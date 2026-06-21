//! Single-arena bump allocator with generative-lifetime scoping and a
//! build-then-freeze collection model.
//!
//! The entire process shares one static `[u8; SIZE]` backing buffer. Scopes
//! are opened with [`scope`] (top-level) or [`Frame::scope`] (nested); each
//! saves the bump offset on entry and restores it on exit via a Drop guard.
//! Collections (`FSpan`, `FStr`) are built incrementally through a closure-
//! scoped [`FBuilder`] that bumps the arena one element at a time, then
//! freeze to their exact length — no over-allocation, no tail padding, no
//! bulk memory initialisation. `FStr` is a type alias for `FSpan<u8>`; if
//! you want UTF-8 substitution for non-UTF-8 bytes, do it at display time
//! (see `main.rs`'s `Bytes` adapter), not during build.
//!
//! Stack-order discipline is a compile-time property: the `for<'id>` HRTB
//! on scope closures means `'id` cannot appear in the closure's return type,
//! so the borrow checker rejects any attempt to leak an allocation past its
//! scope.
//!
//! ## Example
//!
//! ```text
//!   scope(|init| {                         // 'init brand opens
//!      let path = init.str("test", |b| {     // &mut init borrowed
//!          write!(b, "/proc/{pid}/stat").unwrap();
//!      });                                 // FStr<'init> = FSpan<'init, u8>, frozen
//!      init.scope(|tick| {                 // 'tick brand opens
//!          let procs = tick.span("test", |b| {
//!              for pid in pids {
//!                  b.push(ProcInfo { ... });
//!              }
//!          });                             // FSpan<'tick, ProcInfo>, frozen
//!          render(&procs);
//!      });                                 // offset rewinds, 'tick freed
//!   });                                    // offset rewinds, 'init freed
//! ```
//!
//! ## Sizing
//!
//! The arena is a fixed-size byte buffer in `.bss` — it contributes to
//! VmSize but not file size, and pages that are never touched cost zero
//! RSS (Linux demand-zeros on first write). Only pages actually written by
//! builder pushes or `alloc` calls commit physical RAM.
//!
//! ## Threads
//!
//! One global arena, one thread. The `Cell<u32>` offset is not
//! synchronised. Tests serialise via `TEST_LOCK`.

use core::cell::{Cell, UnsafeCell};
use core::fmt;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut};
use core::ptr;
use core::slice;

// `syscall` is only referenced by `uncommit_tail` on Linux; the
// `arena-trace` logger imports it inside `mod trace` directly.
#[cfg(target_os = "linux")]
use crate::syscall;
use crate::zeroable::Zeroable;

// ── Label ────────────────────────────────────────────────────────────────────

/// Allocation tag used by the `arena-trace` feature. Carries a
/// `&'static str` when the feature is on; **zero-sized** when it's off.
///
/// Every bump-producing Frame method takes `label: impl Into<Label>`, so
/// call sites pass a string literal directly — `frame.vec("procs", n)` —
/// and the type system handles the conversion. With the feature off,
/// `From<&'static str> for Label` discards its argument, so at inlined
/// call sites the literal is a dead value and `--gc-sections` drops the
/// rodata entry. Net effect: exactly zero code or data cost when off.
#[cfg(feature = "arena-trace")]
#[derive(Copy, Clone)]
pub struct Label(&'static str);

#[cfg(not(feature = "arena-trace"))]
#[derive(Copy, Clone)]
pub struct Label;

impl From<&'static str> for Label {
    fn from(_s: &'static str) -> Self {
        #[cfg(feature = "arena-trace")]
        { Label(_s) }
        #[cfg(not(feature = "arena-trace"))]
        { Label }
    }
}

#[cfg(feature = "arena-trace")]
impl Label {
    fn as_bytes(self) -> &'static [u8] { self.0.as_bytes() }
}

// ── Trace sink ───────────────────────────────────────────────────────────────

/// Arena event log, written directly to a file via `write(2)`. Enabled
/// by the `arena-trace` cargo feature; off by default. Stays `no_std`:
/// every event builds on a stack buffer and goes out in one `write(2)`,
/// no heap or arena memory involved.
///
/// File path comes from the `ARENA_TRACE` env var at program start, else
/// defaults to `./arena-trace.log`. The fd is opened on the first event
/// with `O_WRONLY | O_CREAT | O_TRUNC`; failure is silent (tracing is a
/// dev tool, not something ltop should crash on).
///
/// Event grammar, one per line:
///
/// ```text
/// A {new_offset} {label}\n   alloc (bump pointer AFTER the alloc)
/// R {new_offset}\n            rewind (scope close / FReserve drop / compact / freeze)
/// U {new_offset}\n            uncommit_tail (madvise MADV_DONTNEED)
/// ```
///
/// Size and nesting depth are recoverable by the renderer: size of the
/// N-th alloc = its new_offset minus the prior event's new_offset;
/// depth = count of active scopes at that seq.
#[cfg(feature = "arena-trace")]
mod trace {
    use core::cell::Cell;
    use crate::syscall;

    /// File descriptor, lazily opened. -1 sentinel = unopened, -2 = open
    /// failed (don't keep trying). Single-threaded access — a bare `Cell`
    /// is enough; the rest of the arena assumes single-threadedness too.
    struct Fd(Cell<i32>);
    // SAFETY: single-threaded; never accessed across threads.
    unsafe impl Sync for Fd {}
    static FD: Fd = Fd(Cell::new(-1));

    fn get_fd() -> i32 {
        let cur = FD.0.get();
        if cur != -1 { return cur; }
        // Default path; no getenv parsing (keeps trace.rs tiny).
        let path = b"./arena-trace.log\0";
        // SAFETY: path is null-terminated; O_WRONLY|O_CREAT|O_TRUNC is
        // what every libc provides; mode is octal 0644.
        let fd = unsafe {
            syscall::open_create(
                path.as_ptr(),
                syscall::O_WRONLY | syscall::O_CREAT | syscall::O_TRUNC,
                0o644,
            )
        };
        FD.0.set(if fd < 0 { -2 } else { fd });
        FD.0.get()
    }

    /// Write a decimal u32 into `buf[pos..]`, returning the new position.
    /// Produces up to 10 ASCII digits; overflow (>= 4 GB) would wrap but
    /// the arena tops out at 512 KB so that's not possible in practice.
    fn push_u32(buf: &mut [u8], mut pos: usize, mut n: u32) -> usize {
        let mut digits = [0u8; 10];
        let mut k = 0;
        if n == 0 {
            digits[0] = b'0';
            k = 1;
        } else {
            while n > 0 {
                digits[k] = b'0' + (n % 10) as u8;
                n /= 10;
                k += 1;
            }
        }
        while k > 0 {
            k -= 1;
            buf[pos] = digits[k];
            pos += 1;
        }
        pos
    }

    fn push(buf: &mut [u8], pos: usize, bytes: &[u8]) -> usize {
        let end = pos + bytes.len();
        buf[pos..end].copy_from_slice(bytes);
        end
    }

    fn write_line(buf: &[u8]) {
        let fd = get_fd();
        if fd < 0 { return; }
        // Partial writes are rare for small line buffers — ignore here.
        syscall::write_all(fd, buf);
    }

    pub(super) fn alloc(label: super::Label, new_offset: u32) {
        // "A {new_offset} {label}\n" — cap the label at 240 bytes to stay
        // within the stack buffer.
        let mut buf = [0u8; 256];
        let mut pos = push(&mut buf, 0, b"A ");
        pos = push_u32(&mut buf, pos, new_offset);
        pos = push(&mut buf, pos, b" ");
        let label = label.as_bytes();
        let label = if label.len() > 240 { &label[..240] } else { label };
        pos = push(&mut buf, pos, label);
        pos = push(&mut buf, pos, b"\n");
        write_line(&buf[..pos]);
    }

    pub(super) fn rewind(new_offset: u32) {
        let mut buf = [0u8; 32];
        let mut pos = push(&mut buf, 0, b"R ");
        pos = push_u32(&mut buf, pos, new_offset);
        pos = push(&mut buf, pos, b"\n");
        write_line(&buf[..pos]);
    }

    pub(super) fn uncommit(new_offset: u32) {
        let mut buf = [0u8; 32];
        let mut pos = push(&mut buf, 0, b"U ");
        pos = push_u32(&mut buf, pos, new_offset);
        pos = push(&mut buf, pos, b"\n");
        write_line(&buf[..pos]);
    }
}

// ── Backing storage ──────────────────────────────────────────────────────────

/// Production arena size (bytes). Peak live usage on a typical host is
/// ~160 KB (init-scope cross-tick buffers + the per-tick rm::Scratch +
/// tick transients), so 512 KB gives 3× headroom for machines with many
/// more processes than our MAX_PROCS (1024) default, or future growth.
/// Virtual memory is free: pages never written don't commit RSS, so
/// over-sizing the arena costs nothing at runtime. RSS tracks the tight
/// working set, not the declared arena size.
#[cfg(not(test))]
pub const SIZE: usize = 512 * 1024;

/// Test arena size. Bigger than you'd think necessary for the FBuilder/FReserve
/// tests because the `rm_matches_nvml` GPU test allocates two 2048-entry
/// FVec<(u32, GpuProc)> buffers (2 × 24 KB) from a single scope.
#[cfg(test)]
pub const SIZE: usize = 128 * 1024;

// `FSpan` stores its backing as a `(u32 offset, u32 len)` pair, so the arena
// must fit in a u32. 128 KB << 4 GB, but guard against someone bumping SIZE
// past that without noticing.
const _: () = assert!(SIZE <= u32::MAX as usize);

/// Max alignment of any T we're willing to allocate. Storage is aligned to
/// this, so `base + offset` is aligned whenever `offset` is a multiple of
/// `MAX_ALIGN`. `bump` rejects higher-alignment requests.
const MAX_ALIGN: usize = 16;

#[repr(C, align(16))]
struct Storage {
    buf: UnsafeCell<[u8; SIZE]>,
    // u32 rather than usize: SIZE <= u32::MAX is statically asserted, and
    // this matches the u32 offsets used in FSpan/FBox/FReserve. Saves 4 bytes
    // in the static, and more importantly makes the cursor type consistent
    // throughout the module.
    offset: Cell<u32>,
}

// SAFETY: ltop is single-threaded. `unsafe impl Sync` is solely so the
// static can exist; no real synchronisation takes place.
unsafe impl Sync for Storage {}

static ARENA: Storage = Storage {
    buf: UnsafeCell::new([0; SIZE]),
    offset: Cell::new(0),
};

/// `(start_address, length_in_bytes)` of the arena's storage. Used by
/// mac startup (`mac_sys::unmap_idle_state`) to mark "do not
/// deallocate" — the arena lives in `__bss` and the kernel can split
/// it across multiple VM regions (the first few bytes share a region
/// with regular `__data`, the bulk lives in its own region), so we
/// need the full range to mark every overlapping region as keep.
pub fn extent() -> (*const u8, usize) {
    (ARENA.buf.get() as *const u8, SIZE)
}

/// Reserve `size` bytes at `align`, advancing the bump pointer. Panics on
/// overflow. Returns the byte offset of the reservation within the arena
/// (callers hand this straight to FBox/FSpan or convert to a pointer via
/// [`ptr_at`]).
///
/// Label-free: callers pass the label separately to `trace::alloc` one
/// frame up, so this function doesn't need it and so the label doesn't
/// appear in the ABI of any `#[inline]` Frame method — when the feature
/// is off, the call site's label literal is a dead value and DCEs.
fn bump(size: usize, align: usize) -> u32 {
    debug_assert!(align.is_power_of_two());
    // Fail loudly on over-alignment: Storage has align MAX_ALIGN, so any T
    // with stricter alignment would end up misaligned in memory even if the
    // offset math said otherwise. ltop's types don't exceed 8; reject early.
    assert!(
        align <= MAX_ALIGN,
        "arena: requested align {} exceeds MAX_ALIGN {}",
        align, MAX_ALIGN,
    );
    // Reject over-size requests before they silently truncate to u32.
    // (SIZE <= u32::MAX is statically asserted, so anything <= SIZE fits.)
    assert!(
        size <= SIZE,
        "arena: requested size {} exceeds SIZE {}",
        size, SIZE,
    );
    let align = align as u32;
    let size = size as u32;
    let cur = ARENA.offset.get();
    let aligned = (cur + (align - 1)) & !(align - 1);
    let end = aligned.checked_add(size).expect("arena: size overflow");
    assert!(
        (end as usize) <= SIZE,
        "arena overflow: need {} bytes at align {}, have {} free",
        size, align, SIZE.saturating_sub(aligned as usize),
    );
    ARENA.offset.set(end);
    aligned
}

/// Compute a mutable pointer into the arena from a byte offset. The only
/// place we cross into usize territory.
fn ptr_at<T>(offset: u32) -> *mut T {
    // SAFETY: callers pass offsets produced by `bump`, which are in-bounds
    // of ARENA.buf by construction.
    unsafe { (ARENA.buf.get() as *mut u8).add(offset as usize) as *mut T }
}

/// Forward-only memmove: copies `len` bytes from `src` to `dst` assuming
/// `dst <= src`. Safe to use when the overlap direction is statically
/// known — every byte is read before its position is overwritten.
///
/// Used by `Frame::replace` and `Frame::compact` in place of
/// `core::ptr::copy`. `ptr::copy` has to handle both overlap directions
/// at runtime, which pulls `compiler_builtins::memmove` (~364 B of
/// __text) into the binary. A forward byte loop saves those bytes at
/// the cost of not handling `dst > src` — a caller that violates the
/// `dst <= src` precondition gets a corrupt copy (debug_assert guards
/// under test).
///
/// Volatile read/write prevents LLVM's loop-idiom pass from rewriting
/// the loop back into a memmove/memcpy call (same trick as the
/// in-crate `bzero` in `src/platform.rs`).
#[inline]
unsafe fn copy_down(src: *const u8, dst: *mut u8, len: usize) {
    debug_assert!(dst as usize <= src as usize,
        "copy_down requires dst <= src (dst={}, src={})",
        dst as usize, src as usize);
    let mut i = 0usize;
    while i < len {
        unsafe {
            let b = core::ptr::read_volatile(src.add(i));
            core::ptr::write_volatile(dst.add(i), b);
        }
        i += 1;
    }
}

// ── Scope ───────────────────────────────────────────────────────────────────

/// Open a top-level scope.
///
/// `'id` is a fresh brand introduced by the `for<'id>` HRTB and made
/// invariant via `PhantomData<fn(&'id ()) -> &'id ()>`; it cannot appear in
/// `R`, so allocations cannot escape the closure. The Drop guard restores
/// the bump offset on scope exit (including via an unwinding panic).
///
/// Returning an allocation out of the closure is a compile error:
///
/// ```compile_fail
/// # use ltop::arena::{self, FBox};
/// let _escaped: FBox<u32> = arena::scope(|f| f.alloc("test", 1u32));
/// ```
///
/// Returning an `FSpan` is also a compile error:
///
/// ```compile_fail
/// # use ltop::arena::{self, FSpan};
/// let _escaped: FSpan<u32> = arena::scope(|f| f.span("test", |b| b.push(1u32)));
/// ```
pub fn scope<R>(f: impl for<'id> FnOnce(&mut Frame<'id>) -> R) -> R {
    let _guard = ScopeGuard(ARENA.offset.get());
    let mut frame = Frame { _brand: PhantomData };
    f(&mut frame)
}

/// Reset the bump offset to its saved position when this guard drops. Runs
/// on normal return AND on unwinding panics, so a panicking scope closure
/// still leaves the arena reusable. Under `panic=abort` (production) the
/// unwind path is elided.
struct ScopeGuard(u32);
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        ARENA.offset.set(self.0);
        #[cfg(feature = "arena-trace")]
        trace::rewind(self.0);
    }
}

/// Current bump offset. Tests only.
#[cfg(test)]
fn offset() -> usize { ARENA.offset.get() as usize }

/// Release physical RAM from the arena's untouched tail back to the kernel
/// via `madvise(MADV_DONTNEED)`. Pages from the bump pointer (rounded up to
/// the next page) through the end of the arena get marked reclaimable:
/// Linux drops them from our RSS and zero-fills on the next touch.
///
/// Call this at a natural low-water point — e.g. at the end of each tick,
/// after all tick-scope allocations have rewound — to release the RAM we
/// touched building tick transients. One syscall; next-tick allocations
/// refault the needed pages on first write.
///
/// Safe to call at any time: `MADV_DONTNEED` is a no-op on pages that are
/// already not-resident, and the arena is single-threaded so there's no
/// concurrency hazard.
///
/// Fallback for unknown OSes (neither Linux nor macOS): no-op. Linux
/// and macOS each have their own impl below.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn uncommit_tail() {}

/// macOS uses `MADV_FREE_REUSABLE` — the Darwin counterpart of Linux's
/// `MADV_DONTNEED` for anonymous memory. Unlike plain `MADV_FREE`
/// (Darwin's default "free" hint), `MADV_FREE_REUSABLE` drops the
/// pages' contribution to `resident_size` immediately and is
/// guaranteed to zero-fault on next access — the two properties we
/// need. Page size is queried via `sysconf(_SC_PAGESIZE)` because
/// Apple silicon uses 16 KB pages while Intel macs use 4 KB; calling
/// it once per tick (2 s steady-state) is cheap.
#[cfg(target_os = "macos")]
pub fn uncommit_tail() {
    let page_size = crate::mac_sys::hw_pagesize() as usize;
    if page_size == 0 { return; }
    let base = ARENA.buf.get() as usize;
    let current_addr = base + ARENA.offset.get() as usize;
    let tail_start = (current_addr + page_size - 1) & !(page_size - 1);
    let arena_end = base + SIZE;
    let len = arena_end.saturating_sub(tail_start) & !(page_size - 1);
    if len > 0 {
        crate::mac_sys::madvise_free_reusable(tail_start as *mut u8, len);
        #[cfg(feature = "arena-trace")]
        trace::uncommit(ARENA.offset.get());
    }
}

#[cfg(target_os = "linux")]
pub fn uncommit_tail() {
    // Page size: 4 KB on every x86_64 Linux kernel we target. Querying
    // `sysconf(_SC_PAGESIZE)` would make this future-proof at the cost of
    // one extra syscall per tick (vs the one we actually want to make).
    const PAGE_SIZE: usize = 4096;
    // ARENA's base address isn't page-aligned (the struct itself is only
    // align(16)), so we have to round the *process address*, not the arena
    // offset, to the next page boundary. Rounding the offset would give a
    // non-page-aligned process address and madvise would fail with EINVAL.
    let base = ARENA.buf.get() as usize;
    let current_addr = base + ARENA.offset.get() as usize;
    let tail_start = (current_addr + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let arena_end = base + SIZE;
    // Round length down to a whole-page multiple so we don't spill past the
    // arena's last byte (madvise only cares about complete pages anyway).
    let len = (arena_end.saturating_sub(tail_start)) & !(PAGE_SIZE - 1);
    if len > 0 {
        // SAFETY: [tail_start .. tail_start + len) is fully inside the
        // arena's backing storage and page-aligned at both ends.
        unsafe {
            syscall::madvise(tail_start as *mut u8, len, syscall::MADV_DONTNEED);
        }
        #[cfg(feature = "arena-trace")]
        trace::uncommit(ARENA.offset.get());
    }
}

// ── Frame ───────────────────────────────────────────────────────────────────

/// Zero-sized scope handle. `'id` brands every allocation; leaking an
/// allocation past the scope is rejected by the borrow checker.
#[repr(transparent)]
pub struct Frame<'id> {
    _brand: PhantomData<fn(&'id ()) -> &'id ()>,
}

/// Compile-time-only marker carrying an invariant `'id` brand.
///
/// **Note: this doesn't prevent any current bug** — it exists for two
/// secondary reasons. First, it gives a name to the "I'm scope-tagged
/// but don't own arena bytes" pattern (see `gpu::agx::State`), so a
/// reader doesn't have to decode the variance dance of bare
/// `PhantomData<fn(&'id ()) -> &'id ()>`. Second, it future-proofs:
/// if such a struct ever gains an arena allocation, the invariant
/// brand is already in place, whereas a covariant
/// `PhantomData<&'id ()>` would silently let the new field leak.
///
/// Construction goes through [`Frame::brand`], so the brand can only come
/// from a live frame. Leaking a `Brand<'id>` out of the scope is a
/// compile error, the same way leaking any other arena type is — even
/// though the leak would be harmless today (Brand is a ZST):
///
/// ```compile_fail
/// # use ltop::arena::{self, Brand};
/// let _escaped: Brand<'_> = arena::scope(|f| f.brand());
/// ```
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Brand<'id> {
    _marker: PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Frame<'id> {
    /// Open a nested scope. `&mut self` ensures the parent frame is
    /// borrowed mutably for the duration of the nested closure, so the
    /// parent can't be used to allocate (which would interleave with the
    /// child's allocations and reclamation).
    ///
    /// Leaking a nested-scope allocation is a compile error:
    ///
    /// ```compile_fail
    /// # use ltop::arena::{self, FBox};
    /// arena::scope(|f| {
    ///     let _escaped: FBox<u32> = f.scope(|sub| sub.alloc("test", 2u32));
    /// });
    /// ```
    /// Mint a `Brand<'id>` carrying this frame's lifetime brand. Use it
    /// to brand-tag a struct that's conceptually scoped to this frame
    /// but doesn't own any arena allocations directly (e.g.,
    /// `gpu::agx::State`, where the cross-tick prev was hoisted out into
    /// a sibling FSpan rotated by `replace2`). Cheap (ZST) and the only
    /// way for downstream code to construct a `Brand`. Doesn't add any
    /// safety today — it's an API-shape pattern; see [`Brand`] for why.
    pub fn brand(&self) -> Brand<'id> {
        Brand { _marker: PhantomData }
    }

    pub fn scope<R>(
        &mut self,
        f: impl for<'sub> FnOnce(&mut Frame<'sub>) -> R,
    ) -> R {
        let _guard = ScopeGuard(ARENA.offset.get());
        let mut sub = Frame { _brand: PhantomData };
        f(&mut sub)
    }

    /// Bytes remaining at the current offset.
    pub fn remaining(&self) -> usize { SIZE - ARENA.offset.get() as usize }

    // ── Single-value allocations ──

    /// Allocate space for `T`, move `value` in, and return an [`FBox`] that
    /// runs `T::drop` when it goes out of scope. Like `Box::new` but the
    /// storage lives in the arena rather than the global heap.
    ///
    /// `label` is a short static byte string identifying this allocation
    /// site for the `arena-trace` feature; it's passed to every bump-
    /// producing method to make the trace self-describing. When the
    /// feature is off the label is unused and should DCE away.
    pub fn alloc<T>(&self, label: impl Into<Label>, value: T) -> FBox<'id, T> {
        let offset = bump(size_of::<T>(), align_of::<T>());
        #[cfg(feature = "arena-trace")]
        trace::alloc(label.into(), ARENA.offset.get());
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();
        // SAFETY: `bump` gave us size_of::<T>() bytes at align_of::<T>().
        unsafe { ptr_at::<T>(offset).write(value); }
        FBox { offset, _marker: PhantomData }
    }

    /// Allocate zero-initialised space for `T` and return an `FBox`.
    /// Safe to call — the `Zeroable` bound moves the "is all-bits-zero
    /// valid for T" obligation to the `unsafe impl Zeroable` site.
    pub fn alloc_zeroed<T: Zeroable>(&self, label: impl Into<Label>) -> FBox<'id, T> {
        let offset = bump(size_of::<T>(), align_of::<T>());
        #[cfg(feature = "arena-trace")]
        trace::alloc(label.into(), ARENA.offset.get());
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();
        // SAFETY: `T: Zeroable` asserts all-bits-zero is a valid `T`;
        // `bump` gave us size_of::<T>() bytes at align_of::<T>().
        unsafe { ptr::write_bytes(ptr_at::<T>(offset) as *mut u8, 0, size_of::<T>()); }
        FBox { offset, _marker: PhantomData }
    }

    // ── Build-then-freeze collections ──

    /// Build an `FSpan` by pushing elements through an [`FBuilder`], then
    /// freeze it. `&mut self` ensures nothing else allocates during the
    /// build — the elements stay contiguous.
    ///
    /// The trace emits **one event per span**, not one per
    /// `FBuilder::push` — an N-element span is conceptually a single
    /// allocation with a single label, and visualising it as N tiny
    /// rects just clutters the diagram. The event records the net bump
    /// after the build closure returns.
    pub fn span<T, F>(&mut self, label: impl Into<Label>, build: F) -> FSpan<'id, T>
    where
        F: FnOnce(&mut FBuilder<'_, 'id, T>),
    {
        // Align the offset once up-front. Every subsequent `push` advances
        // by size_of::<T>(), which is a multiple of align_of::<T>() for any
        // sized type, so alignment is preserved without re-checking.
        let align = align_of::<T>() as u32;
        let cur = ARENA.offset.get();
        let start = (cur + (align - 1)) & !(align - 1);
        ARENA.offset.set(start);

        let mut builder = FBuilder { len: 0, _marker: PhantomData };
        build(&mut builder);

        #[cfg(feature = "arena-trace")]
        if builder.len > 0 {
            trace::alloc(label.into(), ARENA.offset.get());
        }
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();

        FSpan { offset: start, len: builder.len, _marker: PhantomData }
    }

    /// Convenience alias for `span::<u8, _>`. Returns an `FStr` (which
    /// is `FSpan<u8>`); `fmt::Write` is implemented on `FBuilder<u8>` so
    /// `write!(b, ...)` works inside the closure.
    pub fn str<F>(&mut self, label: impl Into<Label>, build: F) -> FStr<'id>
    where
        F: FnOnce(&mut FBuilder<'_, 'id, u8>),
    {
        self.span::<u8, _>(label, build)
    }

    /// Collect an iterator (or gen block) into an `FSpan`. Each item is
    /// bumped into the arena in turn; length is the number of items yielded.
    /// Same guarantees as `span`: `&mut self` locks the frame for the
    /// duration of the iterator, so elements stay contiguous.
    pub fn collect<T, I: IntoIterator<Item = T>>(&mut self, label: impl Into<Label>, iter: I) -> FSpan<'id, T> {
        self.span(label, |b| for item in iter { b.push(item); })
    }

    /// `FSpan` of `n` elements, each `T::default()`. For numeric `T` this
    /// is "n zeros"; for any `T: Default` it's the type's natural empty
    /// value. Use when you want pre-initialised mutable storage (via
    /// `DerefMut<Target=[T]>`) without threading a `repeat_with` iter.
    pub fn zeros<T: Default>(&mut self, label: impl Into<Label>, n: usize) -> FSpan<'id, T> {
        self.collect(label, core::iter::repeat_with(T::default).take(n))
    }

    /// Run `build` with its own access to the arena, then reclaim every
    /// byte it allocated **except** for the returned FSpan's data — which
    /// gets shuffled down to the top-of-arena at the time of the call and
    /// the bump pointer is reset just past it.
    ///
    /// Use this when a computation needs a lot of scratch storage
    /// (adjacency lists, sort keys, temporary indexes) to produce a much
    /// smaller output, and you want the scratch gone by the time the
    /// caller sees the result. `tree_layout` is the canonical example:
    /// it allocates ~30 KB of intermediates and returns an ~8 KB FSpan
    /// of (Indent, u32) — without `compact`, the intermediates stay
    /// pinned in the arena until the enclosing scope exits.
    ///
    /// `T: Copy` because we memmove the result bytes down: types with
    /// `Drop` could cause double-drops, and types containing arena-
    /// relative offsets (e.g., `FSpan<FSpan<U>>`) would have their
    /// inner offsets point at freshly-overwritten memory.
    ///
    /// The `Copy` bound rules out the obvious unsound case: FSpan /
    /// FVec / FBox all have `Drop` impls and therefore are not `Copy`,
    /// so `FSpan<FSpan<T>>` as a return type is a compile error:
    ///
    /// ```compile_fail
    /// # use ltop::arena;
    /// arena::scope(|f| {
    ///     // error: `FSpan<'_, u32>` doesn't satisfy `Copy`.
    ///     let _: arena::FSpan<arena::FSpan<u32>> = f.compact("test", |inner| {
    ///         let scratch: arena::FSpan<u32> =
    ///             inner.collect("test", [1u32, 2, 3].iter().copied());
    ///         inner.collect("test", core::iter::once(scratch))
    ///     });
    /// });
    /// ```
    ///
    /// A user-defined `Copy` type that *happens* to contain a field
    /// interpretable as an arena offset (e.g., a bare `u32` meant to
    /// index into the intermediates) does satisfy `Copy` and slips past
    /// the compiler. That's a documentation caveat — don't put
    /// arena-relative offsets in types you pass through `compact`.
    /// Zero-byte FSpan placeholder: no allocation, length 0, at the
    /// current arena offset. Useful as the initial value of a per-tick
    /// rotating FSpan replaced each iteration via [`replace`]. Until
    /// the first replace, the span is legitimately empty — callers can
    /// `as_slice()` / `iter()` without special-casing.
    pub fn empty<T>(&self) -> FSpan<'id, T> {
        FSpan { offset: ARENA.offset.get(), len: 0, _marker: PhantomData }
    }

    /// Replace `span` with the output of `build`, reclaiming both
    /// `span`'s old bytes and any scratch `build` allocated in between.
    ///
    /// Precondition (debug-asserted): `span` must be at the top of the
    /// arena bump — i.e. no allocations happened between `span`'s
    /// creation and this call. The typical pattern is a rotating
    /// cross-iteration FSpan: each loop iteration consumes the previous
    /// value, produces a new one, and any per-iteration scratch is
    /// reclaimed by the replace's rewind.
    ///
    /// `build` receives a mutable frame (for scratch + the final
    /// result) and a shared reference to the old span (readable for
    /// the whole build — its bytes aren't touched until after `build`
    /// returns). The result FSpan is memmoved down to `span`'s start,
    /// overwriting the old contents; `T: Copy` ensures no destructor
    /// conflicts.
    pub fn replace<T, F>(&mut self, label: impl Into<Label>, span: FSpan<'id, T>, build: F) -> FSpan<'id, T>
    where
        T: Copy,
        F: FnOnce(&mut Frame<'id>, &FSpan<'id, T>) -> FSpan<'id, T>,
    {
        let old_start = span.offset;
        let old_end = span.offset + (span.len as u32) * size_of::<T>() as u32;
        assert_eq!(
            old_end, ARENA.offset.get(),
            "replace: span must be at top of bump (end={}, arena={})",
            old_end, ARENA.offset.get(),
        );

        // build reads `&span` (old bytes), allocates scratch + final
        // result via `self`, and returns the new FSpan.
        let new_span = build(self, &span);
        let new_bytes = (new_span.len as u32) * size_of::<T>() as u32;

        // memmove result down over old bytes. `new_span.offset >= old_start`
        // by construction (new_span was allocated after old_start), so
        // this is always a forward-direction copy — `copy_down` handles
        // it without the 364 B of `compiler_builtins::memmove` that
        // `ptr::copy`'s general-direction form would drag in.
        if new_span.offset != old_start && new_bytes > 0 {
            debug_assert!(new_span.offset > old_start);
            // SAFETY: both source and destination are inside ARENA.buf.
            // The source is valid for `new_bytes` (it's `new_span`'s
            // live range); the destination is valid for `new_bytes`
            // (inside the arena's backing storage; overlaps the old
            // span region, which copy_down's forward iteration
            // handles because dst < src means every byte is read
            // before its slot is overwritten).
            unsafe {
                copy_down(
                    ptr_at::<u8>(new_span.offset),
                    ptr_at::<u8>(old_start),
                    new_bytes as usize,
                );
            }
        }
        ARENA.offset.set(old_start + new_bytes);

        // Trace view: "old span disappears, everything build allocated
        // disappears, new span appears at old_start under `label`". The
        // build closure's internal allocs are already in the log; the
        // rewind here closes their rects, and the synthetic alloc
        // represents the replacement result.
        #[cfg(feature = "arena-trace")]
        {
            trace::rewind(old_start);
            if new_bytes > 0 {
                trace::alloc(label.into(), old_start + new_bytes);
            }
        }
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();

        // T: Copy, so neither span's Drop has real work; forget them to
        // be explicit about ownership transferring to the returned
        // FSpan.
        let new_len = new_span.len;
        core::mem::forget(span);
        core::mem::forget(new_span);
        FSpan { offset: old_start, len: new_len, _marker: PhantomData }
    }

    /// Two-span sibling rotation: replace `span1` (lower) and `span2`
    /// (upper) atomically, reclaiming the build closure's scratch and any
    /// padding between the two new spans.
    ///
    /// Same idea as [`replace`] but for two adjacent cross-iteration FSpans
    /// stacked at the top of the bump. The closure must build new1 below
    /// new2 (i.e., allocate new1's final bytes before new2's). Anything
    /// allocated between them — transient scratch, intermediate buffers —
    /// gets squeezed out by the two memmoves.
    ///
    /// Preconditions (debug-asserted):
    ///   - `span1` ends at or before `span2.offset` (with optional
    ///     alignment padding between them).
    ///   - `span2` ends at the top of the bump.
    ///
    /// On Linux the GPU side wires `T1 = ()` (a ZST), making every byte-
    /// count expression in this method fold to zero at compile time and
    /// the new1 memmove guard `new1_bytes > 0` short-circuit to false.
    pub fn replace2<T1, T2, F>(
        &mut self,
        label1: impl Into<Label>, span1: FSpan<'id, T1>,
        label2: impl Into<Label>, span2: FSpan<'id, T2>,
        build: F,
    ) -> (FSpan<'id, T1>, FSpan<'id, T2>)
    where
        T1: Copy, T2: Copy,
        F: FnOnce(&mut Frame<'id>, &FSpan<'id, T1>, &FSpan<'id, T2>)
            -> (FSpan<'id, T1>, FSpan<'id, T2>),
    {
        let old1_start = span1.offset;
        let old1_end = span1.offset + (span1.len as u32) * size_of::<T1>() as u32;
        let old2_start = span2.offset;
        let old2_end = span2.offset + (span2.len as u32) * size_of::<T2>() as u32;
        debug_assert!(
            old1_end <= old2_start,
            "replace2: span1 must precede span2 (span1 ends at {}, span2 starts at {})",
            old1_end, old2_start,
        );
        assert_eq!(
            old2_end, ARENA.offset.get(),
            "replace2: span2 must be at top of bump (end={}, arena={})",
            old2_end, ARENA.offset.get(),
        );

        let (new1, new2) = build(self, &span1, &span2);
        let new1_bytes = (new1.len as u32) * size_of::<T1>() as u32;
        let new2_bytes = (new2.len as u32) * size_of::<T2>() as u32;

        // Step 1: shift new1 down to span1's old slot. Forward copy is
        // safe because new1.offset > old1_start by construction.
        if new1.offset != old1_start && new1_bytes > 0 {
            debug_assert!(new1.offset > old1_start);
            // SAFETY: src/dst both inside ARENA.buf; src valid for
            // new1_bytes; dst inside the arena. dst < src so copy_down's
            // forward iteration is correct even with overlap.
            unsafe {
                copy_down(
                    ptr_at::<u8>(new1.offset),
                    ptr_at::<u8>(old1_start),
                    new1_bytes as usize,
                );
            }
        }

        // Step 2: shift new2 down to just past new1 (with T2 alignment).
        // dst stays below new2.offset because new2 was allocated after
        // new1's bytes inside `build`, and any padding between them only
        // shrinks under memmove.
        let after_new1 = old1_start + new1_bytes;
        let align_t2 = align_of::<T2>() as u32;
        let new2_dst = (after_new1 + (align_t2 - 1)) & !(align_t2 - 1);
        if new2.offset != new2_dst && new2_bytes > 0 {
            debug_assert!(new2.offset > new2_dst);
            // SAFETY: same reasoning as step 1, applied to new2.
            unsafe {
                copy_down(
                    ptr_at::<u8>(new2.offset),
                    ptr_at::<u8>(new2_dst),
                    new2_bytes as usize,
                );
            }
        }
        ARENA.offset.set(new2_dst + new2_bytes);

        // Trace: one rewind to the lowest reclaimed point, then a synthetic
        // alloc per surviving span under its own label.
        #[cfg(feature = "arena-trace")]
        {
            trace::rewind(old1_start);
            if new1_bytes > 0 {
                trace::alloc(label1.into(), old1_start + new1_bytes);
            } else { let _ = label1.into(); }
            if new2_bytes > 0 {
                trace::alloc(label2.into(), new2_dst + new2_bytes);
            } else { let _ = label2.into(); }
        }
        #[cfg(not(feature = "arena-trace"))]
        {
            let _ = label1.into();
            let _ = label2.into();
        }

        let new1_len = new1.len;
        let new2_len = new2.len;
        core::mem::forget(span1);
        core::mem::forget(span2);
        core::mem::forget(new1);
        core::mem::forget(new2);
        (
            FSpan { offset: old1_start, len: new1_len, _marker: PhantomData },
            FSpan { offset: new2_dst, len: new2_len, _marker: PhantomData },
        )
    }

    pub fn compact<T: Copy, F>(&mut self, label: impl Into<Label>, build: F) -> FSpan<'id, T>
    where F: FnOnce(&mut Frame<'id>) -> FSpan<'id, T>
    {
        let start = ARENA.offset.get();
        let result = build(self);
        let align = align_of::<T>() as u32;
        let dst = (start + (align - 1)) & !(align - 1);
        let bytes = result.len * size_of::<T>() as u32;
        debug_assert!(result.offset >= dst,
            "compact: build returned FSpan at {} but compact started at {}",
            result.offset, start);
        if dst != result.offset && bytes > 0 {
            // SAFETY: both src and dst are within ARENA.buf (src is inside
            // the range `build` allocated from; dst is inside the range we
            // saved before calling build). `result.offset >= dst` by the
            // debug_assert above, so this is always a forward-direction
            // copy — `copy_down` handles it without dragging
            // `compiler_builtins::memmove` into the binary. We reset the
            // bump past dst below, so nothing else observes the source
            // region afterward. `T: Copy` means no destructors to worry
            // about double-running on the source slots.
            unsafe {
                copy_down(ptr_at::<u8>(result.offset),
                          ptr_at::<u8>(dst),
                          bytes as usize);
            }
        }
        ARENA.offset.set(dst + bytes);
        // Trace view: compact = "everything in build disappears, result
        // reappears at dst under a single label". Emit a rewind to the
        // pre-build offset (closes all inner rects), then a synthetic
        // alloc at dst+bytes tagged with compact's label. The renderer
        // sees one labelled rectangle for the compacted result.
        #[cfg(feature = "arena-trace")]
        {
            trace::rewind(start);
            if bytes > 0 {
                trace::alloc(label.into(), dst + bytes);
            }
        }
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();
        FSpan { offset: dst, len: result.len, _marker: PhantomData }
    }

    /// Reserve space for up to `capacity` elements of type `T` and return
    /// an [`FReserve`]. The reserve borrows this frame mutably for its
    /// entire lifetime, so the borrow checker refuses any direct
    /// `frame.alloc` / `frame.scope` / `frame.span` / `frame.reserve`
    /// calls while the reserve is alive — the top-of-bump invariant
    /// [`FReserve::freeze`] depends on is structural, not runtime.
    ///
    /// Transient allocations during the reserve's life go through
    /// [`FReserve::scope`], which re-borrows through the reserve and
    /// preserves top-of-bump (the sub-scope saves/restores the offset
    /// around itself). See [`FReserve`].
    ///
    /// Direct outer-frame allocation while a reserve is alive is a compile
    /// error (the reserve holds a `&mut Frame` exclusively):
    ///
    /// ```compile_fail
    /// # use ltop::arena;
    /// arena::scope(|f| {
    ///     let mut r = f.reserve::<u32>("test", 10);
    ///     f.alloc("test", 42u32);  // error: `f` already borrowed mutably by `r`
    ///     let _v = r.freeze();
    /// });
    /// ```
    ///
    /// ```compile_fail
    /// # use ltop::arena;
    /// arena::scope(|f| {
    ///     let mut r = f.reserve::<u32>("test", 10);
    ///     f.scope(|_s| {});  // error: `f` already borrowed by `r`
    ///     let _v = r.freeze();
    /// });
    /// ```
    ///
    /// ```compile_fail
    /// # use ltop::arena;
    /// arena::scope(|f| {
    ///     let mut r1 = f.reserve::<u32>("test", 10);
    ///     let mut r2 = f.reserve::<u32>("test", 10);  // error: `f` already borrowed by `r1`
    /// });
    /// ```
    pub fn reserve<'a, T>(&'a mut self, label: impl Into<Label>, capacity: u32) -> FReserve<'a, 'id, T> {
        // 64-bit only: size_of::<T>() * (capacity: u32) fits usize trivially.
        // `bump` asserts the result fits the arena.
        let size = size_of::<T>() * capacity as usize;
        let offset = bump(size, align_of::<T>());
        #[cfg(feature = "arena-trace")]
        trace::alloc(label.into(), ARENA.offset.get());
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();
        FReserve { offset, capacity, len: 0, _marker: PhantomData }
    }

    /// Allocate a fixed-capacity, mutable-length [`FVec`] from this frame.
    /// Unlike `reserve`, `vec` takes `&self` — the frame stays usable and
    /// multiple `FVec`s can coexist alongside other allocations. Used for
    /// long-lived "clear-and-refill" buffers at an outer scope (the cross-
    /// tick swap-pair pattern).
    pub fn vec<T>(&self, label: impl Into<Label>, capacity: u32) -> FVec<'id, T> {
        // 64-bit only: size_of::<T>() * (capacity: u32) fits usize trivially.
        // `bump` asserts the result fits the arena.
        let size = size_of::<T>() * capacity as usize;
        let offset = bump(size, align_of::<T>());
        #[cfg(feature = "arena-trace")]
        trace::alloc(label.into(), ARENA.offset.get());
        #[cfg(not(feature = "arena-trace"))]
        let _ = label.into();
        FVec { offset, capacity, len: 0, _marker: PhantomData }
    }

    /// Build a CSR-style nested collection: `n_buckets` buckets of `T`,
    /// stored as a single `(u32 offsets[n+1], T data[total])` pair in the
    /// arena with no per-bucket headers.
    ///
    /// `build` is called twice. Pass 1 tallies the size of each bucket so we
    /// can allocate `data` tight. Pass 2 fills `data` using per-bucket
    /// cursors. The contract is that `build` yields the same `(bucket_idx,
    /// value)` sequence on both passes; divergence panics:
    ///
    /// * A bucket receiving more items in pass 2 trips the `cursor < limit`
    ///   assertion on the offending emit.
    /// * A bucket receiving fewer items trips the final `cursor ==
    ///   offsets[i+1]` equality check after pass 2.
    /// * Same count per bucket but different values/order passes all checks
    ///   and silently stores the pass-2 values — rely on deterministic
    ///   iterators (iter over a slice, Map::iter, etc.) to avoid this.
    ///
    /// Memory cost: `4*(n_buckets+1) + size_of::<T>() * total` bytes,
    /// kept after return. Transient pass-2 cursors (`4*n_buckets`) are
    /// rewound via an inner scope before `nest` returns. vs. the
    /// naive `FVec<FVec<T>>`: no per-bucket FVec headers (12 bytes
    /// each) and no per-bucket alignment padding.
    ///
    /// `T: Copy + Default` lets us pre-zero `data` in one shot and assign
    /// into slots without tracking partial initialisation — fine for the
    /// u32 / small-POD uses that need nest.
    pub fn nest<T, I, F>(&mut self, label: impl Into<Label>, n_buckets: u32, build: F) -> FNest<'id, T>
    where
        T: Copy + Default,
        I: IntoIterator<Item = (u32, T)>,
        F: Fn() -> I,
    {
        let n = n_buckets as usize;
        let label: Label = label.into();

        // offsets[0..=n]: first pass tallies into offsets[i+1] (so [0]
        // stays 0), then prefix-sum converts counts → cumulative offsets
        // in place. offsets[n] is the total item count.
        let mut offsets: FSpan<'id, u32> = self.zeros(label, n + 1);
        for (b, _) in build() {
            assert!(
                (b as usize) < n,
                "FNest: bucket index {} out of range for n_buckets {}",
                b, n,
            );
            offsets[b as usize + 1] += 1;
        }
        for i in 1..=n {
            offsets[i] += offsets[i - 1];
        }
        let total = offsets[n] as usize;

        let mut data: FSpan<'id, T> = self.zeros(label, total);
        let _ = label;  // `label` consumed by the two zeros calls above.

        // Cursors live inside a sub-scope so the arena rewinds past them
        // when `nest` returns — nothing to leak to the caller. We touch
        // `offsets`/`data` (already allocated, fixed bytes) inside the
        // scope; they don't reborrow `self`.
        self.scope(|sub| {
            let mut cursors: FSpan<u32> = sub.collect(
                "nest/cursors",
                offsets.as_slice()[..n].iter().copied(),
            );

            // Pass 2 need not re-check `bu < n`. If a pure closure passes
            // pass 1 it satisfies the bound here too; an impure closure
            // that diverges to `bu >= n` falls through to `offsets[bu+1]`
            // (offsets has length n+1) and panics via slice bounds check.
            let data_slice = data.as_mut_slice();
            for (b, v) in build() {
                let bu = b as usize;
                let limit = offsets[bu + 1];
                assert!(
                    cursors[bu] < limit,
                    "FNest: build closure yielded more items in bucket {} on pass 2 than pass 1",
                    bu,
                );
                data_slice[cursors[bu] as usize] = v;
                cursors[bu] += 1;
            }

            for i in 0..n {
                assert_eq!(
                    cursors[i], offsets[i + 1],
                    "FNest: build closure yielded fewer items in bucket {} on pass 2 than pass 1",
                    i,
                );
            }
        });

        FNest { offsets, data }
    }
}

// ── FBuilder ────────────────────────────────────────────────────────────────

/// Incremental builder for an `FSpan`. Each `push` bumps the arena by
/// `size_of::<T>()` and writes directly into the newly-reserved slot. The
/// `'a` lifetime on `&mut Frame<'id>` is captured as a `PhantomData` so
/// nothing else can allocate from this frame until the builder is dropped.
pub struct FBuilder<'a, 'id, T> {
    len: u32,
    _marker: PhantomData<(&'a mut Frame<'id>, fn() -> T)>,
}

impl<T> FBuilder<'_, '_, T> {
    pub fn len(&self) -> usize { self.len as usize }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Append `value`. Panics on arena overflow.
    pub fn push(&mut self, value: T) {
        let offset = bump(size_of::<T>(), align_of::<T>());
        // No trace event here: `Frame::span` emits a single event for the
        // whole span after the build closure returns.
        // SAFETY: `bump` gave us a properly-sized+aligned uninitialised slot.
        unsafe { ptr_at::<T>(offset).write(value); }
        self.len += 1;
    }

    /// Append `src` by a bulk copy. `T: Copy` so no drop semantics to worry
    /// about.
    pub fn extend_from_slice(&mut self, src: &[T]) where T: Copy {
        if src.is_empty() { return; }
        let n = src.len();
        // size_of::<T>() * src.len() fits usize on our 64-bit targets;
        // `bump` asserts the result fits the arena.
        let size = size_of::<T>() * n;
        let offset = bump(size, align_of::<T>());
        // SAFETY: `bump` gave us properly-sized+aligned storage; src is a
        // valid read of n `T`s.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), ptr_at::<T>(offset), n); }
        self.len += n as u32;
    }

    /// Rewind the last `push` or single-byte append. No-op on empty.
    /// `FBuilder` holds the top of the arena for the duration of the
    /// build closure, so decrementing the arena offset here is sound:
    /// no other allocation can sit above our last byte. Used by the
    /// tty renderer to strip the trailing `\n` that would otherwise
    /// scroll the terminal on a full-height frame.
    pub fn pop(&mut self) {
        if self.len == 0 { return; }
        ARENA.offset.set(ARENA.offset.get() - size_of::<T>() as u32);
        self.len -= 1;
    }
}

/// `write!(builder, ...)` support for byte FBuilders. `fmt::Write::write_str`
/// only accepts `&str`, which is valid UTF-8 by type, so pushing its bytes
/// preserves whatever UTF-8 property callers care about. Arbitrary-byte
/// input goes through `push`/`extend_from_slice` and is treated as raw
/// bytes — no validation here. If you need `?` substitution for non-UTF-8
/// bytes, do it at display time (see `impl Put for [u8]` in `twrite.rs`).
///
/// Tests still use this via `write!`; production renderer goes through
/// [`TinyWriter`] below.
impl fmt::Write for FBuilder<'_, '_, u8> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

/// [`TinyWriter`] is the production render-path writer trait — see
/// `src/twrite.rs`. An `FBuilder<u8>` just pushes raw bytes into its
/// reserved slot; there's no failure mode.
impl crate::twrite::TinyWriter for FBuilder<'_, '_, u8> {
    #[inline] fn put_bytes(&mut self, b: &[u8]) { self.extend_from_slice(b) }
    #[inline] fn put_byte(&mut self, b: u8)     { self.push(b) }
}

// ── FReserve ────────────────────────────────────────────────────────────────

/// Pre-allocated, max-sized arena buffer that is filled incrementally and
/// frozen into an exact-size `FSpan`, with the unused tail reclaimed.
///
/// `FReserve` holds a `PhantomData<&'a mut Frame<'id>>` — the borrow
/// checker treats this as if we held the reference, so any attempt to use
/// the parent frame directly (`frame.alloc`, `frame.scope`,
/// `frame.span`, `frame.reserve`) is refused until the reserve is
/// frozen or dropped. That's what makes the top-of-bump property that
/// `freeze` and the rewinding Drop depend on **structural** rather than
/// runtime-checked. Since the borrow is only a marker, the struct is
/// 12 bytes: three u32s (offset, capacity, len).
///
/// Transient allocations during the reserve's life go through
/// [`FReserve::scope`], which opens a fresh-branded sub-scope and
/// preserves top-of-bump via the save/restore.
pub struct FReserve<'a, 'id, T> {
    offset: u32,    // byte offset of buf[0] within ARENA.buf
    capacity: u32,  // number of T slots reserved
    len: u32,       // slots initialised so far
    // PhantomData carries:
    //  - `&'a mut Frame<'id>` for the borrow-check lockout on the parent
    //    frame (so nothing else can allocate while the reserve is alive);
    //  - `fn(T) -> T` to make FReserve invariant in T.
    _marker: PhantomData<(&'a mut Frame<'id>, fn(T) -> T)>,
}

impl<'a, 'id, T> FReserve<'a, 'id, T> {
    pub fn len(&self) -> usize { self.len as usize }
    pub fn capacity(&self) -> usize { self.capacity as usize }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn is_full(&self) -> bool { self.len == self.capacity }

    /// Raw pointer to slot `i`. Not bounds-checked.
    fn slot_ptr(&self, i: u32) -> *mut T {
        ptr_at::<T>(self.offset + i * size_of::<T>() as u32)
    }

    /// Append `value`. Returns `Err(value)` if the reserve is already full,
    /// handing the caller back the T it tried to push so they can decide
    /// what to do (drop it, log, retry with different container, etc.).
    /// `.unwrap()` if the caller would rather abort on overflow;
    /// `let _ = r.push(x);` to silently drop on overflow.
    #[must_use = "ignoring the Result drops the value silently on overflow"]
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len >= self.capacity { return Err(value); }
        // SAFETY: slot `self.len` is within the reserved range and uninitialised.
        unsafe { self.slot_ptr(self.len).write(value); }
        self.len += 1;
        Ok(())
    }

    /// Append `src` by a bulk copy. `T: Copy` so no drop semantics to worry
    /// about. Panics on capacity overflow (like the bulk-push form of
    /// FBuilder::extend_from_slice; mirror that API for ergonomic migration
    /// from `Vec<u8>` callers that use `extend_from_slice`).
    pub fn extend_from_slice(&mut self, src: &[T]) where T: Copy {
        let n = src.len();
        if n == 0 { return; }
        // usize arithmetic — sidesteps the `n as u32` truncation that a
        // `self.len + (n as u32)` would do; the single assert covers
        // both "n exceeds u32::MAX" and "exceeds reserved capacity".
        let new_len = self.len as usize + n;
        assert!(
            new_len <= self.capacity as usize,
            "FReserve overflow on extend: {} + {} > {}",
            self.len, n, self.capacity,
        );
        // SAFETY: slot_ptr(self.len) is the first uninit slot; we're writing
        // n contiguous Ts starting there, all within the reservation.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.slot_ptr(self.len), n); }
        self.len = new_len as u32;
    }

    /// Drop trailing elements down to `new_len`. Does not reclaim arena
    /// bytes — that happens on `freeze` (or `Drop`).
    pub fn truncate(&mut self, new_len: u32) {
        while self.len > new_len {
            self.len -= 1;
            // SAFETY: slot was initialised by a prior `push`.
            unsafe { ptr::drop_in_place(self.slot_ptr(self.len)); }
        }
    }

    /// Open a sub-scope for transient allocations (scratch buffers, etc.)
    /// during the reserve's lifetime. Structurally identical to
    /// `Frame::scope`: save bump offset, run closure with a fresh `'sub`
    /// brand, restore offset on exit. The save/restore discipline
    /// preserves the reserve's top-of-bump position.
    pub fn scope<R>(&mut self, f: impl for<'sub> FnOnce(&mut Frame<'sub>) -> R) -> R {
        let _guard = ScopeGuard(ARENA.offset.get());
        let mut sub = Frame { _brand: PhantomData };
        f(&mut sub)
    }

    /// Consume the reserve, return an `FSpan` holding exactly `len`
    /// elements, and rewind the arena offset to release the unused tail.
    ///
    /// Takes `self` by value, so the reserve is moved and cannot be used
    /// after — any attempted use is a compile error:
    ///
    /// ```compile_fail
    /// # use ltop::arena;
    /// arena::scope(|f| {
    ///     let mut r = f.reserve::<u32>("test", 4);
    ///     r.push(1).unwrap();
    ///     let _v = r.freeze();
    ///     r.push(2).unwrap();  // error: `r` was moved into `freeze`
    /// });
    /// ```
    ///
    /// Internally, `self` is forgotten via `mem::forget` so that the
    /// element destructors (which the `Drop` impl would otherwise run) are
    /// owned by the returned `FSpan` — no double-free.
    pub fn freeze(self) -> FSpan<'id, T> {
        // By construction (borrow check + scope save/restore), the arena
        // offset is exactly `offset + capacity * size_of::<T>()` right now.
        // Rewind to the used prefix.
        let used_end = self.offset + self.len * size_of::<T>() as u32;
        ARENA.offset.set(used_end);
        #[cfg(feature = "arena-trace")]
        trace::rewind(used_end);

        let fspan = FSpan {
            offset: self.offset,
            len: self.len,
            _marker: PhantomData,
        };
        // Suppress FReserve::drop: element ownership transfers to the FSpan.
        core::mem::forget(self);
        fspan
    }
}

impl<'a, 'id, T> Drop for FReserve<'a, 'id, T> {
    /// If the reserve is dropped WITHOUT calling `freeze` (e.g., early
    /// return or unwinding panic), drop pushed elements and rewind the
    /// arena offset to before the reserve. Safe because our `&mut Frame`
    /// borrow has ensured nothing else is allocated on top of us.
    fn drop(&mut self) {
        self.truncate(0);
        ARENA.offset.set(self.offset);
        #[cfg(feature = "arena-trace")]
        trace::rewind(self.offset);
    }
}

// ── FVec ────────────────────────────────────────────────────────────────────

/// Fixed-capacity, mutable-length buffer in the arena. Allocated via
/// [`Frame::buf`] from a long-lived scope (typically the outermost one);
/// `push`/`pop`/`clear` reuse the same slots across many tick scopes.
///
/// This is the "pre-size at init, clear and refill each tick" shape that
/// `FSpan` (frozen) and `FReserve` (one-shot build) don't fit. `FVec::drop`
/// runs `T::drop` on all pushed elements but does **not** rewind the
/// arena offset — the enclosing scope's `ScopeGuard` is responsible for
/// reclaiming the bytes. That's why `FVec` is typically allocated at the
/// outermost scope: the scope's ScopeGuard owns the arena reclaim.
///
/// Construction via `&self` means multiple `FVec`s can coexist and the
/// parent frame remains usable (unlike `FReserve`, which holds `&mut Frame`).
/// Two `FVec`s of the same type are `mem::swap`-able — the swap moves the
/// (offset, capacity, len) fields, and since elements are in the arena at
/// those offsets, subsequent reads/writes land on the new backing. This
/// is what makes the `prev/next` swap pattern work.
pub struct FVec<'id, T> {
    offset: u32,
    capacity: u32,
    len: u32,
    _marker: PhantomData<&'id mut [T]>,
}

impl<'id, T> FVec<'id, T> {
    pub fn len(&self) -> usize { self.len as usize }
    pub fn capacity(&self) -> usize { self.capacity as usize }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn is_full(&self) -> bool { self.len == self.capacity }

    fn slot_ptr(&self, i: u32) -> *mut T {
        ptr_at::<T>(self.offset + i * size_of::<T>() as u32)
    }

    /// Append `value`. Returns `Err(value)` if full.
    #[must_use = "ignoring the Result drops the value silently on overflow"]
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len >= self.capacity { return Err(value); }
        // SAFETY: slot `self.len` is within the reserved range and uninitialised.
        unsafe { self.slot_ptr(self.len).write(value); }
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.len -= 1;
        // SAFETY: slot `self.len` was initialised by a prior push.
        Some(unsafe { self.slot_ptr(self.len).read() })
    }

    /// Compact in place: for each element, call `keep(&mut elem)`. Elements
    /// for which `keep` returns true are moved toward the front; elements
    /// for which it returns false are dropped. Final `len` equals the number
    /// of kept elements.
    ///
    /// `keep` gets `&mut T` so callers can mutate the element alongside the
    /// keep/discard decision. It cannot look at other elements — cross-element
    /// access would be a separate prior pass.
    pub fn retain_in_place(&mut self, mut keep: impl FnMut(&mut T) -> bool) {
        let mut w = 0u32;
        let end = self.len;
        for r in 0..end {
            // SAFETY: slot r is in bounds and still initialised — we only
            // copy INTO slots < r in prior iterations, never out of slot r.
            let keep_this = keep(unsafe { &mut *self.slot_ptr(r) });
            if keep_this {
                if w != r {
                    // SAFETY: slot w is uninit (either moved-from by an earlier
                    // keep-and-move, or dropped by an earlier discard). slot r
                    // is live; after the copy it becomes moved-from.
                    unsafe { ptr::copy_nonoverlapping(self.slot_ptr(r), self.slot_ptr(w), 1); }
                }
                w += 1;
            } else {
                // SAFETY: slot r is live. After drop, it's uninit (but we
                // won't touch it again — r only increases).
                unsafe { ptr::drop_in_place(self.slot_ptr(r)); }
            }
        }
        self.len = w;
    }

    /// Drop all elements; keep capacity. Between-ticks cleanup pattern.
    pub fn clear(&mut self) { self.truncate(0); }

    /// Drop elements down to `new_len`, keeping capacity.
    pub fn truncate(&mut self, new_len: u32) {
        while self.len > new_len {
            self.len -= 1;
            // SAFETY: slot was initialised by a prior push.
            unsafe { ptr::drop_in_place(self.slot_ptr(self.len)); }
        }
    }

    /// Append `src` by bulk copy. Panics on overflow.
    pub fn extend_from_slice(&mut self, src: &[T]) where T: Copy {
        let n = src.len();
        if n == 0 { return; }
        // usize arithmetic — sidesteps the `n as u32` truncation that a
        // `self.len + (n as u32)` would do; the single assert covers
        // both "n exceeds u32::MAX" and "exceeds capacity".
        let new_len = self.len as usize + n;
        assert!(new_len <= self.capacity as usize, "FVec overflow on extend: {} + {} > {}",
            self.len, n, self.capacity);
        // SAFETY: writing n Ts starting at slot_ptr(self.len), all in range.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.slot_ptr(self.len), n); }
        self.len = new_len as u32;
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: elements 0..len are initialised; &self is shared borrow.
        unsafe { slice::from_raw_parts(self.slot_ptr(0), self.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: elements 0..len are initialised; &mut self is exclusive.
        unsafe { slice::from_raw_parts_mut(self.slot_ptr(0), self.len as usize) }
    }

    /// Consume the FVec, returning an `FSpan` with the same offset and
    /// current `len`. The unused tail `[offset + len*size, offset +
    /// capacity*size)` is **not reclaimed** directly — callers who care
    /// should use `Frame::compact` around the FVec's construction so
    /// the compact machinery memmoves the result down and rewinds the
    /// arena past it.
    ///
    /// `mem::forget` suppresses FVec::drop; element ownership transfers
    /// to the returned FSpan (whose Drop runs element Drops).
    pub fn into_fspan(self) -> FSpan<'id, T> {
        let fspan = FSpan { offset: self.offset, len: self.len, _marker: PhantomData };
        core::mem::forget(self);
        fspan
    }
}

impl<'id> FVec<'id, u8> {
    /// Zero the entire reserved capacity, mark the vec full, and hand back
    /// the whole run as a mutable slice. For receiving a foreign blob whose
    /// size is known up front (e.g. a `sysctl` payload sized via a prior
    /// size-query) into arena space allocated at runtime — a bulk kernel
    /// copy doesn't fit the element-by-element `push` path. The borrow lasts
    /// as long as `&mut self`, so slices parsed out of the blob stay valid
    /// while the FVec handle is in scope.
    pub fn fill_zeroed(&mut self) -> &mut [u8] {
        let n = self.capacity as usize;
        // SAFETY: `vec` reserved `capacity` bytes at this offset (align 1 for
        // u8); zero them so the slice is fully initialised, then expose it.
        unsafe {
            let p = self.slot_ptr(0);
            ptr::write_bytes(p, 0, n);
            self.len = self.capacity;
            slice::from_raw_parts_mut(p, n)
        }
    }
}

impl<T> Deref for FVec<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.as_slice() }
}

impl<T> DerefMut for FVec<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
}

/// `Extend<T>` on FVec lets callers say `fvec.extend(iter)` without a
/// hand-written `for x in iter { fvec.push(x).unwrap(); }` loop. Panics on
/// overflow (capacity exceeded) — matches the `FVec::extend_from_slice`
/// behaviour, and suits the clear-and-refill pattern where the fixed
/// capacity was sized for the worst case.
impl<T> Extend<T> for FVec<'_, T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item).unwrap_or_else(|_| panic!("FVec::extend overflow"));
        }
    }
}

/// `write!(fvec, ...)` support for byte FVecs — truncates on overflow
/// rather than panicking (matches tty::Stdout's soft-fit behaviour).
impl fmt::Write for FVec<'_, u8> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let room = self.capacity() - self.len();
        let n = bytes.len().min(room);
        self.extend_from_slice(&bytes[..n]);
        Ok(())
    }
}

/// [`TinyWriter`] companion to the `fmt::Write` above; same
/// soft-truncation behaviour.
impl crate::twrite::TinyWriter for FVec<'_, u8> {
    fn put_bytes(&mut self, b: &[u8]) {
        let room = self.capacity() - self.len();
        self.extend_from_slice(&b[..b.len().min(room)]);
    }
    fn put_byte(&mut self, b: u8)     { let _ = self.push(b); }
}

impl<T: fmt::Debug> fmt::Debug for FVec<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl<'id, T> Drop for FVec<'id, T> {
    /// Drops live elements. Does NOT rewind the arena — the enclosing
    /// scope's `ScopeGuard` reclaims the bytes.
    fn drop(&mut self) { self.clear(); }
}

// ── FBox ────────────────────────────────────────────────────────────────────

/// Owning handle to a single `T` allocated in the arena. Analogous to
/// `Box<T>`, with two differences:
/// - storage is in the arena, not on the heap (freed when the enclosing
///   scope exits, not when `FBox` drops);
/// - `FBox` is 4 bytes (just a u32 offset), not one word.
///
/// `FBox::drop` runs `T::drop`. This is the difference between `alloc`
/// and a raw `&'id mut T` return — the raw reference would leak `T::drop`
/// for types with non-trivial destructors (same semantics as `Box::leak`).
///
/// No `leak`/into-raw-reference method: if a caller needs a `&mut T` they
/// can `&mut *fbox`, which lives as long as the `FBox` does. `'id` can
/// never coerce to `'static`, so leaking wouldn't save anything.
pub struct FBox<'id, T> {
    offset: u32,
    _marker: PhantomData<&'id mut T>,
}

impl<'id, T> FBox<'id, T> {
    fn ptr(&self) -> *mut T { ptr_at::<T>(self.offset) }
}

impl<T> Deref for FBox<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: FBox owns the T at this offset; &self borrows it shared-ly.
        unsafe { &*self.ptr() }
    }
}

impl<T> DerefMut for FBox<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: FBox owns the T at this offset; &mut self is exclusive.
        unsafe { &mut *self.ptr() }
    }
}

impl<T: fmt::Debug> fmt::Debug for FBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T> Drop for FBox<'_, T> {
    fn drop(&mut self) {
        // SAFETY: T was initialised at construction and hasn't been moved
        // out. No aliasing: `&mut self` in Drop is exclusive.
        unsafe { ptr::drop_in_place(self.ptr()); }
    }
}

// ── FSpan ────────────────────────────────────────────────────────────────────

/// Frozen vector of `T` living in the arena. Stored as a `(u32 offset, u32
/// len)` pair (8 bytes total, independent of `T`); the element slice is
/// reconstructed on demand by adding the offset to the arena base. Elements
/// are mutable in place via `DerefMut`; length is fixed at build time.
///
/// Compared to the alternative `&'id mut [T]` representation (16 bytes on
/// 64-bit), this halves the struct size — matters when FSpans are stored as
/// fields (e.g., a `display: FStr<'id>` on every `ProcInfo`).
pub struct FSpan<'id, T> {
    offset: u32,
    len: u32,
    // Variance and 'id brand via PhantomData on `&'id mut [T]`:
    // - invariant in T (because of &mut), which is what Rust wants for
    //   containers that own Ts;
    // - invariant in 'id (the brand), which is what we need to prevent
    //   lifetime subtyping from leaking allocations.
    _marker: PhantomData<&'id mut [T]>,
}

impl<'id, T> FSpan<'id, T> {
    pub fn len(&self) -> usize { self.len as usize }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: this FSpan owns the range [offset, offset + len *
        // size_of::<T>()) in ARENA.buf, and &self borrows it shared-ly.
        // Rust's usual aliasing rules apply via the returned lifetime.
        unsafe { slice::from_raw_parts(ptr_at::<T>(self.offset), self.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` is exclusive, so no other reference to this
        // range can exist (FSpan is not Clone/Copy; the range is unique to
        // this FSpan).
        unsafe { slice::from_raw_parts_mut(ptr_at::<T>(self.offset), self.len as usize) }
    }
}

impl<'id, T: Copy> FSpan<'id, T> {
    /// Consume the FSpan and return a `&'id [T]` with the arena brand
    /// lifetime, so helpers that allocate internally (e.g.
    /// `Frame::compact`) can return a slice their caller keeps. Skips
    /// FSpan::drop via `mem::forget` — sound because `T: Copy` implies
    /// no destructors to run.
    pub fn leak(self) -> &'id [T] {
        let ptr = ptr_at::<T>(self.offset);
        let len = self.len as usize;
        core::mem::forget(self);
        // SAFETY: the arena bytes at `ptr` hold `len` initialised `T`s
        // for the duration of the `'id` scope; no aliasing because
        // nothing else holds a handle to these bytes after the forget.
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

impl<'id, T> FSpan<'id, T> {
    /// Consume the FSpan, return an FVec that owns the same bytes
    /// with `capacity == len`. The returned FVec can shrink (via
    /// `retain_in_place`, `truncate`, `pop`) but not grow — its
    /// capacity is fixed at the FSpan's length.
    ///
    /// Useful for "build exact-sized via span, then filter" patterns
    /// where pass 2 needs FVec-style mutability that FSpan lacks.
    /// `mem::forget` suppresses FSpan::drop; ownership transfers to
    /// the returned FVec.
    pub fn into_fvec(self) -> FVec<'id, T> {
        let fvec = FVec {
            offset: self.offset,
            capacity: self.len,
            len: self.len,
            _marker: PhantomData,
        };
        core::mem::forget(self);
        fvec
    }
}

impl<T> Deref for FSpan<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.as_slice() }
}

impl<T> DerefMut for FSpan<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
}

impl<T: fmt::Debug> fmt::Debug for FSpan<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl<'id, T> Drop for FSpan<'id, T> {
    fn drop(&mut self) {
        // Drop each element in reverse order. The backing memory itself is
        // reclaimed by the enclosing scope's ScopeGuard.
        let slice = self.as_mut_slice();
        for i in (0..slice.len()).rev() {
            // SAFETY: index in-bounds, element is initialised.
            unsafe { ptr::drop_in_place(&mut slice[i]); }
        }
    }
}

// ── FNest ────────────────────────────────────────────────────────────────────

/// Frozen CSR-style nested collection: `n` buckets of `T`, stored as a
/// length-(n+1) `offsets` array and a flat `data` array. Built via
/// [`Frame::nest`] with a two-pass builder closure; see that method for
/// the construction contract.
///
/// Layout: `offsets[i]..offsets[i+1]` is the half-open data-index range
/// for bucket `i`. `offsets[0] == 0` and `offsets[n] == data.len()`.
///
/// Size: 16 bytes on the stack (two FSpans @ 8 bytes), independent of
/// bucket count or total items.
pub struct FNest<'id, T> {
    // Invariant: offsets.len() >= 1, offsets[0] == 0, offsets is
    // non-decreasing, offsets.last() == data.len().
    offsets: FSpan<'id, u32>,
    data: FSpan<'id, T>,
}

impl<'id, T> FNest<'id, T> {
    /// Number of buckets. `offsets.len() - 1` by construction.
    pub fn len(&self) -> usize { self.offsets.len() - 1 }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Items in bucket `i`. Panics on out-of-range `i`.
    pub fn bucket(&self, i: usize) -> &[T] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.data.as_slice()[start..end]
    }

    /// Iterate buckets as slices, in order.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.len()).map(move |i| self.bucket(i))
    }

    /// All items, flat. Useful for aggregate operations that don't care
    /// about the bucket partition.
    #[allow(dead_code)]
    pub fn flat(&self) -> &[T] { self.data.as_slice() }
}

// ── FStr ────────────────────────────────────────────────────────────────────

/// Frozen byte buffer — literally `FSpan<u8>`. Use `as_slice()` /
/// `.as_ptr()` (inherited through `Deref<Target=[u8]>`) for the raw bytes.
/// Built via `Frame::str` or `Frame::span::<u8, _>`. If you need
/// a `?`-substituted UTF-8 view for display, wrap `&[u8]` in the display
/// adapter at render time rather than during construction.
pub type FStr<'id> = FSpan<'id, u8>;

// ── FSmall ───────────────────────────────────────────────────────────────────

/// 4-byte arena slice handle, for struct fields where FSpan's 8 bytes
/// are too much padding. Packs `(offset, len)` into a single `u32`
/// with 20 bits of offset (1 MB addressable — covers the 512 KB
/// arena with headroom) and 12 bits of length (up to 4095 bytes per
/// slice — fine for display strings).
///
/// All-zeros (`packed == 0`) is a valid empty slice, so `FSmall` is
/// OK to appear in structs initialised via `mem::zeroed()` (see
/// `collect_procs`'s pid-stub pass).
///
/// Convert from an `FSpan` via `FSpan::into_small()`. The FSpan is
/// consumed; FSmall owns the same arena bytes. Drop runs
/// `drop_in_place` on elements (no-op for `T: Copy` types like
/// `u8`), matching FSpan.
pub struct FSmall<'id, T> {
    packed: u32,
    _marker: PhantomData<&'id mut [T]>,
}

impl<'id, T> FSmall<'id, T> {
    const LEN_BITS: u32 = 12;
    const LEN_MASK: u32 = (1 << Self::LEN_BITS) - 1;
    const MAX_OFFSET: u32 = (1 << (32 - Self::LEN_BITS)) - 1;
    const MAX_LEN: u32 = Self::LEN_MASK;

    pub fn len(&self) -> usize { (self.packed & Self::LEN_MASK) as usize }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    fn offset(&self) -> u32 { self.packed >> Self::LEN_BITS }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: this FSmall owns the arena bytes at
        // [offset, offset + len * size_of::<T>()) for the 'id brand's
        // lifetime; `&self` is a shared borrow so no aliasing.
        unsafe { slice::from_raw_parts(ptr_at::<T>(self.offset()), self.len()) }
    }
}

impl<'id, T> Default for FSmall<'id, T> {
    fn default() -> Self { FSmall { packed: 0, _marker: PhantomData } }
}

impl<'id, T> Drop for FSmall<'id, T> {
    fn drop(&mut self) {
        // Drop elements in reverse order. Backing bytes are reclaimed
        // by the enclosing scope's ScopeGuard.
        let slice = unsafe {
            slice::from_raw_parts_mut(ptr_at::<T>(self.offset()), self.len())
        };
        for i in (0..slice.len()).rev() {
            // SAFETY: index in-bounds, element is initialised.
            unsafe { ptr::drop_in_place(&mut slice[i]); }
        }
    }
}

impl<'id, T> FSpan<'id, T> {
    /// Pack into a 4-byte [`FSmall`], consuming self. Panics if
    /// offset or length exceeds FSmall's packed bit budget — the
    /// bounds (1 MB / 4095) are comfortable for ltop's use case
    /// (512 KB arena, <256-byte display strings) but not unlimited.
    pub fn into_small(self) -> FSmall<'id, T> {
        assert!(
            self.offset <= FSmall::<T>::MAX_OFFSET && self.len <= FSmall::<T>::MAX_LEN,
            "FSmall overflow: offset {} / len {} (max {} / {})",
            self.offset, self.len, FSmall::<T>::MAX_OFFSET, FSmall::<T>::MAX_LEN,
        );
        let packed = (self.offset << FSmall::<T>::LEN_BITS) | self.len;
        // Transfer element ownership: FSmall::drop runs drop_in_place.
        core::mem::forget(self);
        FSmall { packed, _marker: PhantomData }
    }
}

/// Packed byte slice handle — `FSmall<u8>`. ProcInfo.display's type.
pub type FSmallStr<'id> = FSmall<'id, u8>;

// ── Test serialisation ───────────────────────────────────────────────────────

/// Any test that touches the global arena must hold this lock while doing
/// so — `cargo test` runs unit tests in parallel by default, but the arena
/// offset is shared state. GPU tests (which use arena::scope for buffers)
/// also grab this lock for the same reason.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt::Write;
    use std::sync::MutexGuard;

    fn lock() -> MutexGuard<'static, ()> { super::test_lock() }

    // ── Basics ──

    #[test]
    fn alloc_primitive() {
        let _g = lock();
        scope(|f| {
            let mut x = f.alloc("test", 42u32);
            assert_eq!(*x, 42);
            *x = 99;
            assert_eq!(*x, 99);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn alloc_multiple_coexist() {
        let _g = lock();
        scope(|f| {
            let mut a = f.alloc("test", 1u64);
            let mut b = f.alloc("test", 2u64);
            *a += 10;
            *b += 20;
            assert_eq!(*a, 11);
            assert_eq!(*b, 22);
        });
    }

    #[test]
    fn alignment_respected() {
        let _g = lock();
        scope(|f| {
            let _small = f.alloc("test", 0xAAu8);
            let big = f.alloc("test", 0xDEADBEEF_CAFEBABEu64);
            let addr = (&*big) as *const u64 as usize;
            assert_eq!(addr % 8, 0);
            assert_eq!(*big, 0xDEADBEEF_CAFEBABE);
        });
    }

    #[test]
    fn fbox_runs_drop() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let _b1 = f.alloc("test", Counter);
            let _b2 = f.alloc("test", Counter);
            // FBoxes drop at end of scope: two Drop calls.
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn fbox_is_four_bytes() {
        use core::mem::size_of;
        assert_eq!(size_of::<FBox<'_, u8>>(), 4);
        assert_eq!(size_of::<FBox<'_, u64>>(), 4);
        // Non-trivial T doesn't change size either.
        struct Big(u64, u64, u64);
        assert_eq!(size_of::<FBox<'_, Big>>(), 4);
    }

    #[test]
    fn alloc_zeroed_primitive_and_array() {
        let _g = lock();
        scope(|f| {
            // Primitives with `Zeroable` impl: zero value should appear.
            let x: FBox<u32> = f.alloc_zeroed("test");
            assert_eq!(*x, 0);
            let y: FBox<u64> = f.alloc_zeroed("test");
            assert_eq!(*y, 0);
            let flag: FBox<bool> = f.alloc_zeroed("test");
            assert!(!*flag);
            // Array of Zeroable is Zeroable (blanket impl).
            let arr: FBox<[u32; 4]> = f.alloc_zeroed("test");
            assert_eq!(*arr, [0, 0, 0, 0]);
        });
    }

    #[test]
    fn alloc_zeroed_user_struct() {
        // Demonstrates the user-defined-type pattern: declare the struct,
        // assert `unsafe impl Zeroable` at the definition site, then use
        // alloc_zeroed at every call site without any more `unsafe`.
        #[repr(C)]
        #[derive(Debug, PartialEq)]
        struct Blob { a: u32, b: u64, arr: [u8; 16] }
        unsafe impl Zeroable for Blob {}

        let _g = lock();
        scope(|f| {
            let b: FBox<Blob> = f.alloc_zeroed("test");
            assert_eq!(*b, Blob { a: 0, b: 0, arr: [0; 16] });
        });
    }

    #[test]
    fn subscope_reclaims_offset() {
        let _g = lock();
        scope(|f| {
            let _a = f.alloc("test", 1u64);
            let before = offset();
            f.scope(|sub| {
                let _b = sub.alloc("test", 2u64);
                assert!(offset() > before);
            });
            assert_eq!(offset(), before);
            let d = f.alloc("test", 4u64);
            assert_eq!(*d, 4);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn alloc_zst() {
        // A zero-sized type should round-trip without advancing the offset.
        let _g = lock();
        scope(|f| {
            let before = offset();
            struct Marker;
            let _m = f.alloc("test", Marker);
            assert_eq!(offset(), before);  // ZST consumed no bytes
        });
    }

    #[test]
    fn alloc_align_16() {
        // 16-byte-aligned type (via repr(align)). Verifies MAX_ALIGN plumbing.
        let _g = lock();
        #[repr(align(16))]
        struct Aligned16(u64, u64);
        scope(|f| {
            let _small = f.alloc("test", 0u8);  // advance offset to a non-16 position
            let a = f.alloc("test", Aligned16(1, 2));
            let addr = (&*a) as *const Aligned16 as usize;
            assert_eq!(addr % 16, 0);
        });
    }

    #[test]
    fn alloc_rejects_over_align() {
        let _g = lock();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[repr(align(32))]
            struct TooAligned(u64);
            scope(|f| { let _ = f.alloc("test", TooAligned(0)); });
        }));
        assert!(r.is_err());
    }

    #[test]
    fn alloc_rejects_over_size() {
        // A reservation bigger than SIZE panics cleanly, not silently
        // returning a too-small allocation via u32 truncation.
        let _g = lock();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scope(|f| {
                let _r: FReserve<u8> = f.reserve("test", SIZE as u32 + 1);
            });
        }));
        assert!(r.is_err());
    }

    #[test]
    fn arena_overflow_panics() {
        let _g = lock();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scope(|f| {
                // Alloc a slab equal to remaining, then try one more byte.
                let rem = f.remaining();
                for _ in 0..rem { f.alloc("test", 0u8); }
                f.alloc("test", 0u8); // boom
            });
        }));
        assert!(r.is_err());
        // ScopeGuard restored offset on unwind.
        assert_eq!(offset(), 0);
    }

    // ── FBuilder / FSpan ──

    #[test]
    fn fspan_is_eight_bytes() {
        // The whole point of the offset+len representation.
        use core::mem::size_of;
        assert_eq!(size_of::<FSpan<'_, u8>>(), 8);
        assert_eq!(size_of::<FSpan<'_, u32>>(), 8);
        assert_eq!(size_of::<FSpan<'_, u64>>(), 8);
        // And FStr (= FSpan<u8>) matches.
        assert_eq!(size_of::<FStr<'_>>(), 8);
    }

    #[test]
    fn span_basic() {
        let _g = lock();
        scope(|f| {
            let v: FSpan<u32> = f.span("test", |b| {
                b.push(10);
                b.push(20);
                b.push(30);
            });
            assert_eq!(v.len(), 3);
            assert_eq!(v.as_slice(), &[10, 20, 30]);
            // Deref<Target=[T]> so iteration works.
            assert_eq!(v.iter().sum::<u32>(), 60);
        });
    }

    #[test]
    fn span_exact_size() {
        // After freeze, the arena has advanced by exactly n * size_of::<T>()
        // bytes (plus alignment padding). No over-allocation.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let _v: FSpan<u32> = f.span("test", |b| {
                b.push(1);
                b.push(2);
                b.push(3);
            });
            let after = offset();
            // 3 × u32 = 12 bytes. Depending on prior alignment we may see a
            // tiny pad before, but not after.
            let consumed = after - before;
            assert!(consumed >= 12 && consumed < 12 + 4);
        });
    }

    #[test]
    fn fbuilder_len_and_is_empty() {
        let _g = lock();
        scope(|f| {
            let _v: FSpan<u32> = f.span("test", |b| {
                assert!(b.is_empty());
                assert_eq!(b.len(), 0);
                b.push(1);
                assert!(!b.is_empty());
                assert_eq!(b.len(), 1);
                b.push(2);
                b.push(3);
                assert_eq!(b.len(), 3);
            });
        });
    }

    #[test]
    fn span_extend_from_empty_slice() {
        // Edge case: pushing zero bytes should be a no-op (not call bump).
        let _g = lock();
        scope(|f| {
            let before = offset();
            let v: FSpan<u32> = f.span("test", |b| {
                b.extend_from_slice(&[]);
            });
            assert_eq!(v.len(), 0);
            assert_eq!(offset() - before, 0);
        });
    }

    #[test]
    fn span_zst() {
        // FSpan of a ZST: len tracks count, no arena bytes consumed.
        let _g = lock();
        scope(|f| {
            let before = offset();
            #[derive(Debug, PartialEq)]
            struct Marker;
            let v: FSpan<Marker> = f.span("test", |b| {
                b.push(Marker);
                b.push(Marker);
                b.push(Marker);
            });
            assert_eq!(v.len(), 3);
            assert_eq!(offset(), before);
        });
    }

    #[test]
    fn span_empty_closure() {
        let _g = lock();
        scope(|f| {
            let v: FSpan<u32> = f.span("test", |_b| {});
            assert_eq!(v.len(), 0);
            assert!(v.is_empty());
        });
    }

    #[test]
    fn zeros_default_init() {
        let _g = lock();
        scope(|f| {
            // Numeric T: Default::default() is 0.
            let v: FSpan<u32> = f.zeros("test", 5);
            assert_eq!(v.as_slice(), &[0, 0, 0, 0, 0]);

            // Mutable via DerefMut<Target=[T]> — zeros is meant to be written.
            let mut v: FSpan<u32> = f.zeros("test", 4);
            for (i, x) in v.iter_mut().enumerate() { *x = (i * 10) as u32; }
            assert_eq!(v.as_slice(), &[0, 10, 20, 30]);

            // Empty n is a clean no-op.
            let empty: FSpan<u64> = f.zeros("test", 0);
            assert!(empty.is_empty());

            // Non-numeric T still works: `bool::default()` is false,
            // `Option::default()` is None.
            let flags: FSpan<bool> = f.zeros("test", 3);
            assert_eq!(flags.as_slice(), &[false, false, false]);
            let opts: FSpan<Option<u32>> = f.zeros("test", 2);
            assert_eq!(opts.as_slice(), &[None, None]);
        });
    }

    #[test]
    fn zeros_exact_size() {
        // `zeros(n)` must bump the arena by exactly n * size_of::<T>() bytes
        // (plus alignment padding). No over- or under-allocation.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let _v: FSpan<u32> = f.zeros("test", 10);
            let after = offset();
            // 10 × u32 = 40 bytes; allow a small leading pad for alignment.
            let consumed = after - before;
            assert!(consumed >= 40 && consumed < 40 + 4,
                "zeros consumed {} bytes, expected ~40", consumed);
        });
    }

    #[test]
    fn compact_reclaims_intermediates() {
        // Inner closure allocates a big scratch (1 KB of u32s) then returns
        // a small 3-element FSpan. After compact returns, arena should only
        // have paid for the 3 elements, not the 1 KB.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let result: FSpan<u32> = f.compact("test", |inner| {
                let _scratch: FSpan<u32> = inner.zeros("test", 256);  // 1 KB of temp
                inner.collect("test", [42u32, 43, 44].iter().copied())
            });
            let after = offset();
            // After compact: arena holds only 3 × u32 = 12 bytes of result.
            let consumed = after - before;
            assert!(consumed >= 12 && consumed < 12 + 4,
                "compact left {} bytes in arena, expected ~12", consumed);
            assert_eq!(result.as_slice(), &[42, 43, 44]);
        });
    }

    #[test]
    fn compact_preserves_contents_and_rewinds() {
        // A compact that allocates 5 scratch FSpans and returns the middle
        // one's contents via a fresh FSpan must still return the right bytes.
        let _g = lock();
        scope(|f| {
            let result: FSpan<u32> = f.compact("test", |inner| {
                let _a: FSpan<u32> = inner.collect("test", [1u32, 2, 3].iter().copied());
                let _b: FSpan<u32> = inner.collect("test", [10u32, 20, 30, 40].iter().copied());
                let c: FSpan<u32> = inner.collect("test", [100u32, 200].iter().copied());
                // Return c; should survive even though more intermediates
                // come after c in the arena.
                let _d: FSpan<u32> = inner.zeros("test", 64);  // overwrites c's old location
                c
            });
            // The returned span's bytes were moved down to the top-of-arena
            // at compact's start. Its contents should still be [100, 200].
            assert_eq!(result.as_slice(), &[100, 200]);
        });
    }

    #[test]
    fn compact_empty_result() {
        let _g = lock();
        scope(|f| {
            let before = offset();
            let result: FSpan<u32> = f.compact("test", |inner| {
                let _big: FSpan<u32> = inner.zeros("test", 100);
                inner.collect("test", core::iter::empty())
            });
            assert!(result.is_empty());
            // Arena offset back to exactly before.
            assert_eq!(offset() - before, 0);
        });
    }

    #[test]
    fn compact_no_move_needed() {
        // If compact's starting offset already equals the result's offset
        // (i.e., the closure returned immediately with no intermediates),
        // no memmove runs and state is still consistent.
        let _g = lock();
        scope(|f| {
            let result: FSpan<u32> = f.compact("test", |inner| {
                // Immediately build the result with no leading scratch.
                inner.collect("test", [7u32, 8, 9].iter().copied())
            });
            assert_eq!(result.as_slice(), &[7, 8, 9]);
        });
    }

    #[test]
    fn span_extend_from_slice() {
        let _g = lock();
        scope(|f| {
            let v: FSpan<u8> = f.span("test", |b| {
                b.extend_from_slice(b"hello ");
                b.extend_from_slice(b"world");
            });
            assert_eq!(v.as_slice(), b"hello world");
        });
    }

    #[test]
    fn fspan_mutates_in_place() {
        let _g = lock();
        scope(|f| {
            let mut v: FSpan<u32> = f.span("test", |b| {
                for i in 0..4 { b.push(i); }
            });
            for x in v.as_mut_slice() { *x *= 10; }
            assert_eq!(v.as_slice(), &[0, 10, 20, 30]);
        });
    }

    #[test]
    fn fspan_drops_elements() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let _v: FSpan<Counter> = f.span("test", |b| {
                b.push(Counter);
                b.push(Counter);
                b.push(Counter);
            });
            // FSpan drops at end of scope → all three destructors run.
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 3);
    }

    // ── SBuilder / FStr ──

    #[test]
    fn str_via_write() {
        let _g = lock();
        scope(|f| {
            let s: FStr = f.str("test", |b| {
                write!(b, "pid={} rss={}M", 12345u32, 678u32).unwrap();
            });
            assert_eq!(s.as_slice(), b"pid=12345 rss=678M");
            assert_eq!(s.len(), 18);
        });
    }

    #[test]
    fn str_exact_size() {
        let _g = lock();
        scope(|f| {
            let before = offset();
            let _s = f.str("test", |b| {
                write!(b, "/proc/1/stat").unwrap();
            });
            let after = offset();
            assert_eq!(after - before, "/proc/1/stat".len());
        });
    }

    #[test]
    fn fstr_raw_bytes_pass_through() {
        // FBuilder<u8>'s `extend_from_slice` writes bytes verbatim — no
        // UTF-8 validation during build. Display-time adapters (see
        // `Bytes<&[u8]>` in main.rs) are where `?` substitution happens.
        let _g = lock();
        scope(|f| {
            let s: FStr = f.str("test", |b| {
                b.extend_from_slice(b"hi \xff there");
            });
            assert_eq!(s.as_slice(), b"hi \xff there");
        });
    }

    #[test]
    fn fstr_as_c_string() {
        // as_ptr is only safe for libc::open(...) if callers manually
        // include a trailing NUL. Demonstrate that pattern.
        let _g = lock();
        scope(|f| {
            let s: FStr = f.str("test", |b| {
                write!(b, "/tmp/path").unwrap();
                b.push(0);
            });
            assert_eq!(s.as_slice(), b"/tmp/path\0");
            // as_ptr (via Deref<Target=[u8]>) points at the first byte.
            let first = unsafe { *s.as_slice().as_ptr() };
            assert_eq!(first, b'/');
        });
    }

    // ── FReserve ──

    #[test]
    fn freserve_is_twelve_bytes() {
        // Three u32s (offset, capacity, len). PhantomData is ZST.
        use core::mem::size_of;
        assert_eq!(size_of::<FReserve<'_, '_, u8>>(), 12);
        assert_eq!(size_of::<FReserve<'_, '_, u64>>(), 12);
        struct Big([u64; 8]);
        assert_eq!(size_of::<FReserve<'_, '_, Big>>(), 12);
    }

    #[test]
    fn reserve_basic() {
        let _g = lock();
        scope(|f| {
            let mut r: FReserve<u32> = f.reserve("test", 10);
            assert_eq!(r.len(), 0);
            assert_eq!(r.capacity(), 10);
            r.push(1).unwrap();
            r.push(2).unwrap();
            r.push(3).unwrap();
            let v: FSpan<u32> = r.freeze();
            assert_eq!(v.as_slice(), &[1, 2, 3]);
            assert_eq!(v.len(), 3);
        });
    }

    #[test]
    fn reserve_empty_freeze() {
        // Reserve space, push nothing, freeze. FSpan is empty; arena rewinds
        // all the way back to before the reserve.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let r: FReserve<u32> = f.reserve("test", 50);
            assert_eq!(offset() - before, 200);
            let v = r.freeze();
            assert_eq!(v.len(), 0);
            assert!(v.is_empty());
            assert_eq!(offset(), before);  // fully reclaimed
        });
    }

    #[test]
    fn reserve_zst() {
        // A reserve of ZSTs consumes no bytes for backing, but push/len/freeze
        // still track counts correctly.
        let _g = lock();
        scope(|f| {
            let before = offset();
            #[derive(Debug, PartialEq)]
            struct Marker;
            let mut r: FReserve<Marker> = f.reserve("test", 10);
            assert_eq!(offset(), before);  // no backing bytes for ZSTs
            r.push(Marker).unwrap();
            r.push(Marker).unwrap();
            r.push(Marker).unwrap();
            let v = r.freeze();
            assert_eq!(v.len(), 3);
            assert_eq!(offset(), before);
        });
    }

    #[test]
    fn reserve_push_returns_err_when_full() {
        let _g = lock();
        scope(|f| {
            let mut r: FReserve<u32> = f.reserve("test", 2);
            assert_eq!(r.push(10), Ok(()));
            assert_eq!(r.push(20), Ok(()));
            // Third push hits capacity — value comes back in Err.
            assert_eq!(r.push(30), Err(30));
            assert_eq!(r.len(), 2);
            // The reserve is still usable after a rejected push.
            let v = r.freeze();
            assert_eq!(v.as_slice(), &[10, 20]);
        });
    }

    #[test]
    fn reserve_freeze_rewinds_tail() {
        // Freeze should trim the arena offset back from capacity to len.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let mut r: FReserve<u64> = f.reserve("test", 100);  // reserves 800 bytes
            assert_eq!(offset(), before + 800);
            r.push(42).unwrap();
            r.push(43).unwrap();
            let _v = r.freeze();  // should rewind to before + 16
            assert_eq!(offset(), before + 16);
        });
    }

    #[test]
    fn reserve_sub_scope_interleaved() {
        // The point of FReserve: transient allocations (via `r.scope`) can
        // happen between `reserve` and `freeze` without disturbing the
        // reserve's top-of-bump position.
        let _g = lock();
        scope(|f| {
            let before_reserve = offset();
            let mut r: FReserve<u32> = f.reserve("test", 16);
            let after_reserve = offset();
            assert_eq!(after_reserve - before_reserve, 64);  // 16 × u32

            r.push(1).unwrap();
            r.push(2).unwrap();

            // Multiple interleaved sub-scopes, each freeing before the next
            // push. Offset returns to after_reserve between them.
            r.scope(|s| {
                let _scratch = s.alloc("test", 0xDEADBEEFu64);
                assert!(offset() > after_reserve);
            });
            assert_eq!(offset(), after_reserve);
            r.push(3).unwrap();

            r.scope(|s| {
                // Bigger scratch, still reclaimed on exit.
                let _big = s.alloc("test", [0u64; 32]);
            });
            assert_eq!(offset(), after_reserve);
            r.push(4).unwrap();

            // Nested sub-scopes inside the reserve also work.
            r.scope(|s| {
                let _a = s.alloc("test", 1u64);
                s.scope(|s2| {
                    let _b = s2.alloc("test", 2u64);
                });
            });
            assert_eq!(offset(), after_reserve);

            let v = r.freeze();
            assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
            // Post-freeze: arena holds only the trimmed FSpan (16 bytes).
            assert_eq!(offset() - before_reserve, 16);
        });
    }

    #[test]
    fn reserve_drop_rewinds_arena() {
        // A reserve dropped without freeze should reclaim its arena bytes
        // entirely — nothing's on top (borrow check guarantees it).
        let _g = lock();
        scope(|f| {
            let before = offset();
            {
                let mut r: FReserve<u32> = f.reserve("test", 100);
                r.push(1).unwrap();
                r.push(2).unwrap();
                assert_eq!(offset() - before, 400);
                // r drops here (no freeze) → should rewind.
            }
            assert_eq!(offset(), before);
        });
    }

    #[test]
    fn reserve_drop_without_freeze_runs_element_drops() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let mut r: FReserve<Counter> = f.reserve("test", 8);
            r.push(Counter).unwrap();
            r.push(Counter).unwrap();
            r.push(Counter).unwrap();
            // Do NOT freeze; r is dropped at end of scope.
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn reserve_freeze_runs_element_drops_via_fspan() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let mut r: FReserve<Counter> = f.reserve("test", 8);
            r.push(Counter).unwrap();
            r.push(Counter).unwrap();
            let _v = r.freeze();  // ownership → FSpan; FReserve's drop elided
            // After freeze, no drops yet.
            assert_eq!(DROPS.load(Ordering::Relaxed), 0);
        });
        // FSpan drops at end of scope, runs both destructors.
        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn reserve_extend_from_slice() {
        let _g = lock();
        scope(|f| {
            let mut r: FReserve<u8> = f.reserve("test", 32);
            r.extend_from_slice(b"hello ");
            r.extend_from_slice(b"world");
            r.extend_from_slice(b"");  // empty is a no-op
            assert_eq!(r.len(), 11);
            let v = r.freeze();
            assert_eq!(v.as_slice(), b"hello world");
        });
    }

    #[test]
    fn reserve_extend_overflow_panics() {
        let _g = lock();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scope(|f| {
                let mut r: FReserve<u32> = f.reserve("test", 3);
                r.extend_from_slice(&[1, 2, 3, 4]);  // boom
            });
        }));
        assert!(r.is_err());
    }

    #[test]
    fn reserve_truncate() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let mut r: FReserve<Counter> = f.reserve("test", 8);
            r.push(Counter).unwrap();
            r.push(Counter).unwrap();
            r.push(Counter).unwrap();
            r.truncate(1);  // drops elements 1 and 2, keeps element 0
            assert_eq!(DROPS.load(Ordering::Relaxed), 2);
            assert_eq!(r.len(), 1);
            let _v = r.freeze();
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 3);
    }

    // ── FVec ──

    #[test]
    fn fvec_is_twelve_bytes() {
        use core::mem::size_of;
        assert_eq!(size_of::<FVec<'_, u8>>(), 12);
        assert_eq!(size_of::<FVec<'_, u64>>(), 12);
    }

    #[test]
    fn fvec_push_clear_reuse() {
        let _g = lock();
        scope(|f| {
            let mut b: FVec<u32> = f.vec("test", 8);
            assert_eq!(b.capacity(), 8);
            b.push(1).unwrap();
            b.push(2).unwrap();
            b.push(3).unwrap();
            assert_eq!(b.as_slice(), &[1, 2, 3]);
            b.clear();
            assert_eq!(b.len(), 0);
            // Reuse
            b.push(10).unwrap();
            b.push(20).unwrap();
            assert_eq!(b.as_slice(), &[10, 20]);
        });
    }

    #[test]
    fn fvec_push_returns_err_when_full() {
        // Mirrors `reserve_push_returns_err_when_full`. main.rs's
        // capacity-bounded `let _ = next.push(...)` sites depend on push
        // handing back the value (rather than aborting) once the FVec is
        // at capacity.
        let _g = lock();
        scope(|f| {
            let mut v: FVec<u32> = f.vec("test", 2);
            assert_eq!(v.push(10), Ok(()));
            assert_eq!(v.push(20), Ok(()));
            assert_eq!(v.push(30), Err(30));
            assert_eq!(v.len(), 2);
            // Still usable after a rejected push.
            assert_eq!(v.as_slice(), &[10, 20]);
        });
    }

    #[test]
    #[should_panic(expected = "FVec overflow on extend")]
    fn fvec_extend_overflow_panics() {
        // The capacity assert in `FVec::extend_from_slice` is the load-
        // bearing safety check now that the prior `checked_add` is gone:
        // if `len + n` exceeds capacity we must panic before
        // `copy_nonoverlapping` writes past the reserved range.
        let _g = lock();
        scope(|f| {
            let mut v: FVec<u32> = f.vec("test", 3);
            v.extend_from_slice(&[1, 2, 3, 4]);  // boom
        });
    }

    #[test]
    fn fvec_coexists_with_other_allocs() {
        // FVec::new takes &self, so other allocations are fine alongside.
        let _g = lock();
        scope(|f| {
            let mut b1: FVec<u32> = f.vec("test", 4);
            let _x = f.alloc("test", 42u64);      // allocates on top of b1
            let mut b2: FVec<u32> = f.vec("test", 4);  // further on top
            b1.push(1).unwrap();
            b2.push(100).unwrap();
            assert_eq!(b1.as_slice(), &[1]);
            assert_eq!(b2.as_slice(), &[100]);
        });
    }

    #[test]
    fn fvec_swap() {
        // The core use case: two FVecs swapped each tick. After swap, reads
        // and writes against each binding hit the other's backing.
        let _g = lock();
        scope(|f| {
            let mut a: FVec<u32> = f.vec("test", 4);
            let mut b: FVec<u32> = f.vec("test", 4);
            a.push(1).unwrap();
            a.push(2).unwrap();
            b.push(100).unwrap();

            core::mem::swap(&mut a, &mut b);

            // a now holds b's old storage; b holds a's old storage.
            assert_eq!(a.as_slice(), &[100]);
            assert_eq!(b.as_slice(), &[1, 2]);

            // Clearing a clears the post-swap `a` (= old b's storage).
            a.clear();
            b.clear();
            // Push into a; it writes to the post-swap slots.
            a.push(9).unwrap();
            assert_eq!(a.as_slice(), &[9]);
        });
    }

    #[test]
    fn fvec_retain_in_place() {
        let _g = lock();
        scope(|f| {
            let mut v: FVec<u32> = f.vec("test", 8);
            for i in 0..6 { v.push(i).unwrap(); }
            // Keep evens; mutate to +100 as a side-effect to prove &mut works.
            v.retain_in_place(|x| { if *x % 2 == 0 { *x += 100; true } else { false } });
            assert_eq!(v.as_slice(), &[100, 102, 104]);
        });
    }

    #[test]
    fn fvec_retain_in_place_drops_discarded() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter(u32);
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let mut v: FVec<Counter> = f.vec("test", 8);
            for i in 0..5 { v.push(Counter(i)).unwrap(); }
            // Discard odd-id elements — drops 2 (ids 1 and 3).
            v.retain_in_place(|c| c.0 % 2 == 0);
            assert_eq!(DROPS.load(Ordering::Relaxed), 2);
            assert_eq!(v.len(), 3);
            // The 3 remaining drop when the FVec drops at scope exit.
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn fvec_retain_in_place_all_kept() {
        let _g = lock();
        scope(|f| {
            let mut v: FVec<u32> = f.vec("test", 4);
            for i in 0..4 { v.push(i).unwrap(); }
            v.retain_in_place(|_| true);
            assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
        });
    }

    #[test]
    fn fvec_retain_in_place_all_discarded() {
        let _g = lock();
        scope(|f| {
            let mut v: FVec<u32> = f.vec("test", 4);
            for i in 0..4 { v.push(i).unwrap(); }
            v.retain_in_place(|_| false);
            assert!(v.is_empty());
        });
    }

    #[test]
    fn fvec_drops_elements_on_clear_and_drop() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        #[derive(Debug)]
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) { DROPS.fetch_add(1, Ordering::Relaxed); }
        }
        DROPS.store(0, Ordering::Relaxed);

        let _g = lock();
        scope(|f| {
            let mut b: FVec<Counter> = f.vec("test", 4);
            b.push(Counter).unwrap();
            b.push(Counter).unwrap();
            b.clear();
            assert_eq!(DROPS.load(Ordering::Relaxed), 2);
            b.push(Counter).unwrap();
            // FVec::drop at end of scope: drops the third.
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 3);
    }

    // ── Realistic tick pattern ──

    #[test]
    fn ltop_shape() {
        #[derive(Debug, PartialEq)]
        struct ProcInfo { pid: u32, rss_kib: u32 }

        fn collect_procs(b: &mut FBuilder<'_, '_, ProcInfo>, pids: &[u32]) {
            for &pid in pids {
                b.push(ProcInfo { pid, rss_kib: pid * 1024 + 7 });
            }
        }

        let _g = lock();
        let mut ticks_run = 0;
        scope(|init| {
            init.scope(|tick_a| {
                let procs: FSpan<ProcInfo> = tick_a.span("test", |b| {
                    collect_procs(b, &[42, 17, 99]);
                });
                let pid_strs: FStr = tick_a.str("test", |b| {
                    for p in procs.iter() {
                        write!(b, "{} ", p.pid).unwrap();
                    }
                });
                assert_eq!(procs.len(), 3);
                assert_eq!(pid_strs.as_slice(), b"42 17 99 ");
            });

            init.scope(|tick_b| {
                let procs: FSpan<ProcInfo> = tick_b.span("test", |b| {
                    collect_procs(b, &[1, 2]);
                });
                assert_eq!(procs.len(), 2);
                assert_eq!(procs[0], ProcInfo { pid: 1, rss_kib: 1031 });
            });
            ticks_run += 1;
        });
        assert_eq!(ticks_run, 1);
        assert_eq!(offset(), 0);
    }

    // ── FNest ──

    #[test]
    fn nest_empty_buckets() {
        let _g = lock();
        scope(|f| {
            let nest: FNest<u32> = f.nest("test", 0, || core::iter::empty());
            assert!(nest.is_empty());
            assert_eq!(nest.len(), 0);
            assert_eq!(nest.flat(), &[] as &[u32]);
            assert_eq!(nest.iter().count(), 0);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn nest_single_bucket() {
        let _g = lock();
        scope(|f| {
            let items = [10u32, 20, 30];
            let nest: FNest<u32> =
                f.nest("test", 1, || items.iter().copied().map(|v| (0u32, v)));
            assert_eq!(nest.len(), 1);
            assert_eq!(nest.bucket(0), &[10, 20, 30]);
            assert_eq!(nest.flat(), &[10, 20, 30]);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn nest_multiple_buckets_preserves_order() {
        // Items with the same bucket appear in emission order in the
        // output. Across buckets, we see the CSR-style layout.
        let _g = lock();
        scope(|f| {
            // (bucket, value)
            let items: &[(u32, u32)] = &[
                (2, 200),
                (0,  10),
                (2, 201),
                (1, 100),
                (0,  11),
                (1, 101),
                (2, 202),
            ];
            let nest: FNest<u32> = f.nest("test", 3, || items.iter().copied());
            assert_eq!(nest.len(), 3);
            assert_eq!(nest.bucket(0), &[10, 11]);
            assert_eq!(nest.bucket(1), &[100, 101]);
            assert_eq!(nest.bucket(2), &[200, 201, 202]);
            assert_eq!(nest.flat().len(), 7);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn nest_empty_buckets_interleaved() {
        // Buckets with zero items should produce empty slices; the
        // offsets array must still have the right length.
        let _g = lock();
        scope(|f| {
            let items: &[(u32, u32)] = &[(0, 1), (3, 2), (3, 3)];
            let nest: FNest<u32> = f.nest("test", 5, || items.iter().copied());
            assert_eq!(nest.len(), 5);
            assert_eq!(nest.bucket(0), &[1]);
            assert_eq!(nest.bucket(1), &[] as &[u32]);
            assert_eq!(nest.bucket(2), &[] as &[u32]);
            assert_eq!(nest.bucket(3), &[2, 3]);
            assert_eq!(nest.bucket(4), &[] as &[u32]);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn nest_skewed_distribution() {
        // One bucket holds most of the items — exercises wider offsets
        // ranges and ensures cursor math is correct on the long bucket.
        let _g = lock();
        scope(|f| {
            let n = 10u32;
            let hot_bucket = 7u32;
            let hot_count = 100u32;
            let nest: FNest<u32> = f.nest("test", n, || {
                (0..hot_count).map(move |v| (hot_bucket, v))
            });
            assert_eq!(nest.len(), n as usize);
            for i in 0..n as usize {
                if i as u32 == hot_bucket {
                    assert_eq!(nest.bucket(i).len(), hot_count as usize);
                    assert_eq!(nest.bucket(i)[0], 0);
                    assert_eq!(nest.bucket(i)[hot_count as usize - 1], hot_count - 1);
                } else {
                    assert!(nest.bucket(i).is_empty());
                }
            }
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn nest_reclaims_cursor_scratch() {
        // After nest returns, the arena holds exactly offsets + data —
        // no cursor scratch. Measure by taking the offset delta.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let items: &[(u32, u32)] = &[
                (0, 1), (1, 10), (0, 2), (2, 100), (1, 11), (2, 101),
            ];
            let nest: FNest<u32> = f.nest("test", 3, || items.iter().copied());
            let after = offset();
            // offsets = 4 * (n+1) = 16 bytes; data = 4 * 6 = 24 bytes.
            // Total = 40 bytes. No cursor leak (which would add ~12).
            assert_eq!(after - before, 4 * (3 + 1) + 4 * 6);
            assert_eq!(nest.flat().len(), 6);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    #[should_panic(expected = "bucket index 5 out of range for n_buckets 3")]
    fn nest_panics_on_out_of_range_bucket() {
        let _g = lock();
        scope(|f| {
            let items: &[(u32, u32)] = &[(5, 0)];
            let _nest: FNest<u32> = f.nest("test", 3, || items.iter().copied());
        });
    }

    #[test]
    #[should_panic]
    fn nest_panics_if_pass_two_bucket_diverges_out_of_range() {
        // An impure closure whose pass 1 stays in-range but pass 2 yields
        // `bu >= n`. No explicit check in pass 2; relies on `offsets[bu+1]`
        // (offsets has length n+1) panicking via slice bounds.
        use core::cell::Cell;
        let _g = lock();
        let call_count = Cell::new(0u32);
        scope(|f| {
            let _nest: FNest<u32> = f.nest("test", 3, || {
                let c = call_count.get();
                call_count.set(c + 1);
                let b: u32 = if c == 0 { 0 } else { 99 };
                core::iter::once((b, 0u32))
            });
        });
    }

    #[test]
    #[should_panic(expected = "yielded more items in bucket 0 on pass 2")]
    fn nest_panics_if_pass_two_emits_more() {
        use core::cell::Cell;
        let _g = lock();
        let call_count = Cell::new(0u32);
        scope(|f| {
            let _nest: FNest<u32> = f.nest("test", 1, || {
                let c = call_count.get();
                call_count.set(c + 1);
                // Pass 1: one item. Pass 2: two.
                let n = if c == 0 { 1 } else { 2 };
                (0..n).map(|v| (0u32, v))
            });
        });
    }

    #[test]
    #[should_panic(expected = "yielded fewer items in bucket 0 on pass 2")]
    fn nest_panics_if_pass_two_emits_fewer() {
        use core::cell::Cell;
        let _g = lock();
        let call_count = Cell::new(0u32);
        scope(|f| {
            let _nest: FNest<u32> = f.nest("test", 1, || {
                let c = call_count.get();
                call_count.set(c + 1);
                // Pass 1: two items. Pass 2: one.
                let n = if c == 0 { 2 } else { 1 };
                (0..n).map(|v| (0u32, v))
            });
        });
    }

    #[test]
    fn nest_calls_build_twice() {
        use core::cell::Cell;
        let _g = lock();
        let call_count = Cell::new(0u32);
        scope(|f| {
            let _nest: FNest<u32> = f.nest("test", 2, || {
                call_count.set(call_count.get() + 1);
                [(0u32, 1u32), (1, 2)].into_iter()
            });
        });
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn nest_iter_yields_buckets_in_order() {
        let _g = lock();
        scope(|f| {
            let items: &[(u32, u32)] = &[(0, 1), (2, 30), (1, 20), (0, 2)];
            let nest: FNest<u32> = f.nest("test", 3, || items.iter().copied());
            let collected: std::vec::Vec<&[u32]> = nest.iter().collect();
            assert_eq!(collected.len(), 3);
            assert_eq!(collected[0], &[1, 2]);
            assert_eq!(collected[1], &[20]);
            assert_eq!(collected[2], &[30]);
        });
        assert_eq!(offset(), 0);
    }

    // ── empty ──

    #[test]
    fn empty_span_has_no_bytes() {
        let _g = lock();
        scope(|f| {
            let before = offset();
            let s: FSpan<u32> = f.empty();
            assert_eq!(s.len(), 0);
            assert!(s.is_empty());
            assert_eq!(s.as_slice(), &[] as &[u32]);
            // empty doesn't advance the bump.
            assert_eq!(offset(), before);
        });
    }

    // ── replace ──

    #[test]
    fn replace_installs_result_at_old_offset() {
        let _g = lock();
        scope(|f| {
            // Anchor some stuff below prev so we can verify offsets
            // land where we expect.
            let _anchor: FBox<u32> = f.alloc("anchor", 0xAA);
            let before = offset();

            let prev: FSpan<u32> = f.empty();
            assert_eq!(prev.len(), 0);

            let next = f.replace("test", prev, |frame, old| {
                assert_eq!(old.len(), 0);
                // Build a small output through a `span` closure.
                frame.span("inner", |b| { b.push(10); b.push(20); b.push(30); })
            });
            assert_eq!(next.as_slice(), &[10, 20, 30]);
            // Arena top is right past next's three u32s.
            assert_eq!(offset(), before + 3 * 4);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn replace_reclaims_build_scratch_and_old_bytes() {
        let _g = lock();
        scope(|f| {
            let before = offset();
            let prev: FSpan<u32> = f.collect("prev", [1u32, 2, 3, 4, 5].iter().copied());
            let after_prev = offset();
            assert_eq!(after_prev - before, 5 * 4);

            let next = f.replace("test", prev, |frame, old| {
                assert_eq!(old.as_slice(), &[1, 2, 3, 4, 5]);
                // Gratuitous scratch allocation that must not leak.
                let _scratch: FSpan<u32> = frame.zeros("scratch", 200);
                frame.collect("new", old.as_slice().iter().map(|x| x * 10))
            });
            assert_eq!(next.as_slice(), &[10, 20, 30, 40, 50]);
            // Arena top is exactly at the new span's end — old bytes
            // overwritten in place, scratch reclaimed.
            assert_eq!(offset(), before + 5 * 4);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn replace_rotates_across_iterations() {
        // Simulate the main-loop pattern: replace runs repeatedly, each
        // iteration's prev is the previous iteration's next. Offsets
        // must not drift; each iteration lands at the same place.
        let _g = lock();
        scope(|f| {
            let anchor_end = {
                let _a: FBox<u64> = f.alloc("anchor", 0);
                offset()
            };

            let mut prev: FSpan<u32> = f.empty();
            for i in 0..8u32 {
                prev = f.replace("test", prev, |frame, old| {
                    // Shift each old value by `i`, append one new element.
                    frame.span("new", |b| {
                        for &v in old.as_slice() { b.push(v + i); }
                        b.push(100 + i);
                    })
                });
                // After iteration i, prev has i+1 elements.
                assert_eq!(prev.len(), (i + 1) as usize);
                // And it sits immediately past the anchor, no drift.
                assert_eq!(offset(), anchor_end + (i + 1) as usize * 4);
            }
            assert_eq!(prev.as_slice().len(), 8);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn replace_empty_result_leaves_offset_at_old_start() {
        let _g = lock();
        scope(|f| {
            let before = offset();
            let prev: FSpan<u32> = f.collect("prev", [1u32, 2, 3].iter().copied());

            let next = f.replace("test", prev, |frame, _old| frame.empty::<u32>());
            assert_eq!(next.len(), 0);
            assert_eq!(offset(), before);
        });
    }

    #[test]
    fn replace_with_inner_scope_rewinds_cleanly() {
        // build can open sub-scopes; their bytes disappear with the
        // scope close, and replace still sees the correct top-of-bump
        // when the closure returns.
        let _g = lock();
        scope(|f| {
            let before = offset();
            let prev: FSpan<u32> = f.empty();
            let next = f.replace("test", prev, |frame, _old| {
                frame.scope(|sub| {
                    let _big: FSpan<u64> = sub.zeros("big", 500);
                    let _: FSpan<u32> = sub.collect("throwaway", 0u32..10);
                });
                frame.collect("result", [7u32, 11, 13].iter().copied())
            });
            assert_eq!(next.as_slice(), &[7, 11, 13]);
            assert_eq!(offset(), before + 3 * 4);
        });
    }

    #[test]
    #[should_panic(expected = "replace: span must be at top of bump")]
    fn replace_panics_if_span_not_at_top() {
        let _g = lock();
        scope(|f| {
            let prev: FSpan<u32> = f.collect("prev", [1u32, 2, 3].iter().copied());
            // Allocate above prev, breaking the top-of-bump invariant.
            let _above: FBox<u32> = f.alloc("above", 0);
            // Now replace's debug_assert should fire.
            let _ = f.replace("test", prev, |frame, _old| frame.empty::<u32>());
        });
    }

    // ── replace2 ─────────────────────────────────────────────────────────

    #[test]
    fn replace2_installs_both_results_in_order() {
        let _g = lock();
        scope(|f| {
            let _anchor: FBox<u32> = f.alloc("anchor", 0xAA);
            let before = offset();

            let s1: FSpan<u32> = f.empty();
            let s2: FSpan<u64> = f.empty();
            let (n1, n2) = f.replace2("a", s1, "b", s2, |frame, _o1, _o2| {
                let n1 = frame.span("n1", |b| { b.push(10u32); b.push(20); b.push(30); });
                // Allocate scratch between the two new spans — must be reclaimed.
                let _scratch: FSpan<u32> = frame.zeros("scratch", 100);
                let n2 = frame.span("n2", |b| { b.push(7u64); b.push(11); });
                (n1, n2)
            });
            assert_eq!(n1.as_slice(), &[10, 20, 30]);
            assert_eq!(n2.as_slice(), &[7, 11]);
            // n1 lands at `before` (3 × u32 = 12 B). n2 needs 8-byte
            // alignment for u64; `before + 12` is already 8-aligned (the
            // u32 anchor put `before = 4`), so no padding. n2 ends 16 B
            // past its start (2 × u64).
            assert_eq!(n1.offset as usize, before);
            assert_eq!(n2.offset as usize, before + 12);
            assert_eq!(offset(), before + 12 + 16);
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn replace2_rotates_across_iterations() {
        let _g = lock();
        scope(|f| {
            let anchor_end = {
                let _a: FBox<u32> = f.alloc("anchor", 0);
                offset()
            };
            let mut s1: FSpan<u32> = f.empty();
            let mut s2: FSpan<u32> = f.empty();
            for i in 0..6u32 {
                let (n1, n2) = f.replace2("a", s1, "b", s2, |frame, o1, o2| {
                    // n1 grows by 1 each iteration; n2 carries old n2 plus i.
                    let n1 = frame.span("a", |b| {
                        for &v in o1.as_slice() { b.push(v); }
                        b.push(i);
                    });
                    let n2 = frame.span("b", |b| {
                        for &v in o2.as_slice() { b.push(v + 1); }
                        b.push(100 + i);
                    });
                    (n1, n2)
                });
                s1 = n1;
                s2 = n2;
                assert_eq!(s1.len(), (i + 1) as usize);
                assert_eq!(s2.len(), (i + 1) as usize);
                assert_eq!(s1.offset as usize, anchor_end);
                assert_eq!(s2.offset as usize, anchor_end + (i + 1) as usize * 4);
                assert_eq!(offset(), anchor_end + 2 * (i + 1) as usize * 4);
            }
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn replace2_zst_first_span_compiles_and_runs() {
        // Mirrors the Linux GPU configuration: T1 = () means new1_bytes is
        // always zero, so the new1 memmove guard short-circuits and the
        // arithmetic for new2_dst folds to `align_up(old1_start, align T2)`
        // at compile time. The runtime asserts and offset bookkeeping must
        // still be correct.
        let _g = lock();
        scope(|f| {
            let _anchor: FBox<u32> = f.alloc("anchor", 0);
            let before = offset();
            let mut s1: FSpan<()> = f.empty();
            let mut s2: FSpan<u32> = f.empty();
            for i in 0..3u32 {
                let (n1, n2) = f.replace2("zst", s1, "real", s2, |frame, o1, o2| {
                    assert_eq!(o1.len(), 0);
                    let n1 = frame.empty::<()>();
                    let n2 = frame.span("real", |b| {
                        for &v in o2.as_slice() { b.push(v); }
                        b.push(i + 1);
                    });
                    (n1, n2)
                });
                s1 = n1;
                s2 = n2;
                assert_eq!(s1.len(), 0);
                assert_eq!(s2.len(), (i + 1) as usize);
                // No bytes consumed for s1 (ZST), so s2 lands at `before`.
                assert_eq!(s2.offset as usize, before);
                assert_eq!(offset(), before + (i + 1) as usize * 4);
            }
        });
        assert_eq!(offset(), 0);
    }

    #[test]
    fn replace2_empty_results_collapse_to_old_start() {
        let _g = lock();
        scope(|f| {
            let _anchor: FBox<u64> = f.alloc("anchor", 0);
            let before = offset();
            let s1: FSpan<u32> = f.collect("s1", [1u32, 2, 3].iter().copied());
            let s2: FSpan<u32> = f.collect("s2", [4u32, 5].iter().copied());
            assert_eq!(offset(), before + 5 * 4);

            let (n1, n2) = f.replace2("a", s1, "b", s2, |frame, _o1, _o2| {
                // Both new spans empty → arena rewinds to old1 start.
                (frame.empty::<u32>(), frame.empty::<u32>())
            });
            assert_eq!(n1.len(), 0);
            assert_eq!(n2.len(), 0);
            assert_eq!(offset(), before);
        });
    }

    #[test]
    #[should_panic(expected = "replace2: span2 must be at top of bump")]
    fn replace2_panics_if_span2_not_at_top() {
        let _g = lock();
        scope(|f| {
            let s1: FSpan<u32> = f.collect("s1", [1u32].iter().copied());
            let s2: FSpan<u32> = f.collect("s2", [2u32].iter().copied());
            let _above: FBox<u32> = f.alloc("above", 0);
            let _ = f.replace2("a", s1, "b", s2, |frame, _o1, _o2| {
                (frame.empty::<u32>(), frame.empty::<u32>())
            });
        });
    }
}
