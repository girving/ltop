//! Libc-backed replacements for the std features we used: `Instant`,
//! `read_dir`, `sleep`, `getpid`, `args`. Plus the `#[global_allocator]`
//! and `#[panic_handler]` that `#![no_std]` requires us to provide ourselves.
//!
//! Each replacement is scoped to exactly what ltop needs, not to be a
//! general-purpose std shim.

use core::time::Duration;

use crate::syscall;

// ── Time ─────────────────────────────────────────────────────────────────────

/// Monotonic timestamp. Replacement for `std::time::Instant`, built from
/// `clock_gettime(CLOCK_MONOTONIC)`. We only ever subtract two of them to
/// get a `Duration`, so we store just the nanosecond count.
#[derive(Clone, Copy, Debug)]
pub struct Instant(u64);

impl Instant {
    pub fn now() -> Self {
        let ts = syscall::get_clock(syscall::CLOCK_MONOTONIC);
        Instant(ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

// ── Sleep ────────────────────────────────────────────────────────────────────

pub fn sleep(d: Duration) {
    let ts = syscall::Timespec {
        tv_sec: d.as_secs() as i64,
        tv_nsec: d.subsec_nanos() as i64,
    };
    unsafe { syscall::nanosleep(&ts, core::ptr::null_mut()); }
}

// ── Process ──────────────────────────────────────────────────────────────────

pub fn getpid() -> u32 {
    syscall::getpid() as u32
}

// ── Directory iteration ──────────────────────────────────────────────────────

/// Iterator over numeric entries in a directory, yielding PID as u32.
/// Non-numeric entries (like `/proc/stat`, `/proc/meminfo`) are skipped.
///
/// Implemented via direct `open` + `getdents64` syscalls rather than
/// `opendir`/`readdir` — opendir allocates the `DIR` struct via
/// calloc, which drags the entire malloc chain (~8 KB) into the binary.
/// Raw getdents64 needs no heap; the getdents64 buffer is supplied by
/// the caller (expected to be an arena-allocated `[u64; DIRENT_BUF_U64S]`),
/// keeping the 4 KB of page-aligned scratch off the tick's stack frame.
///
/// Linux-only: macOS scans PIDs via libproc's `proc_listallpids` and
/// never touches this path.
#[cfg(target_os = "linux")]
pub const DIRENT_BUF_U64S: usize = 512; // 4 KB — comfortably holds ~100+ entries per syscall.

#[cfg(target_os = "linux")]
pub struct PidDir<'a> {
    fd: i32,
    buf_filled: usize, // bytes of `buf` containing valid dirent data
    buf_pos: usize,    // current read offset within `buf`
    buf: &'a mut [u64; DIRENT_BUF_U64S],
}

// Layout the kernel writes at the start of each entry in the buffer.
// Followed by d_name bytes starting at offset DIRENT_NAME_OFFSET, NUL-
// terminated. See `linux_dirent64` in <linux/dirent.h>.
#[cfg(target_os = "linux")]
#[repr(C)]
struct LinuxDirent64 {
    d_ino: u64,
    d_off: i64,
    d_reclen: u16,
    d_type: u8,
}
#[cfg(target_os = "linux")]
const DIRENT_NAME_OFFSET: usize = 19; // size_of::<LinuxDirent64>() packed

#[cfg(target_os = "linux")]
impl<'a> PidDir<'a> {
    pub fn open(path: &core::ffi::CStr, buf: &'a mut [u64; DIRENT_BUF_U64S]) -> Option<Self> {
        let fd = unsafe {
            syscall::open(path.as_ptr() as *const u8,
                          syscall::O_RDONLY | syscall::O_DIRECTORY | syscall::O_CLOEXEC)
        };
        if fd < 0 { return None; }
        Some(Self { fd, buf_filled: 0, buf_pos: 0, buf })
    }

    /// File descriptor of the open directory — usable as the `dirfd`
    /// argument to `openat(2)` so callers avoid re-opening `/proc`.
    pub fn fd(&self) -> i32 { self.fd }

    fn buf_ptr(&self) -> *const u8 { self.buf.as_ptr() as *const u8 }
}

#[cfg(target_os = "linux")]
impl Iterator for PidDir<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        loop {
            // Refill when we've consumed the previous batch.
            if self.buf_pos >= self.buf_filled {
                let n = unsafe {
                    syscall::getdents64(
                        self.fd,
                        self.buf.as_mut_ptr() as *mut u8,
                        core::mem::size_of_val::<[u64; DIRENT_BUF_U64S]>(self.buf),
                    )
                };
                if n <= 0 { return None; }  // 0 = EOF, <0 = error
                self.buf_filled = n as usize;
                self.buf_pos = 0;
            }

            // SAFETY: kernel writes complete dirent records; d_reclen is
            // the byte-width of this entry including d_name. The buffer is
            // u64-aligned so the u64 header fields are naturally aligned.
            let ent_ptr = unsafe { self.buf_ptr().add(self.buf_pos) };
            let ent = unsafe { &*(ent_ptr as *const LinuxDirent64) };
            let reclen = ent.d_reclen as usize;
            let name_ptr = unsafe { ent_ptr.add(DIRENT_NAME_OFFSET) };
            self.buf_pos += reclen;

            // Parse the NUL-terminated name as a pid; skip non-numeric.
            let mut pid: u32 = 0;
            let mut i = 0;
            let mut ok = true;
            loop {
                let b = unsafe { *name_ptr.add(i) };
                if b == 0 { break; }
                if !b.is_ascii_digit() { ok = false; break; }
                pid = pid.wrapping_mul(10).wrapping_add((b - b'0') as u32);
                i += 1;
            }
            if ok && i > 0 { return Some(pid); }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for PidDir<'_> {
    fn drop(&mut self) { syscall::close(self.fd); }
}

// ── argv parsing ─────────────────────────────────────────────────────────────

/// Does argv[1..argc] contain a byte-equal match for `flag`?
pub fn has_arg(argc: i32, argv: *const *const u8, flag: &[u8]) -> bool {
    for i in 1..argc as usize {
        let ptr = unsafe { *argv.add(i) };
        if ptr.is_null() { continue; }
        // Compare up to NUL terminator on argv side.
        let mut j = 0;
        loop {
            let a = unsafe { *ptr.add(j) };
            let b = if j < flag.len() { flag[j] } else { 0 };
            if a != b { break; }
            if a == 0 { return true; }
            j += 1;
        }
    }
    false
}

// ── Runtime glue: global allocator + panic handler ───────────────────────────
//
// Only compiled in the production (non-test) build. `cargo test` pulls in
// libstd, which provides its own allocator and panic handler — defining ours
// there would conflict.

#[cfg(not(test))]
mod rt {
    use core::alloc::{GlobalAlloc, Layout};

    /// Null allocator: aborts on any heap allocation. Production Linux ltop
    /// routes every allocation through the arena, so this should never fire;
    /// it's here so that `#[global_allocator]` is satisfied and so any stray
    /// heap use (regression, macOS code path dragged into a Linux build, etc.)
    /// fails loudly at the first call rather than silently allocating via
    /// a linked libc's malloc.
    ///
    /// Eliminating the libc malloc machinery reclaims ~70 KB of RSS — the
    /// malloc arena, bin metadata, and the thread-safety fallbacks a libc
    /// initialises even for a single-threaded process.
    struct Abort;
    unsafe impl GlobalAlloc for Abort {
        unsafe fn alloc(&self, _: Layout) -> *mut u8 { abort_now() }
        unsafe fn dealloc(&self, _: *mut u8, _: Layout) { abort_now() }
        unsafe fn realloc(&self, _: *mut u8, _: Layout, _: usize) -> *mut u8 {
            abort_now()
        }
    }

    #[global_allocator]
    static ALLOCATOR: Abort = Abort;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        abort_now()
    }

    /// Equivalent to `libc::abort` but without going through libc's signal
    /// machinery (which touches TLS we haven't set up). 134 = 128 + SIGABRT,
    /// the shell convention for "died by SIGABRT".
    fn abort_now() -> ! {
        crate::syscall::exit_group(134)
    }

    // The precompiled `liballoc.rlib` that ships with the stable toolchain was
    // built with `panic=unwind`, so it references `rust_eh_personality`.
    // We've set `panic=abort`, so no unwinding ever happens — the symbol just
    // needs to resolve. The nightly `build-std` path (used by the `./ltop`
    // wrapper) rebuilds alloc with panic=abort and drops the reference, so
    // this stub is only load-bearing on the plain `cargo build --release`.
    #[unsafe(no_mangle)]
    pub extern "C" fn rust_eh_personality() {}

    // LLVM's aarch64-apple-darwin codegen emits `bzero` calls for
    // zero-fills of small-to-medium anonymous storage (stack arrays,
    // struct-valued returns). `compiler-builtins-mem` supplies
    // memcpy/memmove/memcmp/memset/bcmp but not bzero, so an unresolved
    // `bzero` reference ends up as the sole import from libSystem.
    // Providing our own in-crate `bzero` lets the linker dead-strip
    // the whole lazy-binding machinery (dyld_stub_binder, __stubs,
    // __stub_helper, __la_symbol_ptr) that's there solely to bind
    // this one symbol.
    //
    // Body must NOT be `ptr::write_bytes(_, 0, _)` and must NOT be a
    // plain zero-fill loop: both patterns are what LLVM's loop-idiom
    // recognition turns back into `bzero` / `memset` calls. If our
    // bzero loops through `bzero`, the program hangs on the first
    // zero-fill. Volatile writes are not eligible for loop-idiom
    // replacement, so the store loop stays as emitted.
    #[cfg(target_os = "macos")]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn bzero(dst: *mut u8, len: usize) {
        let mut i = 0usize;
        // Byte-align up to an 8-byte boundary first: write_volatile
        // through a *mut u64 requires an aligned pointer per Rust's
        // rules, and LLVM's implicit bzero calls carry no alignment
        // guarantee (AArch64 hardware tolerates the misalignment, the
        // language does not). Mirrors the existing tail loop.
        while i < len && (dst as usize + i) & 7 != 0 {
            unsafe { core::ptr::write_volatile(dst.add(i), 0); }
            i += 1;
        }
        while i + 8 <= len {
            unsafe {
                core::ptr::write_volatile(dst.add(i) as *mut u64, 0);
            }
            i += 8;
        }
        while i < len {
            unsafe { core::ptr::write_volatile(dst.add(i), 0); }
            i += 1;
        }
    }
}
