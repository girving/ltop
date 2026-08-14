//! Custom process entry. Replaces the standard `crt1.o` /
//! `__libc_start_main` path — the production build links no libc at
//! all, so there's no init machinery (TLS setup, atexit, stdio
//! bootstrap) to call into. We do the minimal prep ourselves:
//!
//!   1. `_start` (asm) hands the initial SP to `ltop_entry`.
//!   2. `ltop_entry` (on the kernel-placed stack):
//!        a. Parses argv for `--check` / `--once` and packs the flags
//!           into a u64 while the kernel-populated argv region is
//!           still live (it's about to be madvise'd away).
//!        b. Walks up from the current sp-page with
//!           `madvise(DONTNEED)` — each call either drops a page in
//!           the stack VMA and continues, or returns `-ENOMEM` and
//!           tells us we've stepped past `vma_high`. The last
//!           success address + PAGE is the top of the VMA.
//!        c. Tail-jumps via inline asm to `ltop_main` on a fresh
//!           rsp set to `vma_high - 16` (16 bytes below a page
//!           boundary, preserving SysV/AAPCS alignment). `flags`
//!           travels in the first-arg register.
//!   3. `ltop_main` (running near the top of its page on the new sp):
//!        a. `madvise(DONTNEED)` walks downward from the current
//!           sp-page until `-ENOMEM`, dropping everything including
//!           the kernel's original sp-page that `ltop_entry` was
//!           using. After this, only our current page is committed.
//!        b. Calls `run(flags)`.
//!        c. On return, `exit_group(0)`.
//!
//! Net effect: with a peak stack use < 4080 B (enforced by
//! `tests/stack_frames.rs::no_frame_hits_stack_probe_threshold`),
//! every tick fits inside a single page of stack RSS.
//! `tests/stack_frames.rs::steady_state_stack_rss_one_page` asserts
//! the 4 KB ceiling at runtime from `/proc/PID/smaps`.
//!
//! TLS is never initialised, so any callee that touches `%fs`
//! (`__errno_location`, libc wrappers that write errno) would segfault.
//! Every such site was migrated to `syscall::*` in the D-raw refactor,
//! and the final libc dependency (`qsort`) was replaced by
//! `src/sort.rs`'s heapsort.

#![cfg(all(not(test), libc_free))]

use crate::syscall;

// ── assembly _start ─────────────────────────────────────────────────────────
//
// On entry (both archs): the kernel hands us `sp` pointing at argc,
// with argv/envp/auxv laid out just above. We stash `sp` into the
// first-argument register and call `ltop_entry`. If it ever returns
// (it shouldn't — it `!`s), we tail-issue `exit_group(0)` ourselves.

// x86_64: SysV AMD64 ABI wants rsp % 16 == 0 at a CALL target (i.e.
// rsp+8 aligned for the pushed return address). Kernel delivers
// rsp aligned to 16, so rounding down preserves that. Syscall number
// 231 = SYS_EXIT_GROUP.
#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".global ltop_start",
    "ltop_start:",
    "    mov rdi, rsp",
    "    and rsp, -16",
    "    call ltop_entry",
    "    mov rax, 231",
    "    xor rdi, rdi",
    "    syscall",
);

// aarch64: AAPCS64 requires sp % 16 == 0 at every procedure call;
// the kernel delivers sp aligned to 16 on process entry, so we don't
// re-align (and couldn't easily anyway: aarch64 `and` with `sp` as
// an operand isn't encodeable without a temp, since sp is register
// 31 aliased to xzr outside of the dedicated `sp`-taking variants).
// `bl` is branch-and-link, the aarch64 call instruction. Syscall
// number 94 = SYS_EXIT_GROUP on aarch64 (asm-generic-ish table); the
// number lives in x8, args in x0..x5, and `svc #0` traps to the
// kernel.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".global ltop_start",
    "ltop_start:",
    "    mov x0, sp",
    "    bl ltop_entry",
    "    mov x8, #94",
    "    mov x0, #0",
    "    svc #0",
);

// ── Rust entry: runs on the kernel-provided stack ──────────────────────────

/// Called by `_start` with `sp` pointing at argc. Parses argv while
/// the kernel-placed region is still live, then probes upward with
/// `madvise` to find the top of the stack VMA and tail-jumps to
/// [`ltop_main`] on a fresh sp 16 bytes below that page top.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ltop_entry(sp: *const i64) -> ! {
    // We skip `__libc_start_main`, so nothing walks `.init_array`.
    // Today the section is absent entirely (we drop the self-contained
    // crt bundle); historically it held one entry — a libc ctor that
    // registered `.eh_frame` for unwinding we don't do (`panic=abort`
    // + `-Cforce-unwind-tables=no`). If a future dep sneaks in a real
    // global constructor we'd silently skip it;
    // `tests/stack_frames.rs::init_array_minimal` catches that at
    // test time.

    let argc = unsafe { *sp } as i32;
    let argv = unsafe { sp.add(1) as *const *const u8 };

    // Parse argv flags WHILE the kernel-supplied argv is still valid.
    // `--check` (handled inside `parse_args`) exits before we touch
    // anything else.
    let flags = unsafe { crate::parse_args(argc, argv) };

    // Read AT_PAGESZ from auxv — it lies past argv (argc entries +
    // NULL) and envp (up to its NULL) as (a_type, a_val) i64 pairs,
    // AT_NULL-terminated. This must happen BEFORE the madvise walk
    // below: probe addresses have to be aligned to the real kernel
    // page size (aarch64 kernels ship 4 K, 16 K on Asahi, 64 K on
    // RHEL-alt; a 4 K-aligned probe on a 16 K kernel is EINVAL, which
    // the walk would misread as the VMA boundary).
    let mut q = unsafe { sp.add(1 + argc as usize + 1) };
    while unsafe { *q } != 0 { q = unsafe { q.add(1) }; }
    let mut aux = unsafe { q.add(1) };
    loop {
        let key = unsafe { *aux };
        if key == 0 { break; }
        if key == crate::syscall::page::AT_PAGESZ {
            crate::syscall::page::set(unsafe { *aux.add(1) } as u64);
            break;
        }
        aux = unsafe { aux.add(2) };
    }
    let page = crate::syscall::page::get() as u64;

    // Walk upward from sp's page, dropping each mapped page via
    // `madvise(DONTNEED)`. The last success address is the top page
    // of the stack VMA; the first failure terminates. Since nothing
    // maps within reach of [stack] on Linux (kernel `stack_guard_gap`
    // + our no-mmap binary), the first ENOMEM reliably signals the
    // VMA end — not a walk into something adjacent. The pages we just
    // dropped are zero-filled-on-demand the moment we touch them
    // below (via the tail-jump), so the content going to zero here
    // is exactly what we want.
    let sp_page = sp as u64 & !(page - 1);
    let mut last_mapped = sp_page;
    let mut p = sp_page + page;
    while drop_one(p, page) { last_mapped = p; p += page; }

    // New rsp: 16 B below the top of the last mapped page. SysV /
    // AAPCS want sp 16-aligned at a call target; `top - 16` puts us
    // there with page_size - 16 bytes of headroom (≥ 4080 B), enough
    // for ltop's peak 2 KB stack use to fit inside one page.
    let new_rsp = last_mapped + page - 16;

    // Tail-jump to ltop_main on the fresh rsp. `flags` rides across
    // in the first-arg register (rdi / x0) so we don't have to stash
    // it in memory — the old stack is about to be unreachable.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "mov rsp, {rsp}",
            "push 0",            // dummy return slot so rsp % 16 == 8 at the jmp target
            "jmp {target}",
            rsp = in(reg) new_rsp,
            target = sym ltop_main,
            in("rdi") flags,
            // No `nostack`: the block pushes a dummy return slot, and
            // promising the compiler we don't touch the stack while
            // executing `push` is contract-UB even though control never
            // returns (audit finding). `noreturn` alone loses nothing.
            options(noreturn),
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "mov sp, {rsp}",
            "b {target}",
            rsp = in(reg) new_rsp,
            target = sym ltop_main,
            in("x0") flags,
            options(noreturn, nostack),
        );
    }
}

/// Called via inline-asm tail-jump from [`ltop_entry`] with rsp
/// set to 16 B below the top of the stack VMA's top page. Drops
/// every mapped page below our current sp-page — including the
/// kernel's original sp-page that ltop_entry just vacated — then
/// enters `run`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ltop_main(flags: u64) -> ! {
    // Read current rsp; the jmp target's prologue may have adjusted
    // it down a few bytes for locals, so we compute sp_page from the
    // current value rather than trusting a compile-time offset.
    let cur_sp: u64;
    #[cfg(target_arch = "x86_64")]
    unsafe { core::arch::asm!("mov {}, rsp", out(reg) cur_sp, options(nomem, nostack, preserves_flags)); }
    #[cfg(target_arch = "aarch64")]
    unsafe { core::arch::asm!("mov {}, sp", out(reg) cur_sp, options(nomem, nostack, preserves_flags)); }

    // Walk downward dropping every mapped page until ENOMEM. This
    // sweeps through the old kernel sp-page (ltop_entry's frame) and
    // any intermediate pages that were never madvise'd during the
    // upward walk.
    let page = crate::syscall::page::get() as u64; // stored by ltop_entry
    let mut p = (cur_sp & !(page - 1)).wrapping_sub(page);
    while p != 0 && drop_one(p, page) { p = p.wrapping_sub(page); }

    super::run(flags);
    syscall::exit_group(0)
}

/// `madvise(DONTNEED)` one page. Returns `true` if the page was
/// mapped (drop succeeded), `false` on `-ENOMEM` (unmapped, i.e.
/// we've walked past a VMA boundary).
#[inline]
fn drop_one(addr: u64, page: u64) -> bool {
    unsafe { syscall::madvise(addr as *mut u8, page as usize, syscall::MADV_DONTNEED) == 0 }
}
