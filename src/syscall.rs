//! System-call wrappers. On Linux these are raw `syscall` instructions
//! — the production build links no libc, so going through libc
//! wrappers (which touch TLS for errno) isn't an option. On every
//! other target they delegate to libc: the macOS and test builds
//! still rely on libc-maintained TLS, but neither participates in
//! the D-raw RSS-tuning path.
//!
//! Call-site rule: always go through `crate::syscall::*`, never through
//! `libc::*` directly. Cross-platform conditionals live in one place
//! here, not at every use site.
//!
//! Return-value convention: the raw syscall returns in `rax`.
//!   * `ret >= 0` → success, `ret` is the return value.
//!   * `ret < 0`  → `errno = -ret`; wrappers return the negative i64 so
//!                  callers can inspect the error directly (we never
//!                  need `strerror`).
//!
//! The `syscall` instruction clobbers `rcx` (kernel saves user RIP
//! there) and `r11` (flags). We declare both as `lateout("…") _` so
//! the compiler doesn't rely on them surviving.

// Rust 2024 tightens unsafe-fn bodies to require explicit `unsafe { … }`
// blocks around unsafe ops; every wrapper below is `unsafe fn` and its
// body is entirely raw-syscall or libc calls. Rather than litter each
// one with a single-line `unsafe { … }`, we accept the lint at module
// scope — the function signature is already `unsafe fn`, so the caller
// has already shouldered the soundness argument.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

// ── common constants (Linux-flavoured; macOS ignores via cfg) ──────────────

/// `struct timespec` used by `clock_gettime` / `nanosleep`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}
impl Timespec {
    pub const fn zero() -> Self {
        Self { tv_sec: 0, tv_nsec: 0 }
    }
}

// CLOCK_* IDs are ABI-fixed; hardcoding them on Linux + Darwin keeps
// libc out of the production build entirely (on macOS the libc crate
// declares `#[link(name = "iconv")]` which forces an LC_LOAD_DYLIB
// for libiconv even when zero symbols from it are imported, since
// ld64 has no `--as-needed`). `CLOCK_MONOTONIC` differs between
// kernels: 6 on Darwin, 1 on Linux.
#[cfg(target_os = "linux")]
pub const CLOCK_REALTIME: i32 = 0;
#[cfg(target_os = "linux")]
pub const CLOCK_MONOTONIC: i32 = 1;
#[cfg(target_os = "macos")]
pub const CLOCK_REALTIME: i32 = 0;
#[cfg(target_os = "macos")]
pub const CLOCK_MONOTONIC: i32 = 6;
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub const CLOCK_REALTIME: i32 = libc::CLOCK_REALTIME as i32;
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub const CLOCK_MONOTONIC: i32 = libc::CLOCK_MONOTONIC as i32;

pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_RDWR: i32 = 2;
pub const O_CLOEXEC: i32 = 0o2000000;

// O_DIRECTORY is *not* asm-generic: Linux's arm/arm64 uapi headers
// override it to octal 040000, while x86_64 (and the rest of
// asm-generic) uses 0200000. Passing the wrong bit gives -EINVAL
// from openat, not -ENOTDIR, so it's easy to misdiagnose.
#[cfg(any(not(target_arch = "aarch64"), not(target_os = "linux")))]
pub const O_DIRECTORY: i32 = 0o200000;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub const O_DIRECTORY: i32 = 0o40000;

pub const MADV_DONTNEED: i32 = 4;
// Darwin's `MADV_FREE_REUSABLE` is the analogue of Linux's
// `MADV_DONTNEED`: actively reclaims dirty pages and lets future
// access zero-fault. Plain `MADV_FREE` (= 5) on Darwin only marks
// pages as freeable — the kernel waits for memory pressure before
// actually dropping them, so they keep showing up in `vmmap` as
// dirty until something else needs the RAM. `MADV_DONTNEED` (= 4)
// on Darwin is a hint with even weaker semantics.
#[cfg(target_os = "macos")]
pub const MADV_FREE_REUSABLE: i32 = 7;

// Linux terminal ioctls.
#[cfg(target_os = "linux")]
pub const TCGETS: u64 = 0x5401;
#[cfg(target_os = "linux")]
pub const TCSETS: u64 = 0x5402;
// macOS terminal ioctls. Numbers encode direction + sizeof(termios) + group
// ('t' = 0x74) + command in the standard IOWR macro layout.
// sizeof(struct termios) = 72 on arm64 (LP64: tcflag_t = u64).
#[cfg(target_os = "macos")]
pub const TIOCGETA: u64 = 0x40487413; // TIOCGETA: read termios
#[cfg(target_os = "macos")]
pub const TIOCSETA: u64 = 0x80487414; // TIOCSETA: set termios immediately
#[cfg(target_os = "linux")]
pub const TIOCGWINSZ: u64 = 0x5413;
// macOS: struct winsize is 4 × u16 = 8 bytes; TIOCGWINSZ = _IOR('t',104,struct winsize)
#[cfg(target_os = "macos")]
pub const TIOCGWINSZ: u64 = 0x40087468;

// `AT_FDCWD` differs by ABI: Linux uses -100, the BSD ABI macOS
// inherits uses -2. Only `openat` consumes it — on mac that's the
// `arena-trace` log create (the read path goes through `SYS_OPEN`),
// which is why a wrong value here stays invisible in production.
#[cfg(not(target_os = "macos"))]
pub const AT_FDCWD: i32 = -100;
#[cfg(target_os = "macos")]
pub const AT_FDCWD: i32 = -2;

// ── Linux: raw `syscall` instruction ────────────────────────────────────────
//
// Per-arch submodules define the syscall numbers and the `syscallN`
// inline-asm primitives; `pub use arch::*` lets the rest of the file
// reference `linux::SYS_*` / `linux::syscallN` arch-agnostically.
//
// Note: aarch64 has no `SYS_OPEN`. Both archs have `SYS_OPENAT`, so
// the public `open` / `open_create` wrappers below route through
// `openat(AT_FDCWD, …)` on every arch — one unified code path, and
// the x86_64 cost is a single extra argument register per call.

#[cfg(target_os = "linux")]
mod linux {
    #[cfg(target_arch = "x86_64")] pub use x86_64::*;
    #[cfg(target_arch = "aarch64")] pub use aarch64::*;

    // Linux x86_64 syscall ABI: number in rax; args in rdi/rsi/rdx/r10/r8/r9.
    // The `syscall` instruction clobbers rcx (kernel saves user RIP there)
    // and r11 (flags). arg4 is r10 (not rcx) specifically because of that
    // clobber.
    #[cfg(target_arch = "x86_64")]
    mod x86_64 {
        use core::arch::asm;

        pub const SYS_READ: i64 = 0;
        pub const SYS_WRITE: i64 = 1;
        pub const SYS_CLOSE: i64 = 3;
        pub const SYS_IOCTL: i64 = 16;
        pub const SYS_MADVISE: i64 = 28;
        pub const SYS_NANOSLEEP: i64 = 35;
        pub const SYS_GETPID: i64 = 39;
        pub const SYS_UNAME: i64 = 63;
        pub const SYS_EXIT_GROUP: i64 = 231;
        pub const SYS_CLOCK_GETTIME: i64 = 228;
        pub const SYS_OPENAT: i64 = 257;
        pub const SYS_SCHED_GETAFFINITY: i64 = 204;
        pub const SYS_GETDENTS64: i64 = 217;
        pub const SYS_PRCTL: i64 = 157;
        pub const SYS_SECCOMP: i64 = 317;
        pub const SYS_RT_SIGACTION: i64 = 13;
        // Landlock — same nrs on every Linux arch (5.13+).
        pub const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
        pub const SYS_LANDLOCK_ADD_RULE: i64 = 445;
        pub const SYS_LANDLOCK_RESTRICT_SELF: i64 = 446;

        #[inline]
        pub unsafe fn syscall1(n: i64, a1: i64) -> i64 {
            let ret: i64;
            asm!(
                "syscall",
                inlateout("rax") n => ret,
                in("rdi") a1,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack, preserves_flags),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall2(n: i64, a1: i64, a2: i64) -> i64 {
            let ret: i64;
            asm!(
                "syscall",
                inlateout("rax") n => ret,
                in("rdi") a1,
                in("rsi") a2,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack, preserves_flags),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall3(n: i64, a1: i64, a2: i64, a3: i64) -> i64 {
            let ret: i64;
            asm!(
                "syscall",
                inlateout("rax") n => ret,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack, preserves_flags),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall4(n: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
            let ret: i64;
            asm!(
                "syscall",
                inlateout("rax") n => ret,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                in("r10") a4,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack, preserves_flags),
            );
            ret
        }

        // 5-arg form needed for `prctl(option, arg2, arg3, arg4, arg5)` —
        // the sandbox install path's only 5-arg syscall.
        #[inline]
        pub unsafe fn syscall5(n: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64) -> i64 {
            let ret: i64;
            asm!(
                "syscall",
                inlateout("rax") n => ret,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                in("r10") a4,
                in("r8") a5,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack, preserves_flags),
            );
            ret
        }
    }

    // Linux aarch64 syscall ABI: number in x8; args in x0..x5; return in x0.
    // Plain `in(...)` on the argument registers is sound here because the
    // kernel restores x1..x7 from the saved pt_regs on every syscall return
    // (arch/arm64/kernel/entry.S kernel_exit) — only x0 carries a result,
    // and it's declared as an output. Note `in(...)` promises the asm block
    // preserves the register; it does NOT mark it clobbered. Syscall numbers
    // match Linux's asm-generic table (same as riscv64, loongarch64); they
    // are *not* the same as x86_64's.
    #[cfg(target_arch = "aarch64")]
    mod aarch64 {
        use core::arch::asm;

        pub const SYS_READ: i64 = 63;
        pub const SYS_WRITE: i64 = 64;
        pub const SYS_CLOSE: i64 = 57;
        pub const SYS_IOCTL: i64 = 29;
        pub const SYS_MADVISE: i64 = 233;
        pub const SYS_NANOSLEEP: i64 = 101;
        pub const SYS_GETPID: i64 = 172;
        pub const SYS_UNAME: i64 = 160;
        pub const SYS_EXIT_GROUP: i64 = 94;
        pub const SYS_CLOCK_GETTIME: i64 = 113;
        pub const SYS_OPENAT: i64 = 56;
        pub const SYS_SCHED_GETAFFINITY: i64 = 123;
        pub const SYS_GETDENTS64: i64 = 61;
        pub const SYS_PRCTL: i64 = 167;
        pub const SYS_SECCOMP: i64 = 277;
        pub const SYS_RT_SIGACTION: i64 = 134;
        // Landlock — same nrs on every Linux arch (5.13+).
        pub const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
        pub const SYS_LANDLOCK_ADD_RULE: i64 = 445;
        pub const SYS_LANDLOCK_RESTRICT_SELF: i64 = 446;

        #[inline]
        pub unsafe fn syscall1(n: i64, a1: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0",
                in("x8") n,
                inlateout("x0") a1 => ret,
                options(nostack, preserves_flags),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall2(n: i64, a1: i64, a2: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0",
                in("x8") n,
                inlateout("x0") a1 => ret,
                in("x1") a2,
                options(nostack, preserves_flags),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall3(n: i64, a1: i64, a2: i64, a3: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0",
                in("x8") n,
                inlateout("x0") a1 => ret,
                in("x1") a2,
                in("x2") a3,
                options(nostack, preserves_flags),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall4(n: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0",
                in("x8") n,
                inlateout("x0") a1 => ret,
                in("x1") a2,
                in("x2") a3,
                in("x3") a4,
                options(nostack, preserves_flags),
            );
            ret
        }

        // 5-arg form for `prctl`. See the x86_64 sibling above.
        #[inline]
        pub unsafe fn syscall5(n: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0",
                in("x8") n,
                inlateout("x0") a1 => ret,
                in("x1") a2,
                in("x2") a3,
                in("x3") a4,
                in("x4") a5,
                options(nostack, preserves_flags),
            );
            ret
        }
    }
}

// Darwin aarch64 syscall ABI: number in x16; args in x0..x5; `svc #0x80`
// traps into the kernel. Return in x0. Unlike Linux, Darwin signals
// errors via the carry flag (CF=1 → x0 holds errno, CF=0 → x0 is the
// success value). We fold that into Linux-shape negative-errno with a
// `b.cc`-guarded `neg x0, x0`. So from the Rust side every Darwin
// syscall looks identical to a Linux one: non-negative result on
// success, -errno on error.
//
// Unlike Linux, xnu writes x1 on every BSD syscall return
// (bsd/dev/arm/systemcalls.c arm_prepare_u64_syscall_return: 0 on
// error; uu_rval[1] or 0 on success, by return type), so every BSD
// wrapper must declare x1 as clobbered — the register does NOT survive
// the `svc`. Mach traps write only x0 (osfmk/arm64/sleh.c), but these
// wrappers also serve the traps above, and the extra clobber is free.
//
// Syscall numbers come from xnu's `bsd/kern/syscalls.master`. Apple's
// stance is "libSystem is the ABI, syscall numbers are not" — but in
// practice these have been stable since Mac OS X 10.0. Still, this is
// the non-supported path; going off-libSystem is a deliberate tradeoff
// for binary size (see the "Flavor A" discussion).
#[cfg(target_os = "macos")]
mod darwin {
    #[cfg(target_arch = "aarch64")] pub use aarch64::*;

    #[cfg(target_arch = "aarch64")]
    mod aarch64 {
        use core::arch::asm;

        // Darwin terminates the whole process with `SYS_exit` (no separate
        // `exit_group`). We alias it as `SYS_EXIT_GROUP` for parity with
        // the Linux submodule so the public surface doesn't need to
        // special-case the name.
        pub const SYS_EXIT_GROUP:  i64 = 1;
        pub const SYS_READ:        i64 = 3;
        pub const SYS_WRITE:       i64 = 4;
        pub const SYS_OPEN:        i64 = 5;
        pub const SYS_CLOSE:       i64 = 6;
        pub const SYS_GETPID:      i64 = 20;
        pub const SYS_IOCTL:       i64 = 54;
        pub const SYS_MADVISE:     i64 = 75;
        pub const SYS_GETTIMEOFDAY:i64 = 116;
        pub const SYS_SYSCTL:      i64 = 202;
        pub const SYS_PROC_INFO:   i64 = 336;
        pub const SYS_OPENAT:      i64 = 463;

        // Mach traps (negative syscall numbers in x16 route through the
        // Mach-trap table instead of the BSD-syscall table).
        pub const MACH_ABS_TIME_TRAP:      i64 = -3;
        pub const MACH_TIMEBASE_INFO_TRAP: i64 = -89;
        pub const MACH_WAIT_UNTIL_TRAP:    i64 = -90;

        #[inline]
        pub unsafe fn syscall1(n: i64, a1: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0x80",
                "b.cc 1f",
                "neg x0, x0",
                "1:",
                in("x16") n,
                inlateout("x0") a1 => ret,
                lateout("x1") _,
                options(nostack),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall2(n: i64, a1: i64, a2: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0x80",
                "b.cc 1f",
                "neg x0, x0",
                "1:",
                in("x16") n,
                inlateout("x0") a1 => ret,
                inlateout("x1") a2 => _,
                options(nostack),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall3(n: i64, a1: i64, a2: i64, a3: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0x80",
                "b.cc 1f",
                "neg x0, x0",
                "1:",
                in("x16") n,
                inlateout("x0") a1 => ret,
                inlateout("x1") a2 => _,
                in("x2") a3,
                options(nostack),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall4(n: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0x80",
                "b.cc 1f",
                "neg x0, x0",
                "1:",
                in("x16") n,
                inlateout("x0") a1 => ret,
                inlateout("x1") a2 => _,
                in("x2") a3,
                in("x3") a4,
                options(nostack),
            );
            ret
        }

        #[inline]
        pub unsafe fn syscall6(n: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64) -> i64 {
            let ret: i64;
            asm!(
                "svc #0x80",
                "b.cc 1f",
                "neg x0, x0",
                "1:",
                in("x16") n,
                inlateout("x0") a1 => ret,
                inlateout("x1") a2 => _,
                in("x2") a3,
                in("x3") a4,
                in("x4") a5,
                in("x5") a6,
                options(nostack),
            );
            ret
        }
    }
}

// ── public cross-platform surface ───────────────────────────────────────────

#[inline]
pub unsafe fn read(fd: i32, buf: *mut u8, len: usize) -> i64 {
    #[cfg(target_os = "linux")]
    { linux::syscall3(linux::SYS_READ, fd as i64, buf as i64, len as i64) }
    #[cfg(target_os = "macos")]
    { darwin::syscall3(darwin::SYS_READ, fd as i64, buf as i64, len as i64) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::read(fd, buf as *mut _, len) as i64 }
}

#[inline]
pub unsafe fn write(fd: i32, buf: *const u8, len: usize) -> i64 {
    #[cfg(target_os = "linux")]
    { linux::syscall3(linux::SYS_WRITE, fd as i64, buf as i64, len as i64) }
    #[cfg(target_os = "macos")]
    { darwin::syscall3(darwin::SYS_WRITE, fd as i64, buf as i64, len as i64) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::write(fd, buf as *const _, len) as i64 }
}

/// Safe wrapper around `write`: takes a slice so the pointer + length
/// come from one place and can't disagree. Always inlined so callers
/// pay nothing extra over the raw pointer form. Prefer this at call
/// sites — the only reason to use `write` directly is a partial-write
/// loop that advances into the middle of a buffer.
///
/// Returns the syscall's return value: number of bytes written on
/// success, or a negative `-errno` on failure.
#[inline(always)]
pub fn write_all(fd: i32, buf: &[u8]) -> i64 {
    // SAFETY: `buf` is a live slice, so its pointer is valid for `buf.len()`
    // readable bytes. The kernel write syscall is safe from Rust's POV —
    // the worst it can do on bad args is return an error.
    unsafe { write(fd, buf.as_ptr(), buf.len()) }
}

/// Safe wrapper around `read`: takes a slice so the pointer + length
/// come from one place and can't disagree. Returns the syscall's
/// return value (bytes read on success, 0 at EOF, negative `-errno`).
#[inline(always)]
pub fn read_buf(fd: i32, buf: &mut [u8]) -> i64 {
    unsafe { read(fd, buf.as_mut_ptr(), buf.len()) }
}

/// `open(path, flags)` — path is a NUL-terminated byte pointer. Returns
/// fd on success or a negative i32 containing `-errno`. On Linux this
/// routes through `openat(AT_FDCWD, …)` because aarch64 has no
/// `SYS_OPEN`; the x86_64 kernel is just as happy with either entry,
/// so keeping the two archs on the same syscall avoids a per-arch
/// dispatch here.
#[inline]
pub unsafe fn open(path: *const u8, flags: i32) -> i32 {
    #[cfg(target_os = "linux")]
    { openat(AT_FDCWD, path, flags) }
    // Darwin `open(path, flags, mode)` is a 3-arg call; mode is ignored
    // without `O_CREAT`, so passing 0 is fine here.
    #[cfg(target_os = "macos")]
    { darwin::syscall3(darwin::SYS_OPEN, path as i64, flags as i64, 0) as i32 }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::open(path as *const _, flags) }
}

/// Safe wrapper: open a NUL-terminated path given as `&CStr`. Most
/// callers prefer this — the `&CStr` type makes NUL-termination a
/// compile-time property so the raw-pointer call site doesn't need its
/// own `unsafe` block.
#[inline]
pub fn open_cstr(path: &core::ffi::CStr, flags: i32) -> i32 {
    unsafe { open(path.as_ptr() as *const u8, flags) }
}

#[inline]
pub unsafe fn openat(dirfd: i32, path: *const u8, flags: i32) -> i32 {
    #[cfg(target_os = "linux")]
    { linux::syscall3(linux::SYS_OPENAT, dirfd as i64, path as i64, flags as i64) as i32 }
    #[cfg(target_os = "macos")]
    { darwin::syscall4(darwin::SYS_OPENAT, dirfd as i64, path as i64, flags as i64, 0) as i32 }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::openat(dirfd, path as *const _, flags) }
}

/// Safe wrapper: openat with a `&CStr` path.
#[inline]
pub fn openat_cstr(dirfd: i32, path: &core::ffi::CStr, flags: i32) -> i32 {
    unsafe { openat(dirfd, path.as_ptr() as *const u8, flags) }
}

/// `open(path, flags | O_CREAT, mode)` — four-arg kernel form needed
/// when the file must be created. Goes through `SYS_OPENAT` on both
/// archs (aarch64 has no `SYS_OPEN`) as a 4-arg syscall:
/// `openat(AT_FDCWD, path, flags, mode)`. Used by `arena-trace` to
/// create its log.
#[inline]
pub unsafe fn open_create(path: *const u8, flags: i32, mode: u32) -> i32 {
    #[cfg(target_os = "linux")]
    {
        linux::syscall4(
            linux::SYS_OPENAT,
            AT_FDCWD as i64,
            path as i64,
            flags as i64,
            mode as i64,
        ) as i32
    }
    #[cfg(target_os = "macos")]
    {
        darwin::syscall4(
            darwin::SYS_OPENAT,
            AT_FDCWD as i64,
            path as i64,
            flags as i64,
            mode as i64,
        ) as i32
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::open(path as *const _, flags, mode) }
}

/// `O_CREAT` / `O_TRUNC` — used with `open_create`. The bit values
/// differ between the Linux ABI (`asm-generic/fcntl.h`) and the BSD
/// ABI macOS inherits (`sys/fcntl.h`): Linux `O_CREAT` is `0o100`,
/// macOS `0x0200`. Get this wrong on mac and `open_create` passes
/// stray flags (Linux `O_CREAT` 0x40 lands on mac `O_ASYNC`), so the
/// arena-trace log silently never gets created.
#[cfg(not(target_os = "macos"))]
pub const O_CREAT: i32 = 0o100;
#[cfg(not(target_os = "macos"))]
pub const O_TRUNC: i32 = 0o1000;
#[cfg(target_os = "macos")]
pub const O_CREAT: i32 = 0x0200;
#[cfg(target_os = "macos")]
pub const O_TRUNC: i32 = 0x0400;

/// Safe: takes an integer fd, no pointers. Returns the kernel's status
/// code (0 on success, -errno otherwise); most callers discard it.
#[inline]
pub fn close(fd: i32) -> i32 {
    #[cfg(target_os = "linux")]
    { unsafe { linux::syscall1(linux::SYS_CLOSE, fd as i64) as i32 } }
    #[cfg(target_os = "macos")]
    { unsafe { darwin::syscall1(darwin::SYS_CLOSE, fd as i64) as i32 } }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { unsafe { libc::close(fd) } }
}

/// `ioctl(fd, cmd, arg)` — on x86_64 Linux ioctl is a 3-arg syscall; the
/// third argument carries whatever pointer/int the command expects.
#[inline]
pub unsafe fn ioctl(fd: i32, cmd: u64, arg: *mut core::ffi::c_void) -> i64 {
    #[cfg(target_os = "linux")]
    { linux::syscall3(linux::SYS_IOCTL, fd as i64, cmd as i64, arg as i64) }
    #[cfg(target_os = "macos")]
    { darwin::syscall3(darwin::SYS_IOCTL, fd as i64, cmd as i64, arg as i64) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::ioctl(fd, cmd as _, arg) as i64 }
}

#[inline]
pub unsafe fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32 {
    #[cfg(target_os = "linux")]
    { linux::syscall3(linux::SYS_MADVISE, addr as i64, len as i64, advice as i64) as i32 }
    #[cfg(target_os = "macos")]
    { darwin::syscall3(darwin::SYS_MADVISE, addr as i64, len as i64, advice as i64) as i32 }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::madvise(addr as *mut _, len, advice) }
}

#[inline]
pub unsafe fn nanosleep(req: *const Timespec, rem: *mut Timespec) -> i32 {
    #[cfg(target_os = "linux")]
    { linux::syscall2(linux::SYS_NANOSLEEP, req as i64, rem as i64) as i32 }
    #[cfg(target_os = "macos")]
    { darwin_nanosleep(req, rem) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::nanosleep(req as *const _, rem as *mut _) }
}

#[inline]
pub unsafe fn clock_gettime(clk: i32, ts: *mut Timespec) -> i32 {
    #[cfg(target_os = "linux")]
    { linux::syscall2(linux::SYS_CLOCK_GETTIME, clk as i64, ts as i64) as i32 }
    #[cfg(target_os = "macos")]
    { darwin_clock_gettime(clk, ts) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { libc::clock_gettime(clk as _, ts as *mut _) }
}

// ── Darwin time primitives (Flavor A, phase 4) ─────────────────────────────
//
// clock_gettime on modern Darwin isn't a BSD syscall — libSystem reads
// the kernel commpage directly. For our libSystem-free build we
// reconstruct both clocks from Mach traps + a BSD syscall:
//
//   CLOCK_MONOTONIC  →  MACH_ARM_TRAP_ABSTIME (-3) for ticks,
//                       mach_timebase_info_trap (-89) for ticks→ns ratio
//                       (cached after first call since timebase is fixed)
//   CLOCK_REALTIME   →  gettimeofday syscall (116), 3 args:
//                       (timeval*, timezone*, mach_abs_time*). Seconds
//                       component is all we need; we pass null for the
//                       optional second & third args.
//
// nanosleep is backed by mach_wait_until_trap (-90), which takes an
// absolute deadline in Mach ticks: current_ticks + (duration_ns *
// denom / numer).

#[cfg(target_os = "macos")]
#[inline]
unsafe fn mach_absolute_time() -> u64 {
    use core::arch::asm;
    let ret: u64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") darwin::MACH_ABS_TIME_TRAP,
            lateout("x0") ret,
            options(nostack),
        );
    }
    ret
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct MachTimebaseInfo { numer: u32, denom: u32 }

#[cfg(target_os = "macos")]
#[inline]
unsafe fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32 {
    use core::arch::asm;
    let ret: i64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") darwin::MACH_TIMEBASE_INFO_TRAP,
            inlateout("x0") info as u64 => ret,
            options(nostack),
        );
    }
    ret as i32
}

#[cfg(target_os = "macos")]
#[inline]
unsafe fn mach_wait_until(deadline: u64) -> i32 {
    use core::arch::asm;
    let ret: i64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") darwin::MACH_WAIT_UNTIL_TRAP,
            inlateout("x0") deadline => ret,
            options(nostack),
        );
    }
    ret as i32
}

/// Mach absolute-time ticks per second. Used to convert `pti_total_user` /
/// `pti_total_system` (TASK_ABSOLUTETIME_INFO units, same as mach_absolute_time)
/// into CPU fractions. On M1+ the timebase is {numer=125, denom=3}, giving a
/// 24 MHz clock (1e9 * 3/125 = 24e6 ticks/s). Falls back to 1 GHz (1:1) if
/// the syscall fails, which matches the nanosecond interpretation used as a
/// prior assumption.
#[cfg(target_os = "macos")]
pub fn mach_ticks_per_sec() -> f64 {
    let (numer, denom) = cached_timebase();
    1_000_000_000.0 * denom as f64 / numer as f64
}

/// Cached (numer, denom) of `mach_timebase_info`, packed as
/// `(numer as u64) | ((denom as u64) << 32)`. Queried once per process.
#[cfg(target_os = "macos")]
fn cached_timebase() -> (u32, u32) {
    use core::sync::atomic::{AtomicU64, Ordering};
    static TIMEBASE: AtomicU64 = AtomicU64::new(0);
    let p = TIMEBASE.load(Ordering::Relaxed);
    if p != 0 {
        return (p as u32, (p >> 32) as u32);
    }
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    unsafe { mach_timebase_info(&mut info); }
    if info.numer == 0 || info.denom == 0 {
        return (1, 1); // fall back to 1 tick = 1 ns
    }
    let packed = info.numer as u64 | ((info.denom as u64) << 32);
    TIMEBASE.store(packed, Ordering::Relaxed);
    (info.numer, info.denom)
}

#[cfg(target_os = "macos")]
unsafe fn darwin_clock_gettime(clk: i32, ts: *mut Timespec) -> i32 {
    // Matches the cfg-gated constants at the top of this file: on mac
    // CLOCK_REALTIME=0 and CLOCK_MONOTONIC=6 (libc values).
    if clk == CLOCK_MONOTONIC {
        let ticks = unsafe { mach_absolute_time() };
        let (numer, denom) = cached_timebase();
        // ns = ticks * numer / denom. Native u64 arithmetic.
        //
        // Was previously u128 to avoid overflow for huge uptimes; the
        // u128 path pulled in compiler_builtins' `__udivti3` (~740 B
        // of __text) because arm64 has no u128 division instruction.
        // With the real timebase ratio ({numer=125, denom=3} on M1+)
        // and a 24 MHz base clock, `ticks * numer` overflows u64
        // only past ~190 years of continuous uptime — not a real
        // threat model. Drop back to u64 and keep compiler_builtins
        // out of __text.
        let ns = ticks.wrapping_mul(numer as u64) / denom as u64;
        unsafe {
            (*ts).tv_sec  = (ns / 1_000_000_000) as i64;
            (*ts).tv_nsec = (ns % 1_000_000_000) as i64;
        }
        return 0;
    }
    if clk == CLOCK_REALTIME {
        // Darwin timeval on arm64: { tv_sec: i64, tv_usec: i32 } with 4
        // bytes of padding to 8-byte alignment (total 16 bytes).
        #[repr(C)]
        struct Timeval { tv_sec: i64, tv_usec: i32, _pad: u32 }
        let mut tv = Timeval { tv_sec: 0, tv_usec: 0, _pad: 0 };
        // __gettimeofday(&tv, NULL, NULL).
        let r = darwin::syscall3(
            darwin::SYS_GETTIMEOFDAY,
            &mut tv as *mut _ as i64,
            0, 0,
        ) as i32;
        if r < 0 { return r; }
        unsafe {
            (*ts).tv_sec  = tv.tv_sec;
            (*ts).tv_nsec = tv.tv_usec as i64 * 1000;
        }
        return 0;
    }
    -22 // EINVAL
}

#[cfg(target_os = "macos")]
unsafe fn darwin_nanosleep(req: *const Timespec, _rem: *mut Timespec) -> i32 {
    let req = unsafe { &*req };
    let duration_ns = (req.tv_sec as u64).saturating_mul(1_000_000_000)
        .saturating_add(req.tv_nsec as u64);
    let (numer, denom) = cached_timebase();
    // ticks = ns * denom / numer. u64 math — see darwin_clock_gettime
    // for the overflow argument (realistic sleep durations × small
    // denom stay well under u64 limits; u128 would drag __udivti3).
    let delta_ticks = duration_ns.wrapping_mul(denom as u64) / numer as u64;
    let now = unsafe { mach_absolute_time() };
    let deadline = now.saturating_add(delta_ticks as u64);
    unsafe { mach_wait_until(deadline) }
}

/// Safe wrapper: returns the clock's current value. Caller picks between
/// `CLOCK_REALTIME` (wall clock) and `CLOCK_MONOTONIC` (steady monotonic
/// since boot).
#[inline]
pub fn get_clock(clk: i32) -> Timespec {
    let mut ts = Timespec::zero();
    unsafe { clock_gettime(clk, &mut ts); }
    ts
}

/// Safe: no arguments, no pointers.
#[inline]
pub fn getpid() -> i32 {
    #[cfg(target_os = "linux")]
    { unsafe { linux::syscall1(linux::SYS_GETPID, 0) as i32 } }
    #[cfg(target_os = "macos")]
    { unsafe { darwin::syscall1(darwin::SYS_GETPID, 0) as i32 } }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { unsafe { libc::getpid() } }
}

/// Linux-only: `getdents64` streams directory entries into `buf`. Used
/// by `platform::PidDir` for /proc scanning. macOS uses a different
/// path (libproc) and doesn't reach this.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn getdents64(fd: i32, buf: *mut u8, len: usize) -> i64 {
    linux::syscall3(linux::SYS_GETDENTS64, fd as i64, buf as i64, len as i64)
}

/// Linux-only: `sched_getaffinity(0, size, set)` fills `set` with the
/// CPU affinity bitmask; popcount is the effective nproc.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn sched_getaffinity(pid: i32, size: usize, set: *mut u64) -> i64 {
    linux::syscall3(linux::SYS_SCHED_GETAFFINITY, pid as i64, size as i64, set as i64)
}

/// Linux-only: `prctl(option, arg2, arg3, arg4, arg5)`. Stays `unsafe`
/// because `prctl` is option-polymorphic: some `option` values treat
/// arg2..5 as pointers the kernel reads (`PR_SET_NAME`) or writes
/// (`PR_GET_NAME`, `PR_GET_TID_ADDRESS`, `PR_SET_MM`). A safe
/// signature here would let a caller pick `PR_GET_NAME` with an
/// arbitrary `u64` and have the kernel scribble into live Rust
/// memory — a soundness hole. Specific integer-only options
/// (`PR_SET_NO_NEW_PRIVS`) are sound at the call site; the
/// `unsafe` block there carries the SAFETY argument.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i64 {
    linux::syscall5(linux::SYS_PRCTL, option as i64, arg2 as i64, arg3 as i64, arg4 as i64, arg5 as i64)
}

/// Linux-only: `seccomp(operation, flags, args)`. With
/// `operation = SECCOMP_SET_MODE_FILTER` and `args = &sock_fprog`, the
/// kernel installs the BPF filter program for this thread and all of
/// its descendants. Returns 0 on success, negative `-errno` on failure.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn seccomp(operation: u32, flags: u32, args: *const core::ffi::c_void) -> i64 {
    linux::syscall3(linux::SYS_SECCOMP, operation as i64, flags as i64, args as i64)
}

/// Linux-only: `rt_sigaction(signum, act, oldact, sigsetsize)`. Used by
/// the `sandbox-trap` diagnostic feature to install a SIGSYS handler
/// before swapping seccomp's deny action from KILL_PROCESS to RET_TRAP.
/// `sigsetsize` is the kernel's sigset_t size — 8 bytes on Linux —
/// not the libc-padded 128-byte view.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn rt_sigaction(
    signum: i32,
    act: *const core::ffi::c_void,
    oldact: *mut core::ffi::c_void,
    sigsetsize: usize,
) -> i32 {
    linux::syscall4(
        linux::SYS_RT_SIGACTION,
        signum as i64,
        act as i64,
        oldact as i64,
        sigsetsize as i64,
    ) as i32
}

// ── Landlock (5.13+; -ENOSYS on older kernels, -EOPNOTSUPP if disabled) ───

/// Linux-only: `landlock_create_ruleset(attr, size, flags)`. Returns
/// the new ruleset fd on success, negative `-errno` on failure.
/// Stays `unsafe` because `attr` is read by the kernel for `size`
/// bytes; caller must ensure the pointer is valid for that range.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn landlock_create_ruleset(
    attr: *const core::ffi::c_void,
    size: usize,
    flags: u32,
) -> i64 {
    linux::syscall3(
        linux::SYS_LANDLOCK_CREATE_RULESET,
        attr as i64,
        size as i64,
        flags as i64,
    )
}

/// Linux-only: `landlock_add_rule(ruleset_fd, rule_type, rule_attr, flags)`.
/// `rule_attr` points at a rule-type-specific struct the kernel reads.
#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn landlock_add_rule(
    ruleset_fd: i32,
    rule_type: u32,
    rule_attr: *const core::ffi::c_void,
    flags: u32,
) -> i64 {
    linux::syscall4(
        linux::SYS_LANDLOCK_ADD_RULE,
        ruleset_fd as i64,
        rule_type as i64,
        rule_attr as i64,
        flags as i64,
    )
}

/// Linux-only: `landlock_restrict_self(ruleset_fd, flags)`. Applies
/// the ruleset to the calling thread + descendants. Safe wrapper:
/// integer args only; the only failure mode is the kernel returning
/// `-errno` (no UB).
#[cfg(target_os = "linux")]
#[inline]
pub fn landlock_restrict_self(ruleset_fd: i32, flags: u32) -> i64 {
    // SAFETY: integer-only args; kernel reads no user memory.
    unsafe {
        linux::syscall2(
            linux::SYS_LANDLOCK_RESTRICT_SELF,
            ruleset_fd as i64,
            flags as i64,
        )
    }
}

/// macOS-only: raw `sysctl` syscall (number 202). Apple has marked the
/// syscall deprecated in docs for years but it still works; `sysctlbyname`
/// is the "supported" path but also ends up here after a name→MIB lookup.
/// 6-arg shape: MIB array, length, optional output + its length-in/out,
/// optional input + its length.
#[cfg(target_os = "macos")]
#[inline]
pub unsafe fn sysctl(
    name: *mut i32,
    namelen: u32,
    oldp: *mut core::ffi::c_void,
    oldlenp: *mut usize,
    newp: *mut core::ffi::c_void,
    newlen: usize,
) -> i32 {
    darwin::syscall6(
        darwin::SYS_SYSCTL,
        name as i64, namelen as i64,
        oldp as i64, oldlenp as i64,
        newp as i64, newlen as i64,
    ) as i32
}

/// macOS-only: raw `__proc_info` syscall (number 336). The single kernel
/// entry point behind every `libproc` function (`proc_listallpids`,
/// `proc_pidinfo`, `proc_name`, etc.) — each is a libSystem shim that
/// picks a `callnum` + `flavor` and calls here.
#[cfg(target_os = "macos")]
#[inline]
pub unsafe fn proc_info(
    callnum: i32,
    pid: i32,
    flavor: u32,
    arg: u64,
    buffer: *mut core::ffi::c_void,
    buffersize: i32,
) -> i32 {
    darwin::syscall6(
        darwin::SYS_PROC_INFO,
        callnum as i64, pid as i64,
        flavor as i64, arg as i64,
        buffer as i64, buffersize as i64,
    ) as i32
}

/// Safe wrapper: fill `set` with the current process's CPU affinity mask
/// and return the syscall's return value (bytes written on success,
/// negative `-errno` on failure).
#[cfg(target_os = "linux")]
#[inline]
pub fn sched_getaffinity_self(set: &mut [u64]) -> i64 {
    let size = core::mem::size_of_val(set);
    unsafe { sched_getaffinity(0, size, set.as_mut_ptr()) }
}

#[cfg(target_os = "linux")]
#[inline]
pub unsafe fn uname(buf: *mut u8) -> i32 {
    linux::syscall1(linux::SYS_UNAME, buf as i64) as i32
}

/// Terminates the whole process (all threads) with status `code`. Used
/// from the panic handler and from the custom `_start`'s tail path.
///
/// Safe: no pointer arguments; the process ending isn't a memory-safety
/// concern from Rust's POV.
#[inline]
pub fn exit_group(code: i32) -> ! {
    #[cfg(target_os = "linux")]
    { unsafe {
        linux::syscall1(linux::SYS_EXIT_GROUP, code as i64);
        core::hint::unreachable_unchecked()
    } }
    // Darwin's `SYS_exit` (1) already terminates the whole process; there's
    // no separate `exit_group`.
    #[cfg(target_os = "macos")]
    { unsafe {
        darwin::syscall1(darwin::SYS_EXIT_GROUP, code as i64);
        core::hint::unreachable_unchecked()
    } }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { unsafe { libc::_exit(code) } }
}

// ── tests ──────────────────────────────────────────────────────────────────
//
// Fixed behavioural spec for every syscall wrapper. Runs under the
// default test harness (std + libc) against the same wrappers the
// production path uses, so on Linux we exercise the raw-syscall branch
// with real SYS_* numbers and real inline-asm argument marshalling.
// When porting to a new arch (e.g. aarch64 running under qemu-user),
// every one of these is a direct proof point: a wrong SYS_* number
// fails the functional test (kernel routes to a different syscall or
// ENOSYS); swapped argument registers fail with EINVAL or wrong data.
//
// A few tests are Linux-only: they call syscalls we only implement in
// the Linux branch (`getdents64`, `sched_getaffinity`, `uname`).
//
// `exit_group` is not tested directly — any test that successfully
// exits proves it, and a dedicated test would have to fork.
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    fn elapsed_ns(start: &Timespec, end: &Timespec) -> i64 {
        (end.tv_sec - start.tv_sec) * 1_000_000_000 + (end.tv_nsec - start.tv_nsec)
    }

    fn monotonic_now() -> Timespec {
        let mut ts = Timespec::zero();
        let r = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
        assert_eq!(r, 0, "clock_gettime returned {r}");
        ts
    }

    /// SYS_GETPID, 0-arg: result must match std::process::id.
    #[test]
    fn getpid_matches_std() {
        let ours = getpid();
        let std_pid = std::process::id() as i32;
        assert_eq!(ours, std_pid, "getpid: ours={ours} std={std_pid}");
    }

    /// SYS_CLOCK_GETTIME, 2-arg, writes an output struct via the second
    /// pointer. Checks arg register order and that we can read back a
    /// non-zero monotonic timestamp.
    #[test]
    fn clock_gettime_monotonic_nonzero() {
        let ts = monotonic_now();
        assert!(ts.tv_sec > 0 || ts.tv_nsec > 0, "monotonic ts is zero: {}.{:09}", ts.tv_sec, ts.tv_nsec);
    }

    /// SYS_NANOSLEEP, 2-arg. Requested 2 ms → measured via two
    /// clock_gettime(MONOTONIC) calls must be ≥ 2 ms. Also validates
    /// clock_gettime's write-through-pointer arg ordering (same test
    /// both ways: wrong arg order on either side would fail here).
    #[test]
    fn nanosleep_sleeps_requested() {
        let req = Timespec { tv_sec: 0, tv_nsec: 2_000_000 };
        let mut rem = Timespec::zero();
        let start = monotonic_now();
        let r = unsafe { nanosleep(&req, &mut rem) };
        let end = monotonic_now();
        assert_eq!(r, 0, "nanosleep returned {r}");
        let dt = elapsed_ns(&start, &end);
        assert!(dt >= 2_000_000, "slept only {dt} ns");
    }

    /// SYS_WRITE + SYS_READ round-trip over a socketpair. Exercises:
    ///   - write: arg order (fd, buf, len) → returns bytes written.
    ///   - read:  arg order (fd, buf, len) → returns bytes read and
    ///     writes them into the user buffer.
    /// Uses `std::os::unix::net::UnixStream::pair` so we don't depend
    /// on libc's `pipe`; the raw FDs go straight into our syscall
    /// wrappers.
    #[test]
    fn write_read_roundtrip_via_socketpair() {
        let (a, b) = std::os::unix::net::UnixStream::pair().expect("socketpair");
        let msg = b"ltop-syscall-roundtrip";
        let wrote = write_all(a.as_raw_fd(), msg);
        assert_eq!(wrote, msg.len() as i64, "write returned {wrote}");

        let mut buf = [0u8; 64];
        let n = unsafe { read(b.as_raw_fd(), buf.as_mut_ptr(), buf.len()) };
        assert_eq!(n, msg.len() as i64, "read returned {n}");
        assert_eq!(&buf[..n as usize], msg);
    }

    /// SYS_OPEN on an existing read-only file, then SYS_CLOSE. Uses
    /// /proc/self/comm which is always present on Linux; on macOS
    /// `/etc/hosts` stands in.
    #[test]
    fn open_close_existing_file() {
        #[cfg(target_os = "linux")]
        let path = b"/proc/self/comm\0";
        #[cfg(not(target_os = "linux"))]
        let path = b"/etc/hosts\0";
        let fd = unsafe { open(path.as_ptr(), O_RDONLY) };
        assert!(fd >= 0, "open returned {fd}");
        let r = close(fd);
        assert_eq!(r, 0, "close returned {r}");
    }

    /// SYS_OPEN on a nonexistent path returns a negative errno (-ENOENT).
    /// A wrong SYS_* number here would likely succeed on a different
    /// syscall or fail with a different errno, so the sign is enough.
    #[test]
    fn open_nonexistent_is_negative() {
        let fd = unsafe { open(b"/nonexistent-ltop-syscall-test\0".as_ptr(), O_RDONLY) };
        assert!(fd < 0, "expected negative errno, got fd={fd}");
    }

    /// SYS_OPENAT with AT_FDCWD behaves like SYS_OPEN. Separate SYS_*
    /// number so this is an independent check.
    #[test]
    fn openat_atfdcwd_existing_file() {
        #[cfg(target_os = "linux")]
        let path = b"/proc/self/comm\0";
        #[cfg(not(target_os = "linux"))]
        let path = b"/etc/hosts\0";
        let fd = unsafe { openat(AT_FDCWD, path.as_ptr(), O_RDONLY) };
        assert!(fd >= 0, "openat returned {fd}");
        close(fd);
    }

    /// SYS_READ on an open /proc/self/comm matches that same file read
    /// via std::fs. Catches endian / length-register mistakes that the
    /// socketpair test might miss (real file, kernel-allocated data).
    #[cfg(target_os = "linux")]
    #[test]
    fn read_proc_self_comm_matches_std() {
        let expected = std::fs::read("/proc/self/comm").expect("std read");
        let fd = unsafe { open(b"/proc/self/comm\0".as_ptr(), O_RDONLY) };
        assert!(fd >= 0);
        let mut buf = [0u8; 256];
        let n = unsafe { read(fd, buf.as_mut_ptr(), buf.len()) };
        close(fd);
        assert!(n > 0, "read returned {n}");
        assert_eq!(&buf[..n as usize], expected.as_slice());
    }

    /// SYS_IOCTL on a non-tty fd must fail (ENOTTY). Proves:
    ///   - ioctl reaches the kernel (wrong SYS_* would e.g. return
    ///     ENOSYS or succeed on a different syscall).
    ///   - arg3 is interpreted as a user-pointer (we pass a real buf).
    #[test]
    fn ioctl_non_tty_errors() {
        let fd = unsafe { open(b"/dev/null\0".as_ptr(), O_RDWR) };
        assert!(fd >= 0);
        let mut ws = [0u8; 16];
        let r = unsafe { ioctl(fd, TIOCGWINSZ, ws.as_mut_ptr() as *mut _) };
        close(fd);
        assert!(r < 0, "expected error on /dev/null, got {r}");
    }

    /// SYS_MADVISE(DONTNEED) on an anonymous, page-aligned buffer from
    /// the global allocator returns 0. Validates arg order
    /// (addr, len, advice).
    #[test]
    fn madvise_dontneed_on_aligned_page() {
        use std::alloc::{alloc, dealloc, Layout};
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        unsafe {
            let p = alloc(layout);
            assert!(!p.is_null());
            let r = madvise(p, 4096, MADV_DONTNEED);
            assert_eq!(r, 0, "madvise returned {r}");
            dealloc(p, layout);
        }
    }

    /// SYS_GETDENTS64 on /proc/self reads at least one directory entry.
    /// Linux-only: macOS doesn't implement the same syscall. The
    /// returned bytes start with a dirent64 header whose d_reclen is
    /// a valid stride (≥ 24), so we sanity-check that too.
    #[cfg(target_os = "linux")]
    #[test]
    fn getdents64_reads_proc_self() {
        let fd = unsafe { open(b"/proc/self\0".as_ptr(), O_RDONLY | O_DIRECTORY) };
        assert!(fd >= 0, "open /proc/self: {fd}");
        let mut buf = [0u8; 4096];
        let n = unsafe { getdents64(fd, buf.as_mut_ptr(), buf.len()) };
        close(fd);
        assert!(n > 0, "getdents64 returned {n}");
        // First record's d_reclen lives at byte offset 16 (u16 LE).
        let reclen = u16::from_le_bytes([buf[16], buf[17]]);
        assert!((24..=buf.len() as u16).contains(&reclen), "weird reclen={reclen}");
    }

    /// SYS_SCHED_GETAFFINITY returns bytes written (> 0) and at least
    /// one CPU bit is set. Linux-only.
    #[cfg(target_os = "linux")]
    #[test]
    fn sched_getaffinity_has_cpus() {
        let mut set = [0u64; 16];
        let size = core::mem::size_of_val(&set);
        let r = unsafe { sched_getaffinity(0, size, set.as_mut_ptr()) };
        assert!(r > 0, "sched_getaffinity returned {r}");
        let popcount: u32 = set.iter().map(|w| w.count_ones()).sum();
        assert!(popcount >= 1, "no CPUs in affinity mask");
    }

    /// SYS_UNAME populates the buffer with `struct utsname` whose first
    /// field (sysname) is "Linux". Linux-only.
    #[cfg(target_os = "linux")]
    #[test]
    fn uname_sysname_is_linux() {
        // struct utsname on Linux is 6 × 65 bytes.
        let mut buf = [0u8; 65 * 6];
        let r = unsafe { uname(buf.as_mut_ptr()) };
        assert_eq!(r, 0, "uname returned {r}");
        assert!(buf.starts_with(b"Linux"), "first field is not 'Linux': {:?}", &buf[..16]);
    }
}

// ── Page size (Linux) ────────────────────────────────────────────────────────

/// Kernel page size. Not a constant on Linux: aarch64 kernels ship
/// with 4 KB, 16 KB (Asahi), or 64 KB (RHEL-alt) pages, and both the
/// RSS math (`/proc/*/stat` reports RSS in kernel pages) and every
/// `madvise` alignment depend on the real value.
#[cfg(target_os = "linux")]
pub mod page {
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// auxv key for the page size (`AT_PAGESZ`, elf.h).
    pub const AT_PAGESZ: i64 = 6;

    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    /// Record the auxv-provided page size. Only plausible values stick
    /// (power of two in 1 KB..=1 MB); anything else leaves the lazy
    /// fallback in charge.
    pub fn set(size: u64) {
        if size.is_power_of_two() && (1024..=1 << 20).contains(&size) {
            PAGE_SIZE.store(size as usize, Ordering::Relaxed);
        }
    }

    /// The libc-free build stores AT_PAGESZ in `start.rs::ltop_entry`
    /// before anything pages (the stack-VMA madvise walk needs it);
    /// glibc test/dev builds land in the lazy branch, which reads the
    /// same auxv pairs from /proc/self/auxv.
    pub fn get() -> usize {
        let p = PAGE_SIZE.load(Ordering::Relaxed);
        if p != 0 { return p; }
        if let Some(v) = from_proc_auxv() { set(v as u64); }
        let p = PAGE_SIZE.load(Ordering::Relaxed);
        if p != 0 { return p; }
        PAGE_SIZE.store(4096, Ordering::Relaxed);
        4096
    }

    // In the libc-free build ltop_entry always pre-stores, so the
    // lazy reader would be dead bytes — cfg it away.
    #[cfg(libc_free)]
    fn from_proc_auxv() -> Option<usize> { None }

    /// `/proc/self/auxv` is the kernel's (a_type, a_val) u64 pairs,
    /// AT_NULL-terminated — the same data the initial stack carries.
    #[cfg(not(libc_free))]
    fn from_proc_auxv() -> Option<usize> {
        let fd = super::open_cstr(
            c"/proc/self/auxv",
            super::O_RDONLY | super::O_CLOEXEC,
        );
        if fd < 0 { return None; }
        let mut buf = [0u8; 1024];
        let n = super::read_buf(fd, &mut buf);
        super::close(fd);
        let n = if n > 0 { n as usize } else { 0 };
        let mut i = 0usize;
        while i + 16 <= n {
            let key = u64::from_ne_bytes(buf[i..i + 8].try_into().ok()?);
            let val = u64::from_ne_bytes(buf[i + 8..i + 16].try_into().ok()?);
            if key == 0 { break; }
            if key == AT_PAGESZ as u64 { return Some(val as usize); }
            i += 16;
        }
        None
    }

    /// `/proc/self/smaps`' KernelPageSize is an independent kernel
    /// report of the same value (in kB).
    #[cfg(test)]
    mod tests {
        #[test]
        fn page_size_matches_smaps() {
            let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
            let line = smaps.lines()
                .find(|l| l.starts_with("KernelPageSize:"))
                .expect("KernelPageSize line in smaps");
            let kb: usize = line.split_whitespace().nth(1).unwrap().parse().unwrap();
            assert_eq!(super::get(), kb * 1024);
        }
    }
}
