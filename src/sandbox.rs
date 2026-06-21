//! Linux sandbox: seccomp-bpf for syscall verbs + arguments, plus
//! Landlock for filesystem path restrictions. Together they bound a
//! compromised process to the same set of operations the legitimate
//! tick loop performs.
//!
//! # Why
//!
//! ltop has a meaningful amount of unsafe Rust talking to kernel
//! ABIs (`/proc` parsing, RM ioctls, the libc-free `_start` page
//! dance). A memory-corruption bug pivoted into ROP could otherwise
//! issue any syscall — open arbitrary files, exec, network, etc.
//!
//! # Scope
//!
//! Linux-only. macOS has no seccomp analog. The only in-process route
//! there is `sandbox_init_with_parameters` from libsandbox, which
//! requires importing a libSystem symbol — and the parity tests in
//! `mac_sys.rs` enforce that the production binary imports zero
//! libSystem symbols (`nm -u` is empty). The recommended workaround
//! on mac is launching under `sandbox-exec -p '<profile>' ltop`,
//! which keeps the binary clean and lets the system's own libsandbox
//! compile the SBPL profile against its own kernel — no version-skew
//! risk.
//!
//! # Mechanism
//!
//! Three layered restrictions, applied in order:
//!
//! 1. `prctl(PR_SET_NO_NEW_PRIVS, 1)` — prerequisite for unprivileged
//!    seccomp and Landlock. Once set, `execve` can no longer grant
//!    additional privileges via setuid/setgid/file caps. We never
//!    `execve`, so the visible effect is just that the kernel will
//!    accept seccomp/Landlock from a non-root process.
//!
//! 2. **Landlock** (Linux 5.13+, June 2021) — restricts `openat`
//!    by *path*. seccomp BPF can't deref the path pointer, so
//!    without Landlock a corrupted process could still
//!    `openat("/etc/shadow", ...)` even with read-only flags. The
//!    ruleset allows read-file + read-dir under `/proc` and nothing
//!    else. ltop requires Landlock at install time — kernels older
//!    than 5.13, or kernels built with `CONFIG_SECURITY_LANDLOCK=n`,
//!    abort with a clear stderr message rather than silently
//!    falling back to seccomp-only. Graceful fallback would save
//!    ~50 B but cost the kernel-version invariant; the invariant
//!    is more valuable.
//!
//! 3. **seccomp-bpf** with verb + argument filters. The BPF program:
//!    - rejects any syscall not in the allowlist;
//!    - for `write`, requires `fd ∈ {1, 2}` (stdout/stderr only);
//!    - for `openat`, requires flags exclude any write/create/trunc/
//!      append bit (read-only opens only);
//!    - for `ioctl`, requires `cmd ∈ {TIOCGWINSZ, TCSETS}` ∪
//!      `{any RM ioctl: dir=11 + magic='F'}`.
//!    Default deny action is `SECCOMP_RET_KILL_PROCESS` — disallowed
//!    syscall terminates the whole process immediately, no userspace
//!    handler involvement. Under the `sandbox-trap` cargo feature
//!    the action becomes `RET_TRAP` and a SIGSYS handler prints the
//!    blocked syscall number to stderr before exiting; intended for
//!    diagnosing "ltop dies on kernel X" reports.
//!
//! # When to install
//!
//! From `run()`, after every one-shot startup syscall (open
//! `/dev/nvidiactl`, `sched_getaffinity`, the termios get/set in
//! `tty::enter_raw_mode`, the `/proc/uptime` read in `boot_time`,
//! `gpu::init`'s RM ioctls) and before the first tick. The one-shot
//! syscalls don't need to be in the allowlist; only the verbs the
//! steady-state tick loop and exit path use are.
//!
//! # Future work (not yet implemented)
//!
//! - **macOS sandbox-exec wrapper.** Ship a `tools/ltop-sandbox.sb`
//!   profile + a documented invocation. No code, just packaging.
//! - **Tighter argument filtering.** Forbid `mmap`/`mprotect` with
//!   `PROT_EXEC` if either ever joins the allowlist.

#![cfg(target_os = "linux")]

use crate::syscall;

// ── seccomp / BPF / prctl constants (linux/seccomp.h, linux/filter.h, …) ───

const SECCOMP_SET_MODE_FILTER: u32 = 1;
const SECCOMP_RET_KILL_PROCESS: u32 = 0x80000000;
const SECCOMP_RET_TRAP: u32 = 0x00030000;
const SECCOMP_RET_ALLOW: u32 = 0x7fff0000;

// Classic-BPF instruction encoding (linux/filter.h). `code` packs
// class (low 3 bits) + op-specific bits.
const BPF_LD_W_ABS:  u16 = 0x20; // BPF_LD  | BPF_W   | BPF_ABS — load 4-byte word at absolute offset
const BPF_JMP_JEQ_K: u16 = 0x15; // BPF_JMP | BPF_JEQ | BPF_K   — jump if A == constant
const BPF_RET_K:     u16 = 0x06; // BPF_RET | BPF_K             — return constant
const BPF_ALU_AND_K: u16 = 0x54; // BPF_ALU | BPF_AND | BPF_K   — A &= constant

// linux/audit.h — packs (arch_type | __AUDIT_ARCH_64BIT | __AUDIT_ARCH_LE).
// The kernel writes the caller's arch into seccomp_data.arch; we
// reject anything that isn't our native arch to head off 32-bit /
// x32-ABI selector tricks.
const AUDIT_ARCH_X86_64: u32 = 0xC000003E;
const AUDIT_ARCH_AARCH64: u32 = 0xC00000B7;

// `struct seccomp_data` layout (linux/seccomp.h):
//   int   nr;                     // offset 0,  4 bytes
//   __u32 arch;                   // offset 4,  4 bytes
//   __u64 instruction_pointer;    // offset 8,  8 bytes
//   __u64 args[6];                // offset 16, 6 × 8 = 48 bytes
const SECCOMP_DATA_NR: u32 = 0;
const SECCOMP_DATA_ARCH: u32 = 4;
// args[i] starts at offset 16 + 8*i. We only ever load the low 32
// bits because every arg we filter (write fd, openat flags, ioctl
// cmd) is declared as 32-bit by the kernel syscall handler — the
// high 32 bits are sign/zero-extension the kernel discards.
const SECCOMP_DATA_ARG0_LO: u32 = 16;
const SECCOMP_DATA_ARG1_LO: u32 = 24;
const SECCOMP_DATA_ARG2_LO: u32 = 32;

// linux/prctl.h
const PR_SET_NO_NEW_PRIVS: i32 = 38;

// ── argument-filter constants ──────────────────────────────────────────────

// `openat` flag bits we forbid. Restricting flags this way lets us say
// "only read-only opens" without having to filter on the path string
// (which BPF can't dereference anyway). Landlock handles paths.
//
//   O_WRONLY     = 0x001  - access mode 1
//   O_RDWR       = 0x002  - access mode 2  (mask 0x003 covers both
//                           plus invalid mode 3)
//   O_CREAT      = 0x040  - create on open
//   O_TRUNC      = 0x200  - truncate on open
//   O_APPEND     = 0x400  - append on write
//   __O_TMPFILE  = 0x400000 - kernel rejects without O_RDWR/WRONLY
//                             anyway, but explicit-deny is defensive.
const OPENAT_DENY_FLAGS: u32 = 0x400643;

// Linux terminal ioctls. Same numbers on x86_64 and aarch64 (asm-generic).
const TIOCGWINSZ: u32 = 0x5413;
const TCSETS: u32 = 0x5402;

// RM (NVIDIA driver) ioctls all share `(dir=3) | (magic='F'<<8)` —
// the size and nr fields differ per command. Match by mask + value:
//   bits 30-31 = dir = 0b11
//   bits  8-15 = magic = 0x46 ('F')
// Yields mask=0xc000ff00, value=0xc0004600. Lets every RM cmd through
// without enumerating ~20 size×nr combinations.
const RM_IOCTL_MASK: u32 = 0xc000ff00;
const RM_IOCTL_VALUE: u32 = 0xc000_0000 | (b'F' as u32) << 8;

// ── BPF program ────────────────────────────────────────────────────────────

/// One classic-BPF instruction. Matches `struct sock_filter` in
/// linux/filter.h byte-for-byte (code:u16, jt:u8, jf:u8, k:u32).
#[repr(C)]
#[derive(Clone, Copy)]
struct SockFilter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

/// Header passed to `seccomp(SECCOMP_SET_MODE_FILTER, …)`.
/// `len` is the instruction count; `filter` points at the program.
#[repr(C)]
struct SockFprog {
    len: u16,
    filter: *const SockFilter,
}

#[cfg(target_arch = "x86_64")]
const AUDIT_ARCH: u32 = AUDIT_ARCH_X86_64;
#[cfg(target_arch = "aarch64")]
const AUDIT_ARCH: u32 = AUDIT_ARCH_AARCH64;

/// Final action for syscalls not in the allowlist. KILL_PROCESS in
/// production; RET_TRAP under the `sandbox-trap` feature so the
/// SIGSYS handler can report which syscall was blocked.
#[cfg(not(feature = "sandbox-trap"))]
const DENY_ACTION: u32 = SECCOMP_RET_KILL_PROCESS;
#[cfg(feature = "sandbox-trap")]
const DENY_ACTION: u32 = SECCOMP_RET_TRAP;

// ── per-arch syscall numbers (mirror src/syscall.rs) ───────────────────────

/// "Direct-allow" syscalls — no argument filtering. The argument
/// surface is either none (`exit_group`), trivially safe (`close` of
/// an arbitrary fd), or sufficiently constrained by Landlock + the
/// existing fd set.
///
/// Per-syscall rationale:
///
///   read              - reading from any fd we already hold
///                       (proc dirfd files, RM ioctl returns,
///                       optionally stdin for keypress polling).
///   close             - closing any fd we hold.
///   madvise           - arena page DONTNEED for sub-scope rewinds;
///                       can't escalate (it only releases pages we
///                       own, no information disclosure).
///   nanosleep         - tick spacing.
///   clock_gettime     - `Instant::now()` for tick budget + stamps.
///   exit_group        - normal termination + the sandbox's own
///                       failure path.
///   getdents64        - reading /proc to enumerate PIDs (via the
///                       /proc dirfd; Landlock confines the path).
///   getpid            - collect_procs needs to know our own pid
///                       to filter ourselves out of the tree.
///   rt_sigreturn      - return-from-signal-handler. Required by the
///                       `sandbox-trap` SIGSYS handler; gated on
///                       that feature so non-trap builds don't carry
///                       a JEQ for a syscall they can't reach (we
///                       install no handlers, so the kernel never
///                       invokes rt_sigreturn).
///
/// Notably absent: `restart_syscall`. The kernel only invokes it
/// when a syscall (e.g. nanosleep) is interrupted by a signal whose
/// handler returned with SA_RESTART. We install no handlers in
/// production (or in trap mode — the SIGSYS handler exits, doesn't
/// return), so this path is unreachable.
#[cfg(target_arch = "x86_64")]
const ALLOW_DIRECT: &[u32] = &[
    0,    // read
    3,    // close
    28,   // madvise
    35,   // nanosleep
    39,   // getpid
    217,  // getdents64
    228,  // clock_gettime
    231,  // exit_group
    #[cfg(feature = "sandbox-trap")]
    15,   // rt_sigreturn
];

/// "Filtered" syscalls — allowed only with restricted argument
/// values:
///
///   write             - fd ∈ {1, 2} (stdout, stderr).
///                       Excludes any other fd we hold so a
///                       compromised process can't write to the
///                       proc dirfd (would error anyway), the RM
///                       fd (would be a write-to-driver), etc.
///   openat            - flags exclude write/create/trunc/append
///                       bits; only read-only opens are allowed.
///                       Path is then constrained by Landlock to
///                       /proc.
///   ioctl             - cmd ∈ {TIOCGWINSZ for term_size, TCSETS
///                       for tty raw-mode restore on exit} ∪ all
///                       RM ioctls (matched by magic byte 'F' + dir
///                       bits = 11). Excludes the dangerous
///                       TIOCSTI (tty input injection on the
///                       controlling terminal).
#[cfg(target_arch = "x86_64")]
const SYS_WRITE: u32 = 1;
#[cfg(target_arch = "x86_64")]
const SYS_OPENAT: u32 = 257;
#[cfg(target_arch = "x86_64")]
const SYS_IOCTL: u32 = 16;

#[cfg(target_arch = "aarch64")]
const ALLOW_DIRECT: &[u32] = &[
    63,   // read
    57,   // close
    233,  // madvise
    101,  // nanosleep
    172,  // getpid
    61,   // getdents64
    113,  // clock_gettime
    94,   // exit_group
    #[cfg(feature = "sandbox-trap")]
    139,  // rt_sigreturn
];
#[cfg(target_arch = "aarch64")]
const SYS_WRITE: u32 = 64;
#[cfg(target_arch = "aarch64")]
const SYS_OPENAT: u32 = 56;
#[cfg(target_arch = "aarch64")]
const SYS_IOCTL: u32 = 29;

const N_DIRECT: usize = ALLOW_DIRECT.len();

// ── BPF program layout ─────────────────────────────────────────────────────
//
// The program is shaped as:
//
//   [setup × 3]
//   [direct-allow JEQs × N]
//   [dispatch JEQs × 3] (write, openat, ioctl → check blocks)
//   [write check  × 3]  (LD arg0; JEQ 1 → allow; JEQ 2 → allow else deny)
//   [openat check × 3]  (LD arg2; AND deny-mask; JEQ 0 → allow else deny)
//   [ioctl check  × 5]  (LD arg1; JEQ TIOCGWINSZ → allow;
//                         JEQ TCSETS → allow; AND RM-mask; JEQ RM-value
//                         → allow else deny)
//   [RET deny]
//   [RET allow]
//
// The fixed-size sections are named by const indices so the const-fn
// builder can write each instruction with explicit, named jump
// targets — no magic numbers in the middle.

const I_LD_ARCH: usize = 0;
const I_JEQ_ARCH: usize = 1;
const I_LD_NR: usize = 2;
const I_DIRECT_START: usize = 3;
const I_DISPATCH_WRITE: usize = I_DIRECT_START + N_DIRECT;
const I_DISPATCH_OPENAT: usize = I_DISPATCH_WRITE + 1;
const I_DISPATCH_IOCTL: usize = I_DISPATCH_OPENAT + 1;
const I_CHECK_WRITE: usize = I_DISPATCH_IOCTL + 1;
const I_CHECK_OPENAT: usize = I_CHECK_WRITE + 3;
const I_CHECK_IOCTL: usize = I_CHECK_OPENAT + 3;
const I_RET_DENY: usize = I_CHECK_IOCTL + 5;
const I_RET_ALLOW: usize = I_RET_DENY + 1;
const PROG_LEN: usize = I_RET_ALLOW + 1;

/// Compute the relative jump offset from instruction `from` to
/// instruction `to`, in classic-BPF "skip past the next" units.
/// `pc_after = from + 1 + offset`, so `offset = to - from - 1`.
const fn jmp(from: usize, to: usize) -> u8 {
    (to - from - 1) as u8
}

/// Build the BPF program at compile time.
const fn build_prog() -> [SockFilter; PROG_LEN] {
    let mut prog = [SockFilter { code: 0, jt: 0, jf: 0, k: 0 }; PROG_LEN];

    // [0] LD arch
    prog[I_LD_ARCH] = SockFilter { code: BPF_LD_W_ABS, jt: 0, jf: 0, k: SECCOMP_DATA_ARCH };
    // [1] JEQ AUDIT_ARCH; mismatch → deny.
    prog[I_JEQ_ARCH] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: 0,
        jf: jmp(I_JEQ_ARCH, I_RET_DENY),
        k: AUDIT_ARCH,
    };
    // [2] LD nr
    prog[I_LD_NR] = SockFilter { code: BPF_LD_W_ABS, jt: 0, jf: 0, k: SECCOMP_DATA_NR };

    // Direct-allow JEQs: each match → ALLOW; mismatch → fall through.
    let mut i = 0;
    while i < N_DIRECT {
        let idx = I_DIRECT_START + i;
        prog[idx] = SockFilter {
            code: BPF_JMP_JEQ_K,
            jt: jmp(idx, I_RET_ALLOW),
            jf: 0,
            k: ALLOW_DIRECT[i],
        };
        i += 1;
    }

    // Dispatch JEQs for filtered syscalls. The first two fall through
    // to the next dispatch on jf=0; the last one denies on mismatch.
    prog[I_DISPATCH_WRITE] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_DISPATCH_WRITE, I_CHECK_WRITE),
        jf: 0,
        k: SYS_WRITE,
    };
    prog[I_DISPATCH_OPENAT] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_DISPATCH_OPENAT, I_CHECK_OPENAT),
        jf: 0,
        k: SYS_OPENAT,
    };
    prog[I_DISPATCH_IOCTL] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_DISPATCH_IOCTL, I_CHECK_IOCTL),
        jf: jmp(I_DISPATCH_IOCTL, I_RET_DENY),
        k: SYS_IOCTL,
    };

    // Write check: LD arg0 (fd low 32); JEQ 1 → allow; JEQ 2 → allow else deny.
    prog[I_CHECK_WRITE] = SockFilter {
        code: BPF_LD_W_ABS, jt: 0, jf: 0, k: SECCOMP_DATA_ARG0_LO,
    };
    prog[I_CHECK_WRITE + 1] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_CHECK_WRITE + 1, I_RET_ALLOW),
        jf: 0,
        k: 1, // stdout
    };
    prog[I_CHECK_WRITE + 2] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_CHECK_WRITE + 2, I_RET_ALLOW),
        jf: jmp(I_CHECK_WRITE + 2, I_RET_DENY),
        k: 2, // stderr
    };

    // Openat check: LD arg2 (flags low 32); AND OPENAT_DENY_FLAGS; JEQ 0 → allow else deny.
    prog[I_CHECK_OPENAT] = SockFilter {
        code: BPF_LD_W_ABS, jt: 0, jf: 0, k: SECCOMP_DATA_ARG2_LO,
    };
    prog[I_CHECK_OPENAT + 1] = SockFilter {
        code: BPF_ALU_AND_K, jt: 0, jf: 0, k: OPENAT_DENY_FLAGS,
    };
    prog[I_CHECK_OPENAT + 2] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_CHECK_OPENAT + 2, I_RET_ALLOW),
        jf: jmp(I_CHECK_OPENAT + 2, I_RET_DENY),
        k: 0,
    };

    // Ioctl check: LD arg1 (cmd low 32); JEQ TIOCGWINSZ → allow;
    // JEQ TCSETS → allow; AND RM_MASK; JEQ RM_VALUE → allow else deny.
    prog[I_CHECK_IOCTL] = SockFilter {
        code: BPF_LD_W_ABS, jt: 0, jf: 0, k: SECCOMP_DATA_ARG1_LO,
    };
    prog[I_CHECK_IOCTL + 1] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_CHECK_IOCTL + 1, I_RET_ALLOW),
        jf: 0,
        k: TIOCGWINSZ,
    };
    prog[I_CHECK_IOCTL + 2] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_CHECK_IOCTL + 2, I_RET_ALLOW),
        jf: 0,
        k: TCSETS,
    };
    prog[I_CHECK_IOCTL + 3] = SockFilter {
        code: BPF_ALU_AND_K, jt: 0, jf: 0, k: RM_IOCTL_MASK,
    };
    prog[I_CHECK_IOCTL + 4] = SockFilter {
        code: BPF_JMP_JEQ_K,
        jt: jmp(I_CHECK_IOCTL + 4, I_RET_ALLOW),
        jf: jmp(I_CHECK_IOCTL + 4, I_RET_DENY),
        k: RM_IOCTL_VALUE,
    };

    // Returns.
    prog[I_RET_DENY] = SockFilter { code: BPF_RET_K, jt: 0, jf: 0, k: DENY_ACTION };
    prog[I_RET_ALLOW] = SockFilter { code: BPF_RET_K, jt: 0, jf: 0, k: SECCOMP_RET_ALLOW };

    prog
}

static PROG: [SockFilter; PROG_LEN] = build_prog();

// ── Landlock (linux/landlock.h, kernel 5.13+) ──────────────────────────────

const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2; // 4
const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3; // 8

/// Rule type: restrict accesses to a subtree under `parent_fd`.
const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

// O_PATH: open a path reference without read/write permissions —
// what Landlock recommends for `parent_fd`. asm-generic, same on
// x86_64 and aarch64. (O_DIRECTORY differs per arch but isn't
// strictly required here.)
const O_PATH: i32 = 0o10000000;

/// `struct landlock_path_beneath_attr`. Kernel header is
/// `__attribute__((packed))` — total 12 bytes. With `repr(C)` the
/// struct is 16 bytes (4 trailing pad), but the kernel reads only
/// the first 12 anyway because that's the size passed in
/// `landlock_add_rule`'s rule_type-specific contract. The padding
/// bytes are never read.
#[repr(C)]
struct LandlockPathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

// ── installation ───────────────────────────────────────────────────────────

/// Install the sandbox: PR_SET_NO_NEW_PRIVS, then Landlock (path
/// restrictions), then seccomp (syscall verb + arg filter). Must be
/// called once, late enough in startup that all one-shot setup
/// syscalls have already happened.
///
/// Any failure (prctl, Landlock create/add/restrict, seccomp install)
/// calls `exit_group(2)` after a stderr message — silent fall-through
/// would be a security regression that's hard to notice. Kernels
/// without Landlock (Linux < 5.13 or `CONFIG_SECURITY_LANDLOCK=n`)
/// abort here too: a kernel old enough to lack Landlock predates
/// June 2021 and ltop's threat model assumes both layers are present.
///
/// In `sandbox-trap` mode this also installs a SIGSYS handler so
/// blocked syscalls report their nr to stderr instead of dying
/// silently.
pub fn install() {
    #[cfg(feature = "sandbox-trap")]
    install_sigsys_handler();

    // SAFETY: `PR_SET_NO_NEW_PRIVS` takes a single bool; no
    // pointers, so the integer args are sound under prctl's
    // option-polymorphic signature.
    let r = unsafe { syscall::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if r < 0 { fail(b'1', r); }

    install_landlock();
    install_seccomp();
}

/// Print "ltop: sandbox install failed (S err=N)\n" to stderr and
/// exit. `s` is the ASCII step character documented at each call
/// site, `e` is the negative-errno return from the offending
/// syscall. Single helper because LLVM emits the write_all +
/// exit_group pair once regardless of how many fail-fast sites we
/// have.
#[inline(never)]
#[cold]
fn fail(s: u8, e: i64) -> ! {
    let mut buf = [0u8; 64];
    let prefix = b"ltop: sandbox install failed (";
    buf[..prefix.len()].copy_from_slice(prefix);
    let mut n = prefix.len();
    buf[n] = s;
    n += 1;
    buf[n..n + 5].copy_from_slice(b" err=");
    n += 5;
    let (neg, mut v) = if e < 0 { (true, e.unsigned_abs()) } else { (false, e as u64) };
    if neg { buf[n] = b'-'; n += 1; }
    let mut tmp = [0u8; 20];
    let mut tn = tmp.len();
    if v == 0 { tn -= 1; tmp[tn] = b'0'; }
    while v > 0 { tn -= 1; tmp[tn] = b'0' + (v % 10) as u8; v /= 10; }
    let dn = tmp.len() - tn;
    buf[n..n + dn].copy_from_slice(&tmp[tn..]);
    n += dn;
    buf[n] = b')';
    buf[n + 1] = b'\n';
    n += 2;
    let _ = syscall::write_all(2, &buf[..n]);
    syscall::exit_group(2);
}

fn install_seccomp() {
    let prog = SockFprog {
        len: PROG_LEN as u16,
        filter: PROG.as_ptr(),
    };
    // SAFETY: `prog.filter` points at the static `PROG` array of
    // `prog.len` valid `sock_filter` entries; the kernel reads them
    // and copies into a kernel-side BPF program.
    let r = unsafe {
        syscall::seccomp(
            SECCOMP_SET_MODE_FILTER,
            0,
            &prog as *const SockFprog as *const _,
        )
    };
    if r < 0 { fail(b'2', r); } // seccomp(SET_MODE_FILTER)
}

/// Install a Landlock ruleset that allows read access only under
/// `/proc`. Anything else (writing files, reading other paths,
/// network access on Linux ≥ 6.7 if we gated it, etc.) is denied
/// at the LSM layer.
///
/// Kernel ABI: `landlock_create_ruleset(&attr, sizeof(attr), 0)` →
/// ruleset fd. `landlock_add_rule(rs, PATH_BENEATH, &rule, 0)`.
/// `landlock_restrict_self(rs, 0)` applies the ruleset to this
/// thread + descendants.
fn install_landlock() {
    // The ruleset_attr struct is just a single u64 of access bits.
    // Inlining a u64 here saves the type definition and lets the
    // pointer cast collapse.
    let handled: u64 = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;
    // SAFETY: `&handled` is a valid pointer to 8 bytes the kernel
    // reads as `landlock_ruleset_attr.handled_access_fs`.
    let rs = unsafe {
        syscall::landlock_create_ruleset(
            &handled as *const u64 as *const _,
            core::mem::size_of::<u64>(),
            0,
        )
    };
    // Linux 5.13+ required: -ENOSYS on older kernels and
    // -EOPNOTSUPP when CONFIG_SECURITY_LANDLOCK=n both come through
    // here. We don't differentiate — the message is the same to
    // the user: this kernel can't sandbox ltop, please upgrade.
    if rs < 0 { fail(b'3', rs); } // landlock_create_ruleset
    let rs = rs as i32;

    // Open `/proc` as the parent for the path-beneath rule. O_PATH
    // means we just want a path reference, not read/write access —
    // Landlock only inspects the path the fd refers to.
    let proc_fd = syscall::open_cstr(c"/proc", O_PATH | syscall::O_CLOEXEC);
    if proc_fd < 0 { fail(b'4', proc_fd as i64); } // open(/proc, O_PATH)

    let rule = LandlockPathBeneathAttr {
        allowed_access: LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR,
        parent_fd: proc_fd,
    };
    // SAFETY: `rule` is a valid `landlock_path_beneath_attr`; kernel
    // reads exactly its packed 12 bytes (we pass repr(C) which is
    // 16 with 4 trailing pad — the kernel ignores the pad).
    let r = unsafe {
        syscall::landlock_add_rule(
            rs,
            LANDLOCK_RULE_PATH_BENEATH,
            &rule as *const _ as *const _,
            0,
        )
    };
    syscall::close(proc_fd);
    if r < 0 { fail(b'5', r); } // landlock_add_rule(PATH_BENEATH)

    let r = syscall::landlock_restrict_self(rs, 0);
    syscall::close(rs);
    if r < 0 { fail(b'6', r); } // landlock_restrict_self
}

// ── sandbox-trap: SIGSYS handler ───────────────────────────────────────────
//
// The handler runs in signal context after the kernel rejects a
// syscall (RET_TRAP). It reads `siginfo_t._sigsys._syscall` (the
// blocked syscall number) and prints a short diagnostic to stderr,
// then exits with code 159 (128 + 31 = SIGSYS, the shell convention
// for "killed by SIGSYS"). The mode is opt-in for two reasons:
//
//   1. KILL_PROCESS is strictly safer — there's no userspace handler
//      a corrupted process could hijack mid-flight.
//   2. The handler infrastructure costs ~200 B of `.text` (sigaction
//      install + restorer + the print path). Pay it only when
//      diagnosing.

#[cfg(feature = "sandbox-trap")]
fn install_sigsys_handler() {
    use core::ptr;

    // Layout of `struct sigaction` as the kernel rt_sigaction expects
    // it. This is NOT the libc-extended struct (which pads sa_mask to
    // 128 bytes) — kernel sigaction has the 8-byte kernel sigset_t
    // and uses sa_flags as `unsigned long`.
    //
    // x86_64: requires SA_RESTORER + a userspace sigreturn trampoline
    //   because the kernel doesn't ship a default one for 64-bit.
    // aarch64: kernel uses VDSO sigreturn; no SA_RESTORER needed.
    #[repr(C)]
    struct KSigAction {
        sa_handler: usize,
        sa_flags: u64,
        #[cfg(target_arch = "x86_64")]
        sa_restorer: usize,
        sa_mask: u64,
    }

    const SIGSYS: i32 = 31;
    const SA_SIGINFO: u64 = 0x00000004;
    #[cfg(target_arch = "x86_64")]
    const SA_RESTORER: u64 = 0x04000000;

    let act = KSigAction {
        sa_handler: sigsys_trampoline as usize,
        #[cfg(target_arch = "x86_64")]
        sa_flags: SA_SIGINFO | SA_RESTORER,
        #[cfg(not(target_arch = "x86_64"))]
        sa_flags: SA_SIGINFO,
        #[cfg(target_arch = "x86_64")]
        sa_restorer: sigreturn_trampoline as usize,
        sa_mask: 0,
    };

    // SAFETY: `act` is a valid kernel sigaction laid out per the
    // arch-specific contract above; the pointer is valid for the
    // syscall's read of `sizeof(KSigAction)` bytes, and we pass null
    // for `oldact` so the kernel doesn't write back. `sigsetsize = 8`
    // is the kernel's sigset_t size (NOT libc's 128).
    let r = unsafe {
        syscall::rt_sigaction(
            SIGSYS,
            &act as *const KSigAction as *const _,
            ptr::null_mut(),
            8,
        )
    };
    if r < 0 { fail(b'7', r as i64); } // rt_sigaction(SIGSYS)
}

#[cfg(feature = "sandbox-trap")]
unsafe extern "C" fn sigsys_trampoline(_sig: i32, info: *const u8, _ctx: *const u8) {
    // siginfo_t layout (kernel, both archs of interest):
    //   int si_signo;     // 0
    //   int si_errno;     // 4
    //   int si_code;      // 8
    //   int _pad;         // 12  (alignment to 8 for the union)
    //   union {
    //     ...
    //     struct {        // _sigsys, used when si_signo == SIGSYS
    //       void *si_call_addr;  // 16
    //       int   si_syscall;    // 24
    //       u32   si_arch;       // 28
    //     };
    //   };
    let nr = if info.is_null() {
        -1
    } else {
        // SAFETY: the kernel writes a full siginfo_t before invoking
        // the handler; offset 24 is within the SIGSYS variant.
        unsafe { *(info.add(24) as *const i32) }
    };
    // Print "ltop: blocked syscall <nr>\n" using a fixed stack buffer
    // and decimal-by-hand. Avoids pulling in TinyWriter just for this
    // one diagnostic.
    fn append(buf: &mut [u8], n: &mut usize, bytes: &[u8]) {
        for &b in bytes {
            if *n < buf.len() { buf[*n] = b; *n += 1; }
        }
    }
    let mut buf = [0u8; 48];
    let mut n = 0usize;
    append(&mut buf, &mut n, b"ltop: blocked syscall ");
    let (neg, mut v) = if nr < 0 {
        (true, (nr as i64).unsigned_abs())
    } else {
        (false, nr as u64)
    };
    if neg { append(&mut buf, &mut n, b"-"); }
    let mut tmp = [0u8; 12];
    let mut tn = tmp.len();
    if v == 0 { tn -= 1; tmp[tn] = b'0'; }
    while v > 0 { tn -= 1; tmp[tn] = b'0' + (v % 10) as u8; v /= 10; }
    append(&mut buf, &mut n, &tmp[tn..]);
    append(&mut buf, &mut n, b"\n");
    let _ = syscall::write_all(2, &buf[..n]);
    syscall::exit_group(159);
}

/// x86_64 only: userspace sigreturn trampoline. The kernel lands here
/// when the SIGSYS handler returns; it issues `rt_sigreturn` (syscall
/// 15) which restores the pre-signal context. We never actually reach
/// it because the handler `exit_group`s, but rt_sigaction with
/// SA_RESTORER requires a non-null restorer pointer or the kernel
/// rejects the install.
#[cfg(all(feature = "sandbox-trap", target_arch = "x86_64"))]
#[unsafe(naked)]
unsafe extern "C" fn sigreturn_trampoline() {
    core::arch::naked_asm!(
        "mov rax, 15",   // SYS_rt_sigreturn
        "syscall",
    );
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── structural tests on PROG ───────────────────────────────────────────

    /// Sanity-check the program shape: first three instructions are
    /// LD-arch / JEQ-arch / LD-nr; the trailing two are RET-deny /
    /// RET-allow.
    #[test]
    fn prog_shape_is_well_formed() {
        assert_eq!(PROG[I_LD_ARCH].code, BPF_LD_W_ABS);
        assert_eq!(PROG[I_LD_ARCH].k, SECCOMP_DATA_ARCH);
        assert_eq!(PROG[I_JEQ_ARCH].code, BPF_JMP_JEQ_K);
        assert_eq!(PROG[I_JEQ_ARCH].k, AUDIT_ARCH);
        assert_eq!(PROG[I_LD_NR].code, BPF_LD_W_ABS);
        assert_eq!(PROG[I_LD_NR].k, SECCOMP_DATA_NR);

        assert_eq!(PROG[I_RET_DENY].code, BPF_RET_K);
        assert_eq!(PROG[I_RET_DENY].k, DENY_ACTION);
        assert_eq!(PROG[I_RET_ALLOW].code, BPF_RET_K);
        assert_eq!(PROG[I_RET_ALLOW].k, SECCOMP_RET_ALLOW);
        assert_eq!(PROG.len(), PROG_LEN);
    }

    /// Each direct-allow JEQ jumps to the ALLOW return when matched.
    /// Catches off-by-one regressions in the const-fn jump-offset
    /// computation.
    #[test]
    fn direct_allow_jumps_land_on_allow() {
        for (i, &nr) in ALLOW_DIRECT.iter().enumerate() {
            let inst_idx = I_DIRECT_START + i;
            let entry = PROG[inst_idx];
            assert_eq!(entry.code, BPF_JMP_JEQ_K);
            assert_eq!(entry.k, nr);
            assert_eq!(
                inst_idx + 1 + entry.jt as usize, I_RET_ALLOW,
                "JEQ for syscall {nr} doesn't land on allow",
            );
        }
    }

    /// The arch check's jf must land on the deny return.
    #[test]
    fn arch_mismatch_jumps_to_deny() {
        let arch = PROG[I_JEQ_ARCH];
        assert_eq!(I_JEQ_ARCH + 1 + arch.jf as usize, I_RET_DENY);
    }

    /// Classic BPF jt/jf are u8 (max 255). With current N this is
    /// well under, but make the check explicit so a future
    /// allowlist growth fails at test time, not at runtime install.
    #[test]
    fn jump_distances_fit_in_u8() {
        assert!(PROG_LEN < 256, "PROG_LEN={PROG_LEN} would overflow u8 BPF jump");
    }

    /// `SockFilter` and `SockFprog` must match the kernel ABI byte-
    /// for-byte; the kernel reads `prog.len * sizeof(sock_filter)`
    /// bytes from `prog.filter`.
    #[test]
    fn kernel_struct_layout() {
        assert_eq!(core::mem::size_of::<SockFilter>(), 8);
        assert_eq!(core::mem::align_of::<SockFilter>(), 4);
        assert_eq!(core::mem::size_of::<SockFprog>(), 16);
        assert_eq!(core::mem::align_of::<SockFprog>(), 8);
    }

    // ── BPF interpreter ────────────────────────────────────────────────────

    /// Minimal classic-BPF interpreter, just enough to run our
    /// program (LD-W-ABS, JEQ-K, ALU-AND-K, RET-K). Returns the
    /// constant in the terminating RET. Used as an oracle for what
    /// the kernel will actually do.
    fn run_bpf(prog: &[SockFilter], data: &[u8; 64]) -> u32 {
        let mut pc: usize = 0;
        let mut a: u32 = 0;
        loop {
            let i = prog[pc];
            match i.code {
                BPF_LD_W_ABS => {
                    let off = i.k as usize;
                    a = u32::from_ne_bytes([
                        data[off], data[off + 1], data[off + 2], data[off + 3],
                    ]);
                    pc += 1;
                }
                BPF_JMP_JEQ_K => {
                    pc += 1 + if a == i.k { i.jt } else { i.jf } as usize;
                }
                BPF_ALU_AND_K => {
                    a &= i.k;
                    pc += 1;
                }
                BPF_RET_K => return i.k,
                _ => panic!("unknown BPF instruction code {:#x} at pc {pc}", i.code),
            }
        }
    }

    /// Build a 64-byte `seccomp_data` blob with the given fields.
    /// `args` provides up to 6 u64 values starting at args[0].
    fn make_data(nr: i32, arch: u32, args: &[u64]) -> [u8; 64] {
        let mut d = [0u8; 64];
        d[0..4].copy_from_slice(&nr.to_ne_bytes());
        d[4..8].copy_from_slice(&arch.to_ne_bytes());
        for (i, &a) in args.iter().enumerate().take(6) {
            let off = 16 + i * 8;
            d[off..off + 8].copy_from_slice(&a.to_ne_bytes());
        }
        d
    }

    // ── direct-allow behaviour ─────────────────────────────────────────────

    /// Every direct-allow entry yields RET_ALLOW under the correct arch.
    #[test]
    fn bpf_allows_each_direct_syscall() {
        for &nr in ALLOW_DIRECT {
            let data = make_data(nr as i32, AUDIT_ARCH, &[]);
            let result = run_bpf(&PROG, &data);
            assert_eq!(
                result, SECCOMP_RET_ALLOW,
                "syscall {nr} should be allowed but BPF returned {result:#x}",
            );
        }
    }

    /// A representative set of dangerous syscalls must be denied.
    #[test]
    fn bpf_denies_dangerous_syscalls() {
        let bad_nrs: &[i32] = &[
            // execve: x86_64=59, aarch64=221
            59, 221,
            // ptrace: x86_64=101, aarch64=117
            101, 117,
            // mmap: x86_64=9, aarch64=222
            9, 222,
            // socket: x86_64=41, aarch64=198
            41, 198,
            // bpf: x86_64=321, aarch64=280
            321, 280,
            9999,
            -1,
        ];
        for &nr in bad_nrs {
            if ALLOW_DIRECT.contains(&(nr as u32))
                || nr as u32 == SYS_WRITE
                || nr as u32 == SYS_OPENAT
                || nr as u32 == SYS_IOCTL
            {
                continue;
            }
            let data = make_data(nr, AUDIT_ARCH, &[]);
            let result = run_bpf(&PROG, &data);
            assert_eq!(
                result, DENY_ACTION,
                "syscall {nr} should be denied but BPF returned {result:#x}",
            );
        }
    }

    /// Wrong-arch (32-bit / cross-arch) is denied even when nr would
    /// otherwise be allowed.
    #[test]
    fn bpf_denies_wrong_arch_even_for_allowed_nr() {
        let other_arch = if AUDIT_ARCH == AUDIT_ARCH_X86_64 {
            AUDIT_ARCH_AARCH64
        } else {
            AUDIT_ARCH_X86_64
        };
        let data = make_data(ALLOW_DIRECT[0] as i32, other_arch, &[]);
        let result = run_bpf(&PROG, &data);
        assert_eq!(result, DENY_ACTION);
    }

    /// arch=0 (malformed seccomp_data) is denied.
    #[test]
    fn bpf_denies_arch_zero() {
        let data = make_data(ALLOW_DIRECT[0] as i32, 0, &[]);
        let result = run_bpf(&PROG, &data);
        assert_eq!(result, DENY_ACTION);
    }

    // ── write filter ───────────────────────────────────────────────────────

    /// `write(1, ...)` (stdout) is allowed.
    #[test]
    fn bpf_allows_write_to_stdout() {
        let data = make_data(SYS_WRITE as i32, AUDIT_ARCH, &[1, 0, 0]);
        assert_eq!(run_bpf(&PROG, &data), SECCOMP_RET_ALLOW);
    }

    /// `write(2, ...)` (stderr) is allowed.
    #[test]
    fn bpf_allows_write_to_stderr() {
        let data = make_data(SYS_WRITE as i32, AUDIT_ARCH, &[2, 0, 0]);
        assert_eq!(run_bpf(&PROG, &data), SECCOMP_RET_ALLOW);
    }

    /// `write(0, ...)` (stdin direction-flipped) is denied.
    #[test]
    fn bpf_denies_write_to_stdin() {
        let data = make_data(SYS_WRITE as i32, AUDIT_ARCH, &[0, 0, 0]);
        assert_eq!(run_bpf(&PROG, &data), DENY_ACTION);
    }

    /// `write(3, ...)` and other arbitrary fds are denied. Catches a
    /// compromised process trying to write to the proc dirfd, the
    /// RM fd, or some accidental future fd.
    #[test]
    fn bpf_denies_write_to_other_fds() {
        for fd in [3u64, 4, 100, u32::MAX as u64] {
            let data = make_data(SYS_WRITE as i32, AUDIT_ARCH, &[fd, 0, 0]);
            assert_eq!(
                run_bpf(&PROG, &data), DENY_ACTION,
                "write(fd={fd}) should be denied",
            );
        }
    }

    // ── openat filter ──────────────────────────────────────────────────────

    /// Read-only opens are allowed (with various harmless modifiers).
    #[test]
    fn bpf_allows_read_only_openat() {
        // O_RDONLY = 0; O_CLOEXEC = 0o2000000; O_DIRECTORY varies by
        // arch but isn't in our deny mask. O_NONBLOCK = 0o4000.
        for flags in [0u64, 0o2000000, 0o2000000 | 0o4000] {
            let data = make_data(SYS_OPENAT as i32, AUDIT_ARCH, &[0, 0, flags]);
            assert_eq!(
                run_bpf(&PROG, &data), SECCOMP_RET_ALLOW,
                "openat(flags={flags:#x}) should be allowed",
            );
        }
    }

    /// Write-mode opens are denied: O_WRONLY, O_RDWR, O_CREAT,
    /// O_TRUNC, O_APPEND, __O_TMPFILE, and combinations.
    #[test]
    fn bpf_denies_write_mode_openat() {
        let bad_flags: &[u64] = &[
            0o1,         // O_WRONLY
            0o2,         // O_RDWR
            0o100,       // O_CREAT
            0o1000,      // O_TRUNC
            0o2000,      // O_APPEND
            0o20000000,  // __O_TMPFILE
            0o1 | 0o2000000,            // O_WRONLY | O_CLOEXEC
            0o2 | 0o100,                // O_RDWR | O_CREAT
        ];
        for &flags in bad_flags {
            let data = make_data(SYS_OPENAT as i32, AUDIT_ARCH, &[0, 0, flags]);
            assert_eq!(
                run_bpf(&PROG, &data), DENY_ACTION,
                "openat(flags={flags:#o}) should be denied",
            );
        }
    }

    // ── ioctl filter ───────────────────────────────────────────────────────

    /// TIOCGWINSZ (term_size in tty.rs) is allowed.
    #[test]
    fn bpf_allows_ioctl_tiocgwinsz() {
        let data = make_data(SYS_IOCTL as i32, AUDIT_ARCH, &[0, TIOCGWINSZ as u64, 0]);
        assert_eq!(run_bpf(&PROG, &data), SECCOMP_RET_ALLOW);
    }

    /// TCSETS (tty raw-mode restore on exit) is allowed.
    #[test]
    fn bpf_allows_ioctl_tcsets() {
        let data = make_data(SYS_IOCTL as i32, AUDIT_ARCH, &[0, TCSETS as u64, 0]);
        assert_eq!(run_bpf(&PROG, &data), SECCOMP_RET_ALLOW);
    }

    /// RM ioctls (NV magic 'F' + dir=11) are allowed regardless of
    /// the size/nr fields. This is the bulk of GPU-tick traffic.
    #[test]
    fn bpf_allows_ioctl_rm_family() {
        // Try a few representative RM cmds: dir=3 << 30, magic='F' <<
        // 8, varying size (16 bits) and nr (8 bits).
        for size in [0u32, 0x100, 0x4000, 0x3fff] {
            for nr in [0u32, 0x2A, 0x2B, 0xCA, 0xFF] {
                let cmd = (3u32 << 30) | (size << 16) | ((b'F' as u32) << 8) | nr;
                let data = make_data(SYS_IOCTL as i32, AUDIT_ARCH, &[0, cmd as u64, 0]);
                assert_eq!(
                    run_bpf(&PROG, &data), SECCOMP_RET_ALLOW,
                    "ioctl(cmd={cmd:#x}) should be allowed (RM)",
                );
            }
        }
    }

    /// TIOCSTI is the *dangerous* tty ioctl — it injects characters
    /// into the terminal's input buffer, letting a compromised
    /// process drive the user's shell. Must be denied.
    #[test]
    fn bpf_denies_ioctl_tiocsti() {
        const TIOCSTI: u64 = 0x5412;
        let data = make_data(SYS_IOCTL as i32, AUDIT_ARCH, &[0, TIOCSTI, 0]);
        assert_eq!(run_bpf(&PROG, &data), DENY_ACTION);
    }

    /// TCGETS is used during `tty::enter_raw_mode` (pre-install) but
    /// not afterward; deny.
    #[test]
    fn bpf_denies_ioctl_tcgets() {
        const TCGETS: u64 = 0x5401;
        let data = make_data(SYS_IOCTL as i32, AUDIT_ARCH, &[0, TCGETS, 0]);
        assert_eq!(run_bpf(&PROG, &data), DENY_ACTION);
    }

    /// Arbitrary unmatched ioctl cmds are denied.
    #[test]
    fn bpf_denies_arbitrary_ioctl_cmds() {
        let bad_cmds: &[u64] = &[
            0x12345678,
            0xc0001234,    // dir=3 but wrong magic
            0x40004600,    // magic='F' but dir=2 (read-only)
            0xff,
            u32::MAX as u64,
        ];
        for &cmd in bad_cmds {
            let data = make_data(SYS_IOCTL as i32, AUDIT_ARCH, &[0, cmd, 0]);
            assert_eq!(
                run_bpf(&PROG, &data), DENY_ACTION,
                "ioctl(cmd={cmd:#x}) should be denied",
            );
        }
    }

    // ── feature-gated DENY_ACTION value ────────────────────────────────────

    #[cfg(not(feature = "sandbox-trap"))]
    #[test]
    fn deny_action_is_kill_process_in_production() {
        assert_eq!(DENY_ACTION, SECCOMP_RET_KILL_PROCESS);
    }

    #[cfg(feature = "sandbox-trap")]
    #[test]
    fn deny_action_is_trap_in_diagnostic_mode() {
        assert_eq!(DENY_ACTION, SECCOMP_RET_TRAP);
    }
}
