//! End-to-end check that `sandbox::install()` actually transitions
//! the process to seccomp filter mode, by re-execing this test
//! binary as a child with an env var set; the child installs the
//! sandbox, reads /proc/self/status, and exits 0 only if the kernel
//! reports `Seccomp: 2`. Any non-zero exit fails the parent assert.
//!
//! Linux-only. The two-process design keeps the parent test harness
//! unsandboxed (it'd otherwise be unable to print results, manage
//! threads, etc.) while still getting an honest "the kernel says
//! the filter is in force" verdict from the child.
//!
//! Why a fresh child instead of just installing inline: once the
//! filter is on, every subsequent syscall has to be in the
//! allowlist. The Rust test harness's post-test bookkeeping
//! (mutexes, thread joins, stdout flushing) issues syscalls
//! (`futex`, `sched_yield`, possibly `mmap`) that we deliberately
//! exclude from the allowlist. Letting the test return normally
//! would terminate the whole test process. The child sidesteps
//! that by `exit_group`'ing as soon as the verification finishes.

#![cfg(target_os = "linux")]

use std::env;
use std::process::Command;

const CHILD_ENV: &str = "LTOP_SANDBOX_TEST_CHILD";
const TEST_NAME: &str = "sandbox_install_then_seccomp_field_is_filter_mode";
const CHILD_ENV_SIGSYS: &str = "LTOP_SANDBOX_TEST_CHILD_SIGSYS";
const CHILD_ENV_SLEEP: &str = "LTOP_SANDBOX_TEST_CHILD_SLEEP";

/// The deny action, end to end: a child installs the sandbox and then
/// issues a syscall that is deliberately startup-only (absent from the
/// tick allowlist). The kernel must kill the whole process with
/// SIGSYS — the in-file BPF interpreter tests share the allowlist
/// assumption, so only a real kernel round-trip can falsify it.
#[test]
fn sandbox_denied_syscall_kills_with_sigsys() {
    if env::var(CHILD_ENV_SIGSYS).is_ok() {
        ltop::sandbox::install();
        let mut set = [0u64; 16];
        let _ = ltop::syscall::sched_getaffinity_self(&mut set);
        // Reached ⇒ the filter let a denied syscall through.
        ltop::syscall::exit_group(13);
    }
    let exe = env::current_exe().expect("current_exe");
    let status = Command::new(&exe)
        .env(CHILD_ENV_SIGSYS, "1")
        .args(["--exact", "sandbox_denied_syscall_kills_with_sigsys"])
        .status()
        .expect("failed to spawn child");
    use std::os::unix::process::ExitStatusExt;
    // KILL_PROCESS terminates with SIGSYS (31); under the sandbox-trap
    // feature the SIGSYS handler prints the nr and exits 159 instead.
    assert!(
        status.signal() == Some(31) || status.code() == Some(159),
        "denied syscall did not SIGSYS-kill the child: {status:?}",
    );
}

/// SIGSTOP/SIGCONT landing in the tick loop's nanosleep makes the
/// kernel re-enter via restart_syscall (no handler ran, so
/// -ERESTART_RESTARTBLOCK restarts rather than EINTRs). The filter
/// must allow that re-entry — it once didn't, and Ctrl-Z + fg killed
/// the monitor on resume.
#[test]
fn sandbox_survives_stop_cont_during_nanosleep() {
    if env::var(CHILD_ENV_SLEEP).is_ok() {
        ltop::sandbox::install();
        // ~3 s of 100 ms nanosleeps so the parent's STOP lands in one.
        let ts = ltop::syscall::Timespec { tv_sec: 0, tv_nsec: 100_000_000 };
        for _ in 0..30 {
            unsafe { ltop::syscall::nanosleep(&ts, core::ptr::null_mut()); }
        }
        ltop::syscall::exit_group(0);
    }
    let exe = env::current_exe().expect("current_exe");
    let mut child = Command::new(&exe)
        .env(CHILD_ENV_SLEEP, "1")
        .args(["--exact", "sandbox_survives_stop_cont_during_nanosleep"])
        .spawn()
        .expect("failed to spawn child");
    std::thread::sleep(std::time::Duration::from_millis(700));
    let pid = child.id().to_string();
    for (sig, pause_ms) in [("-STOP", 300u64), ("-CONT", 0)] {
        let ok = Command::new("kill").args([sig, &pid]).status()
            .expect("spawn kill").success();
        assert!(ok, "kill {sig} failed");
        std::thread::sleep(std::time::Duration::from_millis(pause_ms));
    }
    let status = child.wait().expect("wait");
    assert!(status.success(), "child died across STOP/CONT: {status:?}");
}

#[test]
fn sandbox_install_then_seccomp_field_is_filter_mode() {
    if env::var(CHILD_ENV).is_ok() {
        run_child_check();
    }

    // Parent: respawn ourselves with the env var set, asking the
    // harness to run only this test in the child. Re-using
    // `current_exe()` means the child has the lib symbols it needs
    // (`ltop::sandbox::install`, the syscall wrappers) without us
    // having to manage a separate binary target.
    let exe = env::current_exe().expect("current_exe");
    let status = Command::new(&exe)
        .env(CHILD_ENV, "1")
        .args(["--exact", TEST_NAME])
        .status()
        .expect("failed to spawn child");
    assert!(status.success(), "child exit status: {status:?}");
}

/// Child-side verification. Never returns; exits the process with
/// a code that encodes pass/fail.
fn run_child_check() -> ! {
    // Install the sandbox. Any failure here `exit_group(2)`s on its
    // own, which the parent will see as `!status.success()`.
    ltop::sandbox::install();

    // Read /proc/thread-self/status with sandbox-allowed syscalls
    // only. NOT /proc/self/status: the test harness runs each test
    // on a worker thread, and seccomp mode is per-task. /proc/self
    // resolves to the thread-group leader's directory, whose mode
    // stays 0; /proc/thread-self resolves to the *calling* thread,
    // which is the one we just installed seccomp on. Likewise
    // std::fs::read_to_string would call `fstat` (not in our
    // allowlist) to size the buffer; we can't use it.
    let fd = ltop::syscall::open_cstr(c"/proc/thread-self/status", ltop::syscall::O_RDONLY);
    if fd < 0 {
        ltop::syscall::exit_group(10);
    }
    let mut buf = [0u8; 8192];
    let n = ltop::syscall::read_buf(fd, &mut buf);
    ltop::syscall::close(fd);
    if n <= 0 {
        ltop::syscall::exit_group(11);
    }
    let text = &buf[..n as usize];

    // Find "Seccomp:\t<n>" — distinct from the also-present
    // "Seccomp_filters:" line. Mode 2 = SECCOMP_MODE_FILTER.
    let mut found_filter_mode = false;
    for line in text.split(|&b| b == b'\n') {
        if let Some(rest) = line.strip_prefix(b"Seccomp:") {
            let mut i = 0;
            while i < rest.len() && (rest[i] == b'\t' || rest[i] == b' ') {
                i += 1;
            }
            found_filter_mode = i < rest.len() && rest[i] == b'2';
            break;
        }
    }

    ltop::syscall::exit_group(if found_filter_mode { 0 } else { 12 });
}
