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
