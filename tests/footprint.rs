//! Assert that the production `min` binary stays within the
//! per-platform thresholds documented in README.md's Footprint section.
//! The test reads the binary's file size and compares against
//! `MAX_SIZE`, which is set to the README's rounded-up KB number for
//! the host triple (rounded up to the nearest 0.1 KB so a few bytes
//! of LTO drift don't break the test).
//!
//! The test skips with a friendly message if the binary isn't built
//! (so `cargo test --release` works locally without a prior `cargo
//! ltop`). CI workflows materialise the binary via
//! `cargo ltop -- --check` before running `cargo test --release`.
//!
//! A failing assertion means *either* the threshold needs to bump to
//! match an intentional regression (in which case update README.md
//! and `MAX_SIZE` together) or there's an unintended bloat to find
//! and trim before merging. RSS is host-dependent (page size, kernel
//! version, dyld shared cache layout) so it's documented in README
//! but not asserted here — we don't want CI flakes from external
//! state.
//!
//! Currently covered: Linux x86_64, Linux aarch64, macOS aarch64.
//! Thresholds: see `MAX_SIZE` below; the README rounds the same
//! numbers to one decimal place.

use std::fs;
use std::path::Path;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const MIN_BINARY: &str = "target/static-linux-x86_64/min/ltop";
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const MIN_BINARY: &str = "target/static-linux-aarch64/min/ltop";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MIN_BINARY: &str = "target/aarch64-apple-darwin/min/ltop";

// Thresholds in BYTES, rounded up to the nearest 0.1 KB. Each number
// is the integer floor of (display KB × 1024); a few tens of bytes of
// LTO drift fits within the rounding gap. Today's actual sizes
// (post-sandbox: seccomp-bpf with verb + arg filtering, plus
// Landlock for /proc-only path access, plus errno reporting in the
// sandbox install fail path):
//   Linux x86_64   24,272 B  → 23.9 KB
//   Linux aarch64  22,812 B  → 22.5 KB  (arm64 fixed-width insns +
//                                        no compiler_builtins memcpy
//                                        come out smaller than x86_64
//                                        on our control-flow shape)
//   macOS arm64    33,328 B  → 32.6 KB  (verified identical on
//                                        macos-14 CI runner and
//                                        macos-26 dev host — Mach-O
//                                        is reproducible across SDK
//                                        versions for our build
//                                        because we use only the
//                                        syscall ABI)
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const MAX_SIZE: u64 = 239 * 1024 / 10;       // 23.9 KB = 24,473 B
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const MAX_SIZE: u64 = 225 * 1024 / 10;       // 22.5 KB = 23,040 B
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MAX_SIZE: u64 = 326 * 1024 / 10;       // 32.6 KB = 33,382 B

#[test]
#[cfg(any(
    all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")),
    all(target_os = "macos", target_arch = "aarch64"),
))]
fn binary_size_under_threshold() {
    let path = Path::new(MIN_BINARY);
    let Ok(meta) = fs::metadata(path) else {
        eprintln!(
            "skipping: {} not built — run `cargo ltop -- --check` first",
            path.display(),
        );
        return;
    };
    let actual = meta.len();
    assert!(
        actual <= MAX_SIZE,
        "{}: {} bytes > {} threshold ({}-byte regression). Either trim or, \
         if this is intentional, raise *both* this `MAX_SIZE` and the \
         README.md Footprint table together — they're meant to stay in \
         sync.",
        path.display(),
        actual,
        MAX_SIZE,
        actual - MAX_SIZE,
    );
}
