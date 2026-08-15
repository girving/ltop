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
// is the integer floor of (display KB × 1024); ~100 B of LTO/linker
// drift fits within the rounding gap. Today's actual sizes (post
// 2026-07-08 audit fixes):
//   Linux x86_64   ~24,263 B → 23.8 KB
//   Linux aarch64  ~22,876 B → 22.5 KB  (arm64 fixed-width insns +
//                                        no compiler_builtins memcpy
//                                        come out smaller than x86_64
//                                        on our control-flow shape)
//   macOS arm64    33,312 B  → 32.6 KB  (exact — Mach-O is
//                                        reproducible across SDK
//                                        versions for our build
//                                        because we use only the
//                                        syscall ABI)
// The Linux numbers are macOS-host GNU cross-links (see
// .cargo/link-and-strip.sh's Darwin branch) adjusted by the constant
// offset that pipeline shows against CI's binaries at a known
// commit (+11 B x86_64 / +24 B aarch64 — binutils-version and
// gcc-driver-flag differences); CI's GNU output is authoritative and
// this test is where it gets enforced.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const MAX_SIZE: u64 = 238 * 1024 / 10;       // 23.8 KB = 24,371 B
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
        // CI always builds the min binary before `cargo test`; a missing
        // file there is a workflow-ordering bug and must fail loudly —
        // a green run with zero assertions would silently retire every
        // footprint claim. Locally the friendly skip stays.
        assert!(std::env::var_os("CI").is_none(),
            "{} missing under CI — the build step must run before tests",
            path.display());
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

/// The mac binary's two headline invariants, asserted rather than
/// logged (the CI workflow used to only pipe `otool -L`/`nm` to the
/// job log, where a regression needs a human reader to notice —
/// audit design take): zero imported symbols, and exactly one
/// LC_LOAD_DYLIB (libSystem, which dyld requires). Mirrors
/// stack_frames.rs's no_heap_allocator_symbols pattern.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mac_no_imports_and_single_dylib() {
    use std::process::Command;
    if !Path::new(MIN_BINARY).exists() {
        assert!(std::env::var_os("CI").is_none(),
            "{MIN_BINARY} missing under CI — the build step must run before tests");
        eprintln!("skipping: {MIN_BINARY} not built — run `cargo ltop -- --check` first");
        return;
    }
    let nm = Command::new("nm").args(["-u", MIN_BINARY]).output()
        .expect("`nm` unavailable");
    let und = String::from_utf8_lossy(&nm.stdout);
    assert!(und.trim().is_empty(),
        "imported symbols crept in (each is a libSystem call the \
         kernel-as-witness invariant forbids):\n{und}");
    let ot = Command::new("otool").args(["-L", MIN_BINARY]).output()
        .expect("`otool` unavailable");
    let text = String::from_utf8_lossy(&ot.stdout);
    let dylibs: Vec<&str> = text.lines()
        .filter(|l| l.contains(".dylib")).collect();
    assert!(dylibs.len() == 1 && dylibs[0].contains("libSystem"),
        "expected exactly one LC_LOAD_DYLIB (libSystem), got:\n{text}");
}
