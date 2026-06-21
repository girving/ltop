//! Probe for libnvidia-ml.so.1; if found, set `has_nvml` cfg so the test
//! build compiles src/gpu/nvml.rs (the reference backend) and link against
//! libnvidia-ml.so.1 **only for test targets**. Production uses the RM
//! backend (src/gpu/rm.rs) directly via ioctls, with no NVML dependency.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(has_nvml)");
    println!("cargo:rustc-check-cfg=cfg(has_gpu)");
    // Set by the `cargo ltop` / `cargo stack` aliases via rustflags;
    // gates our custom `_start`, argv parser, etc. Declare so rustc
    // doesn't warn about it as an unknown cfg in ordinary builds.
    println!("cargo:rustc-check-cfg=cfg(libc_free)");

    // Production (`cargo ltop` / `cargo stack`, the
    // `static-linux-x86_64` custom target) goes through raw syscalls
    // in `src/syscall.rs` and links no libc at all — `mem*` comes
    // from `compiler-builtins-mem`. So we only emit `-lc` on the
    // other targets (glibc `cargo build --release`, macOS) where
    // `platform.rs` and `gpu/nvml.rs` still call libc functions.
    let target = std::env::var("TARGET").unwrap_or_default();
    let libc_free = target.contains("static-linux");
    if !libc_free {
        println!("cargo:rustc-link-arg-bin=ltop=-lc");
    }

    // macOS GPU backend (src/gpu/agx.rs) reads per-process GPU time from
    // the IOKit registry. We talk to IOKit through raw Mach IPC and
    // parse the kernel's OSSerializeBinary blobs ourselves
    // (mac-iokit-free.md), so the production binary links neither
    // IOKit nor CoreFoundation — only libSystem (which dyld loads
    // anyway). Test builds need IOKit for the libIOKit-vs-MIG parity
    // tests; those tests carry their own `#[link(name = "IOKit",
    // kind = "framework")]` attributes on their extern blocks, so
    // we don't need to add a global link directive here.

    // Presence of /dev/nvidiactl means the NVIDIA kernel driver is loaded
    // and reachable — the ignored GPU tests (`cargo test -- --ignored`)
    // actually need a real device node, not just NVML headers. When set,
    // #[cfg_attr(not(has_gpu), ignore)] lets them run by default.
    //
    // Only emit rerun-if-changed when the path exists. Cargo treats a
    // missing rerun-if-changed target as "always stale" (see
    // `FsStatusOutdated::StaleItem(MissingFile)`), which on a no-GPU
    // host forces build.rs (and by extension the whole crate) to
    // re-link on every `cargo ltop`. The cost of skipping the track:
    // if the NVIDIA driver is loaded *after* a cached build, cargo
    // won't notice until `cargo clean` or `touch build.rs` — a rare
    // workflow. On a GPU host the file exists, the stat is cheap,
    // and changes (driver unload) are picked up automatically.
    if Path::new("/dev/nvidiactl").exists() {
        println!("cargo:rerun-if-changed=/dev/nvidiactl");
        println!("cargo:rustc-cfg=has_gpu");
    }

    let candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        // Debian/Ubuntu cross-toolchain layout: gcc-aarch64-linux-gnu
        // installs aarch64 sysroot libs under /usr/aarch64-linux-gnu/lib.
        "/usr/aarch64-linux-gnu/lib",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/cuda/lib64",
    ];
    for dir in candidates {
        let path = Path::new(dir).join("libnvidia-ml.so.1");
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
            println!("cargo:rustc-cfg=has_nvml");
            // Only emit the linker arg on dynamically-linked targets.
            // Our static production build can't resolve a `.so` dep,
            // and release builds on glibc don't reference NVML symbols
            // anyway — Rust's default `--as-needed` drops the flag at
            // link time.
            if !libc_free {
                println!("cargo:rustc-link-search=native={dir}");
                // `-l:filename` forces GNU ld to link the exact soname,
                // avoiding the `libnvidia-ml.so` dev symlink.
                println!("cargo:rustc-link-arg=-l:libnvidia-ml.so.1");
            }
            return;
        }
    }
}
