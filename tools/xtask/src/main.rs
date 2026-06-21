//! Host-arch dispatcher behind the `cargo stack` / `cargo ltop`
//! aliases. Cargo aliases are static text — they can't pick
//! `--target .cargo/static-linux-<arch>.json` based on the machine
//! running the build. This ~50-line binary reads
//! `env::consts::ARCH` + `env::consts::OS`, assembles the right
//! cargo invocation, and exec-runs it so the user-visible tool is
//! still a single `cargo stack` (or `cargo ltop`) that works on
//! Linux x86_64, Linux aarch64, and macOS aarch64 alike.
//!
//! Cargo invokes us via:
//!
//!   [alias]
//!   stack = "run --package xtask --release --quiet -- stack"
//!   ltop  = "run --package xtask --release --quiet -- ltop"
//!
//! so xtask's argv[1] is the subcommand name; everything after it
//! is forwarded. `cargo` keeps a single leading `--` when aliases
//! are expanded, which we strip here so the inner cargo doesn't see
//! a spurious separator.

use std::process::{exit, Command};

fn main() {
    let mut args = std::env::args().skip(1);
    let subcmd = args.next().unwrap_or_else(|| die("missing subcommand"));
    let mut rest: Vec<String> = args.filter(|a| a != "--").collect();

    // `cargo ltop --install`: build the min binary, then copy it onto
    // $PATH instead of running it. Folded into the `ltop` subcommand
    // (rather than its own alias) so there's one command to remember.
    let do_install = subcmd == "ltop" && rest.first().map(String::as_str) == Some("--install");
    if do_install {
        rest.remove(0);
    }

    // macOS: stock `aarch64-apple-darwin` target (Mach-O, libSystem-
    // dynamic — we can't ditch libc or PIE the way we do on Linux).
    // Linux: custom JSON target that strips crt + libc so the shipped
    // binary is fully static and libc-free.
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let target: &str = match (os, arch) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "x86_64") => ".cargo/static-linux-x86_64.json",
        ("linux", "aarch64") => ".cargo/static-linux-aarch64.json",
        (os, arch) => die(&format!("unsupported host: {os}/{arch}")),
    };

    // Every subcommand routes through `cargo build`; `ltop` then runs
    // the resulting binary (or, with `--install`, copies it to $PATH).
    //
    // `cargo stack`:           min-stack binary tools/stack-svg consumes.
    // `cargo ltop`:            min profile, then run with forwarded args.
    // `cargo ltop --install`:  min profile, then copy to the cargo bin
    //                          dir so the libc-free static binary lands
    //                          on $PATH — same dispatch as `cargo ltop`,
    //                          so the installed binary is byte-identical
    //                          to what you'd run in-tree, on any host.
    // `cargo ltop-dbg`:        min-dbg profile (strip=false), nm/objdump.
    let profile = match subcmd.as_str() {
        "stack" => "min-stack",
        "ltop" => "min",
        "ltop-dbg" => "min-dbg",
        other => die(&format!("unknown xtask subcommand: {other}")),
    };

    let mut c = Command::new("cargo");
    c.args(["build", "--profile", profile, "--target", target]);
    // -Z build-std flags have to live on the command line (not in
    // .cargo/config.toml's [unstable]) so they apply only to this
    // invocation, not to `cargo test` / `cargo build` for host.
    c.args([
        "-Z", "build-std=core,alloc,panic_abort",
        "-Z", "build-std-features=compiler-builtins-mem",
    ]);
    // On mac the Linux-specific flags live in the custom-target
    // config section; we inject the portable subset via RUSTFLAGS
    // only for this invocation so the default `cargo build --release`
    // (same triple, no build-std) isn't affected.
    if os == "macos" {
        // Apple ld flags to tighten a Mach-O that imports nothing:
        //   -dead_strip_dylibs   drops LC_LOAD_DYLIB entries whose
        //                        symbols are unreferenced (libiconv;
        //                        libSystem stays because it's special-
        //                        cased by the loader).
        //   -no_data_const       merges __DATA_CONST into __DATA so we
        //                        don't pay a full 16 KB page for the
        //                        ~500 B of actually-constant data.
        //                        Loses the R-only hardening on const
        //                        data; acceptable for a tiny monitor.
        //   -no_compact_unwind   drops the Apple-specific
        //                        __TEXT,__unwind_info section. Our
        //                        panic handler aborts; no unwinder.
        //   -no_function_starts  Drops LC_FUNCTION_STARTS (16 B) and
        //                        its __LINKEDIT payload. Only used
        //                        by dtrace / ips crash-report symbol
        //                        resolution; execution doesn't need it.
        //   -segprot __TEXT rx rx
        //                        Pin __TEXT's maxprot = initprot =
        //                        r-x (default is rwx/r-x). Tells the
        //                        linker we never need the segment
        //                        writable — may encourage tighter
        //                        layout on some ld64 versions.
        // `-Zemit-stack-sizes` is appended only for the stack subcommand
        // so the shipped `cargo ltop` binary (min profile, packed) doesn't
        // carry the section. On Mach-O it lands in `__LLVM,__stack_sizes`.
        let stack_sizes = if subcmd == "stack" { " -Zemit-stack-sizes" } else { "" };
        c.env("RUSTFLAGS", format!(
            "-Zunstable-options -Cpanic=immediate-abort -Cforce-unwind-tables=no \
             -Clink-arg=-Wl,-dead_strip_dylibs \
             -Clink-arg=-Wl,-no_data_const \
             -Clink-arg=-Wl,-no_compact_unwind \
             -Clink-arg=-Wl,-no_function_starts \
             -Clink-arg=-Wl,-segprot,__TEXT,rx,rx{stack_sizes}"));
    }
    // `cargo stack` / `cargo ltop-dbg` forward extra args (e.g. `-v`)
    // to the build; `ltop` reserves them for the binary.
    if subcmd == "stack" || subcmd == "ltop-dbg" {
        c.args(&rest);
    }
    let s = c.status().unwrap_or_else(|e| die(&format!("cargo build: {e}")));
    if !s.success() { exit(s.code().unwrap_or(1)); }

    // Cargo names the output dir after the target's *short name*: a
    // plain triple (`aarch64-apple-darwin`) as-is, but a custom JSON
    // spec by its file stem (`.cargo/static-linux-aarch64.json` →
    // `static-linux-aarch64`). Derive that, don't reuse `target`.
    let target_dir = {
        let base = target.rsplit('/').next().unwrap_or(target);
        base.strip_suffix(".json").unwrap_or(base)
    };
    let bin = format!("target/{target_dir}/{profile}/ltop");

    // `cargo ltop` (run or `--install`) produces a runnable min binary;
    // on mac, pack it (collapses __TEXT page padding, lesson 19) before
    // running or installing. Always attempt — tested in CI.
    if os == "macos" && subcmd == "ltop" {
        let packed = format!("{bin}.packed");
        let s = Command::new("cargo")
            .args(["run", "--package", "mac-pack", "--release", "--quiet", "--"])
            .arg(&bin)
            .arg(&packed)
            .status()
            .unwrap_or_else(|e| die(&format!("spawn mac-pack: {e}")));
        if !s.success() {
            die(&format!("mac-pack failed (exit {})", s.code().unwrap_or(1)));
        }
        std::fs::rename(&packed, &bin)
            .unwrap_or_else(|e| die(&format!("rename {packed} → {bin}: {e}")));
    }

    // Copy onto $PATH (`fs::copy` preserves the executable bit), or run.
    if do_install {
        let dir = install_dir();
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| die(&format!("create {dir}: {e}")));
        let dest = format!("{dir}/ltop");
        std::fs::copy(&bin, &dest)
            .unwrap_or_else(|e| die(&format!("copy {bin} → {dest}: {e}")));
        eprintln!("xtask: installed minimal ltop → {dest}");
    } else if subcmd == "ltop" {
        let s = Command::new(&bin)
            .args(&rest)
            .status()
            .unwrap_or_else(|e| die(&format!("spawn {bin}: {e}")));
        exit(s.code().unwrap_or(1));
    }
}

// Cargo's install root: $CARGO_INSTALL_ROOT/bin, else $CARGO_HOME/bin,
// else ~/.cargo/bin — the same precedence `cargo install` honours.
fn install_dir() -> String {
    if let Ok(d) = std::env::var("CARGO_INSTALL_ROOT") {
        format!("{d}/bin")
    } else if let Ok(d) = std::env::var("CARGO_HOME") {
        format!("{d}/bin")
    } else if let Ok(h) = std::env::var("HOME") {
        format!("{h}/.cargo/bin")
    } else {
        die("cannot determine install dir: set CARGO_INSTALL_ROOT, CARGO_HOME, or HOME");
    }
}

fn die(msg: &str) -> ! {
    eprintln!("xtask: {msg}");
    exit(2);
}
