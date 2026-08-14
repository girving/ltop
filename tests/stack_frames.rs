//! Verify no single function frame exceeds 4 KB — the threshold rustc
//! uses to decide whether to emit page-walking stack probes.
//!
//! Why this matters for ltop: the production build resets RSP to the top
//! of the stack VMA in `start.rs` and `madvise(DONTNEED)`s everything
//! below the current page. As long as individual Rust frames stay
//! under one page, no function's probes ever reach those dropped
//! pages. The moment a future change grows a frame to ≥ 4 KB, rustc
//! emits `sub rsp, 0x1000; movq $0, (rsp)` probes, which re-commit
//! dropped pages and negate the RSS win until the frame unwinds.
//!
//! This test reads `.stack_sizes` from the `min-stack` binary. If the
//! binary doesn't exist, the test is skipped (build it with
//! `cargo stack`). CI or a pre-push hook should run both:
//!     cargo stack && cargo test --release --test stack_frames

use std::path::Path;

// Per-arch path to the matching min-stack binary produced by
// `cargo stack` — the alias dispatches to the right target JSON
// based on the host. The two custom targets share the same
// `min-stack` profile settings; the ELF parser below is endian- and
// arch-agnostic and the 4 KB probe threshold holds on both archs
// (4 KB pages, rustc emits the same page-walking probes when a
// frame ≥ one page).
#[cfg(target_arch = "x86_64")]
const MIN_STACK_BINARY: &str = "target/static-linux-x86_64/min-stack/ltop";
#[cfg(target_arch = "aarch64")]
const MIN_STACK_BINARY: &str = "target/static-linux-aarch64/min-stack/ltop";

const STACK_PROBE_THRESHOLD: u64 = 4096;

fn load_min_stack() -> Option<Vec<u8>> {
    let path = Path::new(MIN_STACK_BINARY);
    match std::fs::read(path) {
        Ok(b) => Some(b),
        Err(_) => {
            // Linux CI builds min-stack (`cargo stack`) before tests; the
            // mac workflow has no stack stage, so the hard-fail gate is
            // Linux-only.
            #[cfg(target_os = "linux")]
            assert!(std::env::var_os("CI").is_none(),
                "{} missing under CI — the build step must run before tests", path.display());
            eprintln!("skipping: {} not built — run `cargo stack` first", path.display());
            None
        }
    }
}

#[test]
fn no_frame_hits_stack_probe_threshold() {
    let Some(bytes) = load_min_stack() else { return };

    let entries = parse_stack_sizes(&bytes).expect("parse .stack_sizes");
    let mut largest: Option<(u64, u64)> = None;
    for &(addr, size) in &entries {
        if largest.map_or(true, |(_, max)| size > max) {
            largest = Some((addr, size));
        }
    }

    let (addr, max) = largest.expect(".stack_sizes is empty");
    assert!(
        max < STACK_PROBE_THRESHOLD,
        "function at 0x{addr:x} has a {max}-byte frame — ≥ {STACK_PROBE_THRESHOLD} B triggers \
         rustc's page-walking stack probes, which re-commit madvise'd pages and break the \
         stack-relocation RSS win in src/start.rs. Shrink the frame (move the large local \
         to the arena, see the >1 KB rule in CLAUDE.md) or raise the threshold if we've \
         given up on the RSS target."
    );
}

/// `.init_array` should be absent or hold at most one entry. Our
/// custom `_start` (`src/start.rs`) skips `__libc_start_init`, so
/// *any* real global constructor silently never runs. If a dep adds
/// one, this test fires so we notice. After dropping the
/// self-contained crt bundle the section is absent entirely — treat
/// that as 0 entries (which is ≤ 1).
#[test]
fn init_array_minimal() {
    let Some(bytes) = load_min_stack() else { return };
    let entries = init_array_entries(&bytes).unwrap_or(0);
    assert!(
        entries <= 1,
        ".init_array has {entries} entries — our custom _start skips __libc_start_init, \
         so any global constructor silently never runs. Investigate which dep introduced \
         the new ctor and either move it off .init_array or call it explicitly from \
         ltop_entry."
    );
}

/// `.rodata` must land at a lower virtual address than `.text`. This
/// is what lets it ride along in the ELF-header page of the first
/// R-only LOAD segment instead of needing its own LOAD + ~4 KB of
/// page-alignment padding. Enforced by `.cargo/rodata-first.ld`, a
/// linker-script fragment that moves `.rodata` ahead of `.text` via
/// `INSERT AFTER .note.gnu.build-id`. If that stops working (rustc
/// changes its default script, the fragment gets dropped, etc.), the
/// binary grows ~4 KB silently — this test fails first.
#[test]
fn rodata_before_text() {
    let Some(bytes) = load_min_stack() else { return };
    let rodata = section_vaddr(&bytes, ".rodata").expect("no .rodata section");
    let text = section_vaddr(&bytes, ".text").expect("no .text section");
    assert!(
        rodata < text,
        ".rodata at 0x{rodata:x} is not before .text at 0x{text:x} — \
         `.cargo/rodata-first.ld` isn't taking effect. Without this \
         ordering `.rodata` gets its own LOAD segment + page of \
         alignment padding (~4 KB of file size). Check that the \
         rustflag `-Clink-arg=-Wl,-T,.cargo/rodata-first.ld` is still \
         in `.cargo/config.toml` and that the ld `INSERT AFTER` \
         directive still works in the installed toolchain."
    );
}

/// The production binary is strictly arena-allocated and links no libc;
/// `platform::Abort` is installed as `#[global_allocator]` and aborts on
/// every call. So any libc allocator (malloc / calloc / aligned_alloc
/// and the indirect allocators that pull them in — opendir/readdir via
/// calloc, fopen via the FILE struct) should be absent. A hit here
/// usually means a `Vec`/`String`/`Box` snuck into a production path,
/// or a new libc call internally allocates — see CLAUDE.md §9, §10 for
/// the history. Fix by routing through the arena or swapping for a raw
/// syscall.
#[test]
fn no_heap_allocator_symbols() {
    use std::process::Command;
    if !Path::new(MIN_STACK_BINARY).exists() {
        #[cfg(target_os = "linux")]
        assert!(std::env::var_os("CI").is_none(),
            "{MIN_STACK_BINARY} missing under CI — the build step must run before tests");
        eprintln!("skipping: {MIN_STACK_BINARY} not built — run `cargo stack` first");
        return;
    }
    let out = Command::new("nm").arg(MIN_STACK_BINARY).output()
        .expect("`nm` unavailable — install binutils");
    let stdout = std::str::from_utf8(&out.stdout).expect("nm output not utf-8");

    // Exact-match against the standard libc allocator entry points
    // and two libc helpers we've previously caught pulling malloc in.
    const DENYLIST: &[&str] = &[
        "malloc", "calloc", "realloc", "free", "aligned_alloc",
        "__libc_malloc_impl", "__libc_free", "__libc_calloc",
        "alloc_slot", "nontrivial_free", "__malloc_context",
        "opendir", "readdir", "closedir",
    ];
    let hits: Vec<&str> = stdout.lines()
        .filter_map(|l| l.split_whitespace().last())
        .filter(|s| DENYLIST.iter().any(|bad| s == bad))
        .collect();

    assert!(
        hits.is_empty(),
        "binary links heap-allocator symbols: {hits:?}. ltop's #[global_allocator] \
         is an aborting stub (src/platform.rs::Abort) — any live call would crash. \
         A regression here almost always means a new `Vec`/`String`/`Box` in a \
         production path, or an indirect allocator (opendir, fopen, …) just \
         snuck in. See CLAUDE.md §9/§10."
    );
}

/// Steady-state stack RSS of a running ltop must be exactly one page
/// (4 kB). That's the payoff from `src/start.rs`'s rsp-relocation +
/// `madvise(DONTNEED)` walk: every stack VMA page except the one
/// holding the current frame gets dropped, and `no_frame_hits_stack_
/// probe_threshold` above keeps individual frames under 4 KB so the
/// whole tick loop's stack (peak ~2 KB per `tools/stack-svg`) lives
/// in that single page without spilling.
///
/// Regressions this test catches:
///   - `start.rs` stops relocating rsp (sp lands mid-page → 2-page RSS).
///   - The upward `madvise` probe is broken (old kernel sp-page stays
///     committed → 2-page RSS).
///   - The downward `madvise` sweep in `ltop_main` regresses.
///   - A frame grows past 4 KB and rustc inserts page probes that
///     re-commit pages the tick loop never would otherwise touch.
///   - Anything else that balloons the tick's working-set stack.
///
/// Runs ltop as a subprocess for 2.5 s, reads `/proc/PID/smaps`, and
/// asserts the [stack] line's `Rss:` is ≤ 4 kB. Linux-only (relies on
/// Linux-specific smaps; skips on macOS tests).
#[cfg(target_os = "linux")]
#[test]
fn steady_state_stack_rss_one_page() {
    use std::process::{Command, Stdio};
    use std::thread::sleep;
    use std::time::Duration;

    if !Path::new(MIN_STACK_BINARY).exists() {
        #[cfg(target_os = "linux")]
        assert!(std::env::var_os("CI").is_none(),
            "{MIN_STACK_BINARY} missing under CI — the build step must run before tests");
        eprintln!("skipping: {MIN_STACK_BINARY} not built — run `cargo stack` first");
        return;
    }

    // Spawn with stdout/stderr sent to /dev/null — ltop draws a TUI
    // frame every 2s and we don't want that scrolling into the test
    // output.
    let mut child = Command::new(MIN_STACK_BINARY)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped()) // capture for diagnostic on failure
        .spawn().expect("spawn ltop");
    let pid = child.id();

    // Steady state: tick 1 fires at 300 ms, tick 2 at 2.3 s. After
    // tick 1, arena::uncommit_tail has run. Sleep past tick 2 so the
    // stack has exercised its full working-set depth at least once
    // before we sample.
    sleep(Duration::from_millis(2500));

    let smaps = std::fs::read_to_string(format!("/proc/{pid}/smaps"))
        .expect("read /proc/PID/smaps");
    child.kill().ok();
    let status = child.wait().ok();
    // Drain captured stderr after kill+wait so the pipe is closed.
    // ltop's sandbox install path writes diagnostic messages here
    // ("ltop: sandbox install failed (S err=N)") before exit_group;
    // if ltop died of its own accord they're the smoking gun.
    let mut err = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        use std::io::Read;
        let _ = stderr.read_to_string(&mut err);
    }

    // Find the [stack] VMA's Rss field. smaps format: a header line
    // ("addr-addr perms offset …  [name]") followed by key/value
    // entries until the next header.
    let stack_rss_kb = match smaps.split('\n')
        .skip_while(|l| !l.ends_with("[stack]"))
        .find_map(|l| l.strip_prefix("Rss:")
                      .and_then(|s| s.split_whitespace().next())
                      .and_then(|n| n.parse::<u64>().ok()))
    {
        Some(kb) => kb,
        None => panic!(
            "no [stack] Rss line in smaps. smaps len = {} bytes; \
             child wait status = {status:?}; child stderr = {err:?}",
            smaps.len(),
        ),
    };

    assert!(
        stack_rss_kb <= 4,
        "[stack] Rss = {stack_rss_kb} kB, expected ≤ 4 (one page). One of: \
         ltop_entry's rsp-relocation broke, ltop_main's downward madvise \
         sweep broke, or a frame grew past 4 KB and rustc's page probes \
         now re-commit pages the tick loop didn't otherwise touch. Check \
         `tools/stack-svg` output: peak stack ≤ 2 KB was the original budget."
    );
}

// ── minimal ELF64 + .stack_sizes parser ────────────────────────────────────
//
// Only what we need to pull frame sizes. Mirrors the logic in
// tools/stack-svg but inlined here so this test has no deps.

fn u16le(b: &[u8], o: usize) -> u16 { u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) }
fn u32le(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn u64le(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

fn cstr(bytes: &[u8], off: usize) -> &str {
    if off >= bytes.len() { return ""; }
    let end = bytes[off..].iter().position(|&b| b == 0).unwrap_or(bytes.len() - off);
    std::str::from_utf8(&bytes[off..off + end]).unwrap_or("")
}

/// Count 8-byte function-pointer entries in `.init_array`.
fn init_array_entries(elf: &[u8]) -> Option<u64> {
    let (_, size) = find_section(elf, ".init_array")?;
    Some(size / 8)
}

fn find_section<'a>(elf: &'a [u8], target: &str) -> Option<(&'a [u8], u64)> {
    let h = section_header(elf, target)?;
    Some((&elf[h.file_off..h.file_off + h.size], h.size as u64))
}

/// Virtual address of the named section, or None if absent.
fn section_vaddr(elf: &[u8], target: &str) -> Option<u64> {
    Some(section_header(elf, target)?.vaddr)
}

struct SectionHdr { vaddr: u64, file_off: usize, size: usize }

fn section_header(elf: &[u8], target: &str) -> Option<SectionHdr> {
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return None;
    }
    let shoff = u64le(elf, 0x28) as usize;
    let shentsize = u16le(elf, 0x3a) as usize;
    let shnum = u16le(elf, 0x3c) as usize;
    let shstrndx = u16le(elf, 0x3e) as usize;

    let shstr_base = shoff + shstrndx * shentsize;
    let shstr_off = u64le(elf, shstr_base + 0x18) as usize;
    let shstr_size = u64le(elf, shstr_base + 0x20) as usize;
    let shstrtab = &elf[shstr_off..shstr_off + shstr_size];

    for i in 0..shnum {
        let base = shoff + i * shentsize;
        let name_off = u32le(elf, base) as usize;
        let sh_type = u32le(elf, base + 4);
        if cstr(shstrtab, name_off) == target && sh_type != 8 {
            return Some(SectionHdr {
                vaddr: u64le(elf, base + 0x10),
                file_off: u64le(elf, base + 0x18) as usize,
                size: u64le(elf, base + 0x20) as usize,
            });
        }
    }
    None
}

fn parse_stack_sizes(elf: &[u8]) -> Option<Vec<(u64, u64)>> {
    let (data, _) = find_section(elf, ".stack_sizes")?;
    parse_stack_sizes_from_section(data)
}

fn parse_stack_sizes_from_section(data: &[u8]) -> Option<Vec<(u64, u64)>> {
    // Each entry: u64 address, ULEB128 size.
    let mut entries = Vec::new();
    let mut i = 0;
    while i + 8 <= data.len() {
        let addr = u64le(data, i);
        i += 8;
        let mut size = 0u64;
        let mut shift = 0u32;
        loop {
            if i >= data.len() { return Some(entries); }
            let b = data[i];
            i += 1;
            size |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 { break; }
            shift += 7;
            if shift >= 64 { return Some(entries); }
        }
        entries.push((addr, size));
    }
    Some(entries)
}
