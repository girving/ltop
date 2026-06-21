//! inline-bytes: attribute `.text` bytes of a min-stack binary to
//! source-level functions by walking DWARF `DW_TAG_inlined_subroutine`
//! entries. Complements `stack-svg`, which does the same for stack
//! frames — this tool answers "of the N bytes in function F, which
//! inlined callees ate them".
//!
//! Useful against opt-level=z + LTO binaries where one large
//! `.text` symbol (e.g. `ltop::run::{closure#0}` at 14 KB) is the
//! union of a dozen inlined sub-functions. `nm` just tells you the
//! 14 KB exists; DWARF tells you what's inside.
//!
//! Usage:
//!
//!     inline-bytes <binary> [fn-fragment]
//!
//! With no `fn-fragment`: print the top 30 functions by self+inlined
//! byte count. With a fragment: print the full inlined-call tree
//! for every function whose demangled name contains that fragment.
//!
//! The binary needs DWARF + symbols (build with `cargo stack`, which
//! uses the `min-stack` profile that keeps both).

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use gimli::{AttributeValue, DebugInfoOffset, EndianSlice, LittleEndian, UnitSectionOffset};

// ── output bounds ───────────────────────────────────────────────────────────
//
// Printing the full DWARF inline tree of a `min-stack` binary without
// limits overwhelms any terminal: deep monomorphised chains plus wide
// fan-out (e.g. every closure in `core::fmt`) can emit millions of
// lines. These caps keep output to something a human actually reads.

const MAX_DEPTH: usize = 12;
const MAX_CHILDREN_PER_LEVEL: usize = 20;
const MAX_TOTAL_LINES: usize = 5000;
const MIN_BYTES_AT_LEAF: u64 = 8;
const MAX_NEEDLE_HITS: usize = 8;

// ── ELF sections we need to feed gimli ──────────────────────────────────────

fn u16le(b: &[u8], o: usize) -> u16 { u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) }
fn u32le(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn u64le(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

fn cstr(strtab: &[u8], off: usize) -> &str {
    if off >= strtab.len() { return ""; }
    let end = strtab[off..].iter().position(|&b| b == 0).unwrap_or(strtab.len() - off);
    std::str::from_utf8(&strtab[off..off + end]).unwrap_or("<invalid-utf8>")
}

struct Section<'a> { name: String, data: &'a [u8] }

fn parse_sections<'a>(buf: &'a [u8]) -> Result<Vec<Section<'a>>, String> {
    // ELF64 little-endian.
    if buf.len() >= 64 && &buf[..4] == b"\x7fELF" && buf[4] == 2 && buf[5] == 1 {
        return Ok(parse_elf64_le(buf));
    }
    // Mach-O 64-bit little-endian (binary or dSYM companion).
    if buf.len() >= 32 && u32le(buf, 0) == 0xfeedfacf {
        return Ok(parse_macho64_le(buf));
    }
    Err("not an ELF64-little or Mach-O 64-bit-little binary".into())
}

fn parse_elf64_le<'a>(elf: &'a [u8]) -> Vec<Section<'a>> {
    let shoff = u64le(elf, 0x28) as usize;
    let shentsize = u16le(elf, 0x3a) as usize;
    let shnum = u16le(elf, 0x3c) as usize;
    let shstrndx = u16le(elf, 0x3e) as usize;

    // SHT_NOBITS (type 8) has a file offset but no bytes on disk.
    let read = |i: usize| -> (&[u8], u64) {
        let o = shoff + i * shentsize;
        let name_off = u32le(elf, o) as u64;
        let ty = u32le(elf, o + 4);
        let off = u64le(elf, o + 0x18) as usize;
        let size = u64le(elf, o + 0x20) as usize;
        let lo = off.min(elf.len());
        let hi = off.saturating_add(size).min(elf.len());
        let data: &[u8] = if ty == 8 { &[] } else { &elf[lo..hi] };
        (data, name_off)
    };

    let (shstrtab, _) = read(shstrndx);
    let mut out = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let (data, name_off) = read(i);
        out.push(Section { name: cstr(shstrtab, name_off as usize).to_string(), data });
    }
    out
}

/// Mach-O 64-bit walker: pull every `LC_SEGMENT_64`'s sections and
/// feed gimli the DWARF ones. Mach-O names sections like
/// `__DWARF,__debug_info`; gimli expects ELF-style `.debug_info`, so
/// we rewrite `__name` → `.name`. Works on both the binary itself
/// (e.g. dSYM-less builds where DWARF lives in `__DWARF` segment of
/// the executable, rare on macOS) and on `.dSYM` companion files
/// (`<binary>.dSYM/Contents/Resources/DWARF/<binary>` — also Mach-O,
/// holds only DWARF sections, what `dsymutil` produces).
fn parse_macho64_le<'a>(buf: &'a [u8]) -> Vec<Section<'a>> {
    const LC_SEGMENT_64: u32 = 0x19;
    let ncmds = u32le(buf, 0x10) as usize;
    let mut out = Vec::new();
    let mut o = 0x20; // load commands start
    for _ in 0..ncmds {
        if o + 8 > buf.len() { break; }
        let cmd = u32le(buf, o);
        let cmdsize = u32le(buf, o + 4) as usize;
        if cmdsize == 0 { break; }
        if cmd == LC_SEGMENT_64 {
            let nsects = u32le(buf, o + 0x40) as usize;
            // section_64 entries follow the LC_SEGMENT_64 header (72 B).
            for i in 0..nsects {
                let s = o + 72 + i * 80;
                if s + 80 > buf.len() { break; }
                let sectname = cstr_fixed(&buf[s..s + 16]);
                let foff = u32le(buf, s + 0x30) as usize;
                let fsz = u64le(buf, s + 0x28) as usize;
                let lo = foff.min(buf.len());
                let hi = foff.saturating_add(fsz).min(buf.len());
                // Mach-O __debug_info → ELF-style .debug_info for gimli.
                let name = if let Some(rest) = sectname.strip_prefix("__") {
                    format!(".{rest}")
                } else {
                    sectname.to_string()
                };
                out.push(Section { name, data: &buf[lo..hi] });
            }
        }
        o += cmdsize;
    }
    out
}

/// Read a NUL-or-end-terminated string from a fixed-size byte field
/// (Mach-O `sectname[16]` / `segname[16]`).
fn cstr_fixed(b: &[u8]) -> &str {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    std::str::from_utf8(&b[..end]).unwrap_or("<invalid-utf8>")
}

// ── DWARF walk ──────────────────────────────────────────────────────────────

type Slice<'a> = EndianSlice<'a, LittleEndian>;

/// A source-level function as seen through DWARF. A `DW_TAG_subprogram`
/// becomes an `Entry` at the top level; every `DW_TAG_inlined_subroutine`
/// descendant becomes a nested `Entry` under it.
#[derive(Default)]
struct Entry {
    name: String,
    /// [lo, hi) byte ranges in `.text` this call covers. DWARF supports
    /// non-contiguous ranges (function split across cold + hot sections);
    /// we handle both.
    ranges: Vec<(u64, u64)>,
    /// Inlined callees nested inside this call's ranges. Their ranges
    /// are strictly contained in ours (DWARF invariant).
    children: Vec<Entry>,
}

impl Entry {
    fn total_bytes(&self) -> u64 {
        self.ranges.iter().map(|(lo, hi)| hi - lo).sum()
    }
    fn children_bytes(&self) -> u64 {
        self.children.iter().map(|c| c.total_bytes()).sum()
    }
    /// Bytes attributed directly to this function (not to any nested
    /// inlined callee).
    fn self_bytes(&self) -> u64 { self.total_bytes().saturating_sub(self.children_bytes()) }
}

/// Mapping from DW_AT_abstract_origin target DIE offset → demangled
/// qualified name. Built in a first pass so the inline walk can
/// resolve `ltop::run::{closure#0}` (a full path) rather than the bare
/// `{closure#0}` a single `DW_AT_name` would give.
type NameMap = HashMap<DebugInfoOffset, String>;

fn load_dwarf<'a>(sections: &'a [Section<'a>]) -> Result<gimli::Dwarf<Slice<'a>>, String> {
    gimli::Dwarf::load(|id| -> Result<Slice<'a>, gimli::Error> {
        let name = id.name();
        let data = sections.iter().find(|s| s.name == name).map(|s| s.data).unwrap_or(&[]);
        Ok(EndianSlice::new(data, LittleEndian))
    })
    .map_err(|e| format!("load dwarf: {e}"))
}

/// Walk every unit collecting names for DIEs that are candidate
/// `DW_AT_abstract_origin` targets — that is, `DW_TAG_subprogram` DIEs.
/// For each, prefer the demangled `DW_AT_linkage_name` (rustc-demangle
/// expands `_RN...` / `_ZN...` to fully-qualified paths like
/// `ltop::run::{closure#0}`). Fall back to `DW_AT_name` for the
/// handful of subprograms without a linkage name (usually LTO-generated
/// stubs). No ad-hoc namespace-path tracking: abstract_origin targets
/// always carry a linkage name when they come from Rust code, and
/// a bare short name is a reasonable fallback elsewhere.
fn build_name_map<'a>(dwarf: &gimli::Dwarf<Slice<'a>>) -> Result<NameMap, gimli::Error> {
    let mut out = NameMap::new();
    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit_base = match header.offset() {
            UnitSectionOffset::DebugInfoOffset(o) => o.0,
            _ => continue,
        };
        let unit = dwarf.unit(header)?;
        let unit_ref = unit.unit_ref(dwarf);

        let mut entries = unit_ref.entries();
        while let Some((_, entry)) = entries.next_dfs()? {
            if entry.tag() != gimli::DW_TAG_subprogram { continue; }

            let demangled = entry
                .attr_value(gimli::DW_AT_linkage_name)
                .ok()
                .flatten()
                .and_then(|v| unit_ref.attr_string(v).ok())
                .and_then(|r| r.to_string().ok().map(|s| s.to_owned()))
                .map(|m| rustc_demangle::demangle(&m).to_string());

            let name = demangled.or_else(|| entry
                .attr_value(gimli::DW_AT_name)
                .ok()
                .flatten()
                .and_then(|v| unit_ref.attr_string(v).ok())
                .and_then(|r| r.to_string().ok().map(|s| s.to_owned())));

            if let Some(n) = name {
                out.insert(DebugInfoOffset(unit_base + entry.offset().0), n);
            }
        }
    }
    Ok(out)
}

fn dw_addr<'a>(unit: &gimli::UnitRef<Slice<'a>>, entry: &gimli::DebuggingInformationEntry<Slice<'a>>, at: gimli::DwAt) -> Option<u64> {
    let v = entry.attr_value(at).ok()??;
    match v {
        AttributeValue::Addr(a) => Some(a),
        AttributeValue::DebugAddrIndex(i) => unit.address(i).ok(),
        _ => None,
    }
}

/// Extract the address ranges of a subprogram or inlined_subroutine DIE.
/// DWARF encodes ranges two ways:
///   - `low_pc` + `high_pc` (either absolute address or offset from
///     low_pc). Single contiguous range.
///   - `DW_AT_ranges` pointing into `.debug_ranges` or
///     `.debug_rnglists`. Multi-range (for split functions).
///
/// `low_pc == 0` is the DWARF tombstone for a function whose
/// out-of-line body the linker stripped (e.g. a `compiler_builtins`
/// math helper that LTO inlined everywhere and then dead-coded the
/// standalone copy). Compilers leave the DIE behind with the original
/// `(low_pc=0, high_pc=N)` range so DWARF stays self-consistent, but
/// the bytes don't actually live at address 0 — they don't live
/// anywhere. Treat as no range. Without this filter, those phantom
/// ranges show up as huge attributions in the top-30 list (e.g.
/// `fmaf128` 1693 B, `cbrt` 1624 B) even though `nm` confirms no
/// such symbols exist in the binary.
fn entry_ranges<'a>(
    unit: &gimli::UnitRef<Slice<'a>>,
    entry: &gimli::DebuggingInformationEntry<Slice<'a>>,
) -> Vec<(u64, u64)> {
    if let Ok(Some(_)) = entry.attr_value(gimli::DW_AT_ranges) {
        if let Ok(mut it) = unit.die_ranges(entry) {
            let mut out = Vec::new();
            while let Ok(Some(r)) = it.next() {
                // Same tombstone convention applies to range-list
                // entries: a `(0, N)` range means "stripped" once
                // we're past the unit-base offset, since the unit
                // header's own low_pc=0 base is excluded by DWARF
                // from the rangelist enumeration.
                if r.begin > 0 && r.begin < r.end { out.push((r.begin, r.end)); }
            }
            return out;
        }
    }
    if let Some(low) = dw_addr(unit, entry, gimli::DW_AT_low_pc) {
        // See the docstring: low_pc=0 ⇒ stripped, no real range.
        if low == 0 { return Vec::new(); }
        let high = match entry.attr_value(gimli::DW_AT_high_pc).ok().flatten() {
            Some(AttributeValue::Addr(a)) => a,
            Some(AttributeValue::Udata(u)) => low + u,
            Some(AttributeValue::Data1(u)) => low + u as u64,
            Some(AttributeValue::Data2(u)) => low + u as u64,
            Some(AttributeValue::Data4(u)) => low + u as u64,
            Some(AttributeValue::Data8(u)) => low + u,
            _ => low,
        };
        if high > low { return vec![(low, high)]; }
    }
    Vec::new()
}

/// Resolve `DW_AT_abstract_origin` to a qualified function name via
/// `name_map`. The DW_AT_abstract_origin attribute carries a reference
/// to the concrete-function DIE that the inlined call is standing in
/// for; we followed that reference in `build_name_map` to pre-compute
/// the demangled name.
fn resolve_origin_name<'a>(
    unit: &gimli::UnitRef<Slice<'a>>,
    entry: &gimli::DebuggingInformationEntry<Slice<'a>>,
    name_map: &NameMap,
) -> Option<String> {
    let v = entry.attr_value(gimli::DW_AT_abstract_origin).ok()??;
    let unit_base = match unit.header.offset() {
        UnitSectionOffset::DebugInfoOffset(o) => o.0,
        _ => return None,
    };
    let off = match v {
        AttributeValue::UnitRef(r) => DebugInfoOffset(unit_base + r.0),
        AttributeValue::DebugInfoRef(r) => r,
        _ => return None,
    };
    name_map.get(&off).cloned()
}

/// Walk one compilation unit, collecting top-level subprograms and
/// their nested inlined_subroutine trees. A DFS over the DIE hierarchy
/// maintains a parent stack of the currently-open subprogram /
/// inlined_subroutine; tag entries get pushed onto the stack and are
/// attached to whatever is their direct DIE ancestor on the stack.
fn walk_unit<'a>(
    unit: &gimli::UnitRef<Slice<'a>>,
    name_map: &NameMap,
    out: &mut Vec<Entry>,
) -> Result<(), gimli::Error> {
    let unit_base = match unit.header.offset() {
        UnitSectionOffset::DebugInfoOffset(o) => o.0,
        _ => return Ok(()),
    };

    // Stack of (entry, dwarf_depth_at_which_it_opened).
    let mut stack: Vec<(Entry, isize)> = Vec::new();
    let mut entries = unit.entries();
    let mut depth: isize = 0;
    while let Some((delta, entry)) = entries.next_dfs()? {
        depth += delta as isize;
        // Close any open frames whose depth is now behind us.
        while let Some(&(_, od)) = stack.last() {
            if od < depth { break; }
            let (done, _) = stack.pop().unwrap();
            match stack.last_mut() {
                Some((parent, _)) => parent.children.push(done),
                None => out.push(done),
            }
        }

        let tag = entry.tag();
        if tag == gimli::DW_TAG_subprogram {
            // Only take subprograms that have code (not declarations).
            let ranges = entry_ranges(unit, entry);
            if !ranges.is_empty() {
                let name = name_map
                    .get(&DebugInfoOffset(unit_base + entry.offset().0))
                    .cloned()
                    .unwrap_or_else(|| "<unnamed-subprogram>".into());
                stack.push((Entry { name, ranges, children: Vec::new() }, depth));
            }
        } else if tag == gimli::DW_TAG_inlined_subroutine {
            let ranges = entry_ranges(unit, entry);
            if !ranges.is_empty() && !stack.is_empty() {
                let name = resolve_origin_name(unit, entry, name_map)
                    .unwrap_or_else(|| "<unresolved-inline>".into());
                stack.push((Entry { name, ranges, children: Vec::new() }, depth));
            }
        }
    }

    // Close anything still open.
    while let Some((done, _)) = stack.pop() {
        match stack.last_mut() {
            Some((parent, _)) => parent.children.push(done),
            None => out.push(done),
        }
    }
    Ok(())
}

// ── output ──────────────────────────────────────────────────────────────────

/// Bounded printer: caps depth, per-level child count, total lines.
/// Any write error (including our own budget-exceeded signal) returns
/// `Err` and unwinds the recursion cleanly.
struct Printer<W: Write> {
    w: W,
    lines_left: usize,
    truncated: bool,
}

impl<W: Write> Printer<W> {
    fn line(&mut self, args: std::fmt::Arguments<'_>) -> io::Result<()> {
        if self.lines_left == 0 {
            self.truncated = true;
            return Err(io::Error::new(io::ErrorKind::Other, "line budget exhausted"));
        }
        self.lines_left -= 1;
        self.w.write_fmt(args)?;
        self.w.write_all(b"\n")
    }

    fn tree(&mut self, entry: &Entry, indent: usize) -> io::Result<()> {
        let total = entry.total_bytes();
        if indent == 0 {
            self.line(format_args!("{} — total {} B, self {} B",
                entry.name, total, entry.self_bytes()))?;
        } else {
            self.line(format_args!("{:width$}{} — {} B",
                "", entry.name, total, width = indent * 2))?;
        }
        if indent + 1 > MAX_DEPTH { return Ok(()); }

        // Sort by total bytes, descending, then drop tiny leaves so
        // noise doesn't dominate the tail.
        let mut kids: Vec<&Entry> = entry.children.iter()
            .filter(|c| c.total_bytes() >= MIN_BYTES_AT_LEAF)
            .collect();
        kids.sort_by(|a, b| b.total_bytes().cmp(&a.total_bytes()));

        let cap = MAX_CHILDREN_PER_LEVEL;
        let shown = kids.len().min(cap);
        for k in &kids[..shown] {
            self.tree(k, indent + 1)?;
        }
        if kids.len() > cap {
            let rest: u64 = kids[cap..].iter().map(|c| c.total_bytes()).sum();
            self.line(format_args!("{:width$}… {} more, {} B total",
                "", kids.len() - cap, rest, width = (indent + 1) * 2))?;
        }
        Ok(())
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <binary> [fn-fragment]", args[0]);
        return ExitCode::from(2);
    }
    let bytes = match fs::read(&args[1]) {
        Ok(b) => b,
        Err(e) => { eprintln!("read {}: {e}", args[1]); return ExitCode::from(1); }
    };
    let sections = match parse_sections(&bytes) {
        Ok(s) => s,
        Err(e) => { eprintln!("{e}"); return ExitCode::from(1); }
    };
    let dwarf = match load_dwarf(&sections) {
        Ok(d) => d,
        Err(e) => { eprintln!("{e}"); return ExitCode::from(1); }
    };
    let name_map = match build_name_map(&dwarf) {
        Ok(m) => m,
        Err(e) => { eprintln!("build name map: {e}"); return ExitCode::from(1); }
    };

    let mut top: Vec<Entry> = Vec::new();
    let mut units = dwarf.units();
    loop {
        let header = match units.next() {
            Ok(Some(h)) => h,
            Ok(None) => break,
            Err(e) => { eprintln!("units: {e}"); return ExitCode::from(1); }
        };
        let unit = match dwarf.unit(header) {
            Ok(u) => u,
            Err(e) => { eprintln!("unit: {e}"); return ExitCode::from(1); }
        };
        if let Err(e) = walk_unit(&unit.unit_ref(&dwarf), &name_map, &mut top) {
            eprintln!("walk: {e}");
            return ExitCode::from(1);
        }
    }

    let stdout = io::stdout();
    let mut printer = Printer {
        w: BufWriter::new(stdout.lock()),
        lines_left: MAX_TOTAL_LINES,
        truncated: false,
    };

    match args.get(2) {
        Some(needle) => {
            let needle = needle.as_str();
            let mut hits: Vec<&Entry> = top.iter().filter(|e| e.name.contains(needle)).collect();
            hits.sort_by(|a, b| b.total_bytes().cmp(&a.total_bytes()));
            if hits.is_empty() {
                eprintln!("no subprogram matches `{needle}`");
                return ExitCode::from(1);
            }
            let total_hits = hits.len();
            let shown = total_hits.min(MAX_NEEDLE_HITS);
            for e in &hits[..shown] {
                if printer.tree(e, 0).is_err() { break; }
                let _ = printer.w.write_all(b"\n");
            }
            if total_hits > shown {
                let _ = writeln!(printer.w,
                    "… {} more subprograms matched (showing top {} by bytes)",
                    total_hits - shown, shown);
            }
        }
        None => {
            // No filter: top 30 subprograms by total size.
            top.sort_by(|a, b| b.total_bytes().cmp(&a.total_bytes()));
            for e in top.iter().take(30) {
                if writeln!(printer.w, "{:>8} B  self {:>6} B  {}",
                    e.total_bytes(), e.self_bytes(), e.name).is_err() { break; }
            }
        }
    }

    let truncated = printer.truncated;
    // Flush before printing to stderr so output ordering is sane.
    drop(printer);
    if truncated {
        eprintln!("(output truncated at {} lines — tighten your filter)", MAX_TOTAL_LINES);
    }

    ExitCode::SUCCESS
}
