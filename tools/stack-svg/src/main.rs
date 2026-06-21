//! stack-svg: render a static stack-usage analysis as a space-time diagram.
//!
//! Input: an ELF64 binary built with `-Zemit-stack-sizes`, debug info
//! (for DWARF inline decomposition), and symbols preserved. In this
//! workspace that's `cargo stack` → the `min-stack` profile at
//! `target/x86_64-unknown-linux-musl/min-stack/ltop`.
//!
//! Output: SVG to argv[2] (or stdout). Layout mirrors arena-svg — X is
//! a DFS of the call graph (each function's width clamped to [16, 32 KB]
//! like arena events), Y is stack byte offset from the entry point's
//! base. Each rect is one function's frame along that walk.
//!
//! Inlined decomposition: `opt-level=z` + LTO inlines aggressively, so
//! one real function's `.stack_sizes` entry is often the union of many
//! inlined callees' frames (e.g. the 12 KB `ltop::run::{closure#0}` is
//! mostly `collect_procs`'s 4 KB `stat_raw` buffer plus a dozen other
//! inlined locals). We parse `.debug_info` for `DW_TAG_inlined_subroutine`
//! entries and their local variables' `DW_AT_location` stack offsets,
//! and draw each inlined call as a sub-rect painted at its byte range
//! within the containing real frame. Inlined rects use a dashed outline
//! to distinguish them from real-call rects.
//!
//! Recursion: self-loops break on the second visit to a function on the
//! current path, drawn once with a "↻" marker. `--bound FN=N` unrolls
//! FN up to N times before the loop marker.
//!
//! Dependencies: `gimli` for DWARF parsing. `objdump` (binutils) is
//! required for the call-graph extraction — we shell out to it with
//! `-d -C`, which demangles Rust v0 symbols inline.
//!
//! Mach-O (macOS) support: `-Zemit-stack-sizes` is silently a no-op on
//! Mach-O at the time of writing — rustc/LLVM doesn't emit a
//! `.stack_sizes` section. We recover per-function frame sizes by
//! scanning objdump's disassembly for the canonical aarch64 prologue
//! (`sub sp, sp, #imm` or `stp x29, x30, [sp, #-N]!`). DWARF inlined-
//! subroutine decomposition is currently ELF-only because Mach-O
//! debug info often lives in a separate `.dSYM` bundle; the Mac SVG
//! shows real-frame rectangles only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, ExitCode};

use gimli::{AttributeValue, DebugInfoOffset, DwAt, EndianSlice, LittleEndian, Reader, UnitSectionOffset};

// ── Format detection ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Elf,
    MachO,
}

fn detect_format(bytes: &[u8]) -> Result<Format, String> {
    if bytes.len() < 4 {
        return Err("file too small to identify".into());
    }
    if &bytes[..4] == b"\x7fELF" { return Ok(Format::Elf); }
    // Mach-O 64-bit little-endian: MH_MAGIC_64 = 0xfeedfacf, on disk as
    // bytes [cf fa ed fe]. We don't claim to support fat/universal
    // binaries (FAT_MAGIC = 0xcafebabe) or 32-bit Mach-O — the ltop
    // mac targets are aarch64-apple-darwin / x86_64-apple-darwin, both
    // 64-bit thin Mach-O.
    if bytes[..4] == [0xcf, 0xfa, 0xed, 0xfe] { return Ok(Format::MachO); }
    Err("not a recognised binary format (expected ELF64 or Mach-O 64-bit)".into())
}

fn default_binary_path() -> &'static str {
    if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "target/aarch64-apple-darwin/min-stack/ltop"
        } else {
            "target/x86_64-apple-darwin/min-stack/ltop"
        }
    } else if cfg!(target_arch = "aarch64") {
        "target/static-linux-aarch64/min-stack/ltop"
    } else {
        "target/static-linux-x86_64/min-stack/ltop"
    }
}

// ── ELF64 parsing ────────────────────────────────────────────────────────────

// Just enough to find `.symtab`, `.strtab`, and `.stack_sizes`.

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

struct Shdr<'a> {
    name: &'a str,
    ty: u32,
    link: u32,
    data: &'a [u8],
    entsize: u64,
}

fn parse_sections<'a>(elf: &'a [u8]) -> Result<Vec<Shdr<'a>>, String> {
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" {
        return Err("not an ELF file".into());
    }
    if elf[4] != 2 || elf[5] != 1 {
        return Err("not ELF64 little-endian".into());
    }
    let e_shoff = u64le(elf, 0x28) as usize;
    let e_shentsize = u16le(elf, 0x3a) as usize;
    let e_shnum = u16le(elf, 0x3c) as usize;
    let e_shstrndx = u16le(elf, 0x3e) as usize;
    if e_shentsize < 64 {
        return Err(format!("unexpected section header size {e_shentsize}"));
    }

    let read_shdr = |i: usize| -> (&[u8], u64, u64, u64, u32, u32, u64) {
        let o = e_shoff + i * e_shentsize;
        let sh_name = u32le(elf, o) as u64;
        let sh_type = u32le(elf, o + 4);
        let sh_offset = u64le(elf, o + 0x18) as usize;
        let sh_size = u64le(elf, o + 0x20) as usize;
        let sh_link = u32le(elf, o + 0x28);
        let sh_entsize = u64le(elf, o + 0x38);
        // SHT_NOBITS sections (type 8, e.g. .bss) have sh_size > 0 but no
        // backing bytes in the file. Clamp to the file's actual length so
        // we hand back an empty slice instead of panicking.
        let lo = sh_offset.min(elf.len());
        let hi = sh_offset.saturating_add(sh_size).min(elf.len());
        let data: &[u8] = if sh_type == 8 { &[] } else { &elf[lo..hi] };
        (data, sh_name, sh_offset as u64, sh_size as u64, sh_type, sh_link, sh_entsize)
    };

    // Section-name string table (sh_type=SHT_STRTAB at e_shstrndx).
    let (shstrtab, _, _, _, _, _, _) = read_shdr(e_shstrndx);

    let mut out = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let (data, name_off, _, _, ty, link, entsize) = read_shdr(i);
        let name = cstr(shstrtab, name_off as usize);
        out.push(Shdr { name, ty, link, data, entsize });
    }
    Ok(out)
}

fn cstr(strtab: &[u8], off: usize) -> &str {
    if off >= strtab.len() {
        return "";
    }
    let end = strtab[off..].iter().position(|&b| b == 0).unwrap_or(strtab.len() - off);
    std::str::from_utf8(&strtab[off..off + end]).unwrap_or("<invalid-utf8>")
}

// ── Symbols ──────────────────────────────────────────────────────────────────

struct Sym {
    name: String,
    addr: u64,
    #[allow(dead_code)]
    size: u64,
}

fn parse_symbols(sections: &[Shdr<'_>]) -> Vec<Sym> {
    // SHT_SYMTAB = 2; sh_link points at the associated string table.
    let Some((idx, symtab)) = sections.iter().enumerate().find(|(_, s)| s.ty == 2) else {
        return Vec::new();
    };
    let _ = idx;
    let strtab = &sections[symtab.link as usize].data;
    let ent = symtab.entsize as usize;
    if ent == 0 {
        return Vec::new();
    }
    let n = symtab.data.len() / ent;
    let mut syms = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * ent;
        let st_name = u32le(symtab.data, o) as usize;
        let st_info = symtab.data[o + 4];
        let st_value = u64le(symtab.data, o + 8);
        let st_size = u64le(symtab.data, o + 16);
        let ty = st_info & 0xf; // STT_FUNC = 2
        if ty == 2 && st_value != 0 {
            syms.push(Sym {
                name: cstr(strtab, st_name).to_string(),
                addr: st_value,
                size: st_size,
            });
        }
    }
    syms
}

// ── DWARF inlined-subroutine parsing ────────────────────────────────────────
//
// We use gimli to walk `.debug_info`. For each DW_TAG_subprogram we
// record the tree of DW_TAG_inlined_subroutine descendants; for each
// inlined call we extract an approximate frame range by scanning the
// locals (`DW_TAG_variable` / `DW_TAG_formal_parameter`) for
// `DW_OP_fbreg(offset)` locations and reading `DW_AT_byte_size` from
// their type DIEs. frame_lo = min(offset), frame_hi = max(offset + size).
//
// The DWARF is DWARF 4 with rustc defaults; we handle the contiguous
// low_pc+high_pc form. Non-contiguous ranges (via `DW_AT_ranges`) are
// uncommon for inlined subroutines in this codebase and skipped.

type Slice<'a> = EndianSlice<'a, LittleEndian>;

#[derive(Clone)]
struct InlinedCall {
    name: String,
    frame_lo: i64,
    frame_hi: i64, // exclusive
    children: Vec<InlinedCall>,
}

struct InlineMap {
    // Real-function entry address → tree of inlined calls inside it.
    by_addr: HashMap<u64, Vec<InlinedCall>>,
}

/// Qualified name of every DIE that has `DW_AT_name`, keyed by absolute
/// `.debug_info` offset. Built in a first pass by walking every unit's
/// namespace / subprogram / type hierarchy; used in the second pass to
/// resolve `DW_AT_abstract_origin` references across compile units to
/// names like `ltop::run::{closure#0}` rather than the bare short
/// `{closure#0}` that `DW_AT_name` alone gives.
type NameMap = HashMap<DebugInfoOffset, String>;

fn parse_inlines<'a>(sections: &'a [Shdr<'a>]) -> Result<InlineMap, String> {
    let load = |id: gimli::SectionId| -> Result<Slice<'a>, gimli::Error> {
        let name = id.name();
        let data = sections
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.data)
            .unwrap_or(&[]);
        Ok(EndianSlice::new(data, LittleEndian))
    };
    let dwarf = gimli::Dwarf::load(load).map_err(|e| format!("load dwarf: {e}"))?;

    let name_map = build_name_map(&dwarf).map_err(|e| format!("name map: {e}"))?;
    let mut by_addr: HashMap<u64, Vec<InlinedCall>> = HashMap::new();

    let mut units = dwarf.units();
    while let Some(header) = units.next().map_err(|e| format!("units: {e}"))? {
        let unit = dwarf.unit(header).map_err(|e| format!("unit: {e}"))?;
        let unit_ref = unit.unit_ref(&dwarf);
        walk_unit(&unit_ref, &name_map, &mut by_addr)?;
    }
    Ok(InlineMap { by_addr })
}

fn build_name_map<'a>(dwarf: &gimli::Dwarf<Slice<'a>>) -> Result<NameMap, gimli::Error> {
    let mut out = NameMap::new();
    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit_base = match header.offset() {
            UnitSectionOffset::DebugInfoOffset(o) => o.0,
            _ => continue, // skip .debug_types
        };
        let unit = dwarf.unit(header)?;
        let unit_ref = unit.unit_ref(dwarf);
        let mut tree = unit_ref.entries_tree(None)?;
        let root = tree.root()?;
        let mut prefix: Vec<String> = Vec::new();
        walk_names(&unit_ref, root, &mut prefix, unit_base, &mut out)?;
    }
    Ok(out)
}

fn walk_names<'a>(
    unit: &gimli::UnitRef<'_, Slice<'a>>,
    node: gimli::EntriesTreeNode<'_, '_, '_, Slice<'a>>,
    prefix: &mut Vec<String>,
    unit_base: usize,
    out: &mut NameMap,
) -> Result<(), gimli::Error> {
    let entry_offset = node.entry().offset();
    let tag = node.entry().tag();
    let name = read_attr_string(unit, node.entry(), gimli::DW_AT_name);
    if let Some(ref n) = name {
        let full = if prefix.is_empty() {
            n.clone()
        } else {
            format!("{}::{n}", prefix.join("::"))
        };
        let abs = DebugInfoOffset(unit_base + entry_offset.0);
        out.insert(abs, full);
    }
    // Push this DIE into the qualified-name prefix when descending into
    // children, for DIE tags that form a namespace-like scope. Rust's
    // DWARF uses DW_TAG_namespace for modules and DW_TAG_subprogram for
    // the enclosing function of a closure or local type.
    let pushed = matches!(
        tag,
        gimli::DW_TAG_namespace
            | gimli::DW_TAG_subprogram
            | gimli::DW_TAG_structure_type
            | gimli::DW_TAG_class_type
            | gimli::DW_TAG_union_type
            | gimli::DW_TAG_enumeration_type
    ) && name.is_some();
    if pushed {
        prefix.push(name.unwrap());
    }
    let mut children = node.children();
    while let Some(child) = children.next()? {
        walk_names(unit, child, prefix, unit_base, out)?;
    }
    if pushed {
        prefix.pop();
    }
    Ok(())
}

fn walk_unit<'a>(
    unit: &gimli::UnitRef<'_, Slice<'a>>,
    name_map: &NameMap,
    out: &mut HashMap<u64, Vec<InlinedCall>>,
) -> Result<(), String> {
    let mut tree = unit
        .entries_tree(None)
        .map_err(|e| format!("entries_tree: {e}"))?;
    let root = tree.root().map_err(|e| format!("tree root: {e}"))?;
    let mut children = root.children();
    while let Some(node) = children.next().map_err(|e| format!("{e}"))? {
        process_top_level(unit, name_map, node, out)?;
    }
    Ok(())
}

fn process_top_level<'a, 'b>(
    unit: &gimli::UnitRef<'b, Slice<'a>>,
    name_map: &NameMap,
    node: gimli::EntriesTreeNode<'_, '_, '_, Slice<'a>>,
    out: &mut HashMap<u64, Vec<InlinedCall>>,
) -> Result<(), String> {
    let tag = node.entry().tag();
    if tag == gimli::DW_TAG_subprogram {
        let low_pc = get_low_pc(node.entry());
        // Only record subprograms with a concrete entry address — those
        // are the ones that line up with `.stack_sizes` entries.
        if let Some(addr) = low_pc {
            let calls = collect_inlined(unit, name_map, node)?;
            if !calls.is_empty() {
                out.insert(addr, calls);
            }
        }
        return Ok(());
    }
    // Some subprograms sit under namespace / compile-unit DIEs — descend.
    let mut children = node.children();
    while let Some(child) = children.next().map_err(|e| format!("{e}"))? {
        process_top_level(unit, name_map, child, out)?;
    }
    Ok(())
}

fn collect_inlined<'a, 'b>(
    unit: &gimli::UnitRef<'b, Slice<'a>>,
    name_map: &NameMap,
    node: gimli::EntriesTreeNode<'_, '_, '_, Slice<'a>>,
) -> Result<Vec<InlinedCall>, String> {
    let mut out = Vec::new();
    let mut children = node.children();
    while let Some(child) = children.next().map_err(|e| format!("{e}"))? {
        let tag = child.entry().tag();
        if tag == gimli::DW_TAG_inlined_subroutine {
            if let Some(ic) = build_inlined(unit, name_map, child)? {
                out.push(ic);
            }
        } else if tag == gimli::DW_TAG_lexical_block {
            let mut sub = collect_inlined(unit, name_map, child)?;
            out.append(&mut sub);
        }
    }
    Ok(out)
}

fn build_inlined<'a, 'b>(
    unit: &gimli::UnitRef<'b, Slice<'a>>,
    name_map: &NameMap,
    node: gimli::EntriesTreeNode<'_, '_, '_, Slice<'a>>,
) -> Result<Option<InlinedCall>, String> {
    let entry = node.entry().clone();
    let origin = entry
        .attr_value(gimli::DW_AT_abstract_origin)
        .map_err(|e| format!("{e}"))?;
    let Some(name) = origin.and_then(|v| resolve_origin_name(unit, name_map, v)) else {
        return Ok(None);
    };

    // frame range: gathered from locals' DW_AT_location fbreg values,
    // plus recursively from nested inlined calls' ranges (they also
    // sit in the parent frame).
    let mut frame_lo: Option<i64> = None;
    let mut frame_hi: Option<i64> = None;
    let mut nested: Vec<InlinedCall> = Vec::new();

    let mut children = node.children();
    while let Some(child) = children.next().map_err(|e| format!("{e}"))? {
        let tag = child.entry().tag();
        match tag {
            gimli::DW_TAG_variable | gimli::DW_TAG_formal_parameter => {
                if let Some((lo, hi)) = local_frame_range(unit, child.entry())
                    .map_err(|e| format!("{e}"))?
                {
                    frame_lo = Some(frame_lo.map_or(lo, |x| x.min(lo)));
                    frame_hi = Some(frame_hi.map_or(hi, |x| x.max(hi)));
                }
            }
            gimli::DW_TAG_inlined_subroutine => {
                // Child's locals belong to the child, not us. Previously
                // we unioned ic's range into frame_lo/hi; that inflated
                // every outer call to include every nested callee's
                // slots, making ranges meaningless. Collect the child
                // for rendering but leave our own range alone.
                if let Some(ic) = build_inlined(unit, name_map, child)? {
                    nested.push(ic);
                }
            }
            gimli::DW_TAG_lexical_block => {
                // Lexical blocks are a scoping construct within this
                // same inlined body; their locals *are* ours. Recurse
                // and absorb the block's own-locals range (but again
                // not the nested inlined calls' ranges).
                let sub = build_inlined_block_range(unit, name_map, child)?;
                if let Some((lo, hi, mut inner_inlines)) = sub {
                    if let Some(l) = lo {
                        frame_lo = Some(frame_lo.map_or(l, |x| x.min(l)));
                    }
                    if let Some(h) = hi {
                        frame_hi = Some(frame_hi.map_or(h, |x| x.max(h)));
                    }
                    nested.append(&mut inner_inlines);
                }
            }
            _ => {}
        }
    }

    let (lo, hi) = match (frame_lo, frame_hi) {
        (Some(l), Some(h)) if h > l => (l, h),
        // No locals found — still emit a rect so the call is visible,
        // just give it a tiny height.
        _ => (0, 1),
    };
    Ok(Some(InlinedCall {
        name,
        frame_lo: lo,
        frame_hi: hi,
        children: nested,
    }))
}

// Walk a lexical_block subtree. Returns (block_own_lo, block_own_hi,
// inlined-calls-encountered). The own-range reflects only locals
// directly in this block and its nested lexical_block descendants —
// inlined-subroutine descendants contribute only to `inlined`, never
// to the block's range (their locals belong to them).
fn build_inlined_block_range<'a, 'b>(
    unit: &gimli::UnitRef<'b, Slice<'a>>,
    name_map: &NameMap,
    node: gimli::EntriesTreeNode<'_, '_, '_, Slice<'a>>,
) -> Result<Option<(Option<i64>, Option<i64>, Vec<InlinedCall>)>, String> {
    let mut frame_lo: Option<i64> = None;
    let mut frame_hi: Option<i64> = None;
    let mut inlined = Vec::new();
    let mut saw_any = false;
    let mut children = node.children();
    while let Some(child) = children.next().map_err(|e| format!("{e}"))? {
        saw_any = true;
        let tag = child.entry().tag();
        match tag {
            gimli::DW_TAG_variable | gimli::DW_TAG_formal_parameter => {
                if let Some((lo, hi)) = local_frame_range(unit, child.entry())
                    .map_err(|e| format!("{e}"))?
                {
                    frame_lo = Some(frame_lo.map_or(lo, |x| x.min(lo)));
                    frame_hi = Some(frame_hi.map_or(hi, |x| x.max(hi)));
                }
            }
            gimli::DW_TAG_inlined_subroutine => {
                if let Some(ic) = build_inlined(unit, name_map, child)? {
                    inlined.push(ic);
                }
            }
            gimli::DW_TAG_lexical_block => {
                if let Some((lo, hi, mut inner)) = build_inlined_block_range(unit, name_map, child)? {
                    if let Some(l) = lo {
                        frame_lo = Some(frame_lo.map_or(l, |x| x.min(l)));
                    }
                    if let Some(h) = hi {
                        frame_hi = Some(frame_hi.map_or(h, |x| x.max(h)));
                    }
                    inlined.append(&mut inner);
                }
            }
            _ => {}
        }
    }
    if saw_any {
        Ok(Some((frame_lo, frame_hi, inlined)))
    } else {
        Ok(None)
    }
}

/// DW_AT_low_pc for a DIE — only the plain `Addr` form is handled (the
/// default for rustc's contiguous subprograms and inlined subroutines).
fn get_low_pc<R: Reader>(entry: &gimli::DebuggingInformationEntry<'_, '_, R>) -> Option<u64> {
    match entry.attr_value(gimli::DW_AT_low_pc).ok()?? {
        AttributeValue::Addr(a) => Some(a),
        _ => None,
    }
}

/// Resolve `DW_AT_abstract_origin` to the target DIE's fully-qualified
/// name by consulting the NameMap built in a first pass. Falls back to
/// the bare `DW_AT_name` if the DIE isn't in the map.
fn resolve_origin_name<'a, 'b>(
    unit: &gimli::UnitRef<'b, Slice<'a>>,
    name_map: &NameMap,
    val: AttributeValue<Slice<'a>>,
) -> Option<String> {
    let abs_off = match val {
        AttributeValue::UnitRef(o) => {
            let unit_base = match unit.header.offset() {
                UnitSectionOffset::DebugInfoOffset(b) => b.0,
                _ => return None,
            };
            DebugInfoOffset(unit_base + o.0)
        }
        AttributeValue::DebugInfoRef(o) => o,
        _ => return None,
    };
    if let Some(n) = name_map.get(&abs_off) {
        return Some(n.clone());
    }
    // Fallback: read DW_AT_name directly (requires resolving to the
    // target unit; worth the extra work to avoid losing the name).
    resolve_bare_name(unit, val)
}

fn resolve_bare_name<'a, 'b>(
    unit: &gimli::UnitRef<'b, Slice<'a>>,
    val: AttributeValue<Slice<'a>>,
) -> Option<String> {
    match val {
        AttributeValue::UnitRef(o) => {
            let entry = unit.entry(o).ok()?;
            read_attr_string(unit, &entry, gimli::DW_AT_name)
        }
        AttributeValue::DebugInfoRef(off) => {
            let mut iter = unit.dwarf.units();
            while let Some(header) = iter.next().ok()? {
                if let UnitSectionOffset::DebugInfoOffset(h) = header.offset() {
                    let start = h.0;
                    let end = start + header.length_including_self();
                    if off.0 >= start && off.0 < end {
                        let u2 = unit.dwarf.unit(header).ok()?;
                        let u2r = u2.unit_ref(unit.dwarf);
                        let uo = off.to_unit_offset(&u2r.header)?;
                        let entry = u2r.entry(uo).ok()?;
                        return read_attr_string(&u2r, &entry, gimli::DW_AT_name);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn read_attr_string<R: Reader>(
    unit: &gimli::UnitRef<'_, R>,
    entry: &gimli::DebuggingInformationEntry<'_, '_, R>,
    attr: DwAt,
) -> Option<String> {
    let v = entry.attr_value(attr).ok().flatten()?;
    let s = unit.attr_string(v).ok()?;
    Some(s.to_string_lossy().ok()?.into_owned())
}

/// Extract the frame byte-range [fbreg_offset, fbreg_offset + size) for a
/// local variable or parameter. Returns None if the location isn't a
/// plain `DW_OP_fbreg(N)` — we skip register-resident locals and
/// location-list ranges for simplicity (still caught by the aggregate of
/// other locals if at least one is on-stack).
fn local_frame_range<'a, 'b, R: Reader<Offset = usize>>(
    unit: &gimli::UnitRef<'b, R>,
    entry: &gimli::DebuggingInformationEntry<'_, '_, R>,
) -> Result<Option<(i64, i64)>, gimli::Error> {
    let loc = match entry.attr_value(gimli::DW_AT_location)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let expr = match loc {
        AttributeValue::Exprloc(e) => e,
        // Location lists describe per-pc locations; typical for Rust
        // scoped locals. Resolving them requires the current pc, which
        // we don't track during a static walk. Skip for v1.
        _ => return Ok(None),
    };
    let mut ops = expr.operations(unit.encoding());
    let mut offset: Option<i64> = None;
    while let Some(op) = ops.next()? {
        if let gimli::Operation::FrameOffset { offset: o } = op {
            offset = Some(o);
            break;
        }
    }
    let offset = match offset {
        Some(o) => o,
        None => return Ok(None),
    };

    // Size via DW_AT_type → chase to a DIE with DW_AT_byte_size.
    let size = match entry.attr_value(gimli::DW_AT_type)? {
        Some(AttributeValue::UnitRef(type_off)) => byte_size_of(unit, type_off).unwrap_or(8),
        _ => 8,
    };
    Ok(Some((offset, offset + size as i64)))
}

/// Best-effort DW_AT_byte_size, chasing one layer of typedef/modifier
/// indirection. Returns None if the type DIE has no size (e.g. opaque
/// forward declaration); callers substitute a default.
fn byte_size_of<R: Reader<Offset = usize>>(unit: &gimli::UnitRef<'_, R>, off: gimli::UnitOffset) -> Option<u64> {
    let entry = unit.entry(off).ok()?;
    if let Ok(Some(v)) = entry.attr_value(gimli::DW_AT_byte_size) {
        if let Some(n) = v.udata_value() {
            return Some(n);
        }
    }
    match entry.tag() {
        gimli::DW_TAG_typedef
        | gimli::DW_TAG_const_type
        | gimli::DW_TAG_volatile_type
        | gimli::DW_TAG_restrict_type
        | gimli::DW_TAG_atomic_type => {
            if let Ok(Some(AttributeValue::UnitRef(next))) = entry.attr_value(gimli::DW_AT_type) {
                return byte_size_of(unit, next);
            }
        }
        gimli::DW_TAG_pointer_type | gimli::DW_TAG_reference_type | gimli::DW_TAG_rvalue_reference_type => {
            return Some(8);
        }
        gimli::DW_TAG_array_type => {
            // Arrays in Rust/C DWARF: DW_AT_type on the array DIE is the
            // *element* type; the array's length lives under a
            // DW_TAG_subrange_type child (DW_AT_count, or
            // DW_AT_upper_bound + 1). Element size × count gives us the
            // total. Handles the common case where the compiler didn't
            // emit DW_AT_byte_size on the array itself.
            let elem_size = match entry.attr_value(gimli::DW_AT_type).ok().flatten() {
                Some(AttributeValue::UnitRef(elem_off)) => byte_size_of(unit, elem_off)?,
                _ => return None,
            };
            let mut tree = unit.entries_tree(Some(off)).ok()?;
            let root = tree.root().ok()?;
            let mut kids = root.children();
            while let Ok(Some(k)) = kids.next() {
                if k.entry().tag() == gimli::DW_TAG_subrange_type {
                    let count = k
                        .entry()
                        .attr_value(gimli::DW_AT_count)
                        .ok()
                        .flatten()
                        .and_then(|v| v.udata_value())
                        .or_else(|| {
                            k.entry()
                                .attr_value(gimli::DW_AT_upper_bound)
                                .ok()
                                .flatten()
                                .and_then(|v| v.udata_value())
                                .map(|u| u + 1)
                        })?;
                    return Some(elem_size * count);
                }
            }
            return None;
        }
        _ => {}
    }
    None
}

// ── .stack_sizes ────────────────────────────────────────────────────────────

fn parse_stack_sizes(sections: &[Shdr<'_>]) -> HashMap<u64, u64> {
    let mut out = HashMap::new();
    let Some(s) = sections.iter().find(|s| s.name == ".stack_sizes") else {
        return out;
    };
    let data = s.data;
    let mut i = 0;
    while i + 8 <= data.len() {
        let addr = u64le(data, i);
        i += 8;
        let mut size = 0u64;
        let mut shift = 0u32;
        loop {
            if i >= data.len() {
                return out;
            }
            let b = data[i];
            i += 1;
            size |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return out;
            }
        }
        out.insert(addr, size);
    }
    out
}

// ── Call graph from objdump ─────────────────────────────────────────────────

/// Run `objdump -d -C -j .text BIN`, parse for function headers and direct
/// `call` / `jmp` instructions. We track the current function by scanning
/// for lines matching `^[0-9a-f]+ <NAME>:` and attribute every subsequent
/// `call ADDR <TARGET>` to it.
///
/// The demangled TARGET may carry a `+0xN` suffix when the call lands
/// inside a function rather than at its entry. We strip that — a call
/// into the middle of a function is still a call to that function.
fn objdump_calls(binary: &str) -> Result<HashMap<String, Vec<String>>, String> {
    // Disassemble the whole text. We used to pass `-j .text` to skip
    // PLT/stub sections, but Mach-O names that section `__TEXT,__text`
    // and the same `-j .text` matches nothing. The parser's
    // function-header detection (`<symbol>:`) ignores stub trampolines
    // anyway — they have no proper headers in objdump output.
    let out = Command::new("objdump")
        .args(["-d", "-C", binary])
        .output()
        .map_err(|e| format!("spawn objdump: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "objdump failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);

    let mut calls: HashMap<String, Vec<String>> = HashMap::new();
    let mut current: Option<String> = None;

    for line in text.lines() {
        // Function header: "0000000000406aac <main>:"
        if let Some(rest) = line.strip_suffix(":") {
            if let Some((addr_str, sym)) = rest.split_once(" <") {
                if let Some(sym) = sym.strip_suffix('>') {
                    if addr_str.chars().all(|c| c.is_ascii_hexdigit()) {
                        current = Some(sym.to_string());
                        continue;
                    }
                }
            }
        }
        // Instruction lines are tab-indented. We only look for direct
        // calls/jumps that include a "<target>" annotation.
        let Some(cur) = current.as_deref() else {
            continue;
        };
        // Narrow to instructions with a `<...>` target annotation on the
        // right-hand side — that's where objdump writes direct-branch
        // symbol resolutions.
        let Some((pre, after)) = line.rsplit_once(" <") else {
            continue;
        };
        let Some(target) = after.strip_suffix('>') else {
            continue;
        };
        // Instruction mnemonic comes after the bytes column. Scan the
        // whitespace-separated tokens for one that names a call or
        // unconditional branch. x86_64: `call`/`callq` (call) and
        // `jmp`/`jmpq` (tail call when target is another function).
        // aarch64: `bl` (call) and `b` (tail call). Conditional branches
        // (`b.eq`, `b.hs`, …) are intra-function jumps and skipped.
        let has_call_or_jmp = pre
            .split_ascii_whitespace()
            .any(|t| matches!(t, "call" | "callq" | "jmp" | "jmpq" | "bl" | "b"));
        if !has_call_or_jmp {
            continue;
        }
        // Intra-function branches appear as "fn+0xN" targets — a `jmp`
        // to the middle of the current function (loop back-edges, lowered
        // match tables). Drop those; keep entry-point targets, including
        // genuine self-calls like `tree_walk` recursing into itself.
        if let Some((base, _off)) = target.split_once('+') {
            let _ = base;
            continue;
        }
        calls.entry(cur.to_string()).or_default().push(target.to_string());
    }

    // Dedup while preserving first-seen order.
    for v in calls.values_mut() {
        let mut seen = HashSet::new();
        v.retain(|n| seen.insert(n.clone()));
    }

    Ok(calls)
}

// ── Joining stack sizes (address-keyed) with the call graph (name-keyed) ───
//
// `.stack_sizes` is keyed by function address; the call graph from
// `objdump -d -C` uses demangled names. We use `objdump -t -C` to get
// the address ↔ demangled-name mapping, then produce one `stack_by_name`
// table keyed the same way as the call graph.

struct Frames {
    stack_by_name: HashMap<String, u64>,
    addr_by_name: HashMap<String, u64>,
}

/// Parse `objdump -t -C` output for function symbols: the `F .text` lines
/// give us (address, demangled name), matching the names we see on the
/// right-hand side of `call`/`jmp` instructions in `-d -C`.
fn objdump_symtab(binary: &str) -> Result<HashMap<u64, String>, String> {
    let out = Command::new("objdump")
        .args(["-t", "-C", binary])
        .output()
        .map_err(|e| format!("spawn objdump -t: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "objdump -t failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut map = HashMap::new();
    // ELF lines look like:
    //   0000000000406aac g     F .text\t00000000000002d4 main
    //   0000000000401111 l     F .text\t00000000000001c9 ltop::emit_proc_row::<...>
    // Mach-O lines look like (no tab, no size column):
    //   0000000100000678 l     F __TEXT,__text _main
    //   0000000100000678 l     F __TEXT,__text ltop::emit_proc_row::<...>
    //
    // Strategy: locate " F " to peel off the address, then take the
    // section identifier as the next token. After that, ELF has a
    // 16-char hex size (followed by tab + name) and Mach-O goes
    // straight to the name. Detect ELF by the presence of a tab in
    // the post-section remainder.
    for line in text.lines() {
        let Some(f_idx) = line.find(" F ") else { continue; };
        let head = &line[..f_idx];
        let after_f = &line[f_idx + 3..];
        let addr_str = head.split_ascii_whitespace().next().unwrap_or("");
        let Ok(addr) = u64::from_str_radix(addr_str, 16) else { continue; };
        // Drop the section identifier token; keep the rest.
        let rest = match after_f.split_once(char::is_whitespace) {
            Some((_section, r)) => r.trim_start(),
            None => continue,
        };
        // If the next chunk before a tab looks like a hex size, drop it
        // (ELF format). Otherwise the whole rest is the name (Mach-O).
        let name = match rest.split_once('\t') {
            Some((maybe_size, name_tail)) => {
                let s = maybe_size.trim();
                let is_elf_size = s.len() == 16 && s.chars().all(|c| c.is_ascii_hexdigit());
                if is_elf_size { name_tail.trim().to_string() } else { rest.to_string() }
            }
            None => rest.trim().to_string(),
        };
        if !name.is_empty() {
            map.insert(addr, name);
        }
    }
    Ok(map)
}

// ── aarch64 prologue scanning (Mach-O frame-size source) ────────────────────
//
// rustc/LLVM doesn't emit a stack-sizes section on Mach-O, so we recover
// per-function frame sizes by scanning the disassembly. Two prologue
// patterns cover everything LLVM produces for ltop:
//
//   sub  sp, sp, #<imm>          ← single-instruction sp adjustment
//   stp  x29, x30, [sp, #-<N>]!  ← pre-decrement (leaf-style frames)
//
// Multi-step adjustments (frame > 4095 + extras) are rare and capped via
// the .stack_sizes-style threshold check downstream; we capture the first
// `sub sp` we see at function entry. Leaf functions with no `sub sp` get
// frame size 0.
fn objdump_frame_sizes(binary: &str) -> Result<HashMap<String, u64>, String> {
    let out = Command::new("objdump")
        .args(["-d", "-C", binary])
        .output()
        .map_err(|e| format!("spawn objdump -d (frame sizes): {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "objdump -d failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);

    let mut sizes: HashMap<String, u64> = HashMap::new();
    let mut current: Option<String> = None;
    let mut found_size = false;

    for line in text.lines() {
        if let Some(rest) = line.strip_suffix(":") {
            if let Some((addr_str, sym)) = rest.split_once(" <") {
                if let Some(sym) = sym.strip_suffix('>') {
                    if addr_str.chars().all(|c| c.is_ascii_hexdigit()) {
                        if let Some(prev) = current.take() {
                            if !found_size { sizes.insert(prev, 0); }
                        }
                        current = Some(sym.to_string());
                        found_size = false;
                        continue;
                    }
                }
            }
        }
        if found_size { continue; }
        if current.is_none() { continue; }
        if let Some(sz) = parse_aarch64_prologue(line) {
            sizes.insert(current.clone().unwrap(), sz);
            found_size = true;
        }
    }
    if let Some(prev) = current.take() {
        if !found_size { sizes.insert(prev, 0); }
    }
    Ok(sizes)
}

/// Scan one disassembly line for a stack-pointer-adjusting prologue
/// instruction. Returns the frame size if this is the first `sub sp`
/// or `stp …, [sp, #-N]!` we see for a function.
///
/// Lines look like (bytes column varies):
///   "100000678: d10203ff    \tsub\tsp, sp, #0x80"
///   "100000680: a907 7bfd    \tstp\tx29, x30, [sp, #0x70]"
fn parse_aarch64_prologue(line: &str) -> Option<u64> {
    // Drop everything up to the first tab (address + raw bytes column).
    let after_tab = line.split('\t').nth(1)?;
    // `after_tab` is the mnemonic; the operands follow another tab.
    let mnemonic = after_tab.trim();
    let operands = line.split('\t').nth(2).unwrap_or("").trim();
    match mnemonic {
        "sub" => {
            // "sp, sp, #0xN"
            let rest = operands.strip_prefix("sp,")?.trim_start();
            let rest = rest.strip_prefix("sp,")?.trim_start();
            let imm = rest.strip_prefix('#')?;
            // Trim a trailing comment (`; =128`) that some objdump
            // versions append after the immediate.
            let imm = imm.split([',', ' ', ';']).next().unwrap_or("");
            parse_imm(imm)
        }
        "stp" => {
            // Match "x29, x30, [sp, #-N]!" (pre-decrement). N is the
            // frame size. Plain "stp …, [sp, #N]" (no `!`) is a non-
            // updating store and not a frame setup.
            if !operands.starts_with("x29, x30, [sp, #-") { return None; }
            if !operands.contains("]!") { return None; }
            let rest = operands.split("#-").nth(1)?;
            let imm = rest.split(']').next()?;
            parse_imm(imm)
        }
        _ => None,
    }
}

fn parse_imm(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

// ── DFS → event stream ──────────────────────────────────────────────────────

enum Event {
    Push { name: String, size: u64 },
    Pop,
    // An inlined call inside the innermost real frame. `frame_lo`/`hi`
    // are fbreg-relative offsets into that frame — the renderer pairs
    // them with the parent's base to place the sub-rect.
    InlinedPush { name: String, frame_lo: i64, frame_hi: i64 },
    InlinedPop,
    Loop { name: String }, // recursion closed; no further nesting
    Unknown { name: String }, // call to a function we have no stack info for
}

struct Bounds {
    /// Per-function max depth on the current path before we collapse
    /// further recursion into a Loop event. 0 = never unroll (same as
    /// absent, i.e. strict option 1). Default for anything not listed.
    map: HashMap<String, u32>,
    default: u32,
}

fn walk(
    entry: &str,
    calls: &HashMap<String, Vec<String>>,
    frames: &Frames,
    bounds: &Bounds,
    inlines: &InlineMap,
) -> Vec<Event> {
    let mut ev = Vec::new();
    let mut on_path: HashMap<String, u32> = HashMap::new();
    walk_r(entry, calls, frames, bounds, inlines, &mut on_path, &mut ev, 0);
    ev
}

const MAX_DEPTH: u32 = 64;
const MAX_EVENTS: usize = 200_000;

fn walk_r(
    name: &str,
    calls: &HashMap<String, Vec<String>>,
    frames: &Frames,
    bounds: &Bounds,
    inlines: &InlineMap,
    on_path: &mut HashMap<String, u32>,
    ev: &mut Vec<Event>,
    depth: u32,
) {
    if ev.len() >= MAX_EVENTS || depth > MAX_DEPTH {
        return;
    }
    let bound = *bounds.map.get(name).unwrap_or(&bounds.default);
    let count = on_path.get(name).copied().unwrap_or(0);
    if count >= bound.max(1) && count > 0 {
        ev.push(Event::Loop { name: name.to_string() });
        return;
    }
    let Some(&size) = frames.stack_by_name.get(name) else {
        // No .stack_sizes entry → either an extern function (any
        // libc startup / syscall wrappers we still link to, or
        // compiler-builtins mem* helpers) or a symbol outside our
        // build. Record it as Unknown and don't recurse through its
        // callees.
        ev.push(Event::Unknown { name: name.to_string() });
        return;
    };
    ev.push(Event::Push { name: name.to_string(), size });
    *on_path.entry(name.to_string()).or_insert(0) += 1;

    // Emit inlined-call sub-rects within this real frame, if DWARF gave
    // us a decomposition. Inlined calls are emitted first so they sit
    // on the "left side" of the parent's X span, visually grouped before
    // any real callees. (A richer layout would interleave by call-site
    // instruction address, but flat-grouped already gives us the peek-
    // inside structure we want.)
    if let Some(addr) = frames.addr_by_name.get(name) {
        if let Some(ics) = inlines.by_addr.get(addr) {
            for ic in ics {
                emit_inlined(ic, ev);
            }
        }
    }

    if let Some(targets) = calls.get(name) {
        for t in targets {
            walk_r(t, calls, frames, bounds, inlines, on_path, ev, depth + 1);
        }
    }
    *on_path.get_mut(name).unwrap() -= 1;
    if on_path[name] == 0 {
        on_path.remove(name);
    }
    ev.push(Event::Pop);
}

// Minimum visible frame range for an inlined call. Inlined functions
// whose locals all live in registers show up in DWARF with no on-stack
// variables; we record those as (0, 1) placeholders. A 1-byte stripe is
// not useful information — filter them out so the plot stays readable.
const INLINED_MIN_BYTES: i64 = 4;

fn emit_inlined(ic: &InlinedCall, ev: &mut Vec<Event>) {
    if ev.len() >= MAX_EVENTS {
        return;
    }
    let visible = ic.frame_hi - ic.frame_lo >= INLINED_MIN_BYTES;
    if visible {
        ev.push(Event::InlinedPush {
            name: ic.name.clone(),
            frame_lo: ic.frame_lo,
            frame_hi: ic.frame_hi,
        });
    }
    for child in &ic.children {
        emit_inlined(child, ev);
    }
    if visible {
        ev.push(Event::InlinedPop);
    }
}

// ── Rect extraction (mirrors arena-svg) ─────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum RectKind {
    Real,
    Inlined,
}

struct Rect {
    name: String,
    start: u32, // event index at push
    end: u32,   // event index at matching pop
    offset_lo: u64,
    offset_hi: u64,
    kind: RectKind,
}

struct Marker {
    name: String,
    at: u32,     // event index
    offset: u64, // stack offset at which to draw
    kind: MarkerKind,
}

enum MarkerKind {
    Loop,
    Unknown,
}

fn build_rects(events: &[Event]) -> (Vec<Rect>, Vec<Marker>, u64) {
    let mut rects = Vec::new();
    let mut markers = Vec::new();
    // Real-frame opens (start_evt, lo, hi, name). Also doubles as the
    // "parent base" stack: the top entry's `lo` is where the innermost
    // real frame starts, and inlined calls place their sub-rects at
    // `lo + ic.frame_lo..lo + ic.frame_hi`.
    let mut open_real: Vec<(usize, u64, u64, String)> = Vec::new();
    let mut open_inlined: Vec<(usize, u64, u64, String)> = Vec::new();
    let mut cur: u64 = 0;
    let mut peak: u64 = 0;
    for (i, e) in events.iter().enumerate() {
        let i = i as u32;
        match e {
            Event::Push { name, size } => {
                let lo = cur;
                cur += size;
                peak = peak.max(cur);
                open_real.push((i as usize, lo, cur, name.clone()));
            }
            Event::Pop => {
                if let Some((start, lo, hi, name)) = open_real.pop() {
                    rects.push(Rect {
                        name,
                        start: start as u32,
                        end: i,
                        offset_lo: lo,
                        offset_hi: hi,
                        kind: RectKind::Real,
                    });
                    cur = lo;
                }
            }
            Event::InlinedPush { name, frame_lo, frame_hi } => {
                let parent_lo = open_real.last().map(|(_, lo, _, _)| *lo).unwrap_or(0);
                // fbreg offsets are signed; clamp to >= 0 relative to
                // the parent frame. Negative fbreg entries appear on
                // some targets (caller-saved args above fbreg 0); we
                // clamp rather than flip so a tiny clip doesn't collapse
                // the rect to nothing.
                let lo = (parent_lo as i64 + *frame_lo).max(parent_lo as i64) as u64;
                let hi = (parent_lo as i64 + *frame_hi)
                    .max(lo as i64 + 1) as u64;
                // Also clamp above by the real frame's top so a clipped
                // location doesn't wander outside its parent visually.
                let parent_hi = open_real
                    .last()
                    .map(|(_, _, h, _)| *h)
                    .unwrap_or(u64::MAX);
                let hi = hi.min(parent_hi);
                open_inlined.push((i as usize, lo, hi, name.clone()));
            }
            Event::InlinedPop => {
                if let Some((start, lo, hi, name)) = open_inlined.pop() {
                    rects.push(Rect {
                        name,
                        start: start as u32,
                        end: i,
                        offset_lo: lo,
                        offset_hi: hi,
                        kind: RectKind::Inlined,
                    });
                }
            }
            Event::Loop { name } => {
                markers.push(Marker {
                    name: name.clone(),
                    at: i,
                    offset: cur,
                    kind: MarkerKind::Loop,
                });
            }
            Event::Unknown { name } => {
                markers.push(Marker {
                    name: name.clone(),
                    at: i,
                    offset: cur,
                    kind: MarkerKind::Unknown,
                });
            }
        }
    }
    let end = events.len() as u32;
    while let Some((start, lo, hi, name)) = open_real.pop() {
        rects.push(Rect {
            name,
            start: start as u32,
            end,
            offset_lo: lo,
            offset_hi: hi,
            kind: RectKind::Real,
        });
    }
    while let Some((start, lo, hi, name)) = open_inlined.pop() {
        rects.push(Rect {
            name,
            start: start as u32,
            end,
            offset_lo: lo,
            offset_hi: hi,
            kind: RectKind::Inlined,
        });
    }
    (rects, markers, peak)
}

// ── Event X mapping ─────────────────────────────────────────────────────────

const EVENT_X_MIN_STEP: u64 = 16;
const EVENT_X_MAX_STEP: u64 = 32 * 1024;

fn compute_event_x(events: &[Event]) -> (Vec<u64>, u64) {
    let mut xs = Vec::with_capacity(events.len() + 1);
    let mut x: u64 = 0;
    xs.push(x);
    for e in events {
        let step = match e {
            Event::Push { size, .. } => (*size).clamp(EVENT_X_MIN_STEP, EVENT_X_MAX_STEP),
            Event::InlinedPush { frame_lo, frame_hi, .. } => {
                let span = (*frame_hi - *frame_lo).max(0) as u64;
                span.clamp(EVENT_X_MIN_STEP, EVENT_X_MAX_STEP)
            }
            Event::Pop
            | Event::InlinedPop
            | Event::Loop { .. }
            | Event::Unknown { .. } => EVENT_X_MIN_STEP,
        };
        x += step;
        xs.push(x);
    }
    (xs, x)
}

// ── SVG emission (layout vocabulary borrowed from arena-svg) ────────────────

const MARGIN_L: u32 = 60;
const MARGIN_R: u32 = 320;
const MARGIN_T: u32 = 30;
const MARGIN_B: u32 = 40;
const PLOT_W: u32 = 1800;
const PLOT_H: u32 = 700;

fn hash_label(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in s.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

fn hsl(label: &str) -> String {
    let hue = hash_label(label) % 360;
    format!("hsl({hue}, 65%, 55%)")
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}

fn format_bytes(b: u64) -> String {
    if b >= 1024 * 1024 {
        format!("{:.1} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{} KB", b / 1024)
    } else {
        format!("{b} B")
    }
}

/// Trim very long demangled symbols to a readable prefix + suffix.
fn shorten(name: &str) -> String {
    let max = 60;
    if name.len() <= max {
        return name.to_string();
    }
    let head = &name[..max / 2];
    let tail = &name[name.len() - max / 2..];
    format!("{head}…{tail}")
}

fn render_svg(
    rects: &[Rect],
    markers: &[Marker],
    event_x: &[u64],
    total_x: u64,
    peak: u64,
    entry: &str,
) -> String {
    let w = MARGIN_L + PLOT_W + MARGIN_R;
    let h = MARGIN_T + PLOT_H + MARGIN_B;
    let total_x = total_x.max(1);
    let peak = peak.max(1);

    let x = |evt: u32| -> f64 {
        let vx = event_x[evt as usize] as f64;
        MARGIN_L as f64 + (vx / total_x as f64) * PLOT_W as f64
    };
    let y = |off: u64| -> f64 {
        MARGIN_T as f64 + PLOT_H as f64 * (1.0 - off as f64 / peak as f64)
    };

    let mut s = String::with_capacity(rects.len() * 150);
    writeln!(s, r##"<?xml version="1.0" encoding="UTF-8"?>"##).unwrap();
    writeln!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="monospace" font-size="11">"##
    ).unwrap();
    writeln!(s, r##"<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>"##).unwrap();

    // Plot frame + gridlines.
    writeln!(
        s,
        r##"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#888"/>"##,
        MARGIN_L, MARGIN_T, PLOT_W, PLOT_H
    ).unwrap();
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let off = (peak as f64 * frac) as u64;
        let yy = y(off);
        writeln!(
            s,
            r##"<line x1="{}" x2="{}" y1="{:.1}" y2="{:.1}" stroke="#eee"/>"##,
            MARGIN_L, MARGIN_L + PLOT_W, yy, yy,
        ).unwrap();
        writeln!(
            s,
            r##"<text x="{}" y="{:.1}" text-anchor="end" dominant-baseline="middle" fill="#555">{}</text>"##,
            MARGIN_L - 6, yy, format_bytes(off),
        ).unwrap();
    }

    let n_events = (event_x.len().saturating_sub(1)) as u32;
    writeln!(
        s,
        r##"<text x="{}" y="16" font-size="13" fill="#222">stack timeline from {} — {} events, peak {}, {} rects</text>"##,
        MARGIN_L, xml_escape(entry), n_events, format_bytes(peak), rects.len(),
    ).unwrap();

    // Frame rectangles. Draw real rects first (as the background); then
    // inlined rects on top — painted with a dashed stroke so they read
    // as "views into the parent" rather than separate frames.
    let mut real_rects: Vec<&Rect> = rects.iter().filter(|r| r.kind == RectKind::Real).collect();
    // Larger rects first so smaller ones on top aren't covered by them.
    real_rects.sort_by(|a, b| (b.offset_hi - b.offset_lo).cmp(&(a.offset_hi - a.offset_lo)));
    for r in real_rects {
        let x0 = x(r.start);
        let x1 = x(r.end).max(x0 + 0.4);
        let y0 = y(r.offset_hi);
        let y1 = y(r.offset_lo);
        let rw = (x1 - x0).max(0.4);
        let rh = (y1 - y0).max(0.4);
        writeln!(
            s,
            r##"<rect x="{x0:.2}" y="{y0:.2}" width="{rw:.2}" height="{rh:.2}" fill="{}" fill-opacity="0.55" stroke="#222" stroke-width="0.3"><title>{}: [{}..{}) {} @ evt {}..{}</title></rect>"##,
            hsl(&r.name),
            xml_escape(&r.name),
            r.offset_lo,
            r.offset_hi,
            format_bytes(r.offset_hi - r.offset_lo),
            r.start,
            r.end,
        ).unwrap();
    }
    for r in rects.iter().filter(|r| r.kind == RectKind::Inlined) {
        let x0 = x(r.start);
        let x1 = x(r.end).max(x0 + 0.4);
        let y0 = y(r.offset_hi);
        let y1 = y(r.offset_lo);
        let rw = (x1 - x0).max(0.4);
        let rh = (y1 - y0).max(0.4);
        writeln!(
            s,
            r##"<rect x="{x0:.2}" y="{y0:.2}" width="{rw:.2}" height="{rh:.2}" fill="{}" fill-opacity="0.9" stroke="#111" stroke-width="0.3" stroke-dasharray="2,1.5"><title>inlined {}: [{}..{}) {} @ evt {}..{}</title></rect>"##,
            hsl(&r.name),
            xml_escape(&r.name),
            r.offset_lo,
            r.offset_hi,
            format_bytes(r.offset_hi - r.offset_lo),
            r.start,
            r.end,
        ).unwrap();
    }

    // Markers — loops and unknown calls as thin vertical ticks with glyphs.
    for m in markers {
        let xx = x(m.at);
        let yy = y(m.offset);
        let (stroke, glyph, tip) = match m.kind {
            MarkerKind::Loop => ("#c22", "↻", "recursion"),
            MarkerKind::Unknown => ("#888", "?", "no .stack_sizes"),
        };
        writeln!(
            s,
            r##"<line x1="{xx:.1}" x2="{xx:.1}" y1="{:.1}" y2="{:.1}" stroke="{stroke}" stroke-width="0.6" stroke-dasharray="2,2"><title>{} {}</title></line>"##,
            yy - 4.0, yy + 4.0, xml_escape(&m.name), tip,
        ).unwrap();
        writeln!(
            s,
            r##"<text x="{xx:.1}" y="{:.1}" text-anchor="middle" fill="{stroke}" font-size="10">{glyph}</text>"##,
            yy - 6.0,
        ).unwrap();
    }

    // Legend: one row per unique function, sorted by max frame size.
    // We show the tallest frames first since those are the stack cost
    // drivers; the bar width on the y-axis is already the visual cue.
    let mut per_name: BTreeMap<&str, (u64, u64)> = BTreeMap::new(); // (max_size, total_span)
    for r in rects {
        let e = per_name.entry(r.name.as_str()).or_insert((0, 0));
        let size = r.offset_hi - r.offset_lo;
        if size > e.0 {
            e.0 = size;
        }
        e.1 += r.end.saturating_sub(r.start) as u64 * size.max(1);
    }
    let mut legend: Vec<(&str, u64, u64)> =
        per_name.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
    legend.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));

    let legend_x = (MARGIN_L + PLOT_W + 16) as f64;
    let line_h = 13.0;
    writeln!(
        s,
        r##"<text x="{:.1}" y="{:.1}" font-size="12" fill="#222">frame (max size)</text>"##,
        legend_x, (MARGIN_T as f64) - 10.0,
    ).unwrap();
    let mut ly = (MARGIN_T as f64) + line_h * 0.5;
    for (name, max_size, _) in legend.iter().take(42) {
        writeln!(
            s,
            r##"<rect x="{:.1}" y="{:.1}" width="12" height="10" fill="{}"/>"##,
            legend_x, ly - 5.0, hsl(name),
        ).unwrap();
        writeln!(
            s,
            r##"<text x="{:.1}" y="{:.1}" dominant-baseline="middle" fill="#222">{} ({})</text>"##,
            legend_x + 16.0, ly, xml_escape(&shorten(name)), format_bytes(*max_size),
        ).unwrap();
        ly += line_h;
    }

    // Footer: x-axis tick labels (cumulative "stack-byte-events").
    let xl_y = (MARGIN_T + PLOT_H + 14) as f64;
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let vx = (total_x as f64 * frac) as u64;
        let svg_x = MARGIN_L as f64 + frac * PLOT_W as f64;
        writeln!(
            s,
            r##"<text x="{:.1}" y="{:.1}" text-anchor="middle" fill="#555">{}</text>"##,
            svg_x, xl_y, format_bytes(vx),
        ).unwrap();
    }

    s.push_str("</svg>\n");
    s
}

// ── Main ────────────────────────────────────────────────────────────────────

fn parse_bounds(args: &[String]) -> (Bounds, Option<String>, Vec<String>) {
    // Very small arg parser: --bound FN=N, --entry NAME, positional = binary, svg.
    let mut map = HashMap::new();
    let mut default = 1u32;
    let mut entry = None;
    let mut pos = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--bound" && i + 1 < args.len() {
            let spec = &args[i + 1];
            if let Some((fname, n)) = spec.split_once('=') {
                if let Ok(n) = n.parse() {
                    map.insert(fname.to_string(), n);
                }
            }
            i += 2;
        } else if a == "--bound-default" && i + 1 < args.len() {
            if let Ok(n) = args[i + 1].parse() {
                default = n;
            }
            i += 2;
        } else if a == "--entry" && i + 1 < args.len() {
            entry = Some(args[i + 1].clone());
            i += 2;
        } else {
            pos.push(a.clone());
            i += 1;
        }
    }
    (Bounds { map, default }, entry, pos)
}

fn run() -> Result<(), String> {
    let all: Vec<String> = env::args().skip(1).collect();
    let (bounds, entry_arg, pos) = parse_bounds(&all);
    let binary = pos
        .first()
        .map(String::as_str)
        .unwrap_or(default_binary_path());
    let out_path = pos.get(1).map(String::as_str);
    let entry = entry_arg.as_deref().unwrap_or("main");

    let bytes = fs::read(binary).map_err(|e| format!("read {binary}: {e}"))?;
    let format = detect_format(&bytes)?;

    // objdump is our disassembly backend on both formats. If it's missing
    // we surface the error clearly — the binary is useless without a
    // call graph.
    let calls = objdump_calls(binary)?;
    let demangled = objdump_symtab(binary)?;

    // Frame sizes: from `.stack_sizes` on ELF, from prologue scanning
    // on Mach-O (rustc/LLVM doesn't emit a stack-sizes section there).
    // Also collect a `Sym` list for filling `addr_by_name` — on Mach-O
    // we synthesise it from objdump's symtab output since we don't
    // walk the binary's own symbol table.
    let (stack_by_name, addr_by_name): (HashMap<String, u64>, HashMap<String, u64>) =
        match format {
            Format::Elf => {
                let sections = parse_sections(&bytes)?;
                let syms = parse_symbols(&sections);
                let stack_sizes_by_addr = parse_stack_sizes(&sections);
                if stack_sizes_by_addr.is_empty() {
                    return Err(format!(
                        ".stack_sizes section missing or empty in {binary}\n\
                         (did you build with `cargo stack`?)"
                    ));
                }
                let mut stack_by_name: HashMap<String, u64> = HashMap::new();
                let mut addr_by_name: HashMap<String, u64> = HashMap::new();
                for (addr, dname) in &demangled {
                    addr_by_name.insert(dname.clone(), *addr);
                    if let Some(&sz) = stack_sizes_by_addr.get(addr) {
                        stack_by_name.insert(dname.clone(), sz);
                    }
                }
                for s in &syms {
                    addr_by_name.entry(s.name.clone()).or_insert(s.addr);
                    if !stack_by_name.contains_key(&s.name) {
                        if let Some(&sz) = stack_sizes_by_addr.get(&s.addr) {
                            stack_by_name.insert(s.name.clone(), sz);
                        }
                    }
                }
                (stack_by_name, addr_by_name)
            }
            Format::MachO => {
                let stack_by_name = objdump_frame_sizes(binary)?;
                if stack_by_name.is_empty() {
                    return Err(format!(
                        "no aarch64 prologues found in {binary} (did you build with \
                         `cargo stack`? Mach-O frame sizes come from objdump prologue \
                         scans, not a stack-sizes section)"
                    ));
                }
                let mut addr_by_name: HashMap<String, u64> = HashMap::new();
                for (addr, dname) in &demangled {
                    addr_by_name.insert(dname.clone(), *addr);
                }
                (stack_by_name, addr_by_name)
            }
        };
    let frames = Frames { stack_by_name, addr_by_name };

    if !frames.addr_by_name.contains_key(entry) {
        return Err(format!(
            "entry `{entry}` not found in symtab (try --entry _main or a mangled name)"
        ));
    }

    // DWARF inlined-subroutine decomposition: ELF only. Mach-O DWARF
    // commonly lives in a separate .dSYM bundle (or LC_UUID-keyed in
    // __DWARF segment with debug=2), and gimli's Mach-O loader takes
    // some setup we haven't done yet — skip on Mach-O and show real-
    // frame rectangles only.
    let inlines = match format {
        Format::Elf => parse_inlines(&parse_sections(&bytes)?).unwrap_or_else(|e| {
            eprintln!("stack-svg: DWARF parse warning ({e}) — inlined decomposition disabled");
            InlineMap { by_addr: HashMap::new() }
        }),
        Format::MachO => InlineMap { by_addr: HashMap::new() },
    };

    let events = walk(entry, &calls, &frames, &bounds, &inlines);
    let (rects, markers, peak) = build_rects(&events);
    let (event_x, total_x) = compute_event_x(&events);
    let svg = render_svg(&rects, &markers, &event_x, total_x, peak, entry);

    match out_path {
        Some(p) => fs::write(p, &svg).map_err(|e| format!("write {p}: {e}"))?,
        None => io::stdout()
            .write_all(svg.as_bytes())
            .map_err(|e| format!("stdout: {e}"))?,
    }

    eprintln!(
        "stack-svg: {} frames seen, {} events, peak stack {} from `{}` ({} loops, {} unknown)",
        rects.len(),
        events.len(),
        format_bytes(peak),
        entry,
        markers.iter().filter(|m| matches!(m.kind, MarkerKind::Loop)).count(),
        markers.iter().filter(|m| matches!(m.kind, MarkerKind::Unknown)).count(),
    );

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("stack-svg: {e}");
            ExitCode::FAILURE
        }
    }
}
