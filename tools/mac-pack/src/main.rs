//! Mach-O post-processor that packs ltop's min binary into a single
//! 16 KB __TEXT page.
//!
//! ld64's layout places __text at the end of the __TEXT segment, so
//! even when `header + load_commands + __text + __const` total well
//! under 16 KB, ld64 still emits a 32 KB __TEXT segment (header +
//! ~15 KB of zero pad + __text + __const). This tool rewrites the
//! output so __text sits immediately after the load commands and
//! __TEXT shrinks to a single page, dropping the second 16 KB page
//! from the file.
//!
//! Address-preservation trick: shift __text's vmaddr by exactly
//! −0x4000 (one page) and shift __DATA and __LINKEDIT's vmaddrs by
//! the same −0x4000. Every adrp+add reference within __text stays
//! valid — the source instruction and its target both shifted by
//! four pages, so the encoded page-delta is unchanged. No code
//! patching needed.
//!
//! After rewriting, recompute the ad-hoc code signature (SHA-256
//! hashes of each 4 KB code slot). Format matches Go's
//! `cmd/internal/codesign` (our reference implementation) and
//! `ld-prime`'s default output.
//!
//! Usage:
//!   mac-pack <input-binary> <output-binary>
//!
//! Called from xtask after `cargo ltop` builds the unpacked binary.

use std::env;
use std::fs;
use std::process::exit;

use sha2::{Digest, Sha256};

// ─── Mach-O constants ──────────────────────────────────────────────────────

const MH_MAGIC_64: u32 = 0xfeedfacf;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_DYLD_INFO_ONLY: u32 = 0x80000022;
const LC_MAIN: u32 = 0x80000028;
const LC_CODE_SIGNATURE: u32 = 0x1d;
const LC_FUNCTION_STARTS: u32 = 0x26;
const LC_DATA_IN_CODE: u32 = 0x29;
const LC_DYLD_EXPORTS_TRIE: u32 = 0x80000033;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x80000034;

const PAGE: u64 = 0x4000; // 16 KB — arm64 hardware page
const SHIFT: u64 = PAGE;  // we're shrinking __TEXT by one page

// Code-signing constants.
const CSMAGIC_EMBEDDED_SIGNATURE: u32 = 0xfade_0cc0;
const CSMAGIC_CODEDIRECTORY: u32 = 0xfade_0c02;
const CSSLOT_CODEDIRECTORY: u32 = 0;
const CS_HASHTYPE_SHA256: u8 = 2;
const CS_EXECSEG_MAIN_BINARY: u64 = 0x1;
const CS_PAGE_SIZE_BITS: u8 = 12; // log2(4096) — hash-page size
const CS_PAGE_SIZE: usize = 1 << CS_PAGE_SIZE_BITS;
const CS_HASH_SIZE: usize = 32; // SHA-256

// ─── Helpers ───────────────────────────────────────────────────────────────

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("mac-pack: {}", msg.as_ref());
    exit(2);
}

fn passthrough(buf: &[u8], input_path: &str, output_path: &str, reason: &str) {
    fs::write(output_path, buf)
        .unwrap_or_else(|e| die(format!("write {}: {}", output_path, e)));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perm = fs::metadata(input_path)
            .map(|m| m.permissions())
            .unwrap_or_else(|e| die(format!("stat {}: {}", input_path, e)));
        fs::set_permissions(output_path, fs::Permissions::from_mode(perm.mode()))
            .unwrap_or_else(|e| die(format!("chmod {}: {}", output_path, e)));
    }
    // The passthrough path fires on every `cargo ltop` build (current
    // ltop has 3 sections in __TEXT — __text + __const + __cstring —
    // and mac-pack only knows how to collapse the 2-section case;
    // see the README's Footprint section for why we hold at 33 KB).
    // Silent by default — it's not actionable noise. Opt in with
    // `MAC_PACK_VERBOSE=1` if you're debugging the layout.
    if std::env::var_os("MAC_PACK_VERBOSE").is_some() {
        eprintln!("mac-pack: {} ({} B), no packing", reason, buf.len());
    }
}

fn r_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn r_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}
fn w_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn w_u64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn put_be_u32(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_be_bytes()); }
fn put_be_u64(out: &mut Vec<u8>, v: u64) { out.extend_from_slice(&v.to_be_bytes()); }

// ─── Segment / section iteration ───────────────────────────────────────────

/// One LC_SEGMENT_64 entry plus its enclosing load-command range.
struct Segment {
    /// File offset of the load-command header (LC_SEGMENT_64).
    lc_off: usize,
    name: [u8; 16],
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    nsects: u32,
    /// Offsets of each section_64 header within the file (each is 80 B).
    sect_offs: Vec<usize>,
}

impl Segment {
    fn is(&self, n: &[u8]) -> bool {
        let got = self.name.split(|&b| b == 0).next().unwrap_or(&[]);
        got == n
    }
}

/// Walk the load commands and collect every segment.
fn parse_segments(buf: &[u8]) -> Vec<Segment> {
    assert_eq!(r_u32(buf, 0), MH_MAGIC_64, "not a 64-bit little-endian Mach-O");
    let ncmds = r_u32(buf, 16) as usize;
    let sizeofcmds = r_u32(buf, 20) as usize;
    let mut segs = Vec::new();
    let mut off = 32; // sizeof(mach_header_64)
    let end = 32 + sizeofcmds;
    for _ in 0..ncmds {
        if off + 8 > end { break; }
        let cmd = r_u32(buf, off);
        let cmdsize = r_u32(buf, off + 4) as usize;
        if cmd == LC_SEGMENT_64 {
            // struct segment_command_64: cmd, cmdsize, segname[16], vmaddr,
            // vmsize, fileoff, filesize, maxprot, initprot, nsects, flags.
            let mut name = [0u8; 16];
            name.copy_from_slice(&buf[off + 8..off + 24]);
            let nsects = r_u32(buf, off + 64);
            let sect_header_off = off + 72; // after the LC_SEGMENT_64 header
            let mut sect_offs = Vec::with_capacity(nsects as usize);
            for i in 0..nsects as usize {
                sect_offs.push(sect_header_off + i * 80);
            }
            segs.push(Segment {
                lc_off: off,
                name,
                vmaddr: r_u64(buf, off + 24),
                vmsize: r_u64(buf, off + 32),
                fileoff: r_u64(buf, off + 40),
                filesize: r_u64(buf, off + 48),
                nsects,
                sect_offs,
            });
        }
        off += cmdsize;
    }
    segs
}

fn segment_mut_idx(segs: &[Segment], name: &[u8]) -> usize {
    segs.iter().position(|s| s.is(name))
        .unwrap_or_else(|| die(format!("segment {} not found", std::str::from_utf8(name).unwrap_or("?"))))
}

// ─── Per-section reader ────────────────────────────────────────────────────

struct SectionRead {
    addr: u64,
    size: u64,
    offset: u32,
    align: u32,
}

fn read_section(buf: &[u8], hdr_off: usize) -> SectionRead {
    // struct section_64: sectname[16], segname[16], addr, size, offset,
    // align, reloff, nreloc, flags, reserved1, reserved2, reserved3.
    SectionRead {
        addr: r_u64(buf, hdr_off + 32),
        size: r_u64(buf, hdr_off + 40),
        offset: r_u32(buf, hdr_off + 48),
        align: r_u32(buf, hdr_off + 52),
    }
}

fn write_section(buf: &mut [u8], hdr_off: usize, addr: u64, offset: u32) {
    w_u64(buf, hdr_off + 32, addr);
    w_u32(buf, hdr_off + 48, offset);
}


// ─── Ad-hoc code signature ─────────────────────────────────────────────────

/// Compute the total signature size for a binary of `code_size` bytes
/// signed with identifier `id`. Matches Go's `codesign.Size`.
fn sig_size(code_size: u64, id: &str) -> usize {
    // CodeDirectory fixed fields = 13*4 + 4 + 4*8 = 88 bytes.
    const CD_FIXED: usize = 88;
    let ident_off = CD_FIXED;
    let hash_off = ident_off + id.len() + 1; // +1 for trailing NUL
    let n_hashes = (code_size as usize + CS_PAGE_SIZE - 1) / CS_PAGE_SIZE;
    let cdir_size = hash_off + n_hashes * CS_HASH_SIZE;
    // SuperBlob header (12) + one Blob index entry (8) + CodeDirectory.
    12 + 8 + cdir_size
}

/// Emit the complete SuperBlob → CodeDirectory → identifier → hashes
/// byte sequence into `out`. `data` is the binary up to `code_size`
/// (which is the file offset at which the signature itself begins).
fn sign_adhoc(
    out: &mut Vec<u8>,
    data: &[u8],
    id: &str,
    code_size: u64,
    text_off: u64,
    text_size: u64,
) {
    const CD_FIXED: usize = 88;
    let n_hashes = (code_size as usize + CS_PAGE_SIZE - 1) / CS_PAGE_SIZE;
    let ident_off = CD_FIXED;
    let hash_off = ident_off + id.len() + 1;
    let cdir_size = hash_off + n_hashes * CS_HASH_SIZE;
    let total = 12 + 8 + cdir_size;

    // SuperBlob (big-endian, all code-sig fields are BE).
    put_be_u32(out, CSMAGIC_EMBEDDED_SIGNATURE);
    put_be_u32(out, total as u32);
    put_be_u32(out, 1);                                 // count: one CodeDirectory blob

    // Blob index: type = CSSLOT_CODEDIRECTORY, offset = 12 + 8 = 20
    put_be_u32(out, CSSLOT_CODEDIRECTORY);
    put_be_u32(out, (12 + 8) as u32);

    // CodeDirectory (88 fixed bytes).
    put_be_u32(out, CSMAGIC_CODEDIRECTORY);
    put_be_u32(out, cdir_size as u32);                  // length
    put_be_u32(out, 0x0002_0400);                       // version
    put_be_u32(out, 0x0002_0002);                       // flags: adhoc | linkerSigned
    put_be_u32(out, hash_off as u32);                   // hashOffset
    put_be_u32(out, ident_off as u32);                  // identOffset
    put_be_u32(out, 0);                                 // nSpecialSlots
    put_be_u32(out, n_hashes as u32);                   // nCodeSlots
    put_be_u32(out, code_size as u32);                  // codeLimit
    out.push(CS_HASH_SIZE as u8);                       // hashSize
    out.push(CS_HASHTYPE_SHA256);                       // hashType
    out.push(0);                                        // _pad1
    out.push(CS_PAGE_SIZE_BITS);                        // pageSize (log2)
    put_be_u32(out, 0);                                 // _pad2
    put_be_u32(out, 0);                                 // scatterOffset
    put_be_u32(out, 0);                                 // teamOffset
    put_be_u32(out, 0);                                 // _pad3
    put_be_u64(out, 0);                                 // codeLimit64 (0 because < 4 GiB)
    put_be_u64(out, text_off);                          // execSegBase
    put_be_u64(out, text_size);                         // execSegLimit
    put_be_u64(out, CS_EXECSEG_MAIN_BINARY);            // execSegFlags

    // Identifier, NUL-terminated.
    out.extend_from_slice(id.as_bytes());
    out.push(0);

    // Page hashes. Hash each 4 KB slice of `data[0..code_size]`.
    let code_size = code_size as usize;
    for i in 0..n_hashes {
        let start = i * CS_PAGE_SIZE;
        let end = std::cmp::min(start + CS_PAGE_SIZE, code_size);
        let mut h = Sha256::new();
        h.update(&data[start..end]);
        out.extend_from_slice(&h.finalize());
    }
    let _ = total;
}

// ─── The packer ────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        die("usage: mac-pack <input> <output>");
    }
    let input_path = &args[1];
    let output_path = &args[2];

    let mut buf = fs::read(input_path)
        .unwrap_or_else(|e| die(format!("read {}: {}", input_path, e)));

    // ── Parse ──────────────────────────────────────────────────────────────
    let segs = parse_segments(&buf);
    let text_idx = segment_mut_idx(&segs, b"__TEXT");
    let data_idx = segment_mut_idx(&segs, b"__DATA");
    let linkedit_idx = segment_mut_idx(&segs, b"__LINKEDIT");

    let text = &segs[text_idx];
    // If __TEXT is already 1 page (ld64 packed it tightly because the
    // binary is small enough to fit header + load_commands + sections
    // within a single 16 KB page), there is nothing to collapse.
    if text.vmsize == PAGE && text.filesize == PAGE {
        passthrough(&buf, input_path, output_path, "__TEXT already 1 page");
        return;
    }
    // Expect __TEXT to currently span 2 pages and have __text + __const only.
    if text.vmsize != 2 * PAGE || text.filesize != 2 * PAGE {
        die(format!("unexpected __TEXT layout: vmsize=0x{:x} filesize=0x{:x}",
            text.vmsize, text.filesize));
    }
    if text.nsects != 2 {
        // Builds that link Mach frameworks (e.g. IOKit + CoreFoundation
        // for the AGX GPU backend) emit __stubs / __stub_helper /
        // __cstring sections on top of __text + __const. Compacting that
        // layout requires shifting more than two section offsets and
        // also handling the now-non-empty __DATA segment, which is
        // future work — for now, passthrough and ship the larger binary.
        // The non-GPU min build (no framework deps) still hits the
        // 1-page packed path.
        passthrough(&buf, input_path, output_path,
            &format!("__TEXT has {} sections (>2); skip compaction", text.nsects));
        return;
    }
    let text_sect = read_section(&buf, text.sect_offs[0]);
    let const_sect = read_section(&buf, text.sect_offs[1]);
    // ld64 has two layouts for a 2-page __TEXT:
    //   Old (pre-macOS 16): sections placed at the END of the segment
    //     → __text offset ≥ SHIFT (in page 2); subtract SHIFT to compact.
    //   New (macOS 16+):   sections placed immediately after load commands
    //     → __text offset < SHIFT (already in page 1); offsets unchanged.
    // In both cases we strip the unused page and shrink __TEXT to PAGE.
    let sections_in_page2 = text_sect.offset as u64 >= SHIFT;
    let new_text_off = if sections_in_page2 {
        text_sect.offset as u64 - SHIFT
    } else {
        text_sect.offset as u64
    };
    let new_const_off = if sections_in_page2 {
        const_sect.offset as u64 - SHIFT
    } else {
        const_sect.offset as u64
    };
    let text_content_end = new_const_off + const_sect.size;
    // Sanity: load commands must not overlap the new __text location,
    // and the whole __TEXT content must fit in one page.
    let header_and_lc = 32 + r_u32(&buf, 20) as u64; // mach header + sizeofcmds
    if new_text_off < header_and_lc {
        die(format!(
            "packed __text offset {} overlaps load commands (end at {})",
            new_text_off, header_and_lc));
    }
    if text_content_end > PAGE {
        // Content is marginally over the page boundary — can't compact.
        // Pass through; the binary runs fine at 2 pages.
        passthrough(&buf, input_path, output_path,
            &format!("content {text_content_end} B > {PAGE} B, can't compact"));
        return;
    }
    // Also sanity-check the section alignments: -SHIFT preserves all
    // low bits below 0x4000, so the section's align relative to its
    // own page carries over unchanged.
    debug_assert_eq!(new_text_off & ((1u64 << text_sect.align) - 1), 0);
    debug_assert_eq!(new_const_off & ((1u64 << const_sect.align) - 1), 0);

    // Read code-signature command (we'll regenerate the signature).
    let mut codesig_lc_off: Option<usize> = None;
    let mut codesig_dataoff_orig: u64 = 0;
    {
        let ncmds = r_u32(&buf, 16) as usize;
        let mut off = 32;
        for _ in 0..ncmds {
            let cmd = r_u32(&buf, off);
            let cmdsize = r_u32(&buf, off + 4) as usize;
            if cmd == LC_CODE_SIGNATURE {
                codesig_lc_off = Some(off);
                codesig_dataoff_orig = r_u32(&buf, off + 8) as u64;
                break;
            }
            off += cmdsize;
        }
    }
    let codesig_lc_off = codesig_lc_off
        .unwrap_or_else(|| die("no LC_CODE_SIGNATURE in input"));

    // ── Compute new layout ─────────────────────────────────────────────────
    let new_text_vmsize = PAGE;
    let new_text_filesize = text_content_end; // ≤ PAGE; unused tail zerofills

    let old_data = &segs[data_idx];
    let new_data_vmaddr = old_data.vmaddr - SHIFT;
    let new_data_fileoff = PAGE; // always page-aligned

    let old_le = &segs[linkedit_idx];
    let new_le_vmaddr = old_le.vmaddr - SHIFT;
    let new_le_fileoff = PAGE + old_data.filesize; // same value if data.filesize==0

    // __LINKEDIT non-signature payload: everything between old_le.fileoff
    // and the start of the code signature.
    let le_prefix_len = codesig_dataoff_orig - old_le.fileoff;
    let le_prefix_new_end = new_le_fileoff + le_prefix_len;

    // Old LINKEDIT offsets that point into the prefix region get shifted
    // by (new_le_fileoff - old_le.fileoff). We'll apply this to LC_SYMTAB,
    // LC_DYSYMTAB, LC_DYLD_INFO_ONLY, and LC_FUNCTION_STARTS (if present).
    let le_shift: i64 = new_le_fileoff as i64 - old_le.fileoff as i64;

    // ── Build output buffer ────────────────────────────────────────────────
    // Copy the first (header + load commands) bytes from input; we'll
    // mutate the load commands in place and re-emit with the packed layout.
    let mut out = vec![0u8; new_le_fileoff as usize]; // page 1 zeroed
    out[..header_and_lc as usize].copy_from_slice(&buf[..header_and_lc as usize]);
    // __text and __const sections pasted at their new offsets.
    out[new_text_off as usize..(new_text_off + text_sect.size) as usize]
        .copy_from_slice(&buf[text_sect.offset as usize
            ..(text_sect.offset as u64 + text_sect.size) as usize]);
    out[new_const_off as usize..(new_const_off + const_sect.size) as usize]
        .copy_from_slice(&buf[const_sect.offset as usize
            ..(const_sect.offset as u64 + const_sect.size) as usize]);
    // __LINKEDIT prefix (pre-signature) appended at new_le_fileoff.
    let old_le_prefix = &buf[old_le.fileoff as usize
        ..(old_le.fileoff + le_prefix_len) as usize];
    out.extend_from_slice(old_le_prefix);
    debug_assert_eq!(out.len() as u64, le_prefix_new_end);

    // ── Rewrite load commands ──────────────────────────────────────────────
    // __TEXT: vmsize, filesize both shrink to PAGE (vmsize) and
    // text_content_end (filesize ≤ PAGE, remainder zerofills).
    {
        let s = &segs[text_idx];
        w_u64(&mut out, s.lc_off + 32, new_text_vmsize);  // vmsize
        w_u64(&mut out, s.lc_off + 48, new_text_filesize); // filesize
        // __text section
        write_section(&mut out, s.sect_offs[0],
            text.vmaddr + new_text_off,      // addr
            new_text_off as u32);             // offset
        // __const section
        write_section(&mut out, s.sect_offs[1],
            text.vmaddr + new_const_off,
            new_const_off as u32);
    }
    // __DATA: vmaddr shifts by -SHIFT. fileoff = PAGE. Sections' vmaddrs
    // all shift by -SHIFT (they sit inside __DATA; their addr is relative
    // to nothing, it's absolute). Their file offsets — which for __bss
    // is 0 (zerofill) — don't need updating.
    {
        let s = &segs[data_idx];
        w_u64(&mut out, s.lc_off + 24, new_data_vmaddr);
        w_u64(&mut out, s.lc_off + 40, new_data_fileoff);
        for &so in &s.sect_offs {
            let sec = read_section(&buf, so);
            // addr moves; offset stays 0 for zerofill.
            write_section(&mut out, so, sec.addr - SHIFT, sec.offset);
        }
    }
    // __LINKEDIT: vmaddr shifts, fileoff changes.
    {
        let s = &segs[linkedit_idx];
        w_u64(&mut out, s.lc_off + 24, new_le_vmaddr);
        w_u64(&mut out, s.lc_off + 40, new_le_fileoff);
    }

    // Shift LINKEDIT-internal offsets in other LCs.
    {
        let ncmds = r_u32(&out, 16) as usize;
        let mut off = 32;
        for _ in 0..ncmds {
            let cmd = r_u32(&out, off);
            let cmdsize = r_u32(&out, off + 4) as usize;
            match cmd {
                LC_SYMTAB => {
                    let symoff = r_u32(&out, off + 8) as i64 + le_shift;
                    let stroff = r_u32(&out, off + 16) as i64 + le_shift;
                    w_u32(&mut out, off + 8, symoff as u32);
                    w_u32(&mut out, off + 16, stroff as u32);
                }
                LC_DYSYMTAB => {
                    // Offsets at +32, +40, +48, +56, +64, +72 are all file
                    // offsets into __LINKEDIT (tocoff, modtaboff,
                    // extrefsymoff, indirectsymoff, extreloff, locreloff).
                    // Each is followed by a count, so they're 8 bytes apart.
                    for field_off in [32, 40, 48, 56, 64, 72] {
                        let v = r_u32(&out, off + field_off);
                        if v != 0 {
                            w_u32(&mut out, off + field_off,
                                (v as i64 + le_shift) as u32);
                        }
                    }
                }
                LC_DYLD_INFO_ONLY => {
                    // rebase_off, bind_off, weak_bind_off, lazy_bind_off,
                    // export_off at +8, +16, +24, +32, +40.
                    for field_off in [8, 16, 24, 32, 40] {
                        let v = r_u32(&out, off + field_off);
                        if v != 0 {
                            w_u32(&mut out, off + field_off,
                                (v as i64 + le_shift) as u32);
                        }
                    }
                }
                LC_MAIN => {
                    // entryoff at +8 (u64): file offset of main() from
                    // __TEXT segment start. Only adjust when sections
                    // were in page 2 and are being shifted down by SHIFT.
                    if sections_in_page2 {
                        let entryoff = r_u64(&out, off + 8);
                        w_u64(&mut out, off + 8, entryoff - SHIFT);
                    }
                }
                LC_FUNCTION_STARTS
                | LC_DATA_IN_CODE
                | LC_DYLD_EXPORTS_TRIE
                | LC_DYLD_CHAINED_FIXUPS => {
                    // linkedit_data_command: dataoff, datasize (both u32)
                    // at +8, +12. Shift dataoff by the LINKEDIT delta.
                    let dataoff = r_u32(&out, off + 8);
                    if dataoff != 0 {
                        w_u32(&mut out, off + 8, (dataoff as i64 + le_shift) as u32);
                    }
                }
                _ => {}
            }
            off += cmdsize;
        }
    }

    // ── Compute code-signature placement + size ────────────────────────────
    // Signature starts at le_prefix_new_end (right after the non-signature
    // __LINKEDIT payload). Its size depends on codeLimit, which equals the
    // signature's start file offset.
    let code_limit = le_prefix_new_end;
    let id = "ltop\0".trim_end_matches('\0');
    let sig_len = sig_size(code_limit, id);

    // Update LC_CODE_SIGNATURE fields.
    w_u32(&mut out, codesig_lc_off + 8, code_limit as u32);
    w_u32(&mut out, codesig_lc_off + 12, sig_len as u32);

    // Update __LINKEDIT filesize now that we know the signature size.
    let new_le_filesize = (code_limit - new_le_fileoff) + sig_len as u64;
    w_u64(&mut out, segs[linkedit_idx].lc_off + 48, new_le_filesize);

    // Pad the output up to code_limit (should already be there — le prefix
    // ended at le_prefix_new_end == code_limit).
    assert_eq!(out.len() as u64, code_limit);

    // ── Sign ───────────────────────────────────────────────────────────────
    // Fetch final __TEXT fileoff + filesize for execSegBase/Limit.
    let text_file_off = 0u64;
    let text_file_size = new_text_filesize;
    let mut sig = Vec::with_capacity(sig_len);
    sign_adhoc(&mut sig, &out, id, code_limit, text_file_off, text_file_size);
    assert_eq!(sig.len(), sig_len, "emitted signature length mismatch");
    out.extend_from_slice(&sig);

    // ── Write output ───────────────────────────────────────────────────────
    fs::write(output_path, &out)
        .unwrap_or_else(|e| die(format!("write {}: {}", output_path, e)));

    // Copy input permissions to output (preserves executable bit).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perm = fs::metadata(input_path)
            .map(|m| m.permissions())
            .unwrap_or_else(|e| die(format!("stat {}: {}", input_path, e)));
        let mode = perm.mode();
        fs::set_permissions(output_path, fs::Permissions::from_mode(mode))
            .unwrap_or_else(|e| die(format!("chmod {}: {}", output_path, e)));
    }

    let in_size = buf.len();
    let out_size = out.len();
    let _ = &mut buf; // keep until end (so slices in `out` copies are valid)
    eprintln!(
        "mac-pack: {} B → {} B (saved {} B)",
        in_size, out_size, in_size as i64 - out_size as i64
    );
}
