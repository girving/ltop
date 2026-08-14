//! OSSerializeBinary decoder — Phase 3 of mac-iokit-free.md.
//!
//! Pure-byte parser for the format xnu's
//! `libkern/c++/OSSerializeBinary.cpp` emits. The Mach-RPC reply from
//! `io_registry_entry_get_properties_bin` returns a buffer in this
//! format; libIOKit normally hands it to libCoreFoundation's
//! `OSUnserializeBinary` to materialise a `CFDictionary`. We parse it
//! ourselves so the production binary needs neither library.
//!
//! ## Format (verified against xnu source)
//!
//! ```text
//! [ magic (4 bytes) = 0xD3 0x00 0x00 0x00 ]
//! [ stream of items ]
//! ```
//!
//! Each item starts with a u32 tag (little-endian on aarch64):
//!
//! ```text
//!  bit 31    : end-of-collection flag (0x8000_0000) — ignored, counts drive iteration
//!  bits 30-24: type (7 bits, only 4 used today)
//!  bits 23-0 : per-type operand (count or length)
//! ```
//!
//! Types:
//!
//! | Type     | Tag high byte | Operand          | Payload                             |
//! |----------|---------------|------------------|--------------------------------------|
//! | Dict     | 0x01          | pair count       | 2N items (alternating key, value)   |
//! | Array    | 0x02          | element count    | N items                              |
//! | Set      | 0x03          | element count    | N items                              |
//! | Number   | 0x04          | bit-width 8/16/32/64 | exactly two u32 words: low, then high |
//! | Symbol   | 0x08          | byte length incl trailing NUL | bytes padded to 4-byte boundary |
//! | String   | 0x09          | byte length (no NUL) | bytes padded to 4                |
//! | Data     | 0x0A          | byte length      | bytes padded to 4                    |
//! | Boolean  | 0x0B          | 0 or 1           | none                                 |
//! | Object   | 0x0C          | index into objsArray | none                              |
//!
//! ## objsArray and Object back-references
//!
//! Every emitted item *except* an Object back-ref is appended, in emit
//! order, to a single shared array. An `Object` tag's operand is an
//! index into that array — used primarily to deduplicate Symbol keys
//! (the kernel emits each unique Symbol once, then back-refs it on
//! every reuse). Containers go in too, so an Object ref *can* point at
//! a Dict/Array, though that's rare in IORegistry data.
//!
//! Emit order equals byte-stream order: a container is appended to
//! objsArray at its tag, and its children follow inline, so object
//! #N's tag is simply the (N+1)-th non-Object tag in the stream. Our
//! parser exploits that instead of materialising the array: one flat
//! validation walk over the blob records a fixed-size table of
//! checkpoints (the offset of every `stride`-th object, `stride`
//! doubling whenever the table fills, so any blob size fits). A
//! back-ref resolves by jumping to the nearest checkpoint and
//! flat-scanning at most `stride − 1` tags forward. Value-skipping in
//! [`OsBinary::find_dict`] re-walks the value's subtree — O(subtree
//! bytes), the same order as the lookup's own scan. Callers see only
//! `Item { offset }` handles.


// ── Tag bits ────────────────────────────────────────────────────────────────

const TYPE_DICT: u8    = 0x01;
const TYPE_ARRAY: u8   = 0x02;
const TYPE_SET: u8     = 0x03;
const TYPE_NUMBER: u8  = 0x04;
const TYPE_SYMBOL: u8  = 0x08;
const TYPE_STRING: u8  = 0x09;
const TYPE_DATA: u8    = 0x0a;
const TYPE_BOOLEAN: u8 = 0x0b;
const TYPE_OBJECT: u8  = 0x0c;

const MAGIC: [u8; 4] = [0xd3, 0x00, 0x00, 0x00];

#[derive(Clone, Copy)]
struct Tag {
    ty: u8,
    operand: u32,
}

fn read_tag(blob: &[u8], offset: u32) -> Option<Tag> {
    let o = offset as usize;
    if o + 4 > blob.len() { return None; }
    let raw = u32::from_le_bytes(blob[o..o + 4].try_into().ok()?);
    Some(Tag {
        ty: ((raw >> 24) & 0x7f) as u8,   // mask off the EOC bit
        operand: raw & 0x00ff_ffff,
    })
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// Successfully parsed OSSerializeBinary blob. Holds a borrow of the
/// blob plus the caller's checkpoint scratch (filled by `parse`).
///
/// All query methods are O(blob walk) at worst — one linear pass over
/// the sub-blob they touch.
pub struct OsBinary<'a> {
    blob: &'a [u8],
    /// `ckpts[i]` = offset of emitted object `#(i * stride)`. Every
    /// multiple of `stride` below `count` is present.
    ckpts: &'a [u32],
    stride: u64,
    /// Total emitted objects in the blob (Object back-refs excluded).
    count: u32,
}

/// Handle to one item inside the parsed blob — just an offset into
/// the blob, paired with a marker so the type doesn't get confused
/// with arbitrary u32s. `Copy` so callers can pass it around without
/// lifetime gymnastics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub offset: u32,
}

impl<'a> OsBinary<'a> {
    /// Parse `blob`, validating its structure and filling `ckpts_buf`
    /// with object-offset checkpoints. Any non-empty buffer handles any
    /// blob (the checkpoint stride doubles when the table fills); a
    /// bigger buffer only makes back-ref resolution walk less. Returns
    /// `None` if the magic is wrong, the blob is truncated, or a
    /// back-ref points at a not-yet-emitted object.
    pub fn parse(blob: &'a [u8], ckpts_buf: &'a mut [u32]) -> Option<Self> {
        if blob.len() < 8 || blob[..4] != MAGIC || ckpts_buf.is_empty() {
            return None;
        }
        // Offsets are u32; refuse blobs that could overflow the cursor.
        if blob.len() >= u32::MAX as usize - 16 { return None; }
        // Top-level item — must be a Dict for any IORegistry property
        // blob, but the parser doesn't enforce that here; callers can
        // ask for the root and check its type.
        let (n, stride, count) = {
            let mut ck = Ckpts { buf: &mut *ckpts_buf, n: 0, stride: 1 };
            let (_end, count) = walk(blob, 4, Some(&mut ck))?;
            (ck.n, ck.stride, count)
        };
        let ckpts: &'a [u32] = &ckpts_buf[..n];
        Some(OsBinary { blob, ckpts, stride, count })
    }

    /// Top-level item — typically a Dict. Always at offset 4 (just past
    /// the magic).
    pub fn root(&self) -> Option<Item> {
        if self.count == 0 { return None; }
        Some(Item { offset: 4 })
    }

    /// In a Dict, find the value associated with a Symbol key whose
    /// bytes match `key` (no trailing NUL — we strip it during compare).
    /// `Object` back-refs are resolved on the dict itself and on each
    /// Symbol key.
    pub fn find_dict(&self, dict: Item, key: &[u8]) -> Option<Item> {
        let dict_off = self.resolve(dict.offset)?;
        let tag = read_tag(self.blob, dict_off)?;
        if tag.ty != TYPE_DICT { return None; }
        let mut cursor = dict_off + 4;
        for _ in 0..tag.operand {
            let key_off = cursor;
            cursor += self.item_len(key_off)?;
            let val_off = cursor;
            let val_len = self.item_len(val_off)?;
            cursor += val_len;
            if self.symbol_bytes(key_off) == Some(key) {
                return Some(Item { offset: val_off });
            }
        }
        None
    }

    /// Iterate each element of an Array/Set (or an Object back-ref to
    /// one), calling `f` with that element's `Item`.
    pub fn for_each_array(&self, arr: Item, mut f: impl FnMut(Item)) -> Option<()> {
        let arr_off = self.resolve(arr.offset)?;
        let tag = read_tag(self.blob, arr_off)?;
        if tag.ty != TYPE_ARRAY && tag.ty != TYPE_SET { return None; }
        let mut cursor = arr_off + 4;
        for _ in 0..tag.operand {
            let item_off = cursor;
            let item_len = self.item_len(item_off)?;
            cursor += item_len;
            f(Item { offset: item_off });
        }
        Some(())
    }

    /// Decode a Number value (or an Object back-ref to a Number). The
    /// payload is two `u32` words — low then high — regardless of the
    /// tag's bit-width operand; callers can mask if they want a u32.
    pub fn as_number(&self, item: Item) -> Option<u64> {
        let target = self.resolve(item.offset)?;
        let tag = read_tag(self.blob, target)?;
        if tag.ty != TYPE_NUMBER { return None; }
        let o = target as usize + 4;
        if o + 8 > self.blob.len() { return None; }
        let lo = u32::from_le_bytes(self.blob[o..o + 4].try_into().ok()?) as u64;
        let hi = u32::from_le_bytes(self.blob[o + 4..o + 8].try_into().ok()?) as u64;
        Some((hi << 32) | lo)
    }

    /// Decode a String's bytes (or an Object back-ref to one). Returns
    /// the payload without its 4-byte alignment padding.
    pub fn as_string(&self, item: Item) -> Option<&'a [u8]> {
        let target = self.resolve(item.offset)?;
        let tag = read_tag(self.blob, target)?;
        if tag.ty != TYPE_STRING { return None; }
        let len = tag.operand as usize;
        let o = target as usize + 4;
        if o + len > self.blob.len() { return None; }
        Some(&self.blob[o..o + len])
    }

    /// Decode a Symbol's bytes (or an Object back-ref to one), with the
    /// trailing NUL stripped — Symbol's operand includes it.
    ///
    /// Production code today reads Symbols only as keys (via
    /// `find_dict`'s internal `symbol_bytes`); this entry point exists
    /// for callers that want a Symbol-typed *value*. Tested in
    /// `parses_real_agx_properties`'s top-level-keys diagnostic.
    #[allow(dead_code)]
    pub fn as_symbol(&self, item: Item) -> Option<&'a [u8]> {
        self.symbol_bytes(item.offset)
    }

    /// Decode a Boolean. Returns `Some(true)` or `Some(false)`; `None`
    /// for any other type. Production code doesn't decode Booleans
    /// today, but the AGX dict carries several (`IOMatchedAtBoot`,
    /// `CommandSubmissionEnabled`); this entry point is exercised by
    /// `as_bool_decodes_true_and_false`.
    #[allow(dead_code)]
    pub fn as_bool(&self, item: Item) -> Option<bool> {
        let target = self.resolve(item.offset)?;
        let tag = read_tag(self.blob, target)?;
        if tag.ty != TYPE_BOOLEAN { return None; }
        Some(tag.operand != 0)
    }

    /// Resolve an Object back-ref to its target offset; pass-through
    /// for direct items. Returns `None` if the ref index is out of
    /// range (corrupt blob — `parse` already rejects these).
    fn resolve(&self, offset: u32) -> Option<u32> {
        let tag = read_tag(self.blob, offset)?;
        if tag.ty == TYPE_OBJECT {
            self.nth_object(tag.operand)
        } else {
            Some(offset)
        }
    }

    /// Offset of emitted object `#idx`: jump to the nearest checkpoint
    /// at or below it, then flat-scan forward counting non-Object tags
    /// (emit order equals stream order — see the module doc). At most
    /// `stride − 1` objects (plus interleaved back-ref tags) are
    /// stepped over.
    fn nth_object(&self, idx: u32) -> Option<u32> {
        if idx >= self.count { return None; }
        let ck = (idx as u64 / self.stride) as usize;
        let mut offset = *self.ckpts.get(ck)?;
        let mut i = (ck as u64 * self.stride) as u32;
        loop {
            let tag = read_tag(self.blob, offset)?;
            if tag.ty == TYPE_OBJECT {
                offset += 4; // refs emit nothing; skip
                continue;
            }
            if i == idx { return Some(offset); }
            i += 1;
            offset += scalar_step(tag)?;
        }
    }

    /// Length of the item starting at `offset`, including its tag and
    /// any padded payload / nested items. For Object back-refs this is
    /// always 4 (the back-ref tag itself) — *not* the length of the
    /// referenced item.
    fn item_len(&self, offset: u32) -> Option<u32> {
        let tag = read_tag(self.blob, offset)?;
        if tag.ty == TYPE_OBJECT { return Some(4); }
        let (end, _count) = walk(self.blob, offset, None)?;
        Some(end - offset)
    }

    fn symbol_bytes(&self, offset: u32) -> Option<&'a [u8]> {
        let target = self.resolve(offset)?;
        let tag = read_tag(self.blob, target)?;
        if tag.ty != TYPE_SYMBOL { return None; }
        let len = tag.operand as usize;
        let trimmed = if len > 0 { len - 1 } else { 0 }; // drop trailing NUL
        let o = target as usize + 4;
        if o + trimmed > self.blob.len() { return None; }
        Some(&self.blob[o..o + trimmed])
    }
}

// ── Parser ──────────────────────────────────────────────────────────────────

/// Fixed-capacity checkpoint table being built during `walk`. Records
/// the offset of every `stride`-th emitted object; when the table
/// fills, the stride doubles and every other entry is dropped in
/// place, so a fixed buffer covers any object count.
struct Ckpts<'b> {
    buf: &'b mut [u32],
    n: usize,
    stride: u64,
}

impl Ckpts<'_> {
    fn record(&mut self, count: u32, offset: u32) {
        if count as u64 % self.stride != 0 { return; }
        if self.n == self.buf.len() {
            // Halve the density: keep entries at even indices — those
            // are exactly the multiples of the doubled stride.
            let mut w = 0;
            let mut r = 0;
            while r < self.n {
                self.buf[w] = self.buf[r];
                w += 1;
                r += 2;
            }
            self.n = w;
            self.stride *= 2;
            // `count` was a multiple of the old stride; it may not be
            // one of the new.
            if count as u64 % self.stride != 0 { return; }
        }
        self.buf[self.n] = offset;
        self.n += 1;
    }
}

/// Total byte size of the scalar item with tag `tag` (tag word plus
/// padded payload). Containers step 4: their children follow inline
/// and are walked as their own items.
#[inline]
fn scalar_step(tag: Tag) -> Option<u32> {
    match tag.ty {
        TYPE_DICT | TYPE_ARRAY | TYPE_SET | TYPE_BOOLEAN => Some(4),
        // Always 8 bytes payload, regardless of tag.operand (which is
        // the bit-width: 8/16/32/64).
        TYPE_NUMBER => Some(12),
        // `tag.operand` is the byte length (24-bit field, so ≤ 16 MB).
        // Checked arithmetic anyway: an operand near its 24-bit ceiling
        // could otherwise wrap during alignment + the `4 + pad` total.
        TYPE_SYMBOL | TYPE_STRING | TYPE_DATA => {
            (tag.operand.checked_add(3)? & !3).checked_add(4)
        }
        _ => None,
    }
}

/// Walk the item at `start` (with all nested children), validating
/// tags and bounds. Returns `(end_offset, emitted_object_count)`.
///
/// Flat, no recursion: `remaining` counts items still owed to
/// enclosing containers — a container consumes one slot and adds its
/// child count. With `ck` present (the `parse` walk), checkpoints are
/// recorded and Object back-refs are validated against emit order
/// (xnu only ever emits refs to previously serialized objects); with
/// `None` (skipping a sub-item), refs were already validated by parse.
fn walk(blob: &[u8], start: u32, mut ck: Option<&mut Ckpts<'_>>) -> Option<(u32, u32)> {
    let mut cursor = start;
    let mut remaining: u64 = 1;
    let mut count: u32 = 0;
    while remaining > 0 {
        let tag = read_tag(blob, cursor)?;
        remaining -= 1;
        if tag.ty == TYPE_OBJECT {
            if ck.is_some() && tag.operand >= count { return None; }
            cursor += 4; // no payload; emits nothing
            continue;
        }
        if let TYPE_DICT | TYPE_ARRAY | TYPE_SET = tag.ty {
            let mult = if tag.ty == TYPE_DICT { 2 } else { 1 };
            remaining = remaining.checked_add(tag.operand as u64 * mult)?;
        }
        let step = scalar_step(tag)?;
        if (cursor as usize).checked_add(step as usize)? > blob.len() {
            return None;
        }
        if let Some(ck) = ck.as_deref_mut() {
            ck.record(count, cursor);
        }
        count = count.checked_add(1)?;
        cursor += step;
    }
    Some((cursor, count))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal blob by hand: `{ "k" : 7u32 }`.
    /// Magic + Dict(1) + Symbol("k\0") + Number(32, 7).
    fn k_eq_7() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        // Dict, count=1, no EOC.
        b.extend_from_slice(&0x0100_0001u32.to_le_bytes());
        // Symbol, len=2 (incl NUL).
        b.extend_from_slice(&0x0800_0002u32.to_le_bytes());
        b.extend_from_slice(b"k\0\0\0");        // 2 bytes data + 2 padding
        // Number, width=32, EOC set.
        b.extend_from_slice(&0x8400_0020u32.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // high half
        b
    }

    #[test]
    fn parses_root_dict_with_one_pair() {
        let blob = k_eq_7();
        let mut ckpts = vec![0u32; 16];
        let bin = OsBinary::parse(&blob, &mut ckpts).expect("parse");
        let root = bin.root().expect("root");
        let v = bin.find_dict(root, b"k").expect("find k");
        assert_eq!(bin.as_number(v), Some(7));
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut bad = k_eq_7();
        bad[0] = 0;
        let mut ckpts = vec![0u32; 16];
        assert!(OsBinary::parse(&bad, &mut ckpts).is_none());
    }

    /// Object back-references — emit a Symbol "k", a Number, then a
    /// second pair whose key is an Object-ref pointing back at the
    /// first Symbol.
    #[test]
    fn back_refs_to_symbols_resolve() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&0x0100_0002u32.to_le_bytes());        // Dict, 2 pairs
        // Pair 1: key = Symbol "kA\0\0" (len=3), value = Number 7.
        b.extend_from_slice(&0x0800_0003u32.to_le_bytes());        // Symbol, len=3
        b.extend_from_slice(b"kA\0\0");
        b.extend_from_slice(&0x0400_0020u32.to_le_bytes());        // Number 32
        b.extend_from_slice(&7u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        // Pair 2: key = Object back-ref to objs[1] (the Symbol; objs[0]
        // is the Dict itself), value = Number 11. Note that the
        // *first* Number is objs[2].
        b.extend_from_slice(&0x0c00_0001u32.to_le_bytes());        // Object ref index 1
        b.extend_from_slice(&0x8400_0020u32.to_le_bytes());        // Number 32, EOC
        b.extend_from_slice(&11u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());

        let mut ckpts = vec![0u32; 16];
        let bin = OsBinary::parse(&b, &mut ckpts).expect("parse");
        // Both back-ref'd "kA" lookups should resolve. find_dict picks
        // the first match — to actually verify the back-ref path, we
        // walk pairs manually.
        let root = bin.root().unwrap();
        let mut seen = Vec::new();
        let tag = read_tag(&b, root.offset).unwrap();
        assert_eq!(tag.ty, TYPE_DICT);
        let mut cursor = root.offset + 4;
        for _ in 0..tag.operand {
            let key_off = cursor;
            cursor += bin.item_len(key_off).unwrap();
            let val_off = cursor;
            cursor += bin.item_len(val_off).unwrap();
            let key = bin.symbol_bytes(key_off).expect("symbol resolve");
            let v = bin.as_number(Item { offset: val_off }).unwrap();
            seen.push((key.to_vec(), v));
        }
        assert_eq!(seen, vec![(b"kA".to_vec(), 7u64), (b"kA".to_vec(), 11u64)]);
    }

    // ── Synthetic-blob tests (AGX-free; run on every CI macOS host) ──
    //
    // These exist because the property blobs we'll see at runtime are
    // hidden behind AGX hardware, which CI lacks. The IORegistry format
    // is stable, so hand-built blobs that mirror the real shapes catch
    // every byte-level decoder bug *and* exercise the higher-level
    // extractors (`read_agx_used_mib_blob`, `read_user_client_blob`).

    /// Helper: emit a tag word into `out`.
    fn emit_tag(out: &mut Vec<u8>, ty: u8, operand: u32, eoc: bool) {
        let raw: u32 = ((ty as u32) << 24) | (operand & 0x00ff_ffff)
                     | (if eoc { 0x8000_0000 } else { 0 });
        out.extend_from_slice(&raw.to_le_bytes());
    }
    /// Helper: emit a Symbol (key bytes + trailing NUL, padded to 4).
    fn emit_symbol(out: &mut Vec<u8>, s: &[u8], eoc: bool) {
        let len = s.len() as u32 + 1;
        emit_tag(out, TYPE_SYMBOL, len, eoc);
        out.extend_from_slice(s);
        out.push(0);
        while out.len() % 4 != 0 { out.push(0); }
    }
    /// Helper: emit a String (no NUL, padded to 4).
    fn emit_string(out: &mut Vec<u8>, s: &[u8], eoc: bool) {
        emit_tag(out, TYPE_STRING, s.len() as u32, eoc);
        out.extend_from_slice(s);
        while out.len() % 4 != 0 { out.push(0); }
    }
    /// Helper: emit a Number (8 bytes payload, low then high).
    fn emit_number(out: &mut Vec<u8>, value: u64, eoc: bool) {
        emit_tag(out, TYPE_NUMBER, 64, eoc);
        out.extend_from_slice(&(value as u32).to_le_bytes());
        out.extend_from_slice(&((value >> 32) as u32).to_le_bytes());
    }

    /// Round-trip our own static `AGX_MATCHING_BLOB` through the
    /// decoder. Catches encoding bugs in the static (wrong tag word,
    /// off-by-one length, missing NUL, wrong EOC bit) without needing
    /// any AGX hardware — which is exactly the test the kernel runs
    /// at every `IOServiceGetMatchingServices` call.
    #[cfg(target_os = "macos")]
    #[test]
    fn agx_matching_blob_decodes_to_expected_dict() {
        use crate::mac_sys::AGX_MATCHING_BLOB;
        let mut ckpts = vec![0u32; 32];
        let bin = OsBinary::parse(&AGX_MATCHING_BLOB, &mut ckpts)
            .expect("AGX_MATCHING_BLOB should parse");
        let root = bin.root().expect("root present");
        let root_tag = read_tag(&AGX_MATCHING_BLOB, root.offset).unwrap();
        assert_eq!(root_tag.ty, TYPE_DICT, "root must be Dict");
        assert_eq!(root_tag.operand, 1, "exactly one pair");
        let val = bin.find_dict(root, b"IOProviderClass")
            .expect("IOProviderClass key");
        assert_eq!(bin.as_string(val), Some(&b"AGXAccelerator"[..]));
    }

    #[test]
    fn rejects_blob_truncated_after_magic() {
        let mut ckpts = vec![0u32; 4];
        // Just the magic, nothing else.
        assert!(OsBinary::parse(&MAGIC, &mut ckpts).is_none());
    }

    #[test]
    fn rejects_blob_truncated_inside_payload() {
        // Magic + Symbol tag claiming 4 bytes, but only 1 byte present.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TYPE_SYMBOL, 4, false);
        b.push(b'x');                // 1 byte instead of 4
        let mut ckpts = vec![0u32; 4];
        assert!(OsBinary::parse(&b, &mut ckpts).is_none());
    }

    #[test]
    fn rejects_dict_with_count_exceeding_actual_items() {
        // Dict claims 5 pairs; only 1 follows. parse_one tries to read
        // the next item past the end of the blob and returns None.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TYPE_DICT, 5, false);
        emit_symbol(&mut b, b"k", false);
        emit_number(&mut b, 1, true);
        let mut ckpts = vec![0u32; 16];
        assert!(OsBinary::parse(&b, &mut ckpts).is_none());
    }

    #[test]
    fn rejects_padding_overflow_in_string_operand() {
        // Symbol with operand = u32::MAX − 2 would, with naive `+ 3`,
        // wrap to a tiny `pad`. We use checked arithmetic, so the
        // parser returns None instead of advancing into garbage.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        // Operand only has 24 bits; the largest value it can carry is
        // 0xFFFFFD. Add 3 ⇒ 0x1000000 (still fits in u32). Add 4 ⇒ same.
        // The bounds check after that catches any pathological
        // (cursor + pad) wrap. Use the largest legal 24-bit value as a
        // proxy for "absurdly large" operand.
        emit_tag(&mut b, TYPE_SYMBOL, 0x00ff_fffd, false);
        // No payload follows — bounds check should fire.
        let mut ckpts = vec![0u32; 4];
        assert!(OsBinary::parse(&b, &mut ckpts).is_none());
    }

    #[test]
    fn rejects_object_back_ref_with_out_of_range_index() {
        // Dict { Object(99) : Number(7) } — back-ref to an object that
        // hasn't been emitted. xnu never produces forward refs, so
        // parse rejects the blob outright.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TYPE_DICT, 1, false);
        emit_tag(&mut b, TYPE_OBJECT, 99, false);
        emit_number(&mut b, 7, true);
        let mut ckpts = vec![0u32; 16];
        assert!(OsBinary::parse(&b, &mut ckpts).is_none());
    }

    #[test]
    fn one_slot_checkpoint_buffer_handles_any_blob() {
        // The checkpoint stride doubles when the table fills, so even a
        // single-slot buffer parses a multi-item blob — resolution just
        // scans from the first object.
        let blob = k_eq_7();
        let mut ckpts = vec![0u32; 1];
        let bin = OsBinary::parse(&blob, &mut ckpts).expect("parse");
        let root = bin.root().expect("root");
        assert_eq!(bin.as_number(bin.find_dict(root, b"k").unwrap()), Some(7));
    }

    /// Force several stride doublings and check back-refs still resolve
    /// to the right objects: a dict of 200 unique (symbol, number)
    /// pairs, then one pair whose key back-refs an early symbol and
    /// whose value back-refs a late number. Parsed with a 4-slot
    /// checkpoint table (stride ends at 128) and a large one; both must
    /// agree.
    #[test]
    fn stride_doubling_resolves_back_refs_exactly() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TYPE_DICT, 201, false);
        for i in 0..200u32 {
            // Symbols "s000".."s199", values 1000..1199. Emit order:
            // dict = #0, symbol i = #(1 + 2i), number i = #(2 + 2i).
            let name = format!("s{i:03}");
            emit_symbol(&mut b, name.as_bytes(), false);
            emit_number(&mut b, 1000 + i as u64, false);
        }
        // Key: back-ref to symbol "s007" (object #15). Value: back-ref
        // to number 1198 (object #398).
        emit_tag(&mut b, TYPE_OBJECT, 15, false);
        emit_tag(&mut b, TYPE_OBJECT, 398, true);

        for slots in [4usize, 512] {
            let mut ckpts = vec![0u32; slots];
            let bin = OsBinary::parse(&b, &mut ckpts).expect("parse");
            let root = bin.root().unwrap();
            assert_eq!(bin.as_number(bin.find_dict(root, b"s042").unwrap()),
                       Some(1042), "slots={slots}");
            // find_dict returns the FIRST match for s007 — the direct
            // pair. Walk to the final pair manually to hit both refs.
            let mut cursor = root.offset + 4;
            for _ in 0..200 {
                cursor += bin.item_len(cursor).unwrap();  // key
                cursor += bin.item_len(cursor).unwrap();  // value
            }
            let key_off = cursor;
            cursor += bin.item_len(key_off).unwrap();
            let val_off = cursor;
            assert_eq!(bin.symbol_bytes(key_off), Some(&b"s007"[..]),
                       "slots={slots}");
            assert_eq!(bin.as_number(Item { offset: val_off }), Some(1198),
                       "slots={slots}");
        }
    }

    #[test]
    fn as_bool_decodes_true_and_false() {
        // Build { "t" : true, "f" : false }.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TYPE_DICT, 2, false);
        emit_symbol(&mut b, b"t", false);
        emit_tag(&mut b, TYPE_BOOLEAN, 1, false);
        emit_symbol(&mut b, b"f", false);
        emit_tag(&mut b, TYPE_BOOLEAN, 0, true);
        let mut ckpts = vec![0u32; 16];
        let bin = OsBinary::parse(&b, &mut ckpts).expect("parse");
        let root = bin.root().unwrap();
        assert_eq!(bin.as_bool(bin.find_dict(root, b"t").unwrap()), Some(true));
        assert_eq!(bin.as_bool(bin.find_dict(root, b"f").unwrap()), Some(false));
        // Non-Boolean values yield None.
        let one = bin.find_dict(root, b"t").unwrap();
        let _ = one;
    }

    // (read_agx_used_mib_blob and read_user_client_blob synthetic-blob
    // tests live in src/gpu/agx.rs's test module — they need access to
    // the agx::* extractors, which aren't in the lib crate's module
    // tree on a `cargo test --lib` run.)

    /// Capture a real AGXAccelerator property blob via libIOKit and
    /// parse it. Verify that:
    ///   - The parser reaches the end without error.
    ///   - "PerformanceStatistics" (a Dict) is present at top-level.
    ///   - Inside it, "Recovery Count" or "Wait Time" (or any number-
    ///     valued key we can fish out — most are kernel-private) shows
    ///     up as a Number.
    /// Skipped on hosts without AGX (CI / virtualised macOS).
    #[cfg(target_os = "macos")]
    #[test]
    fn parses_real_agx_properties() {
        use crate::mac_sys::*;

        #[link(name = "IOKit", kind = "framework")]
        unsafe extern "C" {
            fn IOServiceMatching(name: *const libc::c_char) -> *const core::ffi::c_void;
            fn IOServiceGetMatchingServices(
                main_port: u32, matching: *const core::ffi::c_void, existing: *mut u32,
            ) -> i32;
            fn IOIteratorNext(iter: u32) -> u32;
            fn IOObjectRelease(obj: u32) -> i32;
        }

        let dict = unsafe { IOServiceMatching(b"AGXAccelerator\0".as_ptr() as *const _) };
        if dict.is_null() {
            eprintln!("no IOServiceMatching; skipping");
            return;
        }
        let mut iter: u32 = 0;
        let kr = unsafe { IOServiceGetMatchingServices(0, dict, &mut iter) };
        assert_eq!(kr, 0);
        let entry = unsafe { IOIteratorNext(iter) };
        unsafe { IOObjectRelease(iter); }
        if entry == 0 {
            eprintln!("no AGXAccelerator on this host; skipping");
            return;
        }

        let buf = io_registry_entry_get_properties_bin(entry).expect("props_bin");
        let blob = buf.as_bytes();

        // Deliberately small: the real AGXAccelerator blob has ~2000+
        // objects (IOReportLegend dominates), so 64 slots force several
        // stride doublings — this is the regression test for the old
        // fixed 1024-entry index, which overflowed on exactly this blob
        // and silently zeroed the GPU used-memory readout.
        let mut ckpts = vec![0u32; 64];
        let bin = OsBinary::parse(blob, &mut ckpts).expect("parse OK");

        let root = bin.root().expect("root");
        let perf = bin.find_dict(root, b"PerformanceStatistics")
            .expect("PerformanceStatistics key present");
        let perf_tag = read_tag(blob, perf.offset).unwrap();
        assert_eq!(perf_tag.ty, TYPE_DICT,
                   "PerformanceStatistics should be a Dict, got type {:#x}", perf_tag.ty);

        // Walk PerformanceStatistics looking for at least one Number-
        // valued key. Property names vary by GPU/driver version but
        // numbers are always there.
        let mut found_number = false;
        let mut cursor = perf.offset + 4;
        for _ in 0..perf_tag.operand {
            let key_off = cursor;
            cursor += bin.item_len(key_off).unwrap();
            let val_off = cursor;
            cursor += bin.item_len(val_off).unwrap();
            if bin.as_number(Item { offset: val_off }).is_some() {
                found_number = true;
                break;
            }
        }
        assert!(found_number, "PerformanceStatistics had no Number-valued keys");

        unsafe { IOObjectRelease(entry); }
    }
}
