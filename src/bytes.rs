//! Integer and byte-slice adapters that implement [`Put`]. Replaces
//! the old `Display`-adapter helpers + the `Str` / `Bytes` wrappers
//! those used to need. The UTF-8 recovery logic that used to live in
//! `Bytes::fmt` now lives in [`twrite::Put for [u8]`][crate::twrite],
//! so callers can just pass a `&[u8]` to `twrite!` directly.
//!
//! These helpers exist mainly to carry width / padding parameters
//! alongside a value; passing a bare `u32` via `twrite!` emits plain
//! decimal digits with no padding.

#![allow(dead_code)]

use crate::twrite::{digit_count, write_u32, Put, TinyWriter};

/// Byte-slice equivalent of `str::split_ascii_whitespace`: iterator
/// over non-empty runs of non-whitespace bytes. Works on raw `&[u8]`
/// so callers don't need to UTF-8-validate `/proc` output first.
pub fn split_ascii_whitespace(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    bytes.split(|&b| b.is_ascii_whitespace()).filter(|s| !s.is_empty())
}

/// Case-insensitive ASCII byte-slice equality. Replaces
/// `<[u8]>::eq_ignore_ascii_case`, whose chunked 16-byte-unrolled
/// path costs ~150 B of `.text` that never executes here — every
/// caller compares against a short literal (≤ 6 bytes). This
/// loop-only version lets both call sites (`ieq` and `icontains`)
/// share one monomorphisation.
pub fn ieq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

/// 32-bit FNV-1a over `s` with ASCII case folded via `| 0x20` (A–Z → a–z;
/// digits and lowercase are fixed points). `const`, so a table of
/// `name_hash(b"literal")` is evaluated at compile time into plain `u32`s —
/// no pointers, so no Mach-O rebase fixups and nothing in `__DATA`.
///
/// Membership in such a table is a fingerprint test, not an exact compare:
/// a non-member collides with an `n`-entry table with probability
/// `n / 2^32` (≈ 10⁻⁸ for our lists), deterministically per name. The
/// callers gate cosmetic filtering only, so that trade buys ~24 B of
/// call-site code plus the literal bytes per entry, against 4 B.
pub const fn name_hash(s: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < s.len() {
        h = (h ^ (s[i] | 0x20) as u32).wrapping_mul(0x0100_0193);
        i += 1;
    }
    h
}

/// Byte-slice equivalent of `str::lines` (without the `\r\n`
/// normalization — /proc doesn't emit CR). Splits on `\n`.
pub fn split_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    bytes.split(|&b| b == b'\n')
}

// ── integer-with-width adapters ────────────────────────────────────────────
//
// Rust's default `Display` for integers routes through
// `Formatter::pad_integral` (~470 B) even when you don't use any
// format specifier. These adapters write digits + pad bytes through
// the writer's primitive `put_*` calls directly.

pub struct U32d(pub u32);
pub struct PadRight(pub u32, pub u8);
pub struct PadLeft(pub u32, pub u8);
pub struct PadZero(pub u32, pub u8);

impl Put for U32d {
    #[inline] fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) { write_u32(w, self.0) }
}
impl Put for PadRight {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        for _ in digit_count(self.0)..self.1 { w.put_byte(b' '); }
        write_u32(w, self.0);
    }
}
impl Put for PadLeft {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        write_u32(w, self.0);
        for _ in digit_count(self.0)..self.1 { w.put_byte(b' '); }
    }
}
impl Put for PadZero {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        for _ in digit_count(self.0)..self.1 { w.put_byte(b'0'); }
        write_u32(w, self.0);
    }
}

/// Format `n` as plain decimal, no padding. Replaces `write!(f, "{}", n)`.
#[inline] pub fn u32d(n: u32) -> U32d { U32d(n) }

/// Right-aligned decimal in a `width`-byte field, padded left with spaces.
#[inline] pub fn pad_right(n: u32, width: u8) -> PadRight { PadRight(n, width) }

/// Left-aligned decimal in a `width`-byte field, padded right with spaces.
#[inline] pub fn pad_left(n: u32, width: u8) -> PadLeft { PadLeft(n, width) }

/// Right-aligned decimal in a `width`-byte field, padded left with `'0'`.
#[inline] pub fn pad_zero(n: u32, width: u8) -> PadZero { PadZero(n, width) }

// ── 1-decimal and 2-decimal float adapters ─────────────────────────────────
//
// Avoid `core::fmt::float` (Grisu + Dragon + cache, ~10 KB). Callers
// always have integer inputs biased by 10 / 100, so we round via
// multiplication and emit `int.frac` digit-wise.

/// `X.Y` (one-decimal) non-negative float, baked into exactly `width`
/// chars. The integer part is right-justified to `width - 2` so the
/// whole output fills `width` without the caller asking for padding.
pub struct F1Wide(pub f64, pub u8);
/// `X.YZ` (two-decimal) non-negative float, no padding. Used for the
/// three load-average numbers in the status header.
pub struct F2(pub f64);

impl Put for F1Wide {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let tenths = (self.0 * 10.0 + 0.5) as u64;
        let int_w = self.1.saturating_sub(2);
        PadRight((tenths / 10) as u32, int_w).put(w);
        w.put_byte(b'.');
        U32d((tenths % 10) as u32).put(w);
    }
}
impl Put for F2 {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let h = (self.0 * 100.0 + 0.5) as u64;
        U32d((h / 100) as u32).put(w);
        w.put_byte(b'.');
        PadZero((h % 100) as u32, 2).put(w);
    }
}

#[inline] pub fn f1_wide(x: f64, width: u8) -> F1Wide { F1Wide(x, width) }
#[inline] pub fn f2(x: f64) -> F2 { F2(x) }

// ── char-repeat adapter ────────────────────────────────────────────────────

/// Writes `c` `n` times — no intermediate String. Used for the
/// horizontal separator between sections of the rendered frame.
pub struct Repeat(pub char, pub usize);

impl Put for Repeat {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let mut buf = [0u8; 4];
        let n = self.0.encode_utf8(&mut buf).len();
        let s = &buf[..n];
        for _ in 0..self.1 { w.put_bytes(s); }
    }
}

#[inline] pub fn repeat(c: char, n: usize) -> Repeat { Repeat(c, n) }

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec::Vec;

    fn put_to_string<P: Put + ?Sized>(p: &P) -> String {
        let mut v: Vec<u8> = Vec::new();
        p.put(&mut v);
        String::from_utf8(v).unwrap()
    }

    #[test] fn name_hash_folds_ascii_case_only() {
        assert_eq!(name_hash(b"Finder"), name_hash(b"finder"));
        assert_eq!(name_hash(b"THINGS3"), name_hash(b"things3"));
        assert_ne!(name_hash(b"finder"), name_hash(b"finder "));
        assert_ne!(name_hash(b"ab"), name_hash(b"ba"));
        assert_ne!(name_hash(b""), name_hash(b"a"));
    }
    #[test] fn u32d_basic()    { assert_eq!(put_to_string(&u32d(42)), "42"); }
    #[test] fn u32d_zero()     { assert_eq!(put_to_string(&u32d(0)), "0"); }
    #[test] fn pad_right_pads_with_spaces() {
        assert_eq!(put_to_string(&pad_right(42, 5)), "   42");
        assert_eq!(put_to_string(&pad_right(12345, 3)), "12345");  // no truncation
    }
    #[test] fn pad_zero_pads_with_zeros() {
        assert_eq!(put_to_string(&pad_zero(7, 3)), "007");
    }
    #[test] fn pad_left_pads_right_with_spaces() {
        assert_eq!(put_to_string(&pad_left(42, 5)), "42   ");
    }
    #[test] fn f1_wide_rounds_and_pads() {
        assert_eq!(put_to_string(&f1_wide(3.14, 4)), " 3.1");
        assert_eq!(put_to_string(&f1_wide(0.0, 3)), "0.0");
    }
    #[test] fn f2_two_decimals() {
        assert_eq!(put_to_string(&f2(1.23)), "1.23");
        assert_eq!(put_to_string(&f2(0.5)), "0.50");
    }
    #[test] fn repeat_char() {
        assert_eq!(put_to_string(&repeat('-', 5)), "-----");
        assert_eq!(put_to_string(&repeat('─', 3)), "───");
        assert_eq!(put_to_string(&repeat('a', 0)), "");
    }
}
