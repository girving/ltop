//! Tiny-write machinery: a concat-style macro + `Put` trait that
//! compiles to direct put-byte calls on the writer. Replaces every
//! `write!(…)` in the terminal render path.
//!
//! The goal is to avoid the per-call-site cost of `write!`:
//!   - ~30 B of stack setup to stage a `core::fmt::Arguments`
//!     (value-pointer + formatter-fn-pointer, per `{}` placeholder)
//!   - Routing through `core::fmt::write` (~445 B) which walks pieces
//!     + args and dispatches each formatter
//!
//! The substitute: every argument to [`twrite!`] implements [`Put`],
//! which writes itself directly into any [`TinyWriter`]. Calls are
//! concrete method invocations, not vtable-indirected; the writer
//! type is a generic, so each site monomorphizes to direct put_*
//! calls and inlines.
//!
//! Infallibility: our only production writer is [`FBuilder<u8>`]
//! over the arena, which push into a pre-reserved buffer and cannot
//! fail. [`TinyWriter`] methods return `()`.
//!
//! Usage:
//! ```ignore
//! // Old
//! write!(out, "pid={}, cpu={}", pid, pad_right(cpu, 4))?;
//!
//! // New
//! twrite!(out, "pid=", pid, ", cpu=", pad_right(cpu, 4));
//! ```
//!
//! Strings, byte slices, integers, and chars all have `Put` impls
//! built in, so the old `Str(&str)` / `Bytes(&[u8])` wrappers are no
//! longer needed.

#![allow(dead_code)]

// ── TinyWriter ─────────────────────────────────────────────────────────────

/// Infallible byte sink. Implemented on [`FBuilder<u8>`] and
/// [`FVec<u8>`] for production; on `Vec<u8>` under `cfg(test)` so
/// tests can write into an owned buffer without pulling in more.
///
/// Implementors only need two primitives. String input rides in via
/// `[Put for str]` → `self.as_bytes()` → `put_bytes`, so nothing here
/// needs to touch UTF-8.
pub trait TinyWriter {
    fn put_bytes(&mut self, b: &[u8]);
    fn put_byte(&mut self, b: u8);

    /// Method-call surface for [`twrite!`]; receives the arg by value so
    /// literal `&'static str` / `&'static [u8]` args are passed in
    /// registers (ptr + len) instead of forcing Rust's rvalue static
    /// promotion of `&"literal"`. The latter places a `&str` in static
    /// storage and requires one rebase fixup per literal in
    /// `__DATA,__const` on Mach-O targets — the 34 entries we saw in
    /// the mac-min binary's `__DATA,__const` came from exactly this
    /// path. By-value eliminates them; the literal bytes themselves
    /// stay in `__TEXT,__const` and materialize at each call site via
    /// an immediate `adrp+add` for the ptr and `mov #imm` for the len.
    fn put_arg<P: Put>(&mut self, p: P) { (&p).put(self) }
}

/// Reborrow blanket so `&mut &mut W: TinyWriter` from the macro's
/// nested borrow shape just works.
impl<W: TinyWriter + ?Sized> TinyWriter for &mut W {
    #[inline] fn put_bytes(&mut self, b: &[u8]) { (**self).put_bytes(b) }
    #[inline] fn put_byte(&mut self, b: u8)     { (**self).put_byte(b) }
}

// ── Put ────────────────────────────────────────────────────────────────────

/// Types that can write themselves into a [`TinyWriter`]. The writer
/// is generic so each site monomorphizes to direct calls.
pub trait Put {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W);
}

/// Forward `&T: Put` through the inner `T: Put`.
impl<T: Put + ?Sized> Put for &T {
    #[inline] fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) { (**self).put(w) }
}

impl Put for str {
    #[inline] fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) { w.put_bytes(self.as_bytes()) }
}

/// `&[u8]` is interpreted as UTF-8 with `?` replacement for the
/// worst malformed bytes — lone continuations, invalid leaders, and
/// truncated sequences. Well-formed ASCII passes through as one
/// `put_bytes` per contiguous run; multi-byte leader + continuations
/// pass through when the byte shape is consistent (`0x80..=0xBF`
/// after a `0xC2..=0xF4` leader).
///
/// Deliberately lax relative to Unicode Table 3-7: we accept
/// "overlong" encodings (e.g. `E0 80 80`) and UTF-16 surrogate code
/// points (`ED A0..=BF ...`). A modern terminal will render these as
/// its replacement glyph and continue; the point is to keep the
/// renderer coherent on worst-case `/proc/<pid>/comm`, not to
/// validate UTF-8 strictly. Strict Table 3-7 validation used to live
/// here and cost ~170 B more `.text`.
impl Put for [u8] {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        // Leader → expected total sequence length, or 0 for bytes
        // that can't start one (lone continuation 0x80..=0xBF,
        // overlong 2-byte leader 0xC0..=0xC1, out-of-range leader
        // 0xF5..=0xFF). The sorted monotone bounds let LLVM compile
        // this to a handful of compares with no jump table.
        let leader_len = |b: u8| -> usize {
            if b < 0x80 { 1 }
            else if b < 0xC2 { 0 }
            else if b < 0xE0 { 2 }
            else if b < 0xF0 { 3 }
            else if b < 0xF5 { 4 }
            else { 0 }
        };
        let mut i = 0;
        while i < self.len() {
            let n = leader_len(self[i]);
            let ok = n > 0
                && i + n <= self.len()
                && self[i + 1..i + n].iter().all(|&b| b & 0xC0 == 0x80);
            if ok {
                w.put_bytes(&self[i..i + n]);
                i += n;
            } else {
                w.put_byte(b'?');
                i += 1;
            }
        }
    }
}

impl Put for char {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let mut buf = [0u8; 4];
        let n = self.encode_utf8(&mut buf).len();
        w.put_bytes(&buf[..n]);
    }
}

impl Put for u32 {
    #[inline] fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) { write_u32(w, *self) }
}

impl Put for u64 {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        // u64::MAX is 20 digits.
        if *self == 0 { w.put_byte(b'0'); return; }
        let mut buf = [0u8; 20];
        let mut i = buf.len();
        let mut m = *self;
        while m > 0 {
            i -= 1;
            buf[i] = b'0' + (m % 10) as u8;
            m /= 10;
        }
        w.put_bytes(&buf[i..]);
    }
}

impl Put for usize {
    #[inline] fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) { (*self as u64).put(w) }
}

/// A bare `u8` argument to [`twrite!`] means a single literal byte —
/// `twrite!(w, b' ')` writes one space. Picking the raw-byte semantic
/// over the decimal one (which would match `u32` / `u64` / etc.) is
/// deliberate: decimal u8 is rarely wanted and typed literals already
/// coerce — `n as u32` renders as decimal, `b'x'` renders as one byte.
/// Lets call sites avoid `put_arg::<&str>` layers for single-char
/// separators without an extra wrapper type.
impl Put for u8 {
    #[inline] fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) { w.put_byte(*self) }
}

impl Put for i32 {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        if *self < 0 {
            w.put_byte(b'-');
            // Negate via u32 to handle i32::MIN cleanly.
            (self.unsigned_abs()).put(w);
        } else {
            (*self as u32).put(w);
        }
    }
}

// ── shared digit writer ────────────────────────────────────────────────────

/// Write `n` as decimal digits (no padding, no prefix). Shared between
/// [`Put for u32`], the `pad_*` adapters in `src/bytes.rs`, and any
/// hand-rolled digit writing elsewhere in the crate.
pub(crate) fn write_u32<W: TinyWriter + ?Sized>(w: &mut W, n: u32) {
    if n == 0 { w.put_byte(b'0'); return; }
    // u32::MAX is 10 digits.
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    let mut m = n;
    while m > 0 {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
    }
    w.put_bytes(&buf[i..]);
}

/// Digit count of `n`. Used by pad_* helpers to decide how many
/// pad bytes to emit.
pub(crate) fn digit_count(mut n: u32) -> u8 {
    // Divide-by-10 loop: the constant division lowers to a
    // multiply-shift, and the whole loop is ~30 B — ilog10's
    // table/branch expansion was ~100 B here.
    let mut d = 1;
    while n >= 10 {
        n /= 10;
        d += 1;
    }
    d
}

// ── test-only writer into a Vec<u8> ────────────────────────────────────────

#[cfg(test)]
impl TinyWriter for alloc::vec::Vec<u8> {
    fn put_bytes(&mut self, b: &[u8]) { self.extend_from_slice(b) }
    fn put_byte(&mut self, b: u8)     { self.push(b) }
}

// ── twrite! macro ──────────────────────────────────────────────────────────

/// Concat-style write. Takes a writer and a list of `Put` arguments;
/// each argument writes itself directly into the writer in order.
/// No format string, no `{}` interpolation, no error propagation.
///
/// ```ignore
/// twrite!(out, "pid=", pid, ", cpu=", pad_right(cpu, 4));
/// ```
///
/// Peephole: a literal `" "` arg is rewritten to `put_byte(b' ')` at
/// macro-expansion time. Saves the `put_arg::<&str>` → `<&str as Put>`
/// → `<str as Put>` → `put_bytes(as_bytes())` chain for the column
/// separators that dominate twrite! call sites.
#[macro_export]
macro_rules! twrite {
    ($w:expr $(,)?) => {{ let _: &mut _ = &mut *$w; }};
    ($w:expr, $($args:tt)*) => {{
        // Bind the writer reference once, then reborrow it per-arg so
        // every put_arg call sees `&mut W` — not `&mut &mut W` that
        // would get a second forwarding monomorphisation through
        // `impl TinyWriter for &mut W`.
        let __w: &mut _ = &mut *$w;
        $crate::__twrite_munch!(__w, $($args)*);
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __twrite_munch {
    ($w:ident $(,)?) => {};
    ($w:ident, " " $(, $($rest:tt)*)?) => {
        $crate::twrite::TinyWriter::put_byte(&mut *$w, b' ');
        $( $crate::__twrite_munch!($w, $($rest)*); )?
    };
    ($w:ident, $arg:expr $(, $($rest:tt)*)?) => {
        $crate::twrite::TinyWriter::put_arg(&mut *$w, $arg);
        $( $crate::__twrite_munch!($w, $($rest)*); )?
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec::Vec;

    fn put_to_string<P: Put + ?Sized>(p: &P) -> String {
        let mut v = Vec::new();
        p.put(&mut v);
        String::from_utf8(v).unwrap()
    }

    #[test]
    fn str_and_slice_roundtrip() {
        assert_eq!(put_to_string("hello"), "hello");
        assert_eq!(put_to_string(&b"hello"[..]), "hello");
        assert_eq!(put_to_string(&""), "");
    }

    #[test]
    fn u32_decimals() {
        assert_eq!(put_to_string(&0u32), "0");
        assert_eq!(put_to_string(&1u32), "1");
        assert_eq!(put_to_string(&12345u32), "12345");
        assert_eq!(put_to_string(&u32::MAX), "4294967295");
    }

    #[test]
    fn u64_decimals() {
        assert_eq!(put_to_string(&0u64), "0");
        assert_eq!(put_to_string(&u64::MAX), "18446744073709551615");
    }

    #[test]
    fn i32_decimals_with_sign() {
        assert_eq!(put_to_string(&-1i32), "-1");
        assert_eq!(put_to_string(&0i32), "0");
        assert_eq!(put_to_string(&42i32), "42");
        assert_eq!(put_to_string(&i32::MIN), "-2147483648");
    }

    #[test]
    fn char_utf8() {
        assert_eq!(put_to_string(&'a'), "a");
        assert_eq!(put_to_string(&'é'), "é");
        assert_eq!(put_to_string(&'🌳'), "🌳");
    }

    #[test]
    fn bytes_utf8_recovery() {
        // Pass-through of valid UTF-8.
        assert_eq!(put_to_string(&b"caf\xc3\xa9"[..]), "café");
        // Lone continuation (0x80) → `?` replacement.
        assert_eq!(put_to_string(&b"a\x80b"[..]), "a?b");
        // 0xC0 / 0xC1 are the overlong 2-byte leaders and always
        // rejected (neither the short-leader range 0xC2..=0xDF nor
        // any other branch accepts them); the trailing 0x80 is then
        // also a lone continuation. Two `?`s.
        assert_eq!(put_to_string(&b"\xc0\x80"[..]), "??");
        // Emoji + box-drawing + interleaved lone-continuation garbage.
        assert_eq!(put_to_string(&b"\xf0\x9f\x8c\xb3\x80\xe2\x94\x80"[..]), "🌳?─");
    }

    /// The lax decoder accepts overlong encodings + UTF-16 surrogate
    /// code points — the strict Table 3-7 rejections used to be ~170 B
    /// of `.text` and the renderer doesn't need that accuracy (terminals
    /// silently replace or display a replacement glyph). Document the
    /// relaxation so a future re-tightening is a conscious choice.
    #[test]
    fn bytes_utf8_overlong_and_surrogate_pass() {
        fn put_to_bytes<P: Put + ?Sized>(p: &P) -> Vec<u8> {
            let mut v = Vec::new(); p.put(&mut v); v
        }
        // Overlong "E0 80 80" → U+0000 (strict would reject; we emit raw).
        assert_eq!(put_to_bytes(&b"\xe0\x80\x80"[..]), b"\xe0\x80\x80");
        // UTF-16 high surrogate "ED A0 80" → U+D800 (strict would reject).
        assert_eq!(put_to_bytes(&b"\xed\xa0\x80"[..]), b"\xed\xa0\x80");
        // U+110000 encoded as "F4 90 80 80" — strict would reject the
        // second byte; we accept the leader as long as the continuation
        // bytes have the right shape.
        assert_eq!(put_to_bytes(&b"\xf4\x90\x80\x80"[..]), b"\xf4\x90\x80\x80");
        // But 0xF5..=0xFF leaders are still rejected — out of UTF-8
        // encoding range entirely.
        assert_eq!(put_to_bytes(&b"\xf5\x80\x80\x80"[..]), b"????");
    }

    #[test]
    fn macro_concat() {
        let mut v: Vec<u8> = Vec::new();
        let pid: u32 = 123;
        let name: &str = "zsh";
        twrite!(&mut v, "pid=", pid, " name=", name);
        assert_eq!(String::from_utf8(v).unwrap(), "pid=123 name=zsh");
    }

    #[test]
    fn macro_trailing_comma() {
        let mut v: Vec<u8> = Vec::new();
        twrite!(&mut v, "a", "b",);
        assert_eq!(String::from_utf8(v).unwrap(), "ab");
    }
}
