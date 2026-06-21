//! In-place sort by the u32 key at each element's byte offset 0.
//!
//! Replaces `libc::qsort` as the backing sort for `map::freeze_map` /
//! `freeze_set` and main.rs's per-tick procs/next sorts. Three reasons
//! we're not calling into musl anymore:
//!
//!   1. **Stack accounting.** musl's `qsort` isn't compiled with
//!      `-Zemit-stack-sizes`, so its recursion depth is invisible to
//!      `tools/stack-svg` and `tests/stack_frames`. Heapsort recurses
//!      nowhere — the test now reflects reality.
//!
//!   2. **Stack RSS.** The custom `_start` in `src/start.rs` madvises
//!      everything below our top page. Any qsort recursion past
//!      ~1.6 KB then re-commits the next page. Non-recursive heapsort
//!      stays firmly in one 4 KB page.
//!
//!   3. **Binary size.** musl's `__qsort_r` is ~905 bytes plus a 15-byte
//!      `qsort` wrapper. This heapsort is ~300 bytes. Net ~600 B off
//!      the `.text`.
//!
//! Algorithm: plain heapsort, iterative. O(n log n) worst case, no
//! auxiliary memory, one stack frame. The key at offset 0 in each
//! element is read as an aligned u32 (public entry const-asserts
//! `size_of::<T>() % 4 == 0` and `align_of::<T>() >= 4`, so every
//! caller satisfies the preconditions at compile time). Swap is a
//! word-at-a-time loop — previously `ptr::swap_nonoverlapping::<u8>`
//! over runtime-length bytes, which lowered to a compiler_builtins
//! `memcpy` call (~164 B). The loop compiles to a few aligned u32
//! load/store pairs with no library dependency.
//!
//! ## `unsafe` hygiene
//!
//! Three layers:
//!
//! 1. `sort_by_u32_key<T>` — public generic surface. Tiny body: compute
//!    `size_of::<T>()` and call `sort_mono`. Monomorphises per `T`,
//!    but only with a handful of instructions each.
//! 2. `sort_mono(base, len, elem_size)` — single non-generic function.
//!    Defines the `key` / `swap` closures that do the raw-pointer
//!    arithmetic. All unsafe code in the module lives here.
//! 3. `sort_raw` / `sift_down` — generic `impl Fn` / `impl FnMut`
//!    algorithm. Because they're only ever called from `sort_mono`
//!    with one pair of closure types, they monomorphise exactly once
//!    — no duplicate copies per `T`, and no `dyn` call overhead. The
//!    algorithm itself is index-only; no raw pointers inside.

use core::mem::{align_of, size_of};

/// Sort `slice` ascending by the first 4 bytes of each element
/// interpreted as a little-endian `u32`.
///
/// Works on any `T` that starts with a `u32` at byte offset 0:
///   * `T = u32` (used by `freeze_set`)
///   * `T = (u32, V)` in Rust's default repr (u32 at offset 0, aligned)
///   * `T = ProcInfo` with `#[repr(C)]` + `pid: u32` first
///
/// Compile-time asserts:
///   * `size_of::<T>() % 4 == 0` so the body can iterate in u32 words
///   * `align_of::<T>() >= 4` so the u32 reads are aligned (no
///     read_unaligned, which used to hide a u32-at-arbitrary-offset
///     promise that real callers never actually needed)
pub fn sort_by_u32_key<T>(slice: &mut [T]) {
    const { assert!(size_of::<T>() % 4 == 0,
        "sort_by_u32_key: size_of::<T>() must be a multiple of 4"); }
    const { assert!(align_of::<T>() >= 4,
        "sort_by_u32_key: align_of::<T>() must be >= 4"); }
    if slice.len() < 2 {
        return;
    }
    // SAFETY: `(base, len, size_of::<T>())` describes exactly the same
    // bytes as `slice`, a live `&mut [T]`. size divisibility and
    // alignment preconditions are const-asserted above.
    unsafe { sort_mono(slice.as_mut_ptr() as *mut u8, slice.len(), size_of::<T>()) };
}

/// Single non-generic entry point. Wraps the raw pointer + element
/// size in two closures whose bodies are the only `unsafe` in the
/// module, then hands them to `sort_raw`. Because `sort_mono` is
/// the sole caller of `sort_raw`, the closure types are fixed and
/// `sort_raw` / `sift_down` monomorphise exactly once in the binary.
///
/// # Safety
/// `base..base + len * elem_size` must be a single writable allocation,
/// `base` must be 4-byte aligned, and `elem_size` must be a positive
/// multiple of 4. All three are ensured by `sort_by_u32_key`'s
/// const-assertions.
#[inline(never)]
unsafe fn sort_mono(base: *mut u8, len: usize, elem_size: usize) {
    // elem_size is always a multiple of 4 at this point; express it in
    // u32 words for the iteration.
    let elem_words = elem_size / 4;
    let base_u32 = base as *mut u32;

    // SAFETY for both closures: every index passed in below is
    // produced by `sort_raw` / `sift_down`, which operate strictly on
    // `0..len`. `base_u32.add(i * elem_words)` for `i < len` points
    // inside the caller-guaranteed `len * elem_size`-byte allocation
    // and at a 4-byte-aligned address (base was 4-aligned; stride is
    // a u32 multiple).
    let key = |i: usize| -> u32 {
        unsafe { base_u32.add(i * elem_words).read() }
    };
    let mut swap = |i: usize, j: usize| {
        // Word-at-a-time swap. Replaces `ptr::swap_nonoverlapping::<u8>`
        // over runtime-length bytes, which lowered to a
        // `compiler_builtins::memcpy` call.
        unsafe {
            let a = base_u32.add(i * elem_words);
            let b = base_u32.add(j * elem_words);
            for k in 0..elem_words {
                let pa = a.add(k);
                let pb = b.add(k);
                let ta = pa.read();
                pa.write(pb.read());
                pb.write(ta);
            }
        }
    };
    sort_raw(len, &key, &mut swap);
}

/// Heapsort `len` items ascending by `key(i)`. Index-only — no raw
/// pointers inside. Monomorphises once per distinct `(K, S)` type pair
/// at the call site; `sort_mono` is the only caller.
fn sort_raw<K: Fn(usize) -> u32, S: FnMut(usize, usize)>(len: usize, key: &K, swap: &mut S) {
    // Phase 1: heapify. Start at the last parent and sift down towards 0.
    let mut i = len / 2;
    while i > 0 {
        i -= 1;
        sift_down(i, len, key, swap);
    }
    // Phase 2: repeatedly swap root (max) with last, shrink the heap,
    // sift new root down. Produces an ascending-sorted array in place.
    let mut n = len;
    while n > 1 {
        n -= 1;
        swap(0, n);
        sift_down(0, n, key, swap);
    }
}

/// Sift the element at index `i` down the heap of size `heap_len`
/// until the max-heap invariant is restored. Non-recursive.
fn sift_down<K: Fn(usize) -> u32, S: FnMut(usize, usize)>(
    mut i: usize,
    heap_len: usize,
    key: &K,
    swap: &mut S,
) {
    loop {
        let left = 2 * i + 1;
        if left >= heap_len {
            return;
        }
        let right = left + 1;
        // Pick the child with the larger key (right if it exists and
        // compares greater; left otherwise).
        let mut max = left;
        if right < heap_len && key(right) > key(left) {
            max = right;
        }
        if key(i) >= key(max) {
            return;
        }
        swap(i, max);
        i = max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_u32() {
        let mut v = [7u32, 3, 9, 1, 4, 1, 8, 2];
        sort_by_u32_key(&mut v);
        assert_eq!(v, [1, 1, 2, 3, 4, 7, 8, 9]);
    }

    #[test]
    fn sorts_empty_and_single() {
        let mut v: [u32; 0] = [];
        sort_by_u32_key(&mut v);
        let mut w = [42u32];
        sort_by_u32_key(&mut w);
        assert_eq!(w, [42]);
    }

    #[test]
    fn sorts_pair_with_value() {
        #[derive(Copy, Clone, PartialEq, Debug)]
        #[repr(C)]
        struct Pair(u32, u64);
        let mut v = [
            Pair(3, 30),
            Pair(1, 10),
            Pair(4, 40),
            Pair(1, 11),
            Pair(5, 50),
            Pair(9, 90),
            Pair(2, 20),
        ];
        sort_by_u32_key(&mut v);
        // Heapsort isn't stable — don't assume the order of the two 1-keys.
        assert_eq!(v[0].0, 1);
        assert_eq!(v[1].0, 1);
        assert_eq!(v[2], Pair(2, 20));
        assert_eq!(v[3], Pair(3, 30));
        assert_eq!(v[4], Pair(4, 40));
        assert_eq!(v[5], Pair(5, 50));
        assert_eq!(v[6], Pair(9, 90));
    }

    #[test]
    fn sorts_already_sorted() {
        let mut v: Vec<u32> = (0..100).collect();
        sort_by_u32_key(&mut v);
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn sorts_reverse() {
        let mut v: Vec<u32> = (0..100).rev().collect();
        sort_by_u32_key(&mut v);
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    /// 10 000 elements from a seeded xorshift PRNG, masked to give many
    /// duplicate keys (exercises the `key(i) >= key(max)` equality path
    /// in `sift_down`). Deterministic — same seed every run.
    #[test]
    fn sorts_fuzzy_with_duplicates() {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut v: Vec<u32> = (0..10_000).map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s as u32) & 0xFF  // mask → ~40 duplicates per key value
        }).collect();
        let mut expected = v.clone();
        expected.sort();
        sort_by_u32_key(&mut v);
        assert_eq!(v, expected);
    }
}
