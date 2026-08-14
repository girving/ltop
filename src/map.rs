//! Flat sorted-vector Map and Set, keyed on `u32`.
//!
//! `Map<'a, V>` is `&'a [(u32, V)]` with binary-search methods; `Set<'a>` is
//! `&'a [u32]`. Both are `Copy` — zero runtime state besides the slice.
//!
//! Callers own the backing storage (a `Vec<...>`), push into it freely, then
//! call `freeze_map` / `freeze_set`. These sort + dedup the vec in place and
//! return a view that borrows it shared-ly — for as long as the view lives,
//! the borrow checker forbids further `push`/`clear`/mutation.
//!
//! The sort is `crate::sort::sort_by_u32_key`, a non-recursive heapsort
//! that keeps the whole binary's worst-case stack depth accountable via
//! `.stack_sizes`. See `src/sort.rs` for the rationale and algorithm.

pub use crate::sort::sort_by_u32_key;

pub struct Map<'a, V>(&'a [(u32, V)]);
pub struct Set<'a>(&'a [u32]);

// Manual Copy/Clone: derive would add unwanted bounds on V.
impl<V> Copy for Map<'_, V> {}
impl<V> Clone for Map<'_, V> { fn clone(&self) -> Self { *self } }
impl Copy for Set<'_> {}
impl Clone for Set<'_> { fn clone(&self) -> Self { *self } }

impl<'a, V> Map<'a, V> {
    /// Wrap a slice that's already sorted by key and de-duplicated.
    /// In debug builds this invariant is checked.
    #[allow(dead_code)]
    pub fn new(sorted: &'a [(u32, V)]) -> Self {
        debug_assert!(sorted.windows(2).all(|w| w[0].0 < w[1].0),
            "Map::new: slice must be sorted by key with no duplicates");
        Self(sorted)
    }

    pub fn get(&self, k: &u32) -> Option<&'a V> {
        self.0.binary_search_by_key(k, |(ek, _)| *ek).ok().map(|i| &self.0[i].1)
    }

    #[allow(dead_code)] pub fn contains_key(&self, k: &u32) -> bool { self.get(k).is_some() }
    #[allow(dead_code)] pub fn len(&self) -> usize { self.0.len() }
    #[allow(dead_code)] pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn iter(&self) -> core::slice::Iter<'a, (u32, V)> { self.0.iter() }
}

impl<'a> Set<'a> {
    #[allow(dead_code)]
    pub fn new(sorted: &'a [u32]) -> Self {
        debug_assert!(sorted.windows(2).all(|w| w[0] < w[1]),
            "Set::new: slice must be sorted with no duplicates");
        Self(sorted)
    }

    pub fn contains(&self, k: &u32) -> bool { self.0.binary_search(k).is_ok() }
    #[allow(dead_code)] pub fn len(&self) -> usize { self.0.len() }
    #[allow(dead_code)] pub fn is_empty(&self) -> bool { self.0.is_empty() }
    #[allow(dead_code)] pub fn iter(&self) -> core::slice::Iter<'a, u32> { self.0.iter() }
}

/// Sort `entries` by key, merging consecutive duplicate keys via `merge`,
/// then return a `Map` borrowing the now-sorted+deduped buffer. While the
/// returned view lives, `entries` is locked read-only by the borrow checker.
///
/// `merge(&mut V, &mut V)` receives the keeper and the doomed element; the
/// doomed element is discarded (overwritten by the next live element) so a
/// closure that just reads `&b` or drains into `a` is enough.
///
/// Requires `V: Copy` because dedup compacts in place via assignment
/// (no drop tracking for removed elements).
pub fn freeze_map<'a, 'id, V: Copy>(
    entries: &'a mut crate::arena::FVec<'id, (u32, V)>,
    mut merge: impl FnMut(&mut V, &mut V),
) -> Map<'a, V> {
    // Compiler-enforced proof of sort_by_u32_key's "u32 key at offset 0"
    // precondition: #[repr(Rust)] tuple layout is only empirically
    // stable, and a future rustc layout change would otherwise silently
    // sort by whatever field landed first (audit finding, sort.rs:65).
    const { assert!(core::mem::offset_of!((u32, V), 0) == 0,
        "freeze_map: rustc no longer lays (u32, V) out with the key first"); }
    sort_by_u32_key(entries.as_mut_slice());
    let new_len = dedup_in_place(entries.as_mut_slice(), |right, left| {
        if left.0 == right.0 {
            merge(&mut left.1, &mut right.1);
            true
        } else { false }
    });
    entries.truncate(new_len as u32);
    Map(entries.as_slice())
}

/// Owned variant of [`freeze_map`]: consumes the FVec and returns a Map
/// whose lifetime is the arena brand `'id` rather than a local borrow.
/// Lets helper functions allocate the intermediate FVec internally and
/// still return a Map the caller can hold — see `build_pid_to_idx`.
///
/// Delegates the sort+dedup+truncate work to [`freeze_map`]; only the
/// final lifetime-lifting (borrow → arena brand) is unique here.
pub fn freeze_map_owned<'id, V: Copy>(
    mut entries: crate::arena::FVec<'id, (u32, V)>,
    merge: impl FnMut(&mut V, &mut V),
) -> Map<'id, V> {
    // Run freeze_map for its side effects; discard the returned Map
    // because its borrow lifetime is shorter than what we're promising
    // the caller. `let _ =` rather than `drop()` because Map is Copy
    // (`&[...]` underneath), so drop is a no-op and the compiler warns.
    let _ = freeze_map(&mut entries, merge);
    // SAFETY: arena bytes at `ptr` hold `len` initialised `(u32, V)`
    // tuples and live for the enclosing scope (`'id` brand). `mem::forget`
    // skips FVec::drop (a no-op for `V: Copy` anyway). The arena's bump
    // pointer is past these bytes so future allocations won't overlap,
    // and nothing else holds an FVec handle to this region after the
    // forget so no aliasing is possible.
    let len = entries.len();
    let ptr = entries.as_slice().as_ptr();
    core::mem::forget(entries);
    Map(unsafe { core::slice::from_raw_parts(ptr, len) })
}

pub fn freeze_set<'a, 'id>(
    keys: &'a mut crate::arena::FVec<'id, u32>,
) -> Set<'a> {
    sort_by_u32_key(keys.as_mut_slice());
    let new_len = dedup_in_place(keys.as_mut_slice(), |right, left| left == right);
    keys.truncate(new_len as u32);
    Set(keys.as_slice())
}

/// Dedup a slice in place, assignment-compacting kept elements to the
/// front. `same(right, left) -> true` means `right` is a duplicate of
/// `left` and should be discarded; the closure may mutate `left` to merge
/// state from `right` before we drop it. Returns the new length.
///
/// `T: Copy` so we can overwrite slots with `slice[w] = slice[r]` without
/// worrying about Drop semantics — our uses (u32 keys, Copy V) all satisfy.
fn dedup_in_place<T: Copy>(
    slice: &mut [T],
    mut same: impl FnMut(&mut T, &mut T) -> bool,
) -> usize {
    if slice.len() < 2 { return slice.len(); }
    let mut w = 1;
    for r in 1..slice.len() {
        // Need simultaneous mutable access to slice[w-1] (last keeper) and
        // slice[r] (candidate). Split the slice at r to get disjoint halves.
        let (left, right) = slice.split_at_mut(r);
        if same(&mut right[0], &mut left[w - 1]) {
            // Candidate is a duplicate; merge ran inside `same`. Skip it.
        } else {
            // Not a duplicate: compact candidate into slot w.
            if w != r { left[w] = right[0]; }
            w += 1;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{self, FVec};

    #[test]
    fn freeze_map_owned_basic() {
        let _g = arena::test_lock();
        arena::scope(|f| {
            let mut buf: FVec<(u32, u32)> = f.vec("test", 4);
            buf.extend([(30u32, 3u32), (10, 1), (20, 2)]);
            let m = freeze_map_owned(buf, |_, _| ());
            assert_eq!(m.len(), 3);
            assert_eq!(m.get(&10), Some(&1));
            assert_eq!(m.get(&20), Some(&2));
            assert_eq!(m.get(&30), Some(&3));
            assert_eq!(m.get(&999), None);
        });
    }

    #[test]
    fn freeze_map_owned_merges_duplicates() {
        let _g = arena::test_lock();
        arena::scope(|f| {
            // Duplicate keys: merge closure should be called and sum the values.
            let mut buf: FVec<(u32, u32)> = f.vec("test", 8);
            buf.extend([(1u32, 10), (1, 20), (1, 5), (2, 100), (3, 0), (3, 7)]);
            let m = freeze_map_owned(buf, |a, b| *a += *b);
            assert_eq!(m.len(), 3);
            assert_eq!(m.get(&1), Some(&35));  // 10 + 20 + 5
            assert_eq!(m.get(&2), Some(&100));
            assert_eq!(m.get(&3), Some(&7));   // 0 + 7
        });
    }

    #[test]
    fn freeze_map_owned_empty() {
        let _g = arena::test_lock();
        arena::scope(|f| {
            let buf: FVec<(u32, u32)> = f.vec("test", 4);
            let m = freeze_map_owned(buf, |_, _| ());
            assert!(m.is_empty());
            assert_eq!(m.get(&0), None);
        });
    }

    #[test]
    fn freeze_map_owned_outlives_allocator_view() {
        // The returned Map should stay valid through further arena
        // allocations — this is the whole point of the 'id lifetime.
        let _g = arena::test_lock();
        arena::scope(|f| {
            let mut buf: FVec<(u32, u32)> = f.vec("test", 4);
            buf.extend([(5u32, 50u32), (2, 20)]);
            let m = freeze_map_owned(buf, |_, _| ());

            // Allocate a bunch more stuff; the Map's backing bytes should
            // be unaffected (they're below the bump pointer, not tied to
            // the FVec handle we `mem::forget`ed).
            let _other: FVec<u64> = f.vec("test", 100);
            let _yet_more: FVec<u32> = f.vec("test", 50);

            assert_eq!(m.get(&2), Some(&20));
            assert_eq!(m.get(&5), Some(&50));
        });
    }

    #[test]
    fn freeze_map_owned_multiple_in_one_scope() {
        // Two freeze_map_owned results should coexist without the second
        // stomping on the first.
        let _g = arena::test_lock();
        arena::scope(|f| {
            let mut b1: FVec<(u32, u32)> = f.vec("test", 2);
            b1.extend([(1u32, 100u32), (2, 200)]);
            let m1 = freeze_map_owned(b1, |_, _| ());

            let mut b2: FVec<(u32, u32)> = f.vec("test", 2);
            b2.extend([(3u32, 300u32), (4, 400)]);
            let m2 = freeze_map_owned(b2, |_, _| ());

            assert_eq!(m1.get(&1), Some(&100));
            assert_eq!(m1.get(&2), Some(&200));
            assert_eq!(m2.get(&3), Some(&300));
            assert_eq!(m2.get(&4), Some(&400));
        });
    }
}
