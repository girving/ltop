//! The `Zeroable` marker trait.
//!
//! Exists so that arena's `alloc_zeroed` can be a safe function at the
//! call site rather than forcing every caller to write `unsafe { ... }`.
//! The obligation ("is all-bits-zero a valid value of this type?") moves
//! to the `unsafe impl Zeroable for …` declaration, which is where the
//! person adding the type has the context to check it.
//!
//! This is the same pattern as `bytemuck::Zeroable`, minus the crate
//! dependency and the derive macro. For ltop's one or two use sites, a
//! one-line `unsafe impl Zeroable for MyType {}` is fine.

/// Types for which the all-bits-zero bit pattern is a valid value.
///
/// # Safety
///
/// Implementors assert that `mem::zeroed::<Self>()` is a valid value of
/// `Self`. Violating this is UB when consumers (such as
/// `arena::Frame::alloc_zeroed`) read the result.
///
/// Safe to impl for:
/// - integers, floats, `bool` (0 is a valid value);
/// - structs of `Zeroable` fields with `#[repr(C)]` or similar predictable
///   layout (zero bytes = every field zero);
/// - arrays of `Zeroable`.
///
/// NOT safe to impl for:
/// - `&T`, `&mut T`, `Box<T>`, `Vec<T>` (null / dangling invariant);
/// - `char` (not every u32 is a valid codepoint);
/// - `NonZero*` (0 is the one value explicitly forbidden);
/// - enums with explicit `#[repr]` discriminants where 0 isn't assigned.
pub unsafe trait Zeroable {}

macro_rules! impl_zeroable {
    ($($ty:ty),* $(,)?) => {
        $( unsafe impl Zeroable for $ty {} )*
    };
}
impl_zeroable!(u8, u16, u32, u64, u128, usize);
impl_zeroable!(i8, i16, i32, i64, i128, isize);
impl_zeroable!(f32, f64, bool);

// An array of Zeroable is Zeroable: the zero-byte pattern covers every
// element slot. Covers `[T; 0]` trivially (no bytes to worry about).
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}
