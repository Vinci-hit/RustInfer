//! Inline-storage shape / stride container.
//!
//! Tensors in this framework cap out at [`MAX_RANK`] dimensions. By storing
//! shape and stride as `[usize; MAX_RANK] + len` we avoid all heap
//! allocation in the hot path — `Dims` is `Copy`, so cloning a `Tensor`'s
//! metadata is a handful of register moves.
//!
//! The type deliberately mimics `&[usize]`: it derefs to a slice, so code
//! written against slice APIs keeps working.

use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// Maximum supported tensor rank.
///
/// Seven is the tightest real-world requirement (z-image patchify uses a
/// 7-D intermediate); eight gives one slot of headroom without growing the
/// struct past 72 bytes.
pub const MAX_RANK: usize = 8;

/// Fixed-capacity, inline-stored `[usize]` used for shapes and strides.
///
/// - `Copy` — metadata ops never allocate.
/// - `PartialEq` / `Eq` — shape comparisons are byte-for-byte.
/// - Transparent `Deref<[usize]>` — plugs into any slice-taking API.
#[derive(Clone, Copy)]
pub struct Dims {
    data: [usize; MAX_RANK],
    len:  u8,
}

impl Dims {
    /// Empty `Dims` (rank 0 — represents a scalar).
    pub const fn new() -> Self {
        Self { data: [0; MAX_RANK], len: 0 }
    }

    /// Build from an arbitrary `&[usize]`.
    ///
    /// # Panics
    /// If `slice.len() > MAX_RANK`. Callers that want a graceful error
    /// should use [`Dims::try_from_slice`] instead.
    pub fn from_slice(slice: &[usize]) -> Self {
        assert!(
            slice.len() <= MAX_RANK,
            "Dims: rank {} exceeds MAX_RANK={}", slice.len(), MAX_RANK,
        );
        let mut data = [0usize; MAX_RANK];
        data[..slice.len()].copy_from_slice(slice);
        Self { data, len: slice.len() as u8 }
    }

    /// Fallible variant of [`Dims::from_slice`].
    pub fn try_from_slice(slice: &[usize]) -> Option<Self> {
        if slice.len() > MAX_RANK { return None; }
        let mut data = [0usize; MAX_RANK];
        data[..slice.len()].copy_from_slice(slice);
        Some(Self { data, len: slice.len() as u8 })
    }

    /// Number of valid entries (the tensor rank).
    #[inline]
    pub const fn len(&self) -> usize { self.len as usize }

    /// `true` iff this is a rank-0 shape.
    #[inline]
    pub const fn is_empty(&self) -> bool { self.len == 0 }

    /// Borrow the populated prefix as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[usize] { &self.data[..self.len as usize] }

    /// Mutable borrow of the populated prefix.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.data[..self.len as usize]
    }

    /// Append an entry. Panics if the rank would exceed `MAX_RANK`.
    #[inline]
    pub fn push(&mut self, v: usize) {
        let i = self.len as usize;
        assert!(i < MAX_RANK, "Dims::push: overflow");
        self.data[i] = v;
        self.len += 1;
    }

    /// Remove and return the entry at `index`, shifting subsequent entries
    /// left. Panics if out of bounds.
    pub fn remove(&mut self, index: usize) -> usize {
        let len = self.len as usize;
        assert!(index < len, "Dims::remove: index {} >= len {}", index, len);
        let removed = self.data[index];
        for i in index..len - 1 {
            self.data[i] = self.data[i + 1];
        }
        self.data[len - 1] = 0;
        self.len -= 1;
        removed
    }

    /// Insert `value` at `index`, shifting subsequent entries right. Panics
    /// on overflow or out-of-bounds.
    pub fn insert(&mut self, index: usize, value: usize) {
        let len = self.len as usize;
        assert!(len < MAX_RANK, "Dims::insert: overflow");
        assert!(index <= len, "Dims::insert: index {} > len {}", index, len);
        for i in (index..len).rev() {
            self.data[i + 1] = self.data[i];
        }
        self.data[index] = value;
        self.len += 1;
    }

    /// Product of all entries (1 for rank-0).
    #[inline]
    pub fn product(&self) -> usize {
        self.as_slice().iter().copied().fold(1usize, usize::saturating_mul)
    }

    /// Row-major (C-contiguous) strides for a given shape.
    pub fn contiguous_strides_for(shape: &[usize]) -> Self {
        let n = shape.len();
        assert!(n <= MAX_RANK, "contiguous_strides_for: rank {} > MAX_RANK", n);
        let mut data = [0usize; MAX_RANK];
        if n > 0 {
            data[n - 1] = 1;
            for i in (0..n - 1).rev() {
                data[i] = data[i + 1] * shape[i + 1];
            }
        }
        Self { data, len: n as u8 }
    }
}

impl Default for Dims {
    fn default() -> Self { Self::new() }
}

impl Deref for Dims {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] { self.as_slice() }
}

impl DerefMut for Dims {
    #[inline]
    fn deref_mut(&mut self) -> &mut [usize] { self.as_mut_slice() }
}

impl Index<usize> for Dims {
    type Output = usize;
    #[inline]
    fn index(&self, i: usize) -> &usize { &self.as_slice()[i] }
}

impl IndexMut<usize> for Dims {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut usize {
        &mut self.as_mut_slice()[i]
    }
}

impl PartialEq for Dims {
    fn eq(&self, other: &Self) -> bool { self.as_slice() == other.as_slice() }
}
impl Eq for Dims {}

impl PartialEq<[usize]> for Dims {
    fn eq(&self, other: &[usize]) -> bool { self.as_slice() == other }
}
impl<const N: usize> PartialEq<[usize; N]> for Dims {
    fn eq(&self, other: &[usize; N]) -> bool { self.as_slice() == other.as_slice() }
}

impl fmt::Debug for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl From<&[usize]> for Dims {
    fn from(s: &[usize]) -> Self { Self::from_slice(s) }
}
impl<const N: usize> From<[usize; N]> for Dims {
    fn from(s: [usize; N]) -> Self { Self::from_slice(&s) }
}
impl From<&Vec<usize>> for Dims {
    fn from(s: &Vec<usize>) -> Self { Self::from_slice(s) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_remove() {
        let mut d = Dims::new();
        d.push(2); d.push(3); d.push(4);
        assert_eq!(d.as_slice(), &[2, 3, 4]);
        assert_eq!(d.remove(1), 3);
        assert_eq!(d.as_slice(), &[2, 4]);
    }

    #[test]
    fn contiguous_strides() {
        let s = Dims::contiguous_strides_for(&[2, 3, 4]);
        assert_eq!(s.as_slice(), &[12, 4, 1]);
        let s0 = Dims::contiguous_strides_for(&[]);
        assert_eq!(s0.len(), 0);
    }

    #[test]
    fn insert_shift() {
        let mut d = Dims::from_slice(&[1, 2, 3]);
        d.insert(1, 99);
        assert_eq!(d.as_slice(), &[1, 99, 2, 3]);
    }
}
