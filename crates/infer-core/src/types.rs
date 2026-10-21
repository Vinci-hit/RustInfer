//! Domain value types — pure math, zero dependencies.

use half::{bf16, f16};
use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

// ─── Dims ────────────────────────────────────────────────────────────────────

pub const MAX_RANK: usize = 8;

#[derive(Clone, Copy)]
pub struct Dims {
    data: [usize; MAX_RANK],
    len: u8,
}

impl Dims {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0; MAX_RANK],
            len: 0,
        }
    }
    pub fn from_slice(s: &[usize]) -> Self {
        assert!(s.len() <= MAX_RANK);
        let mut data = [0usize; MAX_RANK];
        data[..s.len()].copy_from_slice(s);
        Self {
            data,
            len: s.len() as u8,
        }
    }
    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.data[..self.len as usize]
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.data[..self.len as usize]
    }
    pub fn push(&mut self, v: usize) {
        let i = self.len as usize;
        assert!(i < MAX_RANK);
        self.data[i] = v;
        self.len += 1;
    }
    pub fn remove(&mut self, index: usize) -> usize {
        let len = self.len as usize;
        assert!(index < len);
        let removed = self.data[index];
        for i in index..len - 1 {
            self.data[i] = self.data[i + 1];
        }
        self.data[len - 1] = 0;
        self.len -= 1;
        removed
    }
    pub fn product(&self) -> usize {
        self.as_slice()
            .iter()
            .copied()
            .fold(1usize, |a, d| a.checked_mul(d).expect("overflow"))
    }
    pub fn contiguous_strides_for(shape: &[usize]) -> Self {
        let n = shape.len();
        assert!(n <= MAX_RANK);
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
    fn default() -> Self {
        Self::new()
    }
}
impl Deref for Dims {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        self.as_slice()
    }
}
impl DerefMut for Dims {
    fn deref_mut(&mut self) -> &mut [usize] {
        self.as_mut_slice()
    }
}
impl Index<usize> for Dims {
    type Output = usize;
    fn index(&self, i: usize) -> &usize {
        &self.as_slice()[i]
    }
}
impl IndexMut<usize> for Dims {
    fn index_mut(&mut self, i: usize) -> &mut usize {
        &mut self.as_mut_slice()[i]
    }
}
impl PartialEq for Dims {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for Dims {}
impl fmt::Debug for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}
impl<const N: usize> From<[usize; N]> for Dims {
    fn from(s: [usize; N]) -> Self {
        Self::from_slice(&s)
    }
}

// ─── Shape / Strides NewTypes ────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape(pub Dims);
impl Shape {
    pub fn from_slice(s: &[usize]) -> Self {
        Self(Dims::from_slice(s))
    }
    #[inline]
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
    #[inline]
    pub fn numel(&self) -> usize {
        self.0.product()
    }
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }
    pub fn contiguous_strides(&self) -> Strides {
        Strides(Dims::contiguous_strides_for(self.0.as_slice()))
    }
}
impl Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        self.0.as_slice()
    }
}
impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(s: [usize; N]) -> Self {
        Self::from_slice(&s)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Strides(pub Dims);
impl Strides {
    pub fn from_slice(s: &[usize]) -> Self {
        Self(Dims::from_slice(s))
    }
    pub fn contiguous_for(shape: &Shape) -> Self {
        shape.contiguous_strides()
    }
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }
}
impl Deref for Strides {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        self.0.as_slice()
    }
}
impl fmt::Debug for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Strides({:?})", self.0)
    }
}

// ─── DataType / Dtype ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    U8,
    U32,
    I32,
    I8,
}
impl DataType {
    #[inline]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F8E4M3 | DataType::F8E5M2 | DataType::I8 | DataType::U8 => 1,
        }
    }

    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F8E4M3 | DataType::F8E5M2
        )
    }
}

pub trait Dtype: Copy + Send + Sync + 'static + fmt::Debug {
    const DATA_TYPE: DataType;
    const SIZE_BYTES: usize;

    /// Convert one host scalar to the backend-independent reference format.
    fn read_f64(raw: &Self) -> f64;

    /// Convert a backend-independent reference scalar to this dtype.
    fn write_f64(v: f64) -> Self;
}
impl Dtype for f32 {
    const DATA_TYPE: DataType = DataType::F32;
    const SIZE_BYTES: usize = 4;

    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }

    fn write_f64(v: f64) -> Self {
        v as f32
    }
}
impl Dtype for f16 {
    const DATA_TYPE: DataType = DataType::F16;
    const SIZE_BYTES: usize = 2;

    fn read_f64(raw: &Self) -> f64 {
        f64::from(raw.to_f32())
    }

    fn write_f64(v: f64) -> Self {
        f16::from_f32(v as f32)
    }
}
impl Dtype for bf16 {
    const DATA_TYPE: DataType = DataType::BF16;
    const SIZE_BYTES: usize = 2;

    fn read_f64(raw: &Self) -> f64 {
        f64::from(raw.to_f32())
    }

    fn write_f64(v: f64) -> Self {
        bf16::from_f32(v as f32)
    }
}
impl Dtype for i32 {
    const DATA_TYPE: DataType = DataType::I32;
    const SIZE_BYTES: usize = 4;

    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }

    fn write_f64(v: f64) -> Self {
        v as i32
    }
}
impl Dtype for i8 {
    const DATA_TYPE: DataType = DataType::I8;
    const SIZE_BYTES: usize = 1;

    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }

    fn write_f64(v: f64) -> Self {
        v as i8
    }
}

/// Marker: floating-point dtypes only.
pub trait Float: Dtype {}
impl Float for f32 {}
impl Float for f16 {}
impl Float for bf16 {}

/// Whether distinct matrix indices address distinct elements. Empty matrices,
/// transposes and padded layouts are allowed; broadcast outputs are rejected.
/// For positive strides, the minimal collision has row distance s1/gcd(s0,s1)
/// and column distance s0/gcd(s0,s1). No extent/stride multiplication is needed.
pub fn matrix_elements_are_disjoint(rows: usize, cols: usize, s0: usize, s1: usize) -> bool {
    if rows == 0 || cols == 0 {
        return true;
    }
    if (rows > 1 && s0 == 0) || (cols > 1 && s1 == 0) {
        return false;
    }
    if rows <= 1 || cols <= 1 {
        return true;
    }
    let (mut a, mut b) = (s0, s1);
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    s1 / a >= rows || s0 / a >= cols
}

#[cfg(test)]
mod matrix_layout_tests {
    use super::matrix_elements_are_disjoint;
    #[test]
    fn disjoint_layout_matches_exhaustive_address_check() {
        for rows in 0..7 {
            for cols in 0..7 {
                for s0 in 0..9 {
                    for s1 in 0..9 {
                        let mut addresses = std::collections::HashSet::new();
                        let expected =
                            (0..rows).all(|r| (0..cols).all(|c| addresses.insert(r * s0 + c * s1)));
                        assert_eq!(
                            matrix_elements_are_disjoint(rows, cols, s0, s1),
                            expected,
                            "shape=[{rows},{cols}], stride=[{s0},{s1}]"
                        );
                    }
                }
            }
        }
        assert!(matrix_elements_are_disjoint(
            2,
            2,
            usize::MAX,
            usize::MAX - 1
        ));
        assert!(!matrix_elements_are_disjoint(2, 2, usize::MAX, usize::MAX));
    }
}
