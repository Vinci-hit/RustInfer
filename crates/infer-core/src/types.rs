//! Domain value types — pure math, zero dependencies.

use half::{bf16, f16};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{OnceLock, RwLock};

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DTypeId(pub u16);

impl DTypeId {
    pub const F32: DTypeId = DTypeId(0);
    pub const F16: DTypeId = DTypeId(1);
    pub const BF16: DTypeId = DTypeId(2);
    pub const I32: DTypeId = DTypeId(3);
    pub const I8: DTypeId = DTypeId(4);
    pub const F8E4M3: DTypeId = DTypeId(5);
    pub const F8E5M2: DTypeId = DTypeId(6);
    pub const U8: DTypeId = DTypeId(7);
    pub const U32: DTypeId = DTypeId(8);

    pub fn register(spec: DTypeSpec) -> DTypeId {
        static NEXT_ID: AtomicU16 = AtomicU16::new(1024);
        let id = DTypeId(NEXT_ID.fetch_add(1, Ordering::Relaxed));
        dtype_registry()
            .write()
            .expect("dtype registry poisoned")
            .insert(id.0, spec);
        id
    }

    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::U8 => 1,
            id => dtype_registry()
                .read()
                .expect("dtype registry poisoned")
                .get(&id.0)
                .map(|spec| spec.size_bytes)
                .unwrap_or(0),
        }
    }

    pub fn is_float(self) -> bool {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2 => true,
            Self::I32 | Self::I8 | Self::U8 | Self::U32 => false,
            id => dtype_registry()
                .read()
                .expect("dtype registry poisoned")
                .get(&id.0)
                .map(|spec| spec.is_float)
                .unwrap_or(false),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DTypeSpec {
    pub size_bytes: usize,
    pub is_float: bool,
    pub name: &'static str,
}

fn dtype_registry() -> &'static RwLock<HashMap<u16, DTypeSpec>> {
    static REGISTRY: OnceLock<RwLock<HashMap<u16, DTypeSpec>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    BF16,
    F8E4M3,
    I32,
    I8,
}
impl DataType {
    #[inline]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DataType::F32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F8E4M3 | DataType::I8 => 1,
        }
    }

    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F8E4M3
        )
    }
}

pub trait Dtype: Copy + Send + Sync + 'static + fmt::Debug {
    const DATA_TYPE: DataType;
    const SIZE_BYTES: usize;
    const ID: DTypeId;

    /// Convert one host scalar to the backend-independent reference format.
    fn read_f64(raw: &Self) -> f64;

    /// Convert a backend-independent reference scalar to this dtype.
    fn write_f64(v: f64) -> Self;
}
impl Dtype for f32 {
    const DATA_TYPE: DataType = DataType::F32;
    const SIZE_BYTES: usize = 4;
    const ID: DTypeId = DTypeId::F32;

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
    const ID: DTypeId = DTypeId::F16;

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
    const ID: DTypeId = DTypeId::BF16;

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
    const ID: DTypeId = DTypeId::I32;

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
    const ID: DTypeId = DTypeId::I8;

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
