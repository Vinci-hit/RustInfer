//! Strongly-typed tensor storage.
//!
//! `TypedTensor<T>` is the per-dtype carrier under the [`crate::tensor::Tensor`]
//! enum. It owns its metadata (shape, strides, element offset, contiguity
//! flag) and an [`Arc`]-shared [`Buffer`]. A *view* is just a `TypedTensor`
//! whose metadata describes a sub-region of a shared buffer; strides decide
//! whether indexing stays dense.
//!
//! Invariants maintained by every constructor in this module:
//!   - `shape.len() == strides.len()`
//!   - `numel == shape.product()`
//!   - `is_contiguous == Self::compute_contiguous(shape, strides)`
//!   - every addressable element lies within `buffer[offset_elems .. offset_elems + span]`
//!     where `span` is the stride-reachable range.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::base::allocator::{CpuAllocator, DeviceAllocator};
use crate::base::buffer::Buffer;
use crate::base::error::{Error, Result};
use crate::base::DeviceType;

use super::dims::{Dims, MAX_RANK};
use super::dtype::Dtype;

/// Per-dtype tensor storage carrier.
///
/// Clones are cheap: `Dims` is `Copy`, `Buffer` is `Arc`-backed. Cloning
/// does **not** duplicate data; use [`crate::tensor::Tensor::to_owned`] for that.
#[derive(Clone, Debug)]
pub struct TypedTensor<T: Dtype> {
    /// Logical shape.
    pub(crate) shape: Dims,
    /// Element strides (in units of `T`, not bytes).
    pub(crate) strides: Dims,
    /// Offset from `buffer`'s start to this tensor's logical element 0
    /// (in units of `T`).
    pub(crate) offset_elems: usize,
    /// Cached `shape.product()`.
    pub(crate) numel: usize,
    /// Cached result of [`Self::compute_contiguous`]. Every metadata-mutating
    /// constructor must refresh this.
    pub(crate) is_contiguous: bool,
    /// Underlying storage (potentially shared with other views).
    pub(crate) buffer: Buffer,
    _phantom: PhantomData<T>,
}

impl<T: Dtype> TypedTensor<T> {
    // ────────────────────────── constructors ──────────────────────────

    /// Allocate a fresh, zeroed, contiguous tensor on `device`.
    pub fn new(shape: &[usize], device: DeviceType) -> Result<Self> {
        Self::check_rank(shape)?;
        let shape_d = Dims::from_slice(shape);
        let numel = shape_d.product();
        let size_bytes = numel.checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidArgument("shape too large".into()))?;

        let allocator: Arc<dyn DeviceAllocator + Send + Sync> = match device {
            DeviceType::Cpu => Arc::new(CpuAllocator),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                Arc::new(crate::base::allocator::CachingCudaAllocator::instance())
            }
        };
        let buffer = Buffer::new(size_bytes, allocator)?;

        Ok(Self {
            shape: shape_d,
            strides: Dims::contiguous_strides_for(shape),
            offset_elems: 0,
            numel,
            is_contiguous: true,
            buffer,
            _phantom: PhantomData,
        })
    }

    /// Wrap a pre-existing `Buffer` as a contiguous tensor. The buffer size
    /// must match `shape.product() * size_of::<T>()` exactly.
    pub fn from_buffer(buffer: Buffer, shape: &[usize]) -> Result<Self> {
        Self::check_rank(shape)?;
        let shape_d = Dims::from_slice(shape);
        let numel = shape_d.product();
        let expected = numel * std::mem::size_of::<T>();
        if buffer.len_bytes() != expected {
            return Err(Error::InvalidArgument(format!(
                "from_buffer: buffer has {} bytes, expected {} for shape {:?} dtype {:?}",
                buffer.len_bytes(), expected, shape, T::DTYPE
            )).into());
        }
        Ok(Self {
            shape: shape_d,
            strides: Dims::contiguous_strides_for(shape),
            offset_elems: 0,
            numel,
            is_contiguous: true,
            buffer,
            _phantom: PhantomData,
        })
    }

    /// Low-level constructor for view operations. Caller must guarantee
    /// that `buffer` is large enough to cover every addressable element
    /// under `(shape, strides, offset_elems)`.
    pub(crate) fn from_parts(
        buffer: Buffer,
        shape: Dims,
        strides: Dims,
        offset_elems: usize,
    ) -> Self {
        debug_assert_eq!(shape.len(), strides.len());
        let numel = shape.product();
        let is_contiguous = Self::compute_contiguous(&shape, &strides);
        Self {
            shape,
            strides,
            offset_elems,
            numel,
            is_contiguous,
            buffer,
            _phantom: PhantomData,
        }
    }

    // ───────────────────────────── getters ────────────────────────────

    #[inline] pub fn shape(&self)        -> &[usize] { self.shape.as_slice() }
    #[inline] pub fn strides(&self)      -> &[usize] { self.strides.as_slice() }
    #[inline] pub fn ndim(&self)         -> usize    { self.shape.len() }
    #[inline] pub fn numel(&self)        -> usize    { self.numel }
    #[inline] pub fn is_contiguous(&self)-> bool     { self.is_contiguous }
    #[inline] pub fn offset_elems(&self) -> usize    { self.offset_elems }
    #[inline] pub fn buffer(&self)       -> &Buffer  { &self.buffer }
    #[inline] pub fn buffer_mut(&mut self) -> &mut Buffer { &mut self.buffer }

    /// Raw pointer to the *logical* element 0 of this (possibly offset)
    /// view. Always returns `buffer.as_ptr() + offset_elems * size_of::<T>()`.
    #[inline]
    pub fn data_ptr(&self) -> *const T {
        unsafe {
            (self.buffer.as_ptr() as *const T).add(self.offset_elems)
        }
    }

    /// Mutable variant of [`data_ptr`](Self::data_ptr).
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut T {
        unsafe {
            (self.buffer.as_mut_ptr() as *mut T).add(self.offset_elems)
        }
    }

    /// Borrow the contiguous, CPU-resident elements as a slice.
    ///
    /// # Errors
    /// - [`Error::NotContiguous`] if this tensor is a strided view.
    /// - [`Error::DeviceMismatch`] if backing storage is not on CPU.
    pub fn as_slice(&self) -> Result<&[T]> {
        self.check_cpu("as_slice")?;
        self.check_contiguous("as_slice")?;
        unsafe { Ok(std::slice::from_raw_parts(self.data_ptr(), self.numel)) }
    }

    /// Mutable variant of [`as_slice`](Self::as_slice).
    pub fn as_slice_mut(&mut self) -> Result<&mut [T]> {
        self.check_cpu("as_slice_mut")?;
        self.check_contiguous("as_slice_mut")?;
        let n = self.numel;
        unsafe { Ok(std::slice::from_raw_parts_mut(self.data_ptr_mut(), n)) }
    }

    // ──────────────────────────── helpers ─────────────────────────────

    fn check_rank(shape: &[usize]) -> Result<()> {
        if shape.len() > MAX_RANK {
            return Err(Error::InvalidArgument(format!(
                "rank {} exceeds MAX_RANK={}", shape.len(), MAX_RANK
            )).into());
        }
        Ok(())
    }

    fn check_cpu(&self, ctx: &str) -> Result<()> {
        if self.buffer.device() != DeviceType::Cpu {
            return Err(Error::DeviceMismatch {
                expected: DeviceType::Cpu,
                actual: self.buffer.device(),
                in_method: ctx.to_string(),
            }.into());
        }
        Ok(())
    }

    fn check_contiguous(&self, ctx: &str) -> Result<()> {
        if !self.is_contiguous {
            return Err(Error::InvalidArgument(format!(
                "{}: tensor is not contiguous (shape={:?}, strides={:?}); \
                 call .contiguous() first",
                ctx, self.shape(), self.strides()
            )).into());
        }
        Ok(())
    }

    /// C-contiguous iff strides match the row-major formula derived from
    /// shape. Dimensions of size 0 or 1 impose no stride constraint.
    pub(crate) fn compute_contiguous(shape: &Dims, strides: &Dims) -> bool {
        let n = shape.len();
        if n == 0 { return true; }
        if shape.iter().any(|&d| d == 0) { return true; }
        let mut expected = 1usize;
        for i in (0..n).rev() {
            let d = shape[i];
            if d == 1 { continue; }
            if strides[i] != expected { return false; }
            expected = expected.saturating_mul(d);
        }
        true
    }
}
