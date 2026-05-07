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

    /// Returns the logical shape of this tensor as a slice.
    ///
    /// The length of the returned slice equals [`ndim()`](Self::ndim).
    /// Each element represents the size of the corresponding dimension.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = TypedTensor::<f32>::new(&[2, 3, 4], DeviceType::Cpu)?;
    /// assert_eq!(t.shape(), &[2, 3, 4]);
    /// ```
    #[inline]
    pub fn shape(&self) -> &[usize] { self.shape.as_slice() }

    /// Returns the element-unit strides of this tensor as a slice.
    ///
    /// Strides are measured in **elements** (not bytes). To compute byte
    /// strides, multiply each stride by `std::mem::size_of::<T>()`.
    ///
    /// For a contiguous (row-major) tensor with shape `[2, 3, 4]`, the
    /// strides will be `[12, 4, 1]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = TypedTensor::<f32>::new(&[2, 3, 4], DeviceType::Cpu)?;
    /// assert_eq!(t.strides(), &[12, 4, 1]);
    /// ```
    #[inline]
    pub fn strides(&self) -> &[usize] { self.strides.as_slice() }

    /// Returns the number of dimensions (rank) of this tensor.
    ///
    /// A scalar has `ndim() == 0`, a vector `ndim() == 1`, a matrix
    /// `ndim() == 2`, and so on up to [`MAX_RANK`].
    #[inline]
    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Returns the total number of logical elements in this tensor.
    ///
    /// Equal to the product of all shape dimensions. A rank-0 (scalar)
    /// tensor has `numel() == 1`.
    ///
    /// This is a cached value — no computation is performed.
    #[inline]
    pub fn numel(&self) -> usize { self.numel }

    /// Returns whether the tensor's memory layout is C-contiguous (row-major).
    ///
    /// A tensor is contiguous when its strides satisfy the row-major formula:
    /// `strides[i] == strides[i+1] * shape[i+1]` for all `i`, with
    /// `strides[ndim-1] == 1`.
    ///
    /// View operations (e.g. [`permute`](crate::tensor::Tensor::permute),
    /// [`transpose`](crate::tensor::Tensor::transpose)) typically produce
    /// non-contiguous tensors. Many kernels require contiguous input;
    /// call [`contiguous()`](crate::tensor::Tensor::contiguous) to densify.
    #[inline]
    pub fn is_contiguous(&self) -> bool { self.is_contiguous }

    /// Returns the element offset from the start of the underlying buffer
    /// to this tensor's logical element 0.
    ///
    /// For freshly allocated tensors this is always 0. Views created by
    /// [`narrow`](crate::tensor::Tensor::narrow) or
    /// [`select`](crate::tensor::Tensor::select) may have a nonzero offset,
    /// indicating they reference a sub-region of a larger buffer.
    ///
    /// The offset is measured in **elements** (not bytes).
    #[inline]
    pub fn offset_elems(&self) -> usize { self.offset_elems }

    /// Returns a shared reference to the underlying [`Buffer`].
    ///
    /// The buffer holds the raw byte storage and knows its device
    /// (CPU or CUDA). Multiple `TypedTensor` views may share the same
    /// buffer via `Arc`.
    #[inline]
    pub fn buffer(&self) -> &Buffer { &self.buffer }

    /// Returns a mutable reference to the underlying [`Buffer`].
    ///
    /// # Safety Considerations
    ///
    /// If other views share this buffer (via `Clone`), mutating through
    /// this reference affects all of them. Ensure exclusive access or
    /// use [`to_owned()`](crate::tensor::Tensor::to_owned) first.
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut Buffer { &mut self.buffer }

    /// Returns a raw const pointer to this tensor's logical element 0.
    ///
    /// The pointer accounts for [`offset_elems`](Self::offset_elems), so it
    /// always points to `buffer.as_ptr() + offset_elems * size_of::<T>()`.
    ///
    /// # Safety
    ///
    /// The pointer is only valid while the underlying `Buffer` is alive.
    /// Dereferencing beyond `numel` elements (accounting for strides) is UB.
    /// For safe element access, prefer [`as_slice()`](Self::as_slice).
    #[inline]
    pub fn data_ptr(&self) -> *const T {
        // SAFETY: `offset_elems` is bounded by the buffer's allocation via
        // constructor invariants — every addressable element lies within the
        // buffer. We only produce a pointer (no dereference here).
        unsafe {
            (self.buffer.as_ptr() as *const T).add(self.offset_elems)
        }
    }

    /// Returns a raw mutable pointer to this tensor's logical element 0.
    ///
    /// Mutable variant of [`data_ptr`](Self::data_ptr). The same safety
    /// constraints apply. Additionally, the caller must ensure no other
    /// references to the buffer's memory are active.
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut T {
        // SAFETY: Same as `data_ptr` — offset is bounded by constructor invariants.
        // Mutable access is gated by `&mut self`.
        unsafe {
            (self.buffer.as_mut_ptr() as *mut T).add(self.offset_elems)
        }
    }

    /// Borrows the contiguous, CPU-resident elements as an immutable slice.
    ///
    /// This is the primary safe way to read tensor data from Rust code.
    /// The returned slice has exactly [`numel()`](Self::numel) elements,
    /// starting from the tensor's logical element 0.
    ///
    /// # Errors
    ///
    /// - Returns an error if the tensor's backing storage is **not on CPU**
    ///   (e.g. it resides on a CUDA device). Use
    ///   [`to_cpu()`](crate::tensor::Tensor::to_cpu) first.
    /// - Returns an error if the tensor is **not contiguous** (i.e. it is
    ///   a strided view). Call
    ///   [`contiguous()`](crate::tensor::Tensor::contiguous) to densify.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = TypedTensor::<f32>::new(&[3], DeviceType::Cpu)?;
    /// let slice: &[f32] = t.as_slice()?;
    /// assert_eq!(slice.len(), 3);
    /// ```
    pub fn as_slice(&self) -> Result<&[T]> {
        self.check_cpu("as_slice")?;
        self.check_contiguous("as_slice")?;
        // SAFETY: Preconditions verified above — contiguous CPU buffer guarantees
        // that `data_ptr()..data_ptr()+numel` is a valid, aligned, initialized
        // range of `T` elements within the buffer's allocation.
        unsafe { Ok(std::slice::from_raw_parts(self.data_ptr(), self.numel)) }
    }

    /// Borrows the contiguous, CPU-resident elements as a mutable slice.
    ///
    /// Mutable variant of [`as_slice`](Self::as_slice). Allows in-place
    /// modification of tensor data from Rust code.
    ///
    /// # Errors
    ///
    /// Same preconditions as [`as_slice`](Self::as_slice):
    /// - Storage must be on CPU.
    /// - Tensor must be contiguous.
    ///
    /// # Safety Considerations
    ///
    /// If other views share the same underlying buffer, mutations through
    /// this slice are visible to all of them. Use
    /// [`to_owned()`](crate::tensor::Tensor::to_owned) to get exclusive storage.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = TypedTensor::<f32>::new(&[3], DeviceType::Cpu)?;
    /// let slice: &mut [f32] = t.as_slice_mut()?;
    /// slice[0] = 1.0;
    /// slice[1] = 2.0;
    /// slice[2] = 3.0;
    /// ```
    pub fn as_slice_mut(&mut self) -> Result<&mut [T]> {
        self.check_cpu("as_slice_mut")?;
        self.check_contiguous("as_slice_mut")?;
        let n = self.numel;
        // SAFETY: Same as `as_slice` — contiguous CPU layout is verified.
        // Mutable access is gated by `&mut self`.
        unsafe { Ok(std::slice::from_raw_parts_mut(self.data_ptr_mut(), n)) }
    }

    // ──────────────────────────── helpers ─────────────────────────────

    /// Validates that the given shape does not exceed [`MAX_RANK`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `shape.len() > MAX_RANK`.
    fn check_rank(shape: &[usize]) -> Result<()> {
        if shape.len() > MAX_RANK {
            return Err(Error::InvalidArgument(format!(
                "rank {} exceeds MAX_RANK={}", shape.len(), MAX_RANK
            )).into());
        }
        Ok(())
    }

    /// Validates that the tensor's buffer resides on the CPU.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DeviceMismatch`] if the buffer is on a non-CPU device,
    /// including the `ctx` string in the error message for diagnostics.
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

    /// Validates that the tensor is C-contiguous.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the tensor's strides do not
    /// match the row-major layout. The error message includes shape and
    /// strides to aid debugging, and suggests calling `.contiguous()`.
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

    /// Computes whether the given `(shape, strides)` pair represents a
    /// C-contiguous (row-major) layout.
    ///
    /// The algorithm walks dimensions in reverse, checking that each stride
    /// equals the running product of all subsequent dimension sizes.
    /// Size-1 dimensions are skipped (they impose no constraint on stride).
    ///
    /// # Special Cases
    ///
    /// - Rank-0 tensors (scalars) are always contiguous.
    /// - Tensors with any zero-sized dimension are considered contiguous
    ///   (they contain no elements regardless of strides).
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
