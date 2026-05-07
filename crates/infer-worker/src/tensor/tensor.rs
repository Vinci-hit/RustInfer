//! The dtype-erased [`Tensor`] facade.
//!
//! `Tensor` is an enum over `TypedTensor<T>` for each supported dtype.
//! Almost every public method is a one-line dispatch to the inner typed
//! tensor; callers who need raw typed access go through
//! [`Tensor::as_f32`] / [`Tensor::as_bf16`] / etc.

use half::{bf16, f16};
use std::sync::Arc;

use crate::base::allocator::{CpuAllocator, DeviceAllocator};
use crate::base::buffer::Buffer;
use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};

use super::dims::Dims;
use super::typed::TypedTensor;

/// Dtype-erased tensor. Every variant wraps a `TypedTensor<T>` carrying the
/// same storage layout — only the element type differs.
#[derive(Clone, Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    I32(TypedTensor<i32>),
    I8(TypedTensor<i8>),
    F16(TypedTensor<f16>),
    BF16(TypedTensor<bf16>),
}

/// Dispatch a method call uniformly across all [`Tensor`] dtype variants.
///
/// This is the primary mechanism for implementing dtype-erased methods that need
/// to invoke the same operation on the underlying [`TypedTensor<T>`]. It performs
/// an exhaustive match over all five active variants (F32, I32, I8, F16, BF16)
/// and calls the provided method on each typed tensor.
///
/// # Syntax
///
/// ```ignore
/// dispatch_on_tensor!(self, method_name(arg1, arg2, ...))
/// ```
///
/// This expands to:
/// ```ignore
/// match self {
///     Tensor::F32(t)  => t.method_name(arg1, arg2, ...),
///     Tensor::I32(t)  => t.method_name(arg1, arg2, ...),
///     Tensor::I8(t)   => t.method_name(arg1, arg2, ...),
///     Tensor::F16(t)  => t.method_name(arg1, arg2, ...),
///     Tensor::BF16(t) => t.method_name(arg1, arg2, ...),
/// }
/// ```
///
/// # Requirements
///
/// - All five `TypedTensor<T>` variants must support the method with identical signature
/// - Return types must be identical (the match expression produces a uniform type)
/// - Typically used for getter methods that return `&[usize]`, `usize`, `bool`, etc.
///
/// # Examples
///
/// Getting shape from any dtype tensor:
/// ```ignore
/// pub fn shape(&self) -> &[usize] {
///     dispatch_on_tensor!(self, shape())
/// }
/// ```
///
/// Getting the number of elements:
/// ```ignore
/// pub fn numel(&self) -> usize {
///     dispatch_on_tensor!(self, numel())
/// }
/// ```
///
/// # Design Note
///
/// This macro is marked `#[macro_export]` so it's available to external crates
/// for implementing similar dtype-dispatch patterns. It's one of the three
/// critical mechanisms for the dtype-erasure pattern alongside `typed_ref!`
/// and `typed_mut!`.
#[macro_export]
macro_rules! dispatch_on_tensor {
    ($self:expr, $method:ident ( $($args:expr),* $(,)? )) => {
        match $self {
            $crate::tensor::Tensor::F32(t)  => t.$method($($args),*),
            $crate::tensor::Tensor::I32(t)  => t.$method($($args),*),
            $crate::tensor::Tensor::I8(t)   => t.$method($($args),*),
            $crate::tensor::Tensor::F16(t)  => t.$method($($args),*),
            $crate::tensor::Tensor::BF16(t) => t.$method($($args),*),
        }
    };
}

impl Tensor {
    // ────────────────────── constructors (allocating) ─────────────────────

    /// Allocate an uninitialised contiguous tensor of the given shape /
    /// dtype on `device`. Storage contents are unspecified.
    pub fn empty(shape: &[usize], dtype: DataType, device: DeviceType) -> Result<Self> {
        match dtype {
            DataType::F32  => Ok(Tensor::F32 (TypedTensor::<f32 >::new(shape, device)?)),
            DataType::I32  => Ok(Tensor::I32 (TypedTensor::<i32 >::new(shape, device)?)),
            DataType::I8   => Ok(Tensor::I8  (TypedTensor::<i8  >::new(shape, device)?)),
            DataType::F16  => Ok(Tensor::F16 (TypedTensor::<f16 >::new(shape, device)?)),
            DataType::BF16 => Ok(Tensor::BF16(TypedTensor::<bf16>::new(shape, device)?)),
            _ => Err(Error::InvalidArgument(
                format!("empty: unsupported dtype {:?}", dtype)
            ).into()),
        }
    }

    /// Alias for [`Tensor::empty`] preserved for historical call sites.
    #[inline]
    pub fn new(shape: &[usize], dtype: DataType, device: DeviceType) -> Result<Self> {
        Self::empty(shape, dtype, device)
    }

    /// Allocate a zero-initialised contiguous tensor.
    pub fn zeros(shape: &[usize], dtype: DataType, device: DeviceType) -> Result<Self> {
        let mut t = Self::empty(shape, dtype, device)?;
        t.zero_()?;
        Ok(t)
    }

    /// Allocate a contiguous tensor filled with `value` (float semantics).
    /// For integer dtypes the value is truncated.
    pub fn full(
        shape: &[usize],
        value: f64,
        dtype: DataType,
        device: DeviceType,
    ) -> Result<Self> {
        let mut t = Self::empty(shape, dtype, device)?;
        t.fill_(value)?;
        Ok(t)
    }

    /// Allocate a contiguous tensor filled with `1`.
    pub fn ones(shape: &[usize], dtype: DataType, device: DeviceType) -> Result<Self> {
        Self::full(shape, 1.0, dtype, device)
    }

    /// Build a tensor that directly adopts an existing contiguous `Buffer`.
    pub fn from_buffer(buffer: Buffer, shape: &[usize], dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::F32  => Ok(Tensor::F32 (TypedTensor::<f32 >::from_buffer(buffer, shape)?)),
            DataType::I32  => Ok(Tensor::I32 (TypedTensor::<i32 >::from_buffer(buffer, shape)?)),
            DataType::I8   => Ok(Tensor::I8  (TypedTensor::<i8  >::from_buffer(buffer, shape)?)),
            DataType::F16  => Ok(Tensor::F16 (TypedTensor::<f16 >::from_buffer(buffer, shape)?)),
            DataType::BF16 => Ok(Tensor::BF16(TypedTensor::<bf16>::from_buffer(buffer, shape)?)),
            _ => Err(Error::InvalidArgument(
                format!("from_buffer: unsupported dtype {:?}", dtype)
            ).into()),
        }
    }

    // ───────────────────────────── metadata ─────────────────────────────

    /// Logical shape. O(1) borrow, no allocation.
    #[inline]
    pub fn shape(&self) -> &[usize] { dispatch_on_tensor!(self, shape()) }

    /// Element-unit strides. O(1) borrow, no allocation.
    #[inline]
    pub fn strides(&self) -> &[usize] { dispatch_on_tensor!(self, strides()) }

    /// Byte-unit strides (strides × `sizeof(dtype)`). Allocates a `Dims`.
    pub fn byte_strides(&self) -> Dims {
        let es = self.dtype().size_in_bytes();
        let mut out = Dims::new();
        for &s in self.strides() { out.push(s * es); }
        out
    }

    /// Returns the number of dimensions (rank) of this tensor.
    ///
    /// Equivalent to `self.shape().len()`. A scalar has `ndim() == 0`,
    /// a vector `ndim() == 1`, a matrix `ndim() == 2`, etc.
    #[inline]
    pub fn ndim(&self) -> usize { dispatch_on_tensor!(self, ndim()) }

    /// Returns the total number of logical elements in this tensor.
    ///
    /// Equal to the product of all dimensions in [`shape()`](Self::shape).
    /// For a rank-0 scalar tensor, `numel() == 1`.
    #[inline]
    pub fn numel(&self) -> usize { dispatch_on_tensor!(self, numel()) }

    /// Returns whether the tensor is stored in C-contiguous (row-major) order.
    ///
    /// A contiguous tensor has strides that satisfy `strides[i] == strides[i+1] * shape[i+1]`
    /// for all valid `i`, with `strides[ndim-1] == 1`.
    ///
    /// Many kernel operations require contiguous input. Use
    /// [`contiguous()`](Self::contiguous) to produce a dense copy if needed.
    #[inline]
    pub fn is_contiguous(&self) -> bool { dispatch_on_tensor!(self, is_contiguous()) }

    /// Returns the element offset from the buffer start to this tensor's
    /// logical element 0.
    ///
    /// Measured in elements (not bytes). For freshly allocated tensors this
    /// is always 0; views created by [`narrow`](Self::narrow) or
    /// [`select`](Self::select) may have nonzero offsets.
    #[inline]
    pub fn storage_offset(&self) -> usize { dispatch_on_tensor!(self, offset_elems()) }

    /// "This tensor *exclusively* owns its backing storage and exactly fills
    /// it" predicate.
    ///
    /// Returns true iff:
    /// - `is_contiguous() == true`,
    /// - `storage_offset() == 0`,
    /// - the underlying `Buffer` has length **exactly** `numel * sizeof(dtype)`.
    ///
    /// This is the correct precondition for any "fast-path bulk copy of the
    /// whole buffer" (cross-device migration, dtype cast, deep clone). The
    /// looser `is_contiguous() && storage_offset() == 0` test is **not**
    /// sufficient: a prefix-narrowed view (e.g. `base.narrow(0, 0, n)` with
    /// `n < base.shape[0]`) is contiguous and offset 0, yet its buffer still
    /// covers the full base storage.
    pub fn owns_storage_tightly(&self) -> bool {
        if !self.is_contiguous() || self.storage_offset() != 0 {
            return false;
        }
        let expected = self.numel() * self.dtype().size_in_bytes();
        self.buffer().len_bytes() == expected
    }

    /// Dtype tag.
    pub fn dtype(&self) -> DataType {
        match self {
            Tensor::F32(_)  => DataType::F32,
            Tensor::I32(_)  => DataType::I32,
            Tensor::I8(_)   => DataType::I8,
            Tensor::F16(_)  => DataType::F16,
            Tensor::BF16(_) => DataType::BF16,
        }
    }

    /// Device hosting the backing storage.
    #[inline]
    pub fn device(&self) -> DeviceType { self.buffer().device() }

    /// Borrow the underlying storage Buffer.
    pub fn buffer(&self) -> &Buffer {
        match self {
            Tensor::F32(t)  => t.buffer(),
            Tensor::I32(t)  => t.buffer(),
            Tensor::I8(t)   => t.buffer(),
            Tensor::F16(t)  => t.buffer(),
            Tensor::BF16(t) => t.buffer(),
        }
    }

    /// Mutable borrow of the underlying storage Buffer.
    pub fn buffer_mut(&mut self) -> &mut Buffer {
        match self {
            Tensor::F32(t)  => t.buffer_mut(),
            Tensor::I32(t)  => t.buffer_mut(),
            Tensor::I8(t)   => t.buffer_mut(),
            Tensor::F16(t)  => t.buffer_mut(),
            Tensor::BF16(t) => t.buffer_mut(),
        }
    }

    /// Raw pointer to the logical element 0. Byte-level; callers that need
    /// typed access should go through [`Tensor::as_f32`] etc.
    pub fn data_ptr(&self) -> *const u8 {
        let base = self.buffer().as_ptr();
        let offset_bytes = self.storage_offset() * self.dtype().size_in_bytes();
        unsafe { base.add(offset_bytes) }
    }

    /// Mutable variant of [`Tensor::data_ptr`].
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        let offset_bytes = self.storage_offset() * self.dtype().size_in_bytes();
        let base = self.buffer_mut().as_mut_ptr();
        unsafe { base.add(offset_bytes) }
    }

    // ─────────────────── typed accessors (read-only) ────────────────────

    /// Attempts to borrow the inner `TypedTensor<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `F32`.
    pub fn as_f32(&self)  -> Result<&TypedTensor<f32 >> { typed_ref!(self, F32 , "F32") }

    /// Attempts to borrow the inner `TypedTensor<i32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `I32`.
    pub fn as_i32(&self)  -> Result<&TypedTensor<i32 >> { typed_ref!(self, I32 , "I32") }

    /// Attempts to borrow the inner `TypedTensor<i8>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `I8`.
    pub fn as_i8(&self)   -> Result<&TypedTensor<i8  >> { typed_ref!(self, I8  , "I8") }

    /// Attempts to borrow the inner `TypedTensor<f16>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `F16`.
    pub fn as_f16(&self)  -> Result<&TypedTensor<f16 >> { typed_ref!(self, F16 , "F16") }

    /// Attempts to borrow the inner `TypedTensor<bf16>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `BF16`.
    pub fn as_bf16(&self) -> Result<&TypedTensor<bf16>> { typed_ref!(self, BF16, "BF16") }

    /// Attempts to mutably borrow the inner `TypedTensor<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `F32`.
    pub fn as_f32_mut(&mut self)  -> Result<&mut TypedTensor<f32 >> { typed_mut!(self, F32 , "F32") }

    /// Attempts to mutably borrow the inner `TypedTensor<i32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `I32`.
    pub fn as_i32_mut(&mut self)  -> Result<&mut TypedTensor<i32 >> { typed_mut!(self, I32 , "I32") }

    /// Attempts to mutably borrow the inner `TypedTensor<i8>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `I8`.
    pub fn as_i8_mut(&mut self)   -> Result<&mut TypedTensor<i8  >> { typed_mut!(self, I8  , "I8") }

    /// Attempts to mutably borrow the inner `TypedTensor<f16>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `F16`.
    pub fn as_f16_mut(&mut self)  -> Result<&mut TypedTensor<f16 >> { typed_mut!(self, F16 , "F16") }

    /// Attempts to mutably borrow the inner `TypedTensor<bf16>`.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor's dtype is not `BF16`.
    pub fn as_bf16_mut(&mut self) -> Result<&mut TypedTensor<bf16>> { typed_mut!(self, BF16, "BF16") }

    // ───────────────────────── in-place fills ───────────────────────────

    /// `self.fill_(v)` → set every element to `v`. Requires contiguous
    /// storage (strided views are rejected).
    pub fn fill_(&mut self, value: f64) -> Result<()> {
        if !self.is_contiguous() {
            return Err(Error::InvalidArgument(
                "fill_: tensor must be contiguous".into()
            ).into());
        }
        // CPU fast path: write through typed slice.
        if self.device() == DeviceType::Cpu {
            match self {
                Tensor::F32(t)  => { for x in t.as_slice_mut()? { *x = value as f32; } }
                Tensor::I32(t)  => { for x in t.as_slice_mut()? { *x = value as i32; } }
                Tensor::I8(t)   => { for x in t.as_slice_mut()? { *x = value as i8;  } }
                Tensor::F16(t)  => { for x in t.as_slice_mut()? { *x = f16 ::from_f64(value); } }
                Tensor::BF16(t) => { for x in t.as_slice_mut()? { *x = bf16::from_f64(value); } }
            }
            return Ok(());
        }
        // CUDA fill kernel not yet implemented.
        #[cfg(feature = "cuda")]
        {
            return Err(Error::InvalidArgument(
                "fill_: CUDA fill kernel not implemented; \
                 use a CPU tensor or implement a dedicated kernel"
                    .into(),
            )
            .into());
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!()
    }

    /// `self.zero_()` → zero every byte of backing storage.
    pub fn zero_(&mut self) -> Result<()> {
        if !self.is_contiguous() {
            return Err(Error::InvalidArgument(
                "zero_: tensor must be contiguous".into()
            ).into());
        }
        self.buffer_mut().zero_out()
    }

    // ───────────────────────── internal helpers ─────────────────────────

    /// Build a view from (shape, strides, offset) that shares `self`'s
    /// buffer. Used by every zero-copy view operation.
    pub(crate) fn from_view_parts(
        &self,
        shape: Dims,
        strides: Dims,
        offset_elems: usize,
    ) -> Self {
        macro_rules! build {
            ($variant:ident, $t:ty) => {{
                let buf = match self {
                    Tensor::$variant(tt) => tt.buffer().clone(),
                    // SAFETY: The outer match already resolved the variant;
                    // the macro is expanded once per arm, so this is unreachable
                    // by construction.
                    _ => unreachable!("from_view_parts: variant mismatch (bug)"),
                };
                Tensor::$variant(TypedTensor::<$t>::from_parts(buf, shape, strides, offset_elems))
            }};
        }
        match self {
            Tensor::F32(_)  => build!(F32 , f32 ),
            Tensor::I32(_)  => build!(I32 , i32 ),
            Tensor::I8(_)   => build!(I8  , i8  ),
            Tensor::F16(_)  => build!(F16 , f16 ),
            Tensor::BF16(_) => build!(BF16, bf16),
        }
    }

    /// Allocate a fresh contiguous [`Buffer`] on `device` sized for this
    /// tensor's `numel` elements. Currently unused outside tests but kept
    /// as a small convenience for future kernel plumbing.
    #[allow(dead_code)]
    pub(crate) fn alloc_contiguous_buffer(&self, device: DeviceType) -> Result<Buffer> {
        let size_bytes = self.numel() * self.dtype().size_in_bytes();
        let allocator: Arc<dyn DeviceAllocator + Send + Sync> = match device {
            DeviceType::Cpu => Arc::new(CpuAllocator),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                Arc::new(crate::base::allocator::CachingCudaAllocator::instance())
            }
        };
        Buffer::new(size_bytes, allocator)
    }
}

// ── Typed-accessor macros (crate-private dispatch for type-safe access) ─────
//
// These macros provide the inverse operation to `dispatch_on_tensor!`:
// instead of calling a method uniformly on all variants, they extract a
// reference (or mutable reference) to the underlying `TypedTensor<T>` for a
// *specific* dtype, returning an error if the tensor has a different dtype.
//
// Both are extensively used in the `.as_f32()`, `.as_i32_mut()`, etc. public
// methods and in private helpers that need type-specific access.

/// Extract an immutable typed reference, or error if dtype mismatch.
///
/// # Syntax
///
/// ```ignore
/// typed_ref!(tensor, F32, "F32")
/// ```
///
/// This expands to:
/// ```ignore
/// match tensor {
///     Tensor::F32(t) => Ok(t),
///     other => Err(Error::InvalidArgument(format!(
///         "typed access: expected F32, found {:?}", other.dtype()
///     )).into()),
/// }
/// ```
///
/// # Arguments
///
/// - `$self`: The tensor to check (usually `self` or `&self`)
/// - `$variant`: The enum variant name (F32, I32, I8, F16, or BF16) — **without** quotes
/// - `$name`: A display name for error messages — **must be** a string literal (with quotes)
///
/// # Returns
///
/// - `Ok(&TypedTensor<T>)` if the tensor's variant matches
/// - `Err(Error::InvalidArgument)` with a message like "typed access: expected F32, found BF16"
///
/// # Usage
///
/// Implementing `.as_f32()`:
/// ```ignore
/// pub fn as_f32(&self) -> Result<&TypedTensor<f32>> {
///     typed_ref!(self, F32, "F32")
/// }
/// ```
///
/// Typically called by users who have a `Tensor` but need element-level access:
/// ```ignore
/// let tensor = Tensor::F32(...);
/// if let Ok(typed) = tensor.as_f32() {
///     let slice = typed.as_slice()?;
///     // Now we can work with &[f32]
/// }
/// ```
macro_rules! typed_ref {
    ($self:expr, $variant:ident, $name:literal) => {
        match $self {
            Tensor::$variant(t) => Ok(t),
            other => Err(Error::InvalidArgument(format!(
                "typed access: expected {}, found {:?}", $name, other.dtype()
            )).into()),
        }
    };
}

/// Extract a mutable typed reference, or error if dtype mismatch.
///
/// Identical to [`typed_ref!`] but returns `&mut TypedTensor<T>` instead.
/// Requires the input to be mutable, enabling element modification.
///
/// # Syntax
///
/// ```ignore
/// typed_mut!(tensor, F32, "F32")
/// ```
///
/// This expands to:
/// ```ignore
/// match tensor {
///     Tensor::F32(t) => Ok(t),
///     other => {
///         let got = other.dtype();
///         Err(Error::InvalidArgument(format!(
///             "typed access (mut): expected F32, found {:?}", got
///         )).into())
///     }
/// }
/// ```
///
/// # Arguments
///
/// - `$self`: The **mutable** tensor to check (usually `&mut self` or `&mut tensor`)
/// - `$variant`: The enum variant name (F32, I32, I8, F16, or BF16) — **without** quotes
/// - `$name`: A display name for error messages — **must be** a string literal (with quotes)
///
/// # Returns
///
/// - `Ok(&mut TypedTensor<T>)` if the tensor's variant matches
/// - `Err(Error::InvalidArgument)` if dtype mismatch
///
/// # Usage
///
/// Implementing `.as_f32_mut()`:
/// ```ignore
/// pub fn as_f32_mut(&mut self) -> Result<&mut TypedTensor<f32>> {
///     typed_mut!(self, F32, "F32")
/// }
/// ```
///
/// Modifying a typed tensor:
/// ```ignore
/// let mut tensor = Tensor::F32(...);
/// if let Ok(typed) = tensor.as_f32_mut() {
///     let slice = typed.as_slice_mut()?;
///     slice[0] = 42.0;  // Modify element 0
/// }
/// ```
macro_rules! typed_mut {
    ($self:expr, $variant:ident, $name:literal) => {
        match $self {
            Tensor::$variant(t) => Ok(t),
            other => {
                let got = other.dtype();
                Err(Error::InvalidArgument(format!(
                    "typed access (mut): expected {}, found {:?}", $name, got
                )).into())
            }
        }
    };
}
pub(crate) use typed_mut;
pub(crate) use typed_ref;
