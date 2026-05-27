//! `Tensor<T, D>` — the core domain object.
//!
//! A typed, device-aware view over an `Arc<Storage<D>>`. `Clone` and
//! `view` are O(1) Arc bumps; the underlying allocation is freed
//! automatically when the last tensor referencing it drops.

use std::marker::PhantomData;
use std::sync::Arc;

use super::ports::{MemoryPort, OpError, OpResult};
use super::storage::Storage;
use super::types::{Dtype, Shape, Strides};

/// Strongly-typed, device-aware tensor.
/// `T` = element dtype, `D` = device (enforced at compile time).
///
/// Memory is owned by an `Arc<Storage<D>>`. Multiple `Tensor`s may share the
/// same storage (e.g. tied embedding/lm_head, reshape views). The storage is
/// freed when the last reference drops.
pub struct Tensor<T: Dtype, D: MemoryPort> {
    pub(crate) storage: Arc<Storage<D>>,
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) offset_elems: usize,
    pub(crate) numel: usize,
    pub(crate) is_contiguous: bool,
    pub(crate) _marker: PhantomData<T>,
}

impl<T: Dtype, D: MemoryPort> Tensor<T, D> {
    // ─── Construction ──────────────────────────────────────────────

    /// Allocate a contiguous, zero-initialized tensor on `device`.
    pub fn zeros(shape: impl Into<Shape>, device: &D) -> OpResult<Self> {
        let shape = shape.into();
        let numel = shape.numel();
        let size_bytes = numel * T::SIZE_BYTES;
        let storage = Storage::alloc(device, size_bytes)?;
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage,
            shape,
            strides,
            offset_elems: 0,
            numel,
            is_contiguous: true,
            _marker: PhantomData,
        })
    }

    /// Allocate a contiguous tensor on `device` and upload `data` into it.
    /// `data.len()` must equal `shape.numel()`.
    pub fn from_host_slice(
        data: &[T],
        shape: impl Into<Shape>,
        device: &D,
    ) -> OpResult<Self> {
        let shape = shape.into();
        let numel = shape.numel();
        if data.len() != numel {
            return Err(OpError::Shape(format!(
                "from_host_slice: data.len()={} != shape.numel()={}",
                data.len(),
                numel
            )));
        }
        let size_bytes = numel * T::SIZE_BYTES;
        let storage = Storage::alloc(device, size_bytes)?;
        if size_bytes > 0 {
            // SAFETY: storage was just allocated with `size_bytes` bytes.
            unsafe {
                let dst = std::ptr::NonNull::new_unchecked(storage.ptr());
                device.upload(dst, data.as_ptr() as *const u8, size_bytes)?;
            }
        }
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage,
            shape,
            strides,
            offset_elems: 0,
            numel,
            is_contiguous: true,
            _marker: PhantomData,
        })
    }

    /// Allocate from an already-prepared host byte buffer (used by loaders
    /// that have done dtype casting). `bytes.len()` must equal
    /// `shape.numel() * T::SIZE_BYTES`.
    pub fn from_host_bytes(
        bytes: &[u8],
        shape: impl Into<Shape>,
        device: &D,
    ) -> OpResult<Self> {
        let shape = shape.into();
        let numel = shape.numel();
        let size_bytes = numel * T::SIZE_BYTES;
        if bytes.len() != size_bytes {
            return Err(OpError::Shape(format!(
                "from_host_bytes: bytes.len()={} != expected={}",
                bytes.len(),
                size_bytes
            )));
        }
        let storage = Storage::alloc(device, size_bytes)?;
        if size_bytes > 0 {
            // SAFETY: storage just allocated with size_bytes bytes.
            unsafe {
                let dst = std::ptr::NonNull::new_unchecked(storage.ptr());
                device.upload(dst, bytes.as_ptr(), size_bytes)?;
            }
        }
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage,
            shape,
            strides,
            offset_elems: 0,
            numel,
            is_contiguous: true,
            _marker: PhantomData,
        })
    }

    /// Synchronously download the tensor's contents into a fresh `Vec<T>`.
    /// Requires the tensor to be contiguous.
    pub fn to_host_vec(&self) -> OpResult<Vec<T>> {
        if !self.is_contiguous {
            return Err(OpError::NotContiguous(self.shape));
        }
        let mut out: Vec<T> = Vec::with_capacity(self.numel);
        let size_bytes = self.numel * T::SIZE_BYTES;
        if size_bytes > 0 {
            // SAFETY: source is `self.numel * SIZE_BYTES` valid bytes
            // starting at `data_ptr()`; destination is freshly reserved.
            unsafe {
                let src = std::ptr::NonNull::new_unchecked(self.data_ptr() as *mut u8);
                self.storage.device().download(out.as_mut_ptr() as *mut u8, src, size_bytes)?;
                out.set_len(self.numel);
            }
        } else {
            // SAFETY: empty vec.
            unsafe { out.set_len(0); }
        }
        Ok(out)
    }

    /// Construct a view sharing the same storage with a custom shape /
    /// strides / element offset. Caller is responsible for asserting
    /// the view stays within the storage bounds.
    pub fn view_raw(
        &self,
        shape: Shape,
        strides: Strides,
        offset_elems: usize,
        is_contiguous: bool,
    ) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset_elems,
            numel: shape.numel(),
            is_contiguous,
            _marker: PhantomData,
        }
    }

    /// Construct a contiguous view with a new shape (must be compatible
    /// numel with the original). Used for reshape-without-copy.
    pub fn view_contiguous(&self, shape: Shape) -> OpResult<Self> {
        if !self.is_contiguous {
            return Err(OpError::NotContiguous(self.shape));
        }
        if shape.numel() != self.numel {
            return Err(OpError::Shape(format!(
                "view_contiguous: numel mismatch {} -> {}",
                self.numel,
                shape.numel()
            )));
        }
        let strides = shape.contiguous_strides();
        Ok(self.view_raw(shape, strides, self.offset_elems, true))
    }

    /// Slice along `dim` from `start` for `length` elements.
    ///
    /// Zero-copy: returns a view sharing the same `Arc<Storage>`. The result
    /// is non-contiguous unless `dim == 0` and `start == 0` (or the slice
    /// covers the full extent), but downstream kernels that accept row/col
    /// strides (rope, scatter, matmul) handle this directly.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> OpResult<Self> {
        let shape = self.shape.as_slice();
        if dim >= shape.len() {
            return Err(OpError::Shape(format!(
                "narrow: dim {} out of range (ndim={})", dim, shape.len(),
            )));
        }
        if start + length > shape[dim] {
            return Err(OpError::Shape(format!(
                "narrow: dim {} out of bounds (start={}, length={}, extent={})",
                dim, start, length, shape[dim],
            )));
        }
        let strides = self.strides.as_slice();
        let mut new_shape_vec: Vec<usize> = shape.to_vec();
        new_shape_vec[dim] = length;
        let new_shape = Shape::from_slice(&new_shape_vec);
        let extra_offset = start * strides[dim] as usize;
        let new_offset = self.offset_elems + extra_offset;
        // After narrow along dim != 0 with non-full extent, the view is
        // non-contiguous in general. Conservatively mark non-contiguous unless
        // the narrow is identity.
        let is_contig = self.is_contiguous && start == 0 && length == shape[dim];
        Ok(self.view_raw(new_shape, self.strides, new_offset, is_contig))
    }

    // ─── Accessors ─────────────────────────────────────────────────

    #[inline] pub fn shape(&self) -> &Shape { &self.shape }
    #[inline] pub fn strides(&self) -> &Strides { &self.strides }
    #[inline] pub fn ndim(&self) -> usize { self.shape.ndim() }
    #[inline] pub fn numel(&self) -> usize { self.numel }
    #[inline] pub fn is_contiguous(&self) -> bool { self.is_contiguous }
    #[inline] pub fn device(&self) -> &D { self.storage.device() }
    #[inline] pub fn storage(&self) -> &Arc<Storage<D>> { &self.storage }
    #[inline] pub fn offset_elems(&self) -> usize { self.offset_elems }

    /// Raw pointer to the first element (with offset applied).
    ///
    /// On CUDA tensors this points to **device memory** — do not dereference
    /// from host code; pass to kernels only.
    #[inline]
    pub fn data_ptr(&self) -> *const T {
        // SAFETY: storage.ptr() is valid for storage.size() bytes; offset
        // stays within bounds by construction (caller of view_raw is
        // responsible).
        unsafe { (self.storage.ptr() as *const T).add(self.offset_elems) }
    }

    /// Raw mutable pointer (same memory aliasing rules as `data_ptr`).
    ///
    /// Note: takes `&self`, not `&mut self`. The Arc is shared; mutability
    /// of the underlying storage is encoded by the operator-level method
    /// signatures (e.g. `&mut Tensor` only signals "this op writes here",
    /// not exclusive ownership at the storage level).
    #[inline]
    pub fn data_ptr_mut(&self) -> *mut T {
        // SAFETY: see data_ptr.
        unsafe { (self.storage.ptr() as *mut T).add(self.offset_elems) }
    }
}

impl<T: Dtype, D: MemoryPort> Clone for Tensor<T, D> {
    /// O(1) — bumps Arc refcount; new tensor shares the same storage.
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            shape: self.shape,
            strides: self.strides,
            offset_elems: self.offset_elems,
            numel: self.numel,
            is_contiguous: self.is_contiguous,
            _marker: PhantomData,
        }
    }
}

impl<T: Dtype, D: MemoryPort> std::fmt::Debug for Tensor<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("device", &self.storage.device().name())
            .field("dtype", &T::DATA_TYPE)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset_elems", &self.offset_elems)
            .field("numel", &self.numel)
            .field("is_contiguous", &self.is_contiguous)
            .finish()
    }
}
