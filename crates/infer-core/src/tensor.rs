//! `Tensor<T, D>` — the core domain object.
//!
//! A typed, device-aware view over an `Arc<Storage<D>>`. `Clone` and
//! `view` are O(1) Arc bumps; the underlying allocation is freed
//! automatically when the last tensor referencing it drops.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::device::{HostDevice, MemoryPort};
use crate::error::{OpError, OpResult};
use crate::storage::Storage;
use crate::types::{DataType, Dtype, Shape, Strides};

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

    /// Build a tensor from already-validated raw parts that alias `storage`.
    /// `numel` is derived from `shape`. This is the public replacement for the
    /// struct-literal construction op layers used before `Tensor` moved into
    /// `infer-core` (e.g. dtype-bitcast / reinterpret views).
    pub fn from_raw_parts(
        storage: Arc<Storage<D>>,
        shape: Shape,
        strides: Strides,
        offset_elems: usize,
        is_contiguous: bool,
    ) -> Self {
        let numel = shape.numel();
        Tensor {
            storage,
            shape,
            strides,
            offset_elems,
            numel,
            is_contiguous,
            _marker: PhantomData,
        }
    }

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
    pub fn from_host_slice(data: &[T], shape: impl Into<Shape>, device: &D) -> OpResult<Self> {
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
    pub fn from_host_bytes(bytes: &[u8], shape: impl Into<Shape>, device: &D) -> OpResult<Self> {
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
                self.storage
                    .device()
                    .download(out.as_mut_ptr() as *mut u8, src, size_bytes)?;
                out.set_len(self.numel);
            }
        } else {
            // SAFETY: empty vec.
            unsafe {
                out.set_len(0);
            }
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
                "narrow: dim {} out of range (ndim={})",
                dim,
                shape.len(),
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

    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    #[inline]
    pub fn strides(&self) -> &Strides {
        &self.strides
    }
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    #[inline]
    pub fn numel(&self) -> usize {
        self.numel
    }
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }
    #[inline]
    pub fn device(&self) -> &D {
        self.storage.device()
    }
    #[inline]
    pub fn storage(&self) -> &Arc<Storage<D>> {
        &self.storage
    }
    #[inline]
    pub fn offset_elems(&self) -> usize {
        self.offset_elems
    }

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

    /// Element dtype tag (mirrors `T::DATA_TYPE`).
    #[inline]
    pub fn dtype(&self) -> DataType {
        T::DATA_TYPE
    }

    /// Overwrite the tensor's contents with `data` (host → device upload).
    ///
    /// `data.len()` must equal `self.numel()`. The tensor must be contiguous.
    /// Does **not** allocate; reuses the existing storage.
    pub fn upload_from_host(&mut self, data: &[T]) -> OpResult<()> {
        if data.len() != self.numel {
            return Err(OpError::Shape(format!(
                "upload_from_host: data.len()={} != self.numel={}",
                data.len(),
                self.numel,
            )));
        }
        // No is_contiguous check: upload is a linear H2D memcpy;
        // non-contiguous views that are memory-packed (e.g. narrow on dim 0)
        // are fine.
        let size_bytes = self.numel * T::SIZE_BYTES;
        if size_bytes > 0 {
            // SAFETY: self.data_ptr() points to valid device memory of at
            // least self.numel * SIZE_BYTES bytes.
            unsafe {
                let dst = std::ptr::NonNull::new_unchecked(self.data_ptr() as *mut u8);
                self.storage
                    .device()
                    .upload(dst, data.as_ptr() as *const u8, size_bytes)?;
            }
        }
        Ok(())
    }

    /// In-place copy from a same-shape, same-dtype, same-device tensor.
    ///
    /// Performs a device-internal D2D memcpy (zero-copy, no host round-trip)
    /// when the tensors do not share storage. If they share storage (one is a
    /// view of the other), falls back to a host round-trip to avoid undefined
    /// behaviour on overlapping D2D copies.
    /// Returns an error if shape or numel disagrees. Both tensors must be
    /// contiguous — the D2D copy requires contiguous layout to be correct.
    pub fn copy_from(&mut self, src: &Tensor<T, D>) -> OpResult<()> {
        if self.shape != src.shape {
            return Err(OpError::Shape(format!(
                "copy_from: shape mismatch dst={:?} src={:?}",
                self.shape, src.shape
            )));
        }
        if !self.is_contiguous || !src.is_contiguous {
            return Err(OpError::NotContiguous(if self.is_contiguous {
                src.shape
            } else {
                self.shape
            }));
        }
        let n = self.numel;
        if n == 0 {
            return Ok(());
        }
        let bytes = n * T::SIZE_BYTES;

        // If src and self share the same storage, a D2D copy may overlap.
        // Fall back to host round-trip (download src → host, upload host → dst)
        // which is always safe.
        if Arc::as_ptr(&self.storage) == Arc::as_ptr(&src.storage) {
            // SAFETY: host round-trip is safe for overlapping views.
            let host = src.to_host_vec()?;
            let dev = self.storage.device();
            unsafe {
                let dst_nn = std::ptr::NonNull::new_unchecked(self.data_ptr_mut() as *mut u8);
                dev.upload(dst_nn, host.as_ptr() as *const u8, bytes)?;
            }
            return Ok(());
        }

        // Non-overlapping: D2D copy (fast, no host round-trip).
        let dev = self.storage.device();
        // SAFETY: src/dst are different storages, non-overlapping.
        unsafe {
            let dst_nn = std::ptr::NonNull::new_unchecked(self.data_ptr_mut() as *mut u8);
            let src_nn = std::ptr::NonNull::new_unchecked(src.data_ptr() as *mut u8);
            dev.copy_device_to_device(dst_nn, src_nn, bytes)?;
        }
        Ok(())
    }
}

// ─── Float-only helpers ───────────────────────────────────────────────

impl<T: Dtype, D: MemoryPort> Tensor<T, D> {
    /// Allocate a tensor on `device` filled with `randn` samples (Box-Muller).
    ///
    /// Generated host-side as `f32`, optionally cast to `T` per element if
    /// `T != f32`, then uploaded. `seed=None` selects an OS-entropy seed.
    /// Currently supports `T ∈ {f32, bf16, f16}`; other dtypes return an
    /// error.
    pub fn randn(shape: impl Into<Shape>, device: &D, seed: Option<u64>) -> OpResult<Self>
    where
        D: MemoryPort,
    {
        use half::{bf16, f16};
        let shape = shape.into();
        let n = shape.numel();
        // Deterministic SplitMix64 + Box-Muller. Avoids extra deps.
        let mut state: u64 = seed.unwrap_or_else(|| {
            // OS-derived entropy seed; collisions across calls are not a
            // correctness concern for the diffusion pipeline.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xdeadbeef)
        });
        fn next_u64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn unit_f32(state: &mut u64) -> f32 {
            // 24-bit mantissa for [1, 2) → subtract 1 → [0, 1)
            let bits = (next_u64(state) >> 40) as u32;
            f32::from_bits((127 << 23) | bits) - 1.0
        }
        let mut buf_f32: Vec<f32> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            // Avoid log(0).
            let mut u1 = unit_f32(&mut state);
            if u1 < 1e-7 {
                u1 = 1e-7;
            }
            let u2 = unit_f32(&mut state);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z0 = r * theta.cos();
            let z1 = r * theta.sin();
            buf_f32.push(z0);
            i += 1;
            if i < n {
                buf_f32.push(z1);
                i += 1;
            }
        }
        // Allocate destination + cast.
        let storage = Storage::alloc(device, n * T::SIZE_BYTES)?;
        if n > 0 {
            let bytes: Vec<u8> = match T::DATA_TYPE {
                DataType::F32 => {
                    let mut v = Vec::with_capacity(n * 4);
                    for &x in &buf_f32 {
                        v.extend_from_slice(&x.to_le_bytes());
                    }
                    v
                }
                DataType::BF16 => {
                    let mut v = Vec::with_capacity(n * 2);
                    for &x in &buf_f32 {
                        v.extend_from_slice(&bf16::from_f32(x).to_le_bytes());
                    }
                    v
                }
                DataType::F16 => {
                    let mut v = Vec::with_capacity(n * 2);
                    for &x in &buf_f32 {
                        v.extend_from_slice(&f16::from_f32(x).to_le_bytes());
                    }
                    v
                }
                _ => {
                    return Err(OpError::Kernel(format!(
                        "Tensor::randn: unsupported dtype {:?}",
                        T::DATA_TYPE,
                    )));
                }
            };
            // SAFETY: storage holds exactly n * SIZE_BYTES bytes of writable memory.
            unsafe {
                let dst = std::ptr::NonNull::new_unchecked(storage.ptr());
                device.upload(dst, bytes.as_ptr(), bytes.len())?;
            }
        }
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage,
            shape,
            strides,
            offset_elems: 0,
            numel: n,
            is_contiguous: true,
            _marker: PhantomData,
        })
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

/// Host-accessible typed-slice views — available for any `HostDevice` backend
/// (e.g. Cpu), whose bytes live in host-addressable memory. (Previously these
/// were Cpu-specific inherent methods on `Tensor<T, Cpu>`; generalizing to
/// `HostDevice` keeps them as inherent methods on `Tensor` so they move with it
/// into infer-core, without a cross-crate inherent-impl on a foreign type.)
impl<T: Dtype, D: HostDevice + MemoryPort> Tensor<T, D> {
    /// Borrow the tensor as a typed slice (host-accessible + contiguous only).
    pub fn as_slice(&self) -> &[T] {
        assert!(self.is_contiguous(), "as_slice requires contiguous");
        // SAFETY: host-accessible storage; pointer valid for `numel` elements.
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.numel()) }
    }

    /// Mutable typed slice. `&mut self` encodes exclusive access at the call
    /// site even though the underlying Arc is shared.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        assert!(self.is_contiguous(), "as_slice_mut requires contiguous");
        // SAFETY: host-accessible storage; pointer valid for `numel` elements.
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), self.numel()) }
    }
}
