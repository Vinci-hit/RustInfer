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
    /// Validate that a logical tensor view can only address elements inside
    /// `storage`. This stays on the construction path so pointer accessors can
    /// rely on the invariant without repeating the arithmetic.
    fn assert_valid_view(
        storage: &Storage<D>,
        shape: &Shape,
        strides: &Strides,
        offset_elems: usize,
        numel: usize,
        is_contiguous: bool,
    ) {
        assert_eq!(
            shape.ndim(),
            strides.as_slice().len(),
            "Tensor view rank mismatch: shape rank {} != stride rank {}",
            shape.ndim(),
            strides.as_slice().len(),
        );

        let elem_size = std::mem::size_of::<T>();
        assert!(
            elem_size > 0,
            "Tensor views do not support zero-sized dtypes"
        );
        assert_eq!(
            elem_size,
            T::SIZE_BYTES,
            "Tensor dtype byte-size mismatch: Rust layout is {} bytes but Dtype::SIZE_BYTES is {}",
            elem_size,
            T::SIZE_BYTES,
        );
        assert_eq!(
            storage.ptr().addr() % std::mem::align_of::<T>(),
            0,
            "Tensor storage pointer is not aligned for its dtype",
        );

        let storage_elems = storage.size() / elem_size;
        assert!(
            offset_elems <= storage_elems,
            "Tensor view offset {} exceeds storage capacity {} elements",
            offset_elems,
            storage_elems,
        );

        // Empty views touch no element. The offset check above still ensures
        // that data_ptr() is at most one element past the backing allocation.
        if numel == 0 {
            return;
        }

        let max_relative_index = shape
            .as_slice()
            .iter()
            .zip(strides.as_slice())
            .try_fold(0usize, |span, (&extent, &stride)| {
                let dim_span = (extent - 1).checked_mul(stride)?;
                span.checked_add(dim_span)
            })
            .expect("Tensor view address calculation overflow");
        let required_elems = offset_elems
            .checked_add(max_relative_index)
            .and_then(|last| last.checked_add(1))
            .expect("Tensor view address calculation overflow");
        assert!(
            required_elems <= storage_elems,
            "Tensor view requires {} elements but storage capacity is {}",
            required_elems,
            storage_elems,
        );

        // Some operations intentionally treat a non-contiguous tensor as a
        // linear prefix (for example host uploads). Keep that independent
        // access pattern in bounds as well as the logical strided span above.
        let linear_end = offset_elems
            .checked_add(numel)
            .expect("Tensor linear view address calculation overflow");
        assert!(
            linear_end <= storage_elems,
            "Tensor linear view requires {} elements but storage capacity is {}",
            linear_end,
            storage_elems,
        );

        if is_contiguous {
            // Singleton dimensions do not constrain their stride because they
            // never advance the address. Every non-singleton dimension must
            // still describe a packed row-major layout.
            let mut expected_stride = 1usize;
            for (&extent, &stride) in shape.as_slice().iter().zip(strides.as_slice()).rev() {
                if extent > 1 {
                    assert_eq!(
                        stride, expected_stride,
                        "Tensor view marked contiguous has invalid stride {} (expected {})",
                        stride, expected_stride,
                    );
                }
                expected_stride = expected_stride
                    .checked_mul(extent)
                    .expect("Tensor contiguous stride calculation overflow");
            }
        }
    }
    // ─── Construction ──────────────────────────────────────────────

    /// Build a tensor from checked raw parts that alias `storage`.
    /// `numel` is derived from `shape`; invalid ranks, layouts, arithmetic, or
    /// storage bounds panic before a tensor can expose an invalid pointer.
    /// This is the public replacement for the struct-literal construction op
    /// layers used before `Tensor` moved into `infer-core` (e.g. dtype-bitcast /
    /// reinterpret views).
    pub fn from_raw_parts(
        storage: Arc<Storage<D>>,
        shape: Shape,
        strides: Strides,
        offset_elems: usize,
        is_contiguous: bool,
    ) -> Self {
        let numel = shape.numel();
        Self::assert_valid_view(
            &storage,
            &shape,
            &strides,
            offset_elems,
            numel,
            is_contiguous,
        );
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

    /// Construct a view sharing the same storage with a custom shape, strides,
    /// and absolute element offset. Invalid ranks, layouts, arithmetic, or
    /// storage bounds panic before the view is created.
    pub fn view_raw(
        &self,
        shape: Shape,
        strides: Strides,
        offset_elems: usize,
        is_contiguous: bool,
    ) -> Self {
        let numel = shape.numel();
        Self::assert_valid_view(
            &self.storage,
            &shape,
            &strides,
            offset_elems,
            numel,
            is_contiguous,
        );
        Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset_elems,
            numel,
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
        if start > shape[dim] || length > shape[dim] - start {
            return Err(OpError::Shape(format!(
                "narrow: dim {} out of bounds (start={}, length={}, extent={})",
                dim, start, length, shape[dim],
            )));
        }
        let strides = self.strides.as_slice();
        let mut new_shape_vec: Vec<usize> = shape.to_vec();
        new_shape_vec[dim] = length;
        let new_shape = Shape::from_slice(&new_shape_vec);
        let extra_offset = start.checked_mul(strides[dim]).ok_or_else(|| {
            OpError::Shape(format!(
                "narrow: offset overflow (start={}, stride={})",
                start, strides[dim],
            ))
        })?;
        let new_offset = self.offset_elems.checked_add(extra_offset).ok_or_else(|| {
            OpError::Shape(format!(
                "narrow: offset overflow (base={}, extra={})",
                self.offset_elems, extra_offset,
            ))
        })?;
        // Contiguity after narrow: a slice along `dim == 0` stays a single
        // contiguous block (only the base offset shifts), so it remains
        // contiguous for any `start`/`length`. Narrowing an inner dim to less
        // than its full extent introduces gaps between rows and is
        // non-contiguous (unless it is the identity slice).
        let is_contig = self.is_contiguous && (dim == 0 || (start == 0 && length == shape[dim]));
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
        // SAFETY: every construction path validates that offset is within (or,
        // for an empty view, one past) the backing allocation.
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

    /// Reinterpret this tensor's element type as `U` **without touching memory**,
    /// for backends that must narrow a generic `T` to a concrete dtype at a
    /// dispatch boundary (e.g. `MathOps for Cuda` selecting the `f32`/`bf16`/
    /// `f16` kernel). `T` and `U` must have identical storage layout — enforced
    /// here by requiring equal `DATA_TYPE` and `SIZE_BYTES`; a mismatch is a
    /// programmer error and panics rather than silently reinterpreting bytes.
    ///
    /// Zero-cost: this only rewrites the `PhantomData<T>` tag; the `Arc`,
    /// shape, strides, and offset are copied verbatim (an `Arc` refcount bump).
    #[inline]
    pub fn reinterpret<U: Dtype>(&self) -> Tensor<U, D> {
        assert_eq!(
            (T::DATA_TYPE, T::SIZE_BYTES),
            (U::DATA_TYPE, U::SIZE_BYTES),
            "Tensor::reinterpret: layout mismatch {:?} -> {:?}",
            T::DATA_TYPE,
            U::DATA_TYPE,
        );
        Tensor {
            storage: Arc::clone(&self.storage),
            shape: self.shape,
            strides: self.strides,
            offset_elems: self.offset_elems,
            numel: self.numel,
            is_contiguous: self.is_contiguous,
            _marker: PhantomData,
        }
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
    /// Generated host-side and converted through the canonical `Dtype`
    /// scalar API. `seed=None` selects an OS-entropy seed.
    pub fn randn(shape: impl Into<Shape>, device: &D, seed: Option<u64>) -> OpResult<Self>
    where
        D: MemoryPort,
    {
        let shape = shape.into();
        let n = shape.numel();
        if !T::DATA_TYPE.is_float() {
            return Err(OpError::Kernel(format!(
                "Tensor::randn: unsupported dtype {:?}",
                T::DATA_TYPE,
            )));
        }
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
        let values: Vec<T> = buf_f32
            .into_iter()
            .map(|value| T::write_f64(f64::from(value)))
            .collect();
        Self::from_host_slice(&values, shape, device)
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

    /// Mutable typed slice. Panics unless this tensor is the sole owner of its
    /// storage; `&mut self` alone cannot make storage exclusive when cloned
    /// tensors or views retain the same `Arc`.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        assert!(self.is_contiguous(), "as_slice_mut requires contiguous");
        let storage = Arc::get_mut(&mut self.storage)
            .expect("as_slice_mut requires uniquely owned tensor storage");
        // SAFETY: Arc::get_mut proves no other strong or weak Arc can access
        // the storage, the &mut self borrow prevents cloning until the returned
        // slice expires, and construction validated this contiguous range.
        unsafe {
            std::slice::from_raw_parts_mut(
                (storage.ptr() as *mut T).add(self.offset_elems),
                self.numel,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::ptr::NonNull;

    use super::*;
    use crate::device::Device;

    #[derive(Clone, Copy, Debug)]
    struct TestDevice;

    impl Device for TestDevice {
        type ExecCtx = ();

        fn exec_ctx(&self) -> &Self::ExecCtx {
            &()
        }

        fn name(&self) -> &'static str {
            "tensor-test"
        }
    }

    impl HostDevice for TestDevice {}

    fn test_layout(size: usize) -> Layout {
        Layout::from_size_align(size.max(1), 16).expect("valid test layout")
    }

    impl MemoryPort for TestDevice {
        fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>> {
            let ptr = unsafe { std::alloc::alloc_zeroed(test_layout(size)) };
            NonNull::new(ptr).ok_or_else(|| OpError::Kernel("test allocation failed".into()))
        }

        unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize) {
            unsafe { std::alloc::dealloc(ptr.as_ptr(), test_layout(size)) };
        }

        unsafe fn upload(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
            unsafe { std::ptr::copy_nonoverlapping(src, dst.as_ptr(), size) };
            Ok(())
        }

        unsafe fn upload_async(
            &self,
            dst: NonNull<u8>,
            src: *const u8,
            size: usize,
        ) -> OpResult<()> {
            unsafe { self.upload(dst, src, size) }
        }

        unsafe fn download(&self, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()> {
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, size) };
            Ok(())
        }

        fn synchronize(&self) -> OpResult<()> {
            Ok(())
        }

        unsafe fn copy_device_to_device(
            &self,
            dst: NonNull<u8>,
            src: NonNull<u8>,
            size: usize,
        ) -> OpResult<()> {
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), size) };
            Ok(())
        }
    }

    #[test]
    fn raw_parts_reject_out_of_bounds_and_rank_mismatch() {
        let storage = Storage::alloc(&TestDevice, 4 * std::mem::size_of::<f32>()).unwrap();

        let out_of_bounds = catch_unwind(AssertUnwindSafe(|| {
            Tensor::<f32, TestDevice>::from_raw_parts(
                Arc::clone(&storage),
                Shape::from([5]),
                Strides::from_slice(&[1]),
                0,
                true,
            )
        }));
        assert!(out_of_bounds.is_err());

        let rank_mismatch = catch_unwind(AssertUnwindSafe(|| {
            Tensor::<f32, TestDevice>::from_raw_parts(
                Arc::clone(&storage),
                Shape::from([2, 2]),
                Strides::from_slice(&[1]),
                0,
                false,
            )
        }));
        assert!(rank_mismatch.is_err());
    }

    #[test]
    fn raw_views_reject_invalid_contiguous_strided_and_overflowing_ranges() {
        let tensor = Tensor::<f32, TestDevice>::zeros([4], &TestDevice).unwrap();

        let invalid_contiguous = catch_unwind(AssertUnwindSafe(|| {
            tensor.view_raw(Shape::from([2, 2]), Strides::from_slice(&[1, 1]), 0, true)
        }));
        assert!(invalid_contiguous.is_err());

        let strided_out_of_bounds = catch_unwind(AssertUnwindSafe(|| {
            tensor.view_raw(Shape::from([2, 2]), Strides::from_slice(&[3, 1]), 0, false)
        }));
        assert!(strided_out_of_bounds.is_err());

        // The logical stride span is in bounds, but linear operations would
        // overrun from this offset. Construction must protect both patterns.
        let linear_out_of_bounds = catch_unwind(AssertUnwindSafe(|| {
            tensor.view_raw(Shape::from([2, 2]), Strides::from_slice(&[0, 0]), 1, false)
        }));
        assert!(linear_out_of_bounds.is_err());

        let overflowing = catch_unwind(AssertUnwindSafe(|| {
            tensor.view_raw(
                Shape::from([2]),
                Strides::from_slice(&[usize::MAX]),
                1,
                false,
            )
        }));
        assert!(overflowing.is_err());
    }

    #[test]
    fn valid_contiguous_and_narrow_views_preserve_data() {
        let data: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let tensor = Tensor::from_host_slice(&data, [3, 4], &TestDevice).unwrap();

        let raw = tensor.view_raw(Shape::from([2, 4]), Strides::from_slice(&[4, 1]), 4, true);
        assert_eq!(raw.as_slice(), &data[4..]);

        let narrow = tensor.narrow(0, 1, 2).unwrap();
        assert!(narrow.is_contiguous());
        assert_eq!(narrow.offset_elems(), 4);
        assert_eq!(narrow.as_slice(), &data[4..]);

        let inner = tensor.narrow(1, 1, 2).unwrap();
        assert!(!inner.is_contiguous());
        assert_eq!(inner.shape().as_slice(), &[3, 2]);
        assert_eq!(inner.offset_elems(), 1);

        assert!(tensor.narrow(0, usize::MAX, 2).is_err());
    }

    #[test]
    fn mutable_slice_requires_unique_storage() {
        let tensor = Tensor::from_host_slice(&[1_i32, 2, 3], [3], &TestDevice).unwrap();
        let mut alias = tensor.clone();

        let aliased_mutation = catch_unwind(AssertUnwindSafe(|| {
            alias.as_slice_mut()[0] = 9;
        }));
        assert!(aliased_mutation.is_err());

        drop(tensor);
        alias.as_slice_mut()[0] = 9;
        assert_eq!(alias.as_slice(), &[9, 2, 3]);
    }
}
