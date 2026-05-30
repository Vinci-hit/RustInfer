//! `Tensor<T, D>` — the core domain object.
//!
//! A typed, device-aware view over an `Arc<Storage<D>>`. `Clone` and
//! `view` are O(1) Arc bumps; the underlying allocation is freed
//! automatically when the last tensor referencing it drops.

use std::marker::PhantomData;
use std::sync::Arc;

use super::ports::{MemoryPort, OpError, OpResult};
use super::storage::Storage;
use super::types::{DataType, Dtype, Shape, Strides};

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

    /// Element dtype tag (mirrors `T::DATA_TYPE`).
    #[inline]
    pub fn dtype(&self) -> DataType { T::DATA_TYPE }

    /// In-place copy from a same-shape, same-dtype, same-device tensor.
    ///
    /// Performs a device-internal D2D / memcpy. Returns an error if shape
    /// or numel disagrees. The destination must be contiguous (sources may
    /// be strided as long as their numel matches and memcpy semantics are
    /// well-defined — for now we require both contiguous, mirroring the
    /// behaviour we need from the diffusion pipeline).
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
        if n == 0 { return Ok(()); }
        let bytes = n * T::SIZE_BYTES;
        // For CPU<->CPU and Cuda<->Cuda the storage device is identical
        // (D is a single device type via the type parameter), so we can
        // route through the MemoryPort: download to a stack-allocated
        // staging buffer and re-upload. For tiny sizes this is fine; for
        // larger ones backends override with stream-ordered D2D copies.
        // Until the OpBackend gains a `copy_inplace`, we use the host
        // round-trip path which is correct on every device but not the
        // fastest on CUDA.
        let host = src.to_host_vec()?;
        let dev = self.storage.device().clone();
        // Reuse from_host_slice path but write into self.
        // SAFETY: dst is owned, has `bytes` valid bytes.
        unsafe {
            let dst_nn = std::ptr::NonNull::new_unchecked(self.data_ptr_mut() as *mut u8);
            dev.upload(dst_nn, host.as_ptr() as *const u8, bytes)?;
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
            if u1 < 1e-7 { u1 = 1e-7; }
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
                    for &x in &buf_f32 { v.extend_from_slice(&x.to_le_bytes()); }
                    v
                }
                DataType::BF16 => {
                    let mut v = Vec::with_capacity(n * 2);
                    for &x in &buf_f32 { v.extend_from_slice(&bf16::from_f32(x).to_le_bytes()); }
                    v
                }
                DataType::F16 => {
                    let mut v = Vec::with_capacity(n * 2);
                    for &x in &buf_f32 { v.extend_from_slice(&f16::from_f32(x).to_le_bytes()); }
                    v
                }
                _ => return Err(OpError::Kernel(format!(
                    "Tensor::randn: unsupported dtype {:?}", T::DATA_TYPE,
                ))),
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

#[cfg(test)]
mod helper_tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;
    use half::bf16;

    #[test]
    fn randn_f32_cpu_seeded_is_deterministic() {
        let dev = Cpu;
        let a: Tensor<f32, Cpu> = Tensor::randn([4, 8], &dev, Some(42)).unwrap();
        let b: Tensor<f32, Cpu> = Tensor::randn([4, 8], &dev, Some(42)).unwrap();
        let av = a.to_host_vec().unwrap();
        let bv = b.to_host_vec().unwrap();
        assert_eq!(av, bv, "same seed must produce identical samples");
        assert_eq!(av.len(), 32);
    }

    #[test]
    fn randn_f32_cpu_distribution_within_range() {
        let dev = Cpu;
        let n = 4096usize;
        let t: Tensor<f32, Cpu> = Tensor::randn([n], &dev, Some(7)).unwrap();
        let v = t.to_host_vec().unwrap();
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        // Standard normal: mean ≈ 0, var ≈ 1. 4k samples → ±0.1 tolerance.
        assert!(mean.abs() < 0.1, "mean was {}", mean);
        assert!((var - 1.0).abs() < 0.15, "var was {}", var);
        // Should not produce all zeros / all NaN.
        assert!(v.iter().any(|&x| x.abs() > 0.1));
        assert!(v.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn randn_bf16_cpu_distribution_within_range() {
        let dev = Cpu;
        let n = 4096usize;
        let t: Tensor<bf16, Cpu> = Tensor::randn([n], &dev, Some(11)).unwrap();
        let v: Vec<f32> = t.to_host_vec().unwrap().iter().map(|x| x.to_f32()).collect();
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.1, "mean was {}", mean);
        assert!((var - 1.0).abs() < 0.20, "var was {}", var); // wider tol for bf16 quantization
        assert!(v.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn copy_from_cpu_roundtrip() {
        let dev = Cpu;
        let src: Tensor<f32, Cpu> = Tensor::randn([3, 5], &dev, Some(1)).unwrap();
        let mut dst: Tensor<f32, Cpu> = Tensor::zeros([3, 5], &dev).unwrap();
        dst.copy_from(&src).unwrap();
        assert_eq!(src.to_host_vec().unwrap(), dst.to_host_vec().unwrap());
    }

    #[test]
    fn copy_from_shape_mismatch_errors() {
        let dev = Cpu;
        let src: Tensor<f32, Cpu> = Tensor::zeros([3, 5], &dev).unwrap();
        let mut dst: Tensor<f32, Cpu> = Tensor::zeros([4, 5], &dev).unwrap();
        let err = dst.copy_from(&src).unwrap_err();
        match err {
            OpError::Shape(_) => {}
            other => panic!("expected Shape error, got {:?}", other),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn randn_bf16_cuda_seeded_matches_cpu() {
        use crate::infrastructure::cuda::Cuda;
        let cpu = Cpu;
        let cuda = Cuda::new(0).expect("cuda init");
        let n = 1024usize;
        let cpu_t: Tensor<bf16, Cpu> = Tensor::randn([n], &cpu, Some(123)).unwrap();
        let gpu_t: Tensor<bf16, Cuda> = Tensor::randn([n], &cuda, Some(123)).unwrap();
        // Pull GPU back to host.
        let gpu_host = gpu_t.to_host_vec().unwrap();
        let cpu_host = cpu_t.to_host_vec().unwrap();
        for (i, (a, b)) in cpu_host.iter().zip(gpu_host.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "cpu/gpu randn diverged at i={}: cpu={}, gpu={}", i, a.to_f32(), b.to_f32());
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod opbackend_dispatch_tests {
    //! Integration tests that exercise the new diffusion ops through the
    //! `OpBackend` trait dispatch (not just the kernel modules directly).
    //! Catches issues where the trait wiring forgets a method.

    use super::*;
    use crate::domain::ports::OpBackend;
    use crate::infrastructure::cuda::Cuda;
    use half::bf16;

    #[test]
    fn opbackend_apply_rope_interleaved_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (4usize, 2usize, 8usize);
        let half = d / 2;
        let x_host: Vec<f32> = (0..seq * h * d).map(|i| (i as f32) * 0.1).collect();
        let cos_host: Vec<f32> = vec![0.9; seq * half];
        let sin_host: Vec<f32> = vec![0.1; seq * half];
        let mut x: Tensor<f32, Cuda> = Tensor::from_host_slice(&x_host, [seq, h, d], &cuda).unwrap();
        let cos_t: Tensor<f32, Cuda> = Tensor::from_host_slice(&cos_host, [seq, half], &cuda).unwrap();
        let sin_t: Tensor<f32, Cuda> = Tensor::from_host_slice(&sin_host, [seq, half], &cuda).unwrap();
        Cuda::apply_rope_interleaved(&mut x, &cos_t, &sin_t, d).unwrap();
        let got = x.to_host_vec().unwrap();
        // Sanity: values changed.
        assert_ne!(got[0], x_host[0]);
        // First rotated pair: x'[0] = a*cos - b*sin, x'[1] = a*sin + b*cos.
        let (a, b) = (x_host[0], x_host[1]);
        assert!((got[0] - (a * 0.9 - b * 0.1)).abs() < 1e-5);
        assert!((got[1] - (a * 0.1 + b * 0.9)).abs() < 1e-5);
    }

    #[test]
    fn opbackend_concat_seq_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let a: Tensor<bf16, Cuda> = Tensor::from_host_slice(
            &(0..8).map(|i| bf16::from_f32(i as f32)).collect::<Vec<_>>(), [2, 4], &cuda,
        ).unwrap();
        let b: Tensor<bf16, Cuda> = Tensor::from_host_slice(
            &(0..12).map(|i| bf16::from_f32(-(i as f32))).collect::<Vec<_>>(), [3, 4], &cuda,
        ).unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([5, 4], &cuda).unwrap();
        Cuda::concat_seq(&a, &b, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert_eq!(got[0].to_f32(), 0.0);
        assert_eq!(got[7].to_f32(), 7.0);
        assert_eq!(got[8].to_f32(), 0.0); // start of b
        assert_eq!(got[19].to_f32(), -11.0);
    }

    #[test]
    fn opbackend_sdpa_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (3usize, 2usize, 4usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q: Tensor<f32, Cuda> = Tensor::randn([seq, h, d], &cuda, Some(1)).unwrap();
        let k: Tensor<f32, Cuda> = Tensor::randn([seq, h, d], &cuda, Some(2)).unwrap();
        let v: Tensor<f32, Cuda> = Tensor::randn([seq, h, d], &cuda, Some(3)).unwrap();
        let mut out: Tensor<f32, Cuda> = Tensor::zeros([seq, h, d], &cuda).unwrap();
        Cuda::sdpa(&q, &k, &v, &mut out, h, h, d, scale).unwrap();
        // Output is a finite, nonzero tensor.
        let got = out.to_host_vec().unwrap();
        assert!(got.iter().all(|x| x.is_finite()));
        assert!(got.iter().any(|x| x.abs() > 1e-6));
    }

    #[test]
    fn opbackend_pad_with_token_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&[1.0, 2.0, 3.0, 4.0], [1, 4], &cuda).unwrap();
        let pad: Tensor<f32, Cuda> = Tensor::from_host_slice(&[7.0, 7.0, 7.0, 7.0], [4], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([3, 4], &cuda).unwrap();
        Cuda::pad_with_token(&src, &pad, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert_eq!(&got[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert!(got[4..].iter().all(|&x| x == 7.0));
    }

    #[test]
    fn opbackend_cast_dtype_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&[1.0, 2.0, 3.0, 4.0], [4], &cuda).unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([4], &cuda).unwrap();
        Cuda::cast_dtype(&src, &mut dst).unwrap();
        let got: Vec<f32> = dst.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn opbackend_silu_and_tanh_dispatch() {
        let cuda = Cuda::new(0).unwrap();
        let mut s: Tensor<f32, Cuda> = Tensor::from_host_slice(&[1.0, 2.0, -1.0], [3], &cuda).unwrap();
        Cuda::silu_inplace_diff(&mut s).unwrap();
        let s_got = s.to_host_vec().unwrap();
        for (i, &x) in [1.0_f32, 2.0, -1.0].iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            assert!((s_got[i] - expected).abs() < 1e-5);
        }
        let mut t: Tensor<f32, Cuda> = Tensor::from_host_slice(&[0.5, 1.0, -1.0], [3], &cuda).unwrap();
        Cuda::tanh_inplace(&mut t).unwrap();
        let t_got = t.to_host_vec().unwrap();
        for (i, &x) in [0.5_f32, 1.0, -1.0].iter().enumerate() {
            assert!((t_got[i] - x.tanh()).abs() < 1e-5);
        }
    }

    #[test]
    fn opbackend_scalar_mul_from_dev_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let mut x: Tensor<f32, Cuda> = Tensor::from_host_slice(&[1.0, 2.0, 3.0], [3], &cuda).unwrap();
        let scalar: Tensor<f32, Cuda> = Tensor::from_host_slice(&[2.5_f32], [1], &cuda).unwrap();
        Cuda::scalar_mul_inplace_from_dev(&mut x, &scalar).unwrap();
        let got = x.to_host_vec().unwrap();
        assert!((got[0] - 2.5).abs() < 1e-5);
        assert!((got[1] - 5.0).abs() < 1e-5);
        assert!((got[2] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn opbackend_split_cols_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let src_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&src_host, [2, 3], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([2, 2], &cuda).unwrap();
        Cuda::split_cols(&src, &mut dst, 2, 3, 1, 2).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert_eq!(got, vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn opbackend_broadcast_add_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let mut x: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &[1.0, 2.0, 3.0, 4.0], [2, 2], &cuda,
        ).unwrap();
        let bias: Tensor<f32, Cuda> = Tensor::from_host_slice(&[10.0, 20.0], [2], &cuda).unwrap();
        Cuda::broadcast_add_inplace(&mut x, &bias).unwrap();
        let got = x.to_host_vec().unwrap();
        assert_eq!(got, vec![11.0, 22.0, 13.0, 24.0]);
    }
}
