//! I/O, dtype-casting, device migration, random init, strided copy.
//!
//! This module concentrates every Tensor method that either **allocates
//! new storage** or **crosses dtype/device boundaries**. Nothing here
//! touches in-place math — for that, see `tensor/ops.rs`.

use std::sync::Arc;

use half::{bf16, f16};
use safetensors::tensor::TensorView;
use safetensors::Dtype as SafetensorDtype;

use crate::base::allocator::CpuAllocator;
use crate::base::buffer::Buffer;
use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::op::kernels::cpu::cast_kernel;

use super::tensor::Tensor;

impl Tensor {
    // ──────────────────────────── randn ───────────────────────────────

    /// Normal-distribution initialiser `N(0, 1)`.
    ///
    /// - `seed = Some(x)` → reproducible (StdRng).
    /// - `seed = None`    → OS entropy (rand::rng()).
    ///
    /// Generation happens on CPU in F32 regardless of target dtype (so
    /// that Box-Muller is numerically sane), then cast + optionally
    /// uploaded to device.
    pub fn randn(
        shape: &[usize],
        dtype: DataType,
        device: DeviceType,
        seed: Option<u64>,
    ) -> Result<Self> {
        use rand::prelude::*;

        let numel: usize = shape.iter().product();
        let mut rng: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
            None    => Box::new(rand::rng()),
        };

        // Box-Muller N(0, 1)
        let mut f32_data = Vec::with_capacity(numel);
        while f32_data.len() < numel {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            f32_data.push(r * theta.cos());
            if f32_data.len() < numel {
                f32_data.push(r * theta.sin());
            }
        }

        let mut t = Tensor::empty(shape, dtype, DeviceType::Cpu)?;
        match &mut t {
            Tensor::F32(typed) => typed.as_slice_mut()?.copy_from_slice(&f32_data),
            Tensor::BF16(typed) => {
                let v: Vec<bf16> = f32_data.iter().map(|&v| bf16::from_f32(v)).collect();
                typed.as_slice_mut()?.copy_from_slice(&v);
            }
            Tensor::F16(typed) => {
                let v: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
                typed.as_slice_mut()?.copy_from_slice(&v);
            }
            _ => return Err(Error::InvalidArgument(format!(
                "randn: unsupported dtype {:?}", dtype
            )).into()),
        }

        match device {
            DeviceType::Cpu => Ok(t),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => t.to_cuda(id),
        }
    }

    // ─────────────────────────── device moves ─────────────────────────

    /// Migrates this tensor to CPU memory, returning a new tensor.
    ///
    /// If the tensor is already on CPU, returns a cheap clone (the underlying
    /// buffer `Arc` is shared — no data copy occurs).
    ///
    /// For CUDA → CPU transfers, the backing storage is first densified
    /// (if needed) and then copied to a freshly allocated CPU buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if memory allocation or the device-to-host copy fails.
    pub fn to_cpu(&self) -> Result<Self> {
        if self.device() == DeviceType::Cpu {
            return Ok(self.clone());
        }
        // Cross-device copies require a contiguous, zero-offset, *tightly
        // sized* source buffer. A prefix-narrowed view passes the cheap
        // `is_contiguous && offset==0` test but shares the parent's
        // oversized buffer, so use the stricter predicate here.
        let src = if self.owns_storage_tightly() {
            self.clone()
        } else {
            self.contiguous()?
        };

        let nbytes = src.numel() * src.dtype().size_in_bytes();
        debug_assert_eq!(src.buffer().len_bytes(), nbytes);
        let allocator = Arc::new(CpuAllocator);
        let mut cpu_buffer = Buffer::new(nbytes, allocator)?;
        cpu_buffer.copy_from(src.buffer())?;
        Tensor::from_buffer(cpu_buffer, src.shape(), src.dtype())
    }

    /// Migrates this tensor to a specific CUDA device, returning a new tensor.
    ///
    /// If the tensor is already on the target CUDA device, returns a cheap
    /// clone (no data copy). Otherwise allocates GPU memory via the caching
    /// allocator and performs a host-to-device (or device-to-device) copy.
    ///
    /// # Arguments
    ///
    /// - `device_id`: The CUDA device ordinal (0-based).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA device selection, allocation, or the copy fails.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self, device_id: i32) -> Result<Self> {
        if self.device() == DeviceType::Cuda(device_id) {
            return Ok(self.clone());
        }
        let src = if self.owns_storage_tightly() {
            self.clone()
        } else {
            self.contiguous()?
        };

        let nbytes = src.numel() * src.dtype().size_in_bytes();
        debug_assert_eq!(src.buffer().len_bytes(), nbytes);
        let allocator = Arc::new(crate::base::allocator::CachingCudaAllocator::instance());
        crate::cuda::device::set_current_device(device_id)?;
        let mut gpu_buffer = Buffer::new(nbytes, allocator)?;
        gpu_buffer.copy_from(src.buffer())?;
        Tensor::from_buffer(gpu_buffer, src.shape(), src.dtype())
    }

    /// Migrates this tensor to an arbitrary device (CPU or CUDA).
    ///
    /// Convenience wrapper around [`to_cpu()`](Self::to_cpu) and
    /// [`to_cuda()`](Self::to_cuda). If the tensor is already on the target
    /// device, returns a cheap `Arc` clone with no data copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying migration fails.
    pub fn to_device(&self, device: DeviceType) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }
        match device {
            DeviceType::Cpu => self.to_cpu(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => self.to_cuda(id),
        }
    }

    // ───────────────────────── dtype conversion ───────────────────────

    /// Casts this tensor to a different data type, returning a new tensor.
    ///
    /// If the tensor already has `target_dtype`, returns a cheap clone.
    /// Otherwise allocates a new tensor and element-wise converts via the
    /// CPU [`cast_kernel`]. Non-CPU inputs are round-tripped through CPU
    /// for the cast.
    ///
    /// # Supported Casts
    ///
    /// Currently supported conversions (extend `op::kernels::cpu::cast` for more):
    /// - `F32` → `BF16`
    /// - `BF16` → `F32`
    ///
    /// # Errors
    ///
    /// - Returns an error if the cast pair is not supported.
    /// - Returns an error if memory allocation or device transfer fails.
    pub fn to_dtype(&self, target_dtype: DataType) -> Result<Self> {
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        // Cast must see contiguous, CPU-resident, tightly-sized input.
        let src_cpu = if self.device() == DeviceType::Cpu {
            if self.owns_storage_tightly() {
                self.clone()
            } else {
                self.contiguous()?
            }
        } else {
            self.to_cpu()?
        };

        let mut dst_cpu = Tensor::empty(src_cpu.shape(), target_dtype, DeviceType::Cpu)?;

        macro_rules! run {
            ($from:ty, $from_t:expr, $to:ty, $to_t:expr) => {{
                let fs: &[$from] = $from_t.as_slice()?;
                let ts: &mut [$to] = $to_t.as_slice_mut()?;
                cast_kernel::<$from, $to>(fs, ts);
            }};
        }
        // Supported casts are limited by `CastFrom` implementations in
        // `op::kernels::cpu::cast`. Extend there before adding new arms.
        match (&src_cpu, &mut dst_cpu) {
            (Tensor::F32 (s), Tensor::BF16(d)) => run!(f32 , s, bf16, d),
            (Tensor::BF16(s), Tensor::F32 (d)) => run!(bf16, s, f32 , d),
            _ => return Err(Error::InvalidArgument(format!(
                "to_dtype: cast {:?} → {:?} not supported",
                self.dtype(), target_dtype
            )).into()),
        }

        // Ship back to the original device if needed.
        dst_cpu.to_device(self.device())
    }

    // ────────────────────────── safetensors ───────────────────────────

    /// Loads a tensor from a safetensors [`TensorView`], allocating on `device`.
    ///
    /// Copies the raw bytes from the memory-mapped view into a freshly
    /// allocated buffer. For CUDA targets, the data is first staged on CPU
    /// then uploaded via [`to_cuda()`](Self::to_cuda).
    ///
    /// # Arguments
    ///
    /// - `view`: A reference to a safetensors tensor view (typically from an mmap'd file).
    /// - `device`: The target device for the loaded tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the safetensors dtype is unsupported, allocation fails,
    /// or the device transfer fails.
    pub fn from_view(view: &TensorView, device: DeviceType) -> Result<Self> {
        let (shape, dtype, data_bytes) = decode_tensor_view(view)?;
        match device {
            DeviceType::Cpu => {
                let mut t = Tensor::empty(shape, dtype, DeviceType::Cpu)?;
                t.buffer_mut().copy_from_host(data_bytes)?;
                Ok(t)
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(device_id) => {
                let mut cpu = Tensor::empty(shape, dtype, DeviceType::Cpu)?;
                cpu.buffer_mut().copy_from_host(data_bytes)?;
                cpu.to_cuda(device_id)
            }
        }
    }

    /// Loads a tensor from a safetensors view, always allocating on CPU.
    ///
    /// Convenience wrapper for `Self::from_view(view, DeviceType::Cpu)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the safetensors dtype is unsupported or allocation fails.
    pub fn from_view_on_cpu(view: &TensorView) -> Result<Self> {
        Self::from_view(view, DeviceType::Cpu)
    }

    /// Creates a tensor that borrows (zero-copy) from a safetensors view's bytes.
    ///
    /// The returned tensor directly aliases the memory of the safetensors
    /// view without any copy. This is extremely fast for loading but requires
    /// careful lifetime management.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying storage (typically an mmap'd
    /// file) outlives the returned `Tensor`. Dropping the mmap while the tensor
    /// is still in use results in use-after-free.
    ///
    /// # Errors
    ///
    /// Returns an error if the safetensors dtype is unsupported.
    pub unsafe fn from_view_borrowed(view: &TensorView) -> Result<Self> {
        let (shape, dtype, data_bytes) = decode_tensor_view(view)?;
        let buffer = unsafe { Buffer::from_external_slice(data_bytes) };
        Tensor::from_buffer(buffer, shape, dtype)
    }

    // ─────────────────────── cross-tensor copying ─────────────────────

    /// Copies elements from `src` into `self`.
    ///
    /// This is the primary way to move data between tensors. It handles:
    /// - **Same-device contiguous**: fast bulk `memcpy` / `cudaMemcpy`.
    /// - **Same-device strided**: routes through the strided-copy kernel.
    /// - **Cross-device**: migrates `src` to `self`'s device first.
    ///
    /// # Requirements
    ///
    /// - `self.numel() == src.numel()` (element counts must match).
    /// - `self.dtype() == src.dtype()` (no implicit casting).
    ///
    /// # Errors
    ///
    /// Returns an error if shapes or dtypes are incompatible, or if the
    /// underlying copy operation fails.
    pub fn copy_from(&mut self, src: &Tensor) -> Result<()> {
        self.validate_copy_compat(src, "copy_from")?;

        // Fast path: both contiguous on the same device. Even if either
        // side has a nonzero `storage_offset`, we can still do a pure
        // bytewise copy by slicing each Buffer to the exact element
        // range.
        if self.is_contiguous() && src.is_contiguous()
            && self.device() == src.device()
        {
            let elem = self.dtype().size_in_bytes();
            let nbytes = self.numel() * elem;
            let src_off = src.storage_offset() * elem;
            let dst_off = self.storage_offset() * elem;
            let src_slice = src.buffer().slice(src_off, nbytes)?;
            let mut dst_slice = self.buffer().slice(dst_off, nbytes)?;
            dst_slice.copy_from(&src_slice)?;
            return Ok(());
        }
        // General path: same device → strided permute_into; cross-device
        // round-trip through contiguous() first.
        if self.device() != src.device() {
            let src_on_dev = src.to_device(self.device())?;
            return self.copy_from(&src_on_dev);
        }
        let identity: Vec<usize> = (0..src.ndim()).collect();
        // `dst` must be contiguous with zero offset for `permute_into`;
        // otherwise we take the generic strided CPU path.
        if self.is_contiguous() && self.storage_offset() == 0 {
            return src.permute_into(&identity, self);
        }
        let dense = src.contiguous()?;
        strided_copy_same_device(&dense, self)
    }

    /// Asynchronous (stream-ordered) variant of [`copy_from`](Self::copy_from) for CUDA.
    ///
    /// For contiguous same-device CUDA tensors, the copy is enqueued on the
    /// given `stream` and returns immediately (the host does not wait).
    /// For strided or cross-device cases, falls back to the synchronous path.
    ///
    /// CPU tensors are handled synchronously regardless of the stream argument.
    ///
    /// # Arguments
    ///
    /// - `src`: Source tensor (same dtype and element count as `self`).
    /// - `stream`: The CUDA stream on which to enqueue the copy.
    ///
    /// # Errors
    ///
    /// Returns an error if validation or the underlying copy fails.
    #[cfg(feature = "cuda")]
    pub fn copy_from_async(
        &mut self,
        src: &Tensor,
        stream: crate::cuda::ffi::cudaStream_t,
    ) -> Result<()> {
        self.validate_copy_compat(src, "copy_from_async")?;
        if self.is_contiguous() && src.is_contiguous()
            && self.device() == src.device()
        {
            let elem = self.dtype().size_in_bytes();
            let nbytes = self.numel() * elem;
            let src_off = src.storage_offset() * elem;
            let dst_off = self.storage_offset() * elem;
            let src_slice = src.buffer().slice(src_off, nbytes)?;
            let mut dst_slice = self.buffer().slice(dst_off, nbytes)?;
            dst_slice.copy_from_async(&src_slice, stream)?;
            return Ok(());
        }
        // Strided paths don't get stream-ordered treatment yet — the
        // permute kernel schedules itself on the current stream.
        self.copy_from(src)
    }

    /// Copies elements from `src` into `self` on the current CUDA stream.
    ///
    /// Convenience method that automatically selects the appropriate strategy:
    /// - On CUDA: uses [`copy_from_async`](Self::copy_from_async) with the
    ///   current device stream.
    /// - On CPU: falls back to synchronous [`copy_from`](Self::copy_from).
    ///
    /// # Errors
    ///
    /// Returns an error if validation or the underlying copy fails.
    #[inline]
    pub fn copy_from_on_current_stream(&mut self, src: &Tensor) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.device().is_cuda() || src.device().is_cuda() {
                let stream = crate::cuda::get_current_cuda_stream();
                return self.copy_from_async(src, stream);
            }
        }
        self.copy_from(src)
    }

    // ─────────────────────── raw i32 helpers ──────────────────────────

    /// Writes host-side `i32` values into the beginning of this tensor's buffer.
    ///
    /// Copies `count` elements from `src[..count]` into the first `count`
    /// slots of the tensor's raw buffer (byte offset 0). This is primarily
    /// used for Runner ↔ Server shared-memory plumbing where token IDs are
    /// written directly.
    ///
    /// # Arguments
    ///
    /// - `src`: The host-side source slice (must have at least `count` elements).
    /// - `count`: Number of `i32` values to copy.
    ///
    /// # Panics
    ///
    /// Panics if `count > src.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer copy operation fails.
    pub fn write_from_i32_host(&mut self, src: &[i32], count: usize) -> Result<()> {
        assert!(count <= src.len(), "count exceeds src length");
        let copy_bytes = count * std::mem::size_of::<i32>();
        let mut dst_slice = self.buffer().slice(0, copy_bytes)?;
        dst_slice.copy_from_host(&src[..count])?;
        Ok(())
    }

    /// Reads `i32` values from the beginning of this tensor's buffer to a host `Vec`.
    ///
    /// Copies `count` `i32` elements from the tensor's raw buffer (byte offset 0)
    /// into a newly allocated `Vec<i32>` on the host. This is the read-side
    /// counterpart to [`write_from_i32_host`](Self::write_from_i32_host).
    ///
    /// # Arguments
    ///
    /// - `count`: Number of `i32` values to read.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer copy operation fails.
    pub fn read_i32_to_host(&self, count: usize) -> Result<Vec<i32>> {
        let copy_bytes = count * std::mem::size_of::<i32>();
        let src_slice = self.buffer().slice(0, copy_bytes)?;
        let allocator = Arc::new(CpuAllocator);
        let mut cpu_buf = Buffer::new(copy_bytes, allocator)?;
        cpu_buf.copy_from(&src_slice)?;
        let ptr = cpu_buf.as_ptr() as *const i32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec())
    }

    // ───────────────────────── validation helper ──────────────────────

    /// Validates that `self` and `src` are compatible for a copy operation.
    ///
    /// Checks that element counts and dtypes match. Used internally by
    /// [`copy_from`](Self::copy_from) and its async variants.
    ///
    /// # Errors
    ///
    /// - Returns an error if `self.numel() != src.numel()`.
    /// - Returns an error if `self.dtype() != src.dtype()`.
    fn validate_copy_compat(&self, src: &Tensor, ctx: &str) -> Result<()> {
        if self.numel() != src.numel() {
            return Err(Error::InvalidArgument(format!(
                "{}: element count mismatch — dst shape {:?} ({} elems), \
                 src shape {:?} ({} elems)",
                ctx, self.shape(), self.numel(), src.shape(), src.numel()
            )).into());
        }
        if self.dtype() != src.dtype() {
            return Err(Error::InvalidArgument(format!(
                "{}: dtype mismatch — dst {:?}, src {:?}",
                ctx, self.dtype(), src.dtype()
            )).into());
        }
        Ok(())
    }
}

// ───────────────── module-private helpers ─────────────────

/// Decodes a safetensors [`TensorView`] into its shape, dtype, and raw bytes.
///
/// Maps the safetensors-specific dtype enum to our internal [`DataType`].
///
/// # Errors
///
/// Returns an error if the safetensors dtype has no corresponding `DataType`.
fn decode_tensor_view<'a>(view: &'a TensorView) -> Result<(&'a [usize], DataType, &'a [u8])> {
    let shape = view.shape();
    let st_dtype = view.dtype();
    let bytes = view.data();
    let dtype = match st_dtype {
        SafetensorDtype::F32  => DataType::F32,
        SafetensorDtype::F16  => DataType::F16,
        SafetensorDtype::BF16 => DataType::BF16,
        SafetensorDtype::I32  => DataType::I32,
        SafetensorDtype::I8   => DataType::I8,
        other => return Err(Error::InvalidArgument(format!(
            "unsupported safetensors dtype {:?}", other
        )).into()),
    };
    Ok((shape, dtype, bytes))
}

/// Performs an element-by-element stride-aware copy between two tensors on
/// the same device (CPU only).
///
/// This handles the general case where either source or destination (or both)
/// may have non-contiguous strides. Iterates over all elements using a
/// multi-index decomposition, computing source and destination offsets from
/// their respective strides.
///
/// For CUDA tensors, the strided copy goes through [`Tensor::permute_into`] instead,
/// which already handles arbitrary source strides.
///
/// # Errors
///
/// Returns an error if invoked on non-CPU tensors or if dtype pair doesn't match.
fn strided_copy_same_device(src: &Tensor, dst: &mut Tensor) -> Result<()> {
    if src.device() != DeviceType::Cpu {
        return Err(Error::InvalidArgument(
            "strided_copy_same_device: CUDA dst must be contiguous".into()
        ).into());
    }
    let ndim = src.ndim();
    let shape = src.shape().to_vec();
    let src_strides = src.strides().to_vec();
    let dst_strides = dst.strides().to_vec();
    let numel = src.numel();

    // Row-major walker over `shape`.
    macro_rules! walk {
        ($ty:ty, $sr:expr, $ds:expr) => {{
            let sp = $sr.data_ptr();
            let dp = $ds.data_ptr_mut();
            // Precompute row-major strides for the flat index → multi-index
            // decomposition.
            let mut walker = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                walker[i] = walker[i + 1] * shape[i + 1];
            }
            for flat in 0..numel {
                let mut rem = flat;
                let mut so = 0usize;
                let mut dod = 0usize;
                for j in 0..ndim {
                    let c = rem / walker[j];
                    rem %= walker[j];
                    so  += c * src_strides[j];
                    dod += c * dst_strides[j];
                }
                unsafe { *dp.add(dod) = *sp.add(so); }
            }
        }};
    }
    match (src, dst) {
        (Tensor::F32(s),  Tensor::F32(d))  => walk!(f32 , s, d),
        (Tensor::BF16(s), Tensor::BF16(d)) => walk!(bf16, s, d),
        (Tensor::F16(s),  Tensor::F16(d))  => walk!(f16 , s, d),
        (Tensor::I32(s),  Tensor::I32(d))  => walk!(i32 , s, d),
        (Tensor::I8(s),   Tensor::I8(d))   => walk!(i8  , s, d),
        _ => return Err(Error::InvalidArgument(
            "strided_copy_same_device: dtype pair mismatch".into()
        ).into()),
    }
    Ok(())
}
