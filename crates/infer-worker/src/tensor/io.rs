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

    /// Migrate to CPU. Already on CPU → cheap clone.
    pub fn to_cpu(&self) -> Result<Self> {
        if self.device() == DeviceType::Cpu {
            return Ok(self.clone());
        }
        // We materialise in the process — cross-device copies require a
        // contiguous, zero-offset layout.
        let src = if self.is_contiguous() && self.storage_offset() == 0 {
            self.clone()
        } else {
            self.contiguous()?
        };

        let allocator = Arc::new(CpuAllocator);
        let mut cpu_buffer = Buffer::new(src.buffer().len_bytes(), allocator)?;
        cpu_buffer.copy_from(src.buffer())?;
        Tensor::from_buffer(cpu_buffer, src.shape(), src.dtype())
    }

    /// Migrate to a specific CUDA device. Already there → cheap clone.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self, device_id: i32) -> Result<Self> {
        if self.device() == DeviceType::Cuda(device_id) {
            return Ok(self.clone());
        }
        let src = if self.is_contiguous() && self.storage_offset() == 0 {
            self.clone()
        } else {
            self.contiguous()?
        };

        let allocator = Arc::new(crate::base::allocator::CachingCudaAllocator::instance());
        crate::cuda::device::set_current_device(device_id)?;
        let mut gpu_buffer = Buffer::new(src.buffer().len_bytes(), allocator)?;
        gpu_buffer.copy_from(src.buffer())?;
        Tensor::from_buffer(gpu_buffer, src.shape(), src.dtype())
    }

    /// Convenience: go to any device. Same-device case is an Arc clone.
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

    /// Return a new tensor with data cast to `target_dtype`. Runs on CPU
    /// via the generic [`cast_kernel`]; non-CPU sources are round-tripped.
    pub fn to_dtype(&self, target_dtype: DataType) -> Result<Self> {
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        // Cast must see contiguous, CPU-resident input.
        let src_cpu = if self.device() == DeviceType::Cpu {
            if self.is_contiguous() && self.storage_offset() == 0 {
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

    /// Load from a safetensors view, copying into a freshly allocated
    /// tensor on `device`.
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

    /// Convenience: always load to CPU.
    pub fn from_view_on_cpu(view: &TensorView) -> Result<Self> {
        Self::from_view(view, DeviceType::Cpu)
    }

    /// Borrowing zero-copy loader: the returned tensor aliases `view`'s
    /// bytes.
    ///
    /// # Safety
    /// The caller must ensure the mmap/storage behind `view` outlives the
    /// returned Tensor.
    pub unsafe fn from_view_borrowed(view: &TensorView) -> Result<Self> {
        let (shape, dtype, data_bytes) = decode_tensor_view(view)?;
        let buffer = unsafe { Buffer::from_external_slice(data_bytes) };
        Tensor::from_buffer(buffer, shape, dtype)
    }

    // ─────────────────────── cross-tensor copying ─────────────────────

    /// Copy elements from `src` into `self`. Shapes must have identical
    /// element counts and dtypes must match. For layout mismatches where
    /// *either* side is non-contiguous we route through the strided-copy
    /// (identity-permute) kernel so the call is always safe.
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

    /// Stream-ordered variant of [`copy_from`](Self::copy_from) for CUDA.
    /// CPU tensors fall back to the synchronous path.
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

    /// Convenience: stream-ordered on current CUDA stream; plain
    /// `copy_from` on CPU.
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

    /// Write `count` host-side `i32`s into the first `count` slots of this
    /// tensor's buffer. Used by Runner ↔ Server shared-memory plumbing.
    pub fn write_from_i32_host(&mut self, src: &[i32], count: usize) -> Result<()> {
        assert!(count <= src.len(), "count exceeds src length");
        let copy_bytes = count * std::mem::size_of::<i32>();
        let mut dst_slice = self.buffer().slice(0, copy_bytes)?;
        dst_slice.copy_from_host(&src[..count])?;
        Ok(())
    }

    /// Read the first `count` `i32`s from this tensor's buffer back to a
    /// host `Vec`.
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

    fn validate_copy_compat(&self, src: &Tensor, ctx: &str) -> Result<()> {
        if self.numel() != src.numel() {
            anyhow::bail!(
                "{}: element count mismatch — dst shape {:?} ({} elems), \
                 src shape {:?} ({} elems)",
                ctx, self.shape(), self.numel(), src.shape(), src.numel()
            );
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

/// Element-by-element stride-aware copy on a single device (CPU only for
/// now). The CUDA path bounces through `permute_into`, which already
/// handles strided src into contiguous dst.
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
