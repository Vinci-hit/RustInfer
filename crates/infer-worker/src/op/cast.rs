//! Dtype casting for `Tensor`.
//!
//! Two entry points:
//!
//! - [`cast_dtype`]   — convenience; allocates a fresh destination.
//! - [`cast_dtype_into`] — hot-path dst-write variant; caller owns the
//!   destination buffer.
//!
//! CPU delegates to [`Tensor::to_dtype`] (which goes through the CPU
//! `cast_kernel`); CUDA dispatches to dedicated kernels that preserve
//! stride-offset semantics via [`TypedTensor::data_ptr`].

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cast_f32_to_bf16_forward(dst: *mut half::bf16, src: *const f32,        n: i32, stream: cudaStream_t);
    fn cast_bf16_to_f32_forward(dst: *mut f32,        src: *const half::bf16, n: i32, stream: cudaStream_t);
    fn cast_f32_to_f16_forward (dst: *mut half::f16,  src: *const f32,        n: i32, stream: cudaStream_t);
    fn cast_f16_to_f32_forward (dst: *mut f32,        src: *const half::f16,  n: i32, stream: cudaStream_t);
}

/// Allocate a new `Tensor` of `new_dtype` holding the cast of `src`.
pub fn cast_dtype(src: &Tensor, new_dtype: DataType) -> Result<Tensor> {
    if src.dtype() == new_dtype {
        return src.to_owned();
    }
    let mut dst = Tensor::empty(src.shape(), new_dtype, src.device())?;
    cast_dtype_into(src, &mut dst)?;
    Ok(dst)
}

/// `dst = src.cast(dst.dtype())` without allocating.
///
/// `dst` must have identical shape / device to `src`. When
/// `src.dtype() == dst.dtype()` this degrades to `dst.copy_from(src)`.
pub fn cast_dtype_into(src: &Tensor, dst: &mut Tensor) -> Result<()> {
    if src.shape() != dst.shape() {
        return Err(Error::InvalidArgument(format!(
            "cast_dtype_into: shape mismatch src={:?} dst={:?}",
            src.shape(), dst.shape()
        )).into());
    }
    if src.device() != dst.device() {
        return Err(Error::InvalidArgument(format!(
            "cast_dtype_into: device mismatch src={:?} dst={:?}",
            src.device(), dst.device()
        )).into());
    }

    // Identity: no cast, just a copy.
    if src.dtype() == dst.dtype() {
        return dst.copy_from_on_current_stream(src);
    }

    match src.device() {
        DeviceType::Cpu => {
            // Let the Tensor-level routine handle CPU casting (parallel via
            // `cast_kernel`) and copy into the caller's dst slot.
            let tmp = src.to_dtype(dst.dtype())?;
            dst.copy_from_on_current_stream(&tmp)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let n = src.numel() as i32;
            let stream = crate::cuda::get_current_cuda_stream();
            match (src.dtype(), dst.dtype()) {
                (DataType::F32, DataType::BF16) => unsafe {
                    cast_f32_to_bf16_forward(
                        dst.as_bf16_mut()?.data_ptr_mut(),
                        src.as_f32()?.data_ptr(),
                        n, stream);
                }
                (DataType::BF16, DataType::F32) => unsafe {
                    cast_bf16_to_f32_forward(
                        dst.as_f32_mut()?.data_ptr_mut(),
                        src.as_bf16()?.data_ptr(),
                        n, stream);
                }
                (DataType::F32, DataType::F16) => unsafe {
                    cast_f32_to_f16_forward(
                        dst.as_f16_mut()?.data_ptr_mut(),
                        src.as_f32()?.data_ptr(),
                        n, stream);
                }
                (DataType::F16, DataType::F32) => unsafe {
                    cast_f16_to_f32_forward(
                        dst.as_f32_mut()?.data_ptr_mut(),
                        src.as_f16()?.data_ptr(),
                        n, stream);
                }
                (from, to) => return Err(Error::InvalidArgument(format!(
                    "cast_dtype_into CUDA: unsupported {:?} → {:?}", from, to
                )).into()),
            }
            Ok(())
        }
    }
}

// ─────────────────────── tests ───────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    fn f32_from_bf16(t: &Tensor) -> Vec<f32> {
        t.as_bf16().unwrap().as_slice().unwrap()
            .iter().map(|v| v.to_f32()).collect()
    }

    #[test]
    fn cpu_f32_to_bf16_roundtrip() -> Result<()> {
        let mut src = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        src.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let alloc = cast_dtype(&src, DataType::BF16)?;
        let mut into = Tensor::empty(&[4], DataType::BF16, DeviceType::Cpu)?;
        cast_dtype_into(&src, &mut into)?;
        assert_eq!(f32_from_bf16(&alloc), f32_from_bf16(&into));
        Ok(())
    }

    #[test]
    fn identity_is_copy() -> Result<()> {
        let mut src = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        src.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0]);
        let out = cast_dtype(&src, DataType::F32)?;
        assert_eq!(out.as_f32()?.as_slice()?, &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_f32_to_bf16_matches_cpu() -> Result<()> {
        let mut src_cpu = Tensor::empty(&[128], DataType::F32, DeviceType::Cpu)?;
        for (i, v) in src_cpu.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 5.0;
        }
        let src_gpu = src_cpu.to_cuda(0)?;

        let cpu_out = cast_dtype(&src_cpu, DataType::BF16)?;
        let gpu_out = cast_dtype(&src_gpu, DataType::BF16)?.to_cpu()?;
        assert_eq!(f32_from_bf16(&cpu_out), f32_from_bf16(&gpu_out));
        Ok(())
    }
}
