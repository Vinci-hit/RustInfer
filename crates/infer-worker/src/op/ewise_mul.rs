//! Element-wise multiply `dst = a * b` (and in-place `a *= b`).
//!
//! CPU: straight slice loop. CUDA: dedicated kernels in
//! `op/kernels/cuda/ewise_mul/`. Every FFI path uses
//! [`TypedTensor::data_ptr`] / [`TypedTensor::data_ptr_mut`] so that
//! callers can pass strided-offset views (e.g. from `narrow`/`select`)
//! and still hit the right elements.

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn ewise_mul_f32_forward (dst: *mut f32,        a: *const f32,        b: *const f32,        n: i32, stream: cudaStream_t);
    fn ewise_mul_bf16_forward(dst: *mut half::bf16, a: *const half::bf16, b: *const half::bf16, n: i32, stream: cudaStream_t);
    fn ewise_mul_f16_forward (dst: *mut half::f16,  a: *const half::f16,  b: *const half::f16,  n: i32, stream: cudaStream_t);
    fn ewise_mul_inplace_f32_forward (a: *mut f32,        b: *const f32,        n: i32, stream: cudaStream_t);
    fn ewise_mul_inplace_bf16_forward(a: *mut half::bf16, b: *const half::bf16, n: i32, stream: cudaStream_t);
    fn ewise_mul_inplace_f16_forward (a: *mut half::f16,  b: *const half::f16,  n: i32, stream: cudaStream_t);
}

/// `dst = a * b` (same shape). Device-agnostic.
pub fn ewise_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != dst.shape() {
        return Err(Error::InvalidArgument(format!(
            "ewise_mul: shape mismatch a={:?} b={:?} dst={:?}",
            a.shape(), b.shape(), dst.shape())).into());
    }
    let n = a.numel() as i32;
    match a.device() {
        DeviceType::Cpu => ewise_mul_cpu(a, b, dst),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            match a.dtype() {
                DataType::F32 => unsafe {
                    ewise_mul_f32_forward(
                        dst.as_f32_mut()?.data_ptr_mut(),
                        a.as_f32()?.data_ptr(),
                        b.as_f32()?.data_ptr(),
                        n, stream);
                }
                DataType::BF16 => unsafe {
                    ewise_mul_bf16_forward(
                        dst.as_bf16_mut()?.data_ptr_mut(),
                        a.as_bf16()?.data_ptr(),
                        b.as_bf16()?.data_ptr(),
                        n, stream);
                }
                DataType::F16 => unsafe {
                    ewise_mul_f16_forward(
                        dst.as_f16_mut()?.data_ptr_mut(),
                        a.as_f16()?.data_ptr(),
                        b.as_f16()?.data_ptr(),
                        n, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "ewise_mul CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

/// In-place `a *= b` (same shape). Device-agnostic.
pub fn ewise_mul_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument(format!(
            "ewise_mul_inplace: shape mismatch a={:?} b={:?}",
            a.shape(), b.shape())).into());
    }
    let n = a.numel() as i32;
    match a.device() {
        DeviceType::Cpu => ewise_mul_inplace_cpu(a, b),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            match a.dtype() {
                DataType::F32 => unsafe {
                    ewise_mul_inplace_f32_forward(
                        a.as_f32_mut()?.data_ptr_mut(),
                        b.as_f32()?.data_ptr(),
                        n, stream);
                }
                DataType::BF16 => unsafe {
                    ewise_mul_inplace_bf16_forward(
                        a.as_bf16_mut()?.data_ptr_mut(),
                        b.as_bf16()?.data_ptr(),
                        n, stream);
                }
                DataType::F16 => unsafe {
                    ewise_mul_inplace_f16_forward(
                        a.as_f16_mut()?.data_ptr_mut(),
                        b.as_f16()?.data_ptr(),
                        n, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "ewise_mul_inplace CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

// ─────────────────────── CPU paths ───────────────────────

fn ewise_mul_cpu(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    match a.dtype() {
        DataType::F32 => {
            let a_s = a.as_f32()?.as_slice()?;
            let b_s = b.as_f32()?.as_slice()?;
            let d_s = dst.as_f32_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() { d_s[i] = a_s[i] * b_s[i]; }
        }
        DataType::BF16 => {
            let a_s = a.as_bf16()?.as_slice()?;
            let b_s = b.as_bf16()?.as_slice()?;
            let d_s = dst.as_bf16_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() {
                d_s[i] = half::bf16::from_f32(a_s[i].to_f32() * b_s[i].to_f32());
            }
        }
        other => return Err(Error::InvalidArgument(format!(
            "ewise_mul CPU: unsupported dtype {:?}", other)).into()),
    }
    Ok(())
}

fn ewise_mul_inplace_cpu(a: &mut Tensor, b: &Tensor) -> Result<()> {
    match a.dtype() {
        DataType::F32 => {
            let b_vec: Vec<f32> = b.as_f32()?.as_slice()?.to_vec();
            let a_s = a.as_f32_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() { a_s[i] *= b_vec[i]; }
        }
        DataType::BF16 => {
            let b_vec: Vec<half::bf16> = b.as_bf16()?.as_slice()?.to_vec();
            let a_s = a.as_bf16_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() {
                a_s[i] = half::bf16::from_f32(a_s[i].to_f32() * b_vec[i].to_f32());
            }
        }
        other => return Err(Error::InvalidArgument(format!(
            "ewise_mul_inplace CPU: unsupported dtype {:?}", other)).into()),
    }
    Ok(())
}

// ─────────────────────── tests ───────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    #[test]
    fn cpu_f32() -> Result<()> {
        let mut a = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[2.0, 3.0, 4.0, 5.0]);
        let mut dst = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        ewise_mul(&a, &b, &mut dst)?;
        assert_eq!(dst.as_f32()?.as_slice()?, &[2.0, 6.0, 12.0, 20.0]);
        Ok(())
    }

    #[test]
    fn cpu_inplace_f32() -> Result<()> {
        let mut a = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[4.0, 5.0, 6.0]);
        ewise_mul_inplace(&mut a, &b)?;
        assert_eq!(a.as_f32()?.as_slice()?, &[4.0, 10.0, 18.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_f32_matches_cpu() -> Result<()> {
        let mut a = Tensor::empty(&[128], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[128], DataType::F32, DeviceType::Cpu)?;
        for i in 0..128 {
            a.as_f32_mut()?.as_slice_mut()?[i] = (i as f32) * 0.1 + 0.5;
            b.as_f32_mut()?.as_slice_mut()?[i] = (i as f32) * 0.2 + 1.0;
        }
        let mut dst_cpu = Tensor::empty(&[128], DataType::F32, DeviceType::Cpu)?;
        ewise_mul(&a, &b, &mut dst_cpu)?;

        let a_gpu = a.to_cuda(0)?;
        let b_gpu = b.to_cuda(0)?;
        let mut dst_gpu = Tensor::empty(&[128], DataType::F32, DeviceType::Cuda(0))?;
        ewise_mul(&a_gpu, &b_gpu, &mut dst_gpu)?;
        let dst_gpu_cpu = dst_gpu.to_cpu()?;
        assert_eq!(dst_cpu.as_f32()?.as_slice()?, dst_gpu_cpu.as_f32()?.as_slice()?);
        Ok(())
    }
}
