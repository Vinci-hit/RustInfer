//! Concatenate two 2D tensors along dim 0: `[S_a, D] + [S_b, D] → [S_a+S_b, D]`.
//!
//! Device-agnostic. CUDA uses two `cudaMemcpyAsync` D2D copies; CPU uses
//! `ptr::copy_nonoverlapping`. Pointers flow through `data_ptr()` so
//! offset views are handled correctly.

use crate::base::error::{Error, Result};
use crate::base::DeviceType;
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
#[inline]
unsafe fn d2d_memcpy_async(
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    count: usize,
    stream: cudaStream_t,
) -> Result<()> {
    unsafe {
        let rc = crate::cuda::ffi::cudaMemcpyAsync(
            dst, src, count,
            crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream);
        if rc != crate::cuda::ffi::cudaError_cudaSuccess {
            return Err(Error::InternalError(
                format!("cudaMemcpyAsync D2D failed: {}", rc)).into());
        }
    }
    Ok(())
}

/// Concatenate two `[*, D]` tensors along dim 0. Both inputs must share
/// dtype and device; the returned tensor is contiguous and lives on the
/// same device.
pub fn concat_seq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    validate_shapes(a, b, "concat_seq")?;
    let (s_a, d) = (a.shape()[0], a.shape()[1]);
    let s_b = b.shape()[0];
    let mut dst = Tensor::empty(&[s_a + s_b, d], a.dtype(), a.device())?;
    concat_seq_into(a, b, &mut dst)?;
    Ok(dst)
}

/// `dst[..s_a] = a; dst[s_a..] = b`.
pub fn concat_seq_into(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    validate_shapes(a, b, "concat_seq_into")?;
    let (s_a, d) = (a.shape()[0], a.shape()[1]);
    let s_b = b.shape()[0];
    if dst.shape() != [s_a + s_b, d].as_slice() {
        return Err(Error::InvalidArgument(format!(
            "concat_seq_into: dst shape {:?} incompatible with [{}, {}]",
            dst.shape(), s_a + s_b, d)).into());
    }
    if dst.dtype() != a.dtype() || dst.device() != a.device() {
        return Err(Error::InvalidArgument(
            "concat_seq_into: dst dtype/device mismatch".into()).into());
    }

    let bytes_per_row = d * a.dtype().size_in_bytes();
    let a_bytes = s_a * bytes_per_row;
    let b_bytes = s_b * bytes_per_row;

    match a.device() {
        DeviceType::Cpu => unsafe {
            let dst_base = dst.data_ptr_mut();
            std::ptr::copy_nonoverlapping(a.data_ptr(), dst_base, a_bytes);
            std::ptr::copy_nonoverlapping(b.data_ptr(), dst_base.add(a_bytes), b_bytes);
            Ok(())
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            unsafe {
                let dst_base = dst.data_ptr_mut() as *mut core::ffi::c_void;
                d2d_memcpy_async(dst_base, a.data_ptr() as *const _, a_bytes, stream)?;
                d2d_memcpy_async(
                    (dst_base as *mut u8).add(a_bytes) as *mut _,
                    b.data_ptr() as *const _,
                    b_bytes, stream)?;
            }
            Ok(())
        }
    }
}

fn validate_shapes(a: &Tensor, b: &Tensor, ctx: &str) -> Result<()> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(Error::InvalidArgument(
            format!("{ctx}: expected 2D tensors, got a={:?} b={:?}",
                a.shape(), b.shape())).into());
    }
    if a.shape()[1] != b.shape()[1] {
        return Err(Error::InvalidArgument(
            format!("{ctx}: last-dim mismatch a={} b={}",
                a.shape()[1], b.shape()[1])).into());
    }
    if a.dtype() != b.dtype() || a.device() != b.device() {
        return Err(Error::InvalidArgument(
            format!("{ctx}: dtype/device mismatch").into()).into());
    }
    Ok(())
}

// ─────────────────────── tests ───────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    #[test]
    fn concat_cpu_f32() -> Result<()> {
        let mut a = Tensor::empty(&[2, 3], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[1, 3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[7.0, 8.0, 9.0]);

        let out = concat_seq(&a, &b)?;
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(out.as_f32()?.as_slice()?,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn concat_cuda_matches_cpu() -> Result<()> {
        let mut a = Tensor::empty(&[3, 4], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[2, 4], DataType::F32, DeviceType::Cpu)?;
        for i in 0..12 { a.as_f32_mut()?.as_slice_mut()?[i] = i as f32; }
        for i in 0..8  { b.as_f32_mut()?.as_slice_mut()?[i] = (i + 100) as f32; }

        let cpu_out = concat_seq(&a, &b)?;
        let gpu_out = concat_seq(&a.to_cuda(0)?, &b.to_cuda(0)?)?.to_cpu()?;
        assert_eq!(cpu_out.as_f32()?.as_slice()?, gpu_out.as_f32()?.as_slice()?);
        Ok(())
    }
}
