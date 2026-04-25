use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn broadcast_mul_f32_forward(dst: *mut f32, a: *const f32, b: *const f32, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_mul_bf16_forward(dst: *mut half::bf16, a: *const half::bf16, b: *const half::bf16, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_mul_f16_forward(dst: *mut half::f16, a: *const half::f16, b: *const half::f16, rows: i32, d: i32, stream: cudaStream_t);
}

/// dst[i, j] = a[i, j] * b[j]
/// a: [rows, D], b: [D], dst: [rows, D]
#[cfg(feature = "cuda")]
pub fn broadcast_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor, rows: i32, d: i32, stream: cudaStream_t) -> Result<()> {
    match a.dtype() {
        crate::base::DataType::F32 => {
            let ap = a.as_f32()?.buffer().as_ptr() as *const f32;
            let bp = b.as_f32()?.buffer().as_ptr() as *const f32;
            let dp = dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { broadcast_mul_f32_forward(dp, ap, bp, rows, d, stream); }
        }
        crate::base::DataType::BF16 => {
            let ap = a.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let bp = b.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let dp = dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { broadcast_mul_bf16_forward(dp, ap, bp, rows, d, stream); }
        }
        crate::base::DataType::F16 => {
            let ap = a.as_f16()?.buffer().as_ptr() as *const half::f16;
            let bp = b.as_f16()?.buffer().as_ptr() as *const half::f16;
            let dp = dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { broadcast_mul_f16_forward(dp, ap, bp, rows, d, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA broadcast_mul: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}
