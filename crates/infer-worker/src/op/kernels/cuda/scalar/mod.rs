use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn scalar_mul_f32_forward(dst: *mut f32, src: *const f32, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_bf16_forward(dst: *mut half::bf16, src: *const half::bf16, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_f16_forward(dst: *mut half::f16, src: *const half::f16, val: f32, n: i32, stream: cudaStream_t);

    fn scalar_add_f32_forward(dst: *mut f32, src: *const f32, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_add_bf16_forward(dst: *mut half::bf16, src: *const half::bf16, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_add_f16_forward(dst: *mut half::f16, src: *const half::f16, val: f32, n: i32, stream: cudaStream_t);
}

/// dst[i] = src[i] * val  (CUDA)
#[cfg(feature = "cuda")]
pub fn scalar_mul(src: &Tensor, dst: &mut Tensor, val: f32, stream: cudaStream_t) -> Result<()> {
    let n = src.num_elements() as i32;
    match src.dtype() {
        crate::base::DataType::F32 => {
            let s = src.as_f32()?.buffer().as_ptr() as *const f32;
            let d = dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_mul_f32_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let s = src.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let d = dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { scalar_mul_bf16_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::F16 => {
            let s = src.as_f16()?.buffer().as_ptr() as *const half::f16;
            let d = dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { scalar_mul_f16_forward(d, s, val, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA scalar_mul: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}

/// dst[i] = src[i] + val  (CUDA)
#[cfg(feature = "cuda")]
pub fn scalar_add(src: &Tensor, dst: &mut Tensor, val: f32, stream: cudaStream_t) -> Result<()> {
    let n = src.num_elements() as i32;
    match src.dtype() {
        crate::base::DataType::F32 => {
            let s = src.as_f32()?.buffer().as_ptr() as *const f32;
            let d = dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_add_f32_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let s = src.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let d = dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { scalar_add_bf16_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::F16 => {
            let s = src.as_f16()?.buffer().as_ptr() as *const half::f16;
            let d = dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { scalar_add_f16_forward(d, s, val, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA scalar_add: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}
