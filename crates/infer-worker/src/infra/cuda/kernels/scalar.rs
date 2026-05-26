//! Scalar ops CUDA kernel wrapper.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn scalar_mul_inplace_bf16(x: *mut half::bf16, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_inplace_fp16(x: *mut half::f16, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_inplace_fp32(x: *mut f32, val: f32, n: i32, stream: cudaStream_t);
}

pub fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>, scalar: f64) -> OpResult<()> {
    let n = x.numel() as i32;
    let val = scalar as f32;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => scalar_mul_inplace_fp32(x.data_ptr_mut() as _, val, n, stream),
            DataType::BF16 => scalar_mul_inplace_bf16(x.data_ptr_mut() as _, val, n, stream),
            DataType::F16 => scalar_mul_inplace_fp16(x.data_ptr_mut() as _, val, n, stream),
            _ => return Err(OpError::Kernel(format!("scalar_mul: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
