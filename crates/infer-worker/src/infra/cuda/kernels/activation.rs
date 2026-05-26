//! SwiGLU + SiLU CUDA kernel wrappers.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn swiglu_kernel_bf16(x: *mut half::bf16, gate: *const half::bf16, n: i32, stream: cudaStream_t);
    fn swiglu_kernel_fp16(x: *mut half::f16, gate: *const half::f16, n: i32, stream: cudaStream_t);
    fn swiglu_kernel_fp32(x: *mut f32, gate: *const f32, n: i32, stream: cudaStream_t);
    fn silu_kernel_bf16(x: *mut half::bf16, n: i32, stream: cudaStream_t);
    fn silu_kernel_fp16(x: *mut half::f16, n: i32, stream: cudaStream_t);
    fn silu_kernel_fp32(x: *mut f32, n: i32, stream: cudaStream_t);
}

pub fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => silu_kernel_fp32(x.data_ptr_mut() as _, n, stream),
            DataType::BF16 => silu_kernel_bf16(x.data_ptr_mut() as _, n, stream),
            DataType::F16 => silu_kernel_fp16(x.data_ptr_mut() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("silu: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

pub fn swiglu_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>, gate: &Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => swiglu_kernel_fp32(x.data_ptr_mut() as _, gate.data_ptr() as _, n, stream),
            DataType::BF16 => swiglu_kernel_bf16(x.data_ptr_mut() as _, gate.data_ptr() as _, n, stream),
            DataType::F16 => swiglu_kernel_fp16(x.data_ptr_mut() as _, gate.data_ptr() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("swiglu: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
