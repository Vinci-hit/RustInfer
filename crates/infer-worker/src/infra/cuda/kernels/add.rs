//! Add CUDA kernel wrapper.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn add_kernel_bf16x8(c: *mut half::bf16, a: *const half::bf16, b: *const half::bf16, n: i32, stream: cudaStream_t);
    fn add_kernel_fp16x8(c: *mut half::f16, a: *const half::f16, b: *const half::f16, n: i32, stream: cudaStream_t);
    fn add_inplace_kernel_bf16x8(a: *mut half::bf16, b: *const half::bf16, n: i32, stream: cudaStream_t);
    fn add_inplace_kernel_fp16x8(a: *mut half::f16, b: *const half::f16, n: i32, stream: cudaStream_t);
    fn add_kernel_float2_forward(c: *mut f32, a: *const f32, b: *const f32, n: i32, stream: cudaStream_t);
    fn add_inplace_kernel_float2_forward(a: *mut f32, b: *const f32, n: i32, stream: cudaStream_t);
}

pub fn add<T: Dtype>(a: &Tensor<T, Cuda>, b: &Tensor<T, Cuda>, dst: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let n = a.numel() as i32;
    let stream = a.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => add_kernel_float2_forward(dst.data_ptr_mut() as _, a.data_ptr() as _, b.data_ptr() as _, n, stream),
            DataType::BF16 => add_kernel_bf16x8(dst.data_ptr_mut() as _, a.data_ptr() as _, b.data_ptr() as _, n, stream),
            DataType::F16 => add_kernel_fp16x8(dst.data_ptr_mut() as _, a.data_ptr() as _, b.data_ptr() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("add: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

pub fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Cuda>, src: &Tensor<T, Cuda>) -> OpResult<()> {
    let n = dst.numel() as i32;
    let stream = dst.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => add_inplace_kernel_float2_forward(dst.data_ptr_mut() as _, src.data_ptr() as _, n, stream),
            DataType::BF16 => add_inplace_kernel_bf16x8(dst.data_ptr_mut() as _, src.data_ptr() as _, n, stream),
            DataType::F16 => add_inplace_kernel_fp16x8(dst.data_ptr_mut() as _, src.data_ptr() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("add_inplace: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
