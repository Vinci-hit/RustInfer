//! Softmax CUDA kernel wrapper.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn softmax_kernel_bf16x8(output: *mut half::bf16, input: *const half::bf16, rows: i32, dim: i32, stream: cudaStream_t);
    fn softmax_kernel_fp16x8(output: *mut half::f16, input: *const half::f16, rows: i32, dim: i32, stream: cudaStream_t);
    fn softmax_kernel_fp32x4(output: *mut f32, input: *const f32, rows: i32, dim: i32, stream: cudaStream_t);
}

pub fn softmax<T: Dtype>(input: &Tensor<T, Cuda>, output: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let dim = *input.shape().as_slice().last().unwrap_or(&1);
    let rows = (input.numel() / dim) as i32;
    let stream = input.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => softmax_kernel_fp32x4(output.data_ptr_mut() as _, input.data_ptr() as _, rows, dim as i32, stream),
            DataType::BF16 => softmax_kernel_bf16x8(output.data_ptr_mut() as _, input.data_ptr() as _, rows, dim as i32, stream),
            DataType::F16 => softmax_kernel_fp16x8(output.data_ptr_mut() as _, input.data_ptr() as _, rows, dim as i32, stream),
            _ => return Err(OpError::Kernel(format!("softmax: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
