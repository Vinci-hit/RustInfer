//! Argmax sampler CUDA kernel.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn argmax_kernel_bf16(output: *mut i32, input: *const half::bf16, vocab_size: i32, stream: cudaStream_t);
    fn argmax_kernel_fp16(output: *mut i32, input: *const half::f16, vocab_size: i32, stream: cudaStream_t);
    fn argmax_kernel_fp32(output: *mut i32, input: *const f32, vocab_size: i32, stream: cudaStream_t);
}

/// Argmax over a single row of logits → output token ID.
pub fn argmax<T: Dtype>(logits: &Tensor<T, Cuda>, output: &mut Tensor<i32, Cuda>) -> OpResult<()> {
    let vocab_size = *logits.shape().as_slice().last().unwrap() as i32;
    let stream = logits.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => argmax_kernel_fp32(output.data_ptr_mut(), logits.data_ptr() as _, vocab_size, stream),
            DataType::BF16 => argmax_kernel_bf16(output.data_ptr_mut(), logits.data_ptr() as _, vocab_size, stream),
            DataType::F16 => argmax_kernel_fp16(output.data_ptr_mut(), logits.data_ptr() as _, vocab_size, stream),
            _ => return Err(OpError::Kernel(format!("argmax: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
