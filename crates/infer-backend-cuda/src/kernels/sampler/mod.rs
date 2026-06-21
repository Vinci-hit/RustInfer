//! Argmax sampler CUDA kernel.

use infer_core::tensor::Tensor;
use infer_core::ports::{OpError, OpResult};
use infer_core::types::{DataType, Dtype};
use crate::Cuda;
use crate::ffi::cudaStream_t;

unsafe extern "C" {
    // Signature: (logits, vocab_size, result_gpu, workspace, stream)
    fn argmax_cu_bf16_ffi(input: *const half::bf16, batch_size: i32, vocab_size: i32, output: *mut i32, workspace: *mut f32, stream: cudaStream_t);
    fn argmax_cu_fp16_ffi(input: *const half::f16, vocab_size: i32, output: *mut i32, workspace: *mut f32, stream: cudaStream_t);
    fn argmax_cu_f32_ffi(input: *const f32, vocab_size: i32, output: *mut i32, workspace: *mut f32, stream: cudaStream_t);
}


pub fn argmax<T: Dtype>(
    stream: cudaStream_t,
    logits: &Tensor<T, Cuda>,
    output: &mut Tensor<i32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
) -> OpResult<()> {
    let vocab_size = *logits.shape().as_slice().last().unwrap() as i32;
    let batch_size = *logits.shape().as_slice().first().unwrap() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => argmax_cu_f32_ffi(logits.data_ptr() as _, vocab_size, output.data_ptr_mut(), workspace.data_ptr_mut(), stream),
            DataType::BF16 => argmax_cu_bf16_ffi(logits.data_ptr() as _, batch_size, vocab_size, output.data_ptr_mut(), workspace.data_ptr_mut(), stream),
            DataType::F16 => argmax_cu_fp16_ffi(logits.data_ptr() as _, vocab_size, output.data_ptr_mut(), workspace.data_ptr_mut(), stream),
            _ => return Err(OpError::Kernel(format!("argmax: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
