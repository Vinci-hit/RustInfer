//! Embedding CUDA kernel wrapper.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn embedding_kernel_cu_bf16x8(
        output: *mut half::bf16,
        indices: *const i32,
        table: *const half::bf16,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    );
    fn embedding_kernel_cu_fp16x8(
        output: *mut half::f16,
        indices: *const i32,
        table: *const half::f16,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    );
    fn embedding_kernel_cu_fp32x4(
        output: *mut f32,
        indices: *const i32,
        table: *const f32,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    );
}

pub fn embedding<T: Dtype>(
    stream: cudaStream_t,
    table: &Tensor<T, Cuda>,
    indices: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let table_shape = table.shape().as_slice();
    let vocab = table_shape[0] as i32;
    let dim = table_shape[1] as i32;
    let seq_len = indices.numel() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => embedding_kernel_cu_fp32x4(
                output.data_ptr_mut() as _,
                indices.data_ptr(),
                table.data_ptr() as _,
                seq_len,
                dim,
                vocab,
                stream,
            ),
            DataType::BF16 => embedding_kernel_cu_bf16x8(
                output.data_ptr_mut() as _,
                indices.data_ptr(),
                table.data_ptr() as _,
                seq_len,
                dim,
                vocab,
                stream,
            ),
            DataType::F16 => embedding_kernel_cu_fp16x8(
                output.data_ptr_mut() as _,
                indices.data_ptr(),
                table.data_ptr() as _,
                seq_len,
                dim,
                vocab,
                stream,
            ),
            _ => return Err(OpError::Kernel(format!("embedding: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
