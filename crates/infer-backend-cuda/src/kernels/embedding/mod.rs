//! Embedding CUDA kernel wrapper.
//!
//! Replicated and vocabulary-parallel tables use the same lookup kernel. A
//! replicated table is the special case `vocab_start = 0`, while a sharded
//! table masks token ids outside its local vocabulary range.

use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;

unsafe extern "C" {
    fn embedding_kernel_cu_bf16(
        output: *mut half::bf16,
        indices: *const i32,
        table: *const half::bf16,
        token_len: i32,
        dim: i32,
        vocab_start: i32,
        local_vocab_size: i32,
        stream: cudaStream_t,
    );
    fn embedding_kernel_cu_fp16(
        output: *mut half::f16,
        indices: *const i32,
        table: *const half::f16,
        token_len: i32,
        dim: i32,
        vocab_start: i32,
        local_vocab_size: i32,
        stream: cudaStream_t,
    );
    fn embedding_kernel_cu_fp32(
        output: *mut f32,
        indices: *const i32,
        table: *const f32,
        token_len: i32,
        dim: i32,
        vocab_start: i32,
        local_vocab_size: i32,
        stream: cudaStream_t,
    );
}

/// Element types supported by the unified CUDA embedding lookup.
///
/// # Safety
///
/// All pointers must refer to contiguous tensors on the device associated
/// with `stream`, with the shapes described by the scalar arguments.
pub trait EmbeddingKernel: CudaFloat {
    unsafe fn launch(
        output: *mut Self,
        indices: *const i32,
        table: *const Self,
        token_len: i32,
        dim: i32,
        vocab_start: i32,
        local_vocab_size: i32,
        stream: cudaStream_t,
    );
}

macro_rules! impl_embedding_kernel {
    ($ty:ty, $kernel:ident) => {
        impl EmbeddingKernel for $ty {
            #[inline]
            unsafe fn launch(
                output: *mut Self,
                indices: *const i32,
                table: *const Self,
                token_len: i32,
                dim: i32,
                vocab_start: i32,
                local_vocab_size: i32,
                stream: cudaStream_t,
            ) {
                unsafe {
                    $kernel(
                        output,
                        indices,
                        table,
                        token_len,
                        dim,
                        vocab_start,
                        local_vocab_size,
                        stream,
                    )
                }
            }
        }
    };
}

impl_embedding_kernel!(f32, embedding_kernel_cu_fp32);
impl_embedding_kernel!(half::bf16, embedding_kernel_cu_bf16);
impl_embedding_kernel!(half::f16, embedding_kernel_cu_fp16);

fn launch_embedding<T: EmbeddingKernel>(
    stream: cudaStream_t,
    table: &Tensor<T, Cuda>,
    indices: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
    vocab_start: usize,
) -> OpResult<()> {
    if indices.numel() == 0 {
        return Ok(());
    }

    let table_shape = table.shape().as_slice();
    let local_vocab_size = i32::try_from(table_shape[0])
        .map_err(|_| OpError::Shape("local vocabulary exceeds i32".into()))?;
    let dim = i32::try_from(table_shape[1])
        .map_err(|_| OpError::Shape("embedding dim exceeds i32".into()))?;
    let token_len = i32::try_from(indices.numel())
        .map_err(|_| OpError::Shape("token count exceeds i32".into()))?;
    let vocab_start =
        i32::try_from(vocab_start).map_err(|_| OpError::Shape("vocab start exceeds i32".into()))?;

    unsafe {
        T::launch(
            output.data_ptr_mut(),
            indices.data_ptr(),
            table.data_ptr(),
            token_len,
            dim,
            vocab_start,
            local_vocab_size,
            stream,
        );
    }
    Ok(())
}

pub fn embedding<T: EmbeddingKernel>(
    stream: cudaStream_t,
    table: &Tensor<T, Cuda>,
    indices: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    launch_embedding(stream, table, indices, output, 0)
}

pub fn vocab_embedding<T: EmbeddingKernel>(
    stream: cudaStream_t,
    table: &Tensor<T, Cuda>,
    global_indices: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
    vocab_start: usize,
) -> OpResult<()> {
    launch_embedding(stream, table, global_indices, output, vocab_start)
}
