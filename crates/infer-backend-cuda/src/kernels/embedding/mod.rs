//! Embedding CUDA kernel wrapper.
//!
//! Dispatch is an attribute of the element type: [`EmbeddingKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry point, so [`embedding`] is generic with no runtime `match`. Adding a
//! dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

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

/// Element types with an embedding-gather CUDA kernel. The method forwards to
/// this dtype's `extern` entry; the wrapper below is generic over this trait, so
/// the dtype→kernel mapping lives here as a type attribute. Indices are always
/// `i32` regardless of `Self`.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for the given
/// vocab/dim/token layout on `stream`; this just names the FFI entry and
/// performs no checks.
pub trait EmbeddingKernel: CudaFloat {
    /// Gather rows of `table` selected by `indices` into `output`.
    unsafe fn embedding(
        output: *mut Self,
        indices: *const i32,
        table: *const Self,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    );
}

impl EmbeddingKernel for f32 {
    #[inline]
    unsafe fn embedding(
        output: *mut Self,
        indices: *const i32,
        table: *const Self,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    ) {
        unsafe {
            embedding_kernel_cu_fp32x4(output, indices, table, token_len, dim, vocab_size, stream)
        }
    }
}

impl EmbeddingKernel for half::bf16 {
    #[inline]
    unsafe fn embedding(
        output: *mut Self,
        indices: *const i32,
        table: *const Self,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    ) {
        unsafe {
            embedding_kernel_cu_bf16x8(output, indices, table, token_len, dim, vocab_size, stream)
        }
    }
}

impl EmbeddingKernel for half::f16 {
    #[inline]
    unsafe fn embedding(
        output: *mut Self,
        indices: *const i32,
        table: *const Self,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    ) {
        unsafe {
            embedding_kernel_cu_fp16x8(output, indices, table, token_len, dim, vocab_size, stream)
        }
    }
}

pub fn embedding<T: EmbeddingKernel>(
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
        T::embedding(
            output.data_ptr_mut(),
            indices.data_ptr(),
            table.data_ptr(),
            seq_len,
            dim,
            vocab,
            stream,
        );
    }
    Ok(())
}
