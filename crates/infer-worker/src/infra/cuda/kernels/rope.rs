//! RoPE (Rotary Position Embedding) CUDA kernel wrapper.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    pub fn rope_kernel_cu(
        q: *mut f32, k: *mut f32, sin: *const f32, cos: *const f32,
        positions: *const i32, num_tokens: i32, head_num: i32, kv_head_num: i32,
        head_dim: i32, stream: cudaStream_t,
    );
    pub fn rope_kernel_cu_bf16(
        q: *mut half::bf16, k: *mut half::bf16, sin: *const half::bf16, cos: *const half::bf16,
        positions: *const i32, num_tokens: i32, head_num: i32, kv_head_num: i32,
        head_dim: i32, stream: cudaStream_t,
    );
    pub fn rope_kernel_cu_fp16(
        q: *mut half::f16, k: *mut half::f16, sin: *const half::f16, cos: *const half::f16,
        positions: *const i32, num_tokens: i32, head_num: i32, kv_head_num: i32,
        head_dim: i32, stream: cudaStream_t,
    );
}

/// Apply RoPE in-place to Q and K tensors.
/// q: [num_tokens, q_dim], k: [num_tokens, kv_dim]
/// sin/cos: [max_seq_len, head_dim]
/// positions: device pointer to [num_tokens] i32
pub fn rope_inplace<T: Dtype>(
    q: &mut Tensor<T, Cuda>,
    k: &mut Tensor<T, Cuda>,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions_dev: *const i32,
    num_tokens: i32,
    head_num: i32,
    kv_head_num: i32,
    head_dim: i32,
) -> OpResult<()> {
    let stream = q.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => rope_kernel_cu(
                q.data_ptr_mut() as _, k.data_ptr_mut() as _,
                sin.data_ptr() as _, cos.data_ptr() as _,
                positions_dev, num_tokens, head_num, kv_head_num, head_dim, stream,
            ),
            DataType::BF16 => rope_kernel_cu_bf16(
                q.data_ptr_mut() as _, k.data_ptr_mut() as _,
                sin.data_ptr() as _, cos.data_ptr() as _,
                positions_dev, num_tokens, head_num, kv_head_num, head_dim, stream,
            ),
            DataType::F16 => rope_kernel_cu_fp16(
                q.data_ptr_mut() as _, k.data_ptr_mut() as _,
                sin.data_ptr() as _, cos.data_ptr() as _,
                positions_dev, num_tokens, head_num, kv_head_num, head_dim, stream,
            ),
            _ => return Err(OpError::Kernel(format!("rope: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
