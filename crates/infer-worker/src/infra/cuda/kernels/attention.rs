//! Flash Attention CUDA kernel wrapper.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    pub fn launch_flash_attn_prefill_bf16(
        output: *mut half::bf16, q: *const half::bf16, k: *const half::bf16, v: *const half::bf16,
        batch: i32, seq_len: i32, head_num: i32, head_dim: i32,
        kv_head_num: i32, scale: f32, stream: cudaStream_t,
    );
    pub fn launch_flash_attn_prefill_fp16(
        output: *mut half::f16, q: *const half::f16, k: *const half::f16, v: *const half::f16,
        batch: i32, seq_len: i32, head_num: i32, head_dim: i32,
        kv_head_num: i32, scale: f32, stream: cudaStream_t,
    );
}

/// Scaled dot-product attention (prefill path, simplified).
/// q/k/v: [batch, seq_len, head_dim], output: [batch, seq_len, head_dim]
pub fn attention_prefill<T: Dtype>(
    q: &Tensor<T, Cuda>,
    k: &Tensor<T, Cuda>,
    v: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    head_num: i32,
    kv_head_num: i32,
    head_dim: i32,
    scale: f32,
) -> OpResult<()> {
    let shape = q.shape().as_slice();
    let batch = if shape.len() >= 3 { shape[0] } else { 1 };
    let seq_len = if shape.len() >= 3 { shape[1] } else { shape[0] };
    let stream = q.device().config.stream;

    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => launch_flash_attn_prefill_bf16(
                output.data_ptr_mut() as _, q.data_ptr() as _, k.data_ptr() as _, v.data_ptr() as _,
                batch as i32, seq_len as i32, head_num, head_dim, kv_head_num, scale, stream,
            ),
            DataType::F16 => launch_flash_attn_prefill_fp16(
                output.data_ptr_mut() as _, q.data_ptr() as _, k.data_ptr() as _, v.data_ptr() as _,
                batch as i32, seq_len as i32, head_num, head_dim, kv_head_num, scale, stream,
            ),
            _ => return Err(OpError::Kernel(format!("attention: {:?} (only bf16/fp16 supported)", T::DATA_TYPE))),
        }
    }
    Ok(())
}
