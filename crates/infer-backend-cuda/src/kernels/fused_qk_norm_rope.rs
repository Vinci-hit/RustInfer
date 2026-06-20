//! Fused QK Norm + RoPE CUDA kernel wrapper.
//!
//! Fuses Q/K Layernorm (pre-RoPE) with RoPE application into a single kernel.
//! Input:  Q [num_tokens, q_dim], K [num_tokens, kv_dim]
//! Output: Q [num_tokens, q_dim], K [num_tokens, kv_dim]  (RoPE-applied)
//! Layernorm weights: q_weight [q_dim], k_weight [kv_dim], eps
//! RoPE tables: sin [max_seq_len, head_dim/2], cos [max_seq_len, head_dim/2]
//! positions: device pointer [num_tokens] i32

use infer_core::ports::{OpResult, OpError};
use infer_core::types::{DataType, Dtype};
use infer_core::tensor::Tensor;
use crate::Cuda;
use crate::ffi::cudaStream_t;

// ─── C kernel declarations ─────────────────────────────────────────────────────

unsafe extern "C" {
    /// fused_qk_norm_rope for f32
    fn fused_qk_norm_rope_cu_fp32(
        q_out: *mut f32,
        k_out: *mut f32,
        q_in: *const f32,
        k_in: *const f32,
        q_weight: *const f32,
        k_weight: *const f32,
        sin: *const f32,
        cos: *const f32,
        positions: *const i32,
        num_tokens: i32,
        q_dim: i32,
        kv_dim: i32,
        head_dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );

    /// fused_qk_norm_rope for bf16
    fn fused_qk_norm_rope_cu_bf16(
        q_out: *mut half::bf16,
        k_out: *mut half::bf16,
        q_in: *const half::bf16,
        k_in: *const half::bf16,
        q_weight: *const half::bf16,
        k_weight: *const half::bf16,
        sin: *const half::bf16,
        cos: *const half::bf16,
        positions: *const i32,
        num_tokens: i32,
        q_dim: i32,
        kv_dim: i32,
        head_dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );

    /// fused_qk_norm_rope for fp16
    fn fused_qk_norm_rope_cu_fp16(
        q_out: *mut half::f16,
        k_out: *mut half::f16,
        q_in: *const half::f16,
        k_in: *const half::f16,
        q_weight: *const half::f16,
        k_weight: *const half::f16,
        sin: *const half::f16,
        cos: *const half::f16,
        positions: *const i32,
        num_tokens: i32,
        q_dim: i32,
        kv_dim: i32,
        head_dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fused QK Layernorm + RoPE.
///
/// Applies Layernorm to Q and K (along the head_dim axis), then applies RoPE
/// in-place on the normalized Q/K.  All done in one kernel launch.
///
/// # Arguments
/// * `q_out`  — output Q tensor, shape [num_tokens, q_dim]
/// * `k_out`  — output K tensor, shape [num_tokens, kv_dim]
/// * `q_in`   — input  Q tensor, shape [num_tokens, q_dim]
/// * `k_in`   — input  K tensor, shape [num_tokens, kv_dim]
/// * `q_weight` — Q Layernorm weight, shape [q_dim]
/// * `k_weight` — K Layernorm weight, shape [kv_dim]
/// * `sin`    — sin cache, shape [max_seq_len, head_dim/2]
/// * `cos`    — cos cache, shape [max_seq_len, head_dim/2]
/// * `positions_dev` — device pointer to positions [num_tokens] i32
/// * `num_tokens` — number of tokens (batch * seq_len)
/// * `head_num`   — number of Q heads
/// * `kv_head_num` — number of KV heads
/// * `head_dim`    — head dimension
/// * `eps`         — Layernorm epsilon
pub fn fused_qk_norm_rope<T: Dtype>(
    stream: cudaStream_t,
    q_out: &mut Tensor<T, Cuda>,
    k_out: &mut Tensor<T, Cuda>,
    q_in: &Tensor<T, Cuda>,
    k_in: &Tensor<T, Cuda>,
    q_weight: &Tensor<T, Cuda>,
    k_weight: &Tensor<T, Cuda>,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions_dev: *const i32,
    num_tokens: i32,
    head_num: i32,
    kv_head_num: i32,
    head_dim: i32,
    eps: f32,
) -> OpResult<()> {
    let q_dim = head_num * head_dim;
    let kv_dim = kv_head_num * head_dim;

    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => fused_qk_norm_rope_cu_fp32(
                q_out.data_ptr_mut() as _,
                k_out.data_ptr_mut() as _,
                q_in.data_ptr() as _,
                k_in.data_ptr() as _,
                q_weight.data_ptr() as _,
                k_weight.data_ptr() as _,
                sin.data_ptr() as _,
                cos.data_ptr() as _,
                positions_dev,
                num_tokens,
                q_dim,
                kv_dim,
                head_dim,
                eps,
                stream,
            ),
            DataType::BF16 => fused_qk_norm_rope_cu_bf16(
                q_out.data_ptr_mut() as _,
                k_out.data_ptr_mut() as _,
                q_in.data_ptr() as _,
                k_in.data_ptr() as _,
                q_weight.data_ptr() as _,
                k_weight.data_ptr() as _,
                sin.data_ptr() as _,
                cos.data_ptr() as _,
                positions_dev,
                num_tokens,
                q_dim,
                kv_dim,
                head_dim,
                eps,
                stream,
            ),
            DataType::F16 => fused_qk_norm_rope_cu_fp16(
                q_out.data_ptr_mut() as _,
                k_out.data_ptr_mut() as _,
                q_in.data_ptr() as _,
                k_in.data_ptr() as _,
                q_weight.data_ptr() as _,
                k_weight.data_ptr() as _,
                sin.data_ptr() as _,
                cos.data_ptr() as _,
                positions_dev,
                num_tokens,
                q_dim,
                kv_dim,
                head_dim,
                eps,
                stream,
            ),
            _ => {
                return Err(OpError::Kernel(format!(
                    "fused_qk_norm_rope: unsupported dtype {:?}",
                    T::DATA_TYPE
                )))
            }
        }
    }
    Ok(())
}
