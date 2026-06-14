//! CUDA kernel wrappers — paged KV path only.

pub mod activation;
pub mod add;
pub mod attention_paged;
pub mod embedding;
pub mod fused_add_rmsnorm;
pub mod matmul;
pub mod qkv_norm_rope_scatter;
pub mod rmsnorm;
pub mod rope;
pub mod scalar;
pub mod softmax;
// pub mod sampler;
pub mod argmax_batched;
pub mod broadcast_mul;
pub mod cast_dtype;
pub mod concat_seq;
pub mod conv2d;
pub mod ewise_mul;
pub mod gather_merge;
pub mod groupnorm;
pub mod layernorm;
pub mod pad;
pub mod rope_interleaved;
pub mod scatter_kv_paged;
pub mod sdpa;
pub mod split_cols;
pub mod upsample;
