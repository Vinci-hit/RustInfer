//! CUDA kernel wrappers — paged KV path only.

pub mod rmsnorm;
pub mod add;
pub mod matmul;
pub mod softmax;
pub mod activation;
pub mod scalar;
pub mod embedding;
pub mod rope;
pub mod attention_paged;
pub mod fused_add_rmsnorm;
// pub mod sampler;
pub mod split_cols;
pub mod scatter_kv_paged;
pub mod argmax_batched;
pub mod gather_merge;
pub mod groupnorm;
pub mod upsample;
pub mod broadcast_mul;
pub mod ewise_mul;
pub mod layernorm;
pub mod conv2d;
pub mod rope_interleaved;
pub mod cast_dtype;
pub mod sdpa;
pub mod concat_seq;
pub mod pad;
