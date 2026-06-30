//! CUDA kernel wrappers — paged KV path only.
//! Each kernel co-located with its `.cu`/`.h` source in its own dir.

// --- kernels with a co-located mod.rs in their dir ---
pub mod add;
pub mod broadcast_mul;
pub mod embedding;
pub mod ewise_mul;
pub mod flash_attn_gqa; // was attention_paged
pub mod fused_add_rmsnorm;
pub mod gather_merge;
pub mod groupnorm;
pub mod kv_cache; // was scatter_kv_paged
pub mod layernorm;
pub mod matmul;
pub mod qkv_norm_rope_scatter;
pub mod rmsnorm;
pub mod rope;
pub mod rope_interleaved;
pub mod sampler;
pub mod scalar;
pub mod softmax;
pub mod split_cols;
pub mod swiglu; // was activation
pub mod upsample;

// --- extra wrappers sharing a dir (distinct module names) ---
#[path = "cast_fill/cast_dtype.rs"]
pub mod cast_dtype;
#[path = "cast_fill/pad.rs"]
pub mod pad;
#[path = "matmul/sdpa.rs"]
pub mod sdpa;

// --- no-cu (pure Rust / cudnn) modules, kept flat ---
pub mod concat_seq;
pub mod conv2d;
pub mod fused_qk_norm_rope; // note: not yet wired into a caller
