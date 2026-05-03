//! Tensor operators — CPU + CUDA implementations, stride-aware.
//!
//! Everything in `op::*` dispatches internally on [`DeviceType`] and falls
//! through to either `kernels::cpu::*` (always compiled) or
//! `kernels::cuda::*` (gated on the `cuda` feature). Every FFI path forms
//! raw pointers through [`TypedTensor::data_ptr`] /
//! [`TypedTensor::data_ptr_mut`], so operators accept strided / offset
//! views (e.g. from [`Tensor::narrow`], [`Tensor::select`]) without
//! requiring the caller to materialise.
//!
//! # API conventions
//!
//! - **Free functions** (`add`, `swiglu`, `softmax`, …) for algorithms
//!   with no cached state.
//! - **Structs** (`Matmul`, `RMSNorm`, `RoPEOp`, `Embedding`,
//!   `FlashAttnGQA`, `SamplerOp`) for operators that cache device
//!   handles, sin/cos tables, or other precomputed resources.
//! - Most kernels come in pairs: an allocating convenience variant
//!   (`foo(...) -> Result<Tensor>`) and a hot-path `*_into` variant that
//!   writes into a caller-supplied `dst`. Prefer `*_into` inside the
//!   forward loop.
//!
//! # Module index
//!
//! ## Element-wise arithmetic
//! - [`add`]           — `add`, `add_inplace` (tensor + tensor)
//! - [`scalar`]        — `scalar_mul/add(_inplace)`, `scalar_mul_inplace_from_dev`
//! - [`ewise_mul`]     — `ewise_mul(_inplace)` (tensor × tensor)
//! - [`broadcast_mul`] — `broadcast_mul(_inplace)` (`[.., D] * [D]`)
//!
//! ## Activations & pointwise non-linearities
//! - [`activation`] — in-place `silu_inplace`, `tanh_inplace`
//! - [`swiglu`]     — in-place `x = silu(x) * y`
//!
//! ## Normalisation
//! - [`layernorm`]          — LayerNorm
//! - [`rmsnorm`]            — RMSNorm (uses a small stateful struct)
//! - [`fused_add_rmsnorm`]  — fused `y = rmsnorm(x + residual)`
//! - [`groupnorm`]          — GroupNorm + optional fused `groupnorm_silu`
//! - [`softmax`]            — row-wise softmax along the last dim
//!
//! ## Linear algebra
//! - [`matmul`] — dense matmul (supports quantised weights via
//!   `QuantParams`); struct-based entry point `Matmul`
//! - [`conv2d`] — 2D convolution and `conv2d_output_size` helper
//!
//! ## Attention
//! - [`sdpa`]       — `scaled_dot_product_attention`, `dit_sdpa`
//! - [`flash_gqa`]  — grouped-query flash attention (prefill + decode)
//!
//! ## Positional encoding
//! - [`rope`]              — standard RoPE (`RoPEOp` struct with sin/cos cache)
//! - [`rope_interleaved`]  — GPT-J–style interleaved RoPE, in-place
//! - [`timestep_embed`]    — CUDA-only `sinusoid_embedding_from_dev` for diffusion pipelines
//!
//! ## Layout manipulation (allocating helpers around [`crate::tensor`] views)
//! - [`cast`]        — `cast_dtype`, `cast_dtype_into`
//! - [`concat`](mod@concat) — `concat_seq`, `concat_seq_into` (dim-0 concat of 2D tensors)
//! - [`pad`]         — `pad_last_row(_into)`, `pad_with_token(_into)`,
//!                     `overwrite_pad_tokens_inplace`
//! - [`split_cols`]  — split a `[.., D]` tensor into equal column chunks
//! - [`upsample`]    — nearest-neighbour 2× upsample
//!
//! ## Sequence / KV utilities
//! - [`embedding`]   — `Embedding` table lookup (struct holds the table)
//! - [`scatter`]     — `scatter`, `scatter_kv`, `scatter_kv_batch` (KV-cache writes)
//! - [`encode`]      — `EncodeLayer` trait for model-specific tokenise/encode
//! - [`sampler`]     — `Sampler` trait + `ArgmaxSampler` + `SamplerOp` wrapper
//!
//! ## Kernel backends (internal)
//! - [`kernels`] — raw CPU + CUDA kernel implementations that `op::*`
//!   dispatches to. Callers normally shouldn't reach in here.
//!
//! [`DeviceType`]: crate::base::DeviceType
//! [`TypedTensor::data_ptr`]:     crate::tensor::TypedTensor::data_ptr
//! [`TypedTensor::data_ptr_mut`]: crate::tensor::TypedTensor::data_ptr_mut
//! [`Tensor::narrow`]:  crate::tensor::Tensor::narrow
//! [`Tensor::select`]:  crate::tensor::Tensor::select

pub mod kernels;

// element-wise arithmetic
pub mod add;
pub mod scalar;
pub mod ewise_mul;
pub mod broadcast_mul;

// activations / pointwise
pub mod activation;
pub mod swiglu;

// normalisation
pub mod layernorm;
pub mod rmsnorm;
pub mod fused_add_rmsnorm;
pub mod groupnorm;
pub mod softmax;

// linear algebra
pub mod matmul;
pub mod conv2d;

// attention
pub mod sdpa;
pub mod flash_gqa;

// positional encoding
pub mod rope;
pub mod rope_interleaved;
pub mod timestep_embed;

// layout manipulation
pub mod cast;
pub mod concat;
pub mod pad;
pub mod split_cols;
pub mod upsample;

// sequence / KV utilities
pub mod embedding;
pub mod scatter;
pub mod encode;
pub mod sampler;
