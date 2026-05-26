//! Pre-allocated runtime state for diffusion inference.
//!
//! All tensors allocated once at pipeline construction; forward passes
//! borrow sub-views. This makes the hot path allocation-free and
//! CUDA Graph capture-ready.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;

/// Maximum output image resolution constants.
pub const LATENT_CHANNELS: usize = 16;
pub const VAE_SCALE_FACTOR: usize = 8;
pub const SEQ_MULTI_OF: usize = 128;
pub const ADALN_EMBED_DIM: usize = 256;
pub const N_MAX_STEPS: usize = 50;

/// Runtime capacity bounds for the pipeline.
#[derive(Debug, Clone, Copy)]
pub struct ZImageCapacity {
    pub max_height: usize,
    pub max_width: usize,
    pub max_cap_len: usize,
}

impl Default for ZImageCapacity {
    fn default() -> Self {
        Self { max_height: 1024, max_width: 1024, max_cap_len: 512 }
    }
}

impl ZImageCapacity {
    pub fn max_latent_h(&self) -> usize { self.max_height / VAE_SCALE_FACTOR }
    pub fn max_latent_w(&self) -> usize { self.max_width / VAE_SCALE_FACTOR }
}

/// Shape spec needed to size all DitState buffers.
#[derive(Debug, Clone)]
pub struct DitShapeSpec {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,   // = max_img_tokens + max_cap_len (padded to SEQ_MULTI_OF)
    pub cap_feat_dim: usize,
}

/// Pre-allocated workspace for the DiT transformer.
///
/// Each field is a tensor allocated to maximum capacity.
/// Forward passes slice into these buffers.
pub struct DitState<T: Dtype, D: OpBackend> {
    // ─── Per-block scratch (reused each layer) ───
    pub blk_q: Tensor<T, D>,         // [max_seq, q_dim]
    pub blk_k: Tensor<T, D>,         // [max_seq, kv_dim]
    pub blk_v: Tensor<T, D>,         // [max_seq, kv_dim]
    pub blk_attn_out: Tensor<T, D>,  // [max_seq, dim]
    pub blk_gate: Tensor<T, D>,      // [max_seq, intermediate]
    pub blk_up: Tensor<T, D>,        // [max_seq, intermediate]
    pub blk_ffn: Tensor<T, D>,       // [max_seq, dim]

    // ─── Ping-pong hidden states ───
    pub x_padded: Tensor<T, D>,      // [max_seq, dim]
    pub x_padded_tmp: Tensor<T, D>,  // [max_seq, dim]

    // ─── Timestep embedding ───
    pub t_embed: Tensor<T, D>,       // [1, ADALN_EMBED_DIM]
}

impl<T: Dtype, D: OpBackend> DitState<T, D> {
    /// Allocate all buffers according to the shape spec.
    pub fn new(spec: &DitShapeSpec, device: &D) -> OpResult<Self> {
        let q_dim = spec.n_heads * spec.head_dim;
        let kv_dim = spec.n_kv_heads * spec.head_dim;
        let s = spec.max_seq_len;

        Ok(Self {
            blk_q: D::alloc_tensor(Shape::from_slice(&[s, q_dim]), device)?,
            blk_k: D::alloc_tensor(Shape::from_slice(&[s, kv_dim]), device)?,
            blk_v: D::alloc_tensor(Shape::from_slice(&[s, kv_dim]), device)?,
            blk_attn_out: D::alloc_tensor(Shape::from_slice(&[s, spec.dim]), device)?,
            blk_gate: D::alloc_tensor(Shape::from_slice(&[s, spec.intermediate_size]), device)?,
            blk_up: D::alloc_tensor(Shape::from_slice(&[s, spec.intermediate_size]), device)?,
            blk_ffn: D::alloc_tensor(Shape::from_slice(&[s, spec.dim]), device)?,
            x_padded: D::alloc_tensor(Shape::from_slice(&[s, spec.dim]), device)?,
            x_padded_tmp: D::alloc_tensor(Shape::from_slice(&[s, spec.dim]), device)?,
            t_embed: D::alloc_tensor(Shape::from_slice(&[1, ADALN_EMBED_DIM]), device)?,
        })
    }
}

/// Pipeline-level I/O state for the denoise loop.
pub struct PipelineState<T: Dtype, D: OpBackend> {
    /// Working latent [1, LATENT_CHANNELS, latent_h, latent_w]
    pub latents: Tensor<T, D>,
    /// Secondary latent for ping-pong
    pub latents_tmp: Tensor<T, D>,
    /// DiT output per step [1, LATENT_CHANNELS, latent_h, latent_w]
    pub noise_pred: Tensor<T, D>,
}

impl<T: Dtype, D: OpBackend> PipelineState<T, D> {
    pub fn new(cap: &ZImageCapacity, device: &D) -> OpResult<Self> {
        let lh = cap.max_latent_h();
        let lw = cap.max_latent_w();
        let shape = Shape::from_slice(&[1, LATENT_CHANNELS, lh, lw]);
        Ok(Self {
            latents: D::alloc_tensor(shape, device)?,
            latents_tmp: D::alloc_tensor(shape, device)?,
            noise_pred: D::alloc_tensor(shape, device)?,
        })
    }
}
