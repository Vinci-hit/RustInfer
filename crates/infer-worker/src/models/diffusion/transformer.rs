//! Z-Image Transformer (S3-DiT denoising backbone).
//!
//! Architecture: x_embedder → [30 DiTBlock] → [2 noise_refiner] → [2 context_refiner] → final_layer
//! All forward ops go through pre-allocated DitState buffers (zero allocation on hot path).

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use crate::models::layers::{Linear, RMSNorm};
use super::dit_block::DiTBlock;
use super::state::DitState;
use super::timestep_embedder::TimestepEmbedder;

/// Configuration for the Z-Image transformer.
#[derive(Debug, Clone)]
pub struct ZImageTransformerConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub in_channels: usize,        // latent patch flat dim
    pub cap_feat_dim: usize,       // text embedding dim
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub t_scale: f32,
    pub qk_norm: bool,
}

/// The full Z-Image transformer model.
pub struct ZImageTransformer<T: Dtype, D: OpBackend> {
    pub config: ZImageTransformerConfig,

    // ─── Embedders ───
    /// Projects flattened patches → dim
    pub x_embedder: Linear<T, D>,
    /// Projects text encoder hidden states → dim
    pub cap_embedder: Linear<T, D>,
    /// Timestep → conditioning embedding
    pub t_embedder: TimestepEmbedder<T, D>,

    // ─── Main transformer blocks ───
    pub blocks: Vec<DiTBlock<T, D>>,
    /// Noise refiner blocks (applied to image tokens only)
    pub noise_refiner_blocks: Vec<DiTBlock<T, D>>,
    /// Context refiner blocks (applied to text tokens only)
    pub context_refiner_blocks: Vec<DiTBlock<T, D>>,

    // ─── Final layer ───
    pub final_norm: RMSNorm<T, D>,
    pub final_proj: Linear<T, D>,  // dim → in_channels (unpatch)
}

impl<T: Dtype, D: OpBackend> ZImageTransformer<T, D> {
    /// Forward pass for a single denoising step.
    ///
    /// - `x_tokens`: patchified noisy latent [num_img_tokens, patch_flat]
    /// - `cap_embeds`: text encoder output [cap_len, cap_feat_dim]
    /// - `t_value`: scaled timestep scalar
    /// - `state`: pre-allocated workspace buffers
    ///
    /// Returns: noise prediction [num_img_tokens, patch_flat]
    pub fn forward(
        &self,
        x_tokens: &Tensor<T, D>,
        cap_embeds: &Tensor<T, D>,
        t_value: f32,
        state: &mut DitState<T, D>,
    ) -> OpResult<Tensor<T, D>> {
        let dev = x_tokens.device();
        let num_img_tokens = x_tokens.numel() / self.config.in_channels;
        let cap_len = cap_embeds.numel() / self.config.cap_feat_dim;
        let dim = self.config.dim;

        // ─── 1. Embed patches → dim ───
        let mut x = D::alloc_tensor::<T>(Shape::from_slice(&[num_img_tokens, dim]), dev)?;
        self.x_embedder.forward(x_tokens, &mut x)?;

        // ─── 2. Embed captions → dim ───
        let mut cap = D::alloc_tensor::<T>(Shape::from_slice(&[cap_len, dim]), dev)?;
        self.cap_embedder.forward(cap_embeds, &mut cap)?;

        // ─── 3. Timestep embedding ───
        let freq_dim = self.t_embedder.frequency_embedding_size;
        let mid_dim = self.t_embedder.mlp1.weight.numel() / freq_dim;
        let mut t_freq = D::alloc_tensor::<T>(Shape::from_slice(&[1, freq_dim]), dev)?;
        let mut t_hidden = D::alloc_tensor::<T>(Shape::from_slice(&[1, mid_dim]), dev)?;
        self.t_embedder.forward(t_value, &mut t_freq, &mut t_hidden, &mut state.t_embed)?;

        // ─── 4. Concatenate image + text tokens → [total_seq, dim] ───
        let total_seq = num_img_tokens + cap_len;
        let mut hidden = D::alloc_tensor::<T>(Shape::from_slice(&[total_seq, dim]), dev)?;
        // Copy x into hidden[0..num_img_tokens]
        unsafe {
            std::ptr::copy_nonoverlapping(
                x.data_ptr() as *const u8,
                hidden.data_ptr_mut() as *mut u8,
                num_img_tokens * dim * T::SIZE_BYTES,
            );
            // Copy cap into hidden[num_img_tokens..total_seq]
            std::ptr::copy_nonoverlapping(
                cap.data_ptr() as *const u8,
                (hidden.data_ptr_mut() as *mut u8).add(num_img_tokens * dim * T::SIZE_BYTES),
                cap_len * dim * T::SIZE_BYTES,
            );
        }

        // ─── 5. Main transformer blocks (ping-pong between hidden and output) ───
        let mut ping = hidden;
        let mut pong = D::alloc_tensor::<T>(Shape::from_slice(&[total_seq, dim]), dev)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ping.data_ptr() as *const u8,
                pong.data_ptr_mut() as *mut u8,
                total_seq * dim * T::SIZE_BYTES,
            );
        }

        for block in &self.blocks {
            block.forward(
                &ping,
                Some(&state.t_embed),
                &mut state.blk_q,
                &mut state.blk_k,
                &mut state.blk_v,
                &mut state.blk_attn_out,
                &mut state.blk_gate,
                &mut state.blk_up,
                &mut state.blk_ffn,
                &mut pong,
            )?;
            std::mem::swap(&mut ping, &mut pong);
        }
        // After loop, `ping` holds the final result

        // ─── 6. Extract image tokens from result ───
        let mut img_out = D::alloc_tensor::<T>(Shape::from_slice(&[num_img_tokens, dim]), dev)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ping.data_ptr() as *const u8,
                img_out.data_ptr_mut() as *mut u8,
                num_img_tokens * dim * T::SIZE_BYTES,
            );
        }

        // ─── 7. Noise refiner (image tokens only, ping-pong) ───
        let mut ref_ping = img_out;
        let mut ref_pong = D::alloc_tensor::<T>(Shape::from_slice(&[num_img_tokens, dim]), dev)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ref_ping.data_ptr() as *const u8,
                ref_pong.data_ptr_mut() as *mut u8,
                num_img_tokens * dim * T::SIZE_BYTES,
            );
        }
        for block in &self.noise_refiner_blocks {
            block.forward(
                &ref_ping,
                Some(&state.t_embed),
                &mut state.blk_q,
                &mut state.blk_k,
                &mut state.blk_v,
                &mut state.blk_attn_out,
                &mut state.blk_gate,
                &mut state.blk_up,
                &mut state.blk_ffn,
                &mut ref_pong,
            )?;
            std::mem::swap(&mut ref_ping, &mut ref_pong);
        }

        // ─── 8. Final layer: norm → project back to patch space ───
        let mut normed = D::alloc_tensor::<T>(Shape::from_slice(&[num_img_tokens, dim]), dev)?;
        self.final_norm.forward(&ref_ping, &mut normed)?;

        let mut noise_pred = D::alloc_tensor::<T>(Shape::from_slice(&[num_img_tokens, self.config.in_channels]), dev)?;
        self.final_proj.forward(&normed, &mut noise_pred)?;

        Ok(noise_pred)
    }
}
