//! Z-Image Transformer 2D Model (S3-DiT denoising backbone).
//!
//! This is the core DiT that predicts noise/velocity for each denoising step.
//!
//! # Architecture (30 layers, dim=3840)
//!
//! ```text
//! Inputs: image patches [B, N, C·pH·pW], caption embeddings [B, M, cap_dim]
//!
//! ┌─ TimestepEmbedder ──→ adaln_input [B, 256] ──────────────────────┐
//! │                                                                   │
//! ├─ x_embedder(patches) ──→ noise_refiner (2L, AdaLN) ──→ image_tok │
//! ├─ cap_embedder(caption) → context_refiner (2L, no mod) → cap_tok  │
//! │                                                                   │
//! ├─ UnifiedPrepare: concat(image_tok, cap_tok) ──→ unified          │
//! │                                                                   │
//! ├─ 30× ZImageTransformerBlock (AdaLN + RoPE + SwiGLU FFN)         │
//! │                                                                   │
//! └─ FinalLayer ──→ Unpatchify ──→ predicted velocity [B, C, F, H, W]│
//! ```
//!
//! ## Key features
//! - **Single-stream**: image + caption tokens processed jointly (not dual-stream)
//! - **3D RoPE**: frame × height × width positional encoding
//! - **AdaLN modulation**: timestep-conditioned scale/gate via `tanh` gating
//! - **SwiGLU FFN**: `w1·σ(w1x) ⊙ w3x → w2` with hidden_dim = dim/3*8

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

/// DiT transformer configuration.
///
/// Parsed from `transformer/config.json` in the model directory.
#[derive(Debug, Clone)]
pub struct ZImageTransformerConfig {
    pub dim: usize,                    // 3840
    pub n_layers: usize,               // 30
    pub n_refiner_layers: usize,       // 2
    pub n_heads: usize,                // 30
    pub n_kv_heads: usize,             // 30
    pub in_channels: usize,            // 16 (VAE latent channels)
    pub cap_feat_dim: usize,           // 2560 (Qwen3 hidden_size)
    pub all_patch_size: Vec<usize>,    // [2]
    pub all_f_patch_size: Vec<usize>,  // [1]
    pub axes_dims: Vec<usize>,         // [32, 48, 48]  RoPE per-axis dims
    pub axes_lens: Vec<usize>,         // [1536, 512, 512] RoPE per-axis max lens
    pub norm_eps: f32,                 // 1e-5
    pub rope_theta: f32,               // 256.0
    pub t_scale: f32,                  // 1000.0
    pub qk_norm: bool,                 // true
}

/// Z-Image DiT denoising model.
///
/// Holds all layer weights. Constructed by [`ZImagePipeline::from_pretrained`].
pub struct ZImageTransformer2DModel {
    pub config: ZImageTransformerConfig,
    device: DeviceType,
    // TODO: populate layer weights
    //
    // pub t_embedder: TimestepEmbedder,
    // pub x_embedder: Matmul,
    // pub cap_embedder: (RMSNorm, Matmul),
    // pub noise_refiner: Vec<DiTBlock>,      // n_refiner_layers
    // pub context_refiner: Vec<DiTBlock>,     // n_refiner_layers
    // pub layers: Vec<DiTBlock>,             // n_layers
    // pub final_layer: FinalLayer,
    // pub rope_embedder: RopeEmbedder3D,
    // pub x_pad_token: Tensor,
    // pub cap_pad_token: Tensor,
}

impl ZImageTransformer2DModel {
    /// Forward pass: predict velocity for one denoising step.
    ///
    /// # Arguments
    /// - `x`: latent image tensors, each `[C, 1, H, W]`
    /// - `timestep`: normalized timestep tensor `[B]`
    /// - `cap_feats`: caption embeddings, each `[seq_len, cap_feat_dim]`
    ///
    /// # Returns
    /// Predicted velocity tensors, each `[C, 1, H, W]`
    pub fn forward(
        &self,
        _x: &[Tensor],
        _timestep: &Tensor,
        _cap_feats: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        todo!("ZImageTransformer2DModel::forward — implement with existing Op primitives")
    }
}
