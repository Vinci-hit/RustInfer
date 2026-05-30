//! Pre-allocated runtime state for the Z-Image diffusion pipeline.
//!
//! [`DitState`] holds every per-call tensor the transformer forward pass
//! needs (timestep embedding, padded image / caption streams, RoPE caches,
//! per-block scratch). [`PipelineState`] holds the latent ping-pong pair
//! and noise-pred slot used by the denoise loop.
//!
//! Allocate-once-and-reuse so the hot path is alloc-free.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Dtype;

use super::dit_block::DiTBlockScratch;

pub const LATENT_CHANNELS: usize = 16;
pub const VAE_SCALE_FACTOR: usize = 8;
pub const SEQ_MULTI_OF: usize = 128;
pub const ADALN_EMBED_DIM: usize = 256;
pub const N_MAX_STEPS: usize = 50;
pub const T_FREQ_DIM: usize = 256;
pub const T_EMBEDDER_MID: usize = 1024;

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

/// Shape spec describing all transformer-internal buffer sizes.
#[derive(Debug, Clone, Copy)]
pub struct DitShapeSpec {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub cap_feat_dim: usize,
    pub patch_size: usize,
    pub f_patch_size: usize,
    pub patch_in_dim: usize,
    pub final_out_dim: usize,
    pub capacity: ZImageCapacity,
}

impl DitShapeSpec {
    /// Image patch token count at the maximum supported resolution.
    pub fn n_patches_max(&self) -> usize {
        let f_t = 1 / self.f_patch_size.max(1);
        let h_t = self.capacity.max_latent_h() / self.patch_size;
        let w_t = self.capacity.max_latent_w() / self.patch_size;
        f_t.max(1) * h_t * w_t
    }
    /// Image sequence length padded to `SEQ_MULTI_OF`.
    pub fn s_img_max(&self) -> usize {
        round_up(self.n_patches_max(), SEQ_MULTI_OF)
    }
    /// Padded caption sequence length.
    pub fn s_cap_max(&self) -> usize {
        round_up(self.capacity.max_cap_len, SEQ_MULTI_OF)
    }
    /// Unified (image | caption) sequence length used by the main DiT layers.
    pub fn s_total_max(&self) -> usize {
        self.s_img_max() + self.s_cap_max()
    }
}

#[inline]
fn round_up(n: usize, multiple: usize) -> usize {
    debug_assert!(multiple > 0);
    n.div_ceil(multiple) * multiple
}

/// Pipeline-level buffers (denoise I/O only).
pub struct PipelineState<T: Dtype, D: OpBackend> {
    /// `[1, 16, max_lh, max_lw]` working latent.
    pub latents: Tensor<T, D>,
    /// `[1, 16, max_lh, max_lw]` ping-pong partner.
    pub latents_tmp: Tensor<T, D>,
    /// `[1, 16, max_lh, max_lw]` DiT velocity output (after sign flip).
    pub noise_pred: Tensor<T, D>,
    /// `[16, 1, max_lh, max_lw]` 5D view of `latents` consumed by transformer.
    pub latent_5d: Tensor<T, D>,
    pub capacity: ZImageCapacity,
}

impl<T: Dtype, D: OpBackend> PipelineState<T, D> {
    pub fn new(capacity: ZImageCapacity, dev: &D) -> OpResult<Self> {
        let lh = capacity.max_latent_h();
        let lw = capacity.max_latent_w();
        Ok(Self {
            latents: Tensor::zeros([1, LATENT_CHANNELS, lh, lw], dev)?,
            latents_tmp: Tensor::zeros([1, LATENT_CHANNELS, lh, lw], dev)?,
            noise_pred: Tensor::zeros([1, LATENT_CHANNELS, lh, lw], dev)?,
            latent_5d: Tensor::zeros([LATENT_CHANNELS, 1, lh, lw], dev)?,
            capacity,
        })
    }
}

/// All pre-allocated tensors used during a single transformer forward pass.
pub struct DitState<T: Dtype, D: OpBackend> {
    pub spec: DitShapeSpec,

    // Timestep embedding chain.
    pub t_freq: Tensor<T, D>,        // [1, T_FREQ_DIM]
    pub t_hidden: Tensor<T, D>,      // [1, T_EMBEDDER_MID]
    pub t_out: Tensor<T, D>,         // [1, ADALN_EMBED_DIM]
    pub adaln_input: Tensor<T, D>,   // [1, ADALN_EMBED_DIM]

    // Patch embedder.
    pub patches: Tensor<T, D>,       // [n_patches_max, patch_in_dim]
    pub x_emb: Tensor<T, D>,         // [n_patches_max, dim]
    pub x_padded: Tensor<T, D>,      // [s_img_max, dim]
    pub x_padded_tmp: Tensor<T, D>,  // [s_img_max, dim]

    // Caption embedder.
    pub cap_feats_padded: Tensor<T, D>, // [s_cap_max, cap_feat_dim]
    pub cap_normed: Tensor<T, D>,    // [s_cap_max, cap_feat_dim]
    pub cap_emb: Tensor<T, D>,       // [s_cap_max, dim]
    pub cap_padded: Tensor<T, D>,    // [s_cap_max, dim]
    pub cap_padded_tmp: Tensor<T, D>,// [s_cap_max, dim]

    // RoPE caches (F32 device).
    pub x_cos: Tensor<f32, D>,       // [s_img_max, head_dim/2]
    pub x_sin: Tensor<f32, D>,
    pub cap_cos: Tensor<f32, D>,     // [s_cap_max, head_dim/2]
    pub cap_sin: Tensor<f32, D>,
    pub unified_cos: Tensor<f32, D>, // [s_total_max, head_dim/2]
    pub unified_sin: Tensor<f32, D>,

    // Unified main stream.
    pub unified: Tensor<T, D>,       // [s_total_max, dim]
    pub unified_tmp: Tensor<T, D>,   // [s_total_max, dim]

    // Final layer.
    pub final_normed: Tensor<T, D>,  // [s_img_max, dim]
    pub final_scale: Tensor<T, D>,   // [1, dim]
    pub final_out: Tensor<T, D>,     // [s_img_max, final_out_dim]
    pub image_out: Tensor<T, D>,     // [LATENT_CHANNELS, 1, lh, lw]

    /// Per-block scratch shared across all blocks. Block forwards overwrite
    /// these every call; cross-block state flows through `x_padded` /
    /// `cap_padded` / `unified` (above).
    pub block_scratch: DiTBlockScratch<T, D>,
}

impl<T: Dtype, D: OpBackend> DitState<T, D> {
    /// Allocate every transformer-internal buffer up to `spec`'s capacity.
    pub fn new(spec: DitShapeSpec, dev: &D) -> OpResult<Self> {
        let dim = spec.dim;
        let head_dim = spec.head_dim;
        let half_d = head_dim / 2;
        let cap_feat_dim = spec.cap_feat_dim;
        let patch_in_dim = spec.patch_in_dim;
        let final_out_dim = spec.final_out_dim;
        let lh = spec.capacity.max_latent_h();
        let lw = spec.capacity.max_latent_w();

        let s_img = spec.s_img_max();
        let s_cap = spec.s_cap_max();
        let s_tot = spec.s_total_max();
        let n_patches = spec.n_patches_max();

        Ok(Self {
            spec,

            t_freq: Tensor::zeros([1, T_FREQ_DIM], dev)?,
            t_hidden: Tensor::zeros([1, T_EMBEDDER_MID], dev)?,
            t_out: Tensor::zeros([1, ADALN_EMBED_DIM], dev)?,
            adaln_input: Tensor::zeros([1, ADALN_EMBED_DIM], dev)?,

            patches: Tensor::zeros([n_patches, patch_in_dim], dev)?,
            x_emb: Tensor::zeros([n_patches, dim], dev)?,
            x_padded: Tensor::zeros([s_img, dim], dev)?,
            x_padded_tmp: Tensor::zeros([s_img, dim], dev)?,

            cap_feats_padded: Tensor::zeros([s_cap, cap_feat_dim], dev)?,
            cap_normed: Tensor::zeros([s_cap, cap_feat_dim], dev)?,
            cap_emb: Tensor::zeros([s_cap, dim], dev)?,
            cap_padded: Tensor::zeros([s_cap, dim], dev)?,
            cap_padded_tmp: Tensor::zeros([s_cap, dim], dev)?,

            x_cos: Tensor::zeros([s_img, half_d], dev)?,
            x_sin: Tensor::zeros([s_img, half_d], dev)?,
            cap_cos: Tensor::zeros([s_cap, half_d], dev)?,
            cap_sin: Tensor::zeros([s_cap, half_d], dev)?,
            unified_cos: Tensor::zeros([s_tot, half_d], dev)?,
            unified_sin: Tensor::zeros([s_tot, half_d], dev)?,

            unified: Tensor::zeros([s_tot, dim], dev)?,
            unified_tmp: Tensor::zeros([s_tot, dim], dev)?,

            final_normed: Tensor::zeros([s_img, dim], dev)?,
            final_scale: Tensor::zeros([1, dim], dev)?,
            final_out: Tensor::zeros([s_img, final_out_dim], dev)?,
            image_out: Tensor::zeros([LATENT_CHANNELS, 1, lh, lw], dev)?,

            block_scratch: DiTBlockScratch::new(dim, spec.hidden_dim, s_tot, dev)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cuda::Cuda;

    fn z_image_spec() -> DitShapeSpec {
        DitShapeSpec {
            dim: 3840,
            n_heads: 30,
            head_dim: 128,
            hidden_dim: 10240,
            cap_feat_dim: 2560,
            patch_size: 2,
            f_patch_size: 1,
            patch_in_dim: 1 * 2 * 2 * 16,        // 64
            final_out_dim: 2 * 2 * 1 * 16,       // 64
            capacity: ZImageCapacity { max_height: 256, max_width: 256, max_cap_len: 64 },
        }
    }

    #[test]
    fn shape_spec_seq_lengths() {
        let spec = z_image_spec();
        // 256x256 latent = 32x32, /2 = 16x16 patches = 256 tokens, padded to 256 (already divides).
        assert_eq!(spec.n_patches_max(), 16 * 16);
        assert_eq!(spec.s_img_max(), 256);
        assert_eq!(spec.s_cap_max(), 128); // round_up(64, 128) = 128
        assert_eq!(spec.s_total_max(), 384);
    }

    #[test]
    fn pipeline_state_allocates_correct_shapes() {
        let cuda = Cuda::new(0).unwrap();
        let cap = ZImageCapacity { max_height: 512, max_width: 512, max_cap_len: 64 };
        let ps: PipelineState<half::bf16, Cuda> = PipelineState::new(cap, &cuda).unwrap();
        assert_eq!(ps.latents.shape().as_slice(), &[1, 16, 64, 64]);
        assert_eq!(ps.latents_tmp.shape().as_slice(), &[1, 16, 64, 64]);
        assert_eq!(ps.noise_pred.shape().as_slice(), &[1, 16, 64, 64]);
        assert_eq!(ps.latent_5d.shape().as_slice(), &[16, 1, 64, 64]);
    }

    #[test]
    fn dit_state_allocates_without_panic() {
        // Use a tiny spec to avoid huge allocations.
        let cuda = Cuda::new(0).unwrap();
        let spec = DitShapeSpec {
            dim: 64,
            n_heads: 4,
            head_dim: 16,
            hidden_dim: 128,
            cap_feat_dim: 32,
            patch_size: 2,
            f_patch_size: 1,
            patch_in_dim: 1 * 2 * 2 * 16,
            final_out_dim: 2 * 2 * 16,
            capacity: ZImageCapacity { max_height: 64, max_width: 64, max_cap_len: 16 },
        };
        let state: DitState<f32, Cuda> = DitState::new(spec, &cuda).unwrap();
        // Sanity: a few key shapes.
        assert_eq!(state.t_freq.shape().as_slice(), &[1, 256]);
        assert_eq!(state.t_out.shape().as_slice(), &[1, 256]);
        // s_img_max = round_up(4*4, 128) = 128.
        assert_eq!(state.x_padded.shape().as_slice(), &[128, 64]);
        // s_cap_max = round_up(16, 128) = 128.
        assert_eq!(state.cap_padded.shape().as_slice(), &[128, 64]);
        // s_total_max = 256.
        assert_eq!(state.unified.shape().as_slice(), &[256, 64]);
    }
}
