//! VAE Decoder for Z-Image — converts latent [1, 16, H, W] → image [1, 3, 8H, 8W].
//!
//! Architecture: 3-stage upsampling with ResNet blocks + GroupNorm + self-attention.
//! Each stage: Upsample2x → Conv → GroupNorm+SiLU → Conv → GroupNorm+SiLU + residual.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use crate::models::layers::{Conv2D, GroupNorm, Linear};

// ─── Building blocks ─────────────────────────────────────────────────────────

/// ResNet block: two conv layers with GroupNorm+SiLU and a residual skip.
pub struct ResBlock<T: Dtype, D: OpBackend> {
    pub norm1: GroupNorm<T, D>,
    pub conv1: Conv2D<T, D>,
    pub norm2: GroupNorm<T, D>,
    pub conv2: Conv2D<T, D>,
    /// Optional 1×1 conv for channel mismatch in residual path.
    pub skip_conv: Option<Conv2D<T, D>>,
}

impl<T: Dtype, D: OpBackend> ResBlock<T, D> {
    pub fn forward(
        &self,
        x: &Tensor<T, D>,
        buf1: &mut Tensor<T, D>,
        buf2: &mut Tensor<T, D>,
    ) -> OpResult<()> {
        // norm1 + silu + conv1
        self.norm1.forward_silu(x, buf1)?;
        self.conv1.forward(buf1, buf2)?;
        // norm2 + silu + conv2
        self.norm2.forward_silu(buf2, buf1)?;
        self.conv2.forward(buf1, buf2)?;
        // residual: buf2 += skip(x)
        if let Some(ref skip) = self.skip_conv {
            let mut skip_out = buf1.clone(); // reuse buf1
            skip.forward(x, &mut skip_out)?;
            D::add_inplace(buf2, &skip_out)?;
        } else {
            D::add_inplace(buf2, x)?;
        }
        Ok(())
    }
}

/// Self-attention block for VAE mid-layer.
pub struct VaeAttnBlock<T: Dtype, D: OpBackend> {
    pub norm: GroupNorm<T, D>,
    pub q_proj: Linear<T, D>,
    pub k_proj: Linear<T, D>,
    pub v_proj: Linear<T, D>,
    pub o_proj: Linear<T, D>,
    pub num_heads: usize,
    pub head_dim: usize,
}

/// Single upsampling stage: upsample → conv → resblocks.
pub struct UpsampleStage<T: Dtype, D: OpBackend> {
    pub upsample_conv: Conv2D<T, D>,  // 3×3 conv after nearest upsample
    pub res_blocks: Vec<ResBlock<T, D>>,
}

// ─── VAE Decoder ─────────────────────────────────────────────────────────────

/// Configuration for the VAE decoder.
#[derive(Debug, Clone)]
pub struct VaeDecoderConfig {
    pub latent_channels: usize,     // 16
    pub out_channels: usize,        // 3 (RGB)
    pub block_out_channels: Vec<usize>,  // e.g. [512, 512, 256, 128]
    pub num_res_blocks: usize,      // 2
    pub num_groups: usize,          // 32
    pub norm_eps: f32,              // 1e-6
}

impl Default for VaeDecoderConfig {
    fn default() -> Self {
        Self {
            latent_channels: 16,
            out_channels: 3,
            block_out_channels: vec![512, 512, 256, 128],
            num_res_blocks: 2,
            num_groups: 32,
            norm_eps: 1e-6,
        }
    }
}

/// Full VAE decoder.
pub struct VaeDecoder<T: Dtype, D: OpBackend> {
    /// Initial conv: latent_channels → block_out_channels[0]
    pub conv_in: Conv2D<T, D>,
    /// Mid block: resblock + attn + resblock
    pub mid_block: (ResBlock<T, D>, Option<VaeAttnBlock<T, D>>, ResBlock<T, D>),
    /// Upsampling stages (one per channel transition)
    pub up_stages: Vec<UpsampleStage<T, D>>,
    /// Final norm + conv_out: last_channels → out_channels
    pub norm_out: GroupNorm<T, D>,
    pub conv_out: Conv2D<T, D>,
    pub config: VaeDecoderConfig,
}

impl<T: Dtype, D: OpBackend> VaeDecoder<T, D> {
    /// Forward: latent [1, latent_ch, H, W] → image [1, out_ch, 8H, 8W]
    pub fn forward(&self, latent: &Tensor<T, D>, dev: &D) -> OpResult<Tensor<T, D>> {
        let shape = latent.shape().as_slice();
        let (n, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // conv_in
        let ch0 = self.config.block_out_channels[0];
        let mut x = D::alloc_tensor::<T>(Shape::from_slice(&[n, ch0, h, w]), dev)?;
        self.conv_in.forward(latent, &mut x)?;

        // mid block: resblock → attention → resblock
        let mut buf1 = D::alloc_tensor::<T>(x.shape().clone(), dev)?;
        let mut buf2 = D::alloc_tensor::<T>(x.shape().clone(), dev)?;
        self.mid_block.0.forward(&x, &mut buf1, &mut buf2)?;
        std::mem::swap(&mut x, &mut buf2);
        // Self-attention in mid block (if present)
        if let Some(ref attn) = self.mid_block.1 {
            let _shape = x.shape().as_slice();
            let mut normed = D::alloc_tensor::<T>(x.shape().clone(), dev)?;
            attn.norm.forward(&x, &mut normed)?;
            // Simplified: apply attention as identity + residual
            // Full impl: reshape [N,C,H,W]→[H*W,C], QKV proj, SDPA, O proj, reshape back
            D::add_inplace(&mut x, &normed)?;
        }
        self.mid_block.2.forward(&x, &mut buf1, &mut buf2)?;
        std::mem::swap(&mut x, &mut buf2);

        // Up stages
        let mut cur_h = h;
        let mut cur_w = w;
        for (stage_idx, stage) in self.up_stages.iter().enumerate() {
            let out_ch = self.config.block_out_channels[stage_idx + 1];

            // Upsample 2×
            cur_h *= 2;
            cur_w *= 2;
            let in_ch = x.numel() / (n * cur_h / 2 * cur_w / 2);
            let mut upsampled = D::alloc_tensor::<T>(
                Shape::from_slice(&[n, in_ch, cur_h, cur_w]), dev,
            )?;
            D::upsample_nearest_2x(&x, &mut upsampled)?;

            // Post-upsample conv
            let mut conv_out = D::alloc_tensor::<T>(Shape::from_slice(&[n, out_ch, cur_h, cur_w]), dev)?;
            stage.upsample_conv.forward(&upsampled, &mut conv_out)?;
            x = conv_out;

            // ResBlocks
            let mut rb1 = D::alloc_tensor::<T>(x.shape().clone(), dev)?;
            let mut rb2 = D::alloc_tensor::<T>(x.shape().clone(), dev)?;
            for res in &stage.res_blocks {
                res.forward(&x, &mut rb1, &mut rb2)?;
                std::mem::swap(&mut x, &mut rb2);
            }
        }

        // Final norm + silu + conv_out
        let mut normed = D::alloc_tensor::<T>(x.shape().clone(), dev)?;
        self.norm_out.forward_silu(&x, &mut normed)?;

        let out_ch = self.config.out_channels;
        let mut image = D::alloc_tensor::<T>(Shape::from_slice(&[n, out_ch, cur_h, cur_w]), dev)?;
        self.conv_out.forward(&normed, &mut image)?;

        Ok(image)
    }
}
