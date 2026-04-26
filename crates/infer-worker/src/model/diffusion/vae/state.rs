//! Pre-allocated runtime state for the VAE decoder.
//!
//! Sibling of `z_image::state::{PipelineState, DitState}`. Unlike the DiT
//! scratch pool — which can share a single max-sized buffer across all
//! 30 layers because every layer operates at the same sequence length —
//! the VAE up-path progressively doubles spatial resolution while
//! reducing channels, so we pre-allocate **one buffer per stage**.
//!
//! The channel count at each stage is taken as `max(channels encountered
//! at that stage)` so both the "before-upsample" and "after-channel-
//! reduction" tensors at the same spatial resolution can share the slot.
//!
//! This module only *declares and sizes* buffers. Rewiring
//! `VaeDecoder::decode` to consume them is tracked separately.

use std::collections::HashMap;

use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::model::diffusion::buffer::{DiffBufferType, DiffWorkspace};
use crate::model::diffusion::z_image::state::ZImageCapacity;
use crate::tensor::Tensor;

/// GroupNorm group count used by the VAE (must match `NORM_GROUPS` in
/// `decoder.rs`).
#[allow(dead_code)]
const VAE_NORM_GROUPS: usize = 32;

/// Shape spec consumed by [`VaeState::new`].
///
/// We parameterise on the capacity (max latent H/W) and the VAE's
/// `block_out_channels` list so the state can pre-compute every stage's
/// `(channels, height, width)` triple without depending on runtime config.
#[derive(Debug, Clone)]
pub struct VaeShapeSpec {
    pub device: DeviceType,
    pub dtype: DataType,

    /// Batch size supported. Currently always 1 (Z-Image pipeline runs
    /// single-image per call); kept explicit for future CFG/batched
    /// decoding.
    pub batch: usize,

    /// VAE latent channel count (matches `latent_channels` in config).
    pub latent_channels: usize,
    /// RGB output channel count (3).
    pub out_channels: usize,
    /// VAE `block_out_channels`, decoder-order reversed.
    ///
    /// Z-Image config has `[128, 256, 512, 512]`; on the decoder side
    /// stages go `mid_ch = 512 → up[0] 512 → up[1] 512 → up[2] 256 →
    /// up[3] 128`. Keep the original list here and do the reversal
    /// internally.
    pub block_out_channels: Vec<usize>,

    pub capacity: ZImageCapacity,
}

impl VaeShapeSpec {
    /// Channels and spatial size for each stage buffer.
    ///
    /// Returns 5 `(channels, h, w)` triples matching `DiffBufferType`'s
    /// `VaeStage0..=VaeStageOut`.
    pub fn stage_shapes(&self) -> [(usize, usize, usize); 5] {
        // boc in the config is given feed-forward direction; the decoder
        // reverses it.
        let n = self.block_out_channels.len();
        let mid_ch = self.block_out_channels[n - 1];
        let lh = self.capacity.max_latent_h();
        let lw = self.capacity.max_latent_w();

        // Stage 0: conv_in / mid_block output. Spatial = (lh, lw), ch = mid_ch.
        let s0 = (mid_ch, lh, lw);

        // Stages 1..=3 correspond to up_blocks[0..=2], each doubling spatial.
        // Channel count = max(in_ch, out_ch) at that stage so a single
        // slot fits both sides of the `resnets[0]` channel-reduction
        // crossover.
        //
        // up_blocks follow reversed block_out_channels.
        // up_blocks[i] output channels = reversed[i] = boc[n-1-i].
        // up_blocks[i] input channels  = output of previous stage (or mid_ch at i=0).
        let reversed: Vec<usize> = (0..n).map(|i| self.block_out_channels[n - 1 - i]).collect();

        // up_blocks[0]: 512 → 512, upsample → (2lh, 2lw)
        // Stage1 holds the post-upsample tensor of up_blocks[0].
        let s1_ch = reversed[0].max(mid_ch);
        let s1 = (s1_ch, 2 * lh, 2 * lw);

        // up_blocks[1]: 512 → 512, upsample → (4lh, 4lw)
        let s2_ch = reversed[1].max(reversed[0]);
        let s2 = (s2_ch, 4 * lh, 4 * lw);

        // up_blocks[2]: 512 → 256, upsample → (8lh, 8lw)
        // up_blocks[3]: 256 → 128, NO upsample (stays at 8lh, 8lw)
        // Both live at the same spatial resolution, so Stage3 covers the
        // widest channel count in that resolution.
        let mut s3_ch = reversed[2].max(reversed[1]);
        if n >= 4 {
            s3_ch = s3_ch.max(reversed[3]);
        }
        let s3 = (s3_ch, 8 * lh, 8 * lw);

        // Final RGB: conv_out projects Stage3 → out_channels at same spatial.
        let s_out = (self.out_channels, 8 * lh, 8 * lw);

        [s0, s1, s2, s3, s_out]
    }

    /// Largest `(channels, h, w)` across all stage resnets — used to size
    /// the two shared scratch slots (`VaeResnetTmp1/2`).
    ///
    /// Approach: take the max over all stage triples; that's the envelope
    /// in which any intra-stage resnet intermediate needs to live.
    pub fn resnet_scratch_shape(&self) -> (usize, usize, usize) {
        let shapes = self.stage_shapes();
        let mut max_ch = 0usize;
        let mut max_h = 0usize;
        let mut max_w = 0usize;
        for (c, h, w) in shapes.iter() {
            if *c > max_ch { max_ch = *c; }
            if *h > max_h { max_h = *h; }
            if *w > max_w { max_w = *w; }
        }
        (max_ch, max_h, max_w)
    }

    /// Channel count at the mid-block self-attention (= `block_out_channels[-1]`).
    pub fn mid_attn_ch(&self) -> usize {
        *self.block_out_channels.last().unwrap()
    }
}

// ─────────────────────────── VaeState ──────────────────────────────────

pub struct VaeState {
    pub spec: VaeShapeSpec,
    pub buffers: DiffWorkspace,
}

impl VaeState {
    pub fn new(spec: VaeShapeSpec) -> Result<Self> {
        let VaeShapeSpec {
            device,
            dtype,
            batch,
            capacity: _,
            ..
        } = spec.clone();

        let [s0, s1, s2, s3, s_out] = spec.stage_shapes();
        let (rc_ch, rc_h, rc_w) = spec.resnet_scratch_shape();
        let mid_ch = spec.mid_attn_ch();
        let lh = spec.capacity.max_latent_h();
        let lw = spec.capacity.max_latent_w();

        let mut m: DiffWorkspace = HashMap::new();

        m.insert(
            DiffBufferType::VaeStage0,
            Tensor::new(&[batch, s0.0, s0.1, s0.2], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeStage1,
            Tensor::new(&[batch, s1.0, s1.1, s1.2], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeStage2,
            Tensor::new(&[batch, s2.0, s2.1, s2.2], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeStage3,
            Tensor::new(&[batch, s3.0, s3.1, s3.2], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeStageOut,
            Tensor::new(&[batch, s_out.0, s_out.1, s_out.2], dtype, device)?,
        );

        // Resnet scratch: two ping-pong buffers, sized to the largest
        // stage so any resnet's norm/conv intermediate fits.
        m.insert(
            DiffBufferType::VaeResnetTmp1,
            Tensor::new(&[batch, rc_ch, rc_h, rc_w], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeResnetTmp2,
            Tensor::new(&[batch, rc_ch, rc_h, rc_w], dtype, device)?,
        );

        // Mid-block self-attention tensors. Only ever used at (lh, lw)
        // with `mid_ch` channels.
        m.insert(
            DiffBufferType::VaeAttnQ,
            Tensor::new(&[batch, mid_ch, lh, lw], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeAttnK,
            Tensor::new(&[batch, mid_ch, lh, lw], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeAttnV,
            Tensor::new(&[batch, mid_ch, lh, lw], dtype, device)?,
        );
        m.insert(
            DiffBufferType::VaeAttnOut,
            Tensor::new(&[batch, mid_ch, lh, lw], dtype, device)?,
        );

        Ok(Self { spec, buffers: m })
    }

    #[inline]
    pub fn slice(&self, ty: DiffBufferType, shape: &[usize]) -> Result<Tensor> {
        let buf = crate::model::diffusion::buffer::must_get(&self.buffers, ty);
        let zeros = vec![0usize; shape.len()];
        buf.slice(&zeros, shape)
    }

    #[inline]
    pub fn slice_mut(&mut self, ty: DiffBufferType, shape: &[usize]) -> Result<Tensor> {
        let buf = crate::model::diffusion::buffer::must_get_mut(&mut self.buffers, ty);
        let zeros = vec![0usize; shape.len()];
        buf.slice(&zeros, shape)
    }
}

// ─────────────────────────── Tests ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec() -> VaeShapeSpec {
        VaeShapeSpec {
            device: DeviceType::Cpu,
            dtype: DataType::F32,
            batch: 1,
            latent_channels: 16,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            capacity: ZImageCapacity {
                max_height: 256,
                max_width: 256,
                max_cap_len: 64,
            },
        }
    }

    #[test]
    fn stage_shapes_cover_decoder_resolutions() {
        let spec = make_spec();
        // latent 256/8 = 32 × 32
        let lh = 32;
        let [s0, s1, s2, s3, s_out] = spec.stage_shapes();

        assert_eq!(s0, (512, lh, lh));          // mid
        assert_eq!(s1, (512, 2 * lh, 2 * lh));  // up[0] → 64²
        assert_eq!(s2, (512, 4 * lh, 4 * lh));  // up[1] → 128²
        assert_eq!(s3, (512, 8 * lh, 8 * lh));  // up[2]/up[3] → 256² (max ch 512)
        assert_eq!(s_out, (3, 8 * lh, 8 * lh)); // RGB
    }

    #[test]
    fn resnet_scratch_envelopes_largest_stage() {
        let spec = make_spec();
        let (c, h, w) = spec.resnet_scratch_shape();
        // Largest stage is s3 = (512, 256, 256); resnet scratch must fit it.
        assert!(c >= 512);
        assert!(h >= 256 && w >= 256);
    }

    #[test]
    fn state_installs_all_vae_slots() -> Result<()> {
        let st = VaeState::new(make_spec())?;
        let slots = [
            DiffBufferType::VaeStage0,
            DiffBufferType::VaeStage1,
            DiffBufferType::VaeStage2,
            DiffBufferType::VaeStage3,
            DiffBufferType::VaeStageOut,
            DiffBufferType::VaeResnetTmp1,
            DiffBufferType::VaeResnetTmp2,
            DiffBufferType::VaeAttnQ,
            DiffBufferType::VaeAttnK,
            DiffBufferType::VaeAttnV,
            DiffBufferType::VaeAttnOut,
        ];
        for s in slots {
            assert!(
                st.buffers.contains_key(&s),
                "VaeState::new did not install {:?}",
                s
            );
        }
        Ok(())
    }

    #[test]
    fn stage_out_has_rgb_channels() -> Result<()> {
        let st = VaeState::new(make_spec())?;
        assert_eq!(st.buffers[&DiffBufferType::VaeStageOut].shape()[1], 3);
        Ok(())
    }

    #[test]
    fn slice_within_capacity_works() -> Result<()> {
        let spec = make_spec();
        let st = VaeState::new(spec.clone())?;
        let lh = spec.capacity.max_latent_h();
        // Request sub-rectangle: half-height. Stage1 is [1, 512, 2*lh, 2*lh].
        let sub = st.slice(
            DiffBufferType::VaeStage1,
            &[1, 512, lh, 2 * lh],
        )?;
        assert_eq!(sub.shape(), &[1, 512, lh, 2 * lh]);
        Ok(())
    }
}
