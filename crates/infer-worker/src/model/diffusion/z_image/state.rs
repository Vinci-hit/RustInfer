//! Pre-allocated runtime state for the Z-Image pipeline.
//!
//! Two containers live here:
//!
//! - [`PipelineState`] — pipeline-level I/O for the denoise loop
//!   (`Latents`, `LatentsTmp`, `NoisePred`, `Latent5D`).
//! - [`DitState`] — transformer-internal workspace (embeddings, RoPE
//!   cache, per-block scratch pool, final-layer temporaries).
//!
//! Every tensor is allocated once inside `new()` and never re-homed.
//! Forward-pass code (`ZImagePipeline::denoise`,
//! `ZImageTransformer2DModel::forward`, `DiTBlock::forward`) slices
//! sub-views out of these states on each invocation.
//!
//! ## CUDA Graph readiness
//!
//! Because tensor pointers are stable for the state's lifetime, any
//! forward computation run against these states is graph-capture
//! friendly. The one scalar that is *written* between replays is
//! [`DiffBufferType::TValueDev`], a single-element f32 device tensor
//! that holds the current `norm_t * t_scale`; the capturing caller is
//! expected to issue a `cudaMemcpyAsync` into it before each replay.

use std::collections::HashMap;

use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::model::diffusion::buffer::{DiffBufferType, DiffWorkspace};
use crate::tensor::Tensor;

/// Maximum number of denoising steps the state pre-allocates scratch for.
///
/// Stored in [`DiffBufferType::TValueDevVec`] / [`DiffBufferType::DtValueDevVec`]
/// so a single `[N_MAX] f32` upload per `generate()` hands the entire
/// schedule to the denoise CUDA Graph. Z-Image turbo uses 2 steps, the
/// official schedule 9; 16 is comfortable with zero meaningful cost
/// (64 B × 2 slots).
pub const N_MAX_STEPS: usize = 16;

/// Sequence length rounding used by the transformer (must match the
/// `SEQ_MULTI_OF` constant in `transformer.rs`).
const SEQ_MULTI_OF: usize = 128;

/// AdaLN conditioning embedding dim — compile-time constant in the model.
const ADALN_EMBED_DIM: usize = 256;

/// Sinusoidal frequency dim used by the timestep embedder input.
const T_FREQ_DIM: usize = 256;

/// The hidden size of the timestep embedder's intermediate MLP layer.
const T_EMBEDDER_MID: usize = 1024;

/// Number of latent channels (matches `LATENT_CHANNELS` in `pipeline.rs`).
const LATENT_CHANNELS: usize = 16;

// ─────────────────────────── Capacity & Spec ───────────────────────────

/// Runtime-configurable maximum shapes the pipeline must support.
///
/// The state allocates for the largest supported `(height, width,
/// cap_len)`. Any `generate()` request exceeding these bounds will be
/// rejected rather than triggering re-allocation — that keeps CUDA Graph
/// capture valid for the lifetime of the pipeline.
#[derive(Debug, Clone, Copy)]
pub struct ZImageCapacity {
    /// Maximum output image height (pixels).
    pub max_height: usize,
    /// Maximum output image width (pixels).
    pub max_width: usize,
    /// Maximum number of caption tokens (pre-padding) fed to the DiT.
    pub max_cap_len: usize,
}

impl Default for ZImageCapacity {
    fn default() -> Self {
        Self {
            max_height: 1024,
            max_width: 1024,
            max_cap_len: 512,
        }
    }
}

impl ZImageCapacity {
    /// Latent spatial dim at the maximum image resolution.
    ///
    /// Matches the computation in `ZImagePipeline::prepare_latents`:
    /// `latent = 2 * (image / (VAE_SCALE_FACTOR * 2)) = image / 8`.
    #[inline]
    pub fn max_latent_h(&self) -> usize {
        self.max_height / 8
    }
    #[inline]
    pub fn max_latent_w(&self) -> usize {
        self.max_width / 8
    }
}

/// All static shape info the DiT state needs to size its buffers.
///
/// Built once at pipeline construction from the DiT config + capacity.
#[derive(Debug, Clone, Copy)]
pub struct DitShapeSpec {
    // ── runtime config ──
    pub device: DeviceType,
    /// Weight / activation dtype (BF16 on CUDA, F32 on CPU in this repo).
    pub dtype: DataType,

    // ── model dims ──
    pub dim: usize,           // e.g. 3840
    pub n_heads: usize,       // e.g. 30
    pub head_dim: usize,      // e.g. 128
    pub hidden_dim: usize,    // e.g. 10240 (FFN inner size)
    pub cap_feat_dim: usize,  // 2560 for Qwen3 text encoder
    pub patch_size: usize,    // spatial patch (2 for the 2-2 kernel)
    pub f_patch_size: usize,  // temporal patch (1 for T2I)
    /// `patch_in_dim = f_patch_size * patch_size^2 * in_channels (=16)`.
    pub patch_in_dim: usize,
    /// `final_out_dim = patch_size^2 * f_patch_size * out_channels (=16)`.
    pub final_out_dim: usize,

    // ── capacity ──
    pub capacity: ZImageCapacity,
}

impl DitShapeSpec {
    /// Pre-pad image token count at the maximum latent resolution.
    #[inline]
    pub fn n_patches_max(&self) -> usize {
        let f_t = 1 / self.f_patch_size.max(1);
        let h_t = self.capacity.max_latent_h() / self.patch_size;
        let w_t = self.capacity.max_latent_w() / self.patch_size;
        f_t.max(1) * h_t * w_t
    }

    /// Padded image sequence length (rounded up to `SEQ_MULTI_OF`).
    #[inline]
    pub fn s_img_max(&self) -> usize {
        round_up(self.n_patches_max(), SEQ_MULTI_OF)
    }

    /// Padded caption sequence length.
    #[inline]
    pub fn s_cap_max(&self) -> usize {
        round_up(self.capacity.max_cap_len, SEQ_MULTI_OF)
    }

    /// Unified (image | caption) sequence length used by the main 30 DiT
    /// layers.
    #[inline]
    pub fn s_total_max(&self) -> usize {
        self.s_img_max() + self.s_cap_max()
    }
}

#[inline]
fn round_up(n: usize, multiple: usize) -> usize {
    debug_assert!(multiple > 0);
    n.div_ceil(multiple) * multiple
}

// ─────────────────────────── PipelineState ─────────────────────────────

/// Buffers owned at the pipeline level (denoise I/O only).
///
/// Lifetime: one per `ZImagePipeline`, constructed once. Never realloced.
pub struct PipelineState {
    pub device: DeviceType,
    /// Working dtype for latents (matches transformer's weight dtype).
    pub dtype: DataType,
    /// Capacity the state was allocated for. `generate()` must reject
    /// requests exceeding this.
    pub capacity: ZImageCapacity,
    pub buffers: DiffWorkspace,
}

impl PipelineState {
    /// Allocate the three pipeline-level buffers in their max shape.
    ///
    /// This performs up to three `cudaMalloc`s total (on CUDA) and no
    /// further allocations for the rest of the pipeline's lifetime.
    pub fn new(
        capacity: ZImageCapacity,
        dtype: DataType,
        device: DeviceType,
    ) -> Result<Self> {
        let lh = capacity.max_latent_h();
        let lw = capacity.max_latent_w();

        let mut buffers: DiffWorkspace = HashMap::new();

        buffers.insert(
            DiffBufferType::Latents,
            Tensor::new(&[1, LATENT_CHANNELS, lh, lw], dtype, device)?,
        );
        buffers.insert(
            DiffBufferType::LatentsTmp,
            Tensor::new(&[1, LATENT_CHANNELS, lh, lw], dtype, device)?,
        );
        buffers.insert(
            DiffBufferType::NoisePred,
            Tensor::new(&[1, LATENT_CHANNELS, lh, lw], dtype, device)?,
        );
        // NOTE: `Latent5D` used to live here but has moved to `DitState`
        // so the transformer's `forward_denoise_step` can read it from a
        // single state handle (and so the denoise CUDA Graph can bind to
        // one DitState slot).

        Ok(Self { device, dtype, capacity, buffers })
    }

    /// Validate that the requested `(height, width)` fits inside the
    /// state's capacity. Callers should invoke this at the top of
    /// `generate()`.
    pub fn check_request(&self, height: usize, width: usize) -> Result<()> {
        if height > self.capacity.max_height || width > self.capacity.max_width {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "image size {}x{} exceeds configured ZImageCapacity {}x{}",
                height, width, self.capacity.max_height, self.capacity.max_width
            ))
            .into());
        }
        Ok(())
    }

    /// Zero-copy slice of a pre-allocated buffer, sized at `shape`.
    ///
    /// Panics if `ty` was not installed by `new()` (programmer error).
    /// Returns `Err` if `shape` exceeds the buffer's allocated extent.
    #[inline]
    pub fn slice(&self, ty: DiffBufferType, shape: &[usize]) -> Result<Tensor> {
        let buf = crate::model::diffusion::buffer::must_get(&self.buffers, ty);
        let zeros = vec![0usize; shape.len()];
        buf.slice(&zeros, shape)
    }

    /// Mutable variant of [`Self::slice`].
    #[inline]
    pub fn slice_mut(&mut self, ty: DiffBufferType, shape: &[usize]) -> Result<Tensor> {
        let buf = crate::model::diffusion::buffer::must_get_mut(&mut self.buffers, ty);
        let zeros = vec![0usize; shape.len()];
        buf.slice(&zeros, shape)
    }
}

// ─────────────────────────── DitState ──────────────────────────────────

/// Buffers owned by the DiT transformer forward pass.
///
/// All transformer intermediate tensors (per-block scratch + whole-forward
/// working arrays) are in this single map. Block forwards overwrite the
/// `Blk*` entries each call; cross-block state flows through `Unified` /
/// `XPadded` / `CapPadded` which the *caller* (transformer.forward)
/// manages.
///
/// # Invariants
///
/// - `buffers` is populated exclusively by [`DitState::new`] and never
///   mutated structurally afterwards (no inserts, no removes, no swaps).
/// - Every pointer returned from a slice of any buffer remains stable for
///   the state's lifetime — CUDA Graph capture relies on this.
pub struct DitState {
    pub spec: DitShapeSpec,
    pub buffers: DiffWorkspace,

    // ─────────────────────── Host staging buffers ───────────────────────
    //
    // Persistent CPU-side scratch tensors used by the zero-allocation
    // forward path to stage data for a single `cudaMemcpyAsync` into a
    // pre-allocated device slot. Allocated once in `DitState::new`, never
    // re-homed. All of these are plain Host memory (non-pinned) — the
    // memcpy is already fully overlapped with GPU work by the current
    // stream and these payloads are small (≤ few MB).

    /// `[1, T_FREQ_DIM]` weight-dtype CPU — sinusoidal timestep embedding
    /// assembled host-side, uploaded into [`DiffBufferType::TFreq`].
    pub t_freq_host_stage: Tensor,
    /// `[s_img_max, head_dim/2]` F32 CPU — image RoPE cos assembled
    /// host-side, uploaded into [`DiffBufferType::XCos`].
    pub x_cos_host_stage: Tensor,
    /// `[s_img_max, head_dim/2]` F32 CPU — image RoPE sin.
    pub x_sin_host_stage: Tensor,
    /// `[s_cap_max, head_dim/2]` F32 CPU — caption RoPE cos.
    pub cap_cos_host_stage: Tensor,
    /// `[s_cap_max, head_dim/2]` F32 CPU — caption RoPE sin.
    pub cap_sin_host_stage: Tensor,
}

impl DitState {
    /// Allocate every DiT-side buffer, sized for `spec.capacity` and the
    /// model dims in `spec`.
    ///
    /// This is where all per-generate `cudaMalloc` activity is front-loaded:
    /// after `new` returns, no diffusion forward pass should ever allocate.
    pub fn new(spec: DitShapeSpec) -> Result<Self> {
        let DitShapeSpec {
            device,
            dtype,
            dim,
            n_heads: _,
            head_dim,
            hidden_dim,
            cap_feat_dim,
            patch_size: _,
            f_patch_size: _,
            patch_in_dim,
            final_out_dim,
            capacity,
        } = spec;
        let half_d = head_dim / 2;

        let s_img = spec.s_img_max();
        let s_cap = spec.s_cap_max();
        let s_tot = spec.s_total_max();
        let n_patches = spec.n_patches_max();

        let lh = capacity.max_latent_h();
        let lw = capacity.max_latent_w();

        let mut m: DiffWorkspace = HashMap::new();

        // ── timestep ──
        // `TValueDev` is the one slot we *write* between graph replays.
        // Keeping it at [1] (f32, device) means a single cudaMemcpyAsync
        // before each launch is all the host does.
        m.insert(
            DiffBufferType::TValueDev,
            Tensor::new(&[1], DataType::F32, device)?,
        );
        // TFreq holds the sinusoidal timestep embedding that feeds mlp1.
        // It must be in the transformer's weight dtype (typically BF16)
        // so mlp1 can be called directly without an intermediate cast.
        m.insert(
            DiffBufferType::TFreq,
            Tensor::new(&[1, T_FREQ_DIM], dtype, device)?,
        );
        m.insert(
            DiffBufferType::TEmbHidden,
            Tensor::new(&[1, T_EMBEDDER_MID], dtype, device)?,
        );
        m.insert(
            DiffBufferType::TEmbOut,
            Tensor::new(&[1, ADALN_EMBED_DIM], dtype, device)?,
        );
        m.insert(
            DiffBufferType::AdalnInput,
            Tensor::new(&[ADALN_EMBED_DIM], dtype, device)?,
        );
        m.insert(
            DiffBufferType::AdalnSilu,
            Tensor::new(&[ADALN_EMBED_DIM], dtype, device)?,
        );

        // ── patchify / image embedder ──
        m.insert(
            DiffBufferType::Patches,
            Tensor::new(&[n_patches, patch_in_dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::XEmb,
            Tensor::new(&[n_patches, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::XPadded,
            Tensor::new(&[s_img, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::XPaddedTmp,
            Tensor::new(&[s_img, dim], dtype, device)?,
        );

        // ── caption embedder ──
        m.insert(
            DiffBufferType::CapFeatsPadded,
            Tensor::new(&[s_cap, cap_feat_dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::CapNormed,
            Tensor::new(&[s_cap, cap_feat_dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::CapEmb,
            Tensor::new(&[s_cap, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::CapPadded,
            Tensor::new(&[s_cap, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::CapPaddedTmp,
            Tensor::new(&[s_cap, dim], dtype, device)?,
        );

        // ── position ids + RoPE cache ──
        // pos_ids live on the CPU because `RopeEmbedder3D::embed` does a
        // scalar lookup; the resulting cos/sin tables land on-device for
        // the kernels that consume them.
        m.insert(
            DiffBufferType::XPosIds,
            Tensor::new(&[s_img, 3], DataType::I32, DeviceType::Cpu)?,
        );
        m.insert(
            DiffBufferType::CapPosIds,
            Tensor::new(&[s_cap, 3], DataType::I32, DeviceType::Cpu)?,
        );
        m.insert(
            DiffBufferType::XCos,
            Tensor::new(&[s_img, half_d], DataType::F32, device)?,
        );
        m.insert(
            DiffBufferType::XSin,
            Tensor::new(&[s_img, half_d], DataType::F32, device)?,
        );
        m.insert(
            DiffBufferType::CapCos,
            Tensor::new(&[s_cap, half_d], DataType::F32, device)?,
        );
        m.insert(
            DiffBufferType::CapSin,
            Tensor::new(&[s_cap, half_d], DataType::F32, device)?,
        );
        m.insert(
            DiffBufferType::UnifiedCos,
            Tensor::new(&[s_tot, half_d], DataType::F32, device)?,
        );
        m.insert(
            DiffBufferType::UnifiedSin,
            Tensor::new(&[s_tot, half_d], DataType::F32, device)?,
        );

        // ── unified stream for the main 30 DiT layers ──
        m.insert(
            DiffBufferType::Unified,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::UnifiedTmp,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );

        // ── final layer ──
        m.insert(
            DiffBufferType::FinalNormed,
            Tensor::new(&[s_img, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::FinalScale,
            Tensor::new(&[1, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::FinalOut,
            Tensor::new(&[s_img, final_out_dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::ImageOut,
            Tensor::new(&[LATENT_CHANNELS, 1, lh, lw], dtype, device)?,
        );

        // ── DiTBlock scratch pool (shared across all blocks) ──
        m.insert(
            DiffBufferType::BlkModOut,
            Tensor::new(&[1, 4 * dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkNorm1X,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkQkvOut,
            Tensor::new(&[s_tot, 3 * dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkQ,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkK,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkV,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkAttnFlat,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkToOut,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkNorm2Attn,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkNorm1Ffn,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkW1Out,
            Tensor::new(&[s_tot, hidden_dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkW3Out,
            Tensor::new(&[s_tot, hidden_dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkFfnOut,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );
        m.insert(
            DiffBufferType::BlkNorm2Ffn,
            Tensor::new(&[s_tot, dim], dtype, device)?,
        );

        // ─────────────── CUDA Graph-friendly scratch slots ───────────────
        // Per-step timestep / dt schedules, pre-uploaded once per
        // generate() so the denoise graph can index into them with a
        // fixed pointer (only the bytes change across replays).
        m.insert(
            DiffBufferType::TValueDevVec,
            Tensor::new(&[N_MAX_STEPS], DataType::F32, device)?,
        );
        m.insert(
            DiffBufferType::DtValueDevVec,
            Tensor::new(&[N_MAX_STEPS], DataType::F32, device)?,
        );
        // Stable device-side home for the text encoder's output. Written
        // once per generate() from the (possibly freshly-allocated)
        // `prompt_embeds` tensor, then the denoise graph reads from this
        // fixed pointer.
        m.insert(
            DiffBufferType::PromptEmbedsPadded,
            Tensor::new(&[s_cap, cap_feat_dim], dtype, device)?,
        );
        // DiT input view of the current latents. The pipeline copies
        // `[1, C, H, W]` Latents → `[C, 1, H, W]` Latent5D each step.
        // Lives in DitState so the transformer can read it through a
        // single state handle during CUDA Graph capture/replay.
        m.insert(
            DiffBufferType::Latent5D,
            Tensor::new(&[LATENT_CHANNELS, 1, lh, lw], dtype, device)?,
        );

        // ─────────────── Host staging buffers ───────────────
        // Allocated once; written every forward in-place, then HtoD
        // uploaded into the corresponding device slot.
        let t_freq_host_stage =
            Tensor::new(&[1, T_FREQ_DIM], dtype, DeviceType::Cpu)?;
        let x_cos_host_stage =
            Tensor::new(&[s_img, half_d], DataType::F32, DeviceType::Cpu)?;
        let x_sin_host_stage =
            Tensor::new(&[s_img, half_d], DataType::F32, DeviceType::Cpu)?;
        let cap_cos_host_stage =
            Tensor::new(&[s_cap, half_d], DataType::F32, DeviceType::Cpu)?;
        let cap_sin_host_stage =
            Tensor::new(&[s_cap, half_d], DataType::F32, DeviceType::Cpu)?;

        Ok(Self {
            spec,
            buffers: m,
            t_freq_host_stage,
            x_cos_host_stage,
            x_sin_host_stage,
            cap_cos_host_stage,
            cap_sin_host_stage,
        })
    }

    /// Zero-copy slice of a pre-allocated buffer, sized at `shape`.
    ///
    /// Panics if `ty` was not installed by `new()` (programmer error).
    /// Returns `Err` if `shape` exceeds the buffer's allocated extent.
    #[inline]
    pub fn slice(&self, ty: DiffBufferType, shape: &[usize]) -> Result<Tensor> {
        let buf = crate::model::diffusion::buffer::must_get(&self.buffers, ty);
        let zeros = vec![0usize; shape.len()];
        buf.slice(&zeros, shape)
    }

    /// Mutable variant of [`Self::slice`].
    ///
    /// Note: `Tensor::slice` returns an owned `Tensor` that shares the
    /// underlying `Arc<Buffer>` — the `&mut self` here is only needed to
    /// enforce exclusive access while a downstream op writes through it.
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

    fn make_spec() -> DitShapeSpec {
        DitShapeSpec {
            device: DeviceType::Cpu,
            dtype: DataType::F32,
            dim: 3840,
            n_heads: 30,
            head_dim: 128,
            hidden_dim: 10240,
            cap_feat_dim: 2560,
            patch_size: 2,
            f_patch_size: 1,
            patch_in_dim: 1 * 2 * 2 * 16, // 64
            final_out_dim: 2 * 2 * 1 * 16, // 64
            capacity: ZImageCapacity {
                max_height: 256,
                max_width: 256,
                max_cap_len: 64,
            },
        }
    }

    #[test]
    fn capacity_latent_dims() {
        let cap = ZImageCapacity { max_height: 1024, max_width: 768, max_cap_len: 128 };
        assert_eq!(cap.max_latent_h(), 128);
        assert_eq!(cap.max_latent_w(), 96);
    }

    #[test]
    fn shape_spec_derives_seq_lens() {
        let spec = make_spec();
        // latent 32×32, patch=2 → 16×16 = 256 patches (already multiple of 128)
        assert_eq!(spec.n_patches_max(), 256);
        assert_eq!(spec.s_img_max(), 256);
        assert_eq!(spec.s_cap_max(), 128);  // 64 rounds up to SEQ_MULTI_OF=128
        assert_eq!(spec.s_total_max(), 384);
    }

    #[test]
    fn pipeline_state_installs_three_buffers() -> Result<()> {
        let cap = ZImageCapacity { max_height: 256, max_width: 256, max_cap_len: 64 };
        let st = PipelineState::new(cap, DataType::F32, DeviceType::Cpu)?;
        assert!(st.buffers.contains_key(&DiffBufferType::Latents));
        assert!(st.buffers.contains_key(&DiffBufferType::LatentsTmp));
        assert!(st.buffers.contains_key(&DiffBufferType::NoisePred));
        assert_eq!(st.buffers.len(), 3);

        let lh = cap.max_latent_h();
        let lw = cap.max_latent_w();
        assert_eq!(st.buffers[&DiffBufferType::Latents].shape(), &[1, 16, lh, lw]);
        assert_eq!(st.buffers[&DiffBufferType::LatentsTmp].shape(), &[1, 16, lh, lw]);
        assert_eq!(st.buffers[&DiffBufferType::NoisePred].shape(), &[1, 16, lh, lw]);
        Ok(())
    }

    #[test]
    fn pipeline_state_check_request_enforces_capacity() -> Result<()> {
        let cap = ZImageCapacity { max_height: 512, max_width: 512, max_cap_len: 64 };
        let st = PipelineState::new(cap, DataType::F32, DeviceType::Cpu)?;
        assert!(st.check_request(256, 256).is_ok());
        assert!(st.check_request(512, 512).is_ok());
        assert!(st.check_request(513, 512).is_err());
        assert!(st.check_request(512, 1024).is_err());
        Ok(())
    }

    #[test]
    fn dit_state_allocates_expected_shapes() -> Result<()> {
        let spec = make_spec();
        let st = DitState::new(spec)?;

        // spot-check a few key buffers
        let s_tot = spec.s_total_max();
        let s_img = spec.s_img_max();
        assert_eq!(
            st.buffers[&DiffBufferType::Unified].shape(),
            &[s_tot, spec.dim]
        );
        assert_eq!(
            st.buffers[&DiffBufferType::BlkW1Out].shape(),
            &[s_tot, spec.hidden_dim]
        );
        assert_eq!(
            st.buffers[&DiffBufferType::BlkAttnFlat].shape(),
            &[s_tot, spec.dim]
        );
        assert_eq!(st.buffers[&DiffBufferType::TValueDev].shape(), &[1]);
        assert_eq!(
            st.buffers[&DiffBufferType::FinalOut].shape(),
            &[s_img, spec.final_out_dim]
        );
        assert_eq!(
            st.buffers[&DiffBufferType::ImageOut].shape(),
            &[
                16,
                1,
                spec.capacity.max_latent_h(),
                spec.capacity.max_latent_w()
            ]
        );
        Ok(())
    }

    #[test]
    fn dit_state_slice_returns_view_of_requested_shape() -> Result<()> {
        let spec = make_spec();
        let st = DitState::new(spec)?;
        // Ask for a sub-sequence half the max; `Tensor::slice` should succeed
        // and report the new shape.
        let sub = st.slice(DiffBufferType::BlkNorm1X, &[spec.s_img_max(), spec.dim])?;
        assert_eq!(sub.shape(), &[spec.s_img_max(), spec.dim]);
        Ok(())
    }

    #[test]
    fn dit_state_slice_rejects_out_of_bounds() -> Result<()> {
        let spec = make_spec();
        let st = DitState::new(spec)?;
        // Buffer is [s_tot, dim] — asking for [s_tot+32, dim] must fail.
        let oversize = st.slice(
            DiffBufferType::BlkNorm1X,
            &[spec.s_total_max() + 32, spec.dim],
        );
        assert!(oversize.is_err());
        Ok(())
    }

    #[test]
    fn dit_state_all_buffers_installed() -> Result<()> {
        let spec = make_spec();
        let st = DitState::new(spec)?;

        // Enumerate every variant we expect DitState to own. If someone
        // adds a new variant to DiffBufferType but forgets to install it
        // in `DitState::new`, this test locates the gap immediately.
        //
        // Pipeline-scope (Latents/NoisePred/Latent5D) and VAE-scope
        // variants are deliberately excluded — they live in other states.
        let dit_scope = [
            DiffBufferType::TValueDev,
            DiffBufferType::TValueDevVec,
            DiffBufferType::DtValueDevVec,
            DiffBufferType::PromptEmbedsPadded,
            DiffBufferType::Latent5D,
            DiffBufferType::TFreq,
            DiffBufferType::TEmbHidden,
            DiffBufferType::TEmbOut,
            DiffBufferType::AdalnInput,
            DiffBufferType::AdalnSilu,
            DiffBufferType::Patches,
            DiffBufferType::XEmb,
            DiffBufferType::XPadded,
            DiffBufferType::XPaddedTmp,
            DiffBufferType::CapFeatsPadded,
            DiffBufferType::CapNormed,
            DiffBufferType::CapEmb,
            DiffBufferType::CapPadded,
            DiffBufferType::CapPaddedTmp,
            DiffBufferType::XPosIds,
            DiffBufferType::CapPosIds,
            DiffBufferType::XCos,
            DiffBufferType::XSin,
            DiffBufferType::CapCos,
            DiffBufferType::CapSin,
            DiffBufferType::UnifiedCos,
            DiffBufferType::UnifiedSin,
            DiffBufferType::Unified,
            DiffBufferType::UnifiedTmp,
            DiffBufferType::FinalNormed,
            DiffBufferType::FinalScale,
            DiffBufferType::FinalOut,
            DiffBufferType::ImageOut,
            DiffBufferType::BlkModOut,
            DiffBufferType::BlkNorm1X,
            DiffBufferType::BlkQkvOut,
            DiffBufferType::BlkQ,
            DiffBufferType::BlkK,
            DiffBufferType::BlkV,
            DiffBufferType::BlkAttnFlat,
            DiffBufferType::BlkToOut,
            DiffBufferType::BlkNorm2Attn,
            DiffBufferType::BlkNorm1Ffn,
            DiffBufferType::BlkW1Out,
            DiffBufferType::BlkW3Out,
            DiffBufferType::BlkFfnOut,
            DiffBufferType::BlkNorm2Ffn,
        ];
        for ty in dit_scope {
            assert!(
                st.buffers.contains_key(&ty),
                "DitState::new did not install {:?}",
                ty
            );
        }
        Ok(())
    }
}
