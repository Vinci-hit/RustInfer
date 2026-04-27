//! Z-Image Transformer 2D Model (S3-DiT denoising backbone).
//!
//! Full implementation that loads all 30 transformer blocks + 2 noise_refiner +
//! 2 context_refiner + x_embedder + cap_embedder + t_embedder + final_layer.
//!
//! ## Buffer-pool contract
//!
//! [`ZImageTransformer2DModel::forward`] takes a `&mut DitState` handle
//! and *does not allocate* any tensors along the hot path. Every
//! intermediate activation is sliced out of the workspace, keeping the
//! whole forward pass graph-capture-ready.

use std::path::Path;

use crate::OpConfig;
use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};

use crate::model::diffusion::buffer::DiffBufferType as BT;
use crate::model::diffusion::diffusers_loader::DiffusersLoader;
use crate::model::diffusion::z_image::dit_block::DiTBlock;
use crate::model::diffusion::z_image::rope_embedder_3d::RopeEmbedder3D;
use crate::model::diffusion::z_image::state::{DitShapeSpec, DitState, ZImageCapacity};
use crate::model::diffusion::z_image::timestep_embedder::TimestepEmbedder;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::tensor_utils::{
    concat_seq_into, overwrite_pad_tokens_inplace,
    pad_last_row_into, pad_with_token_into,
};
use crate::tensor::Tensor;

const ADALN_EMBED_DIM: usize = 256;
/// Sequence-length padding multiple.
///
/// Fixed at **128** to match the block-M tile of the CUTLASS flash-attention
/// kernel used for self-attention in [`DiTBlock::forward`]:
///
/// - Q tiles are [M=128, K=head_dim]; the kernel issues unpredicated
///   `cp_async` loads that read a full tile even on the trailing block,
///   so the sequence length must be a multiple of 128 for the loads to stay
///   in-bounds and for the result to be mathematically correct.
/// - KV tiles are [N=64, K=head_dim]; 128 is automatically a multiple of 64
///   so the KV side is trivially safe as well.
///
/// The extra padded rows are filled with the learned `x_pad_token` /
/// `cap_pad_token`, so correctness is unchanged — they just participate in
/// attention like any other real token. The marginal FFN / projection cost
/// of the extra tokens is far smaller than what flash-attention saves
/// versus the per-head loop fallback.
const SEQ_MULTI_OF: usize = 128;

// ───────────────────────── Config ─────────────────────────

#[derive(Debug, Clone)]
pub struct ZImageTransformerConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub in_channels: usize,
    pub cap_feat_dim: usize,
    pub all_patch_size: Vec<usize>,
    pub all_f_patch_size: Vec<usize>,
    pub axes_dims: Vec<usize>,
    pub axes_lens: Vec<usize>,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub t_scale: f32,
    pub qk_norm: bool,
}

impl ZImageTransformerConfig {
    pub fn from_json<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let file = std::fs::File::open(config_path)?;
        let v: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| Error::InvalidArgument(format!("Failed to parse DiT config.json: {}", e)))?;

        let get_usize = |k: &str| -> Result<usize> {
            v[k].as_u64()
                .ok_or_else(|| Error::InvalidArgument(format!("Missing or invalid '{}' in DiT config", k)))
                .map(|x| x as usize)
                .map_err(Into::into)
        };
        let get_f32 = |k: &str| -> Result<f32> {
            v[k].as_f64()
                .ok_or_else(|| Error::InvalidArgument(format!("Missing or invalid '{}' in DiT config", k)))
                .map(|x| x as f32)
                .map_err(Into::into)
        };
        let get_usize_arr = |k: &str| -> Result<Vec<usize>> {
            v[k].as_array()
                .ok_or_else(|| Error::InvalidArgument(format!("Missing or invalid '{}' in DiT config", k)))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize)
                    .ok_or_else(|| Error::InvalidArgument(format!("Invalid entry in '{}'", k)).into()))
                .collect()
        };

        Ok(Self {
            dim: get_usize("dim")?,
            n_layers: get_usize("n_layers")?,
            n_refiner_layers: get_usize("n_refiner_layers")?,
            n_heads: get_usize("n_heads")?,
            n_kv_heads: get_usize("n_kv_heads")?,
            in_channels: get_usize("in_channels")?,
            cap_feat_dim: get_usize("cap_feat_dim")?,
            all_patch_size: get_usize_arr("all_patch_size")?,
            all_f_patch_size: get_usize_arr("all_f_patch_size")?,
            axes_dims: get_usize_arr("axes_dims")?,
            axes_lens: get_usize_arr("axes_lens")?,
            norm_eps: v["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: get_f32("rope_theta")?,
            t_scale: get_f32("t_scale")?,
            qk_norm: v["qk_norm"].as_bool().unwrap_or(true),
        })
    }
}

// ───────────────────────── Model ─────────────────────────

/// Shape bundle returned by [`ZImageTransformer2DModel::prepare_denoise_constants`]
/// and consumed by [`ZImageTransformer2DModel::forward_denoise_step`] to
/// slice the state.
///
/// These dimensions are a pure function of the request shape
/// `(f, h, w)` + `cap_feats.shape()[0]`, so they are identical for
/// every step of a single generate().
#[derive(Clone, Copy, Debug)]
pub struct DenoiseShapes {
    pub f: usize,
    pub h: usize,
    pub w: usize,
    pub n_patches: usize,
    pub x_padded_len: usize,
    pub cap_padded_len: usize,
    pub cap_ori_len: usize,
}

pub struct ZImageTransformer2DModel {
    pub config: ZImageTransformerConfig,
    pub device: DeviceType,

    // Embeddings
    pub x_embedder: Matmul,          // Linear(patch_in, dim)
    pub cap_embedder_norm: RMSNorm,  // RMSNorm(cap_feat_dim)
    pub cap_embedder_linear: Matmul, // Linear(cap_feat_dim, dim)
    pub t_embedder: TimestepEmbedder,

    // Pad tokens
    pub x_pad_token: Tensor,   // [1, dim]
    pub cap_pad_token: Tensor, // [1, dim]

    // Refiners
    pub noise_refiner: Vec<DiTBlock>,    // modulation=True
    pub context_refiner: Vec<DiTBlock>,  // modulation=False

    // Main
    pub layers: Vec<DiTBlock>, // modulation=True, 30 layers

    // Final
    pub final_layer_linear: Matmul,             // Linear(dim, out)
    pub final_layer_adaln: Matmul,              // Linear(256, dim)
    pub final_layer_eps: f32,

    // RoPE
    pub rope_embedder: RopeEmbedder3D,

    // Patch key (e.g. "2-1")
    pub patch_key: String,
    pub patch_size: usize,
    pub f_patch_size: usize,
}

impl ZImageTransformer2DModel {
    /// Load from a diffusers transformer directory.
    ///
    /// Expects:
    /// - `{dir}/config.json`
    /// - `{dir}/diffusion_pytorch_model.safetensors.index.json` (+ shards)
    pub fn from_pretrained<P: AsRef<Path>>(
        transformer_dir: P,
        device: DeviceType,
    ) -> Result<Self> {
        let dir = transformer_dir.as_ref();
        let config = ZImageTransformerConfig::from_json(dir.join("config.json"))?;
        let loader = DiffusersLoader::load(dir)?;

        let patch_size = config.all_patch_size[0];
        let f_patch_size = config.all_f_patch_size[0];
        let patch_key = format!("{}-{}", patch_size, f_patch_size);
        let patch_in_dim = f_patch_size * patch_size * patch_size * config.in_channels;
        let out_channels = config.in_channels;
        let final_out_dim = patch_size * patch_size * f_patch_size * out_channels;
        let head_dim = config.dim / config.n_heads;
        let hidden_dim = config.dim / 3 * 8;

        // ── Embeddings ──
        let x_embedder = load_linear(&loader, &format!("all_x_embedder.{}", patch_key), patch_in_dim, config.dim, true, device)?;
        let cap_embedder_norm = RMSNorm::from(
            load_tensor(&loader, "cap_embedder.0.weight", device)?,
            config.norm_eps,
        );
        let cap_embedder_linear = load_linear(&loader, "cap_embedder.1", config.cap_feat_dim, config.dim, true, device)?;
        let t_embedder = load_timestep_embedder(&loader, ADALN_EMBED_DIM, 1024, ADALN_EMBED_DIM, device)?;

        // ── Pad tokens ──
        let x_pad_token = load_tensor(&loader, "x_pad_token", device)?;
        let cap_pad_token = load_tensor(&loader, "cap_pad_token", device)?;

        // ── Refiners ──
        let mut noise_refiner = Vec::with_capacity(config.n_refiner_layers);
        for i in 0..config.n_refiner_layers {
            noise_refiner.push(load_dit_block(
                &loader, &format!("noise_refiner.{}", i),
                config.dim, config.n_heads, head_dim, hidden_dim,
                config.norm_eps, true, device,
            )?);
        }
        let mut context_refiner = Vec::with_capacity(config.n_refiner_layers);
        for i in 0..config.n_refiner_layers {
            context_refiner.push(load_dit_block(
                &loader, &format!("context_refiner.{}", i),
                config.dim, config.n_heads, head_dim, hidden_dim,
                config.norm_eps, false, device,
            )?);
        }

        // ── Main layers ──
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(load_dit_block(
                &loader, &format!("layers.{}", i),
                config.dim, config.n_heads, head_dim, hidden_dim,
                config.norm_eps, true, device,
            )?);
        }

        // ── Final layer ──
        let final_layer_eps = 1e-6f32;
        let final_layer_linear = load_linear(
            &loader, &format!("all_final_layer.{}.linear", patch_key),
            config.dim, final_out_dim, true, device,
        )?;
        let final_layer_adaln = load_linear(
            &loader, &format!("all_final_layer.{}.adaLN_modulation.1", patch_key),
            ADALN_EMBED_DIM, config.dim, true, device,
        )?;

        // ── RoPE ──
        let axes_dims: [usize; 3] = [config.axes_dims[0], config.axes_dims[1], config.axes_dims[2]];
        let axes_lens: [usize; 3] = [config.axes_lens[0], config.axes_lens[1], config.axes_lens[2]];
        let rope_embedder = RopeEmbedder3D::new(axes_dims, axes_lens, config.rope_theta as f64)?;

        Ok(Self {
            config,
            device,
            x_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            t_embedder,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            final_layer_eps,
            final_layer_linear,
            final_layer_adaln,
            rope_embedder,
            patch_key,
            patch_size,
            f_patch_size,
        })
    }

    /// Build a [`DitShapeSpec`] describing this model's buffer sizes for
    /// the given runtime capacity. Consumed by `DitState::new` in the
    /// pipeline constructor to allocate the transformer workspace.
    pub fn shape_spec(&self, capacity: ZImageCapacity) -> DitShapeSpec {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let head_dim = dim / n_heads;
        let hidden_dim = dim / 3 * 8;
        let patch_size = self.patch_size;
        let f_patch_size = self.f_patch_size;
        let in_channels = self.config.in_channels;
        let patch_in_dim = f_patch_size * patch_size * patch_size * in_channels;
        let final_out_dim = patch_size * patch_size * f_patch_size * in_channels;

        DitShapeSpec {
            device: self.device,
            dtype: self.x_embedder.weight.dtype(),
            dim,
            n_heads,
            head_dim,
            hidden_dim,
            cap_feat_dim: self.config.cap_feat_dim,
            patch_size,
            f_patch_size,
            patch_in_dim,
            final_out_dim,
            capacity,
        }
    }

    /// Forward pass for a single sample.
    ///
    /// - `image`: `[C=16, 1, H, W]` latent (dtype must match weights).
    /// - `t_value`: scalar timestep, already normalized by the caller.
    /// - `cap_feats`: `[S_cap, cap_feat_dim]` text embeddings (dtype must
    ///    match weights).
    /// - `state`: transformer workspace. Every per-call buffer (patches,
    ///    embeddings, RoPE cache, final-layer temporaries, plus the
    ///    per-block scratch pool) is sliced out of this state — no
    ///    `Tensor::new` runs along the hot path.
    ///
    /// Returns the predicted velocity shaped `[C=16, 1, H, W]`. The
    /// returned tensor **aliases** `state.ImageOut`; callers must consume
    /// it (typically via `copy_from` into a separate `NoisePred` slot)
    /// before invoking `forward` again.
    pub fn forward(
        &self,
        image: &Tensor,
        t_value: f32,
        cap_feats: &Tensor,
        cuda_config: Option<&OpConfig>,
        state: &mut DitState,
    ) -> Result<Tensor> {
        // ── Timestep embedding (zero-alloc) ──
        // The scalar timestep scaling happens here on host; sinusoid is
        // assembled into `state.t_freq_host_stage`, uploaded once into
        // the pre-allocated TFreq slot, then fed through mlp1 / silu /
        // mlp2, each writing into pre-allocated state slots.
        let scaled_t = t_value * self.config.t_scale;
        {
            // Scope the borrows so `state` can be re-borrowed below.
            let (t_freq_slot_ptr, t_hidden_ptr, t_out_ptr) = {
                let t_freq = state.slice_mut(BT::TFreq, &[1, 256])?;
                let t_hidden = state.slice_mut(BT::TEmbHidden, &[1, self.t_embedder.mlp1.weight.shape()[0]])?;
                let t_out = state.slice_mut(BT::TEmbOut, &[1, ADALN_EMBED_DIM])?;
                (t_freq, t_hidden, t_out)
            };
            // The three views above share `Arc<Buffer>` with the state
            // slots — mutations through them persist in the state.
            let mut t_freq = t_freq_slot_ptr;
            let mut t_hidden = t_hidden_ptr;
            let mut t_out = t_out_ptr;
            self.t_embedder.forward_into(
                scaled_t,
                &mut t_freq,
                &mut t_hidden,
                &mut t_out,
                &mut state.t_freq_host_stage,
                cuda_config,
            )?;
        }
        let adaln_input_view = state.slice(BT::TEmbOut, &[1, ADALN_EMBED_DIM])?
            .view(&[ADALN_EMBED_DIM])?;
        let mut adaln_input = state.slice_mut(BT::AdalnInput, &[ADALN_EMBED_DIM])?;
        adaln_input.copy_from_on_current_stream(&adaln_input_view)?;

        // ── Patchify & embed image (zero-alloc) ──
        let image_shape = image.shape();
        let (_c, f, h, w) = (image_shape[0], image_shape[1], image_shape[2], image_shape[3]);
        let p = self.patch_size;
        let pf = self.f_patch_size;
        let n = (f / pf) * (h / p) * (w / p);
        let patch_in_dim = pf * p * p * self.config.in_channels;

        // Write patchify result directly into the pre-allocated Patches slot.
        let mut patches = state.slice_mut(BT::Patches, &[n, patch_in_dim])?;
        crate::model::diffusion::z_image::patchify::patchify_into(
            image, p, pf, &mut patches,
        )?;

        // x_emb = x_embedder(patches): [n, dim]
        let mut x_emb = state.slice_mut(BT::XEmb, &[n, self.config.dim])?;
        self.x_embedder.forward(&patches, &mut x_emb, cuda_config)?;

        // Pad x to SEQ_MULTI_OF → x_padded [x_padded_len, dim]
        let x_pad = (SEQ_MULTI_OF - n % SEQ_MULTI_OF) % SEQ_MULTI_OF;
        let x_padded_len = n + x_pad;
        let mut x_padded = state.slice_mut(BT::XPadded, &[x_padded_len, self.config.dim])?;
        pad_with_token_into(&x_emb, &self.x_pad_token, &mut x_padded)?;

        // ── Cap embed (zero-alloc except existing helpers) ──
        let cap_ori_len = cap_feats.shape()[0];
        let cap_pad = (SEQ_MULTI_OF - cap_ori_len % SEQ_MULTI_OF) % SEQ_MULTI_OF;
        let cap_padded_len = cap_ori_len + cap_pad;

        // pad_last_row(cap_feats) → CapFeatsPadded slot
        let mut cap_feats_padded = state.slice_mut(
            BT::CapFeatsPadded, &[cap_padded_len, self.config.cap_feat_dim],
        )?;
        pad_last_row_into(cap_feats, &mut cap_feats_padded)?;

        // cap_embedder_norm(cap_feats_padded) → CapNormed slot
        let mut cap_normed = state.slice_mut(
            BT::CapNormed, &[cap_padded_len, self.config.cap_feat_dim],
        )?;
        self.cap_embedder_norm.forward(&cap_feats_padded, &mut cap_normed, cuda_config)?;

        // cap_embedder_linear(cap_normed) → CapEmb slot
        let mut cap_emb = state.slice_mut(BT::CapEmb, &[cap_padded_len, self.config.dim])?;
        self.cap_embedder_linear.forward(&cap_normed, &mut cap_emb, cuda_config)?;

        // overwrite_pad_tokens → CapPadded slot (copy then stamp)
        let mut cap_padded = state.slice_mut(BT::CapPadded, &[cap_padded_len, self.config.dim])?;
        cap_padded.copy_from_on_current_stream(&cap_emb)?;
        overwrite_pad_tokens_inplace(&mut cap_padded, &self.cap_pad_token, cap_ori_len)?;

        // ── Build position ids into pre-allocated CPU slots, compute
        // RoPE cos/sin into pre-allocated device slots — all zero-alloc. ──
        let f_t = f / self.f_patch_size;
        let h_t = h / self.patch_size;
        let w_t = w / self.patch_size;

        let half_d = self.config.dim / self.config.n_heads / 2;

        // Cap pos ids → XPosIds/CapPosIds live on CPU per DitState::new.
        {
            let mut cap_pos_ids = state.slice_mut(BT::CapPosIds, &[cap_padded_len, 3])?;
            fill_cap_pos_ids(&mut cap_pos_ids, cap_padded_len)?;
        }
        {
            let mut x_pos_ids = state.slice_mut(BT::XPosIds, &[x_padded_len, 3])?;
            fill_image_pos_ids(
                &mut x_pos_ids,
                f_t, h_t, w_t,
                cap_padded_len + 1,
                x_padded_len - n,
            )?;
        }

        // RoPE embed into pre-allocated XCos/XSin/CapCos/CapSin slots.
        {
            let x_pos_ids = state.slice(BT::XPosIds, &[x_padded_len, 3])?;
            let mut x_cos = state.slice_mut(BT::XCos, &[x_padded_len, half_d])?;
            let mut x_sin = state.slice_mut(BT::XSin, &[x_padded_len, half_d])?;
            let x_cos_stage = state.x_cos_host_stage
                .slice(&[0, 0], &[x_padded_len, half_d])?;
            let x_sin_stage = state.x_sin_host_stage
                .slice(&[0, 0], &[x_padded_len, half_d])?;
            let mut x_cos_stage = x_cos_stage;
            let mut x_sin_stage = x_sin_stage;
            self.rope_embedder.embed_into(
                &x_pos_ids,
                &mut x_cos, &mut x_sin,
                &mut x_cos_stage, &mut x_sin_stage,
            )?;
        }
        {
            let cap_pos_ids = state.slice(BT::CapPosIds, &[cap_padded_len, 3])?;
            let mut cap_cos = state.slice_mut(BT::CapCos, &[cap_padded_len, half_d])?;
            let mut cap_sin = state.slice_mut(BT::CapSin, &[cap_padded_len, half_d])?;
            let mut cap_cos_stage = state.cap_cos_host_stage
                .slice(&[0, 0], &[cap_padded_len, half_d])?;
            let mut cap_sin_stage = state.cap_sin_host_stage
                .slice(&[0, 0], &[cap_padded_len, half_d])?;
            self.rope_embedder.embed_into(
                &cap_pos_ids,
                &mut cap_cos, &mut cap_sin,
                &mut cap_cos_stage, &mut cap_sin_stage,
            )?;
        }
        let x_cos = state.slice(BT::XCos, &[x_padded_len, half_d])?;
        let x_sin = state.slice(BT::XSin, &[x_padded_len, half_d])?;
        let cap_cos = state.slice(BT::CapCos, &[cap_padded_len, half_d])?;
        let cap_sin = state.slice(BT::CapSin, &[cap_padded_len, half_d])?;

        // ── noise_refiner on x (ping-pong XPadded ↔ XPaddedTmp) ──
        let x_final_ty = self.run_block_chain(
            &self.noise_refiner,
            (BT::XPadded, BT::XPaddedTmp),
            &[x_padded_len, self.config.dim],
            &x_cos, &x_sin, Some(&adaln_input),
            state, cuda_config,
        )?;

        // ── context_refiner on cap (ping-pong CapPadded ↔ CapPaddedTmp) ──
        let cap_final_ty = self.run_block_chain(
            &self.context_refiner,
            (BT::CapPadded, BT::CapPaddedTmp),
            &[cap_padded_len, self.config.dim],
            &cap_cos, &cap_sin, None,
            state, cuda_config,
        )?;

        // ── Unified concat: [x | cap] ──
        let s_tot = x_padded_len + cap_padded_len;

        {
            let x_final = state.slice(x_final_ty, &[x_padded_len, self.config.dim])?;
            let cap_final = state.slice(cap_final_ty, &[cap_padded_len, self.config.dim])?;
            let mut unified = state.slice_mut(BT::Unified, &[s_tot, self.config.dim])?;
            concat_seq_into(&x_final, &cap_final, &mut unified)?;
        }
        let mut unified_cos = state.slice_mut(BT::UnifiedCos, &[s_tot, half_d])?;
        concat_seq_into(&x_cos, &cap_cos, &mut unified_cos)?;
        let mut unified_sin = state.slice_mut(BT::UnifiedSin, &[s_tot, half_d])?;
        concat_seq_into(&x_sin, &cap_sin, &mut unified_sin)?;

        // ── Main layers (ping-pong Unified ↔ UnifiedTmp) ──
        let unified_final_ty = self.run_block_chain(
            &self.layers,
            (BT::Unified, BT::UnifiedTmp),
            &[s_tot, self.config.dim],
            &unified_cos, &unified_sin, Some(&adaln_input),
            state, cuda_config,
        )?;

        // ── Final layer (image-part only; cap tail is discarded) ──
        {
            let unified_final = state.slice(unified_final_ty, &[s_tot, self.config.dim])?;
            let image_prefix = unified_final.slice(&[0, 0], &[x_padded_len, self.config.dim])?;
            self.final_layer_forward(&image_prefix, &adaln_input, state, cuda_config)?;
        }

        // ── Unpatchify into ImageOut slot (zero-alloc). ──
        //
        // FinalOut currently holds [x_padded_len, final_out_dim]; we
        // only want the first `n` token rows. The head slice is a
        // contiguous prefix of FinalOut so `unpatchify_into` can read
        // it directly without materializing an intermediate tensor.
        let final_out = state.slice(BT::FinalOut, &[x_padded_len, self.final_out_dim()])?;
        let out_x = final_out.slice(&[0, 0], &[n, self.final_out_dim()])?;
        let mut image_out = state.slice_mut(
            BT::ImageOut, &[self.config.in_channels, f, h, w],
        )?;
        crate::model::diffusion::z_image::patchify::unpatchify_into(
            &out_x, f, h, w,
            self.config.in_channels, self.patch_size, self.f_patch_size,
            &mut image_out,
        )?;
        Ok(image_out)
    }

    // ───────────────── CUDA-Graph-friendly denoise split ─────────────────
    //
    // Normally `forward` does *everything* each step. That's fine for
    // eager execution, but it makes the step uncapturable because each
    // step re-runs `cap_embedder`, `rope.embed_into`, `concat` etc. —
    // all of which work on shape-fixed inputs and therefore don't need
    // to happen inside the graph.
    //
    // `prepare_denoise_constants` runs every shape-fixed computation
    // **once** per generate() (outside capture); `forward_denoise_step`
    // runs only the per-step work (inside capture). The boundary lives
    // entirely in the workspace: every tensor the step reads without
    // writing is pre-populated by `prepare_denoise_constants` into a
    // slot with a stable device pointer.

    /// Per-request, shape-fixed precomputation. Called **once** per
    /// generate(), **before** the denoise CUDA Graph capture region.
    ///
    /// Populates: `CapFeatsPadded / CapNormed / CapEmb / CapPadded`
    /// (after `context_refiner`), `XPosIds / CapPosIds`,
    /// `XCos / XSin / CapCos / CapSin`, and `UnifiedCos / UnifiedSin`.
    /// Every per-step tensor that is a pure function of request shape
    /// is resolved here so the denoise step can work against fixed
    /// device pointers.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_denoise_constants(
        &self,
        cap_feats: &Tensor,
        f: usize, h: usize, w: usize,
        state: &mut DitState,
        cuda_config: Option<&OpConfig>,
    ) -> Result<DenoiseShapes> {
        let dim = self.config.dim;
        let half_d = dim / self.config.n_heads / 2;
        let p = self.patch_size;
        let pf = self.f_patch_size;

        let f_t = f / pf;
        let h_t = h / p;
        let w_t = w / p;
        let n = f_t * h_t * w_t;
        let x_pad = (SEQ_MULTI_OF - n % SEQ_MULTI_OF) % SEQ_MULTI_OF;
        let x_padded_len = n + x_pad;

        let cap_ori_len = cap_feats.shape()[0];
        let cap_pad = (SEQ_MULTI_OF - cap_ori_len % SEQ_MULTI_OF) % SEQ_MULTI_OF;
        let cap_padded_len = cap_ori_len + cap_pad;

        // ── Cap side: pad → norm → linear → stamp pad_tokens ──
        let mut cap_feats_padded = state.slice_mut(
            BT::CapFeatsPadded, &[cap_padded_len, self.config.cap_feat_dim],
        )?;
        pad_last_row_into(cap_feats, &mut cap_feats_padded)?;

        let mut cap_normed = state.slice_mut(
            BT::CapNormed, &[cap_padded_len, self.config.cap_feat_dim],
        )?;
        self.cap_embedder_norm.forward(&cap_feats_padded, &mut cap_normed, cuda_config)?;

        let mut cap_emb = state.slice_mut(BT::CapEmb, &[cap_padded_len, dim])?;
        self.cap_embedder_linear.forward(&cap_normed, &mut cap_emb, cuda_config)?;

        let mut cap_padded = state.slice_mut(BT::CapPadded, &[cap_padded_len, dim])?;
        cap_padded.copy_from_on_current_stream(&cap_emb)?;
        overwrite_pad_tokens_inplace(&mut cap_padded, &self.cap_pad_token, cap_ori_len)?;

        // ── Position ids into CPU slots ──
        {
            let mut cap_pos_ids = state.slice_mut(BT::CapPosIds, &[cap_padded_len, 3])?;
            fill_cap_pos_ids(&mut cap_pos_ids, cap_padded_len)?;
        }
        {
            let mut x_pos_ids = state.slice_mut(BT::XPosIds, &[x_padded_len, 3])?;
            fill_image_pos_ids(
                &mut x_pos_ids, f_t, h_t, w_t, cap_padded_len + 1, x_padded_len - n,
            )?;
        }

        // ── Rope embed into XCos / XSin / CapCos / CapSin (device) ──
        {
            let x_pos_ids = state.slice(BT::XPosIds, &[x_padded_len, 3])?;
            let mut x_cos = state.slice_mut(BT::XCos, &[x_padded_len, half_d])?;
            let mut x_sin = state.slice_mut(BT::XSin, &[x_padded_len, half_d])?;
            let mut x_cos_stage = state.x_cos_host_stage.slice(&[0, 0], &[x_padded_len, half_d])?;
            let mut x_sin_stage = state.x_sin_host_stage.slice(&[0, 0], &[x_padded_len, half_d])?;
            self.rope_embedder.embed_into(
                &x_pos_ids, &mut x_cos, &mut x_sin,
                &mut x_cos_stage, &mut x_sin_stage,
            )?;
        }
        {
            let cap_pos_ids = state.slice(BT::CapPosIds, &[cap_padded_len, 3])?;
            let mut cap_cos = state.slice_mut(BT::CapCos, &[cap_padded_len, half_d])?;
            let mut cap_sin = state.slice_mut(BT::CapSin, &[cap_padded_len, half_d])?;
            let mut cap_cos_stage = state.cap_cos_host_stage.slice(&[0, 0], &[cap_padded_len, half_d])?;
            let mut cap_sin_stage = state.cap_sin_host_stage.slice(&[0, 0], &[cap_padded_len, half_d])?;
            self.rope_embedder.embed_into(
                &cap_pos_ids, &mut cap_cos, &mut cap_sin,
                &mut cap_cos_stage, &mut cap_sin_stage,
            )?;
        }

        // ── context_refiner on cap (ping-pong CapPadded ↔ CapPaddedTmp);
        //    normalize result back to CapPadded so the denoise step can
        //    always read from a single, stable slot. ──
        let cap_final_ty = self.run_block_chain(
            &self.context_refiner,
            (BT::CapPadded, BT::CapPaddedTmp),
            &[cap_padded_len, dim],
            &state.slice(BT::CapCos, &[cap_padded_len, half_d])?,
            &state.slice(BT::CapSin, &[cap_padded_len, half_d])?,
            None,
            state, cuda_config,
        )?;
        if cap_final_ty != BT::CapPadded {
            let tmp = state.slice(BT::CapPaddedTmp, &[cap_padded_len, dim])?;
            let mut dst = state.slice_mut(BT::CapPadded, &[cap_padded_len, dim])?;
            dst.copy_from_on_current_stream(&tmp)?;
        }

        // ── Unified cos/sin (shape-fixed; concat once for the denoise step) ──
        let s_tot = x_padded_len + cap_padded_len;
        let x_cos = state.slice(BT::XCos, &[x_padded_len, half_d])?;
        let x_sin = state.slice(BT::XSin, &[x_padded_len, half_d])?;
        let cap_cos = state.slice(BT::CapCos, &[cap_padded_len, half_d])?;
        let cap_sin = state.slice(BT::CapSin, &[cap_padded_len, half_d])?;
        let mut unified_cos = state.slice_mut(BT::UnifiedCos, &[s_tot, half_d])?;
        concat_seq_into(&x_cos, &cap_cos, &mut unified_cos)?;
        let mut unified_sin = state.slice_mut(BT::UnifiedSin, &[s_tot, half_d])?;
        concat_seq_into(&x_sin, &cap_sin, &mut unified_sin)?;

        Ok(DenoiseShapes {
            f, h, w,
            n_patches: n,
            x_padded_len, cap_padded_len, cap_ori_len,
        })
    }

    /// Per-step denoise body, designed to be captured into a CUDA Graph.
    ///
    /// Reads:
    ///   - `d_t_scaled` : `[1]` F32 device, `t_value(i) * t_scale`
    ///   - `Latent5D` (populated from `Latents` by the caller before
    ///                 calling us)
    ///   - `CapPadded`, `XCos`, `XSin`, `UnifiedCos`, `UnifiedSin` —
    ///     all filled by `prepare_denoise_constants`
    /// Writes:
    ///   - Returns a view into `ImageOut` holding the predicted velocity.
    ///
    /// Every intermediate lives in a pre-allocated state slot; no
    /// `Tensor::new` runs on this path, and no host-originating data
    /// (other than compile-time constants baked into kernel launches)
    /// is written on the stream, so the entire body is graph-capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_denoise_step(
        &self,
        d_t_scaled: &Tensor,
        shapes: DenoiseShapes,
        state: &mut DitState,
        cuda_config: Option<&OpConfig>,
    ) -> Result<Tensor> {
        let DenoiseShapes {
            f, h, w, n_patches: n,
            x_padded_len, cap_padded_len, cap_ori_len: _,
        } = shapes;
        let dim = self.config.dim;
        let half_d = dim / self.config.n_heads / 2;
        let p = self.patch_size;
        let pf = self.f_patch_size;

        // ── Timestep embedding (device-scalar → sinusoid → mlp) ──
        {
            let mut t_freq = state.slice_mut(BT::TFreq, &[1, 256])?;
            let mut t_hidden = state.slice_mut(
                BT::TEmbHidden, &[1, self.t_embedder.mlp1.weight.shape()[0]],
            )?;
            let mut t_out = state.slice_mut(BT::TEmbOut, &[1, ADALN_EMBED_DIM])?;
            self.t_embedder.forward_from_dev(
                d_t_scaled, &mut t_freq, &mut t_hidden, &mut t_out, cuda_config,
            )?;
        }
        let adaln_input_view = state.slice(BT::TEmbOut, &[1, ADALN_EMBED_DIM])?
            .view(&[ADALN_EMBED_DIM])?;
        let mut adaln_input = state.slice_mut(BT::AdalnInput, &[ADALN_EMBED_DIM])?;
        adaln_input.copy_from_on_current_stream(&adaln_input_view)?;

        // ── Patchify current latent (caller has already populated Latent5D) ──
        let patch_in_dim = pf * p * p * self.config.in_channels;
        let latent_5d = state.slice(BT::Latent5D, &[self.config.in_channels, f, h, w])?;
        let mut patches = state.slice_mut(BT::Patches, &[n, patch_in_dim])?;
        crate::model::diffusion::z_image::patchify::patchify_into(
            &latent_5d, p, pf, &mut patches,
        )?;

        let mut x_emb = state.slice_mut(BT::XEmb, &[n, dim])?;
        self.x_embedder.forward(&patches, &mut x_emb, cuda_config)?;

        let mut x_padded = state.slice_mut(BT::XPadded, &[x_padded_len, dim])?;
        pad_with_token_into(&x_emb, &self.x_pad_token, &mut x_padded)?;

        // Pre-computed cos/sin for image tokens.
        let x_cos = state.slice(BT::XCos, &[x_padded_len, half_d])?;
        let x_sin = state.slice(BT::XSin, &[x_padded_len, half_d])?;

        // ── noise_refiner on x (ping-pong XPadded ↔ XPaddedTmp) ──
        let x_final_ty = self.run_block_chain(
            &self.noise_refiner,
            (BT::XPadded, BT::XPaddedTmp),
            &[x_padded_len, dim],
            &x_cos, &x_sin, Some(&adaln_input),
            state, cuda_config,
        )?;

        // ── Unified concat: [x_final | cap_padded (already refined)] ──
        let s_tot = x_padded_len + cap_padded_len;
        {
            let x_final = state.slice(x_final_ty, &[x_padded_len, dim])?;
            let cap_final = state.slice(BT::CapPadded, &[cap_padded_len, dim])?;
            let mut unified = state.slice_mut(BT::Unified, &[s_tot, dim])?;
            concat_seq_into(&x_final, &cap_final, &mut unified)?;
        }

        // Pre-computed unified cos/sin.
        let unified_cos = state.slice(BT::UnifiedCos, &[s_tot, half_d])?;
        let unified_sin = state.slice(BT::UnifiedSin, &[s_tot, half_d])?;

        // ── Main layers (ping-pong Unified ↔ UnifiedTmp) ──
        let unified_final_ty = self.run_block_chain(
            &self.layers,
            (BT::Unified, BT::UnifiedTmp),
            &[s_tot, dim],
            &unified_cos, &unified_sin, Some(&adaln_input),
            state, cuda_config,
        )?;

        // ── Final layer ──
        {
            let unified_final = state.slice(unified_final_ty, &[s_tot, dim])?;
            let image_prefix = unified_final.slice(&[0, 0], &[x_padded_len, dim])?;
            self.final_layer_forward(&image_prefix, &adaln_input, state, cuda_config)?;
        }

        // ── Unpatchify into ImageOut ──
        let final_out = state.slice(BT::FinalOut, &[x_padded_len, self.final_out_dim()])?;
        let out_x = final_out.slice(&[0, 0], &[n, self.final_out_dim()])?;
        let mut image_out = state.slice_mut(
            BT::ImageOut, &[self.config.in_channels, f, h, w],
        )?;
        crate::model::diffusion::z_image::patchify::unpatchify_into(
            &out_x, f, h, w,
            self.config.in_channels, self.patch_size, self.f_patch_size,
            &mut image_out,
        )?;
        Ok(image_out)
    }

    /// `patch_size² * f_patch_size * in_channels` — output channels per
    /// token after the final linear projection, before `unpatchify`.
    #[inline]
    fn final_out_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.f_patch_size * self.config.in_channels
    }

    /// Run a chain of [`DiTBlock`]s, ping-ponging between the two state
    /// slots in `slots = (primary, tmp)`. Returns the `DiffBufferType`
    /// where the chain's final output landed.
    ///
    /// Each block reads from one slot and writes to the other; the role
    /// flips every iteration. The chain's input must live in `slots.0`
    /// before the call. After `n` iterations the output lives in
    /// `slots.0` if `n` is even, otherwise `slots.1`.
    #[allow(clippy::too_many_arguments)]
    fn run_block_chain(
        &self,
        blocks: &[DiTBlock],
        slots: (BT, BT),
        shape: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        adaln_c: Option<&Tensor>,
        state: &mut DitState,
        cuda_config: Option<&OpConfig>,
    ) -> Result<BT> {
        for (i, block) in blocks.iter().enumerate() {
            let (src_ty, dst_ty) = if i % 2 == 0 { slots } else { (slots.1, slots.0) };
            let src = state.slice(src_ty, shape)?;
            let mut dst = state.slice_mut(dst_ty, shape)?;
            block.forward(&src, cos, sin, adaln_c, state, &mut dst, cuda_config)?;
        }
        Ok(if blocks.len() % 2 == 0 { slots.0 } else { slots.1 })
    }

    fn final_layer_forward(
        &self,
        x: &Tensor,          // [S, dim]
        c: &Tensor,          // [adaln_embed_dim = 256]
        state: &mut DitState,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let seq = x.shape()[0];
        let dim = self.config.dim;
        let out_dim = self.final_out_dim();

        // scale = 1 + adaLN_modulation(silu(c))
        let mut c_silu = state.slice_mut(BT::AdalnSilu, &[ADALN_EMBED_DIM])?;
        c_silu.copy_from_on_current_stream(c)?;
        c_silu.silu()?;
        let c_silu_2d = c_silu.view(&[1, ADALN_EMBED_DIM])?;

        let mut scale = state.slice_mut(BT::FinalScale, &[1, dim])?;
        self.final_layer_adaln.forward(&c_silu_2d, &mut scale, cuda_config)?;
        scale += 1.0_f32;

        // norm_final(x) * scale → FinalNormed
        let mut normed = state.slice_mut(BT::FinalNormed, &[seq, dim])?;
        crate::op::layernorm::layernorm(x, &mut normed, self.final_layer_eps)?;
        normed.mul_row(&scale)?;

        // Linear → FinalOut
        let mut out = state.slice_mut(BT::FinalOut, &[seq, out_dim])?;
        self.final_layer_linear.forward(&normed, &mut out, cuda_config)?;
        Ok(())
    }

    /// Move all weights to CUDA.
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.x_embedder.to_cuda(device_id)?;
        self.cap_embedder_norm.to_cuda(device_id)?;
        self.cap_embedder_linear.to_cuda(device_id)?;
        self.t_embedder.mlp1.to_cuda(device_id)?;
        self.t_embedder.mlp2.to_cuda(device_id)?;
        self.x_pad_token = self.x_pad_token.to_cuda(device_id)?;
        self.cap_pad_token = self.cap_pad_token.to_cuda(device_id)?;
        for b in &mut self.noise_refiner { b.to_cuda(device_id)?; }
        for b in &mut self.context_refiner { b.to_cuda(device_id)?; }
        for b in &mut self.layers { b.to_cuda(device_id)?; }
        self.final_layer_linear.to_cuda(device_id)?;
        self.final_layer_adaln.to_cuda(device_id)?;
        // rope_embedder cache stays on CPU (scalar lookup); no upload needed.
        self.device = DeviceType::Cuda(device_id);
        Ok(())
    }
}

// ───────────────────────── Weight loading helpers ─────────────────────────

fn load_tensor(loader: &DiffusersLoader, name: &str, device: DeviceType) -> Result<Tensor> {
    let view = loader.get_tensor(name)?;
    let t = Tensor::from_view_on_cpu(&view)?;
    let t = if device.is_cpu() && t.dtype() != DataType::F32 {
        t.to_dtype(DataType::F32)?
    } else { t };
    t.to_device(device)
}

fn load_linear(
    loader: &DiffusersLoader,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    device: DeviceType,
) -> Result<Matmul> {
    let w = load_tensor(loader, &format!("{}.weight", prefix), device)?;
    if w.shape() != [out_features, in_features] {
        return Err(Error::InvalidArgument(format!(
            "Linear {} weight shape mismatch: expected [{}, {}], got {:?}",
            prefix, out_features, in_features, w.shape()
        )).into());
    }
    let bias = if has_bias && loader.has_tensor(&format!("{}.bias", prefix)) {
        Some(load_tensor(loader, &format!("{}.bias", prefix), device)?)
    } else {
        None
    };
    Ok(Matmul::from(w, bias))
}

/// Load and fuse the per-block `to_q / to_k / to_v` weights into a single
/// `[3*dim, dim]` matmul.
///
/// Z-Image checkpoints ship Q/K/V as three separate `[dim, dim]` weights, but
/// the three projections share the same input (`norm1_x`) so they can be
/// collapsed into one GEMM of shape `[seq, dim] @ [3*dim, dim]^T → [seq,
/// 3*dim]`. Compared with three independent GEMMs this:
///
/// - replaces three kernel launches with one (saving ~15-20 µs/block of host
///   overhead on the 30-block main stack);
/// - reads `norm1_x` from HBM exactly once (~2.2 MB at seq=384 BF16) instead
///   of three times;
/// - gives cuBLASLt a larger N dimension to pick better tiling heuristics.
///
/// The output's columns split naturally as Q=cols[0..dim), K=cols[dim..2*dim),
/// V=cols[2*dim..3*dim). Downstream code runs three `split_cols` kernel
/// launches to copy each column slab into the contiguous `BlkQ` / `BlkK` /
/// `BlkV` buffers, mirroring the LLM-side `Llama3` / `Qwen3` pipeline.
///
/// These checkpoints never ship a bias on Q/K/V; we refuse if one appears so
/// we don't silently drop it.
fn load_fused_qkv_linear(
    loader: &DiffusersLoader,
    prefix: &str,
    dim: usize,
    device: DeviceType,
) -> Result<Matmul> {
    let names = [
        format!("{}.attention.to_q.weight", prefix),
        format!("{}.attention.to_k.weight", prefix),
        format!("{}.attention.to_v.weight", prefix),
    ];
    // Guard: no bias expected on Z-Image attention projections.
    for kind in ["to_q", "to_k", "to_v"] {
        let b = format!("{}.attention.{}.bias", prefix, kind);
        if loader.has_tensor(&b) {
            return Err(Error::InvalidArgument(format!(
                "load_fused_qkv_linear: unexpected bias on {} — refusing to silently drop it",
                b,
            )).into());
        }
    }

    // Stage each weight on CPU first so we can memcpy the three [dim, dim]
    // slabs into a single [3*dim, dim] buffer without needing a
    // cross-device concat kernel. The resulting fused tensor is uploaded
    // to the target device in one go.
    let load_cpu = |name: &str| -> Result<Tensor> {
        let view = loader.get_tensor(name)?;
        Tensor::from_view_on_cpu(&view)
    };
    let w_q = load_cpu(&names[0])?;
    let w_k = load_cpu(&names[1])?;
    let w_v = load_cpu(&names[2])?;
    for (w, name) in [(&w_q, &names[0]), (&w_k, &names[1]), (&w_v, &names[2])] {
        if w.shape() != [dim, dim] {
            return Err(Error::InvalidArgument(format!(
                "load_fused_qkv_linear: {} has shape {:?}, expected [{}, {}]",
                name, w.shape(), dim, dim,
            )).into());
        }
    }

    // All three weights share a dtype (checkpoint BF16 / F16 / F32).
    let src_dtype = w_q.dtype();
    if w_k.dtype() != src_dtype || w_v.dtype() != src_dtype {
        return Err(Error::InvalidArgument(format!(
            "load_fused_qkv_linear: dtype mismatch q={:?} k={:?} v={:?}",
            w_q.dtype(), w_k.dtype(), w_v.dtype(),
        )).into());
    }

    // For CPU execution the pipeline normalises to F32; for CUDA we keep
    // the checkpoint's native dtype (typically BF16).
    let target_dtype = if device.is_cpu() && src_dtype != DataType::F32 {
        DataType::F32
    } else {
        src_dtype
    };
    let w_q = if w_q.dtype() != target_dtype { w_q.to_dtype(target_dtype)? } else { w_q };
    let w_k = if w_k.dtype() != target_dtype { w_k.to_dtype(target_dtype)? } else { w_k };
    let w_v = if w_v.dtype() != target_dtype { w_v.to_dtype(target_dtype)? } else { w_v };

    // Concatenate along dim 0: [Wq; Wk; Wv] → [3*dim, dim]. Memory order
    // matches the native [out_features, in_features] weight layout so the
    // GEMM output's cols naturally split as [Q | K | V].
    let mut fused = Tensor::new(&[3 * dim, dim], target_dtype, DeviceType::Cpu)?;
    let row_bytes = dim * target_dtype.size_in_bytes();
    match target_dtype {
        DataType::F32 => {
            let dst = fused.as_f32_mut()?.as_slice_mut()?;
            dst[0..dim * dim].copy_from_slice(w_q.as_f32()?.as_slice()?);
            dst[dim * dim..2 * dim * dim].copy_from_slice(w_k.as_f32()?.as_slice()?);
            dst[2 * dim * dim..3 * dim * dim].copy_from_slice(w_v.as_f32()?.as_slice()?);
        }
        DataType::BF16 => {
            let dst = fused.as_bf16_mut()?.as_slice_mut()?;
            dst[0..dim * dim].copy_from_slice(w_q.as_bf16()?.as_slice()?);
            dst[dim * dim..2 * dim * dim].copy_from_slice(w_k.as_bf16()?.as_slice()?);
            dst[2 * dim * dim..3 * dim * dim].copy_from_slice(w_v.as_bf16()?.as_slice()?);
        }
        DataType::F16 => {
            let dst = fused.as_f16_mut()?.as_slice_mut()?;
            dst[0..dim * dim].copy_from_slice(w_q.as_f16()?.as_slice()?);
            dst[dim * dim..2 * dim * dim].copy_from_slice(w_k.as_f16()?.as_slice()?);
            dst[2 * dim * dim..3 * dim * dim].copy_from_slice(w_v.as_f16()?.as_slice()?);
        }
        other => return Err(Error::InvalidArgument(format!(
            "load_fused_qkv_linear: unsupported dtype {:?}", other,
        )).into()),
    }
    let _ = row_bytes; // kept for clarity / future stride-aware variants

    let fused = fused.to_device(device)?;
    Ok(Matmul::from(fused, None))
}

fn load_timestep_embedder(
    loader: &DiffusersLoader,
    freq_dim: usize,
    mid: usize,
    out_dim: usize,
    device: DeviceType,
) -> Result<TimestepEmbedder> {
    let mlp1 = load_linear(loader, "t_embedder.mlp.0", freq_dim, mid, true, device)?;
    let mlp2 = load_linear(loader, "t_embedder.mlp.2", mid, out_dim, true, device)?;
    Ok(TimestepEmbedder {
        mlp1,
        mlp2,
        frequency_embedding_size: freq_dim,
    })
}

fn load_dit_block(
    loader: &DiffusersLoader,
    prefix: &str,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    eps: f32,
    modulation: bool,
    device: DeviceType,
) -> Result<DiTBlock> {
    let attention_norm1 = RMSNorm::from(load_tensor(loader, &format!("{}.attention_norm1.weight", prefix), device)?, eps);
    let attention_norm2 = RMSNorm::from(load_tensor(loader, &format!("{}.attention_norm2.weight", prefix), device)?, eps);
    let ffn_norm1 = RMSNorm::from(load_tensor(loader, &format!("{}.ffn_norm1.weight", prefix), device)?, eps);
    let ffn_norm2 = RMSNorm::from(load_tensor(loader, &format!("{}.ffn_norm2.weight", prefix), device)?, eps);

    let to_qkv = load_fused_qkv_linear(loader, prefix, dim, device)?;
    let to_out = load_linear(loader, &format!("{}.attention.to_out.0", prefix), dim, dim, false, device)?;

    let norm_q = RMSNorm::from(load_tensor(loader, &format!("{}.attention.norm_q.weight", prefix), device)?, eps);
    let norm_k = RMSNorm::from(load_tensor(loader, &format!("{}.attention.norm_k.weight", prefix), device)?, eps);

    let w1 = load_linear(loader, &format!("{}.feed_forward.w1", prefix), dim, hidden_dim, false, device)?;
    let w3 = load_linear(loader, &format!("{}.feed_forward.w3", prefix), dim, hidden_dim, false, device)?;
    let w2 = load_linear(loader, &format!("{}.feed_forward.w2", prefix), hidden_dim, dim, false, device)?;

    let adaln_modulation = if modulation {
        Some(load_linear(loader, &format!("{}.adaLN_modulation.0", prefix), ADALN_EMBED_DIM, 4 * dim, true, device)?)
    } else {
        None
    };

    Ok(DiTBlock {
        attention_norm1, attention_norm2, ffn_norm1, ffn_norm2,
        to_qkv, to_out,
        norm_q, norm_k,
        w1, w3, w2,
        adaln_modulation,
        dim, n_heads, head_dim,
        modulation,
    })
}

// ───────────────────────── Tensor helpers ─────────────────────────

/// Write caption pos ids into a pre-allocated [cap_len, 3] I32 CPU slot.
fn fill_cap_pos_ids(dst: &mut Tensor, cap_len: usize) -> Result<()> {
    debug_assert_eq!(dst.device(), DeviceType::Cpu);
    debug_assert_eq!(dst.dtype(), DataType::I32);
    debug_assert_eq!(dst.shape(), &[cap_len, 3]);
    let data = dst.as_i32_mut()?.as_slice_mut()?;
    for i in 0..cap_len {
        data[i * 3] = (i + 1) as i32;
        data[i * 3 + 1] = 0;
        data[i * 3 + 2] = 0;
    }
    Ok(())
}

/// Write image pos ids into a pre-allocated [n+pad_len, 3] I32 CPU slot.
fn fill_image_pos_ids(
    dst: &mut Tensor,
    f_t: usize, h_t: usize, w_t: usize,
    t_base: usize,
    pad_len: usize,
) -> Result<()> {
    let n = f_t * h_t * w_t;
    let total = n + pad_len;
    debug_assert_eq!(dst.device(), DeviceType::Cpu);
    debug_assert_eq!(dst.dtype(), DataType::I32);
    debug_assert_eq!(dst.shape(), &[total, 3]);
    let data = dst.as_i32_mut()?.as_slice_mut()?;
    let mut idx = 0;
    for fi in 0..f_t {
        for hi in 0..h_t {
            for wi in 0..w_t {
                data[idx * 3] = (t_base + fi) as i32;
                data[idx * 3 + 1] = hi as i32;
                data[idx * 3 + 2] = wi as i32;
                idx += 1;
            }
        }
    }
    for i in n..total {
        data[i * 3] = 0;
        data[i * 3 + 1] = 0;
        data[i * 3 + 2] = 0;
    }
    Ok(())
}
