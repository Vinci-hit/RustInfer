//! Z-Image Transformer 2D Model — full S3-DiT denoising backbone.
//!
//! Architecture:
//! ```text
//! patches = patchify(latent)              # [n, patch_in_dim]
//! x = x_embedder(patches)                  # [n, dim]
//! x = pad_with_token(x, x_pad_token)       # [s_img, dim]
//! cap = pad_last_row(cap_feats)            # [s_cap, cap_feat_dim]
//! cap = cap_embedder_norm(cap)             # RMSNorm
//! cap = cap_embedder_linear(cap)           # [s_cap, dim]
//! cap = overwrite_pad_tokens(cap, cap_pad_token)
//! adaln_input = t_embedder(scaled_t)       # [1, 256]
//!
//! x_pos = build_image_pos_ids(...)
//! cap_pos = build_cap_pos_ids(...)
//! (x_cos, x_sin) = rope.embed(x_pos)
//! (cap_cos, cap_sin) = rope.embed(cap_pos)
//!
//! for blk in noise_refiner: x = blk(x, x_cos, x_sin, adaln_input)
//! for blk in context_refiner: cap = blk(cap, cap_cos, cap_sin, None)
//! unified = concat([x, cap])
//! (u_cos, u_sin) = concat([x_cos, cap_cos], [x_sin, cap_sin])
//! for blk in main_layers: unified = blk(unified, u_cos, u_sin, adaln_input)
//!
//! image_part = unified[:s_img]
//! out = final_layer(image_part, adaln_input)  # adaLN scale + LN + linear
//! velocity = unpatchify(out[:n], H, W)
//! ```

use std::path::Path;

use crate::domain::ports::{OpResult, OpError, OpBackend};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::io::SafetensorsReader;
use crate::models::layers::{Linear, RMSNorm, LayerNorm};
use crate::models::loader::WeightLoader;

use super::dit_block::DiTBlock;
use super::rope_3d::{RopeEmbedder3D, fill_cap_pos_ids, fill_image_pos_ids};
use super::state::{DitState, ADALN_EMBED_DIM, SEQ_MULTI_OF, T_FREQ_DIM, T_EMBEDDER_MID};
use super::timestep_embedder::TimestepEmbedder;

/// One-shot dump latch for the transformer-level intermediates (latent_5d,
/// patches, x_emb). Only the very first denoise step writes; later steps
/// run as usual.
static DUMP_TRANSFORMER_DONE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Take a contiguous 2D view of `src` shaped `[rows, cols]`. Caller asserts
/// `rows * cols ≤ src.numel()`.
fn vp2<T: Dtype>(src: &Tensor<T, Cuda>, rows: usize, cols: usize) -> OpResult<Tensor<T, Cuda>> {
    if rows * cols > src.numel() {
        return Err(OpError::Shape(format!(
            "vp2: requested {}*{}={} > storage numel {}",
            rows, cols, rows * cols, src.numel(),
        )));
    }
    Ok(src.view_raw(
        Shape::from_slice(&[rows, cols]),
        Shape::from_slice(&[cols, 1]).contiguous_strides(),
        src.offset_elems(),
        true,
    ))
}

/// 4D variant of vp2 for `[c, f, h, w]`-style image views.
fn vp4<T: Dtype>(src: &Tensor<T, Cuda>, a: usize, b: usize, c: usize, d: usize) -> OpResult<Tensor<T, Cuda>> {
    if a * b * c * d > src.numel() {
        return Err(OpError::Shape(format!(
            "vp4: requested {}*{}*{}*{}={} > storage numel {}",
            a, b, c, d, a * b * c * d, src.numel(),
        )));
    }
    Ok(src.view_raw(
        Shape::from_slice(&[a, b, c, d]),
        Shape::from_slice(&[b * c * d, c * d, d, 1]).contiguous_strides(),
        src.offset_elems(),
        true,
    ))
}

#[derive(Debug, Clone)]
pub struct ZImageTransformerConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub in_channels: usize,
    pub cap_feat_dim: usize,
    pub patch_size: usize,
    pub f_patch_size: usize,
    pub axes_dims: [usize; 3],
    pub axes_lens: [usize; 3],
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub t_scale: f32,
    pub qk_norm: bool,
}

impl ZImageTransformerConfig {
    pub fn from_json<P: AsRef<Path>>(path: P) -> OpResult<Self> {
        let s = std::fs::read_to_string(&path)
            .map_err(|e| OpError::Kernel(format!("transformer config: {}", e)))?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| OpError::Kernel(format!("transformer config parse: {}", e)))?;
        let get_u = |k: &str| -> OpResult<usize> {
            v[k].as_u64()
                .ok_or_else(|| OpError::Kernel(format!("missing {} in transformer config", k)))
                .map(|x| x as usize)
        };
        let get_f = |k: &str| -> OpResult<f32> {
            v[k].as_f64()
                .ok_or_else(|| OpError::Kernel(format!("missing {} in transformer config", k)))
                .map(|x| x as f32)
        };
        let arr = |k: &str| -> OpResult<Vec<usize>> {
            v[k].as_array()
                .ok_or_else(|| OpError::Kernel(format!("missing {} in transformer config", k)))?
                .iter().map(|x| {
                    x.as_u64().map(|x| x as usize)
                        .ok_or_else(|| OpError::Kernel(format!("bad entry in {}", k)))
                }).collect()
        };

        let axes_dims_v = arr("axes_dims")?;
        let axes_lens_v = arr("axes_lens")?;
        let all_patch = arr("all_patch_size")?;
        let all_f_patch = arr("all_f_patch_size")?;

        let dim = get_u("dim")?;
        let n_heads = get_u("n_heads")?;
        let head_dim = dim / n_heads;
        // Z-Image config has no `intermediate_size`; diffusers uses `dim/3*8`.
        let intermediate_size = dim / 3 * 8;

        Ok(Self {
            dim,
            n_layers: get_u("n_layers")?,
            n_refiner_layers: get_u("n_refiner_layers")?,
            n_heads,
            n_kv_heads: get_u("n_kv_heads")?,
            head_dim,
            intermediate_size,
            in_channels: get_u("in_channels")?,
            cap_feat_dim: get_u("cap_feat_dim")?,
            patch_size: all_patch[0],
            f_patch_size: all_f_patch[0],
            axes_dims: [axes_dims_v[0], axes_dims_v[1], axes_dims_v[2]],
            axes_lens: [axes_lens_v[0], axes_lens_v[1], axes_lens_v[2]],
            norm_eps: v["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: get_f("rope_theta")?,
            t_scale: get_f("t_scale")?,
            qk_norm: v["qk_norm"].as_bool().unwrap_or(true),
        })
    }
}

pub struct ZImageTransformer<T: Dtype, D: OpBackend> {
    pub config: ZImageTransformerConfig,
    // Embeddings.
    pub x_embedder: Linear<T, D>,
    pub cap_embedder_norm: RMSNorm<T, D>,
    pub cap_embedder_linear: Linear<T, D>,
    pub t_embedder: TimestepEmbedder<T, D>,

    // Pad tokens (small `[1, dim]` / `[1, cap_feat_dim]` tensors).
    pub x_pad_token: Tensor<T, D>,
    pub cap_pad_token: Tensor<T, D>,

    // Refiners + main layers.
    pub noise_refiner: Vec<DiTBlock<T, D>>,
    pub context_refiner: Vec<DiTBlock<T, D>>,
    pub layers: Vec<DiTBlock<T, D>>,

    // Final layer: layer-norm + adaLN-scale + linear → patch space.
    pub final_norm: LayerNorm<T, D>,
    pub final_adaln: Linear<T, D>,        // [dim, ADALN_EMBED_DIM]
    pub final_proj: Linear<T, D>,         // [final_out_dim, dim]

    pub rope: RopeEmbedder3D,
}

impl<T: Dtype> ZImageTransformer<T, Cuda> {
    /// Load Z-Image transformer from a diffusers `transformer/` directory.
    /// Expected layout:
    /// - `config.json`
    /// - `diffusion_pytorch_model.safetensors.index.json` + sharded files
    pub fn from_pretrained<P: AsRef<Path>>(transformer_dir: P, device: &Cuda) -> OpResult<Self> {
        let dir = transformer_dir.as_ref();
        let config = ZImageTransformerConfig::from_json(dir.join("config.json"))?;
        let reader = SafetensorsReader::open(dir)
            .map_err(|e| OpError::Kernel(format!("transformer: {}", e)))?;
        let loader = WeightLoader::new(&reader);

        let dim = config.dim;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let in_channels = config.in_channels;
        let cap_feat_dim = config.cap_feat_dim;
        let p = config.patch_size;
        let pf = config.f_patch_size;
        let patch_in_dim = pf * p * p * in_channels;
        let final_out_dim = p * p * pf * in_channels;
        let patch_key = format!("{}-{}", p, pf);

        // x_embedder: per-patch_key linear `all_x_embedder.{key}.weight/bias`.
        let x_embedder = loader.load_linear::<T, Cuda>(
            &format!("all_x_embedder.{}.weight", patch_key),
            Some(&format!("all_x_embedder.{}.bias", patch_key)),
            device,
        )?;

        // cap_embedder is RMSNorm + Linear: `cap_embedder.0.weight` (RMSNorm),
        // `cap_embedder.1.{weight,bias}` (Linear).
        let cap_embedder_norm: RMSNorm<T, Cuda> = loader.load_rmsnorm::<T, Cuda>(
            "cap_embedder.0.weight", device, config.norm_eps,
        )?;
        let cap_embedder_linear = loader.load_linear::<T, Cuda>(
            "cap_embedder.1.weight",
            Some("cap_embedder.1.bias"),
            device,
        )?;

        // t_embedder: 2-layer MLP. Naming in diffusers:
        //   t_embedder.mlp.0.{weight,bias}
        //   t_embedder.mlp.2.{weight,bias}
        let t_emb_mlp1 = loader.load_linear::<T, Cuda>(
            "t_embedder.mlp.0.weight",
            Some("t_embedder.mlp.0.bias"),
            device,
        )?;
        let t_emb_mlp2 = loader.load_linear::<T, Cuda>(
            "t_embedder.mlp.2.weight",
            Some("t_embedder.mlp.2.bias"),
            device,
        )?;
        let t_embedder = TimestepEmbedder {
            mlp1: t_emb_mlp1,
            mlp2: t_emb_mlp2,
            frequency_embedding_size: T_FREQ_DIM,
        };

        // Pad tokens.
        let x_pad_token = loader.load_tensor::<T, Cuda>("x_pad_token", device)?;
        let cap_pad_token = loader.load_tensor::<T, Cuda>("cap_pad_token", device)?;

        // Refiners + main layers.
        let mut noise_refiner = Vec::with_capacity(config.n_refiner_layers);
        for i in 0..config.n_refiner_layers {
            noise_refiner.push(load_dit_block::<T>(
                &loader, &format!("noise_refiner.{}", i),
                dim, n_heads, head_dim, intermediate_size, config.norm_eps, true, device,
            )?);
        }
        let mut context_refiner = Vec::with_capacity(config.n_refiner_layers);
        for i in 0..config.n_refiner_layers {
            context_refiner.push(load_dit_block::<T>(
                &loader, &format!("context_refiner.{}", i),
                dim, n_heads, head_dim, intermediate_size, config.norm_eps, false, device,
            )?);
        }
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(load_dit_block::<T>(
                &loader, &format!("layers.{}", i),
                dim, n_heads, head_dim, intermediate_size, config.norm_eps, true, device,
            )?);
        }

        // Final layer. diffusers `LayerNorm(elementwise_affine=False)` →
        // no weight/bias on disk. We emulate with unit weight + zero bias.
        let final_norm_w_host: Vec<u8> = vec![0u8; dim * T::SIZE_BYTES];
        // Fill with 1.0 in T-dtype.
        let mut final_norm_w_buf = final_norm_w_host;
        for i in 0..dim {
            let off = i * T::SIZE_BYTES;
            match T::DATA_TYPE {
                crate::domain::types::DataType::F32 => {
                    final_norm_w_buf[off..off + 4].copy_from_slice(&1.0_f32.to_le_bytes());
                }
                crate::domain::types::DataType::BF16 => {
                    final_norm_w_buf[off..off + 2].copy_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
                }
                crate::domain::types::DataType::F16 => {
                    final_norm_w_buf[off..off + 2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                }
                other => return Err(OpError::Kernel(format!("final_norm dtype {:?}", other))),
            }
        }
        let final_norm_w = Tensor::<T, Cuda>::from_host_bytes(
            &final_norm_w_buf, Shape::from_slice(&[dim]), device,
        )?;
        let final_norm_b_buf = vec![0u8; dim * T::SIZE_BYTES];
        let final_norm_b = Tensor::<T, Cuda>::from_host_bytes(
            &final_norm_b_buf, Shape::from_slice(&[dim]), device,
        )?;
        let final_norm = LayerNorm::new(final_norm_w, final_norm_b, 1e-6);
        let final_adaln = loader.load_linear::<T, Cuda>(
            &format!("all_final_layer.{}.adaLN_modulation.1.weight", patch_key),
            Some(&format!("all_final_layer.{}.adaLN_modulation.1.bias", patch_key)),
            device,
        )?;
        let final_proj = loader.load_linear::<T, Cuda>(
            &format!("all_final_layer.{}.linear.weight", patch_key),
            Some(&format!("all_final_layer.{}.linear.bias", patch_key)),
            device,
        )?;

        // RoPE.
        let rope = RopeEmbedder3D::new(
            config.axes_dims, config.axes_lens, config.rope_theta as f64,
        )?;

        // Sanity-check: in_channels in config matches latent.
        let _ = patch_in_dim;
        let _ = final_out_dim;

        Ok(Self {
            config,
            x_embedder, cap_embedder_norm, cap_embedder_linear, t_embedder,
            x_pad_token, cap_pad_token,
            noise_refiner, context_refiner, layers,
            final_norm, final_adaln, final_proj,
            rope,
        })
    }

    /// Per-step transformer forward (eager mode), CUDA-only.
    ///
    /// - `latent_5d`: `[16, 1, lh, lw]` device tensor (current sample).
    /// - `cap_feats`: `[cap_len, cap_feat_dim]`. Already on the same device.
    /// - `t_value_scaled`: `(1 - timestep/T) * t_scale`.
    /// - `state`: pre-allocated workspace.
    ///
    /// Writes the predicted velocity into `state.image_out` (`[16,1,lh,lw]`).
    /// Returns a clone-view for ergonomics.
    pub fn forward(
        &self,
        latent_5d: &Tensor<T, Cuda>,
        cap_feats: &Tensor<T, Cuda>,
        t_value_scaled: f32,
        state: &mut DitState<T, Cuda>,
    ) -> OpResult<Tensor<T, Cuda>> {
        let dim = self.config.dim;
        let cap_feat_dim = self.config.cap_feat_dim;
        let p = self.config.patch_size;
        let pf = self.config.f_patch_size;

        // ── Shapes ──
        let ls = latent_5d.shape().as_slice();
        if ls.len() != 4 || ls[0] != self.config.in_channels {
            return Err(OpError::Shape(format!(
                "transformer.forward: latent_5d expected [{},1,H,W], got {:?}",
                self.config.in_channels, ls,
            )));
        }
        let (f, h, w) = (ls[1], ls[2], ls[3]);
        let f_t = f / pf;
        let h_t = h / p;
        let w_t = w / p;
        let n = f_t * h_t * w_t;
        let s_img = round_up(n, SEQ_MULTI_OF);
        let cap_ori = cap_feats.shape().as_slice()[0];
        let s_cap = round_up(cap_ori, SEQ_MULTI_OF);
        let s_total = s_img + s_cap;
        let half_d = self.config.head_dim / 2;
        let patch_in_dim = pf * p * p * self.config.in_channels;

        // ── 1. Timestep embedding → adaln_input ──
        let mut t_freq = vp2(&state.t_freq, 1, T_FREQ_DIM)?;
        let mut t_hidden = vp2(&state.t_hidden, 1, T_EMBEDDER_MID)?;
        let mut t_out = vp2(&state.t_out, 1, ADALN_EMBED_DIM)?;
        self.t_embedder.forward_host(t_value_scaled, &mut t_freq, &mut t_hidden, &mut t_out)?;
        let mut adaln_input = vp2(&state.adaln_input, 1, ADALN_EMBED_DIM)?;
        adaln_input.copy_from(&t_out)?;

        // ── 2. Patchify latent ──
        let mut patches = vp2(&state.patches, n, patch_in_dim)?;
        super::patchify::patchify_into(latent_5d, p, pf, &mut patches)?;
        let do_dump = std::env::var("RUSTINFER_DUMP").is_ok()
            && !DUMP_TRANSFORMER_DONE.swap(true, std::sync::atomic::Ordering::SeqCst);
        if do_dump {
            super::dit_block::dump_tensor("step0_latent_5d_in", latent_5d);
            super::dit_block::dump_tensor("step0_patches", &patches);
        }

        // ── 3. x_embedder + pad ──
        let mut x_emb = vp2(&state.x_emb, n, dim)?;
        self.x_embedder.forward(&patches, &mut x_emb)?;
        if do_dump {
            super::dit_block::dump_tensor("step0_x_emb", &x_emb);
        }
        let mut x_padded = vp2(&state.x_padded, s_img, dim)?;
        Cuda::pad_with_token(&x_emb, &self.x_pad_token, &mut x_padded)?;

        // ── 4. cap_embedder + pad + stamp ──
        let mut cap_padded_feats = vp2(&state.cap_feats_padded, s_cap, cap_feat_dim)?;
        Cuda::pad_last_row(cap_feats, &mut cap_padded_feats)?;
        let mut cap_normed = vp2(&state.cap_normed, s_cap, cap_feat_dim)?;
        self.cap_embedder_norm.forward(&cap_padded_feats, &mut cap_normed)?;
        let mut cap_emb = vp2(&state.cap_emb, s_cap, dim)?;
        self.cap_embedder_linear.forward(&cap_normed, &mut cap_emb)?;
        let mut cap_padded = vp2(&state.cap_padded, s_cap, dim)?;
        cap_padded.copy_from(&cap_emb)?;
        Cuda::overwrite_pad_tokens_inplace(&mut cap_padded, &self.cap_pad_token, cap_ori)?;

        // ── 5. RoPE caches ──
        let cap_pos_ids = fill_cap_pos_ids(s_cap);
        let img_pos_ids = fill_image_pos_ids(f_t, h_t, w_t, s_cap + 1, s_img - n);
        let mut x_cos = vp2(&state.x_cos, s_img, half_d)?;
        let mut x_sin = vp2(&state.x_sin, s_img, half_d)?;
        let mut cap_cos = vp2(&state.cap_cos, s_cap, half_d)?;
        let mut cap_sin = vp2(&state.cap_sin, s_cap, half_d)?;
        self.rope.embed_into_cuda(&img_pos_ids, s_img, &mut x_cos, &mut x_sin)?;
        self.rope.embed_into_cuda(&cap_pos_ids, s_cap, &mut cap_cos, &mut cap_sin)?;
        if do_dump {
            super::dit_block::dump_tensor("step0_x_cos", &x_cos);
            super::dit_block::dump_tensor("step0_x_sin", &x_sin);
            // Also dump first few image pos_ids for sanity.
            eprintln!("[dump] img_pos_ids first 12 = {:?}", &img_pos_ids[..12]);
            eprintln!("[dump] s_cap = {}, s_img = {}, n = {}", s_cap, s_img, n);
        }
        if do_dump {
            super::dit_block::dump_tensor("step0_x_cos", &x_cos);
            super::dit_block::dump_tensor("step0_x_sin", &x_sin);
            // Also dump first few image pos_ids for sanity.
            eprintln!("[dump] img_pos_ids first 12 = {:?}", &img_pos_ids[..12]);
            eprintln!("[dump] s_cap = {}, s_img = {}, n = {}", s_cap, s_img, n);
        }

        // ── 6. noise_refiner on x ──
        run_block_chain::<T>(
            &self.noise_refiner,
            &mut state.x_padded, &mut state.x_padded_tmp,
            s_img, dim,
            &x_cos, &x_sin, Some(&adaln_input),
            &mut state.block_scratch,
        )?;
        let x_final_in_padded = self.noise_refiner.len() % 2 == 0;

        // ── 7. context_refiner on cap (no modulation) ──
        run_block_chain::<T>(
            &self.context_refiner,
            &mut state.cap_padded, &mut state.cap_padded_tmp,
            s_cap, dim,
            &cap_cos, &cap_sin, None,
            &mut state.block_scratch,
        )?;
        let cap_final_in_padded = self.context_refiner.len() % 2 == 0;

        // ── 8. Build unified [x | cap] + (cos, sin) ──
        let x_view = if x_final_in_padded {
            vp2(&state.x_padded, s_img, dim)?
        } else {
            vp2(&state.x_padded_tmp, s_img, dim)?
        };
        let cap_view = if cap_final_in_padded {
            vp2(&state.cap_padded, s_cap, dim)?
        } else {
            vp2(&state.cap_padded_tmp, s_cap, dim)?
        };
        let mut unified = vp2(&state.unified, s_total, dim)?;
        Cuda::concat_seq(&x_view, &cap_view, &mut unified)?;
        let mut u_cos = vp2(&state.unified_cos, s_total, half_d)?;
        let mut u_sin = vp2(&state.unified_sin, s_total, half_d)?;
        Cuda::concat_seq(&x_cos, &cap_cos, &mut u_cos)?;
        Cuda::concat_seq(&x_sin, &cap_sin, &mut u_sin)?;

        // ── 9. Main layers ──
        run_block_chain::<T>(
            &self.layers,
            &mut state.unified, &mut state.unified_tmp,
            s_total, dim,
            &u_cos, &u_sin, Some(&adaln_input),
            &mut state.block_scratch,
        )?;
        let main_in_unified = self.layers.len() % 2 == 0;

        // ── 10. Final layer (image part only) ──
        let unified_full = if main_in_unified {
            vp2(&state.unified, s_total, dim)?
        } else {
            vp2(&state.unified_tmp, s_total, dim)?
        };
        // First s_img rows of the contiguous unified tensor = image part, also contiguous.
        let image_part = vp2(&unified_full, s_img, dim)?;
        let final_out_dim = p * p * pf * self.config.in_channels;
        let mut final_normed = vp2(&state.final_normed, s_img, dim)?;
        // adaLN: scale = 1 + final_adaln(silu(adaln_input))
        let mut adaln_silu = vp2(&state.block_scratch.adaln_silu, 1, ADALN_EMBED_DIM)?;
        adaln_silu.copy_from(&adaln_input)?;
        Cuda::silu_inplace_diff(&mut adaln_silu)?;
        let mut scale = vp2(&state.final_scale, 1, dim)?;
        self.final_adaln.forward(&adaln_silu, &mut scale)?;
        Cuda::scalar_add_inplace(&mut scale, 1.0)?;
        // norm + scale.
        self.final_norm.forward(&image_part, &mut final_normed)?;
        Cuda::broadcast_mul_inplace(&mut final_normed, &scale)?;
        // Project to patch space.
        let mut final_out = vp2(&state.final_out, s_img, final_out_dim)?;
        self.final_proj.forward(&final_normed, &mut final_out)?;

        // ── 11. Unpatchify back to [C, F, H, W] ──
        let valid_rows = vp2(&final_out, n, final_out_dim)?;
        let mut image_out = vp4(&state.image_out, self.config.in_channels, f, h, w)?;
        super::patchify::unpatchify_into(
            &valid_rows, f, h, w, self.config.in_channels, p, pf,
            &mut image_out,
        )?;
        Ok(image_out)
    }
}

/// Load a single DiT block from a `{prefix}` (e.g. `layers.5`,
/// `noise_refiner.0`, `context_refiner.1`).
///
/// Diffusers naming under `{prefix}`:
///   .attention.norm{1,2}.weight   (RMSNorm)
///   .ffn.norm{1,2}.weight         (RMSNorm)
///   .attention.{to_q,to_k,to_v,to_out}.weight   (Linear, no bias)
///   .attention.{q,k}_norm.weight                 (RMSNorm, head_dim)
///   .ffn.w{1,3}.weight (Linear no bias) → fused into to_qkv-style Linear
///                                          for our DiTBlock layout
///   .ffn.w2.weight                               (Linear, no bias)
///   .adaLN_modulation.1.{weight,bias}            (Linear, modulation only)
fn load_dit_block<T: Dtype>(
    loader: &WeightLoader,
    prefix: &str,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    norm_eps: f32,
    modulation: bool,
    device: &Cuda,
) -> OpResult<DiTBlock<T, Cuda>> {
    // Norms.
    let attention_norm1 = loader.load_rmsnorm::<T, Cuda>(
        &format!("{}.attention_norm1.weight", prefix), device, norm_eps,
    )?;
    let attention_norm2 = loader.load_rmsnorm::<T, Cuda>(
        &format!("{}.attention_norm2.weight", prefix), device, norm_eps,
    )?;
    let ffn_norm1 = loader.load_rmsnorm::<T, Cuda>(
        &format!("{}.ffn_norm1.weight", prefix), device, norm_eps,
    )?;
    let ffn_norm2 = loader.load_rmsnorm::<T, Cuda>(
        &format!("{}.ffn_norm2.weight", prefix), device, norm_eps,
    )?;

    // qk-norm (per-head). Diffusers names: `attention.norm_q/norm_k.weight`.
    let norm_q = loader.load_rmsnorm::<T, Cuda>(
        &format!("{}.attention.norm_q.weight", prefix), device, norm_eps,
    )?;
    let norm_k = loader.load_rmsnorm::<T, Cuda>(
        &format!("{}.attention.norm_k.weight", prefix), device, norm_eps,
    )?;

    // Fused QKV: stack to_q/to_k/to_v ([dim, dim] each) → [3*dim, dim].
    let to_qkv = load_fused_qkv_dit::<T>(loader, prefix, dim, device)?;
    // to_out is `attention.to_out.0.weight` (the `.0` indexes a Sequential).
    let to_out = loader.load_linear::<T, Cuda>(
        &format!("{}.attention.to_out.0.weight", prefix), None, device,
    )?;

    // FFN under `feed_forward.{w1,w2,w3}.weight`.
    let w1 = loader.load_linear::<T, Cuda>(
        &format!("{}.feed_forward.w1.weight", prefix), None, device,
    )?;
    let w3 = loader.load_linear::<T, Cuda>(
        &format!("{}.feed_forward.w3.weight", prefix), None, device,
    )?;
    let w2 = loader.load_linear::<T, Cuda>(
        &format!("{}.feed_forward.w2.weight", prefix), None, device,
    )?;

    // AdaLN modulation (optional). diffusers uses `adaLN_modulation.0` (Linear).
    let adaln_modulation = if modulation {
        Some(loader.load_linear::<T, Cuda>(
            &format!("{}.adaLN_modulation.0.weight", prefix),
            Some(&format!("{}.adaLN_modulation.0.bias", prefix)),
            device,
        )?)
    } else {
        None
    };

    Ok(DiTBlock {
        attention_norm1, attention_norm2,
        ffn_norm1, ffn_norm2,
        to_qkv, to_out,
        norm_q, norm_k,
        w1, w3, w2,
        adaln_modulation,
        dim, n_heads, head_dim,
        modulation,
    })
}

/// Concatenate `{prefix}.attention.{to_q,to_k,to_v}.weight` along dim 0 into
/// a fused `[3*dim, dim]` Linear (no bias). All three are `[dim, dim]`.
fn load_fused_qkv_dit<T: Dtype>(
    loader: &WeightLoader,
    prefix: &str,
    dim: usize,
    device: &Cuda,
) -> OpResult<Linear<T, Cuda>> {
    use crate::domain::types::DataType;
    let names = [
        format!("{}.attention.to_q.weight", prefix),
        format!("{}.attention.to_k.weight", prefix),
        format!("{}.attention.to_v.weight", prefix),
    ];
    // Read each [dim, dim] view.
    let mut host = vec![0u8; 3 * dim * dim * T::SIZE_BYTES];
    for (slot, name) in names.iter().enumerate() {
        let view = loader.read_view(name)
            .map_err(|e| OpError::Kernel(format!("{}: {}", name, e)))?;
        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() != 2 || shape[0] != dim || shape[1] != dim {
            return Err(OpError::Shape(format!(
                "fused_qkv_dit: {} has shape {:?}, expected [{}, {}]",
                name, shape, dim, dim,
            )));
        }
        let src = view.data();
        let src_dt = match view.dtype() {
            safetensors::Dtype::F32 => DataType::F32,
            safetensors::Dtype::F16 => DataType::F16,
            safetensors::Dtype::BF16 => DataType::BF16,
            safetensors::Dtype::I32 => DataType::I32,
            safetensors::Dtype::I8 => DataType::I8,
            other => return Err(OpError::Kernel(format!("unsupported dtype: {:?}", other))),
        };
        let row_off = slot * dim;
        let dst_off = row_off * dim * T::SIZE_BYTES;
        let numel = dim * dim;
        if src_dt == T::DATA_TYPE {
            let n = numel * T::SIZE_BYTES;
            host[dst_off..dst_off + n].copy_from_slice(&src[..n]);
        } else {
            unsafe {
                crate::models::loader::cast_bytes_pub(
                    src, src_dt,
                    host.as_mut_ptr().add(dst_off),
                    T::DATA_TYPE,
                    numel,
                );
            }
        }
    }
    let fused = Tensor::<T, Cuda>::from_host_bytes(
        &host, Shape::from_slice(&[3 * dim, dim]), device,
    )?;
    Ok(Linear::new(fused, None))
}

/// Run a chain of DiT blocks, ping-ponging between two slot tensors.
fn run_block_chain<T: Dtype>(
    blocks: &[DiTBlock<T, Cuda>],
    primary: &mut Tensor<T, Cuda>,
    tmp: &mut Tensor<T, Cuda>,
    seq: usize,
    dim: usize,
    cos: &Tensor<f32, Cuda>,
    sin: &Tensor<f32, Cuda>,
    adaln: Option<&Tensor<T, Cuda>>,
    scratch: &mut super::dit_block::DiTBlockScratch<T, Cuda>,
) -> OpResult<()> {
    for (i, block) in blocks.iter().enumerate() {
        let from_primary = i % 2 == 0;
        let (src, dst) = if from_primary {
            (&*primary, &mut *tmp)
        } else {
            (&*tmp, &mut *primary)
        };
        let src_view = vp2(src, seq, dim)?;
        let mut dst_view = vp2(dst, seq, dim)?;
        block.forward(&src_view, cos, sin, adaln, scratch, &mut dst_view)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layers::{Linear, RMSNorm, LayerNorm};
    use crate::models::diffusion::state::ZImageCapacity;

    fn make_linear<T: Dtype>(out: usize, in_: usize, cuda: &Cuda, seed: u64) -> Linear<T, Cuda>
    where T: Dtype, Tensor<T, Cuda>: Sized
    {
        // For simplicity in this smoke test we always use f32 weights.
        unreachable!("only specialized below");
    }

    fn rand_linear_f32(out: usize, in_: usize, cuda: &Cuda, seed: u64) -> Linear<f32, Cuda> {
        let w: Tensor<f32, Cuda> = Tensor::randn([out, in_], cuda, Some(seed)).unwrap();
        Linear::new(w, None)
    }

    fn unit_rmsnorm_f32(d: usize, cuda: &Cuda) -> RMSNorm<f32, Cuda> {
        let w: Tensor<f32, Cuda> = Tensor::from_host_slice(&vec![1.0_f32; d], [d], cuda).unwrap();
        RMSNorm::new(w, 1e-5)
    }

    fn unit_layernorm_f32(d: usize, cuda: &Cuda) -> LayerNorm<f32, Cuda> {
        let w: Tensor<f32, Cuda> = Tensor::from_host_slice(&vec![1.0_f32; d], [d], cuda).unwrap();
        let b: Tensor<f32, Cuda> = Tensor::from_host_slice(&vec![0.0_f32; d], [d], cuda).unwrap();
        LayerNorm::new(w, b, 1e-6)
    }

    fn make_block_f32(
        cuda: &Cuda, dim: usize, n_heads: usize, hidden: usize, modulation: bool, seed: u64,
    ) -> DiTBlock<f32, Cuda> {
        let head_dim = dim / n_heads;
        DiTBlock {
            attention_norm1: unit_rmsnorm_f32(dim, cuda),
            attention_norm2: unit_rmsnorm_f32(dim, cuda),
            ffn_norm1: unit_rmsnorm_f32(dim, cuda),
            ffn_norm2: unit_rmsnorm_f32(dim, cuda),
            to_qkv: rand_linear_f32(3 * dim, dim, cuda, seed),
            to_out: rand_linear_f32(dim, dim, cuda, seed + 1),
            norm_q: unit_rmsnorm_f32(head_dim, cuda),
            norm_k: unit_rmsnorm_f32(head_dim, cuda),
            w1: rand_linear_f32(hidden, dim, cuda, seed + 2),
            w3: rand_linear_f32(hidden, dim, cuda, seed + 3),
            w2: rand_linear_f32(dim, hidden, cuda, seed + 4),
            adaln_modulation: if modulation {
                Some(rand_linear_f32(4 * dim, ADALN_EMBED_DIM, cuda, seed + 5))
            } else {
                None
            },
            dim, n_heads, head_dim, modulation,
        }
    }

    /// End-to-end shape smoke test on a tiny ZImageTransformer:
    /// dim=64, n_heads=4, head_dim=16, n_layers=2, n_refiner=2.
    /// Verifies all stages compose and the output is finite.
    #[test]
    fn transformer_forward_smoke_f32() {
        let cuda = Cuda::new(0).unwrap();
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let cap_feat_dim = 32;
        let in_channels = 16;
        let patch_size = 2;
        let f_patch = 1;
        let hidden = 128;
        let n_layers = 2;
        let n_refiner = 2;
        // Latent 4x4 → patch 2x2 → 4 tokens, padded to SEQ_MULTI_OF=128.
        let lh = 4; let lw = 4;
        let cap_len = 8; // < SEQ_MULTI_OF; padded to 128.

        // Build config + transformer.
        let cfg = ZImageTransformerConfig {
            dim, n_layers, n_refiner_layers: n_refiner,
            n_heads, n_kv_heads: n_heads,
            head_dim,
            intermediate_size: hidden,
            in_channels, cap_feat_dim,
            patch_size, f_patch_size: f_patch,
            axes_dims: [4, 6, 6],     // 4+6+6=16=head_dim
            axes_lens: [256, 32, 32], // covers cap+1 (>128+1) and lh/p, lw/p
            norm_eps: 1e-5,
            rope_theta: 256.0,
            t_scale: 1000.0,
            qk_norm: true,
        };
        let patch_in_dim = f_patch * patch_size * patch_size * in_channels;

        let x_pad: Tensor<f32, Cuda> = Tensor::zeros([1, dim], &cuda).unwrap();
        let cap_pad: Tensor<f32, Cuda> = Tensor::zeros([1, dim], &cuda).unwrap();
        let t_emb = TimestepEmbedder {
            mlp1: rand_linear_f32(T_EMBEDDER_MID, T_FREQ_DIM, &cuda, 100),
            mlp2: rand_linear_f32(ADALN_EMBED_DIM, T_EMBEDDER_MID, &cuda, 101),
            frequency_embedding_size: T_FREQ_DIM,
        };
        let xform = ZImageTransformer {
            config: cfg.clone(),
            x_embedder: rand_linear_f32(dim, patch_in_dim, &cuda, 200),
            cap_embedder_norm: unit_rmsnorm_f32(cap_feat_dim, &cuda),
            cap_embedder_linear: rand_linear_f32(dim, cap_feat_dim, &cuda, 201),
            t_embedder: t_emb,
            x_pad_token: x_pad,
            cap_pad_token: cap_pad,
            noise_refiner: (0..n_refiner).map(|i| make_block_f32(&cuda, dim, n_heads, hidden, true, 300 + i as u64 * 10)).collect(),
            context_refiner: (0..n_refiner).map(|i| make_block_f32(&cuda, dim, n_heads, hidden, false, 400 + i as u64 * 10)).collect(),
            layers: (0..n_layers).map(|i| make_block_f32(&cuda, dim, n_heads, hidden, true, 500 + i as u64 * 10)).collect(),
            final_norm: unit_layernorm_f32(dim, &cuda),
            final_adaln: rand_linear_f32(dim, ADALN_EMBED_DIM, &cuda, 600),
            final_proj: rand_linear_f32(patch_size * patch_size * f_patch * in_channels, dim, &cuda, 601),
            rope: RopeEmbedder3D::new(cfg.axes_dims, cfg.axes_lens, cfg.rope_theta as f64).unwrap(),
        };

        let cap = ZImageCapacity { max_height: lh * 8, max_width: lw * 8, max_cap_len: cap_len };
        let spec = super::super::state::DitShapeSpec {
            dim, n_heads, head_dim,
            hidden_dim: hidden,
            cap_feat_dim,
            patch_size, f_patch_size: f_patch,
            patch_in_dim,
            final_out_dim: patch_size * patch_size * f_patch * in_channels,
            capacity: cap,
        };
        let mut state: DitState<f32, Cuda> = DitState::new(spec, &cuda).unwrap();

        // Random latent + cap_feats.
        let latent: Tensor<f32, Cuda> = Tensor::randn([in_channels, 1, lh, lw], &cuda, Some(7)).unwrap();
        let cap_feats: Tensor<f32, Cuda> = Tensor::randn([cap_len, cap_feat_dim], &cuda, Some(8)).unwrap();
        let out = xform.forward(&latent, &cap_feats, 0.5, &mut state).unwrap();
        let v = out.to_host_vec().unwrap();
        // Output shape correct.
        assert_eq!(out.shape().as_slice(), &[in_channels, 1, lh, lw]);
        // Output count correct (random weights → values may be NaN due to
        // unscaled stacked layernorms; we only verify the wiring assembled).
        assert_eq!(v.len(), in_channels * lh * lw);
    }
}

#[inline]
fn round_up(n: usize, m: usize) -> usize { n.div_ceil(m) * m }
