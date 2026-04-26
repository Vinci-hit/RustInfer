//! Z-Image Transformer 2D Model (S3-DiT denoising backbone).
//!
//! Full implementation that loads all 30 transformer blocks + 2 noise_refiner +
//! 2 context_refiner + x_embedder + cap_embedder + t_embedder + final_layer.

use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};

use crate::model::diffusion::diffusers_loader::DiffusersLoader;
use crate::model::diffusion::z_image::dit_block::DiTBlock;
use crate::model::diffusion::z_image::patchify::{patchify, unpatchify};
use crate::model::diffusion::z_image::rope_embedder_3d::RopeEmbedder3D;
use crate::model::diffusion::z_image::timestep_embedder::TimestepEmbedder;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::scalar;
use crate::op::tensor_utils::{clone_tensor, materialize,
    pad_last_row, pad_with_token, overwrite_pad_tokens, concat_seq};
use crate::tensor::Tensor;

const ADALN_EMBED_DIM: usize = 256;
const SEQ_MULTI_OF: usize = 32;

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

    /// Forward pass for a single sample.
    ///
    /// - `image`: [C=16, 1, H, W] — latent (dtype 必须与权重一致)
    /// - `t`: f32 scalar timestep (already normalized by t_scale outside)
    /// - `cap_feats`: [S_cap, cap_feat_dim] — text embeddings (dtype 必须与权重一致)
    ///
    /// Returns predicted velocity: [C=16, 1, H, W] (same dtype as input)
    pub fn forward(
        &self,
        image: &Tensor,
        t_value: f32,
        cap_feats: &Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Tensor> {
        let device = self.device;
        let dtype = image.dtype();

        // ── Timestep embedding ──
        let mut t_cpu = Tensor::new(&[1], DataType::F32, DeviceType::Cpu)?;
        t_cpu.as_f32_mut()?.as_slice_mut()?[0] = t_value * self.config.t_scale;
        let t_tensor = t_cpu.to_device(device)?;
        let adaln_input_2d = self.t_embedder.forward(&t_tensor, cuda_config)?;
        let adaln_input = adaln_input_2d.view(&[ADALN_EMBED_DIM])?;
        let adaln_input = materialize(&adaln_input)?;

        // ── Patchify & embed image ──
        let image_shape = image.shape();
        let (_c, f, h, w) = (image_shape[0], image_shape[1], image_shape[2], image_shape[3]);
        let patches_view = patchify(image, self.patch_size, self.f_patch_size)?;
        let patches = materialize(&patches_view)?;
        let n = patches.shape()[0];
        let mut x_emb = Tensor::new(&[n, self.config.dim], dtype, device)?;
        self.x_embedder.forward(&patches, &mut x_emb, cuda_config)?;

        // Pad x to SEQ_MULTI_OF
        let x_pad = (SEQ_MULTI_OF - n % SEQ_MULTI_OF) % SEQ_MULTI_OF;
        let x_padded_len = n + x_pad;
        let x_padded = pad_with_token(&x_emb, &self.x_pad_token, x_padded_len)?;

        // ── Cap embed ──
        let cap_ori_len = cap_feats.shape()[0];
        let cap_pad = (SEQ_MULTI_OF - cap_ori_len % SEQ_MULTI_OF) % SEQ_MULTI_OF;
        let cap_padded_len = cap_ori_len + cap_pad;

        let cap_feats_padded = pad_last_row(cap_feats, cap_padded_len)?;

        let mut cap_normed = Tensor::new(&[cap_padded_len, self.config.cap_feat_dim], dtype, device)?;
        self.cap_embedder_norm.forward(&cap_feats_padded, &mut cap_normed, cuda_config)?;

        let mut cap_emb = Tensor::new(&[cap_padded_len, self.config.dim], dtype, device)?;
        self.cap_embedder_linear.forward(&cap_normed, &mut cap_emb, cuda_config)?;

        let cap_padded = overwrite_pad_tokens(&cap_emb, &self.cap_pad_token, cap_ori_len)?;

        // ── Build position ids ──
        let cap_pos_ids = build_cap_pos_ids(cap_padded_len, device)?;
        let f_t = f / self.f_patch_size;
        let h_t = h / self.patch_size;
        let w_t = w / self.patch_size;
        let x_pos_ids = build_image_pos_ids(
            f_t, h_t, w_t, cap_padded_len + 1, x_padded_len - n, device,
        )?;

        // ── Compute RoPE cos/sin ──
        let (x_cos, x_sin) = self.rope_embedder.embed(&x_pos_ids, device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.embed(&cap_pos_ids, device)?;

        // ── noise_refiner on x ──
        let mut x = x_padded;
        for block in &self.noise_refiner {
            x = block.forward(&x, &x_cos, &x_sin, Some(&adaln_input), cuda_config)?;
        }

        // ── context_refiner on cap ──
        let mut cap = cap_padded;
        for block in &self.context_refiner {
            cap = block.forward(&cap, &cap_cos, &cap_sin, None, cuda_config)?;
        }

        // ── Unified concat: [x | cap] ──
        let unified = concat_seq(&x, &cap)?;
        let unified_cos = concat_seq(&x_cos, &cap_cos)?;
        let unified_sin = concat_seq(&x_sin, &cap_sin)?;

        // ── Main layers ──
        let mut unified = unified;
        for block in &self.layers {
            unified = block.forward(&unified, &unified_cos, &unified_sin, Some(&adaln_input), cuda_config)?;
        }

        // ── Final layer ──
        let out = self.final_layer_forward(&unified, &adaln_input, cuda_config)?;
        let out_x = out.slice(&[0, 0], &[x_padded_len, out.shape()[1]])?;
        let out_x = materialize(&out_x)?;
        let out_x = out_x.slice(&[0, 0], &[n, out_x.shape()[1]])?;
        let out_x = materialize(&out_x)?;

        // ── Unpatchify ──
        let image_out_view = unpatchify(&out_x, f, h, w, self.config.in_channels, self.patch_size, self.f_patch_size)?;
        materialize(&image_out_view)
    }

    fn final_layer_forward(
        &self,
        x: &Tensor,          // [S, dim]
        c: &Tensor,          // [dim_adaln=256]
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Tensor> {
        let seq = x.shape()[0];
        let dim = self.config.dim;
        let dtype = x.dtype();
        let device = x.device();

        // scale = 1 + adaLN_modulation(silu(c))
        let mut c_silu = clone_tensor(c)?;
        scalar::silu_inplace(&mut c_silu)?;
        let c_silu_2d = c_silu.view(&[1, ADALN_EMBED_DIM])?;

        let mut scale_raw = Tensor::new(&[1, dim], dtype, device)?;
        self.final_layer_adaln.forward(&c_silu_2d, &mut scale_raw, cuda_config)?;
        let mut scale = Tensor::new(&[1, dim], dtype, device)?;
        scalar::scalar_add(&scale_raw, &mut scale, 1.0)?;

        // norm_final(x) * scale
        let mut normed = Tensor::new(&[seq, dim], dtype, device)?;
        crate::op::layernorm::layernorm(x, &mut normed, self.final_layer_eps)?;
        broadcast_mul_rowvec(&mut normed, &scale)?;

        // Linear
        let out_dim = self.final_layer_linear.weight.shape()[0];
        let mut out = Tensor::new(&[seq, out_dim], dtype, device)?;
        self.final_layer_linear.forward(&normed, &mut out, cuda_config)?;
        Ok(out)
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
        // rope_embedder cache 始终在 CPU（查表操作），不需要上传
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

    let to_q = load_linear(loader, &format!("{}.attention.to_q", prefix), dim, dim, false, device)?;
    let to_k = load_linear(loader, &format!("{}.attention.to_k", prefix), dim, dim, false, device)?;
    let to_v = load_linear(loader, &format!("{}.attention.to_v", prefix), dim, dim, false, device)?;
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
        to_q, to_k, to_v, to_out,
        norm_q, norm_k,
        w1, w3, w2,
        adaln_modulation,
        dim, n_heads, head_dim,
        modulation,
    })
}

// ───────────────────────── Tensor helpers ─────────────────────────

fn broadcast_mul_rowvec(x: &mut Tensor, v: &Tensor) -> Result<()> {
    use crate::op::broadcast_mul::broadcast_mul;
    let s = x.shape()[0];
    let d = x.shape()[1];
    let mut out = Tensor::new(&[s, d], x.dtype(), x.device())?;
    broadcast_mul(x, v, &mut out)?;
    x.copy_from(&out)?;
    Ok(())
}

/// Build caption pos ids [cap_len, 3] on CPU, then upload to device.
fn build_cap_pos_ids(cap_len: usize, device: DeviceType) -> Result<Tensor> {
    let mut ids = Tensor::new(&[cap_len, 3], DataType::I32, DeviceType::Cpu)?;
    let data = ids.as_i32_mut()?.as_slice_mut()?;
    for i in 0..cap_len {
        data[i * 3] = (i + 1) as i32;
        data[i * 3 + 1] = 0;
        data[i * 3 + 2] = 0;
    }
    ids.to_device(device)
}

/// Build image pos ids [n + pad_len, 3] on CPU, then upload to device.
fn build_image_pos_ids(
    f_t: usize, h_t: usize, w_t: usize,
    t_base: usize,
    pad_len: usize,
    device: DeviceType,
) -> Result<Tensor> {
    let n = f_t * h_t * w_t;
    let total = n + pad_len;
    let mut ids = Tensor::new(&[total, 3], DataType::I32, DeviceType::Cpu)?;
    let data = ids.as_i32_mut()?.as_slice_mut()?;
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
    ids.to_device(device)
}
