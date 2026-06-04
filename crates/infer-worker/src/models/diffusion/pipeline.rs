//! Z-Image text-to-image pipeline (Z-Image-Turbo eager-mode CUDA path).
//!
//! Loads transformer + VAE + text encoder + tokenizer from a diffusers-format
//! model directory, then runs the denoise loop end-to-end.

use std::path::{Path, PathBuf};

use crate::domain::ports::{OpResult, OpError, CoreOps};
use crate::domain::tensor::Tensor;
use crate::domain::types::Dtype;
use crate::infrastructure::cuda::Cuda;

use super::scheduler::FlowMatchEulerScheduler;
use super::state::{DitState, DitShapeSpec, PipelineState, ZImageCapacity, LATENT_CHANNELS};
use super::text_encoder::{Qwen3TextEncoder, TEXT_ENCODER_MAX_SEQ_LEN, PAD_TOKEN_ID, apply_chat_template};
use super::transformer::{ZImageTransformer};
use super::vae_decoder::{VaeDecoder};

/// Load a `[16, 1, H, W]` (or `[1, 16, H, W]`) F32 NPY file into a
/// `Tensor<T, Cuda>` of shape `[1, 16, H, W]` (cast to T).
fn load_latent_from_npy<T: Dtype>(
    path: &str, latent_h: usize, latent_w: usize, dev: &Cuda,
) -> OpResult<Tensor<T, Cuda>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| OpError::Kernel(format!("open {}: {}", path, e)))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| OpError::Kernel(format!("read {}: {}", path, e)))?;
    if &buf[..6] != b"\x93NUMPY" {
        return Err(OpError::Kernel("not an NPY file".into()));
    }
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    // Data starts at offset 10 + header_len.
    let data_start = 10 + header_len;
    let elem_bytes = buf.len() - data_start;
    let n_f32 = elem_bytes / 4;
    let expected = LATENT_CHANNELS * latent_h * latent_w;
    if n_f32 != expected {
        return Err(OpError::Shape(format!(
            "latent npy numel {} != expected {} ({}*{}*{})",
            n_f32, expected, LATENT_CHANNELS, latent_h, latent_w,
        )));
    }
    // Read raw f32 values.
    let f32_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(buf.as_ptr().add(data_start) as *const f32, n_f32)
    }.to_vec();
    // Cast to T host bytes.
    let mut host: Vec<u8> = vec![0u8; n_f32 * T::SIZE_BYTES];
    match T::DATA_TYPE {
        crate::domain::types::DataType::F32 => {
            let dst = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut f32, n_f32) };
            dst.copy_from_slice(&f32_data);
        }
        crate::domain::types::DataType::BF16 => {
            let dst = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut half::bf16, n_f32) };
            for (i, &v) in f32_data.iter().enumerate() {
                dst[i] = half::bf16::from_f32(v);
            }
        }
        crate::domain::types::DataType::F16 => {
            let dst = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut half::f16, n_f32) };
            for (i, &v) in f32_data.iter().enumerate() {
                dst[i] = half::f16::from_f32(v);
            }
        }
        other => return Err(OpError::Kernel(format!("load_latent: unsupported {:?}", other))),
    }
    Tensor::<T, Cuda>::from_host_bytes(
        &host,
        crate::domain::types::Shape::from_slice(&[1, LATENT_CHANNELS, latent_h, latent_w]),
        dev,
    )
}

/// Generation parameters.
pub struct GenerateParams {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub sigmas: Option<Vec<f32>>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            height: 1024, width: 1024,
            num_inference_steps: 9, guidance_scale: 0.0,
            seed: Some(42),
            sigmas: None,
        }
    }
}

/// Full Z-Image pipeline.
pub struct ZImagePipeline<T: Dtype> {
    pub transformer: ZImageTransformer<T, Cuda>,
    pub vae: VaeDecoder<T, Cuda>,
    pub text_encoder: Qwen3TextEncoder<T, Cuda>,
    pub tokenizer: tokenizers::Tokenizer,
    pub scheduler: FlowMatchEulerScheduler,
    pub dit_state: DitState<T, Cuda>,
    pub pipeline_state: PipelineState<T, Cuda>,
    pub model_dir: PathBuf,
    pub capacity: ZImageCapacity,
}

impl<T: Dtype> ZImagePipeline<T> {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P, device: &Cuda,
    ) -> OpResult<Self> {
        Self::from_pretrained_with_capacity(model_dir, device, ZImageCapacity::default())
    }

    pub fn from_pretrained_with_capacity<P: AsRef<Path>>(
        model_dir: P,
        device: &Cuda,
        capacity: ZImageCapacity,
    ) -> OpResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        // Scheduler.
        let scheduler = {
            let cfg_path = model_dir.join("scheduler/scheduler_config.json");
            let (num_train, shift) = if cfg_path.exists() {
                let s = std::fs::read_to_string(&cfg_path)
                    .map_err(|e| OpError::Kernel(format!("scheduler cfg: {}", e)))?;
                let v: serde_json::Value = serde_json::from_str(&s)
                    .map_err(|e| OpError::Kernel(format!("scheduler cfg parse: {}", e)))?;
                (
                    v.get("num_train_timesteps").and_then(|x| x.as_u64()).unwrap_or(1000) as usize,
                    v.get("shift").and_then(|x| x.as_f64()).unwrap_or(3.0) as f32,
                )
            } else { (1000, 3.0) };
            FlowMatchEulerScheduler::new(num_train, shift)
        };

        // Text encoder + tokenizer.
        let te_dir = model_dir.join("text_encoder");
        let tok_path = model_dir.join("tokenizer/tokenizer.json");
        let text_encoder: Qwen3TextEncoder<T, Cuda> =
            Qwen3TextEncoder::from_pretrained(&te_dir, &tok_path, device)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| OpError::Kernel(format!("tokenizer load: {}", e)))?;

        // Transformer.
        let transformer_dir = model_dir.join("transformer");
        let transformer: ZImageTransformer<T, Cuda> =
            ZImageTransformer::from_pretrained(&transformer_dir, device)?;

        // VAE.
        let vae_dir = model_dir.join("vae");
        let vae: VaeDecoder<T, Cuda> = VaeDecoder::from_pretrained(&vae_dir, device)?;

        // Pre-allocated state.
        let cfg = &transformer.config;
        let spec = DitShapeSpec {
            dim: cfg.dim,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            hidden_dim: cfg.intermediate_size,
            cap_feat_dim: cfg.cap_feat_dim,
            patch_size: cfg.patch_size,
            f_patch_size: cfg.f_patch_size,
            patch_in_dim: cfg.f_patch_size * cfg.patch_size * cfg.patch_size * cfg.in_channels,
            final_out_dim: cfg.patch_size * cfg.patch_size * cfg.f_patch_size * cfg.in_channels,
            capacity,
        };
        let dit_state: DitState<T, Cuda> = DitState::new(spec, device)?;
        let pipeline_state: PipelineState<T, Cuda> = PipelineState::new(capacity, device)?;

        Ok(Self {
            transformer, vae, text_encoder, tokenizer,
            scheduler, dit_state, pipeline_state,
            model_dir, capacity,
        })
    }

    /// Tokenize a prompt with the chat template, padded to `max_seq_len`.
    /// Returns `(token_ids, attention_mask)` both of length `max_seq_len`.
    pub fn tokenize(&self, prompt: &str, max_seq_len: usize) -> OpResult<(Vec<i32>, Vec<i32>)> {
        let formatted = apply_chat_template(prompt);
        let encoding = self.tokenizer.encode(formatted, true)
            .map_err(|e| OpError::Kernel(format!("tokenize: {}", e)))?;
        let ids = encoding.get_ids();
        let actual_len = ids.len().min(max_seq_len);
        let mut padded = vec![PAD_TOKEN_ID; max_seq_len];
        let mut mask = vec![0_i32; max_seq_len];
        for i in 0..actual_len {
            padded[i] = ids[i] as i32;
            mask[i] = 1;
        }
        Ok((padded, mask))
    }

    /// Run the full pipeline end-to-end. Returns `[1, 3, H, W]` BF16 image
    /// tensor on device (caller may need to cast/clamp/quantize for output).
    pub fn generate(&mut self, prompt: &str, params: &GenerateParams, device: &Cuda)
        -> OpResult<Tensor<T, Cuda>>
    {
        // 1. Tokenize and encode prompt.
        let (tokens, mask) = self.tokenize(prompt, TEXT_ENCODER_MAX_SEQ_LEN)?;
        let prompt_embeds = self.text_encoder.forward(&tokens, &mask, device)?;
        // prompt_embeds shape: [actual_len, cap_feat_dim], dtype T.
        if std::env::var("RUSTINFER_DUMP_PROMPT").is_ok() {
            super::dit_block::dump_tensor("prompt_embeds_rust", &prompt_embeds);
        }
        if std::env::var("RUSTINFER_DUMP_PROMPT").is_ok() {
            super::dit_block::dump_tensor("prompt_embeds_rust", &prompt_embeds);
        }

        // 2. Initialize random latent + scheduler.
        let latent_h = params.height / 8;
        let latent_w = params.width / 8;
        // If RUSTINFER_LATENT_NPY is set to a path of a [16,1,H,W] f32 NPY,
        // load that instead of running randn. This lets us share the exact
        // initial latent with the Python reference for element-wise diff.
        let latent_init: Tensor<T, Cuda> = if let Ok(path) = std::env::var("RUSTINFER_LATENT_NPY") {
            load_latent_from_npy::<T>(&path, latent_h, latent_w, device)?
        } else {
            Tensor::randn(
                [1, LATENT_CHANNELS, latent_h, latent_w], device, params.seed,
            )?
        };
        // Copy into pipeline_state.latents (which is sized to capacity).
        // For simplicity we use pipeline_state.latents shape directly when
        // capacity matches; otherwise a strided copy would be needed. Here
        // we just keep the freshly allocated tensor as our working sample.
        let mut sample = latent_init;
        let _ = &self.pipeline_state; // reserved for future graph capture

        // 3. Configure schedule.
        match params.sigmas.as_deref() {
            Some(s) => self.scheduler.set_timesteps_from_sigmas(s),
            None => {
                let patch = self.transformer.config.patch_size;
                let img_seq_len = (latent_h / patch) * (latent_w / patch);
                self.scheduler.set_timesteps_default(params.num_inference_steps, Some(img_seq_len));
            }
        }
        let _n_steps = self.scheduler.num_steps();

        // 4. Denoise loop.
        let t_scale = self.transformer.config.t_scale;
        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        let _in_channels = self.transformer.config.in_channels;
        for (i, &t) in timesteps.iter().enumerate() {
            // diffusers Z-Image: norm_t = (1 - t/1000), t_scaled = norm_t * t_scale.
            let norm_t = (1000.0 - t) / 1000.0;
            let t_scaled = norm_t * t_scale;

            // Reshape sample [1, C, H, W] → [C, 1, H, W] for transformer.
            let s = sample.shape().as_slice();
            let (b, c, h, w) = (s[0], s[1], s[2], s[3]);
            assert_eq!(b, 1);
            // View-only reshape via view_raw.
            let latent_5d = sample.view_raw(
                crate::domain::types::Shape::from_slice(&[c, 1, h, w]),
                crate::domain::types::Shape::from_slice(&[h * w, h * w, w, 1]).contiguous_strides(),
                sample.offset_elems(), true,
            );

            // Transformer forward.
            let model_out = self.transformer.forward(
                &latent_5d, &prompt_embeds, t_scaled, &mut self.dit_state,
            )?;
            // model_out shape [C, 1, H, W]. Reshape to [1, C, H, W] for scheduler step.
            let mo_4d = model_out.view_raw(
                crate::domain::types::Shape::from_slice(&[1, c, h, w]),
                crate::domain::types::Shape::from_slice(&[c * h * w, h * w, w, 1]).contiguous_strides(),
                model_out.offset_elems(), true,
            );
            // Negate (diffusers does `noise_pred = -model_out`).
            let mut neg_mo: Tensor<T, Cuda> = Tensor::zeros([1, c, h, w], device)?;
            neg_mo.copy_from(&mo_4d)?;
            crate::infrastructure::cuda::Cuda::scalar_mul_inplace(&mut neg_mo, -1.0)?;

            // Scheduler step: produces new sample.
            let mut next_sample: Tensor<T, Cuda> = Tensor::zeros([1, c, h, w], device)?;
            self.scheduler.step(&mut neg_mo, &sample, &mut next_sample)?;
            sample = next_sample;
            let _ = i;
        }

        // 5. VAE pre-rescale: latent / scaling_factor + shift_factor.
        let inv_scale = 1.0 / self.vae.config.scaling_factor as f64;
        let shift = self.vae.config.shift_factor as f64;
        let mut rescaled: Tensor<T, Cuda> = Tensor::zeros(*sample.shape(), device)?;
        rescaled.copy_from(&sample)?;
        crate::infrastructure::cuda::Cuda::scalar_mul_inplace(&mut rescaled, inv_scale)?;
        crate::infrastructure::cuda::Cuda::scalar_add_inplace(&mut rescaled, shift)?;

        // 6. VAE decode.
        let image = self.vae.decode(&rescaled, device)?;
        Ok(image)
    }
}
