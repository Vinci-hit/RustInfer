//! Z-Image / Z-Image-Turbo text-to-image pipeline.
//!
//! Architecture (S3-DiT — Scalable Single-Stream DiT):
//!
//! ```text
//! Prompt ──→ Tokenizer(Qwen3) ──→ TextEncoder(Qwen3, hidden_states[-2])
//!                                        │
//!                                        ▼
//!   randn latent [B,16,H',W'] ──→ DiT Denoiser (30L) ──→ VAE Decoder ──→ RGB Image
//!                                   ↑ timestep
//!                             FlowMatchEulerScheduler
//! ```
//!
//! The Turbo variant uses only 2 denoising steps (`sigmas = [1.0, 0.3]`)
//! with no classifier-free guidance (`guidance_scale = 0.0`).
//!
//! ## Model directory layout (diffusers format)
//!
//! ```text
//! model_dir/
//! ├── model_index.json
//! ├── scheduler/scheduler_config.json
//! ├── text_encoder/{config.json, model*.safetensors}
//! ├── tokenizer/{tokenizer.json, ...}
//! ├── transformer/{config.json, diffusion_pytorch_model*.safetensors}
//! └── vae/{config.json, diffusion_pytorch_model.safetensors}
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::base::{DataType, DeviceType};
use crate::base::error::Result;
use crate::model::diffusion::pipeline::{DiffusionPipeline, DiffusionRequest, DiffusionOutput, DiffusionMetrics};
use crate::model::diffusion::scheduler::{FlowMatchEulerScheduler, Scheduler};
use crate::model::diffusion::z_image::text_encoder::Qwen3TextEncoder;
use crate::model::diffusion::z_image::transformer::ZImageTransformer2DModel;
use crate::model::runtime::InferenceState;
use crate::op::tensor_utils::cast_dtype;
use crate::tensor::Tensor;

/// VAE scale factor (2^(len(block_out_channels)-1) = 8 for standard VAE).
const VAE_SCALE_FACTOR: usize = 8;

/// VAE scaling_factor (from vae/config.json).
const VAE_SCALING_FACTOR: f32 = 0.3611;

/// VAE shift_factor (from vae/config.json).
const VAE_SHIFT_FACTOR: f32 = 0.1159;

/// Turbo sigmas — fixed 2-step schedule.
const TURBO_SIGMAS: &[f32] = &[1.0, 0.3];

/// Number of latent channels.
const LATENT_CHANNELS: usize = 16;

/// Z-Image text-to-image pipeline.
///
/// Owns all components of the diffusers pipeline:
/// text_encoder, transformer (DiT), scheduler.
/// VAE decoder will be added when implemented.
pub struct ZImagePipeline {
    pub device: DeviceType,
    pub scheduler: FlowMatchEulerScheduler,
    pub text_encoder: Qwen3TextEncoder,
    pub text_encoder_state: InferenceState,
    pub transformer: ZImageTransformer2DModel,
    pub vae: crate::model::diffusion::vae::decoder::VaeDecoder,
    pub model_dir: PathBuf,
}

impl ZImagePipeline {
    /// Load from a diffusers-format model directory.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        device: DeviceType,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // 1. Scheduler
        let scheduler = FlowMatchEulerScheduler::new(1000, 3.0);

        // 2. TextEncoder (Qwen3)
        let te_dir = model_dir.join("text_encoder");
        let tok_dir = model_dir.join("tokenizer");
        let text_encoder = Qwen3TextEncoder::new(&te_dir, &tok_dir, device)?;
        let text_encoder_state = text_encoder.create_state()?;

        // 3. Transformer (DiT) — load full weights
        let transformer_dir = model_dir.join("transformer");
        let transformer = ZImageTransformer2DModel::from_pretrained(&transformer_dir, device)?;

        // 4. VAE Decoder
        let vae_dir = model_dir.join("vae");
        let vae = crate::model::diffusion::vae::decoder::VaeDecoder::from_pretrained(&vae_dir, device)?;

        Ok(Self {
            device,
            scheduler,
            text_encoder,
            text_encoder_state,
            transformer,
            vae,
            model_dir: model_dir.to_path_buf(),
        })
    }

    // ─────────────────── Step 1: Encode Prompt ───────────────────────

    /// Encode a text prompt into caption embeddings.
    ///
    /// Returns `[actual_tokens, 2560]` — the text encoder hidden states,
    /// filtered by attention mask (no padding).
    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        self.text_encoder.encode(&mut self.text_encoder_state, prompt)
    }

    // ─────────────────── Step 2: Prepare Latents ─────────────────────

    /// Generate random noise latents.
    ///
    /// Returns `[1, 16, H', W']` where `H' = 2*(H/16)`, `W' = 2*(W/16)`.
    fn prepare_latents(
        &self,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let vae_scale = VAE_SCALE_FACTOR * 2; // 16
        let latent_h = 2 * (height / vae_scale);
        let latent_w = 2 * (width / vae_scale);

        Tensor::randn(
            &[1, LATENT_CHANNELS, latent_h, latent_w],
            DataType::F32,
            self.device,
            seed,
        )
    }

    // ─────────────────── Step 3: Prepare Timesteps ───────────────────

    /// Calculate shift mu and set timesteps.
    fn prepare_timesteps(&mut self, _latent_h: usize, _latent_w: usize) {
        // Turbo 模式使用固定 sigma schedule，不需要 dynamic shifting
        self.scheduler.set_timesteps_from_sigmas(TURBO_SIGMAS);
    }

    // ─────────────────── Step 4: Denoise Loop ────────────────────────

    /// Run the denoising loop.
    ///
    /// For Turbo: 2 steps with sigmas [1.0, 0.3].
    /// Each step: normalize timestep → transformer → negate → scheduler.step
    fn denoise(
        &mut self,
        mut latents: Tensor,
        prompt_embeds: &Tensor,
    ) -> Result<Tensor> {
        let num_steps = self.scheduler.num_steps();
        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();

        self.scheduler.reset();

        for i in 0..num_steps {
            let t = timesteps[i];
            let norm_t = (1000.0 - t) / 1000.0;

            // latents: [1, 16, H, W] → take sample 0 → [16, H, W] → unsqueeze(1) → [16, 1, H, W]
            let shape = latents.shape().to_vec();
            let (_b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let latent_5d = latents.view(&[c, 1, h, w])?;
            let latent_5d = {
                let mut m = Tensor::new(latent_5d.shape(), latent_5d.dtype(), latent_5d.device())?;
                m.copy_from(&latent_5d)?;
                m
            };

            // DiT forward: [16, 1, H, W], norm_t, [S, 2560] → [16, 1, H, W] velocity
            let model_out = self.transformer.forward(
                &latent_5d, norm_t, prompt_embeds,
                if self.device.is_cuda() { self.text_encoder_state.cuda_config.as_ref() } else { None },
            )?;

            // squeeze(1): [16, 1, H, W] → [16, H, W], then view as [1, 16, H, W]
            let noise_pred = model_out.view(&[1, c, h, w])?;
            let mut noise_pred = {
                let mut m = Tensor::new(noise_pred.shape(), noise_pred.dtype(), noise_pred.device())?;
                m.copy_from(&noise_pred)?;
                m
            };
            // negate: noise_pred = -noise_pred
            let neg = {
                let mut n = Tensor::new(noise_pred.shape(), noise_pred.dtype(), noise_pred.device())?;
                crate::op::scalar::scalar_mul(&noise_pred, &mut n, -1.0)?;
                n
            };
            noise_pred = neg;

            latents = self.scheduler.step(&noise_pred, &latents)?;
        }

        Ok(latents)
    }

    // ─────────────────── Step 5: VAE Decode ──────────────────────────

    /// Decode latents to RGB image.
    ///
    /// `latents / scaling_factor + shift_factor → vae.decode()`
    fn vae_decode(&self, latents: &Tensor) -> Result<Tensor> {
        // Rescale: latents / scaling_factor + shift_factor
        let mut rescaled = Tensor::new(latents.shape(), latents.dtype(), latents.device())?;
        crate::op::scalar::scalar_mul(latents, &mut rescaled, 1.0 / VAE_SCALING_FACTOR)?;
        let mut shifted = Tensor::new(latents.shape(), latents.dtype(), latents.device())?;
        crate::op::scalar::scalar_add(&rescaled, &mut shifted, VAE_SHIFT_FACTOR)?;

        // Actual VAE decode: [B, 16, H, W] → [B, 3, H*8, W*8]
        let cfg = if self.device.is_cuda() { self.text_encoder_state.cuda_config.as_ref() } else { None };
        let image = self.vae.decode(&shifted, cfg)?;

        Ok(image)
    }
}

impl DiffusionPipeline for ZImagePipeline {
    fn name(&self) -> &str {
        "z-image-turbo"
    }

    fn generate(&mut self, request: &DiffusionRequest) -> Result<DiffusionOutput> {
        let total_start = Instant::now();
        let weight_dtype = self.transformer.x_embedder.weight.dtype();

        // ── Step 1: Encode prompt ──
        let encode_start = Instant::now();
        let prompt_embeds = self.encode_prompt(&request.prompt)?;
        // 对齐到权重 dtype（text encoder 输出 F32，CUDA 权重为 BF16）
        let prompt_embeds = cast_dtype(&prompt_embeds, weight_dtype)?;
        let encode_prompt_ms = encode_start.elapsed().as_millis() as u64;

        // ── Step 2: Prepare latents ──
        let latents = self.prepare_latents(request.height, request.width, request.seed)?;
        let latent_h = latents.shape()[2];
        let latent_w = latents.shape()[3];
        // latents 从 F32 转到权重 dtype
        let latents = cast_dtype(&latents, weight_dtype)?;

        // ── Step 3: Prepare timesteps ──
        self.prepare_timesteps(latent_h, latent_w);

        // ── Step 4: Denoise loop ──
        let denoise_start = Instant::now();
        let latents = self.denoise(latents, &prompt_embeds)?;
        let denoise_ms = denoise_start.elapsed().as_millis() as u64;

        // ── Step 5: VAE decode ──
        let decode_start = Instant::now();
        let image = self.vae_decode(&latents)?;
        let decode_ms = decode_start.elapsed().as_millis() as u64;

        let total_ms = total_start.elapsed().as_millis() as u64;

        Ok(DiffusionOutput {
            output: image,
            metrics: DiffusionMetrics {
                encode_prompt_ms,
                denoise_ms,
                decode_ms,
                total_ms,
            },
        })
    }
}

// ──────────────────────────── Tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Save a [B, 3, H, W] float tensor (value range [-1, 1]) as a PPM image.
    /// PPM is a raw uncompressed image format that requires no external dependencies.
    fn save_tensor_as_ppm(tensor: &Tensor, path: &str) -> Result<()> {
        let shape = tensor.shape().to_vec();
        assert_eq!(shape.len(), 4, "expected [B,3,H,W], got {:?}", shape);
        let (_b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c, 3, "expected 3 channels, got {}", c);

        // Ensure CPU F32
        let t = tensor.to_device(DeviceType::Cpu)?;
        let t = if t.dtype() != DataType::F32 { t.to_dtype(DataType::F32)? } else { t };
        let data = t.as_f32()?.as_slice()?;

        let mut bytes = Vec::with_capacity(h * w * 3);
        // data layout: [b=0, c, h, w] — channel-major
        let plane = h * w;
        for py in 0..h {
            for px in 0..w {
                for ci in 0..3 {
                    let v = data[ci * plane + py * w + px];
                    // map [-1, 1] → [0, 255]
                    let u = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
                    bytes.push(u);
                }
            }
        }

        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        write!(f, "P6\n{} {}\n255\n", w, h)?;
        f.write_all(&bytes)?;
        Ok(())
    }

    fn get_model_dir() -> &'static Path {
        Path::new("/root/z-image-turbo")
    }

    #[test]
    #[ignore = "需要 Z-Image 模型权重"]
    fn test_pipeline_from_pretrained() -> Result<()> {
        let model_dir = get_model_dir();
        if !model_dir.join("text_encoder/config.json").exists() { return Ok(()); }

        let pipeline = ZImagePipeline::from_pretrained(model_dir, DeviceType::Cpu)?;

        assert_eq!(pipeline.name(), "z-image-turbo");
        assert_eq!(pipeline.transformer.config.dim, 3840);
        assert_eq!(pipeline.text_encoder.output_layer_count, 35);
        Ok(())
    }

    #[test]
    #[ignore = "需要 Z-Image 模型权重"]
    fn test_pipeline_encode_prompt() -> Result<()> {
        let model_dir = get_model_dir();
        if !model_dir.join("text_encoder/config.json").exists() { return Ok(()); }

        let mut pipeline = ZImagePipeline::from_pretrained(model_dir, DeviceType::Cpu)?;
        let embeds = pipeline.encode_prompt("a beautiful sunset over the ocean")?;

        println!("Prompt embeds shape: {:?}", embeds.shape());
        assert_eq!(embeds.shape().len(), 2);
        assert_eq!(embeds.shape()[1], 2560); // Qwen3 hidden_dim
        Ok(())
    }

    #[test]
    fn test_prepare_latents_shape() -> Result<()> {
        // Create a minimal pipeline stub for latent preparation
        let latent_h = 2 * (1024 / 16); // = 128
        let latent_w = 2 * (1024 / 16); // = 128
        let latents = Tensor::randn(
            &[1, LATENT_CHANNELS, latent_h, latent_w],
            DataType::F32,
            DeviceType::Cpu,
            Some(42),
        )?;

        assert_eq!(latents.shape(), &[1, 16, 128, 128]);
        Ok(())
    }

    #[test]
    fn test_scheduler_turbo_steps() {
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(TURBO_SIGMAS);

        assert_eq!(sched.num_steps(), 2);
        assert_eq!(sched.timesteps(), &[1000.0, 300.0]);
        assert_eq!(sched.sigmas(), &[1.0, 0.3, 0.0]);
    }

    #[test]
    fn test_calculate_shift_turbo() {
        use crate::model::diffusion::scheduler::calculate_shift;
        // For 1024x1024: latent = 128x128, image_seq_len = 64*64 = 4096
        let mu = calculate_shift(4096, 256, 4096, 0.5, 1.15);
        assert!((mu - 1.15).abs() < 1e-6);

        // For 512x512: latent = 64x64, image_seq_len = 32*32 = 1024
        let mu = calculate_shift(1024, 256, 4096, 0.5, 1.15);
        assert!(mu > 0.5 && mu < 1.15);
    }

    #[test]
    #[ignore = "需要 Z-Image 模型权重 + CUDA"]
    #[cfg(feature = "cuda")]
    fn test_pipeline_generate_cuda() -> Result<()> {
        let model_dir = get_model_dir();
        if !model_dir.join("text_encoder/config.json").exists() { return Ok(()); }

        eprintln!("[test] loading pipeline...");
        let mut pipeline = ZImagePipeline::from_pretrained(model_dir, DeviceType::Cuda(0))?;
        eprintln!("[test] loaded. generating...");

        // Very small size to isolate correctness / hang issues.
        let request = DiffusionRequest {
            prompt: "a cat".to_string(),
            height: 256,
            width: 256,
            seed: Some(42),
            ..Default::default()
        };

        let output = pipeline.generate(&request)?;
        println!("Output shape: {:?}", output.output.shape());
        println!("Metrics: encode={}ms, denoise={}ms, decode={}ms, total={}ms",
            output.metrics.encode_prompt_ms,
            output.metrics.denoise_ms,
            output.metrics.decode_ms,
            output.metrics.total_ms,
        );
        Ok(())
    }
}
