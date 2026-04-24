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

use std::path::Path;
use crate::base::DeviceType;
use crate::base::error::Result;
use crate::model::diffusion::pipeline::{DiffusionPipeline, DiffusionRequest, DiffusionOutput};
use crate::model::diffusion::scheduler::FlowMatchEulerScheduler;

/// Z-Image text-to-image pipeline.
///
/// Owns all five components of the diffusers pipeline:
/// tokenizer, text_encoder, transformer (DiT), vae, scheduler.
pub struct ZImagePipeline {
    device: DeviceType,
    scheduler: FlowMatchEulerScheduler,
    // TODO: add components as they are implemented
    // pub tokenizer: Box<dyn Tokenizer>,
    // pub text_encoder: Qwen3,           // reuse LLM Qwen3 for hidden_states[-2]
    // pub transformer: ZImageTransformer2DModel,
    // pub vae_decoder: VaeDecoder,
    // pub vae_scale_factor: usize,       // 8
}

impl ZImagePipeline {
    /// Load from a diffusers-format model directory.
    pub fn from_pretrained<P: AsRef<Path>>(
        _model_dir: P,
        device: DeviceType,
    ) -> Result<Self> {
        // TODO: parse model_index.json, load each sub-component from its subfolder

        let scheduler = FlowMatchEulerScheduler::new(1000, 3.0);

        Ok(Self { device, scheduler })
    }
}

impl DiffusionPipeline for ZImagePipeline {
    fn name(&self) -> &str {
        "z-image-turbo"
    }

    fn generate(&mut self, _request: &DiffusionRequest) -> Result<DiffusionOutput> {
        // The full pipeline will be:
        //
        // 1. encode_prompt: tokenize → Qwen3 forward → hidden_states[-2]
        // 2. prepare_latents: randn [B, 16, H', W']
        // 3. scheduler.set_timesteps_from_sigmas([1.0, 0.3], mu)
        // 4. denoise loop (2 steps for Turbo):
        //      timestep = (1000 - t) / 1000
        //      noise_pred = -transformer.forward(latents, timestep, prompt_embeds)
        //      latents = scheduler.step(noise_pred, t, latents)
        // 5. vae_decode: latents / scaling_factor + shift_factor → vae.decode()

        todo!("ZImagePipeline::generate — implement after DiT + VAE ops are ready")
    }
}
