//! Z-Image Pipeline — full text-to-image inference: encode → denoise → decode.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use super::scheduler::FlowMatchEulerScheduler;
use super::state::{DitState, PipelineState, LATENT_CHANNELS};
use super::transformer::ZImageTransformer;
use super::vae_decoder::VaeDecoder;

/// Full Z-Image text-to-image pipeline.
pub struct ZImagePipeline<T: Dtype, D: OpBackend> {
    pub transformer: ZImageTransformer<T, D>,
    pub vae_decoder: VaeDecoder<T, D>,
    pub scheduler: FlowMatchEulerScheduler,
    pub dit_state: DitState<T, D>,
    pub pipeline_state: PipelineState<T, D>,
}

/// Parameters for a single generation request.
pub struct GenerateParams {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self { height: 512, width: 512, num_inference_steps: 4, guidance_scale: 1.0 }
    }
}

impl<T: Dtype, D: OpBackend> ZImagePipeline<T, D> {
    /// Run the full pipeline: text embeddings → denoise loop → VAE decode → image.
    ///
    /// - `cap_embeds`: text encoder output [cap_len, cap_feat_dim]
    /// - `params`: generation parameters (height, width, steps)
    ///
    /// Returns: decoded image tensor [1, 3, height, width]
    pub fn generate(
        &mut self,
        cap_embeds: &Tensor<T, D>,
        params: &GenerateParams,
        dev: &D,
    ) -> OpResult<Tensor<T, D>> {
        let latent_h = params.height / 8;
        let latent_w = params.width / 8;

        // ─── 1. Initialize random latent (in production: random noise) ───
        // For now: zeros (deterministic for testing)
        let latent_shape = Shape::from_slice(&[1, LATENT_CHANNELS, latent_h, latent_w]);
        let mut latent = D::alloc_tensor::<T>(latent_shape, dev)?;

        // ─── 2. Compute sigma schedule ───
        let (sigmas, _timesteps) = self.scheduler.set_timesteps(params.num_inference_steps);

        // ─── 3. Patchify latent → tokens ───
        // Flatten spatial: [1, 16, H, W] → [H*W, 16] (patch_size=1 for simplicity here)
        let num_img_tokens = latent_h * latent_w;
        let patch_flat = LATENT_CHANNELS; // With patch_size=1: patch_flat = channels
        let mut x_tokens = D::alloc_tensor::<T>(Shape::from_slice(&[num_img_tokens, patch_flat]), dev)?;
        // Reshape latent → tokens (memcpy for now; real impl uses patchify)
        unsafe {
            std::ptr::copy_nonoverlapping(
                latent.data_ptr() as *const u8,
                x_tokens.data_ptr_mut() as *mut u8,
                num_img_tokens * patch_flat * T::SIZE_BYTES,
            );
        }

        // ─── 4. Denoise loop ───
        for step in 0..params.num_inference_steps {
            let t_value = sigmas[step] * self.transformer.config.t_scale;

            // Forward through transformer
            let noise_pred = self.transformer.forward(
                &x_tokens,
                cap_embeds,
                t_value,
                &mut self.dit_state,
            )?;

            // Euler step: x_tokens += dt * noise_pred
            let dt = self.scheduler.step_dt(&sigmas, step);
            let mut scaled_noise = D::alloc_tensor::<T>(noise_pred.shape().clone(), dev)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    noise_pred.data_ptr() as *const u8,
                    scaled_noise.data_ptr_mut() as *mut u8,
                    noise_pred.numel() * T::SIZE_BYTES,
                );
            }
            D::scalar_mul_inplace(&mut scaled_noise, dt as f64)?;
            D::add_inplace(&mut x_tokens, &scaled_noise)?;
        }

        // ─── 5. Unpatchify tokens → latent ───
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_tokens.data_ptr() as *const u8,
                latent.data_ptr_mut() as *mut u8,
                num_img_tokens * patch_flat * T::SIZE_BYTES,
            );
        }

        // ─── 6. VAE decode ───
        let image = self.vae_decoder.forward(&latent, dev)?;

        Ok(image)
    }
}
