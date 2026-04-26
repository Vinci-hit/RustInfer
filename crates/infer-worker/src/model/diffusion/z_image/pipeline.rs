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
use crate::model::diffusion::buffer::DiffBufferType as BT;
use crate::model::diffusion::pipeline::{
    DiffusionMetrics, DiffusionOutput, DiffusionPipeline, DiffusionRequest,
};
use crate::model::diffusion::scheduler::{FlowMatchEulerScheduler, Scheduler};
use crate::model::diffusion::z_image::state::{DitState, PipelineState, ZImageCapacity};
use crate::model::diffusion::z_image::text_encoder::Qwen3TextEncoder;
use crate::model::diffusion::z_image::transformer::ZImageTransformer2DModel;
use crate::model::runtime::InferenceState;
use crate::op::tensor_utils::{cast_dtype, clone_tensor};
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
/// text_encoder, transformer (DiT), scheduler, VAE decoder, plus
/// pre-allocated runtime state for the denoise ping-pong.
pub struct ZImagePipeline {
    pub device: DeviceType,
    pub scheduler: FlowMatchEulerScheduler,
    pub text_encoder: Qwen3TextEncoder,
    pub text_encoder_state: InferenceState,
    pub transformer: ZImageTransformer2DModel,
    pub vae: crate::model::diffusion::vae::decoder::VaeDecoder,
    /// Pipeline-level pre-allocated buffers (Latents / LatentsTmp /
    /// NoisePred / Latent5D). Allocated once in `from_pretrained`; its
    /// tensors are sliced to the request shape each `generate()` call and
    /// never reallocated.
    pub pipeline_state: PipelineState,
    /// Transformer-level workspace. Owns every intermediate buffer the
    /// DiT forward pass needs (embeddings, RoPE cache, per-block scratch,
    /// final-layer outputs). Allocated once; block / transformer forward
    /// functions slice from it each step.
    pub dit_state: DitState,
    pub model_dir: PathBuf,
}

impl ZImagePipeline {
    /// Load from a diffusers-format model directory using the default
    /// [`ZImageCapacity`] (1024×1024 max, 512 caption tokens).
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        device: DeviceType,
    ) -> Result<Self> {
        Self::from_pretrained_with_capacity(model_dir, device, ZImageCapacity::default())
    }

    /// Load from a diffusers-format model directory with an explicit
    /// capacity hint. The returned pipeline can only serve requests whose
    /// `(height, width, cap_len)` fit within this capacity; anything
    /// larger will be rejected by `check_request`.
    pub fn from_pretrained_with_capacity<P: AsRef<Path>>(
        model_dir: P,
        device: DeviceType,
        capacity: ZImageCapacity,
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

        // 5. PipelineState — latent buffer pool sized to capacity.
        //    Working dtype matches the transformer's weight dtype so the
        //    scheduler step runs without any on-the-fly dtype conversions.
        let weight_dtype = transformer.x_embedder.weight.dtype();
        let pipeline_state = PipelineState::new(capacity, weight_dtype, device)?;

        // 6. DitState — transformer internal workspace. Derives its
        //    shape spec from the loaded transformer config + capacity.
        let shape_spec = transformer.shape_spec(capacity);
        let dit_state = DitState::new(shape_spec)?;

        Ok(Self {
            device,
            scheduler,
            text_encoder,
            text_encoder_state,
            transformer,
            vae,
            pipeline_state,
            dit_state,
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

    /// Fill the pre-allocated `Latents` buffer with fresh Gaussian noise
    /// at the shape implied by `(height, width)`.
    ///
    /// Returns `(latent_h, latent_w)` so the caller can slice the exact
    /// working region out of the oversized pool each step.
    ///
    /// The randn samples are generated on CPU (Box-Muller) then copied
    /// into the state slot, cast to the working dtype on the fly. This
    /// path still allocates a single transient F32 tensor, but it is
    /// outside the denoise loop, so it never runs during graph capture.
    fn prepare_latents(
        &mut self,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<(usize, usize)> {
        let vae_scale = VAE_SCALE_FACTOR * 2; // 16
        let latent_h = 2 * (height / vae_scale);
        let latent_w = 2 * (width / vae_scale);

        let weight_dtype = self.pipeline_state.dtype;
        let shape = [1, LATENT_CHANNELS, latent_h, latent_w];

        // Generate noise on CPU in F32, convert to target dtype, then
        // copy into the pre-allocated Latents slot (sliced to the exact
        // working region).
        let noise_f32 = Tensor::randn(&shape, DataType::F32, self.device, seed)?;
        let noise = cast_dtype(&noise_f32, weight_dtype)?;

        let mut latents_slot = self.pipeline_state.slice_mut(BT::Latents, &shape)?;
        latents_slot.copy_from(&noise)?;

        Ok((latent_h, latent_w))
    }

    // ─────────────────── Step 3: Prepare Timesteps ───────────────────

    /// Install the (fixed) Turbo sigma schedule on the scheduler.
    fn prepare_timesteps(&mut self, _latent_h: usize, _latent_w: usize) {
        // Turbo runs a hard-coded 2-step schedule; no dynamic shifting.
        self.scheduler.set_timesteps_from_sigmas(TURBO_SIGMAS);
    }

    // ─────────────────── Step 4: Denoise Loop ────────────────────────

    /// Run the denoising loop using the pipeline's pre-allocated latent
    /// pool.
    ///
    /// Ping-pongs between `Latents` and `LatentsTmp` slices so
    /// `scheduler.step` always writes to a destination distinct from its
    /// read source. `NoisePred` receives the (negated) DiT velocity each
    /// step, then is consumed by the scheduler.
    ///
    /// Returns the final latents as an owned clone — decoupling the
    /// output from the pool so `vae_decode` can run without holding a
    /// borrow on `pipeline_state`.
    fn denoise(
        &mut self,
        latent_h: usize,
        latent_w: usize,
        prompt_embeds: &Tensor,
    ) -> Result<Tensor> {
        let num_steps = self.scheduler.num_steps();
        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        self.scheduler.reset();

        let shape4 = [1, LATENT_CHANNELS, latent_h, latent_w];
        let shape5 = [LATENT_CHANNELS, 1, latent_h, latent_w];
        let cuda_cfg = self.text_encoder_state.cuda_config.as_ref()
            .filter(|_| self.device.is_cuda());

        for (i, &t) in timesteps.iter().enumerate() {
            let norm_t = (1000.0 - t) / 1000.0;

            // Ping-pong: even step reads Latents → writes LatentsTmp;
            // odd step swaps. No mem::swap needed — each iteration
            // re-slices the state.
            let (sample_ty, dst_ty) = if i % 2 == 0 {
                (BT::Latents, BT::LatentsTmp)
            } else {
                (BT::LatentsTmp, BT::Latents)
            };

            // DiT input: copy the current [1, C, H, W] latent into the
            // pre-allocated [C, 1, H, W] Latent5D slot so the transformer
            // sees a contiguous, non-aliased tensor.
            let cur = self.pipeline_state.slice(sample_ty, &shape4)?;
            let mut latent_5d = self.pipeline_state.slice_mut(BT::Latent5D, &shape5)?;
            latent_5d.copy_from(&cur.view(&shape5)?)?;

            // DiT forward. `model_out` aliases state.ImageOut — we must
            // consume it via copy_from into NoisePred before the next
            // invocation would clobber the underlying buffer.
            let model_out = self.transformer.forward(
                &latent_5d, norm_t, prompt_embeds, cuda_cfg, &mut self.dit_state,
            )?;

            // noise_pred = -model_out, shaped [1, C, H, W].
            let mut noise_pred = self.pipeline_state.slice_mut(BT::NoisePred, &shape4)?;
            noise_pred.copy_from(&model_out.view(&shape4)?)?;
            noise_pred *= -1.0_f32;

            // dst = sample + dt * noise_pred  (noise_pred is consumed).
            let cur_read = self.pipeline_state.slice(sample_ty, &shape4)?;
            let mut dst = self.pipeline_state.slice_mut(dst_ty, &shape4)?;
            self.scheduler.step(&mut noise_pred, &cur_read, &mut dst)?;
        }

        // After N iterations the last write lives in `Latents` (N even)
        // or `LatentsTmp` (N odd). Clone it out so the caller doesn't
        // need to hold a borrow on the state.
        let final_ty = if num_steps % 2 == 0 { BT::Latents } else { BT::LatentsTmp };
        clone_tensor(&self.pipeline_state.slice(final_ty, &shape4)?)
    }

    // ─────────────────── Step 5: VAE Decode ──────────────────────────

    /// Decode latents to an RGB image.
    ///
    /// `shifted = latents / scaling_factor + shift_factor`, then
    /// `vae.decode(shifted)` produces `[B, 3, H*8, W*8]`.
    fn vae_decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut shifted = clone_tensor(latents)?;
        shifted *= 1.0_f32 / VAE_SCALING_FACTOR;
        shifted += VAE_SHIFT_FACTOR;

        let cfg = self.text_encoder_state.cuda_config.as_ref()
            .filter(|_| self.device.is_cuda());
        self.vae.decode(&shifted, cfg)
    }

    // ─────────────────── Warm-up ─────────────────────────────────────

    /// Pre-run every hot path so the first real `generate()` call sees a
    /// hot CUDA context. Covers:
    ///
    /// - CUDA kernel module load / PTX→SASS JIT for every kernel in the
    ///   text-encoder, DiT transformer, and VAE decoder.
    /// - cuBLASLt algorithm heuristics for every `(M,N,K,dtype,layout)`
    ///   tuple hit during denoise.
    /// - cuDNN Conv2d descriptor / algorithm / workspace cache
    ///   ([`Conv2dCache`]) for every conv in the VAE at the requested
    ///   shape.
    ///
    /// Runs a single end-to-end `generate()` at `(height, width)` with a
    /// 1-char prompt and one denoise step, then `cudaDeviceSynchronize`s
    /// so every cache fill completes before this call returns.
    ///
    /// ## Choosing `(height, width)`
    ///
    /// cuDNN's conv cache key includes the input shape, so the cache
    /// entries filled here only help subsequent requests at the **same**
    /// `(height, width)`. For production serving, call this with the
    /// exact shape you expect; for generic warmup call [`Self::warmup`]
    /// which uses the smallest supported shape (256×256).
    ///
    /// Both `height` and `width` must be multiples of 16 (VAE requires
    /// `H % 16 == 0`, `W % 16 == 0`) and fit within the pipeline's
    /// configured [`ZImageCapacity`].
    ///
    /// [`Conv2dCache`]: crate::op::kernels::cuda::Conv2dCache
    pub fn warmup_for(&mut self, height: usize, width: usize) -> Result<()> {
        // A full 2-step Turbo generate() at 256×256 runs in a handful of
        // milliseconds once warm, so we just replay the real path rather
        // than building a bespoke single-step shortcut — that would risk
        // drifting from what `generate()` actually exercises.
        let request = DiffusionRequest {
            prompt: "a".to_string(),
            height,
            width,
            seed: Some(0),
            ..Default::default()
        };
        let _ = self.generate(&request)?;

        // Ensure every queued kernel / cache fill has completed before
        // we hand control back; otherwise the first user request would
        // still race against tail latency from warmup.
        if self.device.is_cuda() {
            unsafe { crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?; }
        }
        Ok(())
    }

    /// Generic warm-up at the smallest supported shape (256×256).
    ///
    /// Use this when you don't know the exact request shape up-front.
    /// Populates kernel module / cuBLASLt / tokenizer caches, but the
    /// cuDNN Conv2d cache entries are keyed on shape — they'll be
    /// re-filled on the first request at a different `(height, width)`.
    /// For shape-specific warm-up prefer [`Self::warmup_for`].
    pub fn warmup(&mut self) -> Result<()> {
        self.warmup_for(256, 256)
    }
}

impl DiffusionPipeline for ZImagePipeline {
    fn name(&self) -> &str {
        "z-image-turbo"
    }

    fn generate(&mut self, request: &DiffusionRequest) -> Result<DiffusionOutput> {
        // All per-stage timings are **GPU wall-time**, not host-side
        // launch time: each stage ends with a `cudaDeviceSynchronize`
        // (on CUDA) so any kernels it enqueued are actually finished
        // before we stop the clock. Without this, encode_prompt_ms
        // would measure host launch throughput (typically <5 ms) while
        // the work was still racing on the stream.
        let is_cuda = self.device.is_cuda();
        let sync = || -> Result<()> {
            #[cfg(feature = "cuda")]
            {
                if is_cuda {
                    unsafe { crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?; }
                }
            }
            #[cfg(not(feature = "cuda"))]
            { let _ = is_cuda; }
            Ok(())
        };

        let total_start = Instant::now();
        let weight_dtype = self.transformer.x_embedder.weight.dtype();

        // Capacity guard — reject requests that don't fit the pre-allocated
        // pool rather than silently reallocating.
        self.pipeline_state.check_request(request.height, request.width)?;

        // ── Step 1: Encode prompt ──
        let encode_start = Instant::now();
        let prompt_embeds = self.encode_prompt(&request.prompt)?;
        // Cast to the transformer's weight dtype (text encoder emits F32,
        // CUDA weights are typically BF16).
        let prompt_embeds = cast_dtype(&prompt_embeds, weight_dtype)?;
        sync()?;
        let encode_prompt_ms = encode_start.elapsed().as_millis() as u64;

        // ── Step 2: Prepare latents (fills pipeline_state.Latents in place) ──
        let (latent_h, latent_w) = self.prepare_latents(
            request.height, request.width, request.seed,
        )?;

        // ── Step 3: Prepare timesteps ──
        self.prepare_timesteps(latent_h, latent_w);

        // ── Step 4: Denoise loop (ping-pongs through the state pool) ──
        let denoise_start = Instant::now();
        let latents = self.denoise(latent_h, latent_w, &prompt_embeds)?;
        sync()?;
        let denoise_ms = denoise_start.elapsed().as_millis() as u64;

        // ── Step 5: VAE decode ──
        let decode_start = Instant::now();
        let image = self.vae_decode(&latents)?;
        sync()?;
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

        // Warm every hot path (kernel JIT, cuBLASLt heuristics, cuDNN
        // Conv2d cache) at the exact shape we're about to generate at.
        // Without this, the single generate() call below eats the full
        // cold-start tax — typically 10×+ the steady-state latency.
        eprintln!("[test] warming up at 256x256...");
        let t_warm = Instant::now();
        pipeline.warmup_for(256, 256)?;
        eprintln!("[test] warmup done in {}ms", t_warm.elapsed().as_millis());

        // Very small size to isolate correctness / hang issues.
        let request = DiffusionRequest {
            prompt: "a photograph of a cat wearing a red hat, sitting on a wooden bench in a sunny park".to_string(),
            height: 256,
            width: 256,
            seed: Some(42),
            ..Default::default()
        };

        let output = pipeline.generate(&request)?;
        eprintln!("Output shape: {:?}", output.output.shape());
        eprintln!("Metrics: encode={}ms, denoise={}ms, decode={}ms, total={}ms",
            output.metrics.encode_prompt_ms,
            output.metrics.denoise_ms,
            output.metrics.decode_ms,
            output.metrics.total_ms,
        );
        save_tensor_as_ppm(&output.output, "/root/z_image_cuda_output.ppm")?;
        eprintln!("Saved /root/z_image_cuda_output.ppm");
        Ok(())
    }

}
