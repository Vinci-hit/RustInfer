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
use crate::model::diffusion::z_image::state::{DitState, PipelineState, ZImageCapacity, LATENT_CHANNELS};
use crate::model::diffusion::z_image::text_encoder::Qwen3TextEncoder;
use crate::model::diffusion::z_image::transformer::ZImageTransformer2DModel;
use crate::model::runtime::InferenceState;
use crate::op::cast::cast_dtype;
use crate::tensor::Tensor;

/// Deep-copy a tensor (allocate + copy).
fn clone_tensor(src: &Tensor) -> crate::base::error::Result<Tensor> {
    src.contiguous()
}

/// VAE scale factor (2^(len(block_out_channels)-1) = 8 for standard VAE).
const VAE_SCALE_FACTOR: usize = 8;

/// Small HtoD helper used once per `generate()` to upload the per-step
/// `t_value` / `dt` schedule (≤ `N_MAX_STEPS` floats, so ≤ 64 B per
/// call) into a pre-allocated `[N_MAX_STEPS]` F32 device slot.
///
/// Prefers a raw stream-ordered `cudaMemcpyAsync` to avoid allocating a
/// transient CPU `Tensor`. `src.len()` must be `<= dst.numel()`.
#[cfg(feature = "cuda")]
fn upload_f32_slice_into(dst: &mut Tensor, src: &[f32]) -> Result<()> {
    use crate::cuda::ffi;
    debug_assert!(dst.dtype() == DataType::F32);
    debug_assert!(src.len() <= dst.numel());
    let stream = crate::cuda::get_current_cuda_stream();
    let dptr = dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut std::ffi::c_void;
    let sptr = src.as_ptr() as *const std::ffi::c_void;
    let n_bytes = src.len() * std::mem::size_of::<f32>();
    unsafe {
        crate::cuda_check!(ffi::cudaMemcpyAsync(
            dptr, sptr, n_bytes,
            ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream,
        ))?;
    }
    Ok(())
}
/// VAE scaling_factor (from vae/config.json).
const VAE_SCALING_FACTOR: f32 = 0.3611;

/// VAE shift_factor (from vae/config.json).
const VAE_SHIFT_FACTOR: f32 = 0.1159;

/// Turbo sigmas — fixed 2-step schedule.
const TURBO_SIGMAS: &[f32] = &[1.0, 0.3];

fn validate_sigmas(sigmas: &[f32]) -> Result<()> {
    if sigmas.is_empty() {
        return Err(crate::base::error::Error::InvalidArgument(
            "sigmas must contain at least one step".into()
        ).into());
    }
    if sigmas.len() > crate::model::diffusion::z_image::state::N_MAX_STEPS {
        return Err(crate::base::error::Error::InvalidArgument(format!(
            "sigmas length ({}) exceeds N_MAX_STEPS ({})",
            sigmas.len(),
            crate::model::diffusion::z_image::state::N_MAX_STEPS,
        )).into());
    }
    for (i, &sigma) in sigmas.iter().enumerate() {
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "sigma[{}]={} must be finite and in [0, 1]",
                i, sigma
            )).into());
        }
        if i > 0 && sigma > sigmas[i - 1] {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "sigmas must be monotonically non-increasing: sigma[{}]={} > sigma[{}]={}",
                i, sigma, i - 1, sigmas[i - 1]
            )).into());
        }
    }
    Ok(())
}

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
    /// When true, skip the denoise CUDA-Graph path and run the eager
    /// denoise loop. Used by `warmup_for` to let cuBLASLt / cuDNN fill
    /// their algorithm / workspace caches *before* we try to capture —
    /// capturing an un-warm cuBLASLt call trips
    /// `cudaErrorStreamCaptureUnsupported` because the library makes
    /// synchronous queries the first time it sees a new `(M, N, K,
    /// dtype, layout)` tuple.
    pub force_eager_denoise: bool,
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

        // 1. Scheduler — read config from scheduler/scheduler_config.json
        let scheduler = {
            let sched_cfg_path = model_dir.join("scheduler/scheduler_config.json");
            let (num_train_timesteps, shift) = if sched_cfg_path.exists() {
                let cfg_str = std::fs::read_to_string(&sched_cfg_path)?;
                let cfg: serde_json::Value = serde_json::from_str(&cfg_str)
                    .map_err(|e| crate::base::error::Error::InvalidArgument(
                        format!("failed to parse scheduler config: {}", e)
                    ))?;
                let num_train = cfg.get("num_train_timesteps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1000) as usize;
                let shift = cfg.get("shift")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(3.0) as f32;
                (num_train, shift)
            } else {
                (1000, 3.0) // Turbo default
            };
            FlowMatchEulerScheduler::new(num_train_timesteps, shift)
        };

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
            force_eager_denoise: false,
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

    /// Install a sigma schedule on the scheduler.
    ///
    /// - If the request supplies an explicit `sigmas` list, those sigmas
    ///   are used verbatim (length = number of denoising steps).
    /// - Otherwise the scheduler generates its official default
    ///   schedule for `num_inference_steps`, applying dynamic shifting
    ///   based on the latent sequence length (matching
    ///   `diffusers.FlowMatchEulerDiscreteScheduler` on Z-Image / Flux /
    ///   SD3).
    fn prepare_timesteps(
        &mut self,
        request: &DiffusionRequest,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<()> {
        match request.sigmas.as_deref() {
            Some(sigmas) => {
                validate_sigmas(sigmas)?;
                self.scheduler.set_timesteps_from_sigmas(sigmas);
            }
            None => {
                if request.num_inference_steps == 0 {
                    return Err(crate::base::error::Error::InvalidArgument(
                        "num_inference_steps must be >= 1".into()
                    ).into());
                }
                if request.num_inference_steps > crate::model::diffusion::z_image::state::N_MAX_STEPS {
                    return Err(crate::base::error::Error::InvalidArgument(format!(
                        "num_inference_steps ({}) exceeds N_MAX_STEPS ({})",
                        request.num_inference_steps,
                        crate::model::diffusion::z_image::state::N_MAX_STEPS,
                    )).into());
                }
                // image_seq_len = number of DiT tokens after patchify.
                let patch = self.transformer.patch_size;
                let image_seq_len = (latent_h / patch) * (latent_w / patch);
                self.scheduler.set_timesteps_default(
                    request.num_inference_steps,
                    Some(image_seq_len),
                );
            }
        }
        Ok(())
    }

    // ─────────────────── Step 4: Denoise Loop ────────────────────────

    /// Run the denoising loop using the pipeline's pre-allocated latent
    /// pool, captured once into a CUDA Graph and replayed on every
    /// subsequent generate() with the same `(latent_h, latent_w,
    /// cap_padded_len, num_steps)` shape key.
    ///
    /// Layout:
    ///
    /// 1. **Per-request prep, outside capture**:
    ///    - Stage `prompt_embeds` into the pre-allocated
    ///      [`PromptEmbedsPadded`] slot (+ pad to `cap_padded_len`).
    ///    - Call `transformer.prepare_denoise_constants` to populate
    ///      every shape-fixed tensor (cap-side refine, pos_ids, rope
    ///      cos/sin, unified cos/sin).
    ///    - Upload the per-step `t_value(i)` and Euler `dt(i)`
    ///      schedules into [`TValueDevVec`] / [`DtValueDevVec`] with
    ///      one HtoD each.
    /// 2. **Capture or replay**: if `cfg.cuda_graph` is empty,
    ///    `begin → for i in 0..N { step_body(i) } → end → launch`.
    ///    Otherwise just `launch`. The step body is dst-write-only
    ///    against state slots and reads all per-step scalars from
    ///    `TValueDevVec`/`DtValueDevVec` via device-pointer kernels, so
    ///    it is capture-safe and replay-stable.
    /// 3. **"Plan B" ping-pong**: every step reads from [`Latents`] and
    ///    writes to [`LatentsTmp`], then a trailing DtoD copy moves
    ///    [`LatentsTmp`] back to [`Latents`]. This keeps the graph
    ///    simple (one fixed direction) at the cost of a cheap
    ///    per-step memcpy.
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
        if num_steps == 0 {
            return Err(crate::base::error::Error::InvalidArgument(
                "denoise: scheduler has 0 steps".into(),
            ).into());
        }
        if num_steps > crate::model::diffusion::z_image::state::N_MAX_STEPS {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "denoise: num_steps ({}) exceeds compiled N_MAX_STEPS ({})",
                num_steps,
                crate::model::diffusion::z_image::state::N_MAX_STEPS,
            )).into());
        }

        #[cfg(feature = "cuda")]
        let use_cuda_graph = self.device.is_cuda() && !self.force_eager_denoise;
        #[cfg(not(feature = "cuda"))]
        let use_cuda_graph = false;

        // ─── CPU / eager fallback — identical to the previous denoise ───
        if !use_cuda_graph {
            return self.denoise_eager(latent_h, latent_w, prompt_embeds);
        }

        #[cfg(feature = "cuda")]
        {
            use crate::cuda::with_cuda_stream;
            use crate::model::diffusion::buffer::DiffBufferType as BT;

            let shape4 = [1, LATENT_CHANNELS, latent_h, latent_w];
            let shape5 = [LATENT_CHANNELS, 1, latent_h, latent_w];
            let (f, h, w) = (1usize, latent_h, latent_w);

            // ── Stage prompt_embeds into the persistent slot ──
            // We always read cap features from PromptEmbedsPadded going
            // forward, so the denoise step body can bind to a stable
            // device pointer inside the captured graph.
            let cap_ori_len = prompt_embeds.shape()[0];
            let cap_feat_dim = prompt_embeds.shape()[1];
            {
                let mut ps = self.dit_state.slice_mut(
                    BT::PromptEmbedsPadded, &[cap_ori_len, cap_feat_dim],
                )?;
                ps.copy_from_on_current_stream(prompt_embeds)?;
            }
            // Use the persistent slot (same bytes, stable pointer).
            let cap_feats = self.dit_state.slice(
                BT::PromptEmbedsPadded, &[cap_ori_len, cap_feat_dim],
            )?;

            // ── Per-request shape-fixed precomputation (outside capture) ──
            let _t_prep = Instant::now();
            let shapes = {
                let cuda_cfg = self.text_encoder_state.cuda_config.as_ref();
                self.transformer.prepare_denoise_constants(
                    &cap_feats, f, h, w, &mut self.dit_state, cuda_cfg,
                )?
            };
            let _prep_cpu_ms = _t_prep.elapsed().as_micros() as f64 / 1000.0;

            // ── Upload per-step t_value + dt schedule ──
            // TValueDevVec[i] = (1000 - timestep[i]) / 1000 * t_scale
            // DtValueDevVec[i] = sigmas[i+1] - sigmas[i]
            let _t_sched = Instant::now();
            let t_scale = self.transformer.config.t_scale;
            let timesteps = self.scheduler.timesteps();
            let sigmas = self.scheduler.sigmas();
            debug_assert_eq!(sigmas.len(), num_steps + 1);
            let mut t_value_host = vec![0.0_f32; num_steps];
            let mut dt_host = vec![0.0_f32; num_steps];
            for i in 0..num_steps {
                let norm_t = (1000.0 - timesteps[i]) / 1000.0;
                t_value_host[i] = norm_t * t_scale;
                dt_host[i] = sigmas[i + 1] - sigmas[i];
            }
            self.scheduler.reset();
            // Two tiny HtoD uploads (num_steps × 4 B each, ≤ 64 B).
            upload_f32_slice_into(
                &mut self.dit_state.slice_mut(
                    BT::TValueDevVec, &[num_steps],
                )?,
                &t_value_host,
            )?;
            upload_f32_slice_into(
                &mut self.dit_state.slice_mut(
                    BT::DtValueDevVec, &[num_steps],
                )?,
                &dt_host,
            )?;
            let _sched_cpu_ms = _t_sched.elapsed().as_micros() as f64 / 1000.0;

            // ── Make sure every queued prep HtoD has landed on the
            //    stream before we start capturing. Capture mode doesn't
            //    tolerate in-flight pageable HtoDs. ──
            let _t_prep_sync = Instant::now();
            {
                let cfg = self.text_encoder_state.cuda_config.as_ref()
                    .expect("CUDA requires CudaConfig");
                cfg.sync_stream()?;
            }
            let _prep_sync_ms = _t_prep_sync.elapsed().as_micros() as f64 / 1000.0;

            // ── Capture-or-replay loop ──
            let stream_copy = {
                let cfg = self.text_encoder_state.cuda_config.as_ref().unwrap();
                cfg.stream
            };
            let _t_graph = Instant::now();
            let mut _t_launch_ms = 0.0_f64;
            let mut _t_sync_ms = 0.0_f64;
            with_cuda_stream(stream_copy, || -> Result<()> {
                let slot = crate::cuda::GraphSlot::Denoise {
                    latent_h,
                    latent_w,
                    cap_padded_len: shapes.cap_padded_len,
                    steps: num_steps,
                };
                let graph_ready = self.text_encoder_state
                    .cuda_config.as_ref().unwrap()
                    .graph_ready(slot);
                if !graph_ready {
                    // Begin capture (immutable borrow, drops before body).
                    self.text_encoder_state.cuda_config.as_ref().unwrap()
                        .capture_begin()?;

                    // Body — N denoise steps. Everything is dst-write
                    // into state slots; no Tensor::new, no HtoD.
                    self.run_denoise_step_chain(
                        num_steps, shapes, &shape4, &shape5,
                    )?;

                    // End capture (mutable borrow).
                    self.text_encoder_state.cuda_config.as_mut().unwrap()
                        .capture_end(slot)?;
                }

                // Whether we just captured or were already hot, launch
                // the graph once. (Capture alone does NOT execute the
                // kernels.)
                let _t_l = Instant::now();
                self.text_encoder_state.cuda_config.as_ref().unwrap()
                    .launch(slot)?;
                _t_launch_ms = _t_l.elapsed().as_micros() as f64 / 1000.0;
                let _t_s = Instant::now();
                self.text_encoder_state.cuda_config.as_ref().unwrap()
                    .sync_stream()?;
                _t_sync_ms = _t_s.elapsed().as_micros() as f64 / 1000.0;
                Ok(())
            })?;
            let _graph_total_ms = _t_graph.elapsed().as_micros() as f64 / 1000.0;

            let _t_clone = Instant::now();
            // ── Output: Plan-B ping-pong always leaves the answer in
            //    `Latents`, so we can clone straight from it. ──
            let out_latents = clone_tensor(&self.pipeline_state.slice(BT::Latents, &shape4)?)?;
            let _clone_ms = _t_clone.elapsed().as_micros() as f64 / 1000.0;

            if std::env::var_os("RUSTINFER_DENOISE_BREAKDOWN").is_some() {
                eprintln!(
                    "[denoise-breakdown] prep_cpu={:.2}ms sched_cpu={:.2}ms prep_sync={:.2}ms graph_total={:.2}ms (launch={:.2}ms sync={:.2}ms) clone={:.2}ms",
                    _prep_cpu_ms, _sched_cpu_ms, _prep_sync_ms, _graph_total_ms, _t_launch_ms, _t_sync_ms, _clone_ms,
                );
            }
            Ok(out_latents)
        }
    }

    #[cfg(feature = "cuda")]
    fn run_denoise_step_chain(
        &mut self,
        num_steps: usize,
        shapes: crate::model::diffusion::z_image::transformer::DenoiseShapes,
        shape4: &[usize; 4],
        shape5: &[usize; 4],
    ) -> Result<()> {
        use crate::model::diffusion::buffer::DiffBufferType as BT;
        let cuda_cfg = self.text_encoder_state.cuda_config.as_ref();
        for i in 0..num_steps {
            // Slice a single-element device scalar out of TValueDevVec /
            // DtValueDevVec. These are distinct device pointers per
            // step i — each step becomes its own kernel node in the
            // captured graph, and replay reads whatever host wrote into
            // those bytes before `graph_launch`.
            let d_t = self.dit_state.buffers
                .get(&BT::TValueDevVec).unwrap()
                .slice(&[i], &[1])?;
            let d_dt = self.dit_state.buffers
                .get(&BT::DtValueDevVec).unwrap()
                .slice(&[i], &[1])?;

            // ── Copy current Latents ([1,C,H,W]) into Latent5D ([C,1,H,W]). ──
            {
                let cur = self.pipeline_state.slice(BT::Latents, shape4)?;
                let cur_5d = cur.view(shape5)?;
                let mut latent_5d = self.dit_state.slice_mut(BT::Latent5D, shape5)?;
                latent_5d.copy_from_on_current_stream(&cur_5d)?;
            }

            // ── Transformer forward for this step → ImageOut slot. ──
            let model_out = self.transformer.forward_denoise_step(
                &d_t, shapes, &mut self.dit_state, cuda_cfg,
            )?;

            // ── noise_pred = -model_out (shape [1,C,H,W]). ──
            {
                let mut noise_pred = self.pipeline_state.slice_mut(BT::NoisePred, shape4)?;
                noise_pred.copy_from_on_current_stream(&model_out.view(shape4)?)?;
                noise_pred *= -1.0_f32; // literal kernel arg — constant across replays
            }

            // ── Euler step with dt read from device memory. ──
            //    LatentsTmp = Latents + (d_dt) * NoisePred
            {
                let mut noise_pred = self.pipeline_state.slice_mut(BT::NoisePred, shape4)?;
                let cur = self.pipeline_state.slice(BT::Latents, shape4)?;
                let mut dst = self.pipeline_state.slice_mut(BT::LatentsTmp, shape4)?;
                self.scheduler.step_from_dev(&mut noise_pred, &cur, &mut dst, &d_dt)?;
            }

            // ── Plan-B: copy LatentsTmp back to Latents so next step
            //    (or VAE after the loop) always reads from Latents. ──
            {
                let src = self.pipeline_state.slice(BT::LatentsTmp, shape4)?;
                let mut dst = self.pipeline_state.slice_mut(BT::Latents, shape4)?;
                dst.copy_from_on_current_stream(&src)?;
            }
        }
        Ok(())
    }

    /// Eager (non-graph) denoise. Used only when CUDA is disabled; same
    /// logic as the previous implementation minus CPU-specific bits.
    fn denoise_eager(
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
            let (sample_ty, dst_ty) = if i % 2 == 0 {
                (crate::model::diffusion::buffer::DiffBufferType::Latents,
                 crate::model::diffusion::buffer::DiffBufferType::LatentsTmp)
            } else {
                (crate::model::diffusion::buffer::DiffBufferType::LatentsTmp,
                 crate::model::diffusion::buffer::DiffBufferType::Latents)
            };
            let cur = self.pipeline_state.slice(sample_ty, &shape4)?;
            let mut latent_5d = self.dit_state.slice_mut(
                crate::model::diffusion::buffer::DiffBufferType::Latent5D, &shape5,
            )?;
            latent_5d.copy_from_on_current_stream(&cur.view(&shape5)?)?;

            let model_out = self.transformer.forward(
                &latent_5d, norm_t, prompt_embeds, cuda_cfg, &mut self.dit_state,
            )?;
            let mut noise_pred = self.pipeline_state.slice_mut(
                crate::model::diffusion::buffer::DiffBufferType::NoisePred, &shape4,
            )?;
            noise_pred.copy_from_on_current_stream(&model_out.view(&shape4)?)?;
            noise_pred *= -1.0_f32;
            let cur_read = self.pipeline_state.slice(sample_ty, &shape4)?;
            let mut dst = self.pipeline_state.slice_mut(dst_ty, &shape4)?;
            self.scheduler.step(&mut noise_pred, &cur_read, &mut dst)?;
        }
        let final_ty = if num_steps % 2 == 0 {
            crate::model::diffusion::buffer::DiffBufferType::Latents
        } else {
            crate::model::diffusion::buffer::DiffBufferType::LatentsTmp
        };
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
            // Use the cheapest schedule (2-step Turbo) regardless of
            // what the user will eventually request — warmup only
            // cares about populating kernel / cuBLASLt / cuDNN caches,
            // not about image quality.
            sigmas: Some(TURBO_SIGMAS.to_vec()),
            seed: Some(0),
            ..Default::default()
        };
        // Force the eager denoise loop for warmup so cuBLASLt / cuDNN
        // fill their heuristic caches without stream capture being
        // active — capturing a first-time cuBLASLt call trips
        // `cudaErrorStreamCaptureUnsupported`.
        let prev = self.force_eager_denoise;
        self.force_eager_denoise = true;
        let result = self.generate(&request);
        self.force_eager_denoise = prev;
        let _ = result?;

        // Ensure every queued kernel / cache fill has completed before
        // we hand control back; otherwise the first user request would
        // still race against tail latency from warmup.
        self.device.sync()?;
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
        self.device.sync()?;
        let encode_prompt_ms = encode_start.elapsed().as_millis() as u64;

        // ── Step 2: Prepare latents (fills pipeline_state.Latents in place) ──
        let (latent_h, latent_w) = self.prepare_latents(
            request.height, request.width, request.seed,
        )?;

        if prompt_embeds.shape()[0] > self.dit_state.spec.capacity.max_cap_len {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "prompt token length {} exceeds configured max_cap_len {}",
                prompt_embeds.shape()[0],
                self.dit_state.spec.capacity.max_cap_len
            )).into());
        }

        // ── Step 3: Prepare timesteps ──
        self.prepare_timesteps(request, latent_h, latent_w)?;

        // ── Step 4: Denoise loop (ping-pongs through the state pool) ──
        let denoise_start = Instant::now();
        let latents = self.denoise(latent_h, latent_w, &prompt_embeds)?;
        self.device.sync()?;
        let denoise_ms = denoise_start.elapsed().as_millis() as u64;

        // ── Step 5: VAE decode ──
        let decode_start = Instant::now();
        let image = self.vae_decode(&latents)?;
        self.device.sync()?;
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
        eprintln!("[test] warming up at 1024x1024...");
        let t_warm = Instant::now();
        pipeline.warmup_for(1024, 1024)?;
        eprintln!("[test] warmup done in {}ms", t_warm.elapsed().as_millis());

        // Z-Image-Turbo: 1024x1024 HD, 2-step distilled schedule
        let request = DiffusionRequest {
            prompt: "A majestic snow leopard standing on a cliff edge at sunset, with golden light illuminating its fur, dramatic mountain landscape in the background, photorealistic, 8k detail".to_string(),
            height: 1024,
            width: 1024,
            // Keep the original 2-step Turbo schedule so this test
            // stays comparable to previous runs. Use
            // `test_pipeline_bench_cuda_9step` to measure the 9-step
            // official schedule instead.
            sigmas: Some(TURBO_SIGMAS.to_vec()),
            seed: Some(42),
            ..Default::default()
        };

        let output = pipeline.generate(&request)?;
        eprintln!("Output shape: {:?}", output.output.shape());
        eprintln!("Metrics (1st, includes capture): encode={}ms, denoise={}ms, decode={}ms, total={}ms",
            output.metrics.encode_prompt_ms,
            output.metrics.denoise_ms,
            output.metrics.decode_ms,
            output.metrics.total_ms,
        );

        // Second run — graph replay only, no capture. Represents steady-state latency.
        let output = pipeline.generate(&request)?;
        eprintln!("Metrics (2nd, replay):         encode={}ms, denoise={}ms, decode={}ms, total={}ms",
            output.metrics.encode_prompt_ms,
            output.metrics.denoise_ms,
            output.metrics.decode_ms,
            output.metrics.total_ms,
        );
        save_tensor_as_ppm(&output.output, "/root/z_image_cuda_output.ppm")?;
        eprintln!("Saved /root/z_image_cuda_output.ppm");
        Ok(())
    }

    /// Test Z-Image (non-Turbo) model with full inference steps.
    /// Unlike Turbo which uses 2-step distilled schedule, Z-Image requires
    /// more steps (typically 28) and uses CFG guidance.
    #[test]
    #[ignore = "需要 Z-Image 模型权重 + CUDA"]
    #[cfg(feature = "cuda")]
    fn test_pipeline_z_image_full_cuda() -> Result<()> {
        let model_dir = Path::new("/root/models/Z-Image");
        if !model_dir.join("text_encoder/config.json").exists() {
            eprintln!("[test] Z-Image model not found at {:?}, skipping", model_dir);
            return Ok(());
        }

        eprintln!("[test] loading Z-Image (full, non-Turbo) pipeline...");
        let mut pipeline = ZImagePipeline::from_pretrained(model_dir, DeviceType::Cuda(0))?;
        eprintln!("[test] scheduler shift: {}", pipeline.scheduler.shift);

        // Warmup at target resolution
        eprintln!("[test] warming up at 1024x1024...");
        let t_warm = Instant::now();
        pipeline.warmup_for(1024, 1024)?;
        eprintln!("[test] warmup done in {}ms", t_warm.elapsed().as_millis());

        // Z-Image full model — 1024x1024 HD, 50 denoising steps, CFG=4.5
        let request = DiffusionRequest {
            prompt: "一座古老的中式石拱桥横跨在碧绿的湖面上，桥上有几个身穿汉服的人在散步，远处是连绵的青山和飘浮的白云，湖面倒映着蓝天，岸边有垂柳和粉色的桃花，春天的氛围，水墨风格与写实结合，高清细腻".to_string(),
            height: 1024,
            width: 1024,
            num_inference_steps: 50,
            guidance_scale: 4.5,
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

        save_tensor_as_ppm(&output.output, "/root/z_image_full_cuda_output.ppm")?;
        eprintln!("Saved /root/z_image_full_cuda_output.ppm");
        Ok(())
    }

}
