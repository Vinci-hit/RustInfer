//! Diffusion pipeline abstraction.
//!
//! All diffusion pipelines (text-to-image, TTS, video) implement the
//! [`DiffusionPipeline`] trait, providing a uniform interface for the
//! engine layer.

use crate::base::error::Result;
use crate::tensor::Tensor;

// ───────────────────────────── Request / Response ─────────────────────────────

/// Generation request for a diffusion pipeline.
///
/// Different modalities populate different fields:
/// - **text-to-image**: `prompt`, `height`, `width`
/// - **TTS** (future): `prompt` (text), `sample_rate`, etc.
/// - **video** (future): `prompt`, `height`, `width`, `num_frames`
#[derive(Debug, Clone)]
pub struct DiffusionRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,

    // Image / video dimensions
    pub height: usize,
    pub width: usize,

    // Sampling control
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
}

impl Default for DiffusionRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            height: 1024,
            width: 1024,
            num_inference_steps: 8,
            guidance_scale: 0.0,
            seed: None,
        }
    }
}

/// Output produced by a diffusion pipeline.
#[derive(Debug)]
pub struct DiffusionOutput {
    /// The generated tensor.
    /// - **text-to-image**: `[B, C, H, W]` RGB float in `[0, 1]`
    /// - **TTS** (future): `[B, T]` waveform
    pub output: Tensor,
    pub metrics: DiffusionMetrics,
}

/// Performance counters for a single generation call.
#[derive(Debug, Clone, Default)]
pub struct DiffusionMetrics {
    pub encode_prompt_ms: u64,
    pub denoise_ms: u64,
    pub decode_ms: u64,
    pub total_ms: u64,
}

// ───────────────────────────── Trait ──────────────────────────────────────────

/// Unified interface for all diffusion-family models.
///
/// The engine layer calls [`generate`] without caring whether the
/// underlying model produces images, audio, or video.
pub trait DiffusionPipeline: Send {
    /// Human-readable name, e.g. `"z-image-turbo"`.
    fn name(&self) -> &str;

    /// Run the full pipeline: encode → denoise → decode.
    fn generate(&mut self, request: &DiffusionRequest) -> Result<DiffusionOutput>;
}
