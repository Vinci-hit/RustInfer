//! Noise schedulers for diffusion models.
//!
//! Each scheduler defines how to add / remove noise across timesteps.
//! The [`Scheduler`] trait keeps the denoising loop in the pipeline
//! agnostic to the specific scheduling strategy.

use crate::base::error::Result;
use crate::tensor::Tensor;

// ───────────────────────────── Trait ──────────────────────────────────────────

/// Scheduler controls the noise schedule and single-step denoising.
pub trait Scheduler: Send {
    /// Name of the scheduler, e.g. `"flow_match_euler"`.
    fn name(&self) -> &str;

    /// Prepare the timestep sequence for a given number of steps.
    ///
    /// After this call, [`timesteps`] returns the prepared list.
    fn set_timesteps(&mut self, num_inference_steps: usize);

    /// Prepare timesteps from explicit sigma values (used by Turbo models).
    fn set_timesteps_from_sigmas(&mut self, sigmas: &[f32], mu: f32);

    /// Return the current timestep schedule.
    fn timesteps(&self) -> &[f32];

    /// Perform a single denoising step: `x_t → x_{t-1}`.
    ///
    /// - `model_output`: the velocity / noise prediction from the DiT.
    /// - `timestep`: current timestep value.
    /// - `sample`: current noisy latent `x_t`.
    fn step(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
    ) -> Result<Tensor>;
}

// ──────────────────────── Flow Match Euler Discrete ──────────────────────────

/// Flow Matching with Euler discrete steps.
///
/// Used by Z-Image / Z-Image-Turbo.
/// Reference: `diffusers.FlowMatchEulerDiscreteScheduler`
pub struct FlowMatchEulerScheduler {
    pub num_train_timesteps: usize,
    pub shift: f32,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
}

impl FlowMatchEulerScheduler {
    pub fn new(num_train_timesteps: usize, shift: f32) -> Self {
        Self {
            num_train_timesteps,
            shift,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
        }
    }
}

impl Scheduler for FlowMatchEulerScheduler {
    fn name(&self) -> &str {
        "flow_match_euler"
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) {
        let n = num_inference_steps;
        self.sigmas = (0..=n)
            .map(|i| 1.0 - (i as f32 / n as f32))
            .collect();
        self.timesteps = self.sigmas[..n]
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();
    }

    fn set_timesteps_from_sigmas(&mut self, sigmas: &[f32], _mu: f32) {
        // sigmas 直接给定, e.g. [1.0, 0.3] for Turbo 2-step
        self.sigmas = sigmas.to_vec();
        self.sigmas.push(0.0); // append sigma_end = 0
        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn step(
        &self,
        _model_output: &Tensor,
        _timestep: f32,
        _sample: &Tensor,
    ) -> Result<Tensor> {
        // TODO: implement Euler step
        // latent = sample + (sigma_next - sigma_cur) * model_output
        todo!("FlowMatchEulerScheduler::step — will implement with Tensor arithmetic")
    }
}

// ───────────────────────────── Helpers ────────────────────────────────────────

/// Compute adaptive shift `mu` based on image sequence length.
///
/// Linearly interpolates between `base_shift` and `max_shift` as
/// `image_seq_len` goes from `base_seq_len` to `max_seq_len`.
///
/// Reference: `diffusers.pipelines.flux.pipeline_flux.calculate_shift`
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f32;
    let b = base_shift - m * base_seq_len as f32;
    image_seq_len as f32 * m + b
}
