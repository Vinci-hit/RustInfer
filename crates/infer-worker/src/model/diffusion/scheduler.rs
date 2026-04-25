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
    /// Prepare timesteps from explicit sigma values (used by Turbo models).
    fn set_timesteps_from_sigmas(&mut self, sigmas: &[f32]);

    /// Return the current timestep schedule.
    fn timesteps(&self) -> &[f32];

    /// Return the sigma schedule (len = timesteps.len() + 1, last is 0).
    fn sigmas(&self) -> &[f32];

    /// Number of denoising steps.
    fn num_steps(&self) -> usize { self.timesteps().len() }

    /// Reset step counter to 0 (call before each denoising loop).
    fn reset(&mut self);

    /// Perform a single Euler denoising step and advance the internal counter.
    ///
    /// `prev_sample = sample + (sigma_next - sigma_cur) * model_output`
    fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor>;
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
    step_index: usize,
}

impl FlowMatchEulerScheduler {
    pub fn new(num_train_timesteps: usize, shift: f32) -> Self {
        Self {
            num_train_timesteps,
            shift,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            step_index: 0,
        }
    }
}

impl Scheduler for FlowMatchEulerScheduler {
    fn set_timesteps_from_sigmas(&mut self, sigmas: &[f32]) {
        // sigmas 直接给定, e.g. [1.0, 0.3] for Turbo 2-step
        self.sigmas = sigmas.to_vec();
        self.sigmas.push(0.0); // append sigma_end = 0
        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();
        self.step_index = 0;
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    fn reset(&mut self) {
        self.step_index = 0;
    }

    fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;

        // prev_sample = sample + dt * model_output
        let prev_sample = sample + &(model_output * dt);

        self.step_index += 1;
        Ok(prev_sample)
    }
}

// ───────────────────────────── Helpers ────────────────────────────────────────

/// Compute adaptive shift `mu` based on image sequence length.
///
/// Linearly interpolates between `base_shift` and `max_shift` as
/// `image_seq_len` goes from `base_seq_len` to `max_seq_len`.
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

// ───────────────────────────── Tests ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;

    #[test]
    fn test_set_timesteps_from_sigmas_turbo() {
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(&[1.0, 0.3]);

        assert_eq!(sched.timesteps(), &[1000.0, 300.0]);
        assert_eq!(sched.sigmas(), &[1.0, 0.3, 0.0]);
        assert_eq!(sched.num_steps(), 2);
    }

    #[test]
    fn test_euler_step_cpu_f32() -> Result<()> {
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(&[1.0, 0.3]);

        // sigmas = [1.0, 0.3, 0.0]
        // step 0: dt = 0.3 - 1.0 = -0.7
        // step 1: dt = 0.0 - 0.3 = -0.3

        // sample = [1, 2, 3, 4], velocity = [10, 10, 10, 10]
        let mut sample = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        sample.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let mut velocity = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        velocity.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 10.0, 10.0, 10.0]);

        // Step 0: prev = [1,2,3,4] + (-0.7) * [10,10,10,10] = [-6, -5, -4, -3]
        let s1 = sched.step(&velocity, &sample)?;
        let r1 = s1.as_f32()?.as_slice()?;
        assert!((r1[0] - (-6.0)).abs() < 1e-5);
        assert!((r1[1] - (-5.0)).abs() < 1e-5);

        // Step 1: prev = [-6,-5,-4,-3] + (-0.3) * [10,10,10,10] = [-9, -8, -7, -6]
        let s2 = sched.step(&velocity, &s1)?;
        let r2 = s2.as_f32()?.as_slice()?;
        assert!((r2[0] - (-9.0)).abs() < 1e-5);
        assert!((r2[3] - (-6.0)).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_euler_step_full_denoise() -> Result<()> {
        // 验证：从纯噪声 sigma=1 去噪到 sigma=0，如果 velocity = -sample，
        // 则最终结果应为 0
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(&[1.0]); // 单步: sigma 1→0, dt = -1

        let mut noise = Tensor::new(&[8], DataType::F32, DeviceType::Cpu)?;
        noise.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);

        // velocity = sample (flow matching: v predicts direction from noise to clean)
        // prev = noise + (0 - 1) * noise = noise - noise = 0
        let result = sched.step(&noise, &noise)?;
        let r = result.as_f32()?.as_slice()?;
        for &v in r {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }

        Ok(())
    }

    #[test]
    fn test_calculate_shift() {
        let mu = calculate_shift(256, 256, 4096, 0.5, 1.15);
        assert!((mu - 0.5).abs() < 1e-6, "at base_seq_len, should be base_shift");

        let mu = calculate_shift(4096, 256, 4096, 0.5, 1.15);
        assert!((mu - 1.15).abs() < 1e-6, "at max_seq_len, should be max_shift");
    }

    #[test]
    fn test_reset() {
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(&[1.0, 0.5, 0.3]);
        assert_eq!(sched.num_steps(), 3);

        sched.step_index = 2;
        sched.reset();
        assert_eq!(sched.step_index, 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_euler_step_cuda() -> Result<()> {
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(&[1.0, 0.3]);

        let mut sample_cpu = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        sample_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let sample = sample_cpu.to_cuda(0)?;

        let mut vel_cpu = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        vel_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 10.0, 10.0, 10.0]);
        let velocity = vel_cpu.to_cuda(0)?;

        let s1 = sched.step(&velocity, &sample)?;
        let r = s1.to_cpu()?;
        let r1 = r.as_f32()?.as_slice()?;
        // dt = 0.3 - 1.0 = -0.7, prev = [1,2,3,4] + (-0.7)*[10,10,10,10]
        assert!((r1[0] - (-6.0)).abs() < 1e-5);
        assert!((r1[3] - (-3.0)).abs() < 1e-5);

        Ok(())
    }
}
