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
    /// Prepare timesteps from explicit sigma values (used by Turbo models
    /// or whenever a caller wants to override the default schedule).
    ///
    /// Length is the number of denoising steps; `set_timesteps_from_sigmas`
    /// will append a trailing 0 internally so the final destination is
    /// exactly zero noise.
    fn set_timesteps_from_sigmas(&mut self, sigmas: &[f32]);

    /// Prepare timesteps using the scheduler's **built-in default**
    /// schedule — i.e. the one the original reference pipeline would
    /// pick when the user only says "I want N inference steps".
    ///
    /// Implementations that need extra context (e.g. image sequence
    /// length for Flow-Match dynamic shifting) accept it via
    /// `image_seq_len`; schedulers that ignore it (e.g. DDPM) simply
    /// drop the argument.
    fn set_timesteps_default(
        &mut self,
        num_inference_steps: usize,
        image_seq_len: Option<usize>,
    );

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
    /// Writes `dst = sample + (sigma_next - sigma_cur) * model_output`.
    ///
    /// **Destructive**: `model_output` is mutated in place (scaled by
    /// `dt`) so the step can run without any scratch allocation. Callers
    /// that still need the original velocity must copy it before calling.
    /// `sample` and `dst` must be distinct tensors (no aliasing) — the
    /// typical pattern is a ping-pong between two pipeline-owned buffers.
    fn step(
        &mut self,
        model_output: &mut Tensor,
        sample: &Tensor,
        dst: &mut Tensor,
    ) -> Result<()>;
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

    /// Build the **default** Flow-Match sigma schedule for a given
    /// inference-step count. Mirrors `diffusers.FlowMatchEulerDiscreteScheduler.set_timesteps`:
    ///
    /// 1. `sigmas = linspace(1.0, 1/num_train_timesteps, N)` — even
    ///    distribution in t-space (we collapse diffusers' `_sigma_to_t`
    ///    round-trip since it's the identity for Flow-Match).
    /// 2. If `image_seq_len` is provided, compute the adaptive
    ///    `mu = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)`
    ///    (Z-Image / Flux / SD3 defaults) and apply **dynamic shifting**:
    ///      `sigmas = exp(mu) / (exp(mu) + (1/sigma - 1))`.
    /// 3. Otherwise apply **static shifting** with `self.shift`:
    ///      `sigmas = shift * sigma / (1 + (shift - 1) * sigma)`.
    ///
    /// A trailing `0.0` is appended so the last Euler step integrates
    /// all the way to zero noise.
    fn set_timesteps_default(
        &mut self,
        num_inference_steps: usize,
        image_seq_len: Option<usize>,
    ) {
        assert!(num_inference_steps > 0, "num_inference_steps must be >= 1");
        let n = num_inference_steps;
        let sigma_min = 1.0_f32 / self.num_train_timesteps as f32;

        // Linear sigma schedule from 1.0 → sigma_min in N steps.
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let t = if n == 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
                1.0 + t * (sigma_min - 1.0)
            })
            .collect();

        // Apply shift. Dynamic if image_seq_len given (Z-Image / Flux
        // convention), otherwise static with the stored `self.shift`.
        match image_seq_len {
            Some(seq_len) => {
                let mu = calculate_shift(seq_len, 256, 4096, 0.5, 1.15);
                let emu = mu.exp();
                for s in sigmas.iter_mut() {
                    // dynamic shift: σ' = e^μ / (e^μ + (1/σ − 1))
                    *s = emu / (emu + (1.0 / *s - 1.0));
                }
            }
            None => {
                let shift = self.shift;
                for s in sigmas.iter_mut() {
                    // static shift: σ' = shift·σ / (1 + (shift−1)·σ)
                    *s = shift * *s / (1.0 + (shift - 1.0) * *s);
                }
            }
        }

        // Timesteps are sigmas mapped back to [0, num_train_timesteps].
        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();

        // Finalise sigma list with trailing zero.
        sigmas.push(0.0);
        self.sigmas = sigmas;
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

    fn step(
        &mut self,
        model_output: &mut Tensor,
        sample: &Tensor,
        dst: &mut Tensor,
    ) -> Result<()> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;

        // dst = sample + dt * model_output
        //   1) model_output *= dt      (in-place scale, no alloc)
        //   2) dst = sample            (stream-ordered async copy)
        //   3) dst += model_output     (in-place add)
        *model_output *= dt;
        dst.copy_from_on_current_stream(sample)?;
        *dst += &*model_output;

        self.step_index += 1;
        Ok(())
    }
}

impl FlowMatchEulerScheduler {
    /// CUDA Graph-friendly variant of [`Scheduler::step`]: the `dt`
    /// coefficient is read from **device** memory rather than passed as
    /// a kernel-parameter `f32`. This keeps all per-step-varying data
    /// outside the graph's kernel arguments — the caller writes
    /// `d_dt[0]` via `cudaMemcpyAsync` before each replay, everything
    /// else is captured once.
    ///
    /// `d_dt` must be a `[1] f32` tensor on the same device as the
    /// sample tensors. The scheduler's internal `step_index` is not
    /// advanced: the caller drives which element of its pre-computed
    /// `DtValueDevVec` each step reads.
    #[cfg(feature = "cuda")]
    pub fn step_from_dev(
        &mut self,
        model_output: &mut Tensor,
        sample: &Tensor,
        dst: &mut Tensor,
        d_dt: &Tensor,
    ) -> Result<()> {
        crate::op::scalar::scalar_mul_inplace_from_dev(model_output, d_dt)?;
        dst.copy_from_on_current_stream(sample)?;
        *dst += &*model_output;
        Ok(())
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

        let mut dst_a = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        let mut dst_b = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;

        // Step 0: prev = [1,2,3,4] + (-0.7) * [10,10,10,10] = [-6, -5, -4, -3]
        sched.step(&mut velocity, &sample, &mut dst_a)?;
        let r1 = dst_a.as_f32()?.as_slice()?;
        assert!((r1[0] - (-6.0)).abs() < 1e-5);
        assert!((r1[1] - (-5.0)).abs() < 1e-5);

        // Reset velocity to original value since step() mutates it.
        velocity.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 10.0, 10.0, 10.0]);

        // Step 1: prev = [-6,-5,-4,-3] + (-0.3) * [10,10,10,10] = [-9, -8, -7, -6]
        sched.step(&mut velocity, &dst_a, &mut dst_b)?;
        let r2 = dst_b.as_f32()?.as_slice()?;
        assert!((r2[0] - (-9.0)).abs() < 1e-5);
        assert!((r2[3] - (-6.0)).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_euler_step_full_denoise() -> Result<()> {
        // 验证：从纯噪声 sigma=1 去噪到 sigma=0，如果 velocity = sample，
        // 则最终结果应为 0 (dt = -1, prev = noise - noise = 0).
        let mut sched = FlowMatchEulerScheduler::new(1000, 3.0);
        sched.set_timesteps_from_sigmas(&[1.0]); // 单步: sigma 1→0, dt = -1

        let mut noise = Tensor::new(&[8], DataType::F32, DeviceType::Cpu)?;
        noise.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);

        // velocity = a copy of noise (since step() will mutate velocity).
        let mut velocity = crate::op::tensor_utils::clone_tensor(&noise)?;

        let mut dst = Tensor::new(&[8], DataType::F32, DeviceType::Cpu)?;
        sched.step(&mut velocity, &noise, &mut dst)?;
        let r = dst.as_f32()?.as_slice()?;
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
        let mut velocity = vel_cpu.to_cuda(0)?;

        let mut dst = Tensor::new(&[4], DataType::F32, DeviceType::Cuda(0))?;
        sched.step(&mut velocity, &sample, &mut dst)?;
        let r = dst.to_cpu()?;
        let r1 = r.as_f32()?.as_slice()?;
        // dt = 0.3 - 1.0 = -0.7, prev = [1,2,3,4] + (-0.7)*[10,10,10,10]
        assert!((r1[0] - (-6.0)).abs() < 1e-5);
        assert!((r1[3] - (-3.0)).abs() < 1e-5);

        Ok(())
    }
}
