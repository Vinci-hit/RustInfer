//! Flow-Matching Euler discrete scheduler for Z-Image / Z-Image-Turbo.
//!
//! Mirrors `diffusers.FlowMatchEulerDiscreteScheduler` in eager mode, plus a
//! CUDA-Graph-friendly `step_from_dev` variant that reads `dt` from a
//! device-side scalar.
//!
//! ## Schedule construction
//! - [`set_timesteps_default`] — the default Flow-Match schedule for `N`
//!   inference steps. Static shift `σ' = shift·σ / (1+(shift−1)·σ)` when
//!   no `image_seq_len` is supplied (Z-Image-Turbo's case, which sets
//!   `use_dynamic_shifting=False`); dynamic shift via `μ = calculate_shift`
//!   otherwise.
//! - [`set_timesteps_from_sigmas`] — accept an explicit `[σ_0, σ_1, …]`
//!   list (e.g. Turbo's `[1.0, 0.3]`).
//!
//! Both append a trailing `0.0` so the last Euler step integrates all the
//! way to zero noise.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Dtype;

/// Flow Matching with Euler discrete steps.
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

    /// Install an explicit sigma list. Length = number of denoising steps.
    /// Internally appends a trailing `0.0` so `step()` can compute the
    /// final `sigma_next - sigma_cur` correctly.
    pub fn set_timesteps_from_sigmas(&mut self, sigmas: &[f32]) {
        self.sigmas = sigmas.to_vec();
        self.sigmas.push(0.0);
        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();
        self.step_index = 0;
    }

    /// Build the default Flow-Match sigma schedule.
    ///
    /// 1. Linear in t-space: `t = linspace(N_train, 0, N+1)[..N]`
    ///    (we collapse to `1.0 → sigma_min` directly since the t↔σ map is
    ///    the identity for Flow-Match).
    /// 2. Apply shift:
    ///    - dynamic if `image_seq_len.is_some()`:
    ///      `σ' = e^μ / (e^μ + (1/σ − 1))` with
    ///      `μ = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)`
    ///    - static otherwise: `σ' = shift·σ / (1 + (shift−1)·σ)`
    /// 3. `timesteps[i] = σ'[i] * num_train_timesteps`.
    /// 4. Append trailing `0.0` to sigmas.
    pub fn set_timesteps_default(
        &mut self,
        num_inference_steps: usize,
        image_seq_len: Option<usize>,
    ) {
        assert!(num_inference_steps > 0, "num_inference_steps must be >= 1");
        let n = num_inference_steps;
        let sigma_min = 1.0_f32 / self.num_train_timesteps as f32;

        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let t = if n == 1 {
                    0.0
                } else {
                    i as f32 / (n - 1) as f32
                };
                1.0 + t * (sigma_min - 1.0)
            })
            .collect();

        match image_seq_len {
            Some(seq_len) => {
                let mu = calculate_shift(seq_len, 256, 4096, 0.5, 1.15);
                let emu = mu.exp();
                for s in sigmas.iter_mut() {
                    *s = emu / (emu + (1.0 / *s - 1.0));
                }
            }
            None => {
                let shift = self.shift;
                for s in sigmas.iter_mut() {
                    *s = shift * *s / (1.0 + (shift - 1.0) * *s);
                }
            }
        }

        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();

        sigmas.push(0.0);
        self.sigmas = sigmas;
        self.step_index = 0;
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }
    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    /// Single Euler denoising step with the scheduler's internal counter.
    ///
    /// `dst = sample + (σ_next − σ_cur) * model_output`.
    /// **Destructive:** `model_output` is scaled in place by `dt`.
    /// `sample` and `dst` must be distinct buffers.
    pub fn step<T: Dtype, D: OpBackend>(
        &mut self,
        model_output: &mut Tensor<T, D>,
        sample: &Tensor<T, D>,
        dst: &mut Tensor<T, D>,
    ) -> OpResult<()> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = (sigma_next - sigma) as f64;

        // model_output *= dt
        D::scalar_mul_inplace(model_output, dt)?;
        // dst = sample
        dst.copy_from(sample)?;
        // dst += model_output
        D::add_inplace(dst, model_output)?;

        self.step_index += 1;
        Ok(())
    }

    /// CUDA-Graph friendly variant: `dt` is a single-element f32 device tensor
    /// the host writes between launches. Does *not* advance `step_index`.
    pub fn step_from_dev<T: Dtype, D: OpBackend>(
        &mut self,
        model_output: &mut Tensor<T, D>,
        sample: &Tensor<T, D>,
        dst: &mut Tensor<T, D>,
        d_dt: &Tensor<f32, D>,
    ) -> OpResult<()> {
        D::scalar_mul_inplace_from_dev(model_output, d_dt)?;
        dst.copy_from(sample)?;
        D::add_inplace(dst, model_output)?;
        Ok(())
    }
}

/// Linearly interpolate between `base_shift` and `max_shift` as
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn turbo_explicit_sigmas() {
        let mut s = FlowMatchEulerScheduler::new(1000, 3.0);
        s.set_timesteps_from_sigmas(&[1.0, 0.3]);
        assert_eq!(s.timesteps(), &[1000.0, 300.0]);
        assert_eq!(s.sigmas(), &[1.0, 0.3, 0.0]);
        assert_eq!(s.num_steps(), 2);
    }

    #[test]
    fn default_static_shift_3_for_z_image_turbo() {
        let mut s = FlowMatchEulerScheduler::new(1000, 3.0);
        s.set_timesteps_default(9, None);
        assert_eq!(s.num_steps(), 9);
        assert_eq!(s.sigmas().len(), 10);
        // Trailing zero.
        assert_eq!(*s.sigmas().last().unwrap(), 0.0);
        // First sigma after shift: input was 1.0 → 3*1/(1+2*1) = 1.0.
        assert!((s.sigmas()[0] - 1.0).abs() < 1e-6);
        // Sigmas monotonically decreasing.
        for i in 1..s.sigmas().len() {
            assert!(
                s.sigmas()[i] <= s.sigmas()[i - 1] + 1e-6,
                "non-monotonic at {}: {} > {}",
                i,
                s.sigmas()[i],
                s.sigmas()[i - 1]
            );
        }
        // timesteps[i] = sigma[i] * 1000.
        for i in 0..s.num_steps() {
            let expected = s.sigmas()[i] * 1000.0;
            assert!((s.timesteps()[i] - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn default_dynamic_shift_uses_image_seq_len() {
        // diffusers `calculate_shift(seq=4096, base=256, max=4096, b=0.5, m=1.15)`:
        // mu_at_max == max_shift == 1.15.
        let mut s = FlowMatchEulerScheduler::new(1000, 3.0); // shift ignored
        s.set_timesteps_default(4, Some(4096));
        let mu = calculate_shift(4096, 256, 4096, 0.5, 1.15);
        assert!((mu - 1.15).abs() < 1e-6);
        // sigmas[0] should be e^μ / (e^μ + (1/1 - 1)) = 1.0.
        assert!((s.sigmas()[0] - 1.0).abs() < 1e-6);
        // Last meaningful sigma > 0, then trailing 0.
        assert!(s.sigmas()[s.num_steps() - 1] > 0.0);
        assert_eq!(s.sigmas()[s.num_steps()], 0.0);
    }

    #[test]
    fn calculate_shift_linear_endpoints() {
        assert!((calculate_shift(256, 256, 4096, 0.5, 1.15) - 0.5).abs() < 1e-6);
        assert!((calculate_shift(4096, 256, 4096, 0.5, 1.15) - 1.15).abs() < 1e-6);
    }

    #[test]
    fn euler_step_two_step_turbo_cpu_f32() {
        let dev = Cpu;
        let mut s = FlowMatchEulerScheduler::new(1000, 3.0);
        s.set_timesteps_from_sigmas(&[1.0, 0.3]);
        // sigmas = [1.0, 0.3, 0.0]
        // step 0 dt = -0.7
        // step 1 dt = -0.3
        let sample: Tensor<f32, Cpu> =
            Tensor::from_host_slice(&[1.0, 2.0, 3.0, 4.0], [4], &dev).unwrap();
        let mut velocity: Tensor<f32, Cpu> =
            Tensor::from_host_slice(&[10.0; 4], [4], &dev).unwrap();
        let mut dst_a: Tensor<f32, Cpu> = Tensor::zeros([4], &dev).unwrap();
        let mut dst_b: Tensor<f32, Cpu> = Tensor::zeros([4], &dev).unwrap();
        s.step(&mut velocity, &sample, &mut dst_a).unwrap();
        let r1 = dst_a.to_host_vec().unwrap();
        // [1,2,3,4] + (-0.7)*[10,10,10,10] = [-6,-5,-4,-3]
        assert!((r1[0] - (-6.0)).abs() < 1e-5);
        assert!((r1[3] - (-3.0)).abs() < 1e-5);
        // Reset velocity (step() destructively scales it).
        let mut velocity: Tensor<f32, Cpu> =
            Tensor::from_host_slice(&[10.0; 4], [4], &dev).unwrap();
        s.step(&mut velocity, &dst_a, &mut dst_b).unwrap();
        let r2 = dst_b.to_host_vec().unwrap();
        // [-6,-5,-4,-3] + (-0.3)*[10,10,10,10] = [-9,-8,-7,-6]
        assert!((r2[0] - (-9.0)).abs() < 1e-5);
        assert!((r2[3] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn euler_full_denoise_to_zero() {
        // Single step σ:1→0 with velocity = sample → output is zero.
        let dev = Cpu;
        let mut s = FlowMatchEulerScheduler::new(1000, 3.0);
        s.set_timesteps_from_sigmas(&[1.0]);
        let noise: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let sample: Tensor<f32, Cpu> =
            Tensor::from_host_slice(&noise, [noise.len()], &dev).unwrap();
        let mut velocity: Tensor<f32, Cpu> =
            Tensor::from_host_slice(&noise, [noise.len()], &dev).unwrap();
        let mut dst: Tensor<f32, Cpu> = Tensor::zeros([noise.len()], &dev).unwrap();
        s.step(&mut velocity, &sample, &mut dst).unwrap();
        for v in dst.to_host_vec().unwrap() {
            assert!(v.abs() < 1e-6, "expected ~0, got {}", v);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn euler_step_cuda_bf16_matches_cpu_f32() {
        use crate::infrastructure::cuda::Cuda;
        use half::bf16;
        let cpu = Cpu;
        let cuda = Cuda::new(0).unwrap();
        let mut s_cpu = FlowMatchEulerScheduler::new(1000, 3.0);
        let mut s_gpu = FlowMatchEulerScheduler::new(1000, 3.0);
        s_cpu.set_timesteps_from_sigmas(&[1.0, 0.3]);
        s_gpu.set_timesteps_from_sigmas(&[1.0, 0.3]);

        // CPU reference: f32. Use N=16 (multiple of 8 for bf16 vec kernels).
        let n = 16usize;
        let sample_host: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let vel_host: Vec<f32> = vec![10.0; n];
        let cpu_sample: Tensor<f32, Cpu> =
            Tensor::from_host_slice(&sample_host, [n], &cpu).unwrap();
        let mut cpu_vel: Tensor<f32, Cpu> = Tensor::from_host_slice(&vel_host, [n], &cpu).unwrap();
        let mut cpu_dst: Tensor<f32, Cpu> = Tensor::zeros([n], &cpu).unwrap();
        s_cpu.step(&mut cpu_vel, &cpu_sample, &mut cpu_dst).unwrap();
        let cpu_r = cpu_dst.to_host_vec().unwrap();

        // GPU bf16.
        let sample_bf16: Vec<bf16> = sample_host.iter().map(|x| bf16::from_f32(*x)).collect();
        let vel_bf16: Vec<bf16> = vel_host.iter().map(|x| bf16::from_f32(*x)).collect();
        let gpu_sample: Tensor<bf16, Cuda> =
            Tensor::from_host_slice(&sample_bf16, [n], &cuda).unwrap();
        let mut gpu_vel: Tensor<bf16, Cuda> =
            Tensor::from_host_slice(&vel_bf16, [n], &cuda).unwrap();
        let mut gpu_dst: Tensor<bf16, Cuda> = Tensor::zeros([n], &cuda).unwrap();
        s_gpu.step(&mut gpu_vel, &gpu_sample, &mut gpu_dst).unwrap();
        let gpu_r: Vec<f32> = gpu_dst
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();

        for (a, b) in cpu_r.iter().zip(gpu_r.iter()) {
            assert!((a - b).abs() < 0.5, "cpu={} gpu={}", a, b);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn euler_step_from_dev_cuda_matches_normal_step() {
        use crate::infrastructure::cuda::Cuda;
        let cuda = Cuda::new(0).unwrap();
        let mut s = FlowMatchEulerScheduler::new(1000, 3.0);
        s.set_timesteps_from_sigmas(&[1.0, 0.3]);

        let sample_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let vel_host: Vec<f32> = vec![10.0; 4];
        let sample: Tensor<f32, Cuda> = Tensor::from_host_slice(&sample_host, [4], &cuda).unwrap();
        let mut vel: Tensor<f32, Cuda> = Tensor::from_host_slice(&vel_host, [4], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([4], &cuda).unwrap();

        // step_from_dev: dt at step 0 = sigmas[1] - sigmas[0] = -0.7.
        let d_dt: Tensor<f32, Cuda> = Tensor::from_host_slice(&[-0.7_f32], [1], &cuda).unwrap();
        s.step_from_dev(&mut vel, &sample, &mut dst, &d_dt).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert!((got[0] - (-6.0)).abs() < 1e-5);
        assert!((got[3] - (-3.0)).abs() < 1e-5);
    }
}
