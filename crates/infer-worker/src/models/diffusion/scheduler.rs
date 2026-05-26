//! FlowMatchEulerScheduler — Euler discrete diffusion scheduler.
//!
//! Pure math, no device/tensor ops. Operates on Vec<f32> sigma schedules.

/// Flow-matching Euler scheduler for Z-Image.
pub struct FlowMatchEulerScheduler {
    pub num_train_timesteps: usize,
    pub shift: f32,
}

impl FlowMatchEulerScheduler {
    pub fn new(num_train_timesteps: usize, shift: f32) -> Self {
        Self { num_train_timesteps, shift }
    }

    /// Compute the sigma schedule for N inference steps.
    /// Returns (sigmas, timesteps) where sigmas[i] = noise level at step i.
    pub fn set_timesteps(&self, num_steps: usize) -> (Vec<f32>, Vec<f32>) {
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        for i in 0..=num_steps {
            let t = 1.0 - (i as f32 / num_steps as f32);
            // Apply shift: sigma = shift * t / (1 + (shift - 1) * t)
            let sigma = self.shift * t / (1.0 + (self.shift - 1.0) * t);
            sigmas.push(sigma);
        }
        let timesteps: Vec<f32> = sigmas[..num_steps].iter().map(|s| s * self.num_train_timesteps as f32).collect();
        (sigmas, timesteps)
    }

    /// Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * model_output
    /// Returns dt = sigma_{t-1} - sigma_t for this step.
    pub fn step_dt(&self, sigmas: &[f32], step_idx: usize) -> f32 {
        sigmas[step_idx + 1] - sigmas[step_idx]
    }

    /// Apply dynamic shifting based on image resolution.
    pub fn shift_for_resolution(base_shift: f32, image_tokens: usize, base_tokens: usize) -> f32 {
        let ratio = (image_tokens as f32) / (base_tokens as f32);
        base_shift * ratio.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_produces_monotonic_sigmas() {
        let sched = FlowMatchEulerScheduler::new(1000, 3.0);
        let (sigmas, _) = sched.set_timesteps(20);
        // Sigmas should be monotonically decreasing (from ~1 to 0)
        for i in 1..sigmas.len() {
            assert!(sigmas[i] <= sigmas[i - 1], "sigma[{}]={} > sigma[{}]={}", i, sigmas[i], i-1, sigmas[i-1]);
        }
        assert!(sigmas.last().unwrap().abs() < 1e-6, "last sigma should be ~0");
    }

    #[test]
    fn scheduler_dt_negative() {
        let sched = FlowMatchEulerScheduler::new(1000, 3.0);
        let (sigmas, _) = sched.set_timesteps(10);
        // dt should be negative (decreasing noise)
        for i in 0..10 {
            let dt = sched.step_dt(&sigmas, i);
            assert!(dt <= 0.0, "dt[{}] = {} should be <= 0", i, dt);
        }
    }
}
