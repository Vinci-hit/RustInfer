//! Timestep embedding for diffusion models.
//!
//! Converts scalar timestep `t ∈ [0, 1]` into a dense embedding vector
//! via sinusoidal frequency encoding → 2-layer MLP with SiLU activation.
//!
//! ```text
//! t: [B] → ×t_scale → sinusoidal_encoding [B, 256] → MLP [B, 256]
//! ```

use crate::base::{DataType, DeviceType};
use crate::base::error::Result;
use crate::op::matmul::Matmul;
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::config::CudaConfig;

/// Sinusoidal timestep embedding → 2-layer MLP.
///
/// Produces the `adaln_input` vector that modulates DiT blocks.
pub struct TimestepEmbedder {
    /// First linear: [freq_dim → mid_size], with bias
    pub mlp1: Matmul,
    /// Second linear: [mid_size → out_size], with bias
    pub mlp2: Matmul,
    /// Frequency embedding dimension (256)
    pub frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    /// Sinusoidal positional encoding for timesteps.
    ///
    /// `t`: [B] f32 tensor on any device  
    /// Returns: [B, dim] f32 tensor (same device as t)
    pub fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
        let device = t.device();
        let b = t.shape()[0];
        let half = dim / 2;

        // freqs = exp(-ln(10000) * arange(0, half) / half)  → compute on CPU
        let log_max_period = (10000.0_f64).ln();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-log_max_period * i as f64 / half as f64).exp() as f32)
            .collect();

        // t values on CPU
        let t_cpu = if device != DeviceType::Cpu { t.to_cpu()? } else { t.clone() };
        let t_slice = t_cpu.as_f32()?.as_slice()?;

        // args[b, i] = t[b] * freqs[i]  → [B, half]
        // embedding = [cos(args), sin(args)]  → [B, dim]
        let mut emb = Tensor::new(&[b, dim], DataType::F32, DeviceType::Cpu)?;
        {
            let emb_slice = emb.as_f32_mut()?.as_slice_mut()?;
            for bi in 0..b {
                let tv = t_slice[bi];
                for i in 0..half {
                    let arg = tv * freqs[i];
                    emb_slice[bi * dim + i] = arg.cos();
                    emb_slice[bi * dim + half + i] = arg.sin();
                }
            }
        }

        // Move to original device if needed
        match device {
            DeviceType::Cpu => Ok(emb),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => emb.to_cuda(id),
        }
    }

    /// Forward: t → sinusoidal encoding → Linear → SiLU → Linear
    ///
    /// `t`: [B] timestep values (already scaled by t_scale before calling)
    /// Returns: [B, out_size] embedding
    pub fn forward(
        &self,
        t: &Tensor,
        #[cfg(feature = "cuda")] cuda_config: Option<&CudaConfig>,
    ) -> Result<Tensor> {
        // 1. Sinusoidal frequency encoding: [B] → [B, freq_dim]
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size)?;

        // 2. MLP: Linear1 → SiLU → Linear2
        let b = t.shape()[0];
        let mid_size = self.mlp1.weight.shape()[0];
        let out_size = self.mlp2.weight.shape()[0];

        let mut hidden = Tensor::new(&[b, mid_size], t_freq.dtype(), t_freq.device())?;
        self.mlp1.forward(&t_freq, &mut hidden, cuda_config)?;

        hidden.silu_();

        let mut output = Tensor::new(&[b, out_size], hidden.dtype(), hidden.device())?;
        self.mlp2.forward(&hidden, &mut output, cuda_config)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::error::Result;

    #[test]
    fn test_timestep_embedding_shape() -> Result<()> {
        let mut t = Tensor::new(&[2], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.5, 1.0]);

        let emb = TimestepEmbedder::timestep_embedding(&t, 256)?;
        assert_eq!(emb.shape(), &[2, 256]);
        assert_eq!(emb.dtype(), DataType::F32);
        Ok(())
    }

    #[test]
    fn test_timestep_embedding_values() -> Result<()> {
        let mut t = Tensor::new(&[1], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?[0] = 500.0; // typical scaled timestep

        let emb = TimestepEmbedder::timestep_embedding(&t, 256)?;
        let data = emb.as_f32()?.as_slice()?;

        // First 128 values are cos, next 128 are sin
        // cos(500 * 1.0) = cos(500) for freq[0]=1.0
        let expected_cos_0 = (500.0_f32).cos();
        assert!((data[0] - expected_cos_0).abs() < 1e-5,
            "cos[0] = {}, expected {}", data[0], expected_cos_0);

        // All values should be in [-1, 1]
        for &v in data {
            assert!(v.abs() <= 1.0 + 1e-6, "out of range: {v}");
        }
        Ok(())
    }

    #[test]
    fn test_timestep_embedder_forward() -> Result<()> {
        // Build a tiny TimestepEmbedder: freq_dim=8, mid=16, out=8
        let freq_dim = 8;
        let mid = 16;
        let out = 8;

        let mut mlp1 = Matmul::new(freq_dim, mid, true, DataType::F32, DeviceType::Cpu)?;
        // Initialize weights to small values (not zero/uninitialized)
        mlp1.weight.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate()
            .for_each(|(i, v)| *v = (i as f32 * 0.01) - 0.5);
        mlp1.bias.as_mut().unwrap().as_f32_mut()?.as_slice_mut()?.fill(0.0);

        let mut mlp2 = Matmul::new(mid, out, true, DataType::F32, DeviceType::Cpu)?;
        mlp2.weight.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate()
            .for_each(|(i, v)| *v = (i as f32 * 0.01) - 0.3);
        mlp2.bias.as_mut().unwrap().as_f32_mut()?.as_slice_mut()?.fill(0.0);

        let embedder = TimestepEmbedder {
            mlp1,
            mlp2,
            frequency_embedding_size: freq_dim,
        };

        let mut t = Tensor::new(&[2], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.3, 0.7]);

        let result = embedder.forward(&t, None)?;
        assert_eq!(result.shape(), &[2, out]);

        // Output should be finite
        let data = result.as_f32()?.as_slice()?;
        for &v in data {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
        Ok(())
    }

    #[test]
    fn test_silu_basic() -> Result<()> {
        let mut t = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.0, 1.0, -1.0, 2.0]);

        t.silu_();
        let data = t.as_f32()?.as_slice()?;

        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689, silu(2) ≈ 1.7616
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.7311).abs() < 1e-3);
        assert!((data[2] - (-0.2689)).abs() < 1e-3);
        assert!((data[3] - 1.7616).abs() < 1e-3);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_silu_cuda_vs_cpu_f32() -> Result<()> {
        let n = 1024;
        let mut cpu = Tensor::randn(&[n], DataType::F32, DeviceType::Cpu, Some(42))?;
        let mut gpu = cpu.to_cuda(0)?;

        cpu.silu_();
        gpu.silu_();

        let gpu_back = gpu.to_cpu()?;
        let cpu_data = cpu.as_f32()?.as_slice()?;
        let gpu_data = gpu_back.as_f32()?.as_slice()?;
        for i in 0..n {
            assert!(
                (cpu_data[i] - gpu_data[i]).abs() < 1e-5,
                "f32 mismatch at [{}]: cpu={}, gpu={}", i, cpu_data[i], gpu_data[i]
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_silu_cuda_vs_cpu_bf16() -> Result<()> {
        use half::bf16;
        let n = 1024;
        let mut cpu = Tensor::randn(&[n], DataType::BF16, DeviceType::Cpu, Some(7))?;
        let mut gpu = cpu.to_cuda(0)?;

        cpu.silu_();
        gpu.silu_();

        let gpu_back = gpu.to_cpu()?;
        let cpu_data = cpu.as_bf16()?.as_slice()?;
        let gpu_data = gpu_back.as_bf16()?.as_slice()?;
        for i in 0..n {
            let c = cpu_data[i].to_f32();
            let g = gpu_data[i].to_f32();
            assert!(
                (c - g).abs() < 0.02,
                "bf16 mismatch at [{}]: cpu={}, gpu={}", i, c, g
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_silu_cuda_vs_cpu_f16() -> Result<()> {
        use half::f16;
        let n = 1024;
        let mut cpu = Tensor::randn(&[n], DataType::F16, DeviceType::Cpu, Some(99))?;
        let mut gpu = cpu.to_cuda(0)?;

        cpu.silu_();
        gpu.silu_();

        let gpu_back = gpu.to_cpu()?;
        let cpu_data = cpu.as_f16()?.as_slice()?;
        let gpu_data = gpu_back.as_f16()?.as_slice()?;
        for i in 0..n {
            let c = cpu_data[i].to_f32();
            let g = gpu_data[i].to_f32();
            assert!(
                (c - g).abs() < 0.02,
                "f16 mismatch at [{}]: cpu={}, gpu={}", i, c, g
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_silu_cuda_odd_length() -> Result<()> {
        // 奇数长度测试 tail 处理
        let n = 1023;
        let mut cpu = Tensor::randn(&[n], DataType::F32, DeviceType::Cpu, Some(13))?;
        let mut gpu = cpu.to_cuda(0)?;

        cpu.silu_();
        gpu.silu_();

        let gpu_back = gpu.to_cpu()?;
        let cpu_data = cpu.as_f32()?.as_slice()?;
        let gpu_data = gpu_back.as_f32()?.as_slice()?;
        for i in 0..n {
            assert!(
                (cpu_data[i] - gpu_data[i]).abs() < 1e-5,
                "odd len mismatch at [{}]: cpu={}, gpu={}", i, cpu_data[i], gpu_data[i]
            );
        }
        Ok(())
    }
}
