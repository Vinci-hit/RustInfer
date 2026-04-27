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
    /// `target_dtype`: 输出的 dtype（sin/cos 内部用 f32 计算后转换）
    /// Returns: [B, dim] tensor (same device as t)
    pub fn timestep_embedding(t: &Tensor, dim: usize, target_dtype: DataType) -> Result<Tensor> {
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

        // 转到目标 dtype
        let emb = if target_dtype != DataType::F32 {
            emb.to_dtype(target_dtype)?
        } else { emb };

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
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Tensor> {
        let weight_dtype = self.mlp1.weight.dtype();

        // 1. Sinusoidal frequency encoding: [B] → [B, freq_dim]，直接输出权重 dtype
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size, weight_dtype)?;

        // 2. MLP: Linear1 → SiLU → Linear2
        let b = t.shape()[0];
        let mid_size = self.mlp1.weight.shape()[0];
        let out_size = self.mlp2.weight.shape()[0];

        let mut hidden = Tensor::new(&[b, mid_size], weight_dtype, t_freq.device())?;
        self.mlp1.forward(&t_freq, &mut hidden, cuda_config)?;

        hidden.silu_();

        let mut output = Tensor::new(&[b, out_size], weight_dtype, hidden.device())?;
        self.mlp2.forward(&hidden, &mut output, cuda_config)?;

        Ok(output)
    }

    /// Zero-allocation variant of [`Self::forward`] for the diffusion hot
    /// path.
    ///
    /// The caller pre-allocates three device-side slots matching the
    /// transformer's weight dtype:
    ///
    /// - `t_freq_slot`   : `[1, frequency_embedding_size]` — receives the
    ///                     sinusoidal encoding (dtype-cast from F32).
    /// - `t_hidden_slot` : `[1, mlp1.out_features]` — mlp1 output.
    /// - `t_out_slot`    : `[1, mlp2.out_features]` — mlp2 output (this is
    ///                     the returned `adaln_input` tensor).
    ///
    /// `t_value_scaled` is the already-scaled scalar timestep (host side).
    /// `host_staging` is a **persistent** `[1, frequency_embedding_size]`
    /// CPU tensor in the weight dtype that the caller owns for the
    /// lifetime of the pipeline — we compute the sinusoid into it and
    /// then do a single HtoD `copy_from_on_current_stream` to
    /// `t_freq_slot`. This avoids any `Tensor::new` on the hot path.
    ///
    /// Nothing here allocates; everything runs on the caller's current
    /// CUDA stream.
    pub fn forward_into(
        &self,
        t_value_scaled: f32,
        t_freq_slot: &mut Tensor,
        t_hidden_slot: &mut Tensor,
        t_out_slot: &mut Tensor,
        host_staging: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()> {
        let weight_dtype = self.mlp1.weight.dtype();
        let dim = self.frequency_embedding_size;
        let half = dim / 2;

        // Shape / dtype sanity — these are programmer errors, so panic
        // with a descriptive message instead of bubbling up.
        debug_assert_eq!(host_staging.shape(), &[1, dim]);
        debug_assert_eq!(host_staging.dtype(), weight_dtype);
        debug_assert_eq!(host_staging.device(), DeviceType::Cpu);
        debug_assert_eq!(t_freq_slot.shape(), &[1, dim]);
        debug_assert_eq!(t_freq_slot.dtype(), weight_dtype);

        // freqs = exp(-ln(10000) * arange(0, half) / half) — stack-local,
        // no heap alloc; `half = 128` for the standard Z-Image config.
        let log_max_period = (10000.0_f64).ln();

        // Write the sinusoid straight into the persistent host staging
        // buffer (dtype-casted inline). `[cos(args) | sin(args)]`.
        match weight_dtype {
            DataType::F32 => {
                let dst = host_staging.as_f32_mut()?.as_slice_mut()?;
                for i in 0..half {
                    let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
                    let arg = t_value_scaled * freq;
                    dst[i] = arg.cos();
                    dst[half + i] = arg.sin();
                }
            }
            DataType::BF16 => {
                let dst = host_staging.as_bf16_mut()?.as_slice_mut()?;
                for i in 0..half {
                    let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
                    let arg = t_value_scaled * freq;
                    dst[i]         = half::bf16::from_f32(arg.cos());
                    dst[half + i]  = half::bf16::from_f32(arg.sin());
                }
            }
            DataType::F16 => {
                let dst = host_staging.as_f16_mut()?.as_slice_mut()?;
                for i in 0..half {
                    let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
                    let arg = t_value_scaled * freq;
                    dst[i]         = half::f16::from_f32(arg.cos());
                    dst[half + i]  = half::f16::from_f32(arg.sin());
                }
            }
            other => {
                return Err(crate::base::error::Error::InvalidArgument(format!(
                    "TimestepEmbedder::forward_into: unsupported weight dtype {:?}",
                    other,
                )).into());
            }
        }

        // HtoD upload into the pre-allocated TFreq slot, stream-ordered.
        t_freq_slot.copy_from_on_current_stream(host_staging)?;

        // MLP: Linear1 → SiLU → Linear2, all writing into pre-allocated slots.
        self.mlp1.forward(t_freq_slot, t_hidden_slot, cuda_config)?;
        t_hidden_slot.silu()?;
        self.mlp2.forward(t_hidden_slot, t_out_slot, cuda_config)?;
        Ok(())
    }

    /// Zero-alloc, **graph-safe** variant: the scalar timestep is read
    /// from device memory (`d_t_scaled`, `[1] f32` holding `t_value *
    /// t_scale` written by the host before each graph replay). All
    /// compute happens on-device; nothing here uses a per-replay host
    /// scalar, so this is safe to include inside `cudaStreamBeginCapture`
    /// / `cudaStreamEndCapture`.
    #[cfg(feature = "cuda")]
    pub fn forward_from_dev(
        &self,
        d_t_scaled: &Tensor,
        t_freq_slot: &mut Tensor,
        t_hidden_slot: &mut Tensor,
        t_out_slot: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()> {
        // 1. Sinusoid into TFreq slot (directly in weight dtype).
        crate::op::scalar::sinusoid_embedding_from_dev(t_freq_slot, d_t_scaled)?;
        // 2. mlp1 → silu → mlp2, all dst-write into pre-allocated slots.
        self.mlp1.forward(t_freq_slot, t_hidden_slot, cuda_config)?;
        t_hidden_slot.silu()?;
        self.mlp2.forward(t_hidden_slot, t_out_slot, cuda_config)?;
        Ok(())
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

        let emb = TimestepEmbedder::timestep_embedding(&t, 256, DataType::F32)?;
        assert_eq!(emb.shape(), &[2, 256]);
        assert_eq!(emb.dtype(), DataType::F32);
        Ok(())
    }

    #[test]
    fn test_timestep_embedding_values() -> Result<()> {
        let mut t = Tensor::new(&[1], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?[0] = 500.0; // typical scaled timestep

        let emb = TimestepEmbedder::timestep_embedding(&t, 256, DataType::F32)?;
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
    fn test_silu_cuda_vs_cpu_bf16() -> Result<()> {
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
    fn test_silu_cuda_vs_cpu_f16() -> Result<()> {
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
