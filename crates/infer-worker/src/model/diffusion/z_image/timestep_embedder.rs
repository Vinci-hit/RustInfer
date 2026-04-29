//! Timestep embedding: scalar t → sinusoidal encoding → 2-layer MLP.

use crate::base::{DataType, DeviceType};
use crate::base::error::Result;
use crate::op::matmul::Matmul;
use crate::tensor::Tensor;

pub struct TimestepEmbedder {
    pub mlp1: Matmul,
    pub mlp2: Matmul,
    pub frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    /// Zero-alloc forward: sinusoid assembled into `host_staging`, uploaded
    /// to `t_freq_slot`, then mlp1 → silu → mlp2 into pre-allocated slots.
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

        debug_assert_eq!(host_staging.shape(), &[1, dim]);
        debug_assert_eq!(host_staging.dtype(), weight_dtype);
        debug_assert_eq!(host_staging.device(), DeviceType::Cpu);

        let log_max_period = (10000.0_f64).ln();

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
                    "TimestepEmbedder::forward_into: unsupported weight dtype {:?}", other,
                )).into());
            }
        }

        t_freq_slot.copy_from_on_current_stream(host_staging)?;
        self.mlp1.forward(t_freq_slot, t_hidden_slot, cuda_config)?;
        t_hidden_slot.silu()?;
        self.mlp2.forward(t_hidden_slot, t_out_slot, cuda_config)?;
        Ok(())
    }

    /// Graph-safe variant: timestep read from device memory.
    #[cfg(feature = "cuda")]
    pub fn forward_from_dev(
        &self,
        d_t_scaled: &Tensor,
        t_freq_slot: &mut Tensor,
        t_hidden_slot: &mut Tensor,
        t_out_slot: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()> {
        crate::op::scalar::sinusoid_embedding_from_dev(t_freq_slot, d_t_scaled)?;
        self.mlp1.forward(t_freq_slot, t_hidden_slot, cuda_config)?;
        t_hidden_slot.silu()?;
        self.mlp2.forward(t_hidden_slot, t_out_slot, cuda_config)?;
        Ok(())
    }
}
