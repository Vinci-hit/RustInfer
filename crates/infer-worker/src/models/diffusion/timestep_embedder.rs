//! Timestep embedding: scalar t → sinusoidal encoding → 2-layer MLP.
//!
//! The sinusoidal encoding uses `log(10000)` as max period, matching
//! the standard diffusion timestep embedding from the original DDPM paper.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::Dtype;
use crate::domain::tensor::Tensor;
use crate::models::layers::Linear;

/// Timestep embedder: scalar → [1, freq_dim] sinusoid → MLP → [1, out_dim].
pub struct TimestepEmbedder<T: Dtype, D: OpBackend> {
    pub mlp1: Linear<T, D>,  // [mid_dim, freq_dim]
    pub mlp2: Linear<T, D>,  // [out_dim, mid_dim]
    pub frequency_embedding_size: usize,
}

impl<T: Dtype, D: OpBackend> TimestepEmbedder<T, D> {
    /// Compute sinusoidal frequency encoding on CPU, write into `t_freq_slot`.
    ///
    /// Formula: freq[i] = exp(-ln(10000) * i / (dim/2))
    ///          out[i] = cos(t * freq[i]),  out[dim/2 + i] = sin(t * freq[i])
    pub fn compute_sinusoid_cpu(t_value: f32, dim: usize) -> Vec<f32> {
        let half = dim / 2;
        let log_max_period = 10000.0_f64.ln();
        let mut out = vec![0.0f32; dim];
        for i in 0..half {
            let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
            let arg = t_value * freq;
            out[i] = arg.cos();
            out[half + i] = arg.sin();
        }
        out
    }

    /// Forward: t_value → sinusoid → mlp1 → silu → mlp2 → output.
    ///
    /// All intermediate tensors allocated from device via OpBackend.
    pub fn forward(
        &self,
        t_value: f32,
        t_freq: &mut Tensor<T, D>,
        t_hidden: &mut Tensor<T, D>,
        t_out: &mut Tensor<T, D>,
    ) -> OpResult<()> {
        // Step 1: compute sinusoid on CPU and write to t_freq
        // (In production CUDA path, this uses a kernel reading from device scalar)
        let sinusoid = Self::compute_sinusoid_cpu(t_value, self.frequency_embedding_size);
        // Write sinusoid into t_freq — for now copy via raw bytes
        // (Production: H2D async copy or device-side sinusoid kernel)
        let freq_ptr = t_freq.data_ptr_mut() as *mut u8;
        let elem = T::SIZE_BYTES;
        for (i, &val) in sinusoid.iter().enumerate() {
            // Cast f32 → T and write
            let bytes = match T::DATA_TYPE {
                crate::domain::types::DataType::F32 => val.to_le_bytes().to_vec(),
                crate::domain::types::DataType::BF16 => half::bf16::from_f32(val).to_le_bytes().to_vec(),
                crate::domain::types::DataType::F16 => half::f16::from_f32(val).to_le_bytes().to_vec(),
                _ => vec![0u8; elem],
            };
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), freq_ptr.add(i * elem), elem); }
        }

        // Step 2: mlp1
        self.mlp1.forward(t_freq, t_hidden)?;
        // Step 3: silu
        D::silu_inplace(t_hidden)?;
        // Step 4: mlp2
        self.mlp2.forward(t_hidden, t_out)?;
        Ok(())
    }
}
