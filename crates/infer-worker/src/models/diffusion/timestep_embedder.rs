//! Timestep embedder: scalar timestep → sinusoidal encoding → 2-layer MLP.
//!
//! diffusers `ZImageTransformer2DModel` uses
//! `frequency_embedding_size = 256` (= ADALN_EMBED_DIM), an MLP with
//! `mid = 1024`, output dim = 256, SiLU between layers.

use crate::domain::ports::{OpResult, OpError, OpBackend};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::models::layers::Linear;
use half::{bf16, f16};

pub struct TimestepEmbedder<T: Dtype, D: OpBackend> {
    pub mlp1: Linear<T, D>,
    pub mlp2: Linear<T, D>,
    pub frequency_embedding_size: usize,
}

impl<T: Dtype, D: OpBackend> TimestepEmbedder<T, D> {
    /// Forward path with the timestep value materialized on host.
    ///
    /// - `t_value_scaled`: pre-multiplied timestep (typically `(1 - t/T) * t_scale`).
    /// - `t_freq_slot`: `[1, frequency_embedding_size]` device tensor of dtype `T`.
    /// - `t_hidden_slot`: `[1, mid]` (MLP1 output).
    /// - `t_out_slot`: `[1, out_dim]` (MLP2 output).
    pub fn forward_host(
        &self,
        t_value_scaled: f32,
        t_freq_slot: &mut Tensor<T, D>,
        t_hidden_slot: &mut Tensor<T, D>,
        t_out_slot: &mut Tensor<T, D>,
    ) -> OpResult<()> {
        let dim = self.frequency_embedding_size;
        let half = dim / 2;
        let log_max_period = (10000.0_f64).ln();

        // Build sinusoid on host, then upload via MemoryPort.
        let bytes_per_elem = T::SIZE_BYTES;
        let mut host: Vec<u8> = vec![0u8; dim * bytes_per_elem];
        match T::DATA_TYPE {
            DataType::F32 => {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut f32, dim)
                };
                for i in 0..half {
                    let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
                    let arg = t_value_scaled * freq;
                    dst[i] = arg.cos();
                    dst[half + i] = arg.sin();
                }
            }
            DataType::BF16 => {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut bf16, dim)
                };
                for i in 0..half {
                    let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
                    let arg = t_value_scaled * freq;
                    dst[i] = bf16::from_f32(arg.cos());
                    dst[half + i] = bf16::from_f32(arg.sin());
                }
            }
            DataType::F16 => {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut f16, dim)
                };
                for i in 0..half {
                    let freq = (-log_max_period * i as f64 / half as f64).exp() as f32;
                    let arg = t_value_scaled * freq;
                    dst[i] = f16::from_f32(arg.cos());
                    dst[half + i] = f16::from_f32(arg.sin());
                }
            }
            other => return Err(OpError::Kernel(format!(
                "TimestepEmbedder: unsupported dtype {:?}", other,
            ))),
        }

        // Upload to device slot.
        let dev = t_freq_slot.device().clone();
        let dst_bytes = dim * bytes_per_elem;
        if t_freq_slot.numel() != dim {
            return Err(OpError::Shape(format!(
                "TimestepEmbedder: t_freq_slot numel {} != dim {}", t_freq_slot.numel(), dim,
            )));
        }
        unsafe {
            let dn = std::ptr::NonNull::new_unchecked(t_freq_slot.data_ptr_mut() as *mut u8);
            dev.upload(dn, host.as_ptr(), dst_bytes)?;
        }

        // mlp1 → silu → mlp2.
        self.mlp1.forward(t_freq_slot, t_hidden_slot)?;
        D::silu_inplace_diff(t_hidden_slot)?;
        self.mlp2.forward(t_hidden_slot, t_out_slot)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::tensor::Tensor;
    use crate::domain::types::Shape;
    use crate::infrastructure::cuda::Cuda;
    use crate::models::layers::Linear;

    /// Build a tiny embedder with identity-like weights so we can verify the
    /// sinusoid → MLP path numerically end-to-end. mlp1 is `[mid, freq_dim]`
    /// init to `mid_inv * I` (rectangular identity-ish); mlp2 is
    /// `[out, mid]` likewise. SiLU is applied between.
    fn make_embedder(freq_dim: usize, mid: usize, out: usize, cuda: &Cuda) -> TimestepEmbedder<f32, Cuda> {
        // mlp1 weight: shape [mid, freq_dim], values w[i,j] = (i==j ? 1 : 0)
        let mut w1 = vec![0.0_f32; mid * freq_dim];
        for i in 0..mid.min(freq_dim) { w1[i * freq_dim + i] = 1.0; }
        let mut w2 = vec![0.0_f32; out * mid];
        for i in 0..out.min(mid) { w2[i * mid + i] = 1.0; }
        let w1_t: Tensor<f32, Cuda> = Tensor::from_host_slice(&w1, Shape::from_slice(&[mid, freq_dim]), cuda).unwrap();
        let w2_t: Tensor<f32, Cuda> = Tensor::from_host_slice(&w2, Shape::from_slice(&[out, mid]), cuda).unwrap();
        TimestepEmbedder {
            mlp1: Linear::new(w1_t, None),
            mlp2: Linear::new(w2_t, None),
            frequency_embedding_size: freq_dim,
        }
    }

    #[test]
    fn timestep_embedder_t_zero_produces_unit_cos_zero_sin_then_silu() {
        let cuda = Cuda::new(0).unwrap();
        let freq_dim = 8;
        let mid = freq_dim;
        let out = freq_dim;
        let emb = make_embedder(freq_dim, mid, out, &cuda);
        let mut t_freq: Tensor<f32, Cuda> = Tensor::zeros([1, freq_dim], &cuda).unwrap();
        let mut t_hidden: Tensor<f32, Cuda> = Tensor::zeros([1, mid], &cuda).unwrap();
        let mut t_out: Tensor<f32, Cuda> = Tensor::zeros([1, out], &cuda).unwrap();
        // t=0: cos(0*freq)=1, sin(0*freq)=0 → first half = 1, second half = 0.
        emb.forward_host(0.0, &mut t_freq, &mut t_hidden, &mut t_out).unwrap();
        let freq_v = t_freq.to_host_vec().unwrap();
        for v in &freq_v[..freq_dim / 2] { assert!((v - 1.0).abs() < 1e-5); }
        for v in &freq_v[freq_dim / 2..] { assert!(v.abs() < 1e-5); }

        // After mlp1 (identity) + silu + mlp2 (identity):
        // hidden_pre = t_freq, hidden_post = silu(hidden_pre)
        // out = silu(hidden_pre) → first half = silu(1) = 0.7311, second half = silu(0) = 0.
        let out_v = t_out.to_host_vec().unwrap();
        let silu1 = 1.0 / (1.0 + (-1.0_f32).exp()); // sigmoid(1) = silu(1) since x=1
        for v in &out_v[..freq_dim / 2] {
            assert!((v - silu1).abs() < 1e-5, "got {}, expected {}", v, silu1);
        }
        for v in &out_v[freq_dim / 2..] { assert!(v.abs() < 1e-5); }
    }

    #[test]
    fn timestep_embedder_freq_scales_with_t() {
        let cuda = Cuda::new(0).unwrap();
        let freq_dim = 16;
        let emb = make_embedder(freq_dim, freq_dim, freq_dim, &cuda);
        let mut t_freq: Tensor<f32, Cuda> = Tensor::zeros([1, freq_dim], &cuda).unwrap();
        let mut t_hidden: Tensor<f32, Cuda> = Tensor::zeros([1, freq_dim], &cuda).unwrap();
        let mut t_out: Tensor<f32, Cuda> = Tensor::zeros([1, freq_dim], &cuda).unwrap();
        emb.forward_host(1.0, &mut t_freq, &mut t_hidden, &mut t_out).unwrap();
        let v = t_freq.to_host_vec().unwrap();
        let half = freq_dim / 2;
        let log_max = (10000.0_f64).ln();
        for i in 0..half {
            let freq = (-log_max * i as f64 / half as f64).exp() as f32;
            let arg = 1.0_f32 * freq;
            assert!((v[i] - arg.cos()).abs() < 1e-5);
            assert!((v[half + i] - arg.sin()).abs() < 1e-5);
        }
    }
}
