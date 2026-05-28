//! 3D Rotary Position Embedding for DiT (Z-Image).
//!
//! Precomputes per-axis sin/cos caches and embeds 3D positions (t, h, w).
//! The interleaved RoPE formula: x'[2k] = x[2k]*cos - x[2k+1]*sin,
//!                                x'[2k+1] = x[2k+1]*cos + x[2k]*sin

use crate::domain::ports::OpResult;
use crate::domain::types::Shape;
use crate::domain::tensor::Tensor;
use crate::infrastructure::cpu::Cpu;

/// 3D RoPE embedder — precomputes sin/cos for 3 spatial axes.
pub struct RopeEmbedder3D {
    pub theta: f64,
    pub axes_dims: [usize; 3],   // e.g. [32, 48, 48] → half_dim per axis
    pub axes_lens: [usize; 3],   // e.g. [1536, 512, 512] → max position per axis
    /// Precomputed cos cache per axis: [axes_lens[i], axes_dims[i]/2] on CPU (f32)
    cos_cached: [Tensor<f32, Cpu>; 3],
    /// Precomputed sin cache per axis: [axes_lens[i], axes_dims[i]/2] on CPU (f32)
    sin_cached: [Tensor<f32, Cpu>; 3],
}

impl RopeEmbedder3D {
    pub fn new(axes_dims: [usize; 3], axes_lens: [usize; 3], theta: f64) -> Self {
        let cos_cached = [
            Self::precompute_axis(axes_dims[0], axes_lens[0], theta),
            Self::precompute_axis(axes_dims[1], axes_lens[1], theta),
            Self::precompute_axis(axes_dims[2], axes_lens[2], theta),
        ];
        let sin_cached = [
            Self::precompute_axis_sin(axes_dims[0], axes_lens[0], theta),
            Self::precompute_axis_sin(axes_dims[1], axes_lens[1], theta),
            Self::precompute_axis_sin(axes_dims[2], axes_lens[2], theta),
        ];
        Self { theta, axes_dims, axes_lens, cos_cached, sin_cached }
    }

    /// Total half-dim across all 3 axes.
    pub fn half_dim(&self) -> usize {
        self.axes_dims.iter().sum::<usize>() / 2
    }

    fn precompute_axis(dim: usize, max_len: usize, theta: f64) -> Tensor<f32, Cpu> {
        let half_d = dim / 2;
        let mut data = vec![0.0f32; max_len * half_d];
        let freqs: Vec<f64> = (0..half_d)
            .map(|j| 1.0 / theta.powf(2.0 * j as f64 / dim as f64))
            .collect();
        for pos in 0..max_len {
            for j in 0..half_d {
                data[pos * half_d + j] = (pos as f64 * freqs[j]).cos() as f32;
            }
        }
        Tensor::<f32, Cpu>::from_slice(&data, Shape::from_slice(&[max_len, half_d]))
    }

    fn precompute_axis_sin(dim: usize, max_len: usize, theta: f64) -> Tensor<f32, Cpu> {
        let half_d = dim / 2;
        let mut data = vec![0.0f32; max_len * half_d];
        let freqs: Vec<f64> = (0..half_d)
            .map(|j| 1.0 / theta.powf(2.0 * j as f64 / dim as f64))
            .collect();
        for pos in 0..max_len {
            for j in 0..half_d {
                data[pos * half_d + j] = (pos as f64 * freqs[j]).sin() as f32;
            }
        }
        Tensor::<f32, Cpu>::from_slice(&data, Shape::from_slice(&[max_len, half_d]))
    }

    /// Embed 3D positions into sin/cos buffers (CPU staging → device upload).
    ///
    /// - `pos_ids`: [seq_len, 3] i32 on CPU — (t, h, w) per token
    /// - `cos_dst` / `sin_dst`: [seq_len, half_dim] on target device — output
    ///
    /// Gathers from precomputed caches and concatenates across axes.
    pub fn embed_into_cpu(
        &self,
        pos_ids: &[i32],   // flat [seq_len * 3]
        seq_len: usize,
        cos_out: &mut [f32],  // [seq_len * half_dim]
        sin_out: &mut [f32],  // [seq_len * half_dim]
    ) {
        let half_dim = self.half_dim();
        let offsets = [
            0usize,
            self.axes_dims[0] / 2,
            self.axes_dims[0] / 2 + self.axes_dims[1] / 2,
        ];

        for axis in 0..3usize {
            let half_d = self.axes_dims[axis] / 2;
            let offset = offsets[axis];
            let cos_cache = self.cos_cached[axis].as_slice();
            let sin_cache = self.sin_cached[axis].as_slice();

            for token in 0..seq_len {
                let pos = pos_ids[token * 3 + axis] as usize;
                let cache_base = pos * half_d;
                let out_base = token * half_dim + offset;

                cos_out[out_base..out_base + half_d]
                    .copy_from_slice(&cos_cache[cache_base..cache_base + half_d]);
                sin_out[out_base..out_base + half_d]
                    .copy_from_slice(&sin_cache[cache_base..cache_base + half_d]);
            }
        }
    }
}

/// Apply interleaved RoPE in-place on CPU tensor.
/// x: [seq_len, total_dim], cos/sin: [seq_len, half_dim_per_head * num_heads]
/// head_dim: dimension per attention head (RoPE applied per-head).
pub fn apply_rope_interleaved_cpu(
    x: &mut Tensor<f32, Cpu>,
    cos: &Tensor<f32, Cpu>,
    sin: &Tensor<f32, Cpu>,
    head_dim: usize,
) -> OpResult<()> {
    let shape = x.shape().as_slice();
    let seq_len = shape[0];
    let total_dim: usize = shape[1..].iter().product();
    let half_hd = head_dim / 2;

    let cos_data = cos.as_slice();
    let sin_data = sin.as_slice();
    let x_data = unsafe { std::slice::from_raw_parts_mut(x.data_ptr_mut(), x.numel()) };

    for s in 0..seq_len {
        let cos_row = &cos_data[s * half_hd..(s + 1) * half_hd];
        let sin_row = &sin_data[s * half_hd..(s + 1) * half_hd];
        let x_row = &mut x_data[s * total_dim..(s + 1) * total_dim];
        for chunk in x_row.chunks_exact_mut(head_dim) {
            for k in 0..half_hd {
                let x0 = chunk[2 * k];
                let x1 = chunk[2 * k + 1];
                chunk[2 * k] = x0 * cos_row[k] - x1 * sin_row[k];
                chunk[2 * k + 1] = x1 * cos_row[k] + x0 * sin_row[k];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precompute_cache_shapes() {
        let rope = RopeEmbedder3D::new([32, 48, 48], [1536, 512, 512], 256.0);
        assert_eq!(rope.cos_cached[0].shape().as_slice(), &[1536, 16]);
        assert_eq!(rope.cos_cached[1].shape().as_slice(), &[512, 24]);
        assert_eq!(rope.cos_cached[2].shape().as_slice(), &[512, 24]);
        assert_eq!(rope.half_dim(), 64); // (32+48+48)/2
    }

    #[test]
    fn apply_rope_interleaved_math() {
        let head_dim = 4;
        let mut x = Tensor::<f32, Cpu>::from_slice(
            &[1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[1, head_dim]),
        );
        let cos = Tensor::<f32, Cpu>::from_slice(&[0.8, 0.6], Shape::from_slice(&[1, 2]));
        let sin = Tensor::<f32, Cpu>::from_slice(&[0.3, 0.7], Shape::from_slice(&[1, 2]));

        apply_rope_interleaved_cpu(&mut x, &cos, &sin, head_dim).unwrap();

        let result = x.as_slice();
        // x'[0] = 1.0*0.8 - 2.0*0.3 = 0.8 - 0.6 = 0.2
        // x'[1] = 2.0*0.8 + 1.0*0.3 = 1.6 + 0.3 = 1.9
        // x'[2] = 3.0*0.6 - 4.0*0.7 = 1.8 - 2.8 = -1.0
        // x'[3] = 4.0*0.6 + 3.0*0.7 = 2.4 + 2.1 = 4.5
        assert!((result[0] - 0.2).abs() < 1e-5);
        assert!((result[1] - 1.9).abs() < 1e-5);
        assert!((result[2] - (-1.0)).abs() < 1e-5);
        assert!((result[3] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn embed_into_gathers_correctly() {
        let rope = RopeEmbedder3D::new([4, 4, 4], [10, 10, 10], 10000.0);
        let half_dim = rope.half_dim(); // (4+4+4)/2 = 6
        let seq_len = 2;

        // pos_ids: token 0 at (0,0,0), token 1 at (1,2,3)
        let pos_ids = [0i32, 0, 0, 1, 2, 3];
        let mut cos_out = vec![0.0f32; seq_len * half_dim];
        let mut sin_out = vec![0.0f32; seq_len * half_dim];

        rope.embed_into_cpu(&pos_ids, seq_len, &mut cos_out, &mut sin_out);

        // Token 0 at position (0,0,0): cos(0*freq) = 1.0 for all freqs
        for i in 0..half_dim {
            assert!((cos_out[i] - 1.0).abs() < 1e-5, "cos[0][{}] = {}", i, cos_out[i]);
            assert!(sin_out[i].abs() < 1e-5, "sin[0][{}] = {}", i, sin_out[i]);
        }
    }
}
