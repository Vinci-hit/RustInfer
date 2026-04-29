//! 3D Rotary Position Embedding for DiT (Z-Image).

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

pub struct RopeEmbedder3D {
    pub theta: f64,
    pub axes_dims: [usize; 3],
    pub axes_lens: [usize; 3],
    cos_cached: [Tensor; 3],
    sin_cached: [Tensor; 3],
}

impl RopeEmbedder3D {
    pub fn new(
        axes_dims: [usize; 3],
        axes_lens: [usize; 3],
        theta: f64,
    ) -> Result<Self> {
        let (cos_cached, sin_cached) = Self::precompute_freqs(&axes_dims, &axes_lens, theta)?;
        Ok(Self { theta, axes_dims, axes_lens, cos_cached, sin_cached })
    }

    fn precompute_freqs(
        axes_dims: &[usize; 3],
        axes_lens: &[usize; 3],
        theta: f64,
    ) -> Result<([Tensor; 3], [Tensor; 3])> {
        let mut cos_list = Vec::with_capacity(3);
        let mut sin_list = Vec::with_capacity(3);

        for axis in 0..3 {
            let d = axes_dims[axis];
            let e = axes_lens[axis];
            let half_d = d / 2;

            let mut cos_t = Tensor::new(&[e, half_d], DataType::F32, DeviceType::Cpu)?;
            let mut sin_t = Tensor::new(&[e, half_d], DataType::F32, DeviceType::Cpu)?;

            let freqs: Vec<f64> = (0..half_d)
                .map(|j| 1.0 / theta.powf(2.0 * j as f64 / d as f64))
                .collect();

            let cos_slice = cos_t.as_f32_mut()?.as_slice_mut()?;
            let sin_slice = sin_t.as_f32_mut()?.as_slice_mut()?;

            for pos in 0..e {
                let base = pos * half_d;
                for j in 0..half_d {
                    let val = pos as f64 * freqs[j];
                    cos_slice[base + j] = val.cos() as f32;
                    sin_slice[base + j] = val.sin() as f32;
                }
            }

            cos_list.push(cos_t);
            sin_list.push(sin_t);
        }

        Ok((
            [cos_list.remove(0), cos_list.remove(0), cos_list.remove(0)],
            [sin_list.remove(0), sin_list.remove(0), sin_list.remove(0)],
        ))
    }

    pub fn half_dim(&self) -> usize {
        self.axes_dims.iter().sum::<usize>() / 2
    }

    /// Zero-alloc embed: writes cos/sin into pre-allocated device slots via
    /// caller-owned CPU staging buffers.
    pub fn embed_into(
        &self,
        pos_ids_cpu: &Tensor,
        cos_dst: &mut Tensor,
        sin_dst: &mut Tensor,
        cos_host_stage: &mut Tensor,
        sin_host_stage: &mut Tensor,
    ) -> Result<()> {
        if pos_ids_cpu.shape().len() != 2 || pos_ids_cpu.shape()[1] != 3 {
            return Err(Error::InvalidArgument(format!(
                "embed_into: pos_ids must be [seq_len, 3], got {:?}", pos_ids_cpu.shape()
            )).into());
        }
        let seq_len = pos_ids_cpu.shape()[0];
        let half_dim = self.half_dim();

        let ids = pos_ids_cpu.as_i32()?.as_slice()?;
        let cos_out_slice = cos_host_stage.as_f32_mut()?.as_slice_mut()?;
        let sin_out_slice = sin_host_stage.as_f32_mut()?.as_slice_mut()?;

        let offsets = [
            0,
            self.axes_dims[0] / 2,
            self.axes_dims[0] / 2 + self.axes_dims[1] / 2,
        ];

        for axis in 0..3usize {
            let half_d = self.axes_dims[axis] / 2;
            let offset = offsets[axis];
            let cos_cache = self.cos_cached[axis].as_f32()?.as_slice()?;
            let sin_cache = self.sin_cached[axis].as_f32()?.as_slice()?;

            for token in 0..seq_len {
                let pos = ids[token * 3 + axis] as usize;
                let cache_base = pos * half_d;
                let out_base = token * half_dim + offset;

                cos_out_slice[out_base..out_base + half_d]
                    .copy_from_slice(&cos_cache[cache_base..cache_base + half_d]);
                sin_out_slice[out_base..out_base + half_d]
                    .copy_from_slice(&sin_cache[cache_base..cache_base + half_d]);
            }
        }

        cos_dst.copy_from_on_current_stream(cos_host_stage)?;
        sin_dst.copy_from_on_current_stream(sin_host_stage)?;
        Ok(())
    }
}

/// In-place interleaved RoPE (GPT-J style) — CPU fallback for the CUDA
/// kernel in `op/kernels/cuda/rope_interleaved`.
pub fn apply_rope_interleaved(
    x: &mut Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
) -> Result<()> {
    let x_shape = x.shape().to_vec();
    let seq_len = x_shape[0];
    let total_dim = x_shape.iter().skip(1).product::<usize>();
    let half_hd = head_dim / 2;

    let cos_data = cos.as_f32()?.as_slice()?;
    let sin_data = sin.as_f32()?.as_slice()?;

    match x {
        Tensor::F32(typed) => {
            let x_data = typed.as_slice_mut()?;
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
        }
        Tensor::BF16(typed) => {
            let x_data = typed.as_slice_mut()?;
            for s in 0..seq_len {
                let cos_row = &cos_data[s * half_hd..(s + 1) * half_hd];
                let sin_row = &sin_data[s * half_hd..(s + 1) * half_hd];
                let x_row = &mut x_data[s * total_dim..(s + 1) * total_dim];
                for chunk in x_row.chunks_exact_mut(head_dim) {
                    for k in 0..half_hd {
                        let x0 = chunk[2 * k].to_f32();
                        let x1 = chunk[2 * k + 1].to_f32();
                        chunk[2 * k] = half::bf16::from_f32(x0 * cos_row[k] - x1 * sin_row[k]);
                        chunk[2 * k + 1] = half::bf16::from_f32(x1 * cos_row[k] + x0 * sin_row[k]);
                    }
                }
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: unsupported dtype {:?}", x.dtype()
        )).into()),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::error::Result;

    #[test]
    fn test_precompute_cache_shapes() -> Result<()> {
        let rope = RopeEmbedder3D::new([32, 48, 48], [1536, 512, 512], 256.0)?;
        assert_eq!(rope.cos_cached[0].shape(), &[1536, 16]);
        assert_eq!(rope.cos_cached[1].shape(), &[512, 24]);
        assert_eq!(rope.half_dim(), 64);
        Ok(())
    }

    #[test]
    fn test_apply_rope_interleaved_math() -> Result<()> {
        let head_dim = 4;
        let mut x = Tensor::new(&[1, head_dim], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let mut cos = Tensor::new(&[1, 2], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.8, 0.6]);
        let mut sin = Tensor::new(&[1, 2], DataType::F32, DeviceType::Cpu)?;
        sin.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.3, 0.7]);

        apply_rope_interleaved(&mut x, &cos, &sin, head_dim)?;

        let result = x.as_f32()?.as_slice()?;
        let expected = [0.2, 1.9, -1.0, 4.5];
        for i in 0..4 {
            assert!((result[i] - expected[i]).abs() < 1e-5, "mismatch at {}", i);
        }
        Ok(())
    }
}
