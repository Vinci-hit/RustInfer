//! 3D Rotary Position Embedding for DiT (Z-Image).
//!
//! Three precomputed `[axes_lens[i], axes_dims[i]/2]` cos/sin caches on the
//! host. `embed_into` gathers per-axis slices into a unified
//! `[seq_len, half_dim_total]` cos/sin tensor on the target CUDA device.

use crate::domain::ports::{OpResult, OpError, MemoryPort};
use crate::domain::tensor::Tensor;
use crate::infrastructure::cpu::Cpu;
use crate::infrastructure::cuda::Cuda;

pub struct RopeEmbedder3D {
    pub theta: f64,
    pub axes_dims: [usize; 3],
    pub axes_lens: [usize; 3],
    /// `[axes_lens[i], axes_dims[i]/2]` cos cache for axis `i`, on CPU.
    cos_cached: [Vec<f32>; 3],
    /// Same shape as `cos_cached`, sin variant.
    sin_cached: [Vec<f32>; 3],
}

impl RopeEmbedder3D {
    pub fn new(axes_dims: [usize; 3], axes_lens: [usize; 3], theta: f64) -> OpResult<Self> {
        let cos_cached = [
            Self::precompute_axis(axes_dims[0], axes_lens[0], theta, false),
            Self::precompute_axis(axes_dims[1], axes_lens[1], theta, false),
            Self::precompute_axis(axes_dims[2], axes_lens[2], theta, false),
        ];
        let sin_cached = [
            Self::precompute_axis(axes_dims[0], axes_lens[0], theta, true),
            Self::precompute_axis(axes_dims[1], axes_lens[1], theta, true),
            Self::precompute_axis(axes_dims[2], axes_lens[2], theta, true),
        ];
        Ok(Self { theta, axes_dims, axes_lens, cos_cached, sin_cached })
    }

    fn precompute_axis(d: usize, max_len: usize, theta: f64, is_sin: bool) -> Vec<f32> {
        let half_d = d / 2;
        let mut data = vec![0.0_f32; max_len * half_d];
        let freqs: Vec<f64> = (0..half_d)
            .map(|j| 1.0 / theta.powf(2.0 * j as f64 / d as f64))
            .collect();
        for pos in 0..max_len {
            let base = pos * half_d;
            for j in 0..half_d {
                let val = pos as f64 * freqs[j];
                data[base + j] = if is_sin { val.sin() as f32 } else { val.cos() as f32 };
            }
        }
        data
    }

    /// Total `head_dim/2`: sum of per-axis half dims.
    pub fn half_dim(&self) -> usize {
        self.axes_dims.iter().sum::<usize>() / 2
    }

    /// Embed a `[seq_len, 3]` i32 position-id table (host) into pre-allocated
    /// device tensors `cos_dst`, `sin_dst` of shape `[seq_len, half_dim]`.
    /// Internally stages on host then uploads via `MemoryPort::upload`.
    pub fn embed_into_cuda(
        &self,
        pos_ids: &[i32],
        seq_len: usize,
        cos_dst: &mut Tensor<f32, Cuda>,
        sin_dst: &mut Tensor<f32, Cuda>,
    ) -> OpResult<()> {
        if pos_ids.len() != seq_len * 3 {
            return Err(OpError::Shape(format!(
                "embed_into_cuda: pos_ids length {} != seq_len*3 = {}",
                pos_ids.len(), seq_len * 3,
            )));
        }
        let half_dim = self.half_dim();
        if cos_dst.shape().as_slice() != [seq_len, half_dim]
            || sin_dst.shape().as_slice() != [seq_len, half_dim]
        {
            return Err(OpError::Shape(format!(
                "embed_into_cuda: dst shape mismatch (expected [{}, {}], got cos={:?}, sin={:?})",
                seq_len, half_dim, cos_dst.shape(), sin_dst.shape(),
            )));
        }

        let mut cos_host = vec![0.0_f32; seq_len * half_dim];
        let mut sin_host = vec![0.0_f32; seq_len * half_dim];

        let offsets = [
            0_usize,
            self.axes_dims[0] / 2,
            self.axes_dims[0] / 2 + self.axes_dims[1] / 2,
        ];
        for axis in 0..3usize {
            let half_d = self.axes_dims[axis] / 2;
            let offset = offsets[axis];
            let cos_cache = &self.cos_cached[axis];
            let sin_cache = &self.sin_cached[axis];
            for token in 0..seq_len {
                let pos = pos_ids[token * 3 + axis] as usize;
                let cache_base = pos * half_d;
                let out_base = token * half_dim + offset;
                cos_host[out_base..out_base + half_d]
                    .copy_from_slice(&cos_cache[cache_base..cache_base + half_d]);
                sin_host[out_base..out_base + half_d]
                    .copy_from_slice(&sin_cache[cache_base..cache_base + half_d]);
            }
        }

        // Upload.
        let dev = cos_dst.device().clone();
        let bytes = cos_host.len() * 4;
        unsafe {
            let cos_nn = std::ptr::NonNull::new_unchecked(cos_dst.data_ptr_mut() as *mut u8);
            dev.upload(cos_nn, cos_host.as_ptr() as *const u8, bytes)?;
        }
        unsafe {
            let sin_nn = std::ptr::NonNull::new_unchecked(sin_dst.data_ptr_mut() as *mut u8);
            dev.upload(sin_nn, sin_host.as_ptr() as *const u8, bytes)?;
        }
        Ok(())
    }

    /// CPU variant: write cos/sin into pre-allocated host f32 tensors.
    /// Useful for unit tests and the reference path.
    pub fn embed_into_cpu(
        &self,
        pos_ids: &[i32],
        seq_len: usize,
        cos_dst: &mut Tensor<f32, Cpu>,
        sin_dst: &mut Tensor<f32, Cpu>,
    ) -> OpResult<()> {
        let half_dim = self.half_dim();
        if cos_dst.shape().as_slice() != [seq_len, half_dim]
            || sin_dst.shape().as_slice() != [seq_len, half_dim]
        {
            return Err(OpError::Shape(format!(
                "embed_into_cpu: dst shape mismatch (expected [{}, {}], got cos={:?}, sin={:?})",
                seq_len, half_dim, cos_dst.shape(), sin_dst.shape(),
            )));
        }
        if pos_ids.len() != seq_len * 3 {
            return Err(OpError::Shape("embed_into_cpu: pos_ids length mismatch".into()));
        }
        let mut cos_host = vec![0.0_f32; seq_len * half_dim];
        let mut sin_host = vec![0.0_f32; seq_len * half_dim];
        let offsets = [
            0_usize,
            self.axes_dims[0] / 2,
            self.axes_dims[0] / 2 + self.axes_dims[1] / 2,
        ];
        for axis in 0..3usize {
            let half_d = self.axes_dims[axis] / 2;
            let offset = offsets[axis];
            for token in 0..seq_len {
                let pos = pos_ids[token * 3 + axis] as usize;
                let cache_base = pos * half_d;
                let out_base = token * half_dim + offset;
                cos_host[out_base..out_base + half_d]
                    .copy_from_slice(&self.cos_cached[axis][cache_base..cache_base + half_d]);
                sin_host[out_base..out_base + half_d]
                    .copy_from_slice(&self.sin_cached[axis][cache_base..cache_base + half_d]);
            }
        }
        // Write via from_host_slice → re-construct dst storage. The new tensor
        // shares with caller's dst by overwriting the underlying storage bytes.
        // Since Cpu storage is a Vec<u8>, we can poke into it.
        let cpu_dev = cos_dst.device().clone();
        unsafe {
            let cn = std::ptr::NonNull::new_unchecked(cos_dst.data_ptr_mut() as *mut u8);
            cpu_dev.upload(cn, cos_host.as_ptr() as *const u8, cos_host.len() * 4)?;
            let sn = std::ptr::NonNull::new_unchecked(sin_dst.data_ptr_mut() as *mut u8);
            cpu_dev.upload(sn, sin_host.as_ptr() as *const u8, sin_host.len() * 4)?;
        }
        Ok(())
    }
}

/// Build position ids for caption tokens used in Z-Image. Caption tokens
/// occupy axis 0 starting at 1 (axis-0 index 0 is reserved for prefix
/// padding), with axes 1 and 2 fixed to 0.
pub fn fill_cap_pos_ids(seq_len: usize) -> Vec<i32> {
    let mut out = vec![0_i32; seq_len * 3];
    for s in 0..seq_len {
        out[s * 3] = (s + 1) as i32;
    }
    out
}

/// Build position ids for image patch tokens. Image tokens use axis 0 as a
/// global "stream offset" (post-caption), axes 1 and 2 as (h, w) within the
/// patchified latent. Padding tokens share the position of the last real
/// patch.
pub fn fill_image_pos_ids(
    f_t: usize, h_t: usize, w_t: usize,
    axis0_offset: usize,
    pad_count: usize,
) -> Vec<i32> {
    let n = f_t * h_t * w_t;
    let total = n + pad_count;
    let mut out = vec![0_i32; total * 3];
    let mut idx = 0;
    for ft in 0..f_t {
        for ht in 0..h_t {
            for wt in 0..w_t {
                out[idx * 3 + 0] = (axis0_offset + ft) as i32;
                out[idx * 3 + 1] = ht as i32;
                out[idx * 3 + 2] = wt as i32;
                idx += 1;
            }
        }
    }
    // Pad rows: copy the last real row.
    if pad_count > 0 && n > 0 {
        let last = (idx - 1) * 3;
        for i in 0..pad_count {
            let dst = (n + i) * 3;
            out[dst + 0] = out[last + 0];
            out[dst + 1] = out[last + 1];
            out[dst + 2] = out[last + 2];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::tensor::Tensor;

    #[test]
    fn precompute_cache_shapes() {
        let rope = RopeEmbedder3D::new([32, 48, 48], [1536, 512, 512], 256.0).unwrap();
        // cos_cached[0]: 1536 * 16 = 24576 f32.
        assert_eq!(rope.cos_cached[0].len(), 1536 * 16);
        assert_eq!(rope.cos_cached[1].len(), 512 * 24);
        assert_eq!(rope.cos_cached[2].len(), 512 * 24);
        assert_eq!(rope.half_dim(), 64); // (32+48+48)/2
    }

    #[test]
    fn embed_zero_position_is_unit_cos() {
        // Position (0,0,0): cos=1, sin=0 across all bins.
        let rope = RopeEmbedder3D::new([4, 4, 4], [10, 10, 10], 10000.0).unwrap();
        let half_dim = rope.half_dim(); // (4+4+4)/2 = 6
        let cpu = Cpu;
        let mut cos: Tensor<f32, Cpu> = Tensor::zeros([1, half_dim], &cpu).unwrap();
        let mut sin: Tensor<f32, Cpu> = Tensor::zeros([1, half_dim], &cpu).unwrap();
        let pos_ids = vec![0_i32, 0, 0];
        rope.embed_into_cpu(&pos_ids, 1, &mut cos, &mut sin).unwrap();
        let cos_v = cos.to_host_vec().unwrap();
        let sin_v = sin.to_host_vec().unwrap();
        for c in cos_v { assert!((c - 1.0).abs() < 1e-5); }
        for s in sin_v { assert!(s.abs() < 1e-5); }
    }

    #[test]
    fn embed_specific_position_matches_freq_formula() {
        let rope = RopeEmbedder3D::new([4, 4, 4], [10, 10, 10], 10000.0).unwrap();
        let half_dim = rope.half_dim();
        let cpu = Cpu;
        let mut cos: Tensor<f32, Cpu> = Tensor::zeros([1, half_dim], &cpu).unwrap();
        let mut sin: Tensor<f32, Cpu> = Tensor::zeros([1, half_dim], &cpu).unwrap();
        // pos = (3, 0, 0) → axes 0..2 (the first 2 bins out of 6) use pos=3
        let pos_ids = vec![3_i32, 0, 0];
        rope.embed_into_cpu(&pos_ids, 1, &mut cos, &mut sin).unwrap();
        let cos_v = cos.to_host_vec().unwrap();
        let sin_v = sin.to_host_vec().unwrap();
        // Bin 0 of axis 0: freq = 1 / 10000^(2*0/4) = 1.0; cos(3) / sin(3).
        assert!((cos_v[0] - (3.0_f32).cos()).abs() < 1e-5);
        assert!((sin_v[0] - (3.0_f32).sin()).abs() < 1e-5);
        // Bin 1 of axis 0: freq = 1/10000^(2/4) = 1/100; cos(3*0.01) / sin(3*0.01).
        let expected = 0.03_f32;
        assert!((cos_v[1] - expected.cos()).abs() < 1e-5);
        assert!((sin_v[1] - expected.sin()).abs() < 1e-5);
        // Axis 1 / 2 with pos=0 → cos=1, sin=0.
        for i in 2..half_dim {
            assert!((cos_v[i] - 1.0).abs() < 1e-5);
            assert!(sin_v[i].abs() < 1e-5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn embed_into_cuda_matches_cpu() {
        let cpu = Cpu;
        let cuda = Cuda::new(0).unwrap();
        let rope = RopeEmbedder3D::new([32, 48, 48], [1536, 512, 512], 256.0).unwrap();
        let half_dim = rope.half_dim();
        let seq_len = 17;

        // Random pos_ids within bounds.
        let mut pos_ids = Vec::with_capacity(seq_len * 3);
        for s in 0..seq_len {
            pos_ids.push(((s + 5) % 1536) as i32);
            pos_ids.push((s % 512) as i32);
            pos_ids.push(((s * 3) % 512) as i32);
        }

        let mut cos_cpu: Tensor<f32, Cpu> = Tensor::zeros([seq_len, half_dim], &cpu).unwrap();
        let mut sin_cpu: Tensor<f32, Cpu> = Tensor::zeros([seq_len, half_dim], &cpu).unwrap();
        rope.embed_into_cpu(&pos_ids, seq_len, &mut cos_cpu, &mut sin_cpu).unwrap();
        let cpu_cos = cos_cpu.to_host_vec().unwrap();
        let cpu_sin = sin_cpu.to_host_vec().unwrap();

        let mut cos_gpu: Tensor<f32, Cuda> = Tensor::zeros([seq_len, half_dim], &cuda).unwrap();
        let mut sin_gpu: Tensor<f32, Cuda> = Tensor::zeros([seq_len, half_dim], &cuda).unwrap();
        rope.embed_into_cuda(&pos_ids, seq_len, &mut cos_gpu, &mut sin_gpu).unwrap();
        let gpu_cos = cos_gpu.to_host_vec().unwrap();
        let gpu_sin = sin_gpu.to_host_vec().unwrap();

        assert_eq!(cpu_cos, gpu_cos);
        assert_eq!(cpu_sin, gpu_sin);
    }

    #[test]
    fn fill_cap_pos_ids_basic() {
        let ids = fill_cap_pos_ids(3);
        assert_eq!(ids, vec![1, 0, 0,   2, 0, 0,   3, 0, 0]);
    }

    #[test]
    fn fill_image_pos_ids_basic_and_pad() {
        // f_t=1, h_t=2, w_t=2 → 4 patches. Axis-0 starts at 5; pad 2 tokens
        // → repeat last row.
        let ids = fill_image_pos_ids(1, 2, 2, 5, 2);
        assert_eq!(ids.len(), (4 + 2) * 3);
        // patches in (h,w)-major order:
        // (0,0)→(5,0,0), (0,1)→(5,0,1), (1,0)→(5,1,0), (1,1)→(5,1,1).
        assert_eq!(&ids[..3], &[5, 0, 0]);
        assert_eq!(&ids[3..6], &[5, 0, 1]);
        assert_eq!(&ids[6..9], &[5, 1, 0]);
        assert_eq!(&ids[9..12], &[5, 1, 1]);
        // Pad rows = last patch repeated.
        assert_eq!(&ids[12..15], &[5, 1, 1]);
        assert_eq!(&ids[15..18], &[5, 1, 1]);
    }
}
