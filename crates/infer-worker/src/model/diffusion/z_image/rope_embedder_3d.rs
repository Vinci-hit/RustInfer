//! 3D Rotary Position Embedding for DiT (Z-Image).
//!
//! 将位置信息沿 temporal(t)、height(h)、width(w) 三个轴独立编码，
//! 在 head 维度上拼接。与 LLM 1D RoPE 完全独立。
//!
//! ```text
//! 初始化: precompute → cos_cached[3], sin_cached[3]
//!
//! forward:
//!   pos_ids [S, 3] → embed() → cos [S, 64], sin [S, 64]
//!   Q [S, H, D=128] → apply_rope_interleaved(Q, cos, sin) → Q'
//!   K [S, H, D=128] → apply_rope_interleaved(K, cos, sin) → K'
//! ```

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

/// 3D RoPE embedder for DiT models.
///
/// 预计算 3 组独立的 cos/sin 缓存（temporal / height / width），
/// 推理时根据每个 token 的 `[t, h, w]` 位置索引查表拼接。
pub struct RopeEmbedder3D {
    pub theta: f64,
    pub axes_dims: [usize; 3],  // e.g. [32, 48, 48]
    pub axes_lens: [usize; 3],  // e.g. [1536, 512, 512]
    /// 3 组 cos cache, shape: [axes_lens[i], axes_dims[i] / 2], dtype F32
    cos_cached: [Tensor; 3],
    /// 3 组 sin cache, shape: [axes_lens[i], axes_dims[i] / 2], dtype F32
    sin_cached: [Tensor; 3],
}

impl RopeEmbedder3D {
    /// 创建并预计算 3D RoPE cache.
    ///
    /// - `axes_dims`: 每轴分配的旋转维度，sum 应等于 head_dim (e.g. [32, 48, 48] → 128)
    /// - `axes_lens`: 每轴最大位置长度 (e.g. [1536, 512, 512])
    /// - `theta`: 频率基底 (Z-Image 用 256.0)
    pub fn new(
        axes_dims: [usize; 3],
        axes_lens: [usize; 3],
        theta: f64,
    ) -> Result<Self> {
        let (cos_cached, sin_cached) = Self::precompute_freqs(&axes_dims, &axes_lens, theta)?;
        Ok(Self {
            theta,
            axes_dims,
            axes_lens,
            cos_cached,
            sin_cached,
        })
    }

    /// 预计算每轴的 cos/sin 频率表.
    ///
    /// 对轴 i (dim=d, len=e):
    ///   freq[j] = 1.0 / (theta ^ (2j / d)),  j = 0..d/2
    ///   cache[pos][j] = cos/sin(pos * freq[j]),  pos = 0..e
    ///
    /// 返回 3 组 Tensor, 各 shape [axes_lens[i], axes_dims[i] / 2]
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

            // 频率基向量: freq[j] = 1/(theta^(2j/d)), j=0..half_d
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

    /// 总的 half-dim = sum(axes_dims) / 2
    pub fn half_dim(&self) -> usize {
        self.axes_dims.iter().sum::<usize>() / 2
    }

    /// 根据 position IDs 查表拼接出 cos/sin embedding.
    ///
    /// - `pos_ids`: shape [seq_len, 3], dtype I32, 每行 = [t, h, w]
    /// - 返回: (cos, sin), 各 shape [seq_len, half_dim], dtype F32
    ///
    /// 拼接顺序: [temporal_freqs | height_freqs | width_freqs]
    pub fn embed(&self, pos_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        if pos_ids.shape().len() != 2 || pos_ids.shape()[1] != 3 {
            return Err(Error::InvalidArgument(format!(
                "RopeEmbedder3D::embed: pos_ids must be [seq_len, 3], got {:?}",
                pos_ids.shape()
            )).into());
        }

        let seq_len = pos_ids.shape()[0];
        let half_dim = self.half_dim();
        let ids = pos_ids.as_i32()?.as_slice()?;

        let mut cos_out = Tensor::new(&[seq_len, half_dim], DataType::F32, DeviceType::Cpu)?;
        let mut sin_out = Tensor::new(&[seq_len, half_dim], DataType::F32, DeviceType::Cpu)?;

        let cos_out_slice = cos_out.as_f32_mut()?.as_slice_mut()?;
        let sin_out_slice = sin_out.as_f32_mut()?.as_slice_mut()?;

        // 每轴在输出中的起始偏移
        let offsets = [0, self.axes_dims[0] / 2, self.axes_dims[0] / 2 + self.axes_dims[1] / 2];

        for axis in 0..3usize {
            let half_d = self.axes_dims[axis] / 2;
            let offset = offsets[axis];
            let cos_cache = self.cos_cached[axis].as_f32()?.as_slice()?;
            let sin_cache = self.sin_cached[axis].as_f32()?.as_slice()?;

            for token in 0..seq_len {
                let pos = ids[token * 3 + axis] as usize;
                debug_assert!(pos < self.axes_lens[axis],
                    "axis {} pos {} >= max {}", axis, pos, self.axes_lens[axis]);

                let cache_base = pos * half_d;
                let out_base = token * half_dim + offset;

                cos_out_slice[out_base..out_base + half_d]
                    .copy_from_slice(&cos_cache[cache_base..cache_base + half_d]);
                sin_out_slice[out_base..out_base + half_d]
                    .copy_from_slice(&sin_cache[cache_base..cache_base + half_d]);
            }
        }

        Ok((cos_out, sin_out))
    }

    /// 将 cache 搬到 CUDA 设备（用于后续 GPU embed）。
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        for i in 0..3 {
            self.cos_cached[i] = self.cos_cached[i].to_cuda(device_id)?;
            self.sin_cached[i] = self.sin_cached[i].to_cuda(device_id)?;
        }
        Ok(())
    }
}

/// 原地 interleaved RoPE 旋转（GPT-J 风格）。
///
/// 对 Q/K 的每个 head 做偶奇交错旋转：
/// ```text
/// x_out[2k]   = x[2k]   * cos[k] - x[2k+1] * sin[k]
/// x_out[2k+1] = x[2k+1] * cos[k] + x[2k]   * sin[k]
/// ```
///
/// - `x`: [seq_len, n_heads, head_dim] 或 [seq_len, hidden_dim]（会按 head_dim 分头旋转）
/// - `cos`: [seq_len, head_dim/2]
/// - `sin`: [seq_len, head_dim/2]
/// - `head_dim`: 每个 head 的维度 (e.g. 128)
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

    // cos/sin 每行 half_hd 个值
    let cos_data = cos.as_f32()?.as_slice()?;
    let sin_data = sin.as_f32()?.as_slice()?;

    match x {
        Tensor::F32(typed) => {
            let x_data = typed.as_slice_mut()?;
            for s in 0..seq_len {
                let cos_row = &cos_data[s * half_hd..(s + 1) * half_hd];
                let sin_row = &sin_data[s * half_hd..(s + 1) * half_hd];
                let x_row = &mut x_data[s * total_dim..(s + 1) * total_dim];

                // 遍历每个 head（或 head_dim 大小的 chunk）
                for chunk in x_row.chunks_exact_mut(head_dim) {
                    for k in 0..half_hd {
                        let x0 = chunk[2 * k];
                        let x1 = chunk[2 * k + 1];
                        let c = cos_row[k];
                        let sv = sin_row[k];
                        chunk[2 * k] = x0 * c - x1 * sv;
                        chunk[2 * k + 1] = x1 * c + x0 * sv;
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
                        let c = cos_row[k];
                        let sv = sin_row[k];
                        chunk[2 * k] = half::bf16::from_f32(x0 * c - x1 * sv);
                        chunk[2 * k + 1] = half::bf16::from_f32(x1 * c + x0 * sv);
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

        // axis 0: [1536, 16]
        assert_eq!(rope.cos_cached[0].shape(), &[1536, 16]);
        assert_eq!(rope.sin_cached[0].shape(), &[1536, 16]);
        // axis 1: [512, 24]
        assert_eq!(rope.cos_cached[1].shape(), &[512, 24]);
        // axis 2: [512, 24]
        assert_eq!(rope.cos_cached[2].shape(), &[512, 24]);

        assert_eq!(rope.half_dim(), 64); // 16+24+24
        Ok(())
    }

    #[test]
    fn test_precompute_values_at_pos_zero() -> Result<()> {
        let rope = RopeEmbedder3D::new([4, 4, 4], [8, 8, 8], 256.0)?;

        // pos=0 → cos=1, sin=0 for all freqs
        let cos0 = rope.cos_cached[0].as_f32()?.as_slice()?;
        let sin0 = rope.sin_cached[0].as_f32()?.as_slice()?;
        let half_d = 2; // 4/2
        for j in 0..half_d {
            assert!((cos0[j] - 1.0).abs() < 1e-6, "cos[0][{}] = {}", j, cos0[j]);
            assert!(sin0[j].abs() < 1e-6, "sin[0][{}] = {}", j, sin0[j]);
        }
        Ok(())
    }

    #[test]
    fn test_precompute_freq_formula() -> Result<()> {
        // 验证 freq[j] = 1/(theta^(2j/d)) 精确值
        let theta = 256.0_f64;
        let d = 8;
        let rope = RopeEmbedder3D::new([d, d, d], [16, 16, 16], theta)?;

        let cos_data = rope.cos_cached[0].as_f32()?.as_slice()?.to_vec();
        let half_d = d / 2; // 4

        // 验证 pos=1 的 cos 值
        for j in 0..half_d {
            let expected_freq = 1.0 / theta.powf(2.0 * j as f64 / d as f64);
            let expected_cos = (1.0 * expected_freq).cos() as f32;
            let actual = cos_data[1 * half_d + j]; // pos=1, freq j
            assert!(
                (actual - expected_cos).abs() < 1e-5,
                "freq mismatch at j={}: expected {}, got {}", j, expected_cos, actual
            );
        }
        Ok(())
    }

    #[test]
    fn test_embed_caption_tokens() -> Result<()> {
        // Caption tokens: h=0, w=0, only t varies
        let rope = RopeEmbedder3D::new([32, 48, 48], [1536, 512, 512], 256.0)?;

        let seq_len = 4;
        let mut pos_ids = Tensor::new(&[seq_len, 3], DataType::I32, DeviceType::Cpu)?;
        let ids = pos_ids.as_i32_mut()?.as_slice_mut()?;
        // caption: t=1,2,3,4  h=0,w=0
        ids[0] = 1; ids[1] = 0; ids[2] = 0;   // token 0
        ids[3] = 2; ids[4] = 0; ids[5] = 0;   // token 1
        ids[6] = 3; ids[7] = 0; ids[8] = 0;   // token 2
        ids[9] = 4; ids[10] = 0; ids[11] = 0;  // token 3

        let (cos, sin) = rope.embed(&pos_ids)?;
        assert_eq!(cos.shape(), &[4, 64]);
        assert_eq!(sin.shape(), &[4, 64]);

        let cos_data = cos.as_f32()?.as_slice()?;

        // h=0, w=0 → height/width 部分的 cos 应该全是 1.0 (cos(0*freq)=1)
        for token in 0..seq_len {
            let row = &cos_data[token * 64..token * 64 + 64];
            // height part: offset 16..40
            for j in 16..40 {
                assert!((row[j] - 1.0).abs() < 1e-6,
                    "token {} height cos[{}] = {}, expected 1.0", token, j, row[j]);
            }
            // width part: offset 40..64
            for j in 40..64 {
                assert!((row[j] - 1.0).abs() < 1e-6,
                    "token {} width cos[{}] = {}, expected 1.0", token, j, row[j]);
            }
        }

        // temporal 部分 (offset 0..16) 应在不同 token 间变化
        let t0_cos_0 = cos_data[0];      // token 0, temporal freq 0
        let t1_cos_0 = cos_data[64];     // token 1, temporal freq 0
        assert!(
            (t0_cos_0 - t1_cos_0).abs() > 1e-6,
            "temporal cos should differ between tokens"
        );

        Ok(())
    }

    #[test]
    fn test_embed_image_grid() -> Result<()> {
        // 2x2 image grid: (t=10, h=0..2, w=0..2)
        let rope = RopeEmbedder3D::new([32, 48, 48], [1536, 512, 512], 256.0)?;

        let mut pos_ids = Tensor::new(&[4, 3], DataType::I32, DeviceType::Cpu)?;
        let ids = pos_ids.as_i32_mut()?.as_slice_mut()?;
        // (t, h, w): (10,0,0), (10,0,1), (10,1,0), (10,1,1)
        ids[0] = 10; ids[1] = 0; ids[2] = 0;
        ids[3] = 10; ids[4] = 0; ids[5] = 1;
        ids[6] = 10; ids[7] = 1; ids[8] = 0;
        ids[9] = 10; ids[10] = 1; ids[11] = 1;

        let (cos, _sin) = rope.embed(&pos_ids)?;
        let data = cos.as_f32()?.as_slice()?;

        // token 0 和 1: same t, same h, different w → temporal 和 height 部分相同, width 不同
        let row0 = &data[0..64];
        let row1 = &data[64..128];

        // temporal (0..16) 相同
        for j in 0..16 {
            assert!((row0[j] - row1[j]).abs() < 1e-6,
                "temporal should match: j={}, {} vs {}", j, row0[j], row1[j]);
        }
        // height (16..40) 相同
        for j in 16..40 {
            assert!((row0[j] - row1[j]).abs() < 1e-6,
                "height should match: j={}", j);
        }
        // width (40..64) 不同 (w=0 vs w=1)
        let mut any_diff = false;
        for j in 40..64 {
            if (row0[j] - row1[j]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "width part should differ between w=0 and w=1");

        Ok(())
    }

    #[test]
    fn test_apply_rope_interleaved_identity_at_pos0() -> Result<()> {
        // pos=0 → cos=1, sin=0 → x 不变
        let head_dim = 8;
        let half_hd = head_dim / 2;
        let n_heads = 2;
        let seq_len = 1;
        let total = seq_len * n_heads * head_dim;

        let mut x = Tensor::new(&[seq_len, n_heads * head_dim], DataType::F32, DeviceType::Cpu)?;
        let x_data = x.as_f32_mut()?.as_slice_mut()?;
        for i in 0..total { x_data[i] = (i + 1) as f32; }
        let original: Vec<f32> = x_data.to_vec();

        // cos=1, sin=0 (pos=0)
        let mut cos = Tensor::new(&[seq_len, half_hd], DataType::F32, DeviceType::Cpu)?;
        let mut sin = Tensor::new(&[seq_len, half_hd], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.fill(1.0);
        sin.as_f32_mut()?.as_slice_mut()?.fill(0.0);

        apply_rope_interleaved(&mut x, &cos, &sin, head_dim)?;

        let result = x.as_f32()?.as_slice()?;
        for i in 0..total {
            assert!((result[i] - original[i]).abs() < 1e-6,
                "pos=0 should be identity, idx={}: {} vs {}", i, result[i], original[i]);
        }
        Ok(())
    }

    #[test]
    fn test_apply_rope_interleaved_math() -> Result<()> {
        // 手动验证: x=[1,2,3,4], cos=[c0,c1], sin=[s0,s1]
        // 预期: [1*c0-2*s0, 2*c0+1*s0, 3*c1-4*s1, 4*c1+3*s1]
        let head_dim = 4;
        let half_hd = 2;

        let mut x = Tensor::new(&[1, head_dim], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let c0 = 0.8_f32;
        let c1 = 0.6_f32;
        let s0 = 0.3_f32;
        let s1 = 0.7_f32;

        let mut cos = Tensor::new(&[1, half_hd], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[c0, c1]);
        let mut sin = Tensor::new(&[1, half_hd], DataType::F32, DeviceType::Cpu)?;
        sin.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[s0, s1]);

        apply_rope_interleaved(&mut x, &cos, &sin, head_dim)?;

        let result = x.as_f32()?.as_slice()?;
        let expected = [
            1.0 * c0 - 2.0 * s0,  // 0.8 - 0.6 = 0.2
            2.0 * c0 + 1.0 * s0,  // 1.6 + 0.3 = 1.9
            3.0 * c1 - 4.0 * s1,  // 1.8 - 2.8 = -1.0
            4.0 * c1 + 3.0 * s1,  // 2.4 + 2.1 = 4.5
        ];
        for i in 0..4 {
            assert!((result[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: {} vs {}", i, result[i], expected[i]);
        }
        Ok(())
    }

    #[test]
    fn test_apply_rope_interleaved_bf16() -> Result<()> {
        let head_dim = 4;
        let half_hd = 2;

        let mut x = Tensor::new(&[1, head_dim], DataType::BF16, DeviceType::Cpu)?;
        let xd = x.as_bf16_mut()?.as_slice_mut()?;
        xd[0] = half::bf16::from_f32(1.0);
        xd[1] = half::bf16::from_f32(2.0);
        xd[2] = half::bf16::from_f32(3.0);
        xd[3] = half::bf16::from_f32(4.0);

        let mut cos = Tensor::new(&[1, half_hd], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.8, 0.6]);
        let mut sin = Tensor::new(&[1, half_hd], DataType::F32, DeviceType::Cpu)?;
        sin.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.3, 0.7]);

        apply_rope_interleaved(&mut x, &cos, &sin, head_dim)?;

        let result: Vec<f32> = x.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        let expected = [0.2, 1.9, -1.0, 4.5];
        for i in 0..4 {
            assert!((result[i] - expected[i]).abs() < 0.05,
                "bf16 mismatch at {}: {} vs {}", i, result[i], expected[i]);
        }
        Ok(())
    }

    #[test]
    fn test_embed_then_apply_multi_head() -> Result<()> {
        // 完整流程: embed → apply, 多 head
        let rope = RopeEmbedder3D::new([4, 4, 4], [16, 16, 16], 256.0)?;
        let head_dim = 4 + 4 + 4; // = 12 (sum of axes_dims)
        let n_heads = 2;

        // 2 tokens: (t=0,h=0,w=0) 和 (t=1,h=2,w=3)
        let mut pos_ids = Tensor::new(&[2, 3], DataType::I32, DeviceType::Cpu)?;
        let ids = pos_ids.as_i32_mut()?.as_slice_mut()?;
        ids[0] = 0; ids[1] = 0; ids[2] = 0;
        ids[3] = 1; ids[4] = 2; ids[5] = 3;

        let (cos, sin) = rope.embed(&pos_ids)?;
        assert_eq!(cos.shape(), &[2, 6]); // half_dim = 2+2+2 = 6

        // Q tensor [2, 24] = [seq=2, n_heads * head_dim = 2*12]
        let mut q = Tensor::new(&[2, n_heads * head_dim], DataType::F32, DeviceType::Cpu)?;
        q.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate()
            .for_each(|(i, v)| *v = (i as f32 + 1.0) * 0.1);

        apply_rope_interleaved(&mut q, &cos, &sin, head_dim)?;

        // token 0 at pos (0,0,0) → cos=1, sin=0 → 数据不变
        let result = q.as_f32()?.as_slice()?;
        for i in 0..n_heads * head_dim {
            let expected = (i as f32 + 1.0) * 0.1;
            assert!((result[i] - expected).abs() < 1e-5,
                "pos=0 token should be unchanged at {}", i);
        }

        // token 1 at pos (1,2,3) → 应被旋转，与原值不同
        let token1_start = n_heads * head_dim;
        let mut any_diff = false;
        for i in 0..n_heads * head_dim {
            let original = (token1_start + i) as f32 * 0.1 + 0.1;
            if (result[token1_start + i] - original).abs() > 1e-4 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "token at non-zero pos should be rotated");

        Ok(())
    }
}
