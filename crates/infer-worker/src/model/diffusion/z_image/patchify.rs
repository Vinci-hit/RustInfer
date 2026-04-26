//! Patchify / Unpatchify for DiT (Z-Image).
//!
//! 将图像 latent [C, F, H, W] 拆成 token 序列 [num_tokens, patch_flat]，
//! 以及逆操作。纯 tensor reshape + permute，无参数。
//!
//! ```text
//! Patchify:   [C, F, H, W] → view → permute(1,3,5,2,4,6,0) → view → [N, D]
//! Unpatchify: [N, D] → view → permute(6,0,3,1,4,2,5) → view → [C, F, H, W]
//! ```

use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

/// 将图像 latent 拆成 patch token 序列。
///
/// - `image`: [C, F, H, W] contiguous
/// - `patch_size`: spatial patch size (pH = pW)
/// - `f_patch_size`: temporal patch size (pF)
///
/// 返回: [F_t × H_t × W_t, pF × pH × pW × C] contiguous
///
/// 其中 F_t = F/pF, H_t = H/pH, W_t = W/pW
pub fn patchify(
    image: &Tensor,
    patch_size: usize,
    f_patch_size: usize,
) -> Result<Tensor> {
    let shape = image.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "patchify: expected [C, F, H, W], got {:?}", shape
        )).into());
    }
    let (c, f, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);

    if f % p_f != 0 || h % p_h != 0 || w % p_w != 0 {
        return Err(Error::InvalidArgument(format!(
            "patchify: (F={}, H={}, W={}) not divisible by (pF={}, pH={}, pW={})",
            f, h, w, p_f, p_h, p_w
        )).into());
    }

    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
    let num_tokens = f_t * h_t * w_t;
    let patch_flat = p_f * p_h * p_w * c;

    // [C, F, H, W] → [C, f_t, pF, h_t, pH, w_t, pW]
    let t = image.view(&[c, f_t, p_f, h_t, p_h, w_t, p_w])?;
    // permute(1, 3, 5, 2, 4, 6, 0) → [f_t, h_t, w_t, pF, pH, pW, C]
    let t = t.permute(&[1, 3, 5, 2, 4, 6, 0])?;
    // reshape → [num_tokens, patch_flat]
    t.view(&[num_tokens, patch_flat])
}

/// 将 patch token 序列还原为图像 latent。
///
/// - `tokens`: [num_tokens, patch_flat] contiguous
/// - `f, h, w`: 原始图像的 (F, H, W) 尺寸
/// - `out_channels`: C
/// - `patch_size`: pH = pW
/// - `f_patch_size`: pF
///
/// 返回: [C, F, H, W] contiguous
pub fn unpatchify(
    tokens: &Tensor,
    f: usize,
    h: usize,
    w: usize,
    out_channels: usize,
    patch_size: usize,
    f_patch_size: usize,
) -> Result<Tensor> {
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);

    // [num_tokens, patch_flat] → [f_t, h_t, w_t, pF, pH, pW, C]
    let t = tokens.view(&[f_t, h_t, w_t, p_f, p_h, p_w, out_channels])?;
    // permute(6, 0, 3, 1, 4, 2, 5) → [C, f_t, pF, h_t, pH, w_t, pW]
    let t = t.permute(&[6, 0, 3, 1, 4, 2, 5])?;
    // reshape → [C, F, H, W]
    t.view(&[out_channels, f, h, w])
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;

    #[test]
    fn test_patchify_shape() -> Result<()> {
        // [16, 1, 512, 512] → [65536, 64]
        let image = Tensor::new(&[16, 1, 512, 512], DataType::F32, DeviceType::Cpu)?;
        let tokens = patchify(&image, 2, 1)?;
        assert_eq!(tokens.shape(), &[65536, 64]);
        Ok(())
    }

    #[test]
    fn test_unpatchify_shape() -> Result<()> {
        let tokens = Tensor::new(&[65536, 64], DataType::F32, DeviceType::Cpu)?;
        let image = unpatchify(&tokens, 1, 512, 512, 16, 2, 1)?;
        assert_eq!(image.shape(), &[16, 1, 512, 512]);
        Ok(())
    }

    #[test]
    fn test_patchify_unpatchify_roundtrip_small() -> Result<()> {
        // 小例子验证数据完整性: [2, 1, 4, 4], patch=2, f_patch=1
        let c = 2;
        let (f, h, w) = (1, 4, 4);
        let numel = c * f * h * w; // 32

        let mut image = Tensor::new(&[c, f, h, w], DataType::F32, DeviceType::Cpu)?;
        let data = image.as_f32_mut()?.as_slice_mut()?;
        for i in 0..numel { data[i] = i as f32; }

        let tokens = patchify(&image, 2, 1)?;
        assert_eq!(tokens.shape(), &[4, 8]); // 1*2*2=4 tokens, 1*2*2*2=8 flat

        let restored = unpatchify(&tokens, f, h, w, c, 2, 1)?;
        assert_eq!(restored.shape(), &[c, f, h, w]);

        let orig = image.as_f32()?.as_slice()?;
        let rest = restored.as_f32()?.as_slice()?;
        for i in 0..numel {
            assert_eq!(orig[i], rest[i], "roundtrip mismatch at {}", i);
        }
        Ok(())
    }

    #[test]
    fn test_patchify_unpatchify_roundtrip_bf16() -> Result<()> {
        use half::bf16;
        let c = 4;
        let (f, h, w) = (2, 8, 8);
        let numel = c * f * h * w;

        let mut image = Tensor::new(&[c, f, h, w], DataType::BF16, DeviceType::Cpu)?;
        let data = image.as_bf16_mut()?.as_slice_mut()?;
        for i in 0..numel { data[i] = bf16::from_f32(i as f32); }

        let tokens = patchify(&image, 2, 2)?;
        // f_t=1, h_t=4, w_t=4 → 16 tokens, flat=2*2*2*4=32
        assert_eq!(tokens.shape(), &[16, 32]);

        let restored = unpatchify(&tokens, f, h, w, c, 2, 2)?;
        let orig = image.as_bf16()?.as_slice()?;
        let rest = restored.as_bf16()?.as_slice()?;
        for i in 0..numel {
            assert_eq!(orig[i].to_f32(), rest[i].to_f32(), "bf16 roundtrip mismatch at {}", i);
        }
        Ok(())
    }

    #[test]
    fn test_patchify_pixel_order() -> Result<()> {
        // 验证 pixel 在 token 内的排列顺序
        // [C=2, F=1, H=2, W=2], patch=2, f_patch=1 → 1 token, flat=1*2*2*2=8
        let mut image = Tensor::new(&[2, 1, 2, 2], DataType::F32, DeviceType::Cpu)?;
        let data = image.as_f32_mut()?.as_slice_mut()?;
        // C=0: [[0,1],[2,3]], C=1: [[4,5],[6,7]]
        for i in 0..8 { data[i] = i as f32; }

        let tokens = patchify(&image, 2, 1)?;
        assert_eq!(tokens.shape(), &[1, 8]);

        // 预期 token 内顺序: (pf, ph, pw, c) 按 row-major
        // pf=0: ph=0,pw=0,c=0 → pixel(c=0,f=0,h=0,w=0) = 0
        //       ph=0,pw=0,c=1 → pixel(c=1,f=0,h=0,w=0) = 4
        //       ph=0,pw=1,c=0 → pixel(c=0,f=0,h=0,w=1) = 1
        //       ph=0,pw=1,c=1 → pixel(c=1,f=0,h=0,w=1) = 5
        //       ph=1,pw=0,c=0 → pixel(c=0,f=0,h=1,w=0) = 2
        //       ph=1,pw=0,c=1 → pixel(c=1,f=0,h=1,w=0) = 6
        //       ph=1,pw=1,c=0 → pixel(c=0,f=0,h=1,w=1) = 3
        //       ph=1,pw=1,c=1 → pixel(c=1,f=0,h=1,w=1) = 7
        let expected = [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
        let result = tokens.as_f32()?.as_slice()?;
        assert_eq!(result, &expected);
        Ok(())
    }

    // --- view/permute 基础测试 ---

    #[test]
    fn test_view_basic() -> Result<()> {
        let mut t = Tensor::new(&[2, 3], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t2 = t.view(&[3, 2])?;
        assert_eq!(t2.shape(), &[3, 2]);
        assert_eq!(t2.as_f32()?.as_slice()?, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_view_wrong_size() {
        let t = Tensor::new(&[2, 3], DataType::F32, DeviceType::Cpu).unwrap();
        assert!(t.view(&[2, 4]).is_err());
    }

    #[test]
    fn test_permute_2d_transpose() -> Result<()> {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let mut t = Tensor::new(&[2, 3], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t2 = t.permute(&[1, 0])?;
        assert_eq!(t2.shape(), &[3, 2]);
        assert_eq!(t2.as_f32()?.as_slice()?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_permute_3d() -> Result<()> {
        // [2, 3, 4] → permute(2, 0, 1) → [4, 2, 3]
        // new[w, i, j] = old[i, j, w]
        let mut t = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        let data = t.as_f32_mut()?.as_slice_mut()?;
        for i in 0..24 { data[i] = i as f32; }

        let t2 = t.permute(&[2, 0, 1])?;
        assert_eq!(t2.shape(), &[4, 2, 3]);

        let r = t2.as_f32()?.as_slice()?;
        // new[0,0,0] = old[0,0,0] = 0
        assert_eq!(r[0], 0.0);
        // new[1,0,0] = old[0,0,1] = 1
        assert_eq!(r[1 * 6], 1.0);
        // new[0,1,0] = old[1,0,0] = 12
        assert_eq!(r[0 * 6 + 1 * 3 + 0], 12.0);
        // new[3,1,2] = old[1,2,3] = 1*12 + 2*4 + 3 = 23
        assert_eq!(r[3 * 6 + 1 * 3 + 2], 23.0);
        Ok(())
    }

    #[test]
    fn test_permute_identity() -> Result<()> {
        let mut t = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        let data = t.as_f32_mut()?.as_slice_mut()?;
        for i in 0..24 { data[i] = i as f32; }

        let t2 = t.permute(&[0, 1, 2])?;
        assert_eq!(t2.as_f32()?.as_slice()?, t.as_f32()?.as_slice()?);
        Ok(())
    }

    // ────────── Tensor::permute CUDA tests ──────────

    /// 2D transpose: F32 CPU vs CUDA
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_2d_cuda_f32() -> Result<()> {
        let mut t_cpu = Tensor::new(&[2, 3], DataType::F32, DeviceType::Cpu)?;
        t_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let cpu_result = t_cpu.permute(&[1, 0])?;

        let t_gpu = t_cpu.to_cuda(0)?;
        let gpu_result = t_gpu.permute(&[1, 0])?;
        let gpu_cpu = gpu_result.to_cpu()?;

        assert_eq!(gpu_result.shape(), &[3, 2]);
        assert_eq!(cpu_result.as_f32()?.as_slice()?, gpu_cpu.as_f32()?.as_slice()?);
        Ok(())
    }

    /// 3D permute: F32 CPU vs CUDA
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_3d_cuda_f32() -> Result<()> {
        let mut t_cpu = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        let data = t_cpu.as_f32_mut()?.as_slice_mut()?;
        for i in 0..24 { data[i] = i as f32; }

        let cpu_result = t_cpu.permute(&[2, 0, 1])?;

        let t_gpu = t_cpu.to_cuda(0)?;
        let gpu_result = t_gpu.permute(&[2, 0, 1])?;
        let gpu_cpu = gpu_result.to_cpu()?;

        assert_eq!(gpu_result.shape(), &[4, 2, 3]);
        assert_eq!(cpu_result.as_f32()?.as_slice()?, gpu_cpu.as_f32()?.as_slice()?);
        Ok(())
    }

    /// 3D permute: BF16 CPU vs CUDA
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_3d_cuda_bf16() -> Result<()> {
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut t_cpu = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        t_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&values);
        let t_cpu = t_cpu.to_dtype(DataType::BF16)?;

        let cpu_result = t_cpu.permute(&[1, 2, 0])?;

        let t_gpu = t_cpu.to_cuda(0)?;
        let gpu_result = t_gpu.permute(&[1, 2, 0])?;
        let gpu_cpu = gpu_result.to_cpu()?;

        assert_eq!(gpu_result.shape(), &[3, 4, 2]);
        let a = cpu_result.as_bf16()?.as_slice()?;
        let b = gpu_cpu.as_bf16()?.as_slice()?;
        assert_eq!(a, b);
        Ok(())
    }

    /// 7D permute (patchify pattern): F32 CPU vs CUDA
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_7d_cuda_f32() -> Result<()> {
        // Simulates patchify: [C, f_t, pF, h_t, pH, w_t, pW] → [f_t, h_t, w_t, pF, pH, pW, C]
        let shape = [2, 3, 2, 4, 2, 4, 2]; // total = 768
        let perm = [1, 3, 5, 2, 4, 6, 0];
        let n: usize = shape.iter().product();

        let mut t_cpu = Tensor::new(&shape, DataType::F32, DeviceType::Cpu)?;
        let data = t_cpu.as_f32_mut()?.as_slice_mut()?;
        for i in 0..n { data[i] = i as f32; }

        let cpu_result = t_cpu.permute(&perm)?;

        let t_gpu = t_cpu.to_cuda(0)?;
        let gpu_result = t_gpu.permute(&perm)?;
        let gpu_cpu = gpu_result.to_cpu()?;

        let expected_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        assert_eq!(gpu_result.shape(), expected_shape.as_slice());
        assert_eq!(cpu_result.as_f32()?.as_slice()?, gpu_cpu.as_f32()?.as_slice()?);
        Ok(())
    }

    /// I32 permute on CUDA
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_2d_cuda_i32() -> Result<()> {
        let mut t_cpu = Tensor::new(&[3, 4], DataType::I32, DeviceType::Cpu)?;
        let data = t_cpu.as_i32_mut()?.as_slice_mut()?;
        for i in 0..12 { data[i] = i as i32; }

        let cpu_result = t_cpu.permute(&[1, 0])?;

        let t_gpu = t_cpu.to_cuda(0)?;
        let gpu_result = t_gpu.permute(&[1, 0])?;
        let gpu_cpu = gpu_result.to_cpu()?;

        assert_eq!(gpu_result.shape(), &[4, 3]);
        assert_eq!(cpu_result.as_i32()?.as_slice()?, gpu_cpu.as_i32()?.as_slice()?);
        Ok(())
    }

    /// Identity permute on CUDA (no-op check)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_identity_cuda() -> Result<()> {
        let mut t_cpu = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        let data = t_cpu.as_f32_mut()?.as_slice_mut()?;
        for i in 0..24 { data[i] = i as f32; }

        let t_gpu = t_cpu.to_cuda(0)?;
        let gpu_result = t_gpu.permute(&[0, 1, 2])?;
        let gpu_cpu = gpu_result.to_cpu()?;

        assert_eq!(t_cpu.as_f32()?.as_slice()?, gpu_cpu.as_f32()?.as_slice()?);
        Ok(())
    }
}
