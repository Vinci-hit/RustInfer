//! Interleaved (GPT-J style) RoPE rotation, in-place.
//!
//! Input layout: `[seq, n_heads, head_dim]`, with `cos`/`sin` of shape
//! `[seq, head_dim/2]` (always F32). The rotation pairs adjacent entries
//! `(x[2k], x[2k+1])` within each head and applies a planar rotation by
//! `(cos[s, k], sin[s, k])`.
//!
//! Both CPU and CUDA implementations live here — previously the CPU
//! fallback was stashed inside `model::diffusion::z_image::rope_embedder_3d`,
//! which reversed the intended layering (op must not depend on model).

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn rope_interleaved_f32_forward(x: *mut f32,
        cos: *const f32, sin: *const f32,
        seq: i32, n_heads: i32, head_dim: i32, stream: cudaStream_t);
    fn rope_interleaved_bf16_forward(x: *mut half::bf16,
        cos: *const f32, sin: *const f32,
        seq: i32, n_heads: i32, head_dim: i32, stream: cudaStream_t);
}

/// Apply interleaved RoPE in-place. `x` is `[seq, n_heads, head_dim]`,
/// `cos` / `sin` are `[seq, head_dim/2]` in F32.
pub fn apply_rope_interleaved(
    x: &mut Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
) -> Result<()> {
    let shape = x.shape().to_vec();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: expected [seq, n_heads, head_dim], got {:?}", shape
        )).into());
    }
    let (seq, n_heads, d) = (shape[0], shape[1], shape[2]);
    if d != head_dim {
        return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: head_dim mismatch: shape={}, arg={}", d, head_dim
        )).into());
    }
    let half = head_dim / 2;
    let cos_shape = cos.shape();
    if cos_shape.len() != 2 || cos_shape[0] != seq || cos_shape[1] != half {
        return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: cos shape mismatch: {:?} vs expected [{}, {}]",
            cos_shape, seq, half
        )).into());
    }
    if cos.dtype() != DataType::F32 || sin.dtype() != DataType::F32 {
        return Err(Error::InvalidArgument(
            "apply_rope_interleaved: cos/sin must be F32".into()
        ).into());
    }

    match x.device() {
        DeviceType::Cpu => rope_interleaved_cpu(x, cos, sin, head_dim),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            let cos_p = cos.as_f32()?.data_ptr();
            let sin_p = sin.as_f32()?.data_ptr();
            match x.dtype() {
                DataType::F32 => unsafe {
                    rope_interleaved_f32_forward(
                        x.as_f32_mut()?.data_ptr_mut(),
                        cos_p, sin_p,
                        seq as i32, n_heads as i32, head_dim as i32, stream);
                }
                DataType::BF16 => unsafe {
                    rope_interleaved_bf16_forward(
                        x.as_bf16_mut()?.data_ptr_mut(),
                        cos_p, sin_p,
                        seq as i32, n_heads as i32, head_dim as i32, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "apply_rope_interleaved CUDA: unsupported dtype {:?}", other
                )).into()),
            }
            Ok(())
        }
    }
}

// ─────────────────────── CPU kernel ───────────────────────

fn rope_interleaved_cpu(
    x: &mut Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
) -> Result<()> {
    let x_shape = x.shape().to_vec();
    let seq_len = x_shape[0];
    let total_dim: usize = x_shape.iter().skip(1).product();
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
                        chunk[2 * k]     = x0 * cos_row[k] - x1 * sin_row[k];
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
                        chunk[2 * k]     = half::bf16::from_f32(x0 * cos_row[k] - x1 * sin_row[k]);
                        chunk[2 * k + 1] = half::bf16::from_f32(x1 * cos_row[k] + x0 * sin_row[k]);
                    }
                }
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved CPU: unsupported dtype {:?}", x.dtype()
        )).into()),
    }
    Ok(())
}

// ─────────────────────── tests ───────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    /// cos=1, sin=0 ⇒ identity.
    #[test]
    fn cpu_identity_f32() -> Result<()> {
        let seq = 2; let h = 3; let hd = 8; let half = hd / 2;
        let mut x = Tensor::empty(&[seq, h, hd], DataType::F32, DeviceType::Cpu)?;
        for (i, v) in x.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 + 0.5;
        }
        let orig: Vec<f32> = x.as_f32()?.as_slice()?.to_vec();

        let mut cos = Tensor::empty(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        let mut sin = Tensor::empty(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.fill(1.0);
        sin.as_f32_mut()?.as_slice_mut()?.fill(0.0);

        apply_rope_interleaved(&mut x, &cos, &sin, hd)?;
        assert_eq!(x.as_f32()?.as_slice()?, orig.as_slice());
        Ok(())
    }

    /// Explicit single-head rotation check.
    #[test]
    fn cpu_known_rotation_f32() -> Result<()> {
        // head_dim=4, half=2, seq=1, heads=1
        let mut x = Tensor::empty(&[1, 1, 4], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let mut cos = Tensor::empty(&[1, 2], DataType::F32, DeviceType::Cpu)?;
        let mut sin = Tensor::empty(&[1, 2], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.8, 0.6]);
        sin.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.3, 0.7]);

        apply_rope_interleaved(&mut x, &cos, &sin, 4)?;
        // pair 0: (1, 2) with (cos=0.8, sin=0.3)
        //   x0' = 1*0.8 - 2*0.3 = 0.2
        //   x1' = 2*0.8 + 1*0.3 = 1.9
        // pair 1: (3, 4) with (cos=0.6, sin=0.7)
        //   x0' = 3*0.6 - 4*0.7 = -1.0
        //   x1' = 4*0.6 + 3*0.7 = 4.5
        let out = x.as_f32()?.as_slice()?;
        let expected = [0.2f32, 1.9, -1.0, 4.5];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-5,
                "pair mismatch at {}: {} vs {}", i, out[i], expected[i]);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_matches_cpu_f32() -> Result<()> {
        let seq = 4; let h = 2; let hd = 8; let half = hd / 2;
        let mut x_cpu = Tensor::empty(&[seq, h, hd], DataType::F32, DeviceType::Cpu)?;
        for (i, v) in x_cpu.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.11 - 0.5;
        }
        let mut cos = Tensor::empty(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        let mut sin = Tensor::empty(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        for k in 0..(seq * half) {
            cos.as_f32_mut()?.as_slice_mut()?[k] = ((k as f32) * 0.07).cos();
            sin.as_f32_mut()?.as_slice_mut()?[k] = ((k as f32) * 0.07).sin();
        }

        let mut x_gpu = x_cpu.to_cuda(0)?;
        let cos_g = cos.to_cuda(0)?;
        let sin_g = sin.to_cuda(0)?;

        apply_rope_interleaved(&mut x_cpu, &cos, &sin, hd)?;
        apply_rope_interleaved(&mut x_gpu, &cos_g, &sin_g, hd)?;
        let back = x_gpu.to_cpu()?;
        let (a, b) = (x_cpu.as_f32()?.as_slice()?, back.as_f32()?.as_slice()?);
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-4,
                "cuda/cpu mismatch at {}: {} vs {}", i, a[i], b[i]);
        }
        Ok(())
    }
}
