//! Interleaved RoPE for DiT (3D RoPE, applied per-head on `[seq, n_heads, head_dim]`).
//!
//! For each token `s` and head `h`, pairs `(x[2k], x[2k+1])` are rotated by
//! `(cos[s,k], sin[s,k])`:
//!   `x'[2k]   = x[2k] * cos − x[2k+1] * sin`
//!   `x'[2k+1] = x[2k] * sin + x[2k+1] * cos`
//!
//! `cos` / `sin` are F32 caches of shape `[seq, head_dim/2]`.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn rope_interleaved_f32_forward(
        x: *mut f32,
        cos: *const f32, sin: *const f32,
        seq: i32, n_heads: i32, head_dim: i32,
        stream: cudaStream_t,
    );
    fn rope_interleaved_bf16_forward(
        x: *mut half::bf16,
        cos: *const f32, sin: *const f32,
        seq: i32, n_heads: i32, head_dim: i32,
        stream: cudaStream_t,
    );
}

/// Apply interleaved RoPE in-place to `x` of shape `[seq, n_heads, head_dim]`.
/// `cos` / `sin` are `[seq, head_dim/2]`, both F32.
pub fn apply_rope_interleaved<T: Dtype>(
    x: &mut Tensor<T, Cuda>,
    cos: &Tensor<f32, Cuda>,
    sin: &Tensor<f32, Cuda>,
    head_dim: usize,
) -> OpResult<()> {
    let xs = x.shape().as_slice();
    if xs.len() != 3 {
        return Err(OpError::Shape(format!(
            "apply_rope_interleaved: expected [seq, n_heads, head_dim], got {:?}", xs
        )));
    }
    let (seq, n_heads, hd) = (xs[0], xs[1], xs[2]);
    if hd != head_dim {
        return Err(OpError::Shape(format!(
            "apply_rope_interleaved: head_dim mismatch x_shape={} arg={}", hd, head_dim,
        )));
    }
    if head_dim % 2 != 0 {
        return Err(OpError::Shape(format!(
            "apply_rope_interleaved: head_dim must be even, got {}", head_dim,
        )));
    }
    let half = head_dim / 2;
    let cs = cos.shape().as_slice();
    let ss = sin.shape().as_slice();
    if cs != [seq, half] || ss != [seq, half] {
        return Err(OpError::Shape(format!(
            "apply_rope_interleaved: cos/sin shape mismatch cos={:?} sin={:?} expected=[{}, {}]",
            cs, ss, seq, half,
        )));
    }
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => rope_interleaved_f32_forward(
                x.data_ptr_mut() as *mut f32,
                cos.data_ptr() as *const f32,
                sin.data_ptr() as *const f32,
                seq as i32, n_heads as i32, head_dim as i32, stream,
            ),
            DataType::BF16 => rope_interleaved_bf16_forward(
                x.data_ptr_mut() as *mut half::bf16,
                cos.data_ptr() as *const f32,
                sin.data_ptr() as *const f32,
                seq as i32, n_heads as i32, head_dim as i32, stream,
            ),
            _ => return Err(OpError::Kernel(format!(
                "apply_rope_interleaved: unsupported dtype {:?}", T::DATA_TYPE,
            ))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::Shape;
    use half::bf16;

    /// CPU reference: interleaved RoPE.
    fn rope_interleaved_cpu_f32(
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        seq: usize, n_heads: usize, head_dim: usize,
    ) {
        let half = head_dim / 2;
        for s in 0..seq {
            for h in 0..n_heads {
                for k in 0..half {
                    let c = cos[s * half + k];
                    let si = sin[s * half + k];
                    let base = (s * n_heads + h) * head_dim;
                    let a = x[base + 2 * k];
                    let b = x[base + 2 * k + 1];
                    x[base + 2 * k]     = a * c - b * si;
                    x[base + 2 * k + 1] = a * si + b * c;
                }
            }
        }
    }

    #[test]
    fn rope_interleaved_f32_matches_cpu() {
        let cuda = Cuda::new(0).expect("cuda init");
        let (seq, n_heads, head_dim) = (4usize, 2usize, 8usize);
        let half = head_dim / 2;

        // Random input + cos/sin caches.
        let mut x_host: Vec<f32> = (0..seq * n_heads * head_dim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let cos_host: Vec<f32> = (0..seq * half)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let sin_host: Vec<f32> = (0..seq * half)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();

        // CPU reference.
        let mut ref_x = x_host.clone();
        rope_interleaved_cpu_f32(&mut ref_x, &cos_host, &sin_host, seq, n_heads, head_dim);

        // CUDA execution.
        let mut x_dev: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &x_host, Shape::from_slice(&[seq, n_heads, head_dim]), &cuda,
        ).unwrap();
        let cos_dev: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &cos_host, Shape::from_slice(&[seq, half]), &cuda,
        ).unwrap();
        let sin_dev: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &sin_host, Shape::from_slice(&[seq, half]), &cuda,
        ).unwrap();

        apply_rope_interleaved(&mut x_dev, &cos_dev, &sin_dev, head_dim).unwrap();

        let got = x_dev.to_host_vec().unwrap();
        let _ = x_host; // shadow drop
        for (i, (a, b)) in ref_x.iter().zip(got.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mismatch at {}: cpu={} gpu={}", i, a, b);
        }
    }

    #[test]
    fn rope_interleaved_bf16_matches_cpu() {
        let cuda = Cuda::new(0).expect("cuda init");
        let (seq, n_heads, head_dim) = (4usize, 2usize, 8usize);
        let half = head_dim / 2;

        let x_host_f32: Vec<f32> = (0..seq * n_heads * head_dim)
            .map(|i| (i as f32 * 0.21).sin())
            .collect();
        let cos_host: Vec<f32> = (0..seq * half)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let sin_host: Vec<f32> = (0..seq * half)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();

        let mut ref_x = x_host_f32.clone();
        rope_interleaved_cpu_f32(&mut ref_x, &cos_host, &sin_host, seq, n_heads, head_dim);

        let x_host_bf16: Vec<bf16> = x_host_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let mut x_dev: Tensor<bf16, Cuda> = Tensor::from_host_slice(
            &x_host_bf16, Shape::from_slice(&[seq, n_heads, head_dim]), &cuda,
        ).unwrap();
        let cos_dev: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &cos_host, Shape::from_slice(&[seq, half]), &cuda,
        ).unwrap();
        let sin_dev: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &sin_host, Shape::from_slice(&[seq, half]), &cuda,
        ).unwrap();

        apply_rope_interleaved(&mut x_dev, &cos_dev, &sin_dev, head_dim).unwrap();

        let got: Vec<f32> = x_dev.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        for (i, (a, b)) in ref_x.iter().zip(got.iter()).enumerate() {
            // bf16 has ~7-bit mantissa → ~1% relative error at most.
            let abs_err = (a - b).abs();
            let rel_err = abs_err / a.abs().max(1e-3);
            assert!(abs_err < 0.05 || rel_err < 0.03,
                "mismatch at {}: cpu={} gpu={} abs={}", i, a, b, abs_err);
        }
    }
}
