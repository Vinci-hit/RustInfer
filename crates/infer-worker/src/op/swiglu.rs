//! SwiGLU: in-place `x = SiLU(x) * y`.
//!
//! Free-function API — the previous `SwiGLU::new().forward(...)` wrapper
//! was a zero-sized struct that added nothing over a direct call.

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;
use crate::OpConfig;

use super::kernels;

/// In-place SwiGLU: `x[i] = silu(x[i]) * y[i]`. Shapes must match.
pub fn swiglu(y: &Tensor, x: &mut Tensor, cuda_config: Option<&OpConfig>) -> Result<()> {
    match y.device() {
        DeviceType::Cpu => {
            let _ = cuda_config;
            kernels::cpu::swiglu(y, x)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::swiglu(y, x, cuda_config),
    }
}

// ─────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;
    use half::bf16;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol,
                "Mismatch at {}: {} vs {} (diff={})", i, x, y, (x - y).abs());
        }
    }

    fn assert_bf16_close(a: &[bf16], b: &[bf16], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x.to_f32() - y.to_f32()).abs() < tol,
                "BF16 mismatch at {}: {} vs {}", i, x.to_f32(), y.to_f32());
        }
    }

    #[test]
    fn swiglu_cpu_known_values() -> Result<()> {
        // SwiGLU: x[i] = silu(x[i]) * y[i], silu(x) = x / (1 + exp(-x))
        let x_in = [1.0_f32, 2.0, -1.0, 0.5];
        let y_in = [2.0_f32, 1.0, 3.0, 4.0];
        let mut x = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        let mut y = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&x_in);
        y.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&y_in);

        swiglu(&y, &mut x, None)?;

        let got = x.as_f32()?.as_slice()?;
        let expected: Vec<f32> = x_in.iter().zip(y_in.iter())
            .map(|(&xi, &yi)| (xi / (1.0 + (-xi).exp())) * yi)
            .collect();
        assert_close(got, &expected, 1e-5);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn swiglu_cuda_matches_cpu_bf16() -> Result<()> {
        for batch in [1, 4, 8] {
            let (seq, dim) = (20, 256);
            let shape = &[batch, seq, dim];
            let n = batch * seq * dim;

            let x_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i * 7) % 100) as f32 * 0.01)).collect();
            let y_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i * 11) % 80) as f32 * 0.01)).collect();

            // CPU
            let mut x_cpu = Tensor::empty(shape, DataType::BF16, DeviceType::Cpu)?;
            let mut y_cpu = Tensor::empty(shape, DataType::BF16, DeviceType::Cpu)?;
            x_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&x_data);
            y_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&y_data);
            swiglu(&y_cpu, &mut x_cpu, None)?;
            let cpu_out = x_cpu.as_bf16()?.as_slice()?.to_vec();

            // CUDA
            let mut x_g = Tensor::empty(shape, DataType::BF16, DeviceType::Cuda(0))?;
            let mut y_g = Tensor::empty(shape, DataType::BF16, DeviceType::Cuda(0))?;
            x_g.as_bf16_mut()?.buffer_mut().copy_from_host(&x_data)?;
            y_g.as_bf16_mut()?.buffer_mut().copy_from_host(&y_data)?;
            let cfg = crate::cuda::CudaConfig::new()?;
            swiglu(&y_g, &mut x_g, Some(&cfg))?;
            let back = x_g.to_cpu()?;

            assert_bf16_close(&cpu_out, back.as_bf16()?.as_slice()?, 1e-2);
        }
        Ok(())
    }
}
