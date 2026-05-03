//! Element-wise activation functions (in-place).
//!
//! Naming: `<name>_inplace(&mut Tensor)` — mirrors PyTorch's trailing
//! underscore convention (`silu_`, `tanh_`). Each dispatch routes to the
//! CPU or CUDA kernel under `op/kernels/`.

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// In-place SiLU: `x[i] = x[i] * sigmoid(x[i])`.
pub fn silu_inplace(x: &mut Tensor) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => kernels::cpu::silu_inplace(x),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::silu_inplace(
            x, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

/// In-place tanh: `x[i] = tanh(x[i])`.
pub fn tanh_inplace(x: &mut Tensor) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => kernels::cpu::tanh_inplace(x),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::tanh_inplace(
            x, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

// ───────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    #[test]
    fn silu_cpu_matches_reference() -> Result<()> {
        let mut x = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[-2.0, 0.0, 0.5, 2.0]);

        silu_inplace(&mut x)?;

        let out = x.as_f32()?.as_slice()?;
        // silu(x) = x * sigmoid(x)
        for (i, &raw) in [-2.0f32, 0.0, 0.5, 2.0].iter().enumerate() {
            let sig = 1.0 / (1.0 + (-raw).exp());
            let expected = raw * sig;
            assert!((out[i] - expected).abs() < 1e-5,
                "silu mismatch at {}: {} vs {}", i, out[i], expected);
        }
        Ok(())
    }

    #[test]
    fn tanh_cpu_matches_reference() -> Result<()> {
        let mut x = Tensor::empty(&[5], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[-3.0, -1.0, 0.0, 1.0, 3.0]);

        tanh_inplace(&mut x)?;

        let out = x.as_f32()?.as_slice()?;
        for (i, &raw) in [-3.0f32, -1.0, 0.0, 1.0, 3.0].iter().enumerate() {
            let expected = raw.tanh();
            assert!((out[i] - expected).abs() < 1e-5,
                "tanh mismatch at {}: {} vs {}", i, out[i], expected);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn silu_cuda_matches_cpu() -> Result<()> {
        let mut cpu = Tensor::randn(&[512], DataType::F32, DeviceType::Cpu, Some(42))?;
        let mut gpu = cpu.to_cuda(0)?;
        silu_inplace(&mut cpu)?;
        silu_inplace(&mut gpu)?;
        let back = gpu.to_cpu()?;
        let (a, b) = (cpu.as_f32()?.as_slice()?, back.as_f32()?.as_slice()?);
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-5,
                "silu cpu/cuda mismatch at {}", i);
        }
        Ok(())
    }
}
