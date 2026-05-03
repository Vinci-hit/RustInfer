//! Element-wise add: `dst = a + b` and in-place `a += b`.
//!
//! Free-function API — previous versions shipped zero-sized wrapper
//! structs (`Add::new().forward(...)`), which added nothing over a direct
//! call. Callers should use [`add`] / [`add_inplace`].
//!
//! Device/dtype dispatch happens in the per-device kernels; both functions
//! are just thin wrappers picking CPU vs CUDA.

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::OpConfig;
use crate::tensor::Tensor;

use super::kernels;

/// `dst = a + b` (element-wise). Shapes must match.
pub fn add(
    a: &Tensor,
    b: &Tensor,
    dst: &mut Tensor,
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    match a.device() {
        DeviceType::Cpu => {
            let _ = cuda_config;
            kernels::cpu::add(a, b, dst)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::add(a, b, dst, cuda_config),
    }
}

/// `dst += src` (element-wise, in place). Shapes must match.
pub fn add_inplace(
    src: &Tensor,
    dst: &mut Tensor,
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => {
            let _ = cuda_config;
            kernels::cpu::add_inplace(dst, src)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::add_inplace(dst, src, cuda_config),
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
        assert_eq!(a.len(), b.len(), "Slices have different lengths");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol,
                "Mismatch at index {}: a = {}, b = {}", i, x, y);
        }
    }

    fn assert_bf16_close(a: &[bf16], b: &[bf16], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x.to_f32() - y.to_f32()).abs();
            assert!(diff < tol,
                "BF16 mismatch at {}: {} vs {}, diff={}",
                i, x.to_f32(), y.to_f32(), diff);
        }
    }

    #[test]
    fn add_cpu_f32() -> Result<()> {
        let mut a = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
        let mut out = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        add(&a, &b, &mut out, None)?;
        assert_eq!(out.as_f32()?.as_slice()?, &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn add_inplace_cpu_f32() -> Result<()> {
        let mut a = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        let mut dst = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        a  .as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0]);
        dst.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[100.0, 200.0, 300.0]);
        add_inplace(&a, &mut dst, None)?;
        assert_eq!(dst.as_f32()?.as_slice()?, &[101.0, 202.0, 303.0]);
        Ok(())
    }

    #[test]
    fn add_cpu_bf16_batched() -> Result<()> {
        for batch in [1, 2, 4, 8] {
            let (seq, dim) = (32, 64);
            let shape = &[batch, seq, dim];
            let n = batch * seq * dim;

            let mut a = Tensor::empty(shape, DataType::BF16, DeviceType::Cpu)?;
            let mut b = Tensor::empty(shape, DataType::BF16, DeviceType::Cpu)?;
            let mut out = Tensor::empty(shape, DataType::BF16, DeviceType::Cpu)?;

            let ad: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.01)).collect();
            let bd: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.02)).collect();
            a.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&ad);
            b.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&bd);

            add(&a, &b, &mut out, None)?;

            let expected: Vec<bf16> = ad.iter().zip(bd.iter())
                .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32())).collect();
            assert_bf16_close(out.as_bf16()?.as_slice()?, &expected, 1e-2);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn add_cuda_f32_matches_reference() -> Result<()> {
        let size = 32 * 151;
        let mut a = Tensor::empty(&[size], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[size], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.fill(2.0);
        b.as_f32_mut()?.as_slice_mut()?.fill(3.0);

        let a_g = a.to_cuda(0)?;
        let b_g = b.to_cuda(0)?;
        let mut out_g = Tensor::empty(&[size], DataType::F32, DeviceType::Cuda(0))?;
        add(&a_g, &b_g, &mut out_g, None)?;
        let out = out_g.to_cpu()?;
        assert_close(out.as_f32()?.as_slice()?, &vec![5.0_f32; size], 1e-6);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn add_inplace_cuda_bf16_batched() -> Result<()> {
        for batch in [1, 2, 4, 8] {
            let (seq, dim) = (16, 256);
            let shape = &[batch, seq, dim];
            let n = batch * seq * dim;

            let ad: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.005)).collect();
            let od: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.003)).collect();

            let mut a = Tensor::empty(shape, DataType::BF16, DeviceType::Cuda(0))?;
            let mut dst = Tensor::empty(shape, DataType::BF16, DeviceType::Cuda(0))?;
            a.as_bf16_mut()?.buffer_mut().copy_from_host(&ad)?;
            dst.as_bf16_mut()?.buffer_mut().copy_from_host(&od)?;

            let cfg = crate::cuda::CudaConfig::new()?;
            add_inplace(&a, &mut dst, Some(&cfg))?;

            let expected: Vec<bf16> = ad.iter().zip(od.iter())
                .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32())).collect();
            let back = dst.to_cpu()?;
            assert_bf16_close(back.as_bf16()?.as_slice()?, &expected, 1e-2);
        }
        Ok(())
    }
}
