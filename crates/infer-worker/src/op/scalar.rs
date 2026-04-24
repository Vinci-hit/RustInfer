use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// dst[i] = src[i] * val
pub fn scalar_mul(
    src: &Tensor,
    dst: &mut Tensor,
    val: f32,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => kernels::cpu::scalar_mul(src, dst, val),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::scalar_mul(src, dst, val, std::ptr::null_mut()),
    }
}

/// dst[i] = src[i] + val
pub fn scalar_add(
    src: &Tensor,
    dst: &mut Tensor,
    val: f32,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => kernels::cpu::scalar_add(src, dst, val),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::scalar_add(src, dst, val, std::ptr::null_mut()),
    }
}
