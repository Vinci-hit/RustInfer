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
        DeviceType::Cuda(_) => kernels::cuda::scalar_mul(
            src, dst, val, crate::cuda::get_current_cuda_stream(),
        ),
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
        DeviceType::Cuda(_) => kernels::cuda::scalar_add(
            src, dst, val, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

/// 原地 SiLU: x[i] = x[i] * sigmoid(x[i])
pub fn silu_inplace(x: &mut Tensor) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => kernels::cpu::silu_inplace(x),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::silu_inplace(
            x, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

/// 原地 tanh: x[i] = tanh(x[i])
pub fn tanh_inplace(x: &mut Tensor) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => kernels::cpu::tanh_inplace(x),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::tanh_inplace(
            x, crate::cuda::get_current_cuda_stream(),
        ),
    }
}
