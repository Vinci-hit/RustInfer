use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::config::CudaConfig;

use super::kernels;

/// 融合算子: residual += input; norm_output = rmsnorm(residual, weight, eps)
#[allow(unused_variables)]
pub fn fused_add_rmsnorm(
    norm_output: &mut Tensor,
    residual: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    #[cfg(feature = "cuda")] cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    match residual.device() {
        DeviceType::Cpu => kernels::cpu::fused_add_rmsnorm(norm_output, residual, input, weight, eps),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::fused_add_rmsnorm(
            norm_output, residual, input, weight, eps, cuda_config,
        ),
    }
}
