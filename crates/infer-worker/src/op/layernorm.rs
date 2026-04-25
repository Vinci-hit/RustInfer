use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// LayerNorm (无 affine 参数): output = (input - mean) / sqrt(var + eps)
/// input/output: [..., cols], 最后一维做归一化
pub fn layernorm(input: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    match input.device() {
        DeviceType::Cpu => kernels::cpu::layernorm(input, output, eps),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::layernorm(input, output, eps, None),
    }
}
