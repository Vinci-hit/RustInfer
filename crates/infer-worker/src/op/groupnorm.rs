use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// GroupNorm: input[B,C,...] → output[B,C,...], weight[C], bias[C]
#[allow(clippy::too_many_arguments)]
pub fn groupnorm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &mut Tensor,
    num_groups: usize,
    eps: f32,
) -> Result<()> {
    match input.device() {
        DeviceType::Cpu => kernels::cpu::groupnorm(input, weight, bias, output, num_groups, eps),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            kernels::cuda::groupnorm(input, weight, bias, output, num_groups, eps, stream)
        }
    }
}
