use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// Softmax 沿最后一维: output[..., j] = exp(input[..., j] - max) / sum(exp)
pub fn softmax(input: &Tensor, output: &mut Tensor) -> Result<()> {
    match input.device() {
        DeviceType::Cpu => kernels::cpu::softmax(input, output),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            kernels::cuda::softmax(input, output, stream)
        }
    }
}
