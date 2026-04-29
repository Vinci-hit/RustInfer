use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// Upsample nearest 2x: input[B,C,H,W] → output[B,C,2H,2W]
pub fn upsample_nearest_2x(input: &Tensor, output: &mut Tensor) -> Result<()> {
    match input.device() {
        DeviceType::Cpu => kernels::cpu::upsample_nearest_2x(input, output),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            kernels::cuda::upsample_nearest_2x(input, output, stream)
        }
    }
}
