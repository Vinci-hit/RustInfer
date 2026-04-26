use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// Conv2d 前向: input[B,Cin,H,W] * weight[Cout,Cin,kH,kW] + bias[Cout] → output[B,Cout,Hout,Wout]
///
/// - CPU: im2col + GEMM
/// - CUDA: cuDNN cudnnConvolutionForward (自动选择最优算法)
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    output: &mut Tensor,
    stride: usize,
    padding: usize,
    cuda_config: Option<&crate::OpConfig>,
) -> Result<()> {
    match input.device() {
        DeviceType::Cpu => { let _ = cuda_config; kernels::cpu::conv2d(input, weight, bias, output, stride, padding) }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let cfg = cuda_config.ok_or_else(|| crate::base::error::Error::InvalidArgument(
                "conv2d CUDA path requires CudaConfig with cudnn_handle".into()
            ))?;
            kernels::cuda::conv2d_cudnn(input, weight, bias, output, stride, padding, cfg)
        }
    }
}

/// 计算 Conv2d 输出尺寸
pub fn conv2d_output_size(h_in: usize, w_in: usize, kh: usize, kw: usize, stride: usize, padding: usize) -> (usize, usize) {
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    (h_out, w_out)
}
