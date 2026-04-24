use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

use super::kernels;

/// 从 src 的指定列范围拷贝到 dst
#[allow(unused_variables)]
pub fn split_cols_tensor(
    src: &Tensor,
    dst: &mut Tensor,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
    #[cfg(feature = "cuda")] cuda_stream: crate::cuda::ffi::cudaStream_t,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => kernels::cpu::split_cols_tensor(src, dst, rows, total_cols, col_offset, dst_cols),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            match src.dtype() {
                DataType::BF16 => kernels::cuda::split_cols_bf16_tensor(src, dst, rows, total_cols, col_offset, dst_cols, cuda_stream),
                DataType::F16 => kernels::cuda::split_cols_fp16_tensor(src, dst, rows, total_cols, col_offset, dst_cols, cuda_stream),
                other => Err(Error::InvalidArgument(format!(
                    "CUDA split_cols supports BF16/F16 only, got {:?}", other
                )).into()),
            }
        }
    }
}
