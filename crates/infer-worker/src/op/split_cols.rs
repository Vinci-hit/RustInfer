use crate::OpConfig;
use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

use super::kernels;

/// 从 `src` 的列范围 `[col_offset, col_offset+dst_cols)` 拷贝到 `dst`。
///
/// CUDA stream 由本函数内部从 `cuda_config` 解析（`cuda_config=None` 时走
/// thread-local 的 current stream）。调用方只需照常传 `Option<&OpConfig>`，
/// 不需要自己 `resolve_stream`。
#[allow(unused_variables)]
pub fn split_cols_tensor(
    src: &Tensor,
    dst: &mut Tensor,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => kernels::cpu::split_cols_tensor(src, dst, rows, total_cols, col_offset, dst_cols),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::CudaConfig::resolve_stream(cuda_config);
            match src.dtype() {
                DataType::BF16 => kernels::cuda::split_cols_bf16_tensor(src, dst, rows, total_cols, col_offset, dst_cols, stream),
                DataType::F16 => kernels::cuda::split_cols_fp16_tensor(src, dst, rows, total_cols, col_offset, dst_cols, stream),
                DataType::F32 => kernels::cuda::split_cols_f32_tensor(src, dst, rows, total_cols, col_offset, dst_cols, stream),
                other => Err(Error::InvalidArgument(format!(
                    "CUDA split_cols supports BF16/F16/F32 only, got {:?}", other
                )).into()),
            }
        }
    }
}
