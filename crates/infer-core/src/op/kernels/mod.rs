pub mod cpu;
pub mod cuda;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::config::CudaConfig;

/// Unified fused_add_rmsnorm entry.
/// Dispatches by tensor device and keeps cfg-gating out of model forward paths.
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
        DeviceType::Cpu => cpu::fused_add_rmsnorm(norm_output, residual, input, weight, eps),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => cuda::fused_add_rmsnorm(
            norm_output,
            residual,
            input,
            weight,
            eps,
            cuda_config,
        ),
    }
}

/// Unified fused scatter_kv entry.
#[allow(unused_variables)]
pub fn scatter_kv(
    dst_k: &mut Tensor,
    src_k: &Tensor,
    dst_v: &mut Tensor,
    src_v: &Tensor,
    pos: &Tensor,
    #[cfg(feature = "cuda")] cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    match dst_k.device() {
        DeviceType::Cpu => cpu::scatter_kv(dst_k, src_k, dst_v, src_v, pos),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => cuda::scatter_kv(dst_k, src_k, dst_v, src_v, pos, cuda_config),
    }
}

/// Unified split-cols entry.
/// Keeps a single callsite signature for CPU/CUDA paths.
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
        DeviceType::Cpu => cpu::split_cols_tensor(src, dst, rows, total_cols, col_offset, dst_cols),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            if src.dtype() != DataType::BF16 {
                return Err(Error::InvalidArgument(format!(
                    "CUDA split_cols currently supports BF16 only, got {:?}",
                    src.dtype()
                )).into());
            }
            cuda::split_cols_bf16_tensor(
                src,
                dst,
                rows,
                total_cols,
                col_offset,
                dst_cols,
                cuda_stream,
            )
        }
    }
}
