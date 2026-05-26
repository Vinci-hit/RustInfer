//! Fused add + RMSNorm CUDA kernel.
//! residual += input; output = rmsnorm(residual, weight, eps)

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn fused_add_rmsnorm_kernel_cu_bf16(output: *mut half::bf16, residual: *mut half::bf16, input: *const half::bf16, weight: *const half::bf16, rows: i32, dim: i32, eps: f32, stream: cudaStream_t);
    fn fused_add_rmsnorm_kernel_cu_fp16(output: *mut half::f16, residual: *mut half::f16, input: *const half::f16, weight: *const half::f16, rows: i32, dim: i32, eps: f32, stream: cudaStream_t);
    fn fused_add_rmsnorm_kernel_cu_fp32(output: *mut f32, residual: *mut f32, input: *const f32, weight: *const f32, rows: i32, dim: i32, eps: f32, stream: cudaStream_t);
}

/// Fused: residual += input; output = rmsnorm(residual, weight, eps)
pub fn fused_add_rmsnorm<T: Dtype>(
    output: &mut Tensor<T, Cuda>,
    residual: &mut Tensor<T, Cuda>,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = weight.numel();
    let rows = (input.numel() / dim) as i32;
    let stream = input.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => fused_add_rmsnorm_kernel_cu_fp32(output.data_ptr_mut() as _, residual.data_ptr_mut() as _, input.data_ptr() as _, weight.data_ptr() as _, rows, dim as i32, eps, stream),
            DataType::BF16 => fused_add_rmsnorm_kernel_cu_bf16(output.data_ptr_mut() as _, residual.data_ptr_mut() as _, input.data_ptr() as _, weight.data_ptr() as _, rows, dim as i32, eps, stream),
            DataType::F16 => fused_add_rmsnorm_kernel_cu_fp16(output.data_ptr_mut() as _, residual.data_ptr_mut() as _, input.data_ptr() as _, weight.data_ptr() as _, rows, dim as i32, eps, stream),
            _ => return Err(OpError::Kernel(format!("fused_add_rmsnorm: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
