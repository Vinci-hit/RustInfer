//! Fused add + RMSNorm CUDA kernel.
//! residual += input; output = rmsnorm(residual, weight, eps)
//!
//! Dispatch is an attribute of the element type: [`FusedAddRmsNormKernel`] is
//! implemented once per supported dtype. BF16/F16 use fused entry points;
//! FP32 composes existing add and RMSNorm operators.
//! Adding a dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn fused_add_rmsnorm_kernel_cu_bf16(
        output: *mut half::bf16,
        residual: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn fused_add_rmsnorm_kernel_cu_fp16(
        output: *mut half::f16,
        residual: *mut half::f16,
        input: *const half::f16,
        weight: *const half::f16,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

/// Float dispatch for residual addition followed by RMSNorm. The FP32 path
/// composes existing operators; no FP32 fused CUDA entry is compiled.
pub trait FusedAddRmsNormKernel: CudaFloat {
    fn forward(
        stream: cudaStream_t,
        output: &mut Tensor<Self, Cuda>,
        residual: &mut Tensor<Self, Cuda>,
        input: &Tensor<Self, Cuda>,
        weight: &Tensor<Self, Cuda>,
        eps: f32,
    ) -> OpResult<()>;
}

impl FusedAddRmsNormKernel for f32 {
    fn forward(
        stream: cudaStream_t,
        output: &mut Tensor<Self, Cuda>,
        residual: &mut Tensor<Self, Cuda>,
        input: &Tensor<Self, Cuda>,
        weight: &Tensor<Self, Cuda>,
        eps: f32,
    ) -> OpResult<()> {
        super::add::add_inplace(stream, residual, input)?;
        super::rmsnorm::rmsnorm(stream, residual, weight, output, eps)
    }
}

macro_rules! fused {
    ($ty:ty, $entry:ident) => {
        impl FusedAddRmsNormKernel for $ty {
            fn forward(
                stream: cudaStream_t,
                output: &mut Tensor<Self, Cuda>,
                residual: &mut Tensor<Self, Cuda>,
                input: &Tensor<Self, Cuda>,
                weight: &Tensor<Self, Cuda>,
                eps: f32,
            ) -> OpResult<()> {
                unsafe {
                    $entry(
                        output.data_ptr_mut(),
                        residual.data_ptr_mut(),
                        input.data_ptr(),
                        weight.data_ptr(),
                        (input.numel() / weight.numel()) as i32,
                        weight.numel() as i32,
                        eps,
                        stream,
                    );
                }
                Ok(())
            }
        }
    };
}
fused!(half::bf16, fused_add_rmsnorm_kernel_cu_bf16);
fused!(half::f16, fused_add_rmsnorm_kernel_cu_fp16);

/// Fused: residual += input; output = rmsnorm(residual, weight, eps).
pub fn fused_add_rmsnorm<T: FusedAddRmsNormKernel>(
    stream: cudaStream_t,
    output: &mut Tensor<T, Cuda>,
    residual: &mut Tensor<T, Cuda>,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    T::forward(stream, output, residual, input, weight, eps)
}
