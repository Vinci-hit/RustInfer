//! Fused add + RMSNorm CUDA kernel.
//! residual += input; output = rmsnorm(residual, weight, eps)
//!
//! Dispatch is an attribute of the element type: [`FusedAddRmsNormKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry point, so [`fused_add_rmsnorm`] is generic with no runtime `match`.
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
    fn fused_add_rmsnorm_kernel_cu_fp32(
        output: *mut f32,
        residual: *mut f32,
        input: *const f32,
        weight: *const f32,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

/// Element types with a fused add+RMSNorm CUDA kernel. The method forwards to
/// this dtype's `extern` entry; the wrapper below is generic over this trait, so
/// the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `rows * dim`
/// elements on `stream`; this just names the FFI entry and performs no checks.
pub trait FusedAddRmsNormKernel: CudaFloat {
    /// `residual += input; output = rmsnorm(residual, weight, eps)`.
    unsafe fn fused_add_rmsnorm(
        output: *mut Self,
        residual: *mut Self,
        input: *const Self,
        weight: *const Self,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

impl FusedAddRmsNormKernel for f32 {
    #[inline]
    unsafe fn fused_add_rmsnorm(
        output: *mut Self,
        residual: *mut Self,
        input: *const Self,
        weight: *const Self,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            fused_add_rmsnorm_kernel_cu_fp32(
                output, residual, input, weight, rows, dim, eps, stream,
            )
        }
    }
}

impl FusedAddRmsNormKernel for half::bf16 {
    #[inline]
    unsafe fn fused_add_rmsnorm(
        output: *mut Self,
        residual: *mut Self,
        input: *const Self,
        weight: *const Self,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            fused_add_rmsnorm_kernel_cu_bf16(
                output, residual, input, weight, rows, dim, eps, stream,
            )
        }
    }
}

impl FusedAddRmsNormKernel for half::f16 {
    #[inline]
    unsafe fn fused_add_rmsnorm(
        output: *mut Self,
        residual: *mut Self,
        input: *const Self,
        weight: *const Self,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            fused_add_rmsnorm_kernel_cu_fp16(
                output, residual, input, weight, rows, dim, eps, stream,
            )
        }
    }
}

/// Fused: residual += input; output = rmsnorm(residual, weight, eps)
pub fn fused_add_rmsnorm<T: FusedAddRmsNormKernel>(
    stream: cudaStream_t,
    output: &mut Tensor<T, Cuda>,
    residual: &mut Tensor<T, Cuda>,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = weight.numel();
    let rows = (input.numel() / dim) as i32;
    unsafe {
        T::fused_add_rmsnorm(
            output.data_ptr_mut(),
            residual.data_ptr_mut(),
            input.data_ptr(),
            weight.data_ptr(),
            rows,
            dim as i32,
            eps,
            stream,
        );
    }
    Ok(())
}
