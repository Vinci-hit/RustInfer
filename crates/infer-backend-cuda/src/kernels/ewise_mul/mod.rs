//! Element-wise multiply CUDA kernel wrapper.
//!
//! Dispatch is an attribute of the element type: [`EwiseMulKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry point, so [`ewise_mul`] is generic with no runtime `match`. Adding a
//! dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn ewise_mul_f32_forward(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn ewise_mul_bf16_forward(
        dst: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        n: i32,
        stream: cudaStream_t,
    );
    fn ewise_mul_f16_forward(
        dst: *mut half::f16,
        a: *const half::f16,
        b: *const half::f16,
        n: i32,
        stream: cudaStream_t,
    );
}

/// Element types with an elementwise-multiply CUDA kernel. The method forwards
/// to this dtype's `extern` entry; the wrapper below is generic over this
/// trait, so the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `n` elements on
/// `stream`; this just names the FFI entry and performs no checks.
pub trait EwiseMulKernel: CudaFloat {
    /// `dst = a * b`, elementwise over `n` elements.
    unsafe fn ewise_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        stream: cudaStream_t,
    );
}

impl EwiseMulKernel for f32 {
    #[inline]
    unsafe fn ewise_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        stream: cudaStream_t,
    ) {
        unsafe { ewise_mul_f32_forward(dst, a, b, n, stream) }
    }
}

impl EwiseMulKernel for half::bf16 {
    #[inline]
    unsafe fn ewise_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        stream: cudaStream_t,
    ) {
        unsafe { ewise_mul_bf16_forward(dst, a, b, n, stream) }
    }
}

impl EwiseMulKernel for half::f16 {
    #[inline]
    unsafe fn ewise_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        stream: cudaStream_t,
    ) {
        unsafe { ewise_mul_f16_forward(dst, a, b, n, stream) }
    }
}

pub fn ewise_mul<T: EwiseMulKernel>(
    stream: cudaStream_t,
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = a.numel() as i32;
    unsafe { T::ewise_mul(dst.data_ptr_mut(), a.data_ptr(), b.data_ptr(), n, stream) }
    Ok(())
}
