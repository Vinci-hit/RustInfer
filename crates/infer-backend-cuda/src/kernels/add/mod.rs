//! Add CUDA kernel wrapper.
//!
//! Dispatch is an attribute of the element type: [`AddKernel`] is implemented
//! once per supported dtype and names that dtype's `extern "C"` entry point, so
//! [`add`]/[`add_inplace`] are generic with no runtime `match`. Adding a dtype
//! is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn add_kernel_bf16x8(
        c: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
    fn add_kernel_fp16x8(
        c: *mut half::f16,
        a: *const half::f16,
        b: *const half::f16,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
    fn add_inplace_kernel_bf16x8(
        a: *mut half::bf16,
        b: *const half::bf16,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
    fn add_inplace_kernel_fp16x8(
        a: *mut half::f16,
        b: *const half::f16,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
    fn add_kernel_float2_forward(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
    fn add_inplace_kernel_float2_forward(
        a: *mut f32,
        b: *const f32,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
}

/// Element types with an elementwise-add CUDA kernel. The two methods each
/// forward to this dtype's `extern` entry; the wrapper below is generic over
/// this trait, so the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `n` elements on
/// `stream`; this just names the FFI entry and performs no checks.
pub trait AddKernel: CudaFloat {
    /// `c = a + b`, elementwise over `n` elements.
    unsafe fn add(
        c: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    );
    /// `a += b`, elementwise over `n` elements.
    unsafe fn add_inplace(a: *mut Self, b: *const Self, n: i32, num_sm: i32, stream: cudaStream_t);
}

impl AddKernel for f32 {
    #[inline]
    unsafe fn add(
        c: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    ) {
        unsafe { add_kernel_float2_forward(c, a, b, n, num_sm, stream) }
    }
    #[inline]
    unsafe fn add_inplace(a: *mut Self, b: *const Self, n: i32, num_sm: i32, stream: cudaStream_t) {
        unsafe { add_inplace_kernel_float2_forward(a, b, n, num_sm, stream) }
    }
}

impl AddKernel for half::bf16 {
    #[inline]
    unsafe fn add(
        c: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    ) {
        unsafe { add_kernel_bf16x8(c, a, b, n, num_sm, stream) }
    }
    #[inline]
    unsafe fn add_inplace(a: *mut Self, b: *const Self, n: i32, num_sm: i32, stream: cudaStream_t) {
        unsafe { add_inplace_kernel_bf16x8(a, b, n, num_sm, stream) }
    }
}

impl AddKernel for half::f16 {
    #[inline]
    unsafe fn add(
        c: *mut Self,
        a: *const Self,
        b: *const Self,
        n: i32,
        num_sm: i32,
        stream: cudaStream_t,
    ) {
        unsafe { add_kernel_fp16x8(c, a, b, n, num_sm, stream) }
    }
    #[inline]
    unsafe fn add_inplace(a: *mut Self, b: *const Self, n: i32, num_sm: i32, stream: cudaStream_t) {
        unsafe { add_inplace_kernel_fp16x8(a, b, n, num_sm, stream) }
    }
}

pub fn add<T: AddKernel>(
    stream: cudaStream_t,
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = a.numel() as i32;
    let num_sm = a.device().config.device_info().sm_count;
    unsafe {
        T::add(
            dst.data_ptr_mut(),
            a.data_ptr(),
            b.data_ptr(),
            n,
            num_sm,
            stream,
        )
    }
    Ok(())
}

pub fn add_inplace<T: AddKernel>(
    stream: cudaStream_t,
    dst: &mut Tensor<T, Cuda>,
    src: &Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = dst.numel() as i32;
    let num_sm = dst.device().config.device_info().sm_count;
    unsafe { T::add_inplace(dst.data_ptr_mut(), src.data_ptr(), n, num_sm, stream) }
    Ok(())
}
