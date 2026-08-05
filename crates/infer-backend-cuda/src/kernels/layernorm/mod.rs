//! LayerNorm CUDA kernel wrapper.
//! The .cu kernel computes (x - mean) / std. We apply weight*x + bias via
//! broadcast_mul + a simple bias-add loop on top.
//!
//! Dispatch is an attribute of the element type: [`LayerNormKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry points, so [`layernorm`] is generic with no runtime `match`. Adding a
//! dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn layernorm_f32_forward(
        output: *mut f32,
        input: *const f32,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn layernorm_bf16_forward(
        output: *mut half::bf16,
        input: *const half::bf16,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn layernorm_f16_forward(
        output: *mut half::f16,
        input: *const half::f16,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

// Reuse broadcast_mul from its module
unsafe extern "C" {
    fn broadcast_mul_f32_forward(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_mul_bf16_forward(
        dst: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_mul_f16_forward(
        dst: *mut half::f16,
        a: *const half::f16,
        b: *const half::f16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    // broadcast_add (reuse the add kernel from broadcast_mul.cu)
    fn broadcast_add_inplace_f32_forward(
        a: *mut f32,
        b: *const f32,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_bf16_forward(
        a: *mut half::bf16,
        b: *const half::bf16,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_f16_forward(
        a: *mut half::f16,
        b: *const half::f16,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    );
}

/// Element types with a LayerNorm CUDA kernel. The three methods forward to
/// this dtype's `extern` entries (normalize, broadcast-multiply, broadcast-add);
/// the wrapper below is generic over this trait, so the dtype→kernel mapping
/// lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `rows * cols`
/// elements on `stream`; this just names the FFI entries and performs no checks.
pub trait LayerNormKernel: CudaFloat {
    /// Normalize (zero-mean, unit-variance) rows of `input` into `output`.
    unsafe fn layernorm(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    /// `dst = a * b` with `b` broadcast per row.
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    /// `a += b` with `b` broadcast per row.
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    );
}

impl LayerNormKernel for f32 {
    #[inline]
    unsafe fn layernorm(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe { layernorm_f32_forward(output, input, rows, cols, eps, stream) }
    }
    #[inline]
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_mul_f32_forward(dst, a, b, rows, d, stream) }
    }
    #[inline]
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_add_inplace_f32_forward(a, b, rows, d, row_stride, stream) }
    }
}

impl LayerNormKernel for half::bf16 {
    #[inline]
    unsafe fn layernorm(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe { layernorm_bf16_forward(output, input, rows, cols, eps, stream) }
    }
    #[inline]
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_mul_bf16_forward(dst, a, b, rows, d, stream) }
    }
    #[inline]
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_add_inplace_bf16_forward(a, b, rows, d, row_stride, stream) }
    }
}

impl LayerNormKernel for half::f16 {
    #[inline]
    unsafe fn layernorm(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe { layernorm_f16_forward(output, input, rows, cols, eps, stream) }
    }
    #[inline]
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_mul_f16_forward(dst, a, b, rows, d, stream) }
    }
    #[inline]
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        row_stride: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_add_inplace_f16_forward(a, b, rows, d, row_stride, stream) }
    }
}

/// Full layernorm: normalize → scale by weight → add bias.
pub fn layernorm<T: LayerNormKernel>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    bias: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = *input.shape().as_slice().last().unwrap();
    let rows = (input.numel() / dim) as i32;
    let cols = dim as i32;

    unsafe {
        // Step 1: normalize (zero-mean, unit-variance)
        T::layernorm(
            output.data_ptr_mut(),
            input.data_ptr(),
            rows,
            cols,
            eps,
            stream,
        );

        // Step 2: output *= weight (broadcast multiply)
        T::broadcast_mul(
            output.data_ptr_mut(),
            output.data_ptr(),
            weight.data_ptr(),
            rows,
            cols,
            stream,
        );

        // Step 3: output += bias (broadcast add)
        T::broadcast_add_inplace(
            output.data_ptr_mut(),
            bias.data_ptr(),
            rows,
            cols,
            cols,
            stream,
        );
    }
    Ok(())
}
