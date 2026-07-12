//! RMSNorm CUDA kernel wrapper.
//!
//! Dispatch is an attribute of the element type: [`RmsNormKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry point, so [`rmsnorm`]/[`rmsnorm_inplace`] are generic with no runtime
//! `match`. Adding a dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::Dtype;

// ─── C kernel declarations ───────────────────────────────────────────────────

unsafe extern "C" {
    fn rmsnorm_kernel_cu_dim(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        outer0: i32,
        outer1: i32,
        dim: i32,
        in_stride0: i64,
        in_stride1: i64,
        out_stride0: i64,
        out_stride1: i64,
        eps: f32,
        stream: cudaStream_t,
    );
    fn rmsnorm_kernel_cu_bf16x8(
        output: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        outer0: i32,
        outer1: i32,
        dim: i32,
        in_stride0: i64,
        in_stride1: i64,
        out_stride0: i64,
        out_stride1: i64,
        eps: f32,
        stream: cudaStream_t,
    );
    fn rmsnorm_kernel_cu_fp16x8(
        output: *mut half::f16,
        input: *const half::f16,
        weight: *const half::f16,
        outer0: i32,
        outer1: i32,
        dim: i32,
        in_stride0: i64,
        in_stride1: i64,
        out_stride0: i64,
        out_stride1: i64,
        eps: f32,
        stream: cudaStream_t,
    );
}

// ─── Binding trait ───────────────────────────────────────────────────────────

/// Element types with an RMSNorm CUDA kernel. The method forwards to this
/// dtype's `extern` entry; the wrappers below are generic over this trait, so
/// the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers matching the layouts on
/// `stream`; this just names the FFI entry and performs no checks.
pub trait RmsNormKernel: CudaFloat {
    /// `out = rmsnorm(inp, w, eps)` over the given input/output layouts.
    unsafe fn rmsnorm(
        out: *mut Self,
        inp: *const Self,
        w: *const Self,
        il: Layout,
        ol: Layout,
        eps: f32,
        stream: cudaStream_t,
    );
}

impl RmsNormKernel for f32 {
    #[inline]
    unsafe fn rmsnorm(
        out: *mut Self,
        inp: *const Self,
        w: *const Self,
        il: Layout,
        ol: Layout,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            rmsnorm_kernel_cu_dim(
                out, inp, w, il.outer0, il.outer1, il.dim, il.stride0, il.stride1, ol.stride0,
                ol.stride1, eps, stream,
            )
        }
    }
}

impl RmsNormKernel for half::bf16 {
    #[inline]
    unsafe fn rmsnorm(
        out: *mut Self,
        inp: *const Self,
        w: *const Self,
        il: Layout,
        ol: Layout,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            rmsnorm_kernel_cu_bf16x8(
                out, inp, w, il.outer0, il.outer1, il.dim, il.stride0, il.stride1, ol.stride0,
                ol.stride1, eps, stream,
            )
        }
    }
}

impl RmsNormKernel for half::f16 {
    #[inline]
    unsafe fn rmsnorm(
        out: *mut Self,
        inp: *const Self,
        w: *const Self,
        il: Layout,
        ol: Layout,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            rmsnorm_kernel_cu_fp16x8(
                out, inp, w, il.outer0, il.outer1, il.dim, il.stride0, il.stride1, ol.stride0,
                ol.stride1, eps, stream,
            )
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// `output = rmsnorm(input, weight, eps)` on CUDA.
pub fn rmsnorm<T: RmsNormKernel>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = weight.numel();
    let in_layout = derive_layout(input, dim)?;
    let out_layout = derive_layout(output, dim)?;

    unsafe {
        T::rmsnorm(
            output.data_ptr_mut(),
            input.data_ptr(),
            weight.data_ptr(),
            in_layout,
            out_layout,
            eps,
            stream,
        )
    }
    Ok(())
}

/// In-place: `x = rmsnorm(x, weight, eps)`.
pub fn rmsnorm_inplace<T: RmsNormKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = weight.numel();
    let layout = derive_layout(x, dim)?;
    let ptr = x.data_ptr_mut();

    unsafe {
        T::rmsnorm(
            ptr,
            ptr as *const _,
            weight.data_ptr(),
            layout,
            layout,
            eps,
            stream,
        )
    }
    Ok(())
}

// ─── Internal ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Layout {
    outer0: i32,
    outer1: i32,
    dim: i32,
    stride0: i64,
    stride1: i64,
}

fn derive_layout<T: Dtype, D: infer_core::ports::MemoryPort>(
    t: &Tensor<T, D>,
    dim: usize,
) -> OpResult<Layout> {
    let shape = t.shape().as_slice();
    let strides = t.strides().as_slice();
    if shape.is_empty() || *shape.last().unwrap() != dim || *strides.last().unwrap() != 1 {
        return Err(OpError::Shape(format!(
            "rmsnorm: bad layout shape={:?} strides={:?} dim={}",
            shape, strides, dim
        )));
    }
    if t.is_contiguous() {
        return Ok(Layout {
            outer0: (t.numel() / dim) as i32,
            outer1: 1,
            dim: dim as i32,
            stride0: dim as i64,
            stride1: 0,
        });
    }
    match t.ndim() {
        2 => Ok(Layout {
            outer0: shape[0] as i32,
            outer1: 1,
            dim: dim as i32,
            stride0: strides[0] as i64,
            stride1: 0,
        }),
        3 => Ok(Layout {
            outer0: shape[0] as i32,
            outer1: shape[1] as i32,
            dim: dim as i32,
            stride0: strides[0] as i64,
            stride1: strides[1] as i64,
        }),
        _ => Err(OpError::Shape(
            "rmsnorm: unsupported rank for strided input".into(),
        )),
    }
}
