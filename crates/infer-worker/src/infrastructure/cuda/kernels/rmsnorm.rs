//! RMSNorm CUDA kernel wrapper — generic over T: Float.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

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

// ─── Public API ──────────────────────────────────────────────────────────────

/// `output = rmsnorm(input, weight, eps)` on CUDA.
pub fn rmsnorm<T: Dtype>(
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
        dispatch::<T>(
            output.data_ptr_mut() as *mut _,
            input.data_ptr() as *const _,
            weight.data_ptr() as *const _,
            in_layout,
            out_layout,
            eps,
            stream,
        )
    }
}

/// In-place: `x = rmsnorm(x, weight, eps)`.
pub fn rmsnorm_inplace<T: Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = weight.numel();
    let layout = derive_layout(x, dim)?;
    let ptr = x.data_ptr_mut();

    unsafe {
        dispatch::<T>(
            ptr as *mut _,
            ptr as *const _,
            weight.data_ptr() as *const _,
            layout,
            layout,
            eps,
            stream,
        )
    }
}

// ─── Internal ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Layout {
    outer0: i32,
    outer1: i32,
    dim: i32,
    stride0: i64,
    stride1: i64,
}

fn derive_layout<T: Dtype, D: crate::domain::ports::MemoryPort>(
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

unsafe fn dispatch<T: Dtype>(
    out: *mut std::ffi::c_void,
    inp: *const std::ffi::c_void,
    w: *const std::ffi::c_void,
    il: Layout,
    ol: Layout,
    eps: f32,
    stream: cudaStream_t,
) -> OpResult<()> {
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => rmsnorm_kernel_cu_dim(
                out as _, inp as _, w as _, il.outer0, il.outer1, il.dim, il.stride0, il.stride1,
                ol.stride0, ol.stride1, eps, stream,
            ),
            DataType::BF16 => rmsnorm_kernel_cu_bf16x8(
                out as _, inp as _, w as _, il.outer0, il.outer1, il.dim, il.stride0, il.stride1,
                ol.stride0, ol.stride1, eps, stream,
            ),
            DataType::F16 => rmsnorm_kernel_cu_fp16x8(
                out as _, inp as _, w as _, il.outer0, il.outer1, il.dim, il.stride0, il.stride1,
                ol.stride0, ol.stride1, eps, stream,
            ),
            _ => {
                return Err(OpError::Kernel(format!(
                    "rmsnorm: unsupported dtype {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}
