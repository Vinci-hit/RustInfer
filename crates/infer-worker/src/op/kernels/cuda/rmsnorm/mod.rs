//! RMSNorm CUDA kernel 的 Rust 包装层。
//!
//! 数据视图模型：[outer0, outer1, dim] 三维 strided（最后一维 dense）。
//! 调用方喂 `&Tensor`，本层从 `(shape, strides, weight.shape[0])` 推导这五元组：
//!
//! - **dense（任意 rank）**：`is_contiguous() && shape.last() == dim`
//!     → outer0 = numel/dim, outer1 = 1, stride0 = dim, stride1 = 0
//!     覆盖 1-D / 2-D / 3-D dense 等所有传统用法。
//! - **2-D strided（最后一维 dense，shape[1] == dim）**：
//!     → outer0 = shape[0], outer1 = 1, stride0 = strides[0], stride1 = 0
//!     覆盖 `qkv.narrow(0, ..., total_T)` 这种行前缀视图。
//! - **3-D strided 按 head 切（shape == [outer0, outer1, dim]，最后一维 dense）**：
//!     → outer0 / outer1 / stride0 / stride1 直接来自 view
//!     覆盖 `qkv.narrow(1, 0, q_dim).reshape([T, head_num, head_dim])` 这种 QK-norm 用法。
//!
//! `forward` 与 `forward_inplace` 共享一条 dispatch；in-place 即 output==input。
//!
//! 不准 fallback：不满足约束（last-dim 非 dense、stride 不对齐 8 等）直接报
//! `InvalidArgument`。

use crate::base::error::{Error, Result};
use crate::base::DataType;
use crate::cuda::config::CudaConfig;
use crate::tensor::Tensor;

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
        stream: crate::cuda::ffi::cudaStream_t,
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
        stream: crate::cuda::ffi::cudaStream_t,
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
        stream: crate::cuda::ffi::cudaStream_t,
    );
}

/// 一个 RMSNorm 行视图：`outer0 * outer1` 行，每行 `dim` 个 dense 元素，
/// 行 r 的物理 element 偏移 = `(r/outer1)*stride0 + (r%outer1)*stride1`。
#[derive(Clone, Copy)]
struct Layout {
    outer0: i32,
    outer1: i32,
    dim: i32,
    stride0: i64,
    stride1: i64,
}

/// 推导 `Tensor` 的 RMSNorm 行布局。失败直接 InvalidArgument。
fn derive_layout(t: &Tensor, dim_required: usize) -> Result<Layout> {
    let shape = t.shape();
    let strides = t.strides();
    if shape.is_empty() {
        return Err(Error::InvalidArgument(
            "RMSNorm CUDA: input must not be 0-D".to_string(),
        )
        .into());
    }
    if *shape.last().unwrap() != dim_required {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA: last-dim ({}) must equal weight dim ({})",
            shape.last().unwrap(),
            dim_required
        ))
        .into());
    }
    if *strides.last().unwrap() != 1 {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA: last-dim must be dense (stride==1), got strides {:?}",
            strides
        ))
        .into());
    }

    // dense: 任意 rank → 全部 flatten 到 outer0
    if t.is_contiguous() {
        let outer0 = (t.numel() / dim_required) as i32;
        return Ok(Layout {
            outer0,
            outer1: 1,
            dim: dim_required as i32,
            stride0: dim_required as i64,
            stride1: 0,
        });
    }

    // strided 2-D：单一 outer0
    if t.ndim() == 2 {
        let outer0 = shape[0] as i32;
        let stride0 = strides[0] as i64;
        if (stride0 as usize) < dim_required {
            return Err(Error::InvalidArgument(format!(
                "RMSNorm CUDA: 2-D row_stride ({}) < dim ({})",
                stride0, dim_required
            ))
            .into());
        }
        return Ok(Layout {
            outer0,
            outer1: 1,
            dim: dim_required as i32,
            stride0,
            stride1: 0,
        });
    }

    // strided 3-D：按 head 切，shape == [outer0, outer1, dim]
    if t.ndim() == 3 {
        let outer0 = shape[0] as i32;
        let outer1 = shape[1] as i32;
        let stride0 = strides[0] as i64;
        let stride1 = strides[1] as i64;
        return Ok(Layout {
            outer0,
            outer1,
            dim: dim_required as i32,
            stride0,
            stride1,
        });
    }

    Err(Error::InvalidArgument(format!(
        "RMSNorm CUDA: input rank {} not supported (must be contiguous, or 2-D/3-D strided with last-dim dense)",
        t.ndim()
    ))
    .into())
}

/// half 路径需要 dim/各 stride 都对齐 8 elem（保证 float4 16-byte 对齐）；
/// f32 路径需要对齐 4。
fn check_align(layout: &Layout, dtype: DataType) -> Result<()> {
    let chunk: i64 = match dtype {
        DataType::F32 => 4,
        DataType::BF16 | DataType::F16 => 8,
        other => {
            return Err(Error::InvalidArgument(format!(
                "RMSNorm CUDA: unsupported dtype {:?}",
                other
            ))
            .into());
        }
    };
    let bad = (layout.dim as i64) % chunk != 0
        || layout.stride0 % chunk != 0
        || layout.stride1 % chunk != 0;
    if bad {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA: dim/stride not aligned to {} elems (dim={}, stride0={}, stride1={})",
            chunk, layout.dim, layout.stride0, layout.stride1
        ))
        .into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch(
    output_ptr: *mut std::ffi::c_void,
    input_ptr: *const std::ffi::c_void,
    weight_ptr: *const std::ffi::c_void,
    in_layout: Layout,
    out_layout: Layout,
    eps: f32,
    dtype: DataType,
    stream: crate::cuda::ffi::cudaStream_t,
) -> Result<()> {
    debug_assert_eq!(in_layout.outer0, out_layout.outer0);
    debug_assert_eq!(in_layout.outer1, out_layout.outer1);
    debug_assert_eq!(in_layout.dim, out_layout.dim);
    unsafe {
        match dtype {
            DataType::F32 => rmsnorm_kernel_cu_dim(
                output_ptr as *mut f32,
                input_ptr as *const f32,
                weight_ptr as *const f32,
                in_layout.outer0,
                in_layout.outer1,
                in_layout.dim,
                in_layout.stride0,
                in_layout.stride1,
                out_layout.stride0,
                out_layout.stride1,
                eps,
                stream,
            ),
            DataType::BF16 => rmsnorm_kernel_cu_bf16x8(
                output_ptr as *mut half::bf16,
                input_ptr as *const half::bf16,
                weight_ptr as *const half::bf16,
                in_layout.outer0,
                in_layout.outer1,
                in_layout.dim,
                in_layout.stride0,
                in_layout.stride1,
                out_layout.stride0,
                out_layout.stride1,
                eps,
                stream,
            ),
            DataType::F16 => rmsnorm_kernel_cu_fp16x8(
                output_ptr as *mut half::f16,
                input_ptr as *const half::f16,
                weight_ptr as *const half::f16,
                in_layout.outer0,
                in_layout.outer1,
                in_layout.dim,
                in_layout.stride0,
                in_layout.stride1,
                out_layout.stride0,
                out_layout.stride1,
                eps,
                stream,
            ),
            other => {
                return Err(Error::InvalidArgument(format!(
                    "RMSNorm CUDA: unsupported dtype {:?}",
                    other
                ))
                .into());
            }
        }
    }
    Ok(())
}

/// `output = rmsnorm(input, weight, eps)`。
pub fn rmsnorm(
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    if weight.ndim() != 1 {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA: weight must be 1-D, got shape {:?}",
            weight.shape()
        ))
        .into());
    }
    let dim = weight.shape()[0];
    let dtype = input.dtype();
    if output.dtype() != dtype || weight.dtype() != dtype {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA: dtype mismatch input={:?} weight={:?} output={:?}",
            dtype,
            weight.dtype(),
            output.dtype()
        ))
        .into());
    }
    let in_layout = derive_layout(input, dim)?;
    let out_layout = derive_layout(output, dim)?;
    if (in_layout.outer0, in_layout.outer1) != (out_layout.outer0, out_layout.outer1) {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA: output ({},{}) != input ({},{}) row count",
            out_layout.outer0, out_layout.outer1, in_layout.outer0, in_layout.outer1
        ))
        .into());
    }
    check_align(&in_layout, dtype)?;
    check_align(&out_layout, dtype)?;

    let stream = CudaConfig::resolve_stream(cuda_config);
    let in_ptr = data_ptr_const(input)?;
    let out_ptr = data_ptr_mut(output)?;
    let w_ptr = data_ptr_const(weight)?;
    unsafe { dispatch(out_ptr, in_ptr, w_ptr, in_layout, out_layout, eps, dtype, stream) }
}

/// `x = rmsnorm(x, weight, eps)`，原地。
pub fn rmsnorm_inplace(
    x: &mut Tensor,
    weight: &Tensor,
    eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    if weight.ndim() != 1 {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA in-place: weight must be 1-D, got shape {:?}",
            weight.shape()
        ))
        .into());
    }
    let dim = weight.shape()[0];
    let dtype = x.dtype();
    if weight.dtype() != dtype {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CUDA in-place: dtype mismatch x={:?} weight={:?}",
            dtype,
            weight.dtype()
        ))
        .into());
    }
    let layout = derive_layout(x, dim)?;
    check_align(&layout, dtype)?;

    let stream = CudaConfig::resolve_stream(cuda_config);
    let ptr = data_ptr_mut(x)?;
    let w_ptr = data_ptr_const(weight)?;
    unsafe { dispatch(ptr, ptr as *const _, w_ptr, layout, layout, eps, dtype, stream) }
}

fn data_ptr_const(t: &Tensor) -> Result<*const std::ffi::c_void> {
    Ok(match t.dtype() {
        DataType::F32 => t.as_f32()?.data_ptr() as *const _,
        DataType::BF16 => t.as_bf16()?.data_ptr() as *const _,
        DataType::F16 => t.as_f16()?.data_ptr() as *const _,
        other => {
            return Err(Error::InvalidArgument(format!(
                "RMSNorm CUDA: unsupported dtype {:?}",
                other
            ))
            .into());
        }
    })
}

fn data_ptr_mut(t: &mut Tensor) -> Result<*mut std::ffi::c_void> {
    Ok(match t.dtype() {
        DataType::F32 => t.as_f32_mut()?.data_ptr_mut() as *mut _,
        DataType::BF16 => t.as_bf16_mut()?.data_ptr_mut() as *mut _,
        DataType::F16 => t.as_f16_mut()?.data_ptr_mut() as *mut _,
        other => {
            return Err(Error::InvalidArgument(format!(
                "RMSNorm CUDA: unsupported dtype {:?}",
                other
            ))
            .into());
        }
    })
}
