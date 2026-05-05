//! RMSNorm CPU 内核（通用 strided 版本）。
//!
//! 与 CUDA 路径同模型：input/output 抽象为 `[outer0, outer1, dim]` 三维 strided
//! 视图，最后一维 dense（element stride==1），按 `outer0 * outer1` 行处理。
//!
//! 用 `ndarray::ArrayView2`（先把 3-D flatten 成等价 2-D `[outer0*outer1, dim]`
//! 的 logical row 流）来跑——因为 strided 行偏移 = `o0*stride0 + o1*stride1`
//! 不一定能用单一 row_stride 表达，所以用一个手写 row 偏移函数；row 内最后一维
//! dense，可以直接 `from_raw_parts` 取 `&[T]` slice。
//!
//! 不准 fallback：不满足约束直接 `InvalidArgument`。

use crate::base::error::{Error, Result};
use crate::base::DataType;
use crate::tensor::Tensor;
use half::bf16;
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Layout {
    outer0: usize,
    outer1: usize,
    dim: usize,
    stride0: usize,
    stride1: usize,
}

impl Layout {
    #[inline]
    fn rows(&self) -> usize {
        self.outer0 * self.outer1
    }
    #[inline]
    fn row_offset(&self, row: usize) -> usize {
        if self.outer1 == 1 {
            row * self.stride0
        } else {
            (row / self.outer1) * self.stride0 + (row % self.outer1) * self.stride1
        }
    }
}

fn derive_layout(t: &Tensor, dim_required: usize) -> Result<Layout> {
    let shape = t.shape();
    let strides = t.strides();
    if shape.is_empty() {
        return Err(Error::InvalidArgument(
            "RMSNorm CPU: input must not be 0-D".to_string(),
        )
        .into());
    }
    if *shape.last().unwrap() != dim_required {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CPU: last-dim ({}) != weight dim ({})",
            shape.last().unwrap(),
            dim_required
        ))
        .into());
    }
    if *strides.last().unwrap() != 1 {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CPU: last-dim must be dense (stride==1), got strides {:?}",
            strides
        ))
        .into());
    }

    if t.is_contiguous() {
        let outer0 = t.numel() / dim_required;
        return Ok(Layout {
            outer0,
            outer1: 1,
            dim: dim_required,
            stride0: dim_required,
            stride1: 0,
        });
    }
    if t.ndim() == 2 {
        let outer0 = shape[0];
        let stride0 = strides[0];
        if stride0 < dim_required {
            return Err(Error::InvalidArgument(format!(
                "RMSNorm CPU: 2-D row_stride ({}) < dim ({})",
                stride0, dim_required
            ))
            .into());
        }
        return Ok(Layout {
            outer0,
            outer1: 1,
            dim: dim_required,
            stride0,
            stride1: 0,
        });
    }
    if t.ndim() == 3 {
        return Ok(Layout {
            outer0: shape[0],
            outer1: shape[1],
            dim: dim_required,
            stride0: strides[0],
            stride1: strides[1],
        });
    }
    Err(Error::InvalidArgument(format!(
        "RMSNorm CPU: input rank {} not supported (must be contiguous, or 2-D/3-D strided with last-dim dense)",
        t.ndim()
    ))
    .into())
}

trait RmsnormElem: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}
impl RmsnormElem for f32 {
    #[inline]
    fn to_f32(self) -> f32 { self }
    #[inline]
    fn from_f32(v: f32) -> Self { v }
}
impl RmsnormElem for bf16 {
    #[inline]
    fn to_f32(self) -> f32 { self.to_f32() }
    #[inline]
    fn from_f32(v: f32) -> Self { bf16::from_f32(v) }
}

/// 单行 RMSNorm（in-place 安全：先 pass1 再 pass2）。
fn rmsnorm_row<T: RmsnormElem>(
    row_in: &[T],
    row_out: &mut [T],
    weight_f32: &[f32],
    dim_recip: f32,
    eps: f32,
) {
    debug_assert_eq!(row_in.len(), row_out.len());
    debug_assert_eq!(row_in.len(), weight_f32.len());
    let mut sum_sq = 0.0f32;
    for &v in row_in {
        let f = v.to_f32();
        sum_sq += f * f;
    }
    let rsqrt = (sum_sq * dim_recip + eps).sqrt().recip();
    for ((dst, &src), &w) in row_out.iter_mut().zip(row_in.iter()).zip(weight_f32.iter()) {
        *dst = T::from_f32(src.to_f32() * rsqrt * w);
    }
}

/// 按 row 并行执行 RMSNorm。每行独立 → 不同 row 不会写入同一 (in_off, out_off)
/// 段（in-place 时也成立，每行 in/out 完全重合，但仅本行使用）。
///
/// SAFETY：`in_base` / `out_base` 必须指向至少
/// `max(row_offset) + dim` 个 `T` 的有效 storage；layout 的所有偏移必须落在
/// 该范围内。两条指针可以别名同一段 storage（in-place）。
unsafe fn run_parallel<T: RmsnormElem>(
    in_base: *const T,
    out_base: *mut T,
    weight_f32: &[f32],
    in_layout: Layout,
    out_layout: Layout,
    eps: f32,
) {
    let rows = in_layout.rows();
    let dim = in_layout.dim;
    let dim_recip = 1.0f32 / dim as f32;
    let in_addr = in_base as usize;
    let out_addr = out_base as usize;
    (0..rows).into_par_iter().for_each(move |r| {
        let in_off = in_layout.row_offset(r);
        let out_off = out_layout.row_offset(r);
        // SAFETY: 保证由调用方提供 —— 见 fn 注释。
        unsafe {
            let in_ptr = (in_addr as *const T).add(in_off);
            let out_ptr = (out_addr as *mut T).add(out_off);
            let row_in = std::slice::from_raw_parts(in_ptr, dim);
            let row_out = std::slice::from_raw_parts_mut(out_ptr, dim);
            rmsnorm_row::<T>(row_in, row_out, weight_f32, dim_recip, eps);
        }
    });
}

fn run_dispatch(
    in_ptr: *const u8,
    out_ptr: *mut u8,
    weight: &Tensor,
    in_layout: Layout,
    out_layout: Layout,
    dtype: DataType,
    eps: f32,
) -> Result<()> {
    match dtype {
        DataType::F32 => {
            let weight_f32 = weight.as_f32()?.as_slice()?.to_vec();
            // SAFETY: layout/dtype/storage 已校验。
            unsafe {
                run_parallel::<f32>(
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    &weight_f32,
                    in_layout,
                    out_layout,
                    eps,
                );
            }
            Ok(())
        }
        DataType::BF16 => {
            let weight_f32: Vec<f32> = weight
                .as_bf16()?
                .as_slice()?
                .iter()
                .map(|&v| v.to_f32())
                .collect();
            // SAFETY: 同上。
            unsafe {
                run_parallel::<bf16>(
                    in_ptr as *const bf16,
                    out_ptr as *mut bf16,
                    &weight_f32,
                    in_layout,
                    out_layout,
                    eps,
                );
            }
            Ok(())
        }
        other => Err(Error::InvalidArgument(format!(
            "RMSNorm CPU: unsupported dtype {:?}",
            other
        ))
        .into()),
    }
}

pub fn rmsnorm(input: &Tensor, weight: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    if weight.ndim() != 1 {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CPU: weight must be 1-D, got shape {:?}",
            weight.shape()
        ))
        .into());
    }
    let dim = weight.shape()[0];
    let dtype = input.dtype();
    if output.dtype() != dtype || weight.dtype() != dtype {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CPU: dtype mismatch input={:?} weight={:?} output={:?}",
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
            "RMSNorm CPU: output ({},{}) != input ({},{}) row count",
            out_layout.outer0, out_layout.outer1, in_layout.outer0, in_layout.outer1
        ))
        .into());
    }
    let in_ptr = data_ptr_const(input)? as *const u8;
    let out_ptr = data_ptr_mut(output)? as *mut u8;
    run_dispatch(in_ptr, out_ptr, weight, in_layout, out_layout, dtype, eps)
}

pub fn rmsnorm_inplace(x: &mut Tensor, weight: &Tensor, eps: f32) -> Result<()> {
    if weight.ndim() != 1 {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CPU in-place: weight must be 1-D, got shape {:?}",
            weight.shape()
        ))
        .into());
    }
    let dim = weight.shape()[0];
    let dtype = x.dtype();
    if weight.dtype() != dtype {
        return Err(Error::InvalidArgument(format!(
            "RMSNorm CPU in-place: dtype mismatch x={:?} weight={:?}",
            dtype,
            weight.dtype()
        ))
        .into());
    }
    let layout = derive_layout(x, dim)?;
    let ptr = data_ptr_mut(x)? as *mut u8;
    run_dispatch(ptr as *const u8, ptr, weight, layout, layout, dtype, eps)
}

fn data_ptr_const(t: &Tensor) -> Result<*const std::ffi::c_void> {
    Ok(match t.dtype() {
        DataType::F32 => t.as_f32()?.data_ptr() as *const _,
        DataType::BF16 => t.as_bf16()?.data_ptr() as *const _,
        DataType::F16 => t.as_f16()?.data_ptr() as *const _,
        other => {
            return Err(Error::InvalidArgument(format!(
                "RMSNorm CPU: unsupported dtype {:?}",
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
                "RMSNorm CPU: unsupported dtype {:?}",
                other
            ))
            .into());
        }
    })
}
