use crate::base::error::{Result, Error};
use crate::tensor::Tensor;
use half::bf16;
use ndarray::{ArrayView1, ArrayViewMut1};
use ndarray_linalg::Norm;
use rayon::prelude::*;

/// RMSNorm 的 CPU 内核实现，使用 ndarray 库进行高性能计算。
pub fn rmsnorm(input: &Tensor, weight: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match (input.dtype(), weight.dtype(), output.dtype()) {
        (crate::base::DataType::F32, crate::base::DataType::F32, crate::base::DataType::F32) => {
            rmsnorm_f32(input, weight, output, eps)
        }
        (crate::base::DataType::BF16, crate::base::DataType::BF16, crate::base::DataType::BF16) => {
            rmsnorm_bf16(input, weight, output, eps)
        }
        _ => {
            // 如果输入和输出的数据类型不匹配，则返回错误
            Err(Error::InvalidArgument(format!(
                "Unsupported data type combination for rmsnorm: input={:?}, weight={:?}, output={:?}",
                input.dtype(), weight.dtype(), output.dtype()
            )).into())
        }
    }
}

/// In-place CPU RMSNorm: `x = rmsnorm(x, weight, eps)`.
///
/// Each row is independent and processed inside a single `par_chunks_mut`
/// closure: we first read `x[row]` twice to compute the L2 norm → rsqrt,
/// then rewrite `x[row]` with `weight * x[row] * rsqrt`. Because each
/// row chunk is owned exclusively by one rayon task, the in-place update
/// is data-race free.
pub fn rmsnorm_inplace(x: &mut Tensor, weight: &Tensor, eps: f32) -> Result<()> {
    match (x.dtype(), weight.dtype()) {
        (crate::base::DataType::F32, crate::base::DataType::F32) => rmsnorm_inplace_f32(x, weight, eps),
        (crate::base::DataType::BF16, crate::base::DataType::BF16) => rmsnorm_inplace_bf16(x, weight, eps),
        (xd, wd) => Err(Error::InvalidArgument(format!(
            "Unsupported data type combination for rmsnorm_inplace: x={:?}, weight={:?}",
            xd, wd
        )).into()),
    }
}

/// F32版本的RMSNorm实现
fn rmsnorm_f32(input: &Tensor, weight: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    // --- 1. 获取 Rust slice ---
    // 类型和设备检查已在上层完成，这里可以直接 unwrap
    let input_typed = input.as_f32().unwrap();
    let weight_typed = weight.as_f32().unwrap();
    let output_typed = output.as_f32_mut().unwrap();

    let input_slice = input_typed.as_slice().unwrap();
    let weight_slice = weight_typed.as_slice().unwrap();
    let output_slice = output_typed.as_slice_mut().unwrap();
    
    let dim = weight_slice.len();
    let dim_sqrt = (dim as f32).sqrt();
    // --- 2. 将 slice 转换为 ndarray 视图 (零拷贝) ---
    let weight_view = ArrayView1::from(weight_slice);
    
    output_slice
        .par_chunks_mut(dim)
        .zip(input_slice.par_chunks(dim))
        .for_each(|(output_chunk, input_chunk)| {
            let input_row_view = ArrayView1::from(input_chunk);
            let mut output_row_view = ArrayViewMut1::from(output_chunk);
            let norm = input_row_view.norm_l2();
            let rms = norm / dim_sqrt;
            let rsqrt = (rms * rms + eps).sqrt().recip();
            
            output_row_view.assign(&(&weight_view * (&input_row_view * rsqrt)));
        });
    
    Ok(())
}

fn rmsnorm_inplace_f32(x: &mut Tensor, weight: &Tensor, eps: f32) -> Result<()> {
    let weight_slice = weight.as_f32()?.as_slice()?;
    let x_slice = x.as_f32_mut()?.as_slice_mut()?;

    let dim = weight_slice.len();
    let dim_sqrt = (dim as f32).sqrt();
    let weight_view = ArrayView1::from(weight_slice);

    x_slice
        .par_chunks_mut(dim)
        .for_each(|row| {
            // Pass 1: read only — compute rsqrt from the row's L2 norm.
            let row_view = ArrayView1::from(&*row);
            let norm = row_view.norm_l2();
            let rms = norm / dim_sqrt;
            let rsqrt = (rms * rms + eps).sqrt().recip();
            // Pass 2: overwrite row in place.
            let mut row_mut = ArrayViewMut1::from(row);
            let scaled = &weight_view * rsqrt;
            row_mut *= &scaled;
        });
    Ok(())
}

/// BF16版本的RMSNorm实现
fn rmsnorm_bf16(input: &Tensor, weight: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    // --- 1. 获取 Rust slice ---
    let input_typed = input.as_bf16()?;
    let weight_typed = weight.as_bf16()?;
    let output_typed = output.as_bf16_mut()?;

    let input_slice = input_typed.as_slice()?;
    let weight_slice = weight_typed.as_slice()?;
    let output_slice = output_typed.as_slice_mut()?;
    
    let dim = weight_slice.len();
    let dim_sqrt = (dim as f32).sqrt();
    
    // --- 2. 将 slice 转换为 ndarray 视图 (零拷贝) ---
    // 对于BF16，我们需要将数据转换为F32进行计算以保证精度
    output_slice
        .par_chunks_mut(dim)
        .zip(input_slice.par_chunks(dim))
        .for_each(|(output_chunk, input_chunk)| {
            // 将BF16输入转换为F32进行计算
            let input_f32: Vec<f32> = input_chunk.iter().map(|&x| x.to_f32()).collect();
            let weight_f32: Vec<f32> = weight_slice.iter().map(|&x| x.to_f32()).collect();
            
            let input_row_view = ArrayView1::from(input_f32.as_slice());
            let weight_view = ArrayView1::from(weight_f32.as_slice());
            
            let norm = input_row_view.norm_l2();
            let rms = norm / dim_sqrt;
            let rsqrt = (rms * rms + eps).sqrt().recip();
            
            // 计算结果并转换回BF16
            let result: Vec<bf16> = input_row_view.iter()
                .zip(weight_view.iter())
                .map(|(&inp, &wgt)| bf16::from_f32(inp * rsqrt * wgt))
                .collect();
                
            output_chunk.copy_from_slice(&result);
        });
    
    Ok(())
}

fn rmsnorm_inplace_bf16(x: &mut Tensor, weight: &Tensor, eps: f32) -> Result<()> {
    let weight_slice = weight.as_bf16()?.as_slice()?;
    let x_slice = x.as_bf16_mut()?.as_slice_mut()?;

    let dim = weight_slice.len();
    let dim_sqrt = (dim as f32).sqrt();
    let weight_f32: Vec<f32> = weight_slice.iter().map(|&w| w.to_f32()).collect();

    x_slice
        .par_chunks_mut(dim)
        .for_each(|row| {
            // Pass 1: read-only norm computation in f32 for precision.
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                let f = v.to_f32();
                sum_sq += f * f;
            }
            let rms = (sum_sq / dim as f32).sqrt();
            let rsqrt = (rms * rms + eps).sqrt().recip();
            // Pass 2: in-place overwrite.
            for (v, &w) in row.iter_mut().zip(weight_f32.iter()) {
                let f = v.to_f32() * rsqrt * w;
                *v = bf16::from_f32(f);
            }
            let _ = dim_sqrt; // silence unused (kept for API symmetry)
        });
    Ok(())
}
