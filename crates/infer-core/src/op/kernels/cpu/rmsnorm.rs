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

/// Fused CPU path: residual += input; norm_output = rmsnorm(residual, weight)
/// Supports both F32 and BF16.
pub fn fused_add_rmsnorm(
    norm_output: &mut Tensor,
    residual: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<()> {
    match (
        norm_output.dtype(),
        residual.dtype(),
        input.dtype(),
        weight.dtype(),
    ) {
        (
            crate::base::DataType::F32,
            crate::base::DataType::F32,
            crate::base::DataType::F32,
            crate::base::DataType::F32,
        ) => fused_add_rmsnorm_f32(norm_output, residual, input, weight, eps),
        (
            crate::base::DataType::BF16,
            crate::base::DataType::BF16,
            crate::base::DataType::BF16,
            crate::base::DataType::BF16,
        ) => fused_add_rmsnorm_bf16(norm_output, residual, input, weight, eps),
        _ => Err(Error::InvalidArgument(format!(
            "Unsupported dtype combination for fused_add_rmsnorm: norm_output={:?}, residual={:?}, input={:?}, weight={:?}",
            norm_output.dtype(),
            residual.dtype(),
            input.dtype(),
            weight.dtype()
        ))
        .into()),
    }
}

fn fused_add_rmsnorm_f32(
    norm_output: &mut Tensor,
    residual: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<()> {
    let dim = weight.shape()[0];

    let residual_slice = residual.as_f32_mut()?.as_slice_mut()?;
    let input_slice = input.as_f32()?.as_slice()?;
    let weight_slice = weight.as_f32()?.as_slice()?;
    let norm_output_slice = norm_output.as_f32_mut()?.as_slice_mut()?;

    if residual_slice.len() != input_slice.len() || residual_slice.len() != norm_output_slice.len() {
        return Err(Error::InvalidArgument("fused_add_rmsnorm_f32 shape mismatch".to_string()).into());
    }

    residual_slice
        .par_chunks_mut(dim)
        .zip(input_slice.par_chunks(dim))
        .zip(norm_output_slice.par_chunks_mut(dim))
        .for_each(|((res_row, inp_row), out_row)| {
            let mut sum_sq = 0.0f32;
            for j in 0..dim {
                res_row[j] += inp_row[j];
                let v = res_row[j];
                sum_sq += v * v;
            }
            let rsqrt = (sum_sq / dim as f32 + eps).sqrt().recip();
            for j in 0..dim {
                out_row[j] = res_row[j] * rsqrt * weight_slice[j];
            }
        });

    Ok(())
}

fn fused_add_rmsnorm_bf16(
    norm_output: &mut Tensor,
    residual: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<()> {
    let dim = weight.shape()[0];

    let residual_slice = residual.as_bf16_mut()?.as_slice_mut()?;
    let input_slice = input.as_bf16()?.as_slice()?;
    let weight_slice = weight.as_bf16()?.as_slice()?;
    let norm_output_slice = norm_output.as_bf16_mut()?.as_slice_mut()?;

    if residual_slice.len() != input_slice.len() || residual_slice.len() != norm_output_slice.len() {
        return Err(Error::InvalidArgument("fused_add_rmsnorm_bf16 shape mismatch".to_string()).into());
    }

    residual_slice
        .par_chunks_mut(dim)
        .zip(input_slice.par_chunks(dim))
        .zip(norm_output_slice.par_chunks_mut(dim))
        .for_each(|((res_row, inp_row), out_row)| {
            let mut sum_sq = 0.0f32;
            for j in 0..dim {
                let v = res_row[j].to_f32() + inp_row[j].to_f32();
                res_row[j] = bf16::from_f32(v);
                sum_sq += v * v;
            }
            let rsqrt = (sum_sq / dim as f32 + eps).sqrt().recip();
            for j in 0..dim {
                out_row[j] = bf16::from_f32(res_row[j].to_f32() * rsqrt * weight_slice[j].to_f32());
            }
        });

    Ok(())
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
