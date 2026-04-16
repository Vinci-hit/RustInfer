use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;
use rayon::prelude::*;

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
