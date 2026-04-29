use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;

/// CPU GroupNorm: input[B, C, H, W] → output[B, C, H, W]
/// 按通道分组归一化，每组内独立计算 mean 和 var。
/// weight[C], bias[C] 是可学习的 affine 参数。
pub fn groupnorm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &mut Tensor,
    num_groups: usize,
    eps: f32,
) -> Result<()> {
    match input.dtype() {
        crate::base::DataType::F32 => groupnorm_f32(input, weight, bias, output, num_groups, eps),
        crate::base::DataType::BF16 => groupnorm_bf16(input, weight, bias, output, num_groups, eps),
        _ => Err(Error::InvalidArgument(format!(
            "CPU groupnorm: unsupported dtype {:?}", input.dtype()
        )).into()),
    }
}

fn groupnorm_f32(
    input: &Tensor, weight: &Tensor, bias: &Tensor, output: &mut Tensor,
    num_groups: usize, eps: f32,
) -> Result<()> {
    let shape = input.shape(); // [B, C, H, W]
    let batch = shape[0];
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;

    let in_data = input.as_f32()?.as_slice()?;
    let w_data = weight.as_f32()?.as_slice()?;
    let b_data = bias.as_f32()?.as_slice()?;
    let out_data = output.as_f32_mut()?.as_slice_mut()?;

    for b in 0..batch {
        for g in 0..num_groups {
            let c_start = g * channels_per_group;
            // 计算 group 内均值
            let mut sum = 0.0f64;
            for c in c_start..c_start + channels_per_group {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    sum += in_data[base + s] as f64;
                }
            }
            let mean = sum / group_size as f64;

            // 计算方差
            let mut var_sum = 0.0f64;
            for c in c_start..c_start + channels_per_group {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    let diff = in_data[base + s] as f64 - mean;
                    var_sum += diff * diff;
                }
            }
            let rstd = 1.0 / ((var_sum / group_size as f64) + eps as f64).sqrt();

            // 归一化 + affine
            for c in c_start..c_start + channels_per_group {
                let base = (b * channels + c) * spatial;
                let w = w_data[c] as f64;
                let bi = b_data[c] as f64;
                for s in 0..spatial {
                    let val = (in_data[base + s] as f64 - mean) * rstd;
                    out_data[base + s] = (val * w + bi) as f32;
                }
            }
        }
    }
    Ok(())
}

fn groupnorm_bf16(
    input: &Tensor, weight: &Tensor, bias: &Tensor, output: &mut Tensor,
    num_groups: usize, eps: f32,
) -> Result<()> {
    let shape = input.shape();
    let batch = shape[0];
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;

    let in_data = input.as_bf16()?.as_slice()?;
    let w_data = weight.as_bf16()?.as_slice()?;
    let b_data = bias.as_bf16()?.as_slice()?;
    let out_data = output.as_bf16_mut()?.as_slice_mut()?;

    for b in 0..batch {
        for g in 0..num_groups {
            let c_start = g * channels_per_group;
            let mut sum = 0.0f64;
            for c in c_start..c_start + channels_per_group {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    sum += in_data[base + s].to_f32() as f64;
                }
            }
            let mean = sum / group_size as f64;

            let mut var_sum = 0.0f64;
            for c in c_start..c_start + channels_per_group {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    let diff = in_data[base + s].to_f32() as f64 - mean;
                    var_sum += diff * diff;
                }
            }
            let rstd = 1.0 / ((var_sum / group_size as f64) + eps as f64).sqrt();

            for c in c_start..c_start + channels_per_group {
                let base = (b * channels + c) * spatial;
                let w = w_data[c].to_f32() as f64;
                let bi = b_data[c].to_f32() as f64;
                for s in 0..spatial {
                    let val = (in_data[base + s].to_f32() as f64 - mean) * rstd;
                    out_data[base + s] = bf16::from_f32((val * w + bi) as f32);
                }
            }
        }
    }
    Ok(())
}
