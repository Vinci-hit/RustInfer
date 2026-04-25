use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::{bf16, f16};

/// LayerNorm without affine: output[i] = (input[i] - mean) / sqrt(var + eps)
/// input/output: [rows, cols], each row normalized independently.
pub fn layernorm(input: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    let shape = input.shape();
    let cols = *shape.last().unwrap();
    let rows = input.num_elements() / cols;

    match (input, output) {
        (Tensor::F32(ti), Tensor::F32(to)) => {
            let i_s = ti.as_slice()?;
            let o_s = to.as_slice_mut()?;
            for r in 0..rows {
                let base = r * cols;
                let row = &i_s[base..base + cols];
                let mean: f32 = row.iter().sum::<f32>() / cols as f32;
                let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
                let rstd = 1.0 / (var + eps).sqrt();
                for j in 0..cols {
                    o_s[base + j] = (row[j] - mean) * rstd;
                }
            }
        }
        (Tensor::BF16(ti), Tensor::BF16(to)) => {
            let i_s = ti.as_slice()?;
            let o_s = to.as_slice_mut()?;
            for r in 0..rows {
                let base = r * cols;
                let mean: f32 = i_s[base..base + cols].iter().map(|x| x.to_f32()).sum::<f32>() / cols as f32;
                let var: f32 = i_s[base..base + cols].iter().map(|x| { let d = x.to_f32() - mean; d * d }).sum::<f32>() / cols as f32;
                let rstd = 1.0 / (var + eps).sqrt();
                for j in 0..cols {
                    o_s[base + j] = bf16::from_f32((i_s[base + j].to_f32() - mean) * rstd);
                }
            }
        }
        (Tensor::F16(ti), Tensor::F16(to)) => {
            let i_s = ti.as_slice()?;
            let o_s = to.as_slice_mut()?;
            for r in 0..rows {
                let base = r * cols;
                let mean: f32 = i_s[base..base + cols].iter().map(|x| x.to_f32()).sum::<f32>() / cols as f32;
                let var: f32 = i_s[base..base + cols].iter().map(|x| { let d = x.to_f32() - mean; d * d }).sum::<f32>() / cols as f32;
                let rstd = 1.0 / (var + eps).sqrt();
                for j in 0..cols {
                    o_s[base + j] = f16::from_f32((i_s[base + j].to_f32() - mean) * rstd);
                }
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "layernorm: unsupported dtype {:?}", input.dtype()
        )).into()),
    }
    Ok(())
}
