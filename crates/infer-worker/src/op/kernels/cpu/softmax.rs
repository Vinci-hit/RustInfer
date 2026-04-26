use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;

/// CPU Softmax: 沿最后一维做 softmax（数值稳定版本）
/// input: [..., D], output: [..., D]
pub fn softmax(input: &Tensor, output: &mut Tensor) -> Result<()> {
    match input.dtype() {
        crate::base::DataType::F32 => softmax_f32(input, output),
        crate::base::DataType::BF16 => softmax_bf16(input, output),
        _ => Err(Error::InvalidArgument(format!(
            "CPU softmax: unsupported dtype {:?}", input.dtype()
        )).into()),
    }
}

fn softmax_f32(input: &Tensor, output: &mut Tensor) -> Result<()> {
    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    let n_rows = input.num_elements() / last_dim;

    let in_data = input.as_f32()?.as_slice()?;
    let out_data = output.as_f32_mut()?.as_slice_mut()?;

    for row in 0..n_rows {
        let base = row * last_dim;
        // 数值稳定: 减去 max
        let max_val = in_data[base..base + last_dim].iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for j in 0..last_dim {
            let v = (in_data[base + j] - max_val).exp();
            out_data[base + j] = v;
            sum += v;
        }
        let inv_sum = 1.0 / sum;
        for j in 0..last_dim {
            out_data[base + j] *= inv_sum;
        }
    }
    Ok(())
}

fn softmax_bf16(input: &Tensor, output: &mut Tensor) -> Result<()> {
    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    let n_rows = input.num_elements() / last_dim;

    let in_data = input.as_bf16()?.as_slice()?;
    let out_data = output.as_bf16_mut()?.as_slice_mut()?;

    for row in 0..n_rows {
        let base = row * last_dim;
        let max_val = in_data[base..base + last_dim].iter()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b.to_f32()));
        let mut sum = 0.0f32;
        for j in 0..last_dim {
            let v = (in_data[base + j].to_f32() - max_val).exp();
            out_data[base + j] = bf16::from_f32(v);
            sum += v;
        }
        let inv_sum = 1.0 / sum;
        for j in 0..last_dim {
            out_data[base + j] = bf16::from_f32(out_data[base + j].to_f32() * inv_sum);
        }
    }
    Ok(())
}
