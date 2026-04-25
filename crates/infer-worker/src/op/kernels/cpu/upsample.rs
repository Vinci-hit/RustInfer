use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

/// CPU Upsample nearest 2x: input[B, C, H, W] → output[B, C, 2H, 2W]
pub fn upsample_nearest_2x(input: &Tensor, output: &mut Tensor) -> Result<()> {
    match input.dtype() {
        crate::base::DataType::F32 => upsample_nearest_2x_f32(input, output),
        crate::base::DataType::BF16 => upsample_nearest_2x_bf16(input, output),
        _ => Err(Error::InvalidArgument(format!(
            "CPU upsample_nearest_2x: unsupported dtype {:?}", input.dtype()
        )).into()),
    }
}

fn upsample_nearest_2x_f32(input: &Tensor, output: &mut Tensor) -> Result<()> {
    let shape = input.shape(); // [B, C, H, W]
    let batch = shape[0];
    let channels = shape[1];
    let h_in = shape[2];
    let w_in = shape[3];
    let h_out = h_in * 2;
    let w_out = w_in * 2;

    let in_data = input.as_f32()?.as_slice()?;
    let out_data = output.as_f32_mut()?.as_slice_mut()?;

    for b in 0..batch {
        for c in 0..channels {
            let in_base = (b * channels + c) * h_in * w_in;
            let out_base = (b * channels + c) * h_out * w_out;
            for h in 0..h_in {
                for w in 0..w_in {
                    let val = in_data[in_base + h * w_in + w];
                    let oh = h * 2;
                    let ow = w * 2;
                    out_data[out_base + oh * w_out + ow] = val;
                    out_data[out_base + oh * w_out + ow + 1] = val;
                    out_data[out_base + (oh + 1) * w_out + ow] = val;
                    out_data[out_base + (oh + 1) * w_out + ow + 1] = val;
                }
            }
        }
    }
    Ok(())
}

fn upsample_nearest_2x_bf16(input: &Tensor, output: &mut Tensor) -> Result<()> {
    let shape = input.shape();
    let batch = shape[0];
    let channels = shape[1];
    let h_in = shape[2];
    let w_in = shape[3];
    let h_out = h_in * 2;
    let w_out = w_in * 2;

    let in_data = input.as_bf16()?.as_slice()?;
    let out_data = output.as_bf16_mut()?.as_slice_mut()?;

    for b in 0..batch {
        for c in 0..channels {
            let in_base = (b * channels + c) * h_in * w_in;
            let out_base = (b * channels + c) * h_out * w_out;
            for h in 0..h_in {
                for w in 0..w_in {
                    let val = in_data[in_base + h * w_in + w];
                    let oh = h * 2;
                    let ow = w * 2;
                    out_data[out_base + oh * w_out + ow] = val;
                    out_data[out_base + oh * w_out + ow + 1] = val;
                    out_data[out_base + (oh + 1) * w_out + ow] = val;
                    out_data[out_base + (oh + 1) * w_out + ow + 1] = val;
                }
            }
        }
    }
    Ok(())
}
