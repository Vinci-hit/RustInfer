use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

/// CPU Conv2d: input[B,Cin,H,W] * weight[Cout,Cin,kH,kW] + bias[Cout] → output[B,Cout,Hout,Wout]
///
/// 实现方式: im2col + GEMM。将每个 patch 展开为列向量，然后用矩阵乘法计算。
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    output: &mut Tensor,
    stride: usize,
    padding: usize,
) -> Result<()> {
    match input.dtype() {
        crate::base::DataType::F32 => conv2d_f32(input, weight, bias, output, stride, padding),
        _ => Err(Error::InvalidArgument(format!(
            "CPU conv2d: unsupported dtype {:?}", input.dtype()
        )).into()),
    }
}

fn conv2d_f32(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    output: &mut Tensor,
    stride: usize,
    padding: usize,
) -> Result<()> {
    let in_shape = input.shape();   // [B, Cin, H, W]
    let w_shape = weight.shape();   // [Cout, Cin, kH, kW]
    let out_shape = output.shape(); // [B, Cout, Hout, Wout]

    let batch = in_shape[0];
    let c_in = in_shape[1];
    let h_in = in_shape[2];
    let w_in = in_shape[3];
    let c_out = w_shape[0];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let h_out = out_shape[2];
    let w_out = out_shape[3];

    let in_data = input.as_f32()?.as_slice()?;
    let w_data = weight.as_f32()?.as_slice()?;
    let out_data = output.as_f32_mut()?.as_slice_mut()?;

    // weight reshaped: [Cout, Cin*kH*kW]
    let col_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;

    // im2col buffer
    let mut col = vec![0.0f32; col_size * spatial_out];

    for b in 0..batch {
        let in_base = b * c_in * h_in * w_in;

        // im2col: 将 input[b] 展开为 col[col_size, spatial_out]
        for oh in 0..h_out {
            for ow in 0..w_out {
                let out_col = oh * w_out + ow;
                for ic in 0..c_in {
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * stride + fh;
                            let iw = ow * stride + fw;
                            let col_row = ic * kh * kw + fh * kw + fw;
                            let val = if ih >= padding && ih < h_in + padding
                                && iw >= padding && iw < w_in + padding
                            {
                                let ih = ih - padding;
                                let iw = iw - padding;
                                in_data[in_base + ic * h_in * w_in + ih * w_in + iw]
                            } else {
                                0.0
                            };
                            col[col_row * spatial_out + out_col] = val;
                        }
                    }
                }
            }
        }

        // GEMM: output[b] = weight_mat @ col
        // weight_mat: [Cout, col_size], col: [col_size, spatial_out]
        // output: [Cout, spatial_out]
        let out_base = b * c_out * spatial_out;
        for oc in 0..c_out {
            let w_row_base = oc * col_size;
            for s in 0..spatial_out {
                let mut sum = 0.0f32;
                for k in 0..col_size {
                    sum += w_data[w_row_base + k] * col[k * spatial_out + s];
                }
                out_data[out_base + oc * spatial_out + s] = sum;
            }
        }
    }

    // Add bias
    if let Some(bias_t) = bias {
        let bias_data = bias_t.as_f32()?.as_slice()?;
        for b in 0..batch {
            let out_base = b * c_out * spatial_out;
            for oc in 0..c_out {
                let bias_val = bias_data[oc];
                for s in 0..spatial_out {
                    out_data[out_base + oc * spatial_out + s] += bias_val;
                }
            }
        }
    }

    Ok(())
}
