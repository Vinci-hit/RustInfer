use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::{bf16, f16};

/// dst[i, j] = a[i, j] * b[j]
/// a: [rows, D], b: [D], dst: [rows, D]
pub fn broadcast_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    let shape = a.shape();
    let d = *shape.last().unwrap();
    let rows = a.num_elements() / d;

    match (a, b, dst) {
        (Tensor::F32(ta), Tensor::F32(tb), Tensor::F32(td)) => {
            let a_s = ta.as_slice()?;
            let b_s = tb.as_slice()?;
            let d_s = td.as_slice_mut()?;
            for r in 0..rows {
                let base = r * d;
                for j in 0..d {
                    d_s[base + j] = a_s[base + j] * b_s[j];
                }
            }
        }
        (Tensor::BF16(ta), Tensor::BF16(tb), Tensor::BF16(td)) => {
            let a_s = ta.as_slice()?;
            let b_s = tb.as_slice()?;
            let d_s = td.as_slice_mut()?;
            for r in 0..rows {
                let base = r * d;
                for j in 0..d {
                    d_s[base + j] = bf16::from_f32(a_s[base + j].to_f32() * b_s[j].to_f32());
                }
            }
        }
        (Tensor::F16(ta), Tensor::F16(tb), Tensor::F16(td)) => {
            let a_s = ta.as_slice()?;
            let b_s = tb.as_slice()?;
            let d_s = td.as_slice_mut()?;
            for r in 0..rows {
                let base = r * d;
                for j in 0..d {
                    d_s[base + j] = f16::from_f32(a_s[base + j].to_f32() * b_s[j].to_f32());
                }
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "broadcast_mul: unsupported dtype {:?}", a.dtype()
        )).into()),
    }
    Ok(())
}
