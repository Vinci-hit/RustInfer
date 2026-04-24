use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

/// Scatter operation on CPU: dst[pos, :] = src[0, :]
pub fn scatter(dst: &mut Tensor, src: &Tensor, pos: &Tensor) -> Result<()> {
    let pos_val = pos.as_i32()?.as_slice()?[0] as usize;
    let kvdim = src.shape()[1];
    let max_seq_len = dst.shape()[0];
    if pos_val >= max_seq_len {
        return Err(Error::InvalidArgument(format!(
            "scatter position out of range: pos={}, max_seq_len={}",
            pos_val, max_seq_len
        ))
        .into());
    }

    match src.dtype() {
        crate::base::DataType::F32 => {
            let src_slice = src.as_f32()?.as_slice()?;
            let dst_slice = dst.as_f32_mut()?.as_slice_mut()?;
            let dst_start = pos_val * kvdim;
            dst_slice[dst_start..dst_start + kvdim].copy_from_slice(&src_slice[..kvdim]);
        }
        crate::base::DataType::BF16 => {
            let src_slice = src.as_bf16()?.as_slice()?;
            let dst_slice = dst.as_bf16_mut()?.as_slice_mut()?;
            let dst_start = pos_val * kvdim;
            dst_slice[dst_start..dst_start + kvdim].copy_from_slice(&src_slice[..kvdim]);
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "scatter CPU: unsupported dtype {:?}",
                src.dtype()
            ))
            .into());
        }
    }

    Ok(())
}

/// Fused scatter on CPU: write both K and V cache at the same position.
pub fn scatter_kv(
    dst_k: &mut Tensor,
    src_k: &Tensor,
    dst_v: &mut Tensor,
    src_v: &Tensor,
    pos: &Tensor,
) -> Result<()> {
    scatter(dst_k, src_k, pos)?;
    scatter(dst_v, src_v, pos)?;
    Ok(())
}
