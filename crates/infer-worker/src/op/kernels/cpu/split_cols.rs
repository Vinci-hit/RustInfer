use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

/// CPU column split utility:
/// copy columns [col_offset, col_offset + dst_cols) from src[rows, total_cols] to dst[rows, dst_cols].
pub fn split_cols_tensor(
    src: &Tensor,
    dst: &mut Tensor,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
) -> Result<()> {
    if col_offset + dst_cols > total_cols {
        return Err(Error::InvalidArgument(format!(
            "split_cols out of bounds: offset {} + cols {} > total {}",
            col_offset, dst_cols, total_cols
        ))
        .into());
    }

    match src.dtype() {
        crate::base::DataType::BF16 => {
            let src_slice = src.as_bf16()?.as_slice()?;
            let dst_slice = dst.as_bf16_mut()?.as_slice_mut()?;
            for r in 0..rows {
                let src_row_start = r * total_cols + col_offset;
                let dst_row_start = r * dst_cols;
                dst_slice[dst_row_start..dst_row_start + dst_cols]
                    .copy_from_slice(&src_slice[src_row_start..src_row_start + dst_cols]);
            }
            Ok(())
        }
        crate::base::DataType::F32 => {
            let src_slice = src.as_f32()?.as_slice()?;
            let dst_slice = dst.as_f32_mut()?.as_slice_mut()?;
            for r in 0..rows {
                let src_row_start = r * total_cols + col_offset;
                let dst_row_start = r * dst_cols;
                dst_slice[dst_row_start..dst_row_start + dst_cols]
                    .copy_from_slice(&src_slice[src_row_start..src_row_start + dst_cols]);
            }
            Ok(())
        }
        _ => Err(Error::InvalidArgument(format!(
            "split_cols CPU: unsupported dtype {:?}",
            src.dtype()
        ))
        .into()),
    }
}
