use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

/// CPU implementation of scatter operation
///
/// Copies src[0, :] to dst[pos, :]
///
/// # Arguments
/// * `dst` - Destination tensor with shape [max_seq_len, kvdim]
/// * `src` - Source tensor with shape [1, kvdim]
/// * `pos` - Position tensor (I32 scalar) indicating where to write in dst
pub fn scatter(dst: &mut Tensor, src: &Tensor, pos: &Tensor) -> Result<()> {
    // Validate shapes
    if src.shape().len() != 2 || src.shape()[0] != 1 {
        return Err(Error::InvalidArgument(
            format!("Scatter src must have shape [1, kvdim], got {:?}", src.shape())
        ).into());
    }

    if dst.shape().len() != 2 {
        return Err(Error::InvalidArgument(
            format!("Scatter dst must have shape [max_seq_len, kvdim], got {:?}", dst.shape())
        ).into());
    }

    if pos.shape().len() != 1 || pos.shape()[0] != 1 {
        return Err(Error::InvalidArgument(
            format!("Scatter pos must be a scalar (shape [1]), got {:?}", pos.shape())
        ).into());
    }

    let kvdim = src.shape()[1];
    let max_seq_len = dst.shape()[0];

    if dst.shape()[1] != kvdim {
        return Err(Error::InvalidArgument(
            format!("Scatter kvdim mismatch: src has {}, dst has {}", kvdim, dst.shape()[1])
        ).into());
    }

    // Get position value
    let pos_value = pos.as_i32()?.as_slice()?[0] as usize;

    // Validate position
    if pos_value >= max_seq_len {
        return Err(Error::InvalidArgument(
            format!("Scatter position {} is out of bounds for max_seq_len {}", pos_value, max_seq_len)
        ).into());
    }

    // Dispatch based on data type
    match src.dtype() {
        crate::base::DataType::F32 => {
            scatter_f32(dst, src, pos_value, kvdim)
        }
        crate::base::DataType::BF16 => {
            scatter_bf16(dst, src, pos_value, kvdim)
        }
        _ => {
            Err(Error::InvalidArgument(
                format!("Scatter does not support data type {:?}", src.dtype())
            ).into())
        }
    }
}

/// F32 scatter implementation
fn scatter_f32(dst: &mut Tensor, src: &Tensor, pos: usize, kvdim: usize) -> Result<()> {
    let src_data = src.as_f32()?.as_slice()?;
    let dst_data = dst.as_f32_mut()?.as_slice_mut()?;

    let dst_offset = pos * kvdim;

    // Copy src[0, :] to dst[pos, :]
    dst_data[dst_offset..dst_offset + kvdim].copy_from_slice(&src_data[0..kvdim]);

    Ok(())
}

/// BF16 scatter implementation
fn scatter_bf16(dst: &mut Tensor, src: &Tensor, pos: usize, kvdim: usize) -> Result<()> {
    let src_data = src.as_bf16()?.as_slice()?;
    let dst_data = dst.as_bf16_mut()?.as_slice_mut()?;

    let dst_offset = pos * kvdim;

    // Copy src[0, :] to dst[pos, :]
    dst_data[dst_offset..dst_offset + kvdim].copy_from_slice(&src_data[0..kvdim]);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use half::bf16;

    #[test]
    fn test_scatter_f32() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::F32;
        let max_seq_len = 10;
        let kvdim = 8;

        // Create source tensor [1, kvdim]
        let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
        let src_data: Vec<f32> = (0..kvdim).map(|i| (i as f32) * 10.0).collect();
        src.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&src_data);

        // Create destination tensor [max_seq_len, kvdim]
        let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
        let dst_data = vec![0.0f32; max_seq_len * kvdim];
        dst.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&dst_data);

        // Create position tensor
        let mut pos = Tensor::new(&[1], DataType::I32, device)?;
        pos.as_i32_mut()?.as_slice_mut()?[0] = 5;

        // Execute scatter
        scatter(&mut dst, &src, &pos)?;

        // Verify
        let result = dst.as_f32()?.as_slice()?;

        // Check scattered row
        for i in 0..kvdim {
            assert_eq!(result[5 * kvdim + i], src_data[i]);
        }

        // Check other rows are unchanged
        for row in 0..max_seq_len {
            if row != 5 {
                for i in 0..kvdim {
                    assert_eq!(result[row * kvdim + i], 0.0);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_scatter_bf16() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;
        let max_seq_len = 16;
        let kvdim = 64;

        // Create source tensor [1, kvdim]
        let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
        let src_data: Vec<bf16> = (0..kvdim).map(|i| bf16::from_f32((i as f32) * 0.5 + 1.0)).collect();
        src.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&src_data);

        // Create destination tensor [max_seq_len, kvdim]
        let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
        let dst_data = vec![bf16::from_f32(-1.0); max_seq_len * kvdim];
        dst.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&dst_data);

        // Create position tensor
        let mut pos = Tensor::new(&[1], DataType::I32, device)?;
        pos.as_i32_mut()?.as_slice_mut()?[0] = 10;

        // Execute scatter
        scatter(&mut dst, &src, &pos)?;

        // Verify
        let result = dst.as_bf16()?.as_slice()?;

        // Check scattered row
        for i in 0..kvdim {
            assert_eq!(result[10 * kvdim + i], src_data[i]);
        }

        // Check other rows are unchanged
        for row in 0..max_seq_len {
            if row != 10 {
                for i in 0..kvdim {
                    assert_eq!(result[row * kvdim + i].to_f32(), -1.0);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_scatter_edge_cases() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::F32;
        let max_seq_len = 8;
        let kvdim = 4;

        // Test scatter to first position (pos=0)
        {
            let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
            src.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

            let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
            dst.as_f32_mut()?.as_slice_mut()?.fill(0.0);

            let mut pos = Tensor::new(&[1], DataType::I32, device)?;
            pos.as_i32_mut()?.as_slice_mut()?[0] = 0;

            scatter(&mut dst, &src, &pos)?;

            let result = dst.as_f32()?.as_slice()?;
            assert_eq!(&result[0..4], &[1.0, 2.0, 3.0, 4.0]);
        }

        // Test scatter to last position (pos=max_seq_len-1)
        {
            let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
            src.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

            let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
            dst.as_f32_mut()?.as_slice_mut()?.fill(0.0);

            let mut pos = Tensor::new(&[1], DataType::I32, device)?;
            pos.as_i32_mut()?.as_slice_mut()?[0] = (max_seq_len - 1) as i32;

            scatter(&mut dst, &src, &pos)?;

            let result = dst.as_f32()?.as_slice()?;
            let last_row_start = (max_seq_len - 1) * kvdim;
            assert_eq!(&result[last_row_start..last_row_start + 4], &[5.0, 6.0, 7.0, 8.0]);
        }

        Ok(())
    }
}
