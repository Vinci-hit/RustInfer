use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::{bf16, f16};

/// dst[i] = src[i] * val
pub fn scalar_mul(src: &Tensor, dst: &mut Tensor, val: f32) -> Result<()> {
    match (src, dst) {
        (Tensor::F32(s), Tensor::F32(d)) => {
            let s = s.as_slice()?;
            let d = d.as_slice_mut()?;
            d.iter_mut().zip(s).for_each(|(o, &i)| *o = i * val);
        }
        (Tensor::BF16(s), Tensor::BF16(d)) => {
            let s = s.as_slice()?;
            let d = d.as_slice_mut()?;
            d.iter_mut().zip(s).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() * val));
        }
        (Tensor::F16(s), Tensor::F16(d)) => {
            let s = s.as_slice()?;
            let d = d.as_slice_mut()?;
            d.iter_mut().zip(s).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() * val));
        }
        _ => return Err(Error::InvalidArgument(format!(
            "scalar_mul: unsupported dtype {:?}", src.dtype()
        )).into()),
    }
    Ok(())
}

/// dst[i] = src[i] + val
pub fn scalar_add(src: &Tensor, dst: &mut Tensor, val: f32) -> Result<()> {
    match (src, dst) {
        (Tensor::F32(s), Tensor::F32(d)) => {
            let s = s.as_slice()?;
            let d = d.as_slice_mut()?;
            d.iter_mut().zip(s).for_each(|(o, &i)| *o = i + val);
        }
        (Tensor::BF16(s), Tensor::BF16(d)) => {
            let s = s.as_slice()?;
            let d = d.as_slice_mut()?;
            d.iter_mut().zip(s).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() + val));
        }
        (Tensor::F16(s), Tensor::F16(d)) => {
            let s = s.as_slice()?;
            let d = d.as_slice_mut()?;
            d.iter_mut().zip(s).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() + val));
        }
        _ => return Err(Error::InvalidArgument(format!(
            "scalar_add: unsupported dtype {:?}", src.dtype()
        )).into()),
    }
    Ok(())
}
