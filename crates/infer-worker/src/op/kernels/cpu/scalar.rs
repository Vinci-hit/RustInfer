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

/// 原地 SiLU: x[i] = x[i] * sigmoid(x[i])  where sigmoid(x) = 1/(1+exp(-x))
pub fn silu_inplace(x: &mut Tensor) -> Result<()> {
    #[inline]
    fn silu_f32(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

    match x {
        Tensor::F32(t) => {
            t.as_slice_mut()?.iter_mut().for_each(|v| *v = silu_f32(*v));
        }
        Tensor::BF16(t) => {
            t.as_slice_mut()?.iter_mut().for_each(|v| *v = bf16::from_f32(silu_f32(v.to_f32())));
        }
        Tensor::F16(t) => {
            t.as_slice_mut()?.iter_mut().for_each(|v| *v = f16::from_f32(silu_f32(v.to_f32())));
        }
        _ => return Err(Error::InvalidArgument(format!(
            "silu_inplace: unsupported dtype {:?}", x.dtype()
        )).into()),
    }
    Ok(())
}
