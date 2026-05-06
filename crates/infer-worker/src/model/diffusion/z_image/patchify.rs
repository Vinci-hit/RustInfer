//! Patchify / Unpatchify for DiT (Z-Image).
//!
//! Only zero-alloc `_into` variants are provided for production use.

use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

pub fn patchify_into(
    image: &Tensor,
    patch_size: usize,
    f_patch_size: usize,
    dst: &mut Tensor,
) -> Result<()> {
    let shape = image.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "patchify_into: expected [C, F, H, W], got {:?}", shape
        )).into());
    }
    let (c, f, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);

    if f % p_f != 0 || h % p_h != 0 || w % p_w != 0 {
        return Err(Error::InvalidArgument(format!(
            "patchify_into: (F={f}, H={h}, W={w}) not divisible by (pF={p_f}, pH={p_h}, pW={p_w})"
        )).into());
    }
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
    let num_tokens = f_t * h_t * w_t;
    let patch_flat = p_f * p_h * p_w * c;
    if dst.shape() != [num_tokens, patch_flat].as_slice() {
        return Err(Error::InvalidArgument(format!(
            "patchify_into: dst shape {:?} != [{num_tokens}, {patch_flat}]", dst.shape()
        )).into());
    }

    let src_view = image.reshape(&[c, f_t, p_f, h_t, p_h, w_t, p_w])?;
    let mut dst_7d = dst.reshape(&[f_t, h_t, w_t, p_f, p_h, p_w, c])?;
    src_view.permute_into(&[1, 3, 5, 2, 4, 6, 0], &mut dst_7d)?;
    Ok(())
}

pub fn unpatchify_into(
    tokens: &Tensor,
    f: usize, h: usize, w: usize,
    out_channels: usize,
    patch_size: usize,
    f_patch_size: usize,
    dst: &mut Tensor,
) -> Result<()> {
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);

    if dst.shape() != [out_channels, f, h, w].as_slice() {
        return Err(Error::InvalidArgument(format!(
            "unpatchify_into: dst shape {:?} != [{out_channels}, {f}, {h}, {w}]", dst.shape()
        )).into());
    }

    let src_view = tokens.reshape(&[f_t, h_t, w_t, p_f, p_h, p_w, out_channels])?;
    let mut dst_7d = dst.reshape(&[out_channels, f_t, p_f, h_t, p_h, w_t, p_w])?;
    src_view.permute_into(&[6, 0, 3, 1, 4, 2, 5], &mut dst_7d)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;

    fn patchify(image: &Tensor, patch_size: usize, f_patch_size: usize) -> Result<Tensor> {
        let shape = image.shape();
        let (c, f, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
        let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
        let t = image.reshape(&[c, f_t, p_f, h_t, p_h, w_t, p_w])?;
        // `permute` yields a zero-copy strided view; `reshape` densifies
        // before collapsing the axes.
        let t = t.permute(&[1, 3, 5, 2, 4, 6, 0])?;
        t.reshape(&[f_t * h_t * w_t, p_f * p_h * p_w * c])
    }

    fn unpatchify(
        tokens: &Tensor, f: usize, h: usize, w: usize,
        out_channels: usize, patch_size: usize, f_patch_size: usize,
    ) -> Result<Tensor> {
        let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
        let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
        let t = tokens.reshape(&[f_t, h_t, w_t, p_f, p_h, p_w, out_channels])?;
        let t = t.permute(&[6, 0, 3, 1, 4, 2, 5])?;
        t.reshape(&[out_channels, f, h, w])
    }

    #[test]
    fn test_roundtrip() -> Result<()> {
        let (c, f, h, w) = (2, 1, 4, 4);
        let numel = c * f * h * w;
        let mut image = Tensor::new(&[c, f, h, w], DataType::F32, DeviceType::Cpu)?;
        let data = image.as_f32_mut()?.as_slice_mut()?;
        for i in 0..numel { data[i] = i as f32; }

        let tokens = patchify(&image, 2, 1)?;
        let restored = unpatchify(&tokens, f, h, w, c, 2, 1)?;

        let orig = image.as_f32()?.as_slice()?;
        let rest = restored.as_f32()?.as_slice()?;
        for i in 0..numel {
            assert_eq!(orig[i], rest[i], "roundtrip mismatch at {i}");
        }
        Ok(())
    }
}
