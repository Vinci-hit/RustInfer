//! Sequence padding: extend a `[N, D]` tensor to `[target_len, D]`.
//!
//! Three flavours, all device-agnostic:
//!
//! - [`pad_last_row`] / [`pad_last_row_into`] — repeat the final row of
//!   `src` into the tail slots. Used when a pad-token tensor isn't handy.
//! - [`pad_with_token`] / [`pad_with_token_into`] — broadcast a provided
//!   `[D]` token into the tail slots.
//! - [`overwrite_pad_tokens_inplace`] — rewrite rows `[keep_prefix..]` of
//!   an existing tensor with the broadcast token.
//!
//! CUDA paths go through kernels in
//! `op/kernels/cuda/{broadcast_row, cast_fill}/`; D2D memcpy uses
//! `cudaMemcpyAsync`. FFI pointer formation goes via `data_ptr()` so
//! strided/offset views are supported.

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn broadcast_row_bf16_forward(dst: *mut half::bf16, row: *const half::bf16, num_rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_row_f32_forward (dst: *mut f32,        row: *const f32,        num_rows: i32, d: i32, stream: cudaStream_t);
    fn fill_repeat_last_row_bf16_forward(dst: *mut half::bf16, n_src: i32, target_len: i32, d: i32, stream: cudaStream_t);
    fn fill_repeat_last_row_f32_forward (dst: *mut f32,        n_src: i32, target_len: i32, d: i32, stream: cudaStream_t);
}

#[cfg(feature = "cuda")]
#[inline]
unsafe fn d2d_memcpy_async(
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    count: usize,
    stream: cudaStream_t,
) -> Result<()> {
    unsafe {
        let rc = crate::cuda::ffi::cudaMemcpyAsync(
            dst, src, count,
            crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream);
        if rc != crate::cuda::ffi::cudaError_cudaSuccess {
            return Err(Error::InternalError(
                format!("cudaMemcpyAsync D2D failed: {}", rc)).into());
        }
    }
    Ok(())
}

// ────────────────────────── pad_last_row ──────────────────────────

/// Allocate `[target_len, D]` and pad by repeating `src`'s last row.
pub fn pad_last_row(src: &Tensor, target_len: usize) -> Result<Tensor> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row: expected [N, D], got {:?}", shape)).into());
    }
    let (_n, d) = (shape[0], shape[1]);
    let mut dst = Tensor::empty(&[target_len, d], src.dtype(), src.device())?;
    pad_last_row_into(src, &mut dst)?;
    Ok(dst)
}

/// `dst[..n_src] = src; dst[n_src..] = src[n_src - 1]` (last-row repeat).
///
/// `target_len` is inferred from `dst.shape()[0]`.
pub fn pad_last_row_into(src: &Tensor, dst: &mut Tensor) -> Result<()> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: expected [N, D], got {:?}", shape)).into());
    }
    let (n, d) = (shape[0], shape[1]);
    if dst.shape().len() != 2 || dst.shape()[1] != d {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: dst shape {:?} incompatible with [target_len, {}]",
            dst.shape(), d)).into());
    }
    let target_len = dst.shape()[0];
    if target_len < n {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: target_len ({}) < n ({})", target_len, n)).into());
    }
    if src.dtype() != dst.dtype() || src.device() != dst.device() {
        return Err(Error::InvalidArgument(
            "pad_last_row_into: dtype/device mismatch".into()).into());
    }
    if target_len == n {
        return dst.copy_from_on_current_stream(src);
    }

    let bytes_per_row = d * src.dtype().size_in_bytes();
    let src_bytes = n * bytes_per_row;

    match src.device() {
        DeviceType::Cpu => unsafe {
            let src_base = src.data_ptr();
            let dst_base = dst.data_ptr_mut();
            std::ptr::copy_nonoverlapping(src_base, dst_base, src_bytes);
            let last_ptr = src_base.add((n - 1) * bytes_per_row);
            for r in n..target_len {
                let dp = dst_base.add(r * bytes_per_row);
                std::ptr::copy_nonoverlapping(last_ptr, dp, bytes_per_row);
            }
            Ok(())
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            unsafe {
                d2d_memcpy_async(
                    dst.data_ptr_mut() as *mut _,
                    src.data_ptr()      as *const _,
                    src_bytes, stream)?;
            }
            match src.dtype() {
                DataType::BF16 => unsafe {
                    fill_repeat_last_row_bf16_forward(
                        dst.as_bf16_mut()?.data_ptr_mut(),
                        n as i32, target_len as i32, d as i32, stream);
                }
                DataType::F32 => unsafe {
                    fill_repeat_last_row_f32_forward(
                        dst.as_f32_mut()?.data_ptr_mut(),
                        n as i32, target_len as i32, d as i32, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "pad_last_row_into CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

// ────────────────────────── pad_with_token ──────────────────────────

/// Allocate `[target_len, D]` and pad by broadcasting a `[D]` token tensor
/// into the tail slots.
pub fn pad_with_token(src: &Tensor, pad_token: &Tensor, target_len: usize) -> Result<Tensor> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token: expected [N, D], got {:?}", shape)).into());
    }
    let d = shape[1];
    let mut dst = Tensor::empty(&[target_len, d], src.dtype(), src.device())?;
    pad_with_token_into(src, pad_token, &mut dst)?;
    Ok(dst)
}

/// `dst[..n_src] = src; dst[n_src..] = pad_token broadcasted`.
pub fn pad_with_token_into(
    src: &Tensor, pad_token: &Tensor, dst: &mut Tensor,
) -> Result<()> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: expected [N, D], got {:?}", shape)).into());
    }
    let (n, d) = (shape[0], shape[1]);
    if dst.shape().len() != 2 || dst.shape()[1] != d {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: dst shape {:?} incompatible with [target_len, {}]",
            dst.shape(), d)).into());
    }
    let target_len = dst.shape()[0];
    if target_len < n {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: target_len ({}) < n ({})", target_len, n)).into());
    }
    if pad_token.numel() != d {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: pad_token has {} elems, expected {}",
            pad_token.numel(), d)).into());
    }
    if src.dtype() != dst.dtype() || src.device() != dst.device() {
        return Err(Error::InvalidArgument(
            "pad_with_token_into: dtype/device mismatch".into()).into());
    }
    if target_len == n {
        return dst.copy_from_on_current_stream(src);
    }

    let bytes_per_row = d * src.dtype().size_in_bytes();
    let src_bytes = n * bytes_per_row;

    match src.device() {
        DeviceType::Cpu => unsafe {
            let src_base = src.data_ptr();
            let dst_base = dst.data_ptr_mut();
            std::ptr::copy_nonoverlapping(src_base, dst_base, src_bytes);
            let pad_ptr = pad_token.data_ptr();
            for r in n..target_len {
                std::ptr::copy_nonoverlapping(
                    pad_ptr,
                    dst_base.add(r * bytes_per_row),
                    bytes_per_row);
            }
            Ok(())
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            if src_bytes > 0 {
                unsafe {
                    d2d_memcpy_async(
                        dst.data_ptr_mut() as *mut _,
                        src.data_ptr()      as *const _,
                        src_bytes, stream)?;
                }
            }
            let pad_rows = target_len - n;
            if pad_rows > 0 {
                match src.dtype() {
                    DataType::BF16 => unsafe {
                        // dst starts at row `n` in the destination view.
                        let dst_view = dst.as_bf16_mut()?.data_ptr_mut().add(n * d);
                        let pad_ptr  = pad_token.as_bf16()?.data_ptr();
                        broadcast_row_bf16_forward(
                            dst_view, pad_ptr, pad_rows as i32, d as i32, stream);
                    }
                    DataType::F32 => unsafe {
                        let dst_view = dst.as_f32_mut()?.data_ptr_mut().add(n * d);
                        let pad_ptr  = pad_token.as_f32()?.data_ptr();
                        broadcast_row_f32_forward(
                            dst_view, pad_ptr, pad_rows as i32, d as i32, stream);
                    }
                    other => return Err(Error::InvalidArgument(format!(
                        "pad_with_token_into CUDA: unsupported dtype {:?}", other
                    )).into()),
                }
            }
            Ok(())
        }
    }
}

// ────────────────────── overwrite_pad_tokens ──────────────────────

/// In-place rewrite: `x[keep_prefix.., :] = pad_token` (broadcast).
pub fn overwrite_pad_tokens_inplace(
    x: &mut Tensor, pad_token: &Tensor, keep_prefix: usize,
) -> Result<()> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "overwrite_pad_tokens_inplace: expected [N, D], got {:?}", shape)).into());
    }
    let (total, d) = (shape[0], shape[1]);
    if keep_prefix > total {
        return Err(Error::InvalidArgument(format!(
            "overwrite_pad_tokens_inplace: keep_prefix ({}) > total ({})",
            keep_prefix, total)).into());
    }
    if pad_token.numel() != d {
        return Err(Error::InvalidArgument(format!(
            "overwrite_pad_tokens_inplace: pad_token has {} elems, expected {}",
            pad_token.numel(), d)).into());
    }
    let pad_rows = total - keep_prefix;
    if pad_rows == 0 { return Ok(()); }

    let bytes_per_row = d * x.dtype().size_in_bytes();

    match x.device() {
        DeviceType::Cpu => unsafe {
            let pad_ptr = pad_token.data_ptr();
            let base = x.data_ptr_mut();
            for r in keep_prefix..total {
                std::ptr::copy_nonoverlapping(
                    pad_ptr,
                    base.add(r * bytes_per_row),
                    bytes_per_row);
            }
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            match x.dtype() {
                DataType::BF16 => unsafe {
                    let dst_view = x.as_bf16_mut()?.data_ptr_mut().add(keep_prefix * d);
                    let pad_ptr  = pad_token.as_bf16()?.data_ptr();
                    broadcast_row_bf16_forward(
                        dst_view, pad_ptr, pad_rows as i32, d as i32, stream);
                }
                DataType::F32 => unsafe {
                    let dst_view = x.as_f32_mut()?.data_ptr_mut().add(keep_prefix * d);
                    let pad_ptr  = pad_token.as_f32()?.data_ptr();
                    broadcast_row_f32_forward(
                        dst_view, pad_ptr, pad_rows as i32, d as i32, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "overwrite_pad_tokens_inplace CUDA: unsupported dtype {:?}", other
                )).into()),
            }
        }
    }
    Ok(())
}

// ─────────────────────── tests ───────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    fn f32_tensor(shape: &[usize], data: &[f32]) -> Result<Tensor> {
        let mut t = Tensor::empty(shape, DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(data);
        Ok(t)
    }

    #[test]
    fn pad_last_row_cpu() -> Result<()> {
        let src = f32_tensor(&[3, 4], &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ])?;
        let dst = pad_last_row(&src, 5)?;
        assert_eq!(dst.shape(), &[5, 4]);
        // Rows 3 and 4 should both be copies of row 2.
        let d = dst.as_f32()?.as_slice()?;
        assert_eq!(&d[12..16], &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(&d[16..20], &[9.0, 10.0, 11.0, 12.0]);
        Ok(())
    }

    #[test]
    fn pad_with_token_cpu() -> Result<()> {
        let src = f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let pad = f32_tensor(&[3], &[-1.0, -2.0, -3.0])?;
        let dst = pad_with_token(&src, &pad, 4)?;
        assert_eq!(dst.shape(), &[4, 3]);
        let d = dst.as_f32()?.as_slice()?;
        assert_eq!(&d[0..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&d[6..9], &[-1.0, -2.0, -3.0]);
        assert_eq!(&d[9..12], &[-1.0, -2.0, -3.0]);
        Ok(())
    }

    #[test]
    fn overwrite_pad_tokens_cpu() -> Result<()> {
        let mut x = f32_tensor(&[4, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ])?;
        let pad = f32_tensor(&[3], &[-1.0, -1.0, -1.0])?;
        overwrite_pad_tokens_inplace(&mut x, &pad, 2)?;
        let d = x.as_f32()?.as_slice()?;
        assert_eq!(&d[0..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&d[6..9], &[-1.0, -1.0, -1.0]);
        assert_eq!(&d[9..12], &[-1.0, -1.0, -1.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn pad_last_row_cuda_matches_cpu() -> Result<()> {
        let src = f32_tensor(&[3, 4], &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ])?;
        let cpu_out = pad_last_row(&src, 6)?;

        let src_g = src.to_cuda(0)?;
        let gpu_out = pad_last_row(&src_g, 6)?.to_cpu()?;
        assert_eq!(cpu_out.as_f32()?.as_slice()?, gpu_out.as_f32()?.as_slice()?);
        Ok(())
    }
}
