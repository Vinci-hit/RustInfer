//! Sequence padding utilities used by the Z-Image transformer.
//!
//! - [`pad_with_token_into`]: `dst[..N] = src; dst[N..target] = pad_token`
//! - [`pad_last_row_into`]: `dst[..N] = src; dst[N..target] = src[N-1]`
//! - [`overwrite_pad_tokens_inplace`]: rewrite rows `[keep_prefix..]` with
//!   the broadcast pad token (used after `cap_embedder` so the pad slots
//!   contain the *embedded* token instead of the embedder of pad-id).
//!
//! Implemented via the `broadcast_row_*_forward` and
//! `fill_repeat_last_row_*_forward` kernels in `cast_fill.cu`, plus D2D
//! memcpy for the prefix.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::{
    cudaError_cudaSuccess, cudaMemcpyAsync, cudaMemcpyKind, cudaStream_t,
};
use half::bf16;

unsafe extern "C" {
    fn broadcast_row_bf16_forward(
        dst: *mut bf16,
        row: *const bf16,
        num_rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_row_f32_forward(
        dst: *mut f32,
        row: *const f32,
        num_rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn fill_repeat_last_row_bf16_forward(
        dst: *mut bf16,
        n_src: i32,
        target_len: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn fill_repeat_last_row_f32_forward(
        dst: *mut f32,
        n_src: i32,
        target_len: i32,
        d: i32,
        stream: cudaStream_t,
    );
}

unsafe fn d2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    n: usize,
    stream: cudaStream_t,
) -> OpResult<()> {
    if n == 0 {
        return Ok(());
    }
    let code = unsafe {
        cudaMemcpyAsync(
            dst,
            src,
            n,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream,
        )
    };
    if code != cudaError_cudaSuccess {
        return Err(OpError::Kernel(format!("D2D memcpy: {:?}", code)));
    }
    Ok(())
}

/// `dst[..n] = src; dst[n..target] = pad_token broadcasted`.
pub fn pad_with_token_into<T: Dtype>(
    stream: cudaStream_t,
    src: &Tensor<T, Cuda>,
    pad_token: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let ss = src.shape().as_slice();
    let ds = dst.shape().as_slice();
    if ss.len() != 2 || ds.len() != 2 || ss[1] != ds[1] {
        return Err(OpError::Shape(format!(
            "pad_with_token: src={:?} dst={:?}",
            ss, ds,
        )));
    }
    let (n, d) = (ss[0], ss[1]);
    let target = ds[0];
    if target < n {
        return Err(OpError::Shape(format!(
            "pad_with_token: target {} < n {}",
            target, n
        )));
    }
    if pad_token.numel() != d {
        return Err(OpError::Shape(format!(
            "pad_with_token: pad_token has {} elems, expected {}",
            pad_token.numel(),
            d,
        )));
    }
    let bytes_per_row = d * T::SIZE_BYTES;
    let src_bytes = n * bytes_per_row;

    unsafe {
        d2d(
            dst.data_ptr_mut() as _,
            src.data_ptr() as _,
            src_bytes,
            stream,
        )?;
        if target > n {
            let pad_rows = target - n;
            let dst_pad_base = (dst.data_ptr_mut() as *mut u8).add(src_bytes);
            match T::DATA_TYPE {
                DataType::BF16 => broadcast_row_bf16_forward(
                    dst_pad_base as *mut bf16,
                    pad_token.data_ptr() as *const bf16,
                    pad_rows as i32,
                    d as i32,
                    stream,
                ),
                DataType::F32 => broadcast_row_f32_forward(
                    dst_pad_base as *mut f32,
                    pad_token.data_ptr() as *const f32,
                    pad_rows as i32,
                    d as i32,
                    stream,
                ),
                other => {
                    return Err(OpError::Kernel(format!(
                        "pad_with_token: unsupported dtype {:?}",
                        other,
                    )));
                }
            }
        }
    }
    Ok(())
}

/// `dst[..n] = src; dst[n..target] = src[n-1]` — repeat last row.
pub fn pad_last_row_into<T: Dtype>(
    stream: cudaStream_t,
    src: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let ss = src.shape().as_slice();
    let ds = dst.shape().as_slice();
    if ss.len() != 2 || ds.len() != 2 || ss[1] != ds[1] {
        return Err(OpError::Shape(format!(
            "pad_last_row: src={:?} dst={:?}",
            ss, ds,
        )));
    }
    let (n, d) = (ss[0], ss[1]);
    let target = ds[0];
    if n == 0 {
        return Err(OpError::Shape(
            "pad_last_row: src must have at least one row".into(),
        ));
    }
    if target < n {
        return Err(OpError::Shape(format!(
            "pad_last_row: target {} < n {}",
            target, n
        )));
    }
    let bytes_per_row = d * T::SIZE_BYTES;
    let src_bytes = n * bytes_per_row;
    unsafe {
        d2d(
            dst.data_ptr_mut() as _,
            src.data_ptr() as _,
            src_bytes,
            stream,
        )?;
        if target > n {
            match T::DATA_TYPE {
                DataType::BF16 => fill_repeat_last_row_bf16_forward(
                    dst.data_ptr_mut() as *mut bf16,
                    n as i32,
                    target as i32,
                    d as i32,
                    stream,
                ),
                DataType::F32 => fill_repeat_last_row_f32_forward(
                    dst.data_ptr_mut() as *mut f32,
                    n as i32,
                    target as i32,
                    d as i32,
                    stream,
                ),
                other => {
                    return Err(OpError::Kernel(format!(
                        "pad_last_row: unsupported dtype {:?}",
                        other,
                    )));
                }
            }
        }
    }
    Ok(())
}

/// `dst[keep_prefix..] = pad_token broadcasted`. Rows `[0..keep_prefix)` stay
/// untouched.
pub fn overwrite_pad_tokens_inplace<T: Dtype>(
    stream: cudaStream_t,
    dst: &mut Tensor<T, Cuda>,
    pad_token: &Tensor<T, Cuda>,
    keep_prefix: usize,
) -> OpResult<()> {
    let ds = dst.shape().as_slice();
    if ds.len() != 2 {
        return Err(OpError::Shape(format!("overwrite_pad: dst={:?}", ds)));
    }
    let (target, d) = (ds[0], ds[1]);
    if keep_prefix > target {
        return Err(OpError::Shape(format!(
            "overwrite_pad: keep_prefix {} > target {}",
            keep_prefix, target,
        )));
    }
    if pad_token.numel() != d {
        return Err(OpError::Shape(format!(
            "overwrite_pad: pad_token {} elems, expected {}",
            pad_token.numel(),
            d,
        )));
    }
    if keep_prefix == target {
        return Ok(());
    }
    let pad_rows = target - keep_prefix;
    let bytes_per_row = d * T::SIZE_BYTES;
    unsafe {
        let dst_base = (dst.data_ptr_mut() as *mut u8).add(keep_prefix * bytes_per_row);
        match T::DATA_TYPE {
            DataType::BF16 => broadcast_row_bf16_forward(
                dst_base as *mut bf16,
                pad_token.data_ptr() as *const bf16,
                pad_rows as i32,
                d as i32,
                stream,
            ),
            DataType::F32 => broadcast_row_f32_forward(
                dst_base as *mut f32,
                pad_token.data_ptr() as *const f32,
                pad_rows as i32,
                d as i32,
                stream,
            ),
            other => {
                return Err(OpError::Kernel(format!(
                    "overwrite_pad: unsupported dtype {:?}",
                    other,
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_with_token_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let d = 4usize;
        let n = 2usize;
        let target = 5usize;
        let src_host: Vec<f32> = (0..n * d).map(|i| (i + 1) as f32).collect();
        let pad_host: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&src_host, [n, d], &cuda).unwrap();
        let pad: Tensor<f32, Cuda> = Tensor::from_host_slice(&pad_host, [d], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([target, d], &cuda).unwrap();
        pad_with_token_into(cuda.config.stream, &src, &pad, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        for i in 0..n * d {
            assert_eq!(got[i], src_host[i]);
        }
        for r in n..target {
            for c in 0..d {
                assert_eq!(
                    got[r * d + c],
                    pad_host[c],
                    "row {} col {}: got {}, expected {}",
                    r,
                    c,
                    got[r * d + c],
                    pad_host[c]
                );
            }
        }
    }

    #[test]
    fn pad_last_row_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let d = 3usize;
        let n = 2usize;
        let target = 4usize;
        let src_host: Vec<f32> = vec![1.0, 2.0, 3.0, 9.0, 8.0, 7.0];
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&src_host, [n, d], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([target, d], &cuda).unwrap();
        pad_last_row_into(cuda.config.stream, &src, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        // First two rows = src.
        for i in 0..n * d {
            assert_eq!(got[i], src_host[i]);
        }
        // Last 2 rows = src[n-1] = [9,8,7].
        for r in n..target {
            assert_eq!(&got[r * d..(r + 1) * d], &src_host[(n - 1) * d..n * d]);
        }
    }

    #[test]
    fn overwrite_pad_tokens_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let d = 4usize;
        let target = 5usize;
        let init: Vec<f32> = (0..target * d).map(|i| i as f32).collect();
        let pad_host: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0];
        let mut dst: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&init, [target, d], &cuda).unwrap();
        let pad: Tensor<f32, Cuda> = Tensor::from_host_slice(&pad_host, [d], &cuda).unwrap();
        let keep = 2usize;
        overwrite_pad_tokens_inplace(cuda.config.stream, &mut dst, &pad, keep).unwrap();
        let got = dst.to_host_vec().unwrap();
        // [0..keep) untouched.
        for i in 0..keep * d {
            assert_eq!(got[i], init[i]);
        }
        // [keep..) = pad.
        for r in keep..target {
            for c in 0..d {
                assert_eq!(got[r * d + c], pad_host[c]);
            }
        }
    }

    #[test]
    fn pad_with_token_bf16_basic() {
        use half::bf16;
        let cuda = Cuda::new(0).unwrap();
        let d = 4usize;
        let n = 1usize;
        let target = 3usize;
        let src_host: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let pad_host: Vec<bf16> = vec![-1.0, -2.0, -3.0, -4.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let src: Tensor<bf16, Cuda> = Tensor::from_host_slice(&src_host, [n, d], &cuda).unwrap();
        let pad: Tensor<bf16, Cuda> = Tensor::from_host_slice(&pad_host, [d], &cuda).unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([target, d], &cuda).unwrap();
        pad_with_token_into(cuda.config.stream, &src, &pad, &mut dst).unwrap();
        let got: Vec<f32> = dst
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        for c in 0..d {
            assert_eq!(got[c], src_host[c].to_f32());
        }
        for r in n..target {
            for c in 0..d {
                assert_eq!(got[r * d + c], pad_host[c].to_f32());
            }
        }
    }
}
