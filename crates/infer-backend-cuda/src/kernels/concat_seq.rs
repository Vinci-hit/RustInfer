//! Sequence concatenation: `[S_a, D] + [S_b, D] → [S_a+S_b, D]` along dim 0.
//!
//! Implemented as two stream-ordered D2D memcpy.

use crate::Cuda;
use crate::ffi::{cudaError_cudaSuccess, cudaMemcpyAsync, cudaMemcpyKind, cudaStream_t};
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::Dtype;

/// In-place concat: `dst = [a; b]` along dim 0.
///
/// `a`, `b` must be 2D with matching last dim and dtype; `dst` must be
/// `[a.shape()[0] + b.shape()[0], a.shape()[1]]`.
pub fn concat_seq_into<T: Dtype>(
    stream: cudaStream_t,
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let as_ = a.shape().as_slice();
    let bs_ = b.shape().as_slice();
    let ds_ = dst.shape().as_slice();
    if as_.len() != 2 || bs_.len() != 2 || ds_.len() != 2 {
        return Err(OpError::Shape(format!(
            "concat_seq: 2D required, got a={:?} b={:?} dst={:?}",
            as_, bs_, ds_,
        )));
    }
    if as_[1] != bs_[1] || ds_[1] != as_[1] {
        return Err(OpError::Shape(format!(
            "concat_seq: last-dim mismatch a={} b={} dst={}",
            as_[1], bs_[1], ds_[1],
        )));
    }
    if ds_[0] != as_[0] + bs_[0] {
        return Err(OpError::Shape(format!(
            "concat_seq: dst rows {} != a.rows + b.rows = {} + {}",
            ds_[0], as_[0], bs_[0],
        )));
    }
    let d = as_[1];
    let bytes_per_row = d * T::SIZE_BYTES;
    let a_bytes = as_[0] * bytes_per_row;
    let b_bytes = bs_[0] * bytes_per_row;
    unsafe {
        let dst_base = dst.data_ptr_mut() as *mut std::ffi::c_void;
        if a_bytes > 0 {
            let code = cudaMemcpyAsync(
                dst_base,
                a.data_ptr() as *const std::ffi::c_void,
                a_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            if code != cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("concat_seq a-copy: {:?}", code)));
            }
        }
        if b_bytes > 0 {
            let code = cudaMemcpyAsync(
                (dst_base as *mut u8).add(a_bytes) as *mut std::ffi::c_void,
                b.data_ptr() as *const std::ffi::c_void,
                b_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            if code != cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("concat_seq b-copy: {:?}", code)));
            }
        }
    }
    Ok(())
}

/// Concatenate columns using two pitched, stream-ordered device copies.
pub fn concat_cols_into<T: Dtype>(
    stream: cudaStream_t,
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let sa = a.shape().as_slice();
    let sb = b.shape().as_slice();
    let sd = dst.shape().as_slice();
    if sa.len() != 2
        || sb.len() != 2
        || sd.len() != 2
        || sa[0] != sb[0]
        || sd[0] != sa[0]
        || sa[1].checked_add(sb[1]) != Some(sd[1])
        || !a.is_contiguous()
        || !b.is_contiguous()
        || !dst.is_contiguous()
    {
        return Err(OpError::Shape(
            "concat_cols requires contiguous [N,A], [N,B], [N,A+B]".into(),
        ));
    }
    let (rows, ac, bc, dc) = (sa[0], sa[1], sb[1], sd[1]);
    if rows == 0 || dc == 0 {
        return Ok(());
    }
    let dst_start = dst.data_ptr() as usize;
    let dst_end = dst_start + dst.numel() * T::SIZE_BYTES;
    for src in [a, b] {
        let start = src.data_ptr() as usize;
        let end = start + src.numel() * T::SIZE_BYTES;
        if start < dst_end && dst_start < end {
            return Err(OpError::Shape("concat_cols output overlaps input".into()));
        }
    }
    if a.device().device_id != b.device().device_id
        || a.device().device_id != dst.device().device_id
    {
        return Err(OpError::Shape("concat_cols device mismatch".into()));
    }
    for (src, cols, offset) in [(a, ac, 0), (b, bc, ac)] {
        if cols == 0 {
            continue;
        }
        // SAFETY: validated disjoint contiguous matrices; pitch and extent fit each row.
        let code = unsafe {
            crate::ffi::cudaMemcpy2DAsync(
                dst.data_ptr_mut().add(offset).cast(),
                dc * T::SIZE_BYTES,
                src.data_ptr().cast(),
                cols * T::SIZE_BYTES,
                cols * T::SIZE_BYTES,
                rows,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            )
        };
        if code != cudaError_cudaSuccess {
            return Err(crate::error::classify_sync_error(code, "concat_cols"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn concat_cols_preserves_row_order_and_rejects_aliases() {
        let device = Cuda::new(0).unwrap();
        let a = Tensor::from_host_slice(&[1f32, 2., 3., 4.], [2, 2], &device).unwrap();
        let b = Tensor::from_host_slice(&[10f32, 20.], [2, 1], &device).unwrap();
        let mut dst = Tensor::zeros([2, 3], &device).unwrap();
        concat_cols_into(device.config.stream, &a, &b, &mut dst).unwrap();
        assert_eq!(dst.to_host_vec().unwrap(), [1., 2., 10., 3., 4., 20.]);
        let a = a.narrow(1, 0, 1).unwrap();
        assert!(concat_cols_into(device.config.stream, &a, &b, &mut dst).is_err());
        let a = dst.narrow(0, 0, 1).unwrap();
        let empty = Tensor::zeros([1, 0], &device).unwrap();
        let mut overlap = a.clone();
        assert!(concat_cols_into(device.config.stream, &a, &empty, &mut overlap).is_err());
        let a = Tensor::<f32, _>::zeros([0, 2], &device).unwrap();
        let b = Tensor::zeros([0, 1], &device).unwrap();
        let mut dst = Tensor::zeros([0, 3], &device).unwrap();
        concat_cols_into(device.config.stream, &a, &b, &mut dst).unwrap();
    }

    use super::*;
    use half::bf16;

    #[test]
    fn concat_seq_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let d = 4usize;
        let s_a = 3usize;
        let s_b = 2usize;
        let a_host: Vec<f32> = (0..s_a * d).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..s_b * d).map(|i| 100.0 + i as f32).collect();
        let a: Tensor<f32, Cuda> = Tensor::from_host_slice(&a_host, [s_a, d], &cuda).unwrap();
        let b: Tensor<f32, Cuda> = Tensor::from_host_slice(&b_host, [s_b, d], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([s_a + s_b, d], &cuda).unwrap();
        concat_seq_into(cuda.config.stream, &a, &b, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        for i in 0..s_a * d {
            assert_eq!(got[i], a_host[i]);
        }
        for i in 0..s_b * d {
            assert_eq!(got[s_a * d + i], b_host[i]);
        }
    }

    #[test]
    fn concat_seq_bf16_basic() {
        let cuda = Cuda::new(0).unwrap();
        let d = 8usize;
        let a_host: Vec<bf16> = (0..2 * d).map(|i| bf16::from_f32(i as f32)).collect();
        let b_host: Vec<bf16> = (0..3 * d).map(|i| bf16::from_f32(-(i as f32))).collect();
        let a: Tensor<bf16, Cuda> = Tensor::from_host_slice(&a_host, [2, d], &cuda).unwrap();
        let b: Tensor<bf16, Cuda> = Tensor::from_host_slice(&b_host, [3, d], &cuda).unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([5, d], &cuda).unwrap();
        concat_seq_into(cuda.config.stream, &a, &b, &mut dst).unwrap();
        let got: Vec<f32> = dst
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        for i in 0..2 * d {
            assert_eq!(got[i], a_host[i].to_f32());
        }
        for i in 0..3 * d {
            assert_eq!(got[2 * d + i], b_host[i].to_f32());
        }
    }

    #[test]
    fn concat_seq_shape_mismatch_errors() {
        let cuda = Cuda::new(0).unwrap();
        let a: Tensor<f32, Cuda> = Tensor::zeros([2, 4], &cuda).unwrap();
        let b: Tensor<f32, Cuda> = Tensor::zeros([3, 5], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([5, 4], &cuda).unwrap();
        let err = concat_seq_into(cuda.config.stream, &a, &b, &mut dst).unwrap_err();
        match err {
            OpError::Shape(_) => {}
            other => panic!("got {:?}", other),
        }
    }
}

#[cfg(test)]
mod column_dtype_tests {
    use super::*;
    #[test]
    fn concat_cols_bf16() {
        use half::bf16;
        let device = Cuda::new(0).unwrap();
        let a = Tensor::from_host_slice(&[bf16::from_f32(1.), bf16::from_f32(2.)], [2, 1], &device)
            .unwrap();
        let b = Tensor::from_host_slice(
            &[
                bf16::from_f32(3.),
                bf16::from_f32(4.),
                bf16::from_f32(5.),
                bf16::from_f32(6.),
            ],
            [2, 2],
            &device,
        )
        .unwrap();
        let mut dst = Tensor::zeros([2, 3], &device).unwrap();
        concat_cols_into(device.config.stream, &a, &b, &mut dst).unwrap();
        assert_eq!(
            dst.to_host_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            [1., 3., 4., 2., 5., 6.]
        );
    }
}
