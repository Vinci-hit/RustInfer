//! Sequence concatenation: `[S_a, D] + [S_b, D] → [S_a+S_b, D]` along dim 0.
//!
//! Implemented as two stream-ordered D2D memcpy.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::tensor::Tensor;
use crate::domain::types::Dtype;
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::{cudaMemcpyAsync, cudaMemcpyKind, cudaError_cudaSuccess};

/// In-place concat: `dst = [a; b]` along dim 0.
///
/// `a`, `b` must be 2D with matching last dim and dtype; `dst` must be
/// `[a.shape()[0] + b.shape()[0], a.shape()[1]]`.
pub fn concat_seq_into<T: Dtype>(
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let as_ = a.shape().as_slice();
    let bs_ = b.shape().as_slice();
    let ds_ = dst.shape().as_slice();
    if as_.len() != 2 || bs_.len() != 2 || ds_.len() != 2 {
        return Err(OpError::Shape(format!(
            "concat_seq: 2D required, got a={:?} b={:?} dst={:?}", as_, bs_, ds_,
        )));
    }
    if as_[1] != bs_[1] || ds_[1] != as_[1] {
        return Err(OpError::Shape(format!(
            "concat_seq: last-dim mismatch a={} b={} dst={}", as_[1], bs_[1], ds_[1],
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
    let stream = a.device().config.stream;
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

#[cfg(test)]
mod tests {
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
        concat_seq_into(&a, &b, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        for i in 0..s_a * d { assert_eq!(got[i], a_host[i]); }
        for i in 0..s_b * d { assert_eq!(got[s_a * d + i], b_host[i]); }
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
        concat_seq_into(&a, &b, &mut dst).unwrap();
        let got: Vec<f32> = dst.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        for i in 0..2 * d { assert_eq!(got[i], a_host[i].to_f32()); }
        for i in 0..3 * d { assert_eq!(got[2 * d + i], b_host[i].to_f32()); }
    }

    #[test]
    fn concat_seq_shape_mismatch_errors() {
        let cuda = Cuda::new(0).unwrap();
        let a: Tensor<f32, Cuda> = Tensor::zeros([2, 4], &cuda).unwrap();
        let b: Tensor<f32, Cuda> = Tensor::zeros([3, 5], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([5, 4], &cuda).unwrap();
        let err = concat_seq_into(&a, &b, &mut dst).unwrap_err();
        match err { OpError::Shape(_) => {}, other => panic!("got {:?}", other) }
    }
}
