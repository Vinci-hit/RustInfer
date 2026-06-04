//! Dtype casting CUDA wrappers — F32 ↔ BF16, F32 ↔ F16.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;
use half::{bf16, f16};

unsafe extern "C" {
    fn cast_f32_to_bf16_forward(dst: *mut bf16, src: *const f32, n: i32, stream: cudaStream_t);
    fn cast_bf16_to_f32_forward(dst: *mut f32, src: *const bf16, n: i32, stream: cudaStream_t);
    fn cast_f32_to_f16_forward(dst: *mut f16, src: *const f32, n: i32, stream: cudaStream_t);
    fn cast_f16_to_f32_forward(dst: *mut f32, src: *const f16, n: i32, stream: cudaStream_t);
}

/// Cast `src: Tensor<S, Cuda>` into `dst: Tensor<D, Cuda>` of the same shape.
///
/// Identity copy when `S == D` is delegated to `Tensor::copy_from`. Currently
/// supports the F32 ↔ BF16 and F32 ↔ F16 pairs (the only ones Z-Image needs).
pub fn cast_dtype<S: Dtype, D: Dtype>(
    src: &Tensor<S, Cuda>,
    dst: &mut Tensor<D, Cuda>,
) -> OpResult<()> {
    if src.shape() != dst.shape() {
        return Err(OpError::Shape(format!(
            "cast_dtype: shape mismatch src={:?} dst={:?}",
            src.shape(), dst.shape(),
        )));
    }
    if !src.is_contiguous() || !dst.is_contiguous() {
        return Err(OpError::NotContiguous(*src.shape()));
    }
    let n = src.numel() as i32;
    let stream = src.device().config.stream;
    unsafe {
        match (S::DATA_TYPE, D::DATA_TYPE) {
            (DataType::F32, DataType::BF16) => cast_f32_to_bf16_forward(
                dst.data_ptr_mut() as *mut bf16, src.data_ptr() as *const f32, n, stream,
            ),
            (DataType::BF16, DataType::F32) => cast_bf16_to_f32_forward(
                dst.data_ptr_mut() as *mut f32, src.data_ptr() as *const bf16, n, stream,
            ),
            (DataType::F32, DataType::F16) => cast_f32_to_f16_forward(
                dst.data_ptr_mut() as *mut f16, src.data_ptr() as *const f32, n, stream,
            ),
            (DataType::F16, DataType::F32) => cast_f16_to_f32_forward(
                dst.data_ptr_mut() as *mut f32, src.data_ptr() as *const f16, n, stream,
            ),
            (a, b) if a == b => {
                // Same dtype → memcpy via stream-ordered device copy.
                let bytes = src.numel() * S::SIZE_BYTES;
                if bytes > 0 {
                    let code = crate::infrastructure::cuda::ffi::cudaMemcpyAsync(
                        dst.data_ptr_mut() as *mut std::ffi::c_void,
                        src.data_ptr() as *const std::ffi::c_void,
                        bytes,
                        crate::infrastructure::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream,
                    );
                    if code != crate::infrastructure::cuda::ffi::cudaError_cudaSuccess {
                        return Err(OpError::Kernel(format!("cast_dtype memcpy: {:?}", code)));
                    }
                }
            }
            (s, d) => return Err(OpError::Kernel(format!(
                "cast_dtype: unsupported {:?} → {:?}", s, d,
            ))),
        }
    }
    Ok(())
}

/// Allocate a new `Tensor<D, Cuda>` and cast `src` into it.
pub fn cast_dtype_new<S: Dtype, D: Dtype>(
    src: &Tensor<S, Cuda>,
) -> OpResult<Tensor<D, Cuda>> {
    let mut dst: Tensor<D, Cuda> = Tensor::zeros(*src.shape(), src.device())?;
    cast_dtype(src, &mut dst)?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::domain::types::Shape;

    #[test]
    fn cast_f32_to_bf16_roundtrip() {
        let cuda = Cuda::new(0).expect("cuda init");
        let n = 1024usize;
        let src_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&src_host, [n], &cuda).unwrap();

        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([n], &cuda).unwrap();
        cast_dtype(&src, &mut dst).unwrap();
        let got: Vec<f32> = dst.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();

        for (i, (a, b)) in src_host.iter().zip(got.iter()).enumerate() {
            // BF16 has 7 mantissa bits → ~1% relative error.
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-3);
            assert!(abs < 0.05 || rel < 0.01,
                "cast f32→bf16 mismatch at {}: src={} got={} abs={}", i, a, b, abs);
        }
    }

    #[test]
    fn cast_bf16_to_f32_lossless() {
        let cuda = Cuda::new(0).expect("cuda init");
        let n = 256usize;
        let src_host: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 * 0.5)).collect();
        let src: Tensor<bf16, Cuda> = Tensor::from_host_slice(&src_host, [n], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([n], &cuda).unwrap();
        cast_dtype(&src, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        for (i, (a, b)) in src_host.iter().zip(got.iter()).enumerate() {
            assert_eq!(a.to_f32(), *b, "lossless bf16→f32 at {}", i);
        }
    }

    #[test]
    fn cast_dtype_shape_mismatch_errors() {
        let cuda = Cuda::new(0).expect("cuda init");
        let src: Tensor<f32, Cuda> = Tensor::zeros([4, 4], &cuda).unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([4, 5], &cuda).unwrap();
        let err = cast_dtype(&src, &mut dst).unwrap_err();
        match err {
            OpError::Shape(_) => {}
            other => panic!("expected Shape error, got {:?}", other),
        }
    }

    #[test]
    fn cast_dtype_new_allocates_correctly() {
        let cuda = Cuda::new(0).expect("cuda init");
        let src_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&src_host, [4], &cuda).unwrap();
        let dst: Tensor<bf16, Cuda> = cast_dtype_new(&src).unwrap();
        let got: Vec<f32> = dst.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        for (a, b) in src_host.iter().zip(got.iter()) {
            assert!((a - b).abs() < 0.05);
        }
    }
}
