//! Scalar ops CUDA kernel wrappers (scalar mul/add, silu/tanh, device-scalar variants).

use crate::Cuda;
use crate::ffi::cudaStream_t;
use half::{bf16, f16};
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
    // dst = src * val
    fn scalar_mul_f32_forward(
        dst: *mut f32,
        src: *const f32,
        val: f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn scalar_mul_bf16_forward(
        dst: *mut bf16,
        src: *const bf16,
        val: f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn scalar_mul_f16_forward(
        dst: *mut f16,
        src: *const f16,
        val: f32,
        n: i32,
        stream: cudaStream_t,
    );
    // dst = src + val
    fn scalar_add_f32_forward(
        dst: *mut f32,
        src: *const f32,
        val: f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn scalar_add_bf16_forward(
        dst: *mut bf16,
        src: *const bf16,
        val: f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn scalar_add_f16_forward(
        dst: *mut f16,
        src: *const f16,
        val: f32,
        n: i32,
        stream: cudaStream_t,
    );
    // x = silu(x), in-place
    fn silu_inplace_f32_forward(data: *mut f32, n: i32, stream: cudaStream_t);
    fn silu_inplace_bf16_forward(data: *mut bf16, n: i32, stream: cudaStream_t);
    fn silu_inplace_f16_forward(data: *mut f16, n: i32, stream: cudaStream_t);
    // x = tanh(x), in-place
    fn tanh_inplace_f32_forward(data: *mut f32, n: i32, stream: cudaStream_t);
    fn tanh_inplace_bf16_forward(data: *mut bf16, n: i32, stream: cudaStream_t);
    fn tanh_inplace_f16_forward(data: *mut f16, n: i32, stream: cudaStream_t);
    // x *= *d_val (device-side scalar pointer; CUDA Graph friendly)
    fn scalar_mul_inplace_from_dev_f32_forward(
        x: *mut f32,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn scalar_mul_inplace_from_dev_bf16_forward(
        x: *mut bf16,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn scalar_mul_inplace_from_dev_f16_forward(
        x: *mut f16,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    );
}

/// In-place scalar multiply: `x *= val`. Implemented as `dst=src,val` with
/// `dst == src` aliased to the same buffer.
pub fn scalar_mul_inplace<T: Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    scalar: f64,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let val = scalar as f32;
    let p = x.data_ptr_mut();
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => scalar_mul_f32_forward(p as _, p as _, val, n, stream),
            DataType::BF16 => scalar_mul_bf16_forward(p as _, p as _, val, n, stream),
            DataType::F16 => scalar_mul_f16_forward(p as _, p as _, val, n, stream),
            _ => {
                return Err(OpError::Kernel(format!(
                    "scalar_mul_inplace: {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}

/// In-place scalar add: `x += val`.
pub fn scalar_add_inplace<T: Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    scalar: f64,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let val = scalar as f32;
    let p = x.data_ptr_mut();
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => scalar_add_f32_forward(p as _, p as _, val, n, stream),
            DataType::BF16 => scalar_add_bf16_forward(p as _, p as _, val, n, stream),
            DataType::F16 => scalar_add_f16_forward(p as _, p as _, val, n, stream),
            _ => {
                return Err(OpError::Kernel(format!(
                    "scalar_add_inplace: {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}

/// In-place SiLU activation: `x = x * sigmoid(x)`.
pub fn silu_inplace<T: Dtype>(stream: cudaStream_t, x: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    let p = x.data_ptr_mut();
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => silu_inplace_f32_forward(p as _, n, stream),
            DataType::BF16 => silu_inplace_bf16_forward(p as _, n, stream),
            DataType::F16 => silu_inplace_f16_forward(p as _, n, stream),
            _ => return Err(OpError::Kernel(format!("silu_inplace: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

/// In-place tanh activation.
pub fn tanh_inplace<T: Dtype>(stream: cudaStream_t, x: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    let p = x.data_ptr_mut();
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => tanh_inplace_f32_forward(p as _, n, stream),
            DataType::BF16 => tanh_inplace_bf16_forward(p as _, n, stream),
            DataType::F16 => tanh_inplace_f16_forward(p as _, n, stream),
            _ => return Err(OpError::Kernel(format!("tanh_inplace: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

/// CUDA-Graph-friendly scalar mul: scalar lives in device memory at `d_val`
/// (an `[1] f32` tensor). Reads the byte at replay time, so the host can
/// rewrite the byte between graph launches without re-capturing.
pub fn scalar_mul_inplace_from_dev<T: Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    d_val: &Tensor<f32, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let p = x.data_ptr_mut();
    let dv = d_val.data_ptr();
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => scalar_mul_inplace_from_dev_f32_forward(p as _, dv, n, stream),
            DataType::BF16 => scalar_mul_inplace_from_dev_bf16_forward(p as _, dv, n, stream),
            DataType::F16 => scalar_mul_inplace_from_dev_f16_forward(p as _, dv, n, stream),
            _ => {
                return Err(OpError::Kernel(format!(
                    "scalar_mul_inplace_from_dev: {:?}",
                    T::DATA_TYPE
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
    fn scalar_mul_inplace_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let host: Vec<f32> = vec![1.0, 2.0, -3.0, 4.5];
        let mut t: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [4], &cuda).unwrap();
        scalar_mul_inplace(cuda.config.stream, &mut t, 2.5).unwrap();
        let got = t.to_host_vec().unwrap();
        let expected: Vec<f32> = host.iter().map(|x| x * 2.5).collect();
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn scalar_mul_inplace_bf16_basic() {
        let cuda = Cuda::new(0).unwrap();
        let host: Vec<bf16> = vec![1.0, 2.0, -3.0, 4.5]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let mut t: Tensor<bf16, Cuda> = Tensor::from_host_slice(&host, [4], &cuda).unwrap();
        scalar_mul_inplace(cuda.config.stream, &mut t, 2.0).unwrap();
        let got: Vec<f32> = t
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let expected: Vec<f32> = host.iter().map(|x| x.to_f32() * 2.0).collect();
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 0.05);
        }
    }

    #[test]
    fn scalar_add_inplace_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let host: Vec<f32> = vec![1.0, -1.0, 2.5, 0.0];
        let mut t: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [4], &cuda).unwrap();
        scalar_add_inplace(cuda.config.stream, &mut t, 0.5).unwrap();
        let got = t.to_host_vec().unwrap();
        for (a, &b) in host.iter().zip(got.iter()) {
            assert!((a + 0.5 - b).abs() < 1e-5);
        }
    }

    #[test]
    fn silu_inplace_f32_matches_reference() {
        let cuda = Cuda::new(0).unwrap();
        let host: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0];
        let mut t: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [host.len()], &cuda).unwrap();
        silu_inplace(cuda.config.stream, &mut t).unwrap();
        let got = t.to_host_vec().unwrap();
        for (i, &x) in host.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "silu mismatch at {}: x={}, got={}, expected={}",
                i,
                x,
                got[i],
                expected
            );
        }
    }

    #[test]
    fn tanh_inplace_f32_matches_reference() {
        let cuda = Cuda::new(0).unwrap();
        let host: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0];
        let mut t: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [host.len()], &cuda).unwrap();
        tanh_inplace(cuda.config.stream, &mut t).unwrap();
        let got = t.to_host_vec().unwrap();
        for (i, &x) in host.iter().enumerate() {
            let expected = x.tanh();
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "tanh mismatch at {}: x={}, got={}, expected={}",
                i,
                x,
                got[i],
                expected
            );
        }
    }

    #[test]
    fn scalar_mul_from_dev_f32() {
        let cuda = Cuda::new(0).unwrap();
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut t: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [4], &cuda).unwrap();
        let scalar: Tensor<f32, Cuda> = Tensor::from_host_slice(&[3.0_f32], [1], &cuda).unwrap();
        scalar_mul_inplace_from_dev(cuda.config.stream, &mut t, &scalar).unwrap();
        let got = t.to_host_vec().unwrap();
        let expected: Vec<f32> = host.iter().map(|x| x * 3.0).collect();
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
