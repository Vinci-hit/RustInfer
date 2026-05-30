//! Broadcast multiply CUDA kernel wrapper.
//! broadcast_mul: dst[i] = a[i] * b[i % D]  (b is [D], a is [rows, D])

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn broadcast_mul_f32_forward(dst: *mut f32, a: *const f32, b: *const f32, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_mul_bf16_forward(dst: *mut half::bf16, a: *const half::bf16, b: *const half::bf16, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_mul_f16_forward(dst: *mut half::f16, a: *const half::f16, b: *const half::f16, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_add_inplace_f32_forward(a: *mut f32, b: *const f32, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_add_inplace_bf16_forward(a: *mut half::bf16, b: *const half::bf16, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_add_inplace_f16_forward(a: *mut half::f16, b: *const half::f16, rows: i32, d: i32, stream: cudaStream_t);
}

/// In-place broadcast multiply: x[i,j] *= scale[j].
pub fn broadcast_mul_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>, scale: &Tensor<T, Cuda>) -> OpResult<()> {
    let dim = scale.numel() as i32;
    let rows = (x.numel() as i32) / dim;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => broadcast_mul_f32_forward(x.data_ptr_mut() as _, x.data_ptr() as _, scale.data_ptr() as _, rows, dim, stream),
            DataType::BF16 => broadcast_mul_bf16_forward(x.data_ptr_mut() as _, x.data_ptr() as _, scale.data_ptr() as _, rows, dim, stream),
            DataType::F16 => broadcast_mul_f16_forward(x.data_ptr_mut() as _, x.data_ptr() as _, scale.data_ptr() as _, rows, dim, stream),
            _ => return Err(OpError::Kernel(format!("broadcast_mul: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

/// In-place broadcast add: x[i,j] += bias[j].
pub fn broadcast_add_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>, bias: &Tensor<T, Cuda>) -> OpResult<()> {
    let dim = bias.numel() as i32;
    let rows = (x.numel() as i32) / dim;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => broadcast_add_inplace_f32_forward(x.data_ptr_mut() as _, bias.data_ptr() as _, rows, dim, stream),
            DataType::BF16 => broadcast_add_inplace_bf16_forward(x.data_ptr_mut() as _, bias.data_ptr() as _, rows, dim, stream),
            DataType::F16 => broadcast_add_inplace_f16_forward(x.data_ptr_mut() as _, bias.data_ptr() as _, rows, dim, stream),
            _ => return Err(OpError::Kernel(format!("broadcast_add: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn broadcast_mul_inplace_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let rows = 3usize;
        let dim = 4usize;
        let x_host: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        let scale_host: Vec<f32> = vec![1.0, 0.5, 2.0, -1.0];
        let mut x: Tensor<f32, Cuda> = Tensor::from_host_slice(&x_host, [rows, dim], &cuda).unwrap();
        let scale: Tensor<f32, Cuda> = Tensor::from_host_slice(&scale_host, [dim], &cuda).unwrap();
        broadcast_mul_inplace(&mut x, &scale).unwrap();
        let got = x.to_host_vec().unwrap();
        for r in 0..rows {
            for c in 0..dim {
                let expected = x_host[r * dim + c] * scale_host[c];
                assert!((got[r * dim + c] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn broadcast_add_inplace_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let rows = 3usize;
        let dim = 4usize;
        let x_host: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        let bias_host: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let mut x: Tensor<f32, Cuda> = Tensor::from_host_slice(&x_host, [rows, dim], &cuda).unwrap();
        let bias: Tensor<f32, Cuda> = Tensor::from_host_slice(&bias_host, [dim], &cuda).unwrap();
        broadcast_add_inplace(&mut x, &bias).unwrap();
        let got = x.to_host_vec().unwrap();
        for r in 0..rows {
            for c in 0..dim {
                let expected = x_host[r * dim + c] + bias_host[c];
                assert!((got[r * dim + c] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn broadcast_mul_inplace_bf16_basic() {
        let cuda = Cuda::new(0).unwrap();
        let rows = 4usize;
        let dim = 8usize;
        let x_host: Vec<bf16> = (0..rows * dim).map(|i| bf16::from_f32(i as f32 * 0.1)).collect();
        let scale_host: Vec<bf16> = (0..dim).map(|i| bf16::from_f32((i + 1) as f32 * 0.5)).collect();
        let mut x: Tensor<bf16, Cuda> = Tensor::from_host_slice(&x_host, [rows, dim], &cuda).unwrap();
        let scale: Tensor<bf16, Cuda> = Tensor::from_host_slice(&scale_host, [dim], &cuda).unwrap();
        broadcast_mul_inplace(&mut x, &scale).unwrap();
        let got: Vec<f32> = x.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        for r in 0..rows {
            for c in 0..dim {
                let expected = x_host[r * dim + c].to_f32() * scale_host[c].to_f32();
                let got_v = got[r * dim + c];
                let abs = (got_v - expected).abs();
                let rel = abs / expected.abs().max(1e-3);
                assert!(abs < 0.05 || rel < 0.02,
                    "[r={},c={}] got={} expected={}", r, c, got_v, expected);
            }
        }
    }
}
