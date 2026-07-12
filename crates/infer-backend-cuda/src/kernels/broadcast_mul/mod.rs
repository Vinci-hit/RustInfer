//! Broadcast multiply CUDA kernel wrapper.
//! broadcast_mul: dst[i] = a[i] * b[i % D]  (b is [D], a is [rows, D])
//!
//! Dispatch is an attribute of the element type: [`BroadcastMulKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry points, so [`broadcast_mul_inplace`]/[`broadcast_add_inplace`] are
//! generic with no runtime `match`. Adding a dtype is one `impl`; an
//! unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn broadcast_mul_f32_forward(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_mul_bf16_forward(
        dst: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_mul_f16_forward(
        dst: *mut half::f16,
        a: *const half::f16,
        b: *const half::f16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_f32_forward(
        a: *mut f32,
        b: *const f32,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_bf16_forward(
        a: *mut half::bf16,
        b: *const half::bf16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_f16_forward(
        a: *mut half::f16,
        b: *const half::f16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
}

/// Element types with broadcast multiply/add CUDA kernels. The two methods
/// forward to this dtype's `extern` entries; the wrappers below are generic
/// over this trait, so the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `rows * d` elements
/// on `stream`; this just names the FFI entries and performs no checks.
pub trait BroadcastMulKernel: CudaFloat {
    /// `dst = a * b` with `b` (`[d]`) broadcast over `rows`.
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    /// `a += b` with `b` (`[d]`) broadcast over `rows`.
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
}

impl BroadcastMulKernel for f32 {
    #[inline]
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_mul_f32_forward(dst, a, b, rows, d, stream) }
    }
    #[inline]
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_add_inplace_f32_forward(a, b, rows, d, stream) }
    }
}

impl BroadcastMulKernel for half::bf16 {
    #[inline]
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_mul_bf16_forward(dst, a, b, rows, d, stream) }
    }
    #[inline]
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_add_inplace_bf16_forward(a, b, rows, d, stream) }
    }
}

impl BroadcastMulKernel for half::f16 {
    #[inline]
    unsafe fn broadcast_mul(
        dst: *mut Self,
        a: *const Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_mul_f16_forward(dst, a, b, rows, d, stream) }
    }
    #[inline]
    unsafe fn broadcast_add_inplace(
        a: *mut Self,
        b: *const Self,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    ) {
        unsafe { broadcast_add_inplace_f16_forward(a, b, rows, d, stream) }
    }
}

/// In-place broadcast multiply: x[i,j] *= scale[j].
pub fn broadcast_mul_inplace<T: BroadcastMulKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    scale: &Tensor<T, Cuda>,
) -> OpResult<()> {
    let dim = scale.numel();
    if dim == 0 || !x.numel().is_multiple_of(dim) {
        return Err(OpError::Shape(format!(
            "broadcast_mul_inplace: x.numel()={} not a multiple of scale.numel()={}",
            x.numel(),
            dim
        )));
    }
    let rows = (x.numel() / dim) as i32;
    let dim = dim as i32;
    unsafe {
        T::broadcast_mul(
            x.data_ptr_mut(),
            x.data_ptr(),
            scale.data_ptr(),
            rows,
            dim,
            stream,
        );
    }
    Ok(())
}

/// In-place broadcast add: x[i,j] += bias[j].
pub fn broadcast_add_inplace<T: BroadcastMulKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    bias: &Tensor<T, Cuda>,
) -> OpResult<()> {
    let dim = bias.numel();
    if dim == 0 || !x.numel().is_multiple_of(dim) {
        return Err(OpError::Shape(format!(
            "broadcast_add_inplace: x.numel()={} not a multiple of bias.numel()={}",
            x.numel(),
            dim
        )));
    }
    let rows = (x.numel() / dim) as i32;
    let dim = dim as i32;
    unsafe {
        T::broadcast_add_inplace(x.data_ptr_mut(), bias.data_ptr(), rows, dim, stream);
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
        let mut x: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&x_host, [rows, dim], &cuda).unwrap();
        let scale: Tensor<f32, Cuda> = Tensor::from_host_slice(&scale_host, [dim], &cuda).unwrap();
        broadcast_mul_inplace(cuda.config.stream, &mut x, &scale).unwrap();
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
        let mut x: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&x_host, [rows, dim], &cuda).unwrap();
        let bias: Tensor<f32, Cuda> = Tensor::from_host_slice(&bias_host, [dim], &cuda).unwrap();
        broadcast_add_inplace(cuda.config.stream, &mut x, &bias).unwrap();
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
        let x_host: Vec<bf16> = (0..rows * dim)
            .map(|i| bf16::from_f32(i as f32 * 0.1))
            .collect();
        let scale_host: Vec<bf16> = (0..dim)
            .map(|i| bf16::from_f32((i + 1) as f32 * 0.5))
            .collect();
        let mut x: Tensor<bf16, Cuda> =
            Tensor::from_host_slice(&x_host, [rows, dim], &cuda).unwrap();
        let scale: Tensor<bf16, Cuda> = Tensor::from_host_slice(&scale_host, [dim], &cuda).unwrap();
        broadcast_mul_inplace(cuda.config.stream, &mut x, &scale).unwrap();
        let got: Vec<f32> = x
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        for r in 0..rows {
            for c in 0..dim {
                let expected = x_host[r * dim + c].to_f32() * scale_host[c].to_f32();
                let got_v = got[r * dim + c];
                let abs = (got_v - expected).abs();
                let rel = abs / expected.abs().max(1e-3);
                assert!(
                    abs < 0.05 || rel < 0.02,
                    "[r={},c={}] got={} expected={}",
                    r,
                    c,
                    got_v,
                    expected
                );
            }
        }
    }
}
