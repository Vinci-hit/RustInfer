//! Scalar ops CUDA kernel wrappers (scalar mul/add, silu/tanh, device-scalar variants).
//!
//! Dispatch is an attribute of the element type: [`ScalarKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry points, so the wrappers below are generic with no runtime `match`.
//! Adding a dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use half::{bf16, f16};
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

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

/// Element types with the scalar-op CUDA kernels. Each method forwards to this
/// dtype's `extern` entry; the wrappers below are generic over this trait, so
/// the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `n` elements on
/// `stream` (and `d_val` a valid `[1] f32` device pointer for
/// [`scalar_mul_inplace_from_dev`]); this just names the FFI entries and
/// performs no checks.
pub trait ScalarKernel: CudaFloat {
    /// `dst = src * val`, elementwise over `n` elements.
    unsafe fn scalar_mul(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t);
    /// `dst = src + val`, elementwise over `n` elements.
    unsafe fn scalar_add(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t);
    /// `data = silu(data)`, in place over `n` elements.
    unsafe fn silu_inplace(data: *mut Self, n: i32, stream: cudaStream_t);
    /// `data = tanh(data)`, in place over `n` elements.
    unsafe fn tanh_inplace(data: *mut Self, n: i32, stream: cudaStream_t);
    /// `x *= *d_val`, reading the scalar from device memory at replay time.
    unsafe fn scalar_mul_inplace_from_dev(
        x: *mut Self,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    );
}

impl ScalarKernel for f32 {
    #[inline]
    unsafe fn scalar_mul(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t) {
        unsafe { scalar_mul_f32_forward(dst, src, val, n, stream) }
    }
    #[inline]
    unsafe fn scalar_add(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t) {
        unsafe { scalar_add_f32_forward(dst, src, val, n, stream) }
    }
    #[inline]
    unsafe fn silu_inplace(data: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { silu_inplace_f32_forward(data, n, stream) }
    }
    #[inline]
    unsafe fn tanh_inplace(data: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { tanh_inplace_f32_forward(data, n, stream) }
    }
    #[inline]
    unsafe fn scalar_mul_inplace_from_dev(
        x: *mut Self,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    ) {
        unsafe { scalar_mul_inplace_from_dev_f32_forward(x, d_val, n, stream) }
    }
}

impl ScalarKernel for bf16 {
    #[inline]
    unsafe fn scalar_mul(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t) {
        unsafe { scalar_mul_bf16_forward(dst, src, val, n, stream) }
    }
    #[inline]
    unsafe fn scalar_add(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t) {
        unsafe { scalar_add_bf16_forward(dst, src, val, n, stream) }
    }
    #[inline]
    unsafe fn silu_inplace(data: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { silu_inplace_bf16_forward(data, n, stream) }
    }
    #[inline]
    unsafe fn tanh_inplace(data: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { tanh_inplace_bf16_forward(data, n, stream) }
    }
    #[inline]
    unsafe fn scalar_mul_inplace_from_dev(
        x: *mut Self,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    ) {
        unsafe { scalar_mul_inplace_from_dev_bf16_forward(x, d_val, n, stream) }
    }
}

impl ScalarKernel for f16 {
    #[inline]
    unsafe fn scalar_mul(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t) {
        unsafe { scalar_mul_f16_forward(dst, src, val, n, stream) }
    }
    #[inline]
    unsafe fn scalar_add(dst: *mut Self, src: *const Self, val: f32, n: i32, stream: cudaStream_t) {
        unsafe { scalar_add_f16_forward(dst, src, val, n, stream) }
    }
    #[inline]
    unsafe fn silu_inplace(data: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { silu_inplace_f16_forward(data, n, stream) }
    }
    #[inline]
    unsafe fn tanh_inplace(data: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { tanh_inplace_f16_forward(data, n, stream) }
    }
    #[inline]
    unsafe fn scalar_mul_inplace_from_dev(
        x: *mut Self,
        d_val: *const f32,
        n: i32,
        stream: cudaStream_t,
    ) {
        unsafe { scalar_mul_inplace_from_dev_f16_forward(x, d_val, n, stream) }
    }
}

/// In-place scalar multiply: `x *= val`. Implemented as `dst=src,val` with
/// `dst == src` aliased to the same buffer.
pub fn scalar_mul_inplace<T: ScalarKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    scalar: f64,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let val = scalar as f32;
    let p = x.data_ptr_mut();
    unsafe {
        T::scalar_mul(p, p, val, n, stream);
    }
    Ok(())
}

/// In-place scalar add: `x += val`.
pub fn scalar_add_inplace<T: ScalarKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    scalar: f64,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let val = scalar as f32;
    let p = x.data_ptr_mut();
    unsafe {
        T::scalar_add(p, p, val, n, stream);
    }
    Ok(())
}

/// In-place SiLU activation: `x = x * sigmoid(x)`.
pub fn silu_inplace<T: ScalarKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let p = x.data_ptr_mut();
    unsafe {
        T::silu_inplace(p, n, stream);
    }
    Ok(())
}

/// In-place tanh activation.
pub fn tanh_inplace<T: ScalarKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let p = x.data_ptr_mut();
    unsafe {
        T::tanh_inplace(p, n, stream);
    }
    Ok(())
}

/// CUDA-Graph-friendly scalar mul: scalar lives in device memory at `d_val`
/// (an `[1] f32` tensor). Reads the byte at replay time, so the host can
/// rewrite the byte between graph launches without re-capturing.
pub fn scalar_mul_inplace_from_dev<T: ScalarKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    d_val: &Tensor<f32, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    let p = x.data_ptr_mut();
    let dv = d_val.data_ptr();
    unsafe {
        T::scalar_mul_inplace_from_dev(p, dv, n, stream);
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
        let host: Vec<bf16> = [1.0, 2.0, -3.0, 4.5]
            .into_iter()
            .map(bf16::from_f32)
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
