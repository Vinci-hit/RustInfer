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
    fn affine_layer_norm_forward(
        x: *const std::ffi::c_void,
        w: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        dtype: i32,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );

    fn gelu_forward(x: *mut std::ffi::c_void, dtype: i32, n: i32, tanh: bool, stream: cudaStream_t);
    fn rotary_angles_forward(
        x: *mut std::ffi::c_void,
        dtype: i32,
        sin: *const f32,
        cos: *const f32,
        rows: i32,
        heads: i32,
        dim: i32,
        half_dim: i32,
        row_stride: i64,
        stream: cudaStream_t,
    );
    fn zero_norm_f16_forward(
        input: *const f16,
        weight: *const f16,
        output: *mut f16,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    );

    fn zero_norm_bf16_forward(
        input: *const bf16,
        weight: *const bf16,
        output: *mut bf16,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    );

    fn zero_norm_f32_forward(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    );

    fn sigmoid_mul_f32_forward(output: *mut f32, gate: *const f32, n: i32, stream: cudaStream_t);
    fn sigmoid_mul_bf16_forward(output: *mut bf16, gate: *const bf16, n: i32, stream: cudaStream_t);
    fn sigmoid_mul_f16_forward(output: *mut f16, gate: *const f16, n: i32, stream: cudaStream_t);
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

fn float_code<T: infer_core::dtype::Dtype>() -> OpResult<i32> {
    use infer_core::types::DTypeId;
    match T::ID {
        DTypeId::F32 => Ok(0),
        DTypeId::F16 => Ok(1),
        DTypeId::BF16 => Ok(2),
        _ => Err(infer_core::ports::OpError::unsupported(
            "CUDA",
            "vision scalar dtype",
        )),
    }
}

pub fn gelu<T: infer_core::dtype::Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    tanh: bool,
) -> OpResult<()> {
    use infer_core::ports::OpError;
    if !x.is_contiguous() {
        return Err(OpError::Shape("gelu: non-contiguous input".into()));
    }
    let n = i32::try_from(x.numel()).map_err(|_| OpError::Shape("gelu: size overflow".into()))?;
    let dtype = float_code::<T>()?;
    if n > 0 {
        unsafe {
            gelu_forward(x.data_ptr_mut().cast(), dtype, n, tanh, stream);
        }
    }
    Ok(())
}

pub fn rope_with_angles<T: infer_core::dtype::Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    sin: &Tensor<f32, Cuda>,
    cos: &Tensor<f32, Cuda>,
    dim: usize,
) -> OpResult<()> {
    use infer_core::ports::OpError;
    if x.shape().len() != 2
        || sin.shape().len() != 2
        || sin.shape() != cos.shape()
        || dim == 0
        || x.shape()[1] % dim != 0
        || sin.shape()[0] != x.shape()[0]
        || sin.shape()[1] == 0
        || sin.shape()[1] * 2 > dim
        || x.strides()[1] != 1
        || !sin.is_contiguous()
        || !cos.is_contiguous()
    {
        return Err(OpError::Shape(
            "rope_with_angles: invalid shapes/strides".into(),
        ));
    }
    let cv = |n| i32::try_from(n).map_err(|_| OpError::Shape("rotary size overflow".into()));
    let dtype = float_code::<T>()?;
    if x.shape()[0] > 0 {
        unsafe {
            rotary_angles_forward(
                x.data_ptr_mut().cast(),
                dtype,
                sin.data_ptr(),
                cos.data_ptr(),
                cv(x.shape()[0])?,
                cv(x.shape()[1] / dim)?,
                cv(dim)?,
                cv(sin.shape()[1])?,
                x.strides()[0] as i64,
                stream,
            );
        }
    }
    Ok(())
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
    unsafe fn zero_norm(
        input: *const Self,
        weight: *const Self,
        output: *mut Self,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    /// `dst = src * val`, elementwise over `n` elements.
    unsafe fn sigmoid_mul(output: *mut Self, gate: *const Self, n: i32, stream: cudaStream_t);
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
    unsafe fn zero_norm(
        input: *const Self,
        weight: *const Self,
        output: *mut Self,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            zero_norm_f32_forward(
                input,
                weight,
                output,
                rows,
                heads,
                dim,
                input_stride,
                output_stride,
                eps,
                stream,
            )
        }
    }
    unsafe fn sigmoid_mul(output: *mut Self, gate: *const Self, n: i32, stream: cudaStream_t) {
        unsafe { sigmoid_mul_f32_forward(output, gate, n, stream) }
    }

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
    unsafe fn zero_norm(
        input: *const Self,
        weight: *const Self,
        output: *mut Self,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            zero_norm_bf16_forward(
                input,
                weight,
                output,
                rows,
                heads,
                dim,
                input_stride,
                output_stride,
                eps,
                stream,
            )
        }
    }
    unsafe fn sigmoid_mul(output: *mut Self, gate: *const Self, n: i32, stream: cudaStream_t) {
        unsafe { sigmoid_mul_bf16_forward(output, gate, n, stream) }
    }

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
    unsafe fn zero_norm(
        input: *const Self,
        weight: *const Self,
        output: *mut Self,
        rows: i32,
        heads: i32,
        dim: i32,
        input_stride: i32,
        output_stride: i32,
        eps: f32,
        stream: cudaStream_t,
    ) {
        unsafe {
            zero_norm_f16_forward(
                input,
                weight,
                output,
                rows,
                heads,
                dim,
                input_stride,
                output_stride,
                eps,
                stream,
            )
        }
    }
    unsafe fn sigmoid_mul(output: *mut Self, gate: *const Self, n: i32, stream: cudaStream_t) {
        unsafe { sigmoid_mul_f16_forward(output, gate, n, stream) }
    }

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

pub fn sigmoid_mul<T: ScalarKernel>(
    stream: cudaStream_t,
    output: &mut Tensor<T, Cuda>,
    gate: &Tensor<T, Cuda>,
) -> OpResult<()> {
    use infer_core::ports::OpError;
    if output.shape() != gate.shape() || !output.is_contiguous() || !gate.is_contiguous() {
        return Err(OpError::Shape(
            "sigmoid_mul: expected matching contiguous tensors".into(),
        ));
    }
    let n = i32::try_from(output.numel())
        .map_err(|_| OpError::Shape("sigmoid_mul: size exceeds i32".into()))?;
    if n > 0 {
        unsafe {
            T::sigmoid_mul(output.data_ptr_mut(), gate.data_ptr(), n, stream);
        }
    }
    Ok(())
}

#[cfg(test)]
mod sigmoid_tests {
    use super::*;
    use crate::CudaScope;
    use infer_core::dtype::Dtype;
    use infer_core::ports::FusedOps;

    fn check<T: Dtype>() {
        let cuda = Cuda::new(0).unwrap();
        let scope = CudaScope::new(cuda.clone());
        // More than one block, with a tail and saturating gates.
        let gates: Vec<T> = (0..513)
            .map(|i| T::write_f64((i as f64 - 256.0) / 3.0))
            .collect();
        let values: Vec<T> = (0..513)
            .map(|i| T::write_f64((i as f64 * 0.13).sin()))
            .collect();
        let expected: Vec<f32> = values
            .iter()
            .zip(&gates)
            .map(|(v, g)| {
                let g = T::read_f64(g) as f32;
                let sigmoid = T::write_f64((1.0 / (1.0 + (-g).exp())) as f64);
                let result =
                    T::write_f64(((T::read_f64(v) as f32) * (T::read_f64(&sigmoid) as f32)) as f64);
                T::read_f64(&result) as f32
            })
            .collect();
        let gate = Tensor::from_host_slice(&gates, [513], &cuda).unwrap();
        let mut output = Tensor::from_host_slice(&values, [513], &cuda).unwrap();
        <Cuda as FusedOps>::sigmoid_mul(&scope, &mut output, &gate).unwrap();
        for (got, expected) in output.to_host_vec().unwrap().iter().zip(expected) {
            let got = T::read_f64(got) as f32;
            assert!(
                (got - expected).abs() < 1e-5,
                "got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn sigmoid_mul_matches_activation_dtype_rounding() {
        check::<f32>();
        check::<bf16>();
        check::<f16>();
    }
}

pub fn rmsnorm_zero_centered<T: ScalarKernel>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    use infer_core::ports::OpError;
    let dim = weight.numel();
    if dim == 0
        || input.shape() != output.shape()
        || input.shape().len() != 2
        || input.shape()[1] % dim != 0
        || input.strides()[1] != 1
        || output.strides()[1] != 1
        || !weight.is_contiguous()
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(OpError::Shape(
            "rmsnorm_zero_centered: invalid shapes, strides or epsilon".into(),
        ));
    }
    let cv = |n: usize| {
        i32::try_from(n)
            .map_err(|_| OpError::Shape("rmsnorm_zero_centered: size exceeds i32".into()))
    };
    let rows = cv(input.shape()[0])?;
    let heads = cv(input.shape()[1] / dim)?;
    if rows > 0 {
        unsafe {
            T::zero_norm(
                input.data_ptr(),
                weight.data_ptr(),
                output.data_ptr_mut(),
                rows,
                heads,
                cv(dim)?,
                cv(input.strides()[0])?,
                cv(output.strides()[0])?,
                eps,
                stream,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod zero_norm_tests {
    use super::*;
    use crate::CudaScope;
    use infer_core::ports::FusedOps;

    #[test]
    fn zero_centered_norm_keeps_scale_in_fp32_and_handles_strided_heads() {
        let cuda = Cuda::new(0).unwrap();
        let scope = CudaScope::new(cuda.clone());
        let values: Vec<bf16> = (0..24)
            .map(|i| bf16::from_f32((i as f32 - 9.0) * 0.25))
            .collect();
        let tensor = Tensor::from_host_slice(&values, [2, 12], &cuda).unwrap();
        let mut view = tensor.narrow(1, 2, 8).unwrap();
        let input = view.clone();
        let weights: Vec<bf16> = [0.001953125, -0.00390625, 0.125, -0.25]
            .map(bf16::from_f32)
            .to_vec();
        let weight = Tensor::from_host_slice(&weights, [4], &cuda).unwrap();
        <Cuda as FusedOps>::rmsnorm_zero_centered(&scope, &input, &weight, &mut view, 1e-6)
            .unwrap();
        let actual = tensor.to_host_vec().unwrap();
        for row in 0..2 {
            for i in [0, 1, 10, 11] {
                assert_eq!(actual[row * 12 + i], values[row * 12 + i]);
            }
            for head in 0..2 {
                let offset = row * 12 + 2 + head * 4;
                let x = &values[offset..offset + 4];
                let inv = (x.iter().map(|v| v.to_f32().powi(2)).sum::<f32>() / 4.0 + 1e-6)
                    .sqrt()
                    .recip();
                for i in 0..4 {
                    let expected =
                        bf16::from_f32(x[i].to_f32() * inv * (1.0 + weights[i].to_f32()));
                    assert_eq!(actual[offset + i], expected);
                }
            }
        }
    }
}

pub fn layer_norm<T: infer_core::dtype::Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    bias: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    use infer_core::ports::OpError;
    let shape = input.shape().as_slice();
    if !eps.is_finite()
        || eps <= 0.0
        || shape.len() != 2
        || shape[1] == 0
        || output.shape() != input.shape()
        || weight.shape().as_slice() != [shape[1]]
        || bias.shape() != weight.shape()
        || !input.is_contiguous()
        || !output.is_contiguous()
        || !weight.is_contiguous()
        || !bias.is_contiguous()
    {
        return Err(OpError::Shape("layer_norm shape/stride mismatch".into()));
    }
    let rows =
        i32::try_from(shape[0]).map_err(|_| OpError::Shape("layer_norm rows overflow".into()))?;
    let cols =
        i32::try_from(shape[1]).map_err(|_| OpError::Shape("layer_norm cols overflow".into()))?;
    let dtype = float_code::<T>()?;
    if rows > 0 {
        unsafe {
            affine_layer_norm_forward(
                input.data_ptr().cast(),
                weight.data_ptr().cast(),
                bias.data_ptr().cast(),
                output.data_ptr_mut().cast(),
                dtype,
                rows,
                cols,
                eps,
                stream,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod vision_tests {
    use super::*;
    use infer_core::ports::{FusedOps, MathOps};
    #[test]
    fn affine_norm_linear_and_partial_rotary_preserve_bf16_semantics() {
        let cuda = Cuda::new(0).unwrap();
        let scope = cuda.scope();
        let values: Vec<_> = (0..32)
            .map(|i| bf16::from_f32((i as f32 - 12.0) * 0.125))
            .collect();
        let input = Tensor::from_host_slice(&values, [2, 16], &cuda).unwrap();
        let weights: Vec<_> = (0..16)
            .map(|i| bf16::from_f32(0.7 + i as f32 * 0.1))
            .collect();
        let biases: Vec<_> = (0..16)
            .map(|i| bf16::from_f32(-0.3 + i as f32 * 0.01))
            .collect();
        let weight = Tensor::from_host_slice(&weights, [16], &cuda).unwrap();
        let bias = Tensor::from_host_slice(&biases, [16], &cuda).unwrap();
        let mut out = Tensor::zeros([2, 16], &cuda).unwrap();
        Cuda::layer_norm(&scope, &input, &weight, &bias, &mut out, 1e-6).unwrap();
        let actual = out.to_host_vec().unwrap();
        for row in 0..2 {
            let x: Vec<f32> = values[row * 16..(row + 1) * 16]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let mean = x.iter().sum::<f32>() / 16.0;
            let inv = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 16.0 + 1e-6)
                .sqrt()
                .recip();
            for j in 0..16 {
                assert_eq!(
                    actual[row * 16 + j],
                    bf16::from_f32((x[j] - mean) * inv * weights[j].to_f32() + biases[j].to_f32())
                );
            }
        }
        let mut view = input.narrow(1, 2, 8).unwrap();
        let sin = Tensor::from_host_slice(&[1.0f32, 0.0, 0.0, 1.0], [2, 2], &cuda).unwrap();
        let cos = Tensor::from_host_slice(&[0.0f32, 1.0, 1.0, 0.0], [2, 2], &cuda).unwrap();
        Cuda::rope_with_angles(&scope, &mut view, &sin, &cos, 8).unwrap();
        let actual = input.to_host_vec().unwrap();
        for row in 0..2 {
            let j = row;
            let base = row * 16 + 2;
            for k in 0..16 {
                let expected = if k == 2 + j {
                    -values[base + j + 2].to_f32()
                } else if k == 4 + j {
                    values[base + j].to_f32()
                } else {
                    values[row * 16 + k].to_f32()
                };
                assert_eq!(actual[row * 16 + k].to_f32(), expected);
            }
        }
        // Dot product 1.00390625 rounds to 1.0 in BF16; adding the bias
        // before that rounding must produce 1.0078125.
        let x = Tensor::from_host_slice(
            &[bf16::from_f32(1.0), bf16::from_f32(0.00390625)],
            [1, 2],
            &cuda,
        )
        .unwrap();
        let w = Tensor::from_host_slice(&[bf16::from_f32(1.0); 2], [1, 2], &cuda).unwrap();
        let b = Tensor::from_host_slice(&[bf16::from_f32(0.001953125)], [1], &cuda).unwrap();
        let mut o = Tensor::zeros([1, 1], &cuda).unwrap();
        Cuda::linear(&scope, &x, &w, &b, &mut o).unwrap();
        assert_eq!(o.to_host_vec().unwrap()[0].to_f32(), 1.0078125);
        let mut g =
            Tensor::from_host_slice(&[bf16::from_f32(-1.0), bf16::from_f32(1.0)], [1, 2], &cuda)
                .unwrap();
        Cuda::gelu_inplace(&scope, &mut g, false).unwrap();
        assert_eq!(
            g.to_host_vec().unwrap(),
            vec![bf16::from_f32(-0.15865525), bf16::from_f32(0.84134475)]
        );
    }
}
