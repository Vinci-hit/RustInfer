//! Softmax CUDA kernel wrapper.
//!
//! Dispatch is an attribute of the element type: [`SoftmaxKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry point, so [`softmax`] is generic with no runtime `match`. Adding a
//! dtype is one `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use half::{bf16, f16};
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn softmax_f32_forward(
        output: *mut f32,
        input: *const f32,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    );
    fn softmax_bf16_forward(
        output: *mut bf16,
        input: *const bf16,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    );
    fn softmax_f16_forward(
        output: *mut f16,
        input: *const f16,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    );
}

/// Element types with a softmax CUDA kernel. The method forwards to this
/// dtype's `extern` entry; the wrapper below is generic over this trait, so the
/// dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `rows * cols`
/// elements on `stream`; this just names the FFI entry and performs no checks.
pub trait SoftmaxKernel: CudaFloat {
    /// Row-wise softmax of `input` into `output` over `rows × cols`.
    unsafe fn softmax(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    );
}

impl SoftmaxKernel for f32 {
    #[inline]
    unsafe fn softmax(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) {
        unsafe { softmax_f32_forward(output, input, rows, cols, stream) }
    }
}

impl SoftmaxKernel for bf16 {
    #[inline]
    unsafe fn softmax(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) {
        unsafe { softmax_bf16_forward(output, input, rows, cols, stream) }
    }
}

impl SoftmaxKernel for f16 {
    #[inline]
    unsafe fn softmax(
        output: *mut Self,
        input: *const Self,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) {
        unsafe { softmax_f16_forward(output, input, rows, cols, stream) }
    }
}

/// Row-wise softmax over the last dimension.
pub fn softmax<T: SoftmaxKernel>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let dim = *input.shape().as_slice().last().unwrap_or(&1);
    let rows = (input.numel() / dim) as i32;
    unsafe {
        T::softmax(
            output.data_ptr_mut(),
            input.data_ptr(),
            rows,
            dim as i32,
            stream,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference: row-wise softmax with subtract-max for stability.
    fn softmax_ref(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0;
            let exps: Vec<f32> = row
                .iter()
                .map(|v| {
                    let e = (v - m).exp();
                    s += e;
                    e
                })
                .collect();
            for (i, &e) in exps.iter().enumerate() {
                out[r * cols + i] = e / s;
            }
        }
        out
    }

    #[test]
    fn softmax_f32_basic() {
        let cuda = Cuda::new(0).unwrap();
        let rows = 3usize;
        let cols = 8usize;
        let host: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.3).sin()).collect();
        let input: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [rows, cols], &cuda).unwrap();
        let mut output: Tensor<f32, Cuda> = Tensor::zeros([rows, cols], &cuda).unwrap();
        softmax(cuda.config.stream, &input, &mut output).unwrap();
        let got = output.to_host_vec().unwrap();
        let expected = softmax_ref(&host, rows, cols);
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-5, "expected={} got={}", a, b);
        }
        // Sums to 1.
        for r in 0..rows {
            let s: f32 = got[r * cols..(r + 1) * cols].iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_bf16_close_to_reference() {
        let cuda = Cuda::new(0).unwrap();
        let rows = 4usize;
        let cols = 16usize;
        let host_f32: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.1).cos()).collect();
        let host_bf16: Vec<bf16> = host_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let input: Tensor<bf16, Cuda> =
            Tensor::from_host_slice(&host_bf16, [rows, cols], &cuda).unwrap();
        let mut output: Tensor<bf16, Cuda> = Tensor::zeros([rows, cols], &cuda).unwrap();
        softmax(cuda.config.stream, &input, &mut output).unwrap();
        let got: Vec<f32> = output
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let host_rt: Vec<f32> = host_bf16.iter().map(|x| x.to_f32()).collect();
        let expected = softmax_ref(&host_rt, rows, cols);
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 0.005, "expected={} got={}", a, b);
        }
    }
}
