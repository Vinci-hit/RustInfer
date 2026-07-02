//! SiLU + SwiGLU CUDA kernel wrappers.
//!
//! Dispatch is an attribute of the element type: [`SwigluKernel`] is
//! implemented once per supported dtype and names that dtype's `extern "C"`
//! entry points, so [`silu_inplace`]/[`swiglu_inplace`] are generic with no
//! runtime `match`. Adding a dtype is one `impl`; an unsupported dtype fails to
//! compile.
//!
//! Packed SwiGLU has a narrower supported set (native BF16 kernel + an F32
//! software fallback, no F16), so it gets its own [`SwigluPackedKernel`]
//! implemented only for the dtypes it actually handles.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    // SwiGLU: input_output_x = silu(input_output_x) * input_y, in place.
    // Signature in .cu: (input_y, input_output_x, num_elements, stream)
    fn swiglu_inplace_cu_bf16x8(
        y: *const half::bf16,
        x: *mut half::bf16,
        n: i32,
        stream: cudaStream_t,
    );
    fn swiglu_inplace_cu_fp16x8(
        y: *const half::f16,
        x: *mut half::f16,
        n: i32,
        stream: cudaStream_t,
    );
    fn swiglu_inplace_kernel_cu_fp32x4(y: *const f32, x: *mut f32, n: i32, stream: cudaStream_t);

    // Packed SwiGLU: gate_up [rows, 2*inter] → out [rows, inter]
    fn swiglu_packed_cu_bf16(
        gate_up: *const half::bf16,
        out: *mut half::bf16,
        rows: i32,
        inter: i32,
        stream: cudaStream_t,
    );

    // SiLU in-place: x = silu(x).
    fn silu_inplace_bf16_forward(x: *mut half::bf16, n: i32, stream: cudaStream_t);
    fn silu_inplace_f16_forward(x: *mut half::f16, n: i32, stream: cudaStream_t);
    fn silu_inplace_f32_forward(x: *mut f32, n: i32, stream: cudaStream_t);
}

/// Element types with the elementwise SiLU / SwiGLU CUDA kernels. Each method
/// forwards to this dtype's `extern` entry; the wrappers below are generic over
/// this trait, so the dtype→kernel mapping lives here as a type attribute.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for `n` elements on
/// `stream`; this just names the FFI entries and performs no checks.
pub trait SwigluKernel: CudaFloat {
    /// `x = silu(x)`, in place over `n` elements.
    unsafe fn silu_inplace(x: *mut Self, n: i32, stream: cudaStream_t);
    /// `x = silu(x) * y`, in place over `n` elements.
    unsafe fn swiglu_inplace(y: *const Self, x: *mut Self, n: i32, stream: cudaStream_t);
}

impl SwigluKernel for f32 {
    #[inline]
    unsafe fn silu_inplace(x: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { silu_inplace_f32_forward(x, n, stream) }
    }
    #[inline]
    unsafe fn swiglu_inplace(y: *const Self, x: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { swiglu_inplace_kernel_cu_fp32x4(y, x, n, stream) }
    }
}

impl SwigluKernel for half::bf16 {
    #[inline]
    unsafe fn silu_inplace(x: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { silu_inplace_bf16_forward(x, n, stream) }
    }
    #[inline]
    unsafe fn swiglu_inplace(y: *const Self, x: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { swiglu_inplace_cu_bf16x8(y, x, n, stream) }
    }
}

impl SwigluKernel for half::f16 {
    #[inline]
    unsafe fn silu_inplace(x: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { silu_inplace_f16_forward(x, n, stream) }
    }
    #[inline]
    unsafe fn swiglu_inplace(y: *const Self, x: *mut Self, n: i32, stream: cudaStream_t) {
        unsafe { swiglu_inplace_cu_fp16x8(y, x, n, stream) }
    }
}

/// Element types with a packed-SwiGLU path. BF16 has a native fused kernel;
/// F32 has a software fallback (split → silu → ewise-multiply). F16 has neither,
/// so it is intentionally *not* implemented — `swiglu_packed::<f16>` fails to
/// compile (the old code returned a runtime error for it).
pub trait SwigluPackedKernel: CudaFloat {
    /// `out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]`.
    fn swiglu_packed(
        stream: cudaStream_t,
        gate_up: &Tensor<Self, Cuda>,
        out: &mut Tensor<Self, Cuda>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()>;
}

impl SwigluPackedKernel for half::bf16 {
    #[inline]
    fn swiglu_packed(
        stream: cudaStream_t,
        gate_up: &Tensor<Self, Cuda>,
        out: &mut Tensor<Self, Cuda>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        unsafe {
            swiglu_packed_cu_bf16(
                gate_up.data_ptr(),
                out.data_ptr_mut(),
                rows as i32,
                inter as i32,
                stream,
            );
        }
        Ok(())
    }
}

impl SwigluPackedKernel for f32 {
    #[inline]
    fn swiglu_packed(
        stream: cudaStream_t,
        gate_up: &Tensor<Self, Cuda>,
        out: &mut Tensor<Self, Cuda>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        // Generic fallback: split gate_up [rows, 2*inter] into
        // gate / up halves, apply silu to gate, then ewise multiply.
        let dev = gate_up.device().clone();
        let mut gate: Tensor<Self, Cuda> = Tensor::zeros([rows, inter], &dev)?;
        let mut up: Tensor<Self, Cuda> = Tensor::zeros([rows, inter], &dev)?;
        super::split_cols::split_cols(
            stream,
            gate_up,
            &mut gate,
            rows as i32,
            (2 * inter) as i32,
            0,
            inter as i32,
        )?;
        super::split_cols::split_cols(
            stream,
            gate_up,
            &mut up,
            rows as i32,
            (2 * inter) as i32,
            inter as i32,
            inter as i32,
        )?;
        silu_inplace(stream, &mut gate)?;
        super::ewise_mul::ewise_mul(stream, &gate, &up, out)?;
        Ok(())
    }
}

pub fn silu_inplace<T: SwigluKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    unsafe {
        T::silu_inplace(x.data_ptr_mut(), n, stream);
    }
    Ok(())
}

pub fn swiglu_inplace<T: SwigluKernel>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    gate: &Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    unsafe {
        T::swiglu_inplace(gate.data_ptr(), x.data_ptr_mut(), n, stream);
    }
    Ok(())
}

/// Packed SwiGLU: gate_up `[rows, 2*inter]` → out `[rows, inter]`,
/// where `out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]`.
///
/// Replaces 2 × `split_cols` + `swiglu_inplace` with a single fused kernel
/// (BF16); F32 uses the split → silu → ewise-multiply software fallback.
pub fn swiglu_packed<T: SwigluPackedKernel>(
    stream: cudaStream_t,
    gate_up: &Tensor<T, Cuda>,
    out: &mut Tensor<T, Cuda>,
    rows: usize,
    inter: usize,
) -> OpResult<()> {
    if inter % 8 != 0 {
        return Err(OpError::Shape(format!(
            "swiglu_packed: inter ({}) must be a multiple of 8",
            inter
        )));
    }
    T::swiglu_packed(stream, gate_up, out, rows, inter)
}
