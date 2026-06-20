//! SiLU + SwiGLU CUDA kernel wrappers.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

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

pub fn silu_inplace<T: Dtype>(stream: cudaStream_t, x: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => silu_inplace_f32_forward(x.data_ptr_mut() as _, n, stream),
            DataType::BF16 => silu_inplace_bf16_forward(x.data_ptr_mut() as _, n, stream),
            DataType::F16 => silu_inplace_f16_forward(x.data_ptr_mut() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("silu: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

pub fn swiglu_inplace<T: Dtype>(
    stream: cudaStream_t,
    x: &mut Tensor<T, Cuda>,
    gate: &Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = x.numel() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => swiglu_inplace_kernel_cu_fp32x4(
                gate.data_ptr() as _,
                x.data_ptr_mut() as _,
                n,
                stream,
            ),
            DataType::BF16 => {
                swiglu_inplace_cu_bf16x8(gate.data_ptr() as _, x.data_ptr_mut() as _, n, stream)
            }
            DataType::F16 => {
                swiglu_inplace_cu_fp16x8(gate.data_ptr() as _, x.data_ptr_mut() as _, n, stream)
            }
            _ => return Err(OpError::Kernel(format!("swiglu: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

/// Packed SwiGLU: gate_up `[rows, 2*inter]` → out `[rows, inter]`,
/// where `out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]`.
///
/// Replaces 2 × `split_cols` + `swiglu_inplace` with a single fused kernel.
pub fn swiglu_packed<T: Dtype>(
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
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => swiglu_packed_cu_bf16(
                gate_up.data_ptr() as _,
                out.data_ptr_mut() as _,
                rows as i32,
                inter as i32,
                stream,
            ),
            DataType::F32 => {
                // Generic fallback: split gate_up [rows, 2*inter] into
                // gate / up halves, apply silu to gate, then ewise multiply.
                let dev = gate_up.device().clone();
                let mut gate: Tensor<T, Cuda> = Tensor::zeros([rows, inter], &dev)?;
                let mut up: Tensor<T, Cuda> = Tensor::zeros([rows, inter], &dev)?;
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
            }
            _ => {
                return Err(OpError::Kernel(format!(
                    "swiglu_packed: unsupported dtype {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}
