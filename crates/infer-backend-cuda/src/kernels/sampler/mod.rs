//! Argmax sampler CUDA kernel.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
    // BF16 C signature: (logits, selected_rows_device, batch_size, vocab_size,
    // result_gpu, workspace, stream). `selected_rows_device` is nullable — null
    // means "argmax every row 0..batch". The previous binding OMITTED this
    // pointer, shifting every subsequent argument by one (batch→selected_rows,
    // vocab→batch, output→vocab, ...) so the kernel never wrote `output` and
    // decode emitted token 0 ("!") for every position.
    fn argmax_cu_bf16_ffi(
        input: *const half::bf16,
        selected_rows: *const i32,
        batch_size: i32,
        vocab_size: i32,
        output: *mut i32,
        workspace: *mut f32,
        stream: cudaStream_t,
    );
    fn argmax_cu_fp16_ffi(
        input: *const half::f16,
        vocab_size: i32,
        output: *mut i32,
        workspace: *mut f32,
        stream: cudaStream_t,
    );
    fn argmax_cu_f32_ffi(
        input: *const f32,
        vocab_size: i32,
        output: *mut i32,
        workspace: *mut f32,
        stream: cudaStream_t,
    );
}

pub fn argmax<T: Dtype>(
    stream: cudaStream_t,
    logits: &Tensor<T, Cuda>,
    output: &mut Tensor<i32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
    // BF16 only: device i32 list of rows to argmax. `None` → every row.
    // Length determines the kernel's `batch_size` (one output id per selected
    // row). f16/f32 bindings have no selector pin yet — `Some` returns Err.
    selected_rows: Option<&Tensor<i32, Cuda>>,
) -> OpResult<()> {
    let vocab_size = *logits.shape().as_slice().last().unwrap() as i32;
    let logits_rows = *logits.shape().as_slice().first().unwrap() as i32;
    // When `selected_rows` is provided, the kernel argmaxes ONLY those rows;
    // the C entry's `batch_size` is the selector length, not logits rows.
    let (sel_ptr, eff_batch) = match selected_rows {
        None => (std::ptr::null::<i32>(), logits_rows),
        Some(sel) => (sel.data_ptr(), sel.numel() as i32),
    };
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => {
                if selected_rows.is_some() {
                    return Err(OpError::Kernel(
                        "argmax: selected_rows is bf16-only (f32 binding has no selector)".into(),
                    ));
                }
                argmax_cu_f32_ffi(
                    logits.data_ptr() as _,
                    vocab_size,
                    output.data_ptr_mut(),
                    workspace.data_ptr_mut(),
                    stream,
                )
            }
            DataType::BF16 => argmax_cu_bf16_ffi(
                logits.data_ptr() as _,
                sel_ptr,
                eff_batch,
                vocab_size,
                output.data_ptr_mut(),
                workspace.data_ptr_mut(),
                stream,
            ),
            DataType::F16 => {
                if selected_rows.is_some() {
                    return Err(OpError::Kernel(
                        "argmax: selected_rows is bf16-only (f16 binding has no selector)".into(),
                    ));
                }
                argmax_cu_fp16_ffi(
                    logits.data_ptr() as _,
                    vocab_size,
                    output.data_ptr_mut(),
                    workspace.data_ptr_mut(),
                    stream,
                )
            }
            _ => return Err(OpError::Kernel(format!("argmax: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
