//! Batched argmax for ragged batch.
//!
//! Strategy: pick the last logits row per sequence (using cu_q_lens), then
//! launch the single-row `argmax_cu_*_ffi` once per sequence. Batch=1 path
//! is the same call site as before — zero-cost wrapper.
//!
//! The lower-overhead `argmax_batch_cu_*_ffi` kernel assumes a contiguous
//! `[batch, vocab]` layout, which doesn't fit our ragged Q output without
//! an explicit gather; we keep the implementation simple here and let
//! Phase 4 fuse the gather + argmax into a single kernel if needed.
//!
//! ## Decode-only fast path
//!
//! When every seq has q_len == 1 (i.e. logits is already `[batch, vocab]`
//! contiguous and `last_row[i] == i`), `argmax_batched_decode_into` issues
//! a SINGLE `argmax_batch_cu_*_ffi` launch that writes directly into a
//! caller-provided device output. Zero D2H, zero alloc — graph-capturable.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::{self, cudaStream_t};

unsafe extern "C" {
    fn argmax_cu_bf16_ffi(
        input: *const half::bf16,
        selected_rows_device: *const i32,
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

pub fn argmax_batched_decode_into<T: Dtype>(
    logits: &Tensor<T, Cuda>,
    out_dev: &mut Tensor<i32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
) -> OpResult<()> {
    let shape = logits.shape().as_slice();
    if shape.len() < 2 {
        return Err(OpError::Shape(format!(
            "argmax_batched_decode_into: logits must be 2D, got {:?}",
            shape,
        )));
    }
    let batch = shape[0];
    let vocab = shape[1];
    if out_dev.numel() < batch {
        return Err(OpError::Shape(format!(
            "argmax_batched_decode_into: out_dev too small ({} < batch {})",
            out_dev.numel(),
            batch,
        )));
    }
    if batch == 0 {
        return Ok(());
    }

    // Only support BF16
    if T::DATA_TYPE != DataType::BF16 {
        return Err(OpError::Kernel(format!(
            "argmax_batched_decode_into: dtype {:?} not implemented",
            T::DATA_TYPE,
        )));
    }

    let stream = logits.device().config.stream;

    // Single path: argmax per row
    unsafe {
        argmax_cu_bf16_ffi(
            logits.data_ptr() as _,
            std::ptr::null(),
            batch as i32,
            vocab as i32,
            out_dev.data_ptr_mut(),
            workspace.data_ptr_mut(),
            stream,
        );
    }
    Ok(())
}

pub fn argmax_batched<T: Dtype>(
    logits: &Tensor<T, Cuda>,
    cu_q_lens: &Tensor<i32, Cuda>,
    batch: usize,
    out_dev: &mut Tensor<i32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
    rows: &mut Tensor<i32, Cuda>,
) -> OpResult<Vec<i32>> {
    if batch == 0 {
        return Ok(Vec::new());
    }
    let total_rows = logits.shape().as_slice()[0];
    let vocab = logits.numel() / total_rows;
    let device = logits.device();
    let stream = device.config.stream;
    // Pull cu_q_lens to host (small, batch+1 i32s).
    let cu = cu_q_lens.to_host_vec()?;
    if cu.len() != batch + 1 {
        return Err(OpError::Shape(format!(
            "argmax_batched: cu_q_lens.len()={} expected batch+1={}",
            cu.len(),
            batch + 1,
        )));
    }

    let elem_bytes = T::SIZE_BYTES;
    let mut selected_rows = Vec::with_capacity(batch);
    for seq in 0..batch {
        selected_rows.push(cu[seq + 1] - 1); // 提取出那几个关键行号
    }
    // Narrow rows to [batch] so upload_from_host length matches.
    let mut rows = rows.narrow(0, 0, batch)?;
    rows.upload_from_host(&selected_rows)?;
    for seq in 0..batch {
        let last_row = (cu[seq + 1] - 1) as usize;
        let row_ptr =
            unsafe { (logits.data_ptr() as *const u8).add(last_row * vocab * elem_bytes) };
        let out_ptr = unsafe { out_dev.data_ptr_mut().add(seq) };
        unsafe {
            match T::DATA_TYPE {
                DataType::BF16 => argmax_cu_bf16_ffi(
                    logits.data_ptr() as _,
                    rows.data_ptr(),
                    batch as i32,
                    vocab as i32,
                    out_ptr,
                    workspace.data_ptr_mut(),
                    stream,
                ),
                DataType::F16 => argmax_cu_fp16_ffi(
                    row_ptr as _,
                    vocab as i32,
                    out_ptr,
                    workspace.data_ptr_mut(),
                    stream,
                ),
                DataType::F32 => argmax_cu_f32_ffi(
                    row_ptr as _,
                    vocab as i32,
                    out_ptr,
                    workspace.data_ptr_mut(),
                    stream,
                ),
                _ => {
                    return Err(OpError::Kernel(format!(
                        "argmax_batched: dtype {:?}",
                        T::DATA_TYPE
                    )));
                }
            }
        }
        break;
    }
    // Sync once before D2H read.
    unsafe {
        let code = ffi::cudaStreamSynchronize(stream);
        if code != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!("argmax_batched sync: {:?}", code)));
        }
    }
    out_dev.to_host_vec()
}
