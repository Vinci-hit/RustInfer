// In src/op/kernels/cuda/argmax.rs

use crate::base::error::{Result, Error};
use crate::base::DataType;
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};
use half::bf16;

// --- FFI 声明 ---
unsafe extern "C" {
    fn argmax_cu_f32_ffi(
        logits_ptr: *const f32,
        vocab_size: i32,
        result_ptr_gpu: *mut i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn argmax_cu_bf16_ffi(
        logits_ptr: *const bf16,
        vocab_size: i32,
        result_ptr_gpu: *mut i32,
        stream: cuda::ffi::cudaStream_t,
    );

    // Batch argmax CUDA kernel (BF16 only)
    // Input: logits [batch_size, vocab_size] (BF16)
    // Output: token_ids [batch_size] (I32 on GPU)
    fn argmax_batch_cu_bf16_ffi(
        logits_ptr: *const bf16,
        batch_size: i32,
        vocab_size: i32,
        output_ptr_gpu: *mut i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// 在 GPU 上执行 argmax，并通过 D2H 拷贝隐式同步返回结果。
/// 使用 CudaConfig 中的预分配 result buffer 以支持 CUDA graphs。
pub fn argmax(logits: &Tensor, output_token: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let vocab_size = logits.shape()[0];

    let cuda_cfg = cuda_config
        .ok_or_else(|| Error::InvalidArgument("CudaConfig required for CUDA argmax".to_string()))?;

    let stream = cuda_cfg.stream;
    match logits.dtype() {
        DataType::F32 => {
            let logits_ptr = logits.as_f32()?.buffer().as_ptr() as *const f32;
            unsafe {
                argmax_cu_f32_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    output_token.as_i32_mut()?.buffer_mut().as_mut_ptr() as *mut i32,
                    stream,
                )
            }
        }
        DataType::BF16 => {
            let logits_ptr = logits.as_bf16()?.buffer().as_ptr() as *const bf16;
            unsafe {
                argmax_cu_bf16_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    output_token.as_i32_mut()?.buffer_mut().as_mut_ptr() as *mut i32,
                    stream,
                )
            }
        }
        unsupported => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported dtype '{:?}' for CUDA argmax kernel", unsupported
            )).into());
        }
    };

    Ok(())
}

/// Batch argmax on CUDA (BF16 only)
///
/// Design:
/// - Input: logits [batch_size, vocab_size] (BF16 on GPU)
/// - Output: token_ids [batch_size] (I32 on GPU)
/// - Output tensor is pre-allocated and reused across batches
/// - Server copies result to CPU after each batch
/// - Supports CUDA Graph fusion for high-throughput inference
pub fn argmax_batch(logits: &Tensor, output_tokens: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(
            format!("Expected 2D logits [batch_size, vocab_size], got shape {:?}", shape)
        ).into());
    }

    // Only support BF16 for now (focus on production use case)
    if logits.dtype() != DataType::BF16 {
        return Err(Error::InvalidArgument(
            format!("argmax_batch only supports BF16, got {:?}", logits.dtype())
        ).into());
    }

    let batch_size = shape[0] as i32;
    let vocab_size = shape[1] as i32;

    if output_tokens.shape() != &[batch_size as usize] {
        return Err(Error::InvalidArgument(
            format!("Expected output shape [{}], got {:?}", batch_size, output_tokens.shape())
        ).into());
    }

    if output_tokens.dtype() != DataType::I32 {
        return Err(Error::InvalidArgument(
            format!("Output must be I32, got {:?}", output_tokens.dtype())
        ).into());
    }

    let cuda_cfg = cuda_config
        .ok_or_else(|| Error::InvalidArgument("CudaConfig required for CUDA argmax_batch".to_string()))?;

    let stream = cuda_cfg.stream;
    let logits_ptr = logits.as_bf16()?.buffer().as_ptr() as *const bf16;

    unsafe {
        argmax_batch_cu_bf16_ffi(
            logits_ptr,
            batch_size,
            vocab_size,
            output_tokens.as_i32_mut()?.buffer_mut().as_mut_ptr() as *mut i32,
            stream,
        )
    }

    Ok(())
}
