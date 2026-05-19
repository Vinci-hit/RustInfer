// In src/op/kernels/cuda/argmax.rs

use crate::base::error::{Result, Error};
use crate::base::DataType;
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};
use half::bf16; // 确保引入 bf16 类型

// --- FFI 声明 ---
// 现在我们为每种支持的数据类型提供一个独立的 FFI 函数
unsafe extern "C" {
    fn argmax_cu_f32_ffi(
        logits_ptr: *const f32,
        vocab_size: i32,
        result_ptr_gpu: *mut i32, // << 指向 GPU 内存
        stream: cuda::ffi::cudaStream_t,
    );

    fn argmax_cu_bf16_ffi(
        logits_ptr: *const bf16, // Rust 的 bf16 类型
        vocab_size: i32,
        result_ptr_gpu: *mut i32, // << 指向 GPU 内存
        stream: cuda::ffi::cudaStream_t,
    );

    fn argmax_cu_fp16_ffi(
        logits_ptr: *const half::f16, // Rust 的 f16 类型
        vocab_size: i32,
        result_ptr_gpu: *mut i32, // << 指向 GPU 内存
        stream: cuda::ffi::cudaStream_t,
    );

    // Batched argmax: logits [B, vocab_size], out [B]
    fn argmax_batch_cu_bf16_ffi(
        logits_ptr: *const bf16,
        batch_size: i32,
        vocab_size: i32,
        row_stride: i32,
        out_ptr: *mut i32,
        stream: cuda::ffi::cudaStream_t,
    );
    fn argmax_batch_cu_f32_ffi(
        logits_ptr: *const f32,
        batch_size: i32,
        vocab_size: i32,
        row_stride: i32,
        out_ptr: *mut i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// 在 GPU 上执行 argmax，并通过 D2H 拷贝隐式同步返回结果。
/// 使用 CudaConfig 中的预分配 result buffer 以支持 CUDA graphs。
pub fn argmax(logits: &Tensor, output_token: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    // --- 1. 形状检查 ---
    let vocab_size = logits.shape()[0];

    // --- 2. 获取 CUDA stream ---
    let cuda_cfg = cuda_config
        .ok_or_else(|| Error::InvalidArgument("CudaConfig required for CUDA argmax".to_string()))?;

    let stream = cuda_cfg.stream;
    // --- 4. 根据 logits 的类型，调用不同的 FFI 函数 ---
    match logits.dtype() {
        DataType::F32 => {
            // 提取类型化指针
            let logits_ptr = logits.as_f32()?.data_ptr();

            // 调用 f32 专用的 FFI 函数
            unsafe {
                argmax_cu_f32_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    output_token.as_i32_mut()?.data_ptr_mut(),
                    stream,
                )
            }
        }
        DataType::BF16 => {
            // 提取类型化指针
            let logits_ptr = logits.as_bf16()?.data_ptr();

            // 调用 bf16 专用的 FFI 函数
            unsafe {
                argmax_cu_bf16_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    output_token.as_i32_mut()?.data_ptr_mut(),
                    stream,
                )
            }
        }
        DataType::F16 => {
            // 提取类型化指针
            let logits_ptr = logits.as_f16()?.data_ptr();

            // 调用 bf16 专用的 FFI 函数
            unsafe {
                argmax_cu_fp16_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    output_token.as_i32_mut()?.data_ptr_mut(),
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
/// Batched argmax：`logits[B, vocab_size]` → `out[B]`。
/// 每个 CUDA block 处理一个 seq 行。
pub fn argmax_batch(logits: &Tensor, out: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let batch_size = logits.shape()[0];
    let vocab_size = logits.shape()[1];
    argmax_batch_strided(
        logits, vocab_size, vocab_size, 0,
        batch_size, out, cuda_config,
    )
}

/// Batched argmax strided：logits 从 col_offset 起，每行跨 row_stride 个元素，
/// 扫描 vocab_size 个元素。可直接从非连续 [B, full_vocab] 里取前 vocab_size 列做 argmax，
/// 省去 split_cols。
#[allow(clippy::too_many_arguments)]
pub fn argmax_batch_strided(
    logits: &Tensor,
    vocab_size: usize,
    row_stride: usize,
    col_offset: usize,
    batch_size: usize,
    out: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let cuda_cfg = cuda_config
        .ok_or_else(|| Error::InvalidArgument("CudaConfig required for argmax_batch".to_string()))?;
    let stream = cuda_cfg.stream;
    let out_ptr = out.as_i32_mut()?.data_ptr_mut();
    match logits.dtype() {
        DataType::BF16 => unsafe {
            let base = logits.as_bf16()?.data_ptr();
            argmax_batch_cu_bf16_ffi(
                base.add(col_offset),
                batch_size as i32, vocab_size as i32, row_stride as i32,
                out_ptr, stream,
            );
        },
        DataType::F32 => unsafe {
            let base = logits.as_f32()?.data_ptr();
            argmax_batch_cu_f32_ffi(
                base.add(col_offset),
                batch_size as i32, vocab_size as i32, row_stride as i32,
                out_ptr, stream,
            );
        },
        other => return Err(Error::InvalidArgument(
            format!("argmax_batch_strided: unsupported dtype {:?}", other)
        ).into()),
    }
    Ok(())
}
