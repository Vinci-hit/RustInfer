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
            let logits_ptr = logits.as_f32()?.buffer().as_ptr() as *const f32;

            // 调用 f32 专用的 FFI 函数
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
            // 提取类型化指针
            let logits_ptr = logits.as_bf16()?.buffer().as_ptr() as *const bf16;

            // 调用 bf16 专用的 FFI 函数
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