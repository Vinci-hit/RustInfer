// In src/op/kernels/cuda/argmax.rs

use crate::base::error::{Result, Error};
use crate::base::DataType;
use crate::cuda::error::CudaError;
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
pub fn argmax(logits: &Tensor, cuda_config: Option<&CudaConfig>) -> Result<i32> {
    // --- 1. 形状检查 ---
    let vocab_size = logits.shape()[0];

    // --- 2. 在 GPU 上为结果分配一个临时 Tensor ---
    // 这个张量非常小，只包含一个 i32
    let mut result_gpu = Tensor::new(&[1], DataType::I32, logits.device())?;//动态，非静态图了。
    // --- 3. 获取 CUDA stream ---
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // --- 4. 外部类型分发 (在 Rust 中进行) ---
    // 根据 logits 的类型，调用不同的 FFI 函数
    match logits.dtype() {
        DataType::F32 => {
            // 提取类型化指针
            let logits_ptr = logits.as_f32()?.buffer().as_ptr() as *const f32;
            let result_ptr_gpu = result_gpu.as_i32_mut()?.buffer_mut().as_mut_ptr() as *mut i32;
            
            // 调用 f32 专用的 FFI 函数
            unsafe {
                argmax_cu_f32_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    result_ptr_gpu,
                    stream,
                )
            }
        }
        DataType::BF16 => {
            // 提取类型化指针
            let logits_ptr = logits.as_bf16()?.buffer().as_ptr() as *const bf16;
            let result_ptr_gpu = result_gpu.as_i32_mut()?.buffer_mut().as_mut_ptr() as *mut i32;

            // 调用 bf16 专用的 FFI 函数
            unsafe {
                argmax_cu_bf16_ffi(
                    logits_ptr,
                    vocab_size as i32,
                    result_ptr_gpu,
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

    // --- 6. 将结果从 GPU 拷贝回 CPU (隐式同步) ---
    // `to_cpu()` 会执行一个阻塞的 D2H memcpy，确保在继续之前内核已完成。
    let result_cpu = result_gpu.to_cpu()?;

    // --- 7. 从 CPU 张量中提取最终的 i32 值 ---
    let result_slice = result_cpu.as_i32()?.as_slice()?;
    
    Ok(result_slice[0])
}