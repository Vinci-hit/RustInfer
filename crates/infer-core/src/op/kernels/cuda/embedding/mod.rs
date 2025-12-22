use crate::base::error::{Error, Result};
use crate::cuda::CudaConfig;
use crate::tensor::Tensor;
use crate::cuda;

// ============================================================================
//  手动 FFI 声明
// ============================================================================
unsafe extern "C" {
    fn embedding_kernel_cu_fp32x4(
        output: *mut f32,
        input_token_ids: *const i32,
        weight: *const f32,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    fn embedding_kernel_cu_bf16x8(
        output: *mut half::bf16,
        input_token_ids: *const i32,
        weight: *const half::bf16,
        token_len: i32,
        dim: i32,
        vocab_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Embedding 的 CUDA 内核包装函数
pub fn embedding(
    input_tokens: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    cuda_config: Option<&CudaConfig>
) -> Result<()> {
    // --- 1. 获取具体类型 ---
    let tokens_typed = input_tokens.as_i32()?; // 输入必须是 i32
    let vocab_size = weight.shape()[0];
    let dim = weight.shape()[1];
    let token_len = input_tokens.shape()[0];
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);
    let tokens_ptr = tokens_typed.buffer().as_ptr() as *const i32;
    
    // --- 2. 数据类型分发 ---
    let dtype = output.dtype();
    match dtype {
        crate::base::DataType::F32 => {
            let weight_typed = weight.as_f32()?;
            let out_typed = output.as_f32_mut()?;
            
            // --- 3. 检查前置条件 (对齐) ---
            if !dim.is_multiple_of(4) {
                return Err(Error::InvalidArgument(
                    "CUDA Embedding kernel (fp32x4) requires the embedding dimension to be a multiple of 4.".into()
                ).into());
            }
            
            let weight_ptr = weight_typed.buffer().as_ptr() as *const f32;
            let out_ptr = out_typed.buffer_mut().as_mut_ptr() as *mut f32;
            
            // --- 4. 调用 FFI 函数 ---
            unsafe {
                embedding_kernel_cu_fp32x4(
                    out_ptr,
                    tokens_ptr,
                    weight_ptr,
                    token_len as i32,
                    dim as i32,
                    vocab_size as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::BF16 => {
            let weight_typed = weight.as_bf16()?;
            let out_typed = output.as_bf16_mut()?;
            
            // --- 3. 检查前置条件 (对齐) ---
            if !dim.is_multiple_of(8) {
                return Err(Error::InvalidArgument(
                    "CUDA Embedding kernel (bf16x8) requires the embedding dimension to be a multiple of 8.".into()
                ).into());
            }
            
            let weight_ptr = weight_typed.buffer().as_ptr() as *const half::bf16;
            let out_ptr = out_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            
            // --- 4. 调用 FFI 函数 ---
            unsafe {
                embedding_kernel_cu_bf16x8(
                    out_ptr,
                    tokens_ptr,
                    weight_ptr,
                    token_len as i32,
                    dim as i32,
                    vocab_size as i32,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for embedding CUDA kernel: {:?}", dtype
            )).into());
        }
    }

    Ok(())
}