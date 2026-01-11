use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};

// --- FFI 声明 ---
// 假设 C/C++ 端的 CUDA kernel 签名如下：
unsafe extern "C" {
    pub fn rope_kernel_cu(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        input_q: *mut f32,
        input_k: *mut f32,
        input_pos: *const i32,
        seq_len:i32,
        sin_cache: *const f32,
        cos_cache: *const f32,
        stream: cuda::ffi::cudaStream_t,
    );
    
    // BF16版本的ROPE CUDA kernel
    pub fn rope_kernel_cu_bf16(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        input_q: *mut half::bf16,
        input_k: *mut half::bf16,
        input_pos: *const i32,
        seq_len: i32,
        sin_cache: *const half::bf16,
        cos_cache: *const half::bf16,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Rotary Positional Embedding (RoPE) 的 CUDA 内核包装函数
///
/// 这是一个就地 (in-place) 操作，会修改 input_q 和 input_k 的内容。
///
/// # Arguments
/// * `dim`: Q 和 K 向量的总旋转维度。
/// * `kv_dim`: K 向量旋转的维度。
/// * `head_size`: Attention Head 的大小。
/// * `input_q`: Query 张量, 被就地修改。
/// * `input_k`: Key 张量, 被就地修改。
/// * `input_pos`: 包含当前位置索引的张量。
/// * `sin_cache`: 正弦缓存张量。
/// * `cos_cache`: 余弦缓存张量。
/// * `stream`: 可选的 CUDA stream。
#[allow(clippy::too_many_arguments)]
pub fn rope(
    dim: usize,
    kv_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    input_pos: &Tensor,
    seq_len:i32,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // --- 1. 根据数据类型获取具体类型和指针 ---
    let dtype = input_q.dtype();
    
    // Pos 是 i32，需要不可变指针
    let pos = input_pos.as_i32()?.buffer().as_ptr() as *const i32;

    // --- 2. 维度检查和转换 ---
    
    // 维度转换为 i32。注意：这里我们信任上层 Op 已经做了充分的边界检查。
    let dim_i32 = dim as i32;
    let kv_dim_i32 = kv_dim as i32;
    let head_size_i32 = head_size as i32;

    if dim_i32 < 0 || kv_dim_i32 < 0 || head_size_i32 < 0 {
         return Err(Error::InvalidArgument("RoPE dimensions must be non-negative.".to_string()).into());
    }

    // --- 3. 获取 CUDA stream ---
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // --- 4. 根据数据类型调用相应的 FFI 函数 ---
    match dtype {
        crate::base::DataType::F32 => {
            // Q 和 K 需要可变指针
            let q_ptr = input_q.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            let k_ptr = input_k.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;

            // Cache 是 f32，需要不可变指针
            let sin_ptr = sin_cache.as_f32()?.buffer().as_ptr() as *const f32;
            let cos_ptr = cos_cache.as_f32()?.buffer().as_ptr() as *const f32;

            unsafe {
                rope_kernel_cu(
                    dim_i32,
                    kv_dim_i32,
                    head_size_i32,
                    q_ptr,       // *mut f32
                    k_ptr,       // *mut f32
                    pos,     // *const i32
                    seq_len,
                    sin_ptr,     // *const f32
                    cos_ptr,     // *const f32
                    stream,
                );
            }
        }
        crate::base::DataType::BF16 => {
            // Q 和 K 需要可变指针
            let q_ptr = input_q.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let k_ptr = input_k.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

            // Cache 是 bf16，需要不可变指针
            let sin_ptr = sin_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let cos_ptr = cos_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;

            unsafe {
                rope_kernel_cu_bf16(
                    dim_i32,
                    kv_dim_i32,
                    head_size_i32,
                    q_ptr,       // *mut bf16
                    k_ptr,       // *mut bf16
                    pos,     // const  * i32
                    seq_len,
                    sin_ptr,     // *const bf16
                    cos_ptr,     // *const bf16
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for ROPE CUDA kernel: {:?}", dtype
            )).into());
        }
    }
    
    Ok(())
}

unsafe extern "C" {
    pub fn sin_cos_cache_calc_cu(
        head_size: i32,
        max_seq_len: i32,
        sin_cache: *mut f32,
        cos_cache: *mut f32,
        stream: cuda::ffi::cudaStream_t,
    );
    
    // BF16版本的sin_cos_cache_calc CUDA kernel
    pub fn sin_cos_cache_calc_cu_bf16(
        head_size: i32,
        max_seq_len: i32,
        sin_cache: *mut half::bf16,
        cos_cache: *mut half::bf16,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// 计算并填充正弦和余弦旋转嵌入 (RoPE) 的缓存的 CUDA 内核包装函数。
///
/// # Arguments
/// * `head_size`: 旋转维度的大小 (K)。
/// * `max_seq_len`: 序列的最大长度 (M)。
/// * `sin_cache`: 正弦值输出张量, 形状 [max_seq_len, head_size]。
/// * `cos_cache`: 余弦值输出张量, 形状 [max_seq_len, head_size]。
/// * `cuda_config`: 可选的 CUDA 配置，用于获取 stream。
pub fn sin_cos_cache_calc_cuda(
    head_size: usize,
    max_seq_len: usize,
    sin_cache: &mut Tensor,
    cos_cache: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = sin_cache.dtype();
   
    // --- 2. 维度转换 ---
    
    // 维度转换为 i32
    let head_size_i32 = head_size as i32;
    let max_seq_len_i32 = max_seq_len as i32;

    if head_size_i32 < 0 || max_seq_len_i32 < 0 {
         return Err(Error::InvalidArgument("Dimensions must be non-negative.".to_string()).into());
    }

    // --- 3. 获取 CUDA stream ---
    // 参照你提供的 sgemv 风格
    let mut stream: cuda::ffi::cudaStream_t = std::ptr::null_mut();
    if let Some(config) = cuda_config {
        stream = config.stream; 
    }

    // --- 4. 根据数据类型调用相应的 FFI 函数 ---
    match dtype {
        crate::base::DataType::F32 => {
            // sin_cache 和 cos_cache 都是输出，需要可变指针 (*mut f32)
            let sin_ptr = sin_cache.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            let cos_ptr = cos_cache.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;

            unsafe {
                sin_cos_cache_calc_cu(
                    head_size_i32,
                    max_seq_len_i32,
                    sin_ptr,
                    cos_ptr,
                    stream,
                );
            }
        }
        crate::base::DataType::BF16 => {
            // sin_cache 和 cos_cache 都是输出，需要可变指针 (*mut bf16)
            let sin_ptr = sin_cache.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let cos_ptr = cos_cache.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

            unsafe {
                sin_cos_cache_calc_cu_bf16(
                    head_size_i32,
                    max_seq_len_i32,
                    sin_ptr,
                    cos_ptr,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for sin_cos_cache_calc CUDA kernel: {:?}", dtype
            )).into());
        }
    }
    
    Ok(())
}