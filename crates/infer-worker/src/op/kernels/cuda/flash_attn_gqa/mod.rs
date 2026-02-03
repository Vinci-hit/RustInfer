use crate::base::error::{Error,Result};
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};

// --- FFI 声明 ---
// 假设 C/C++ 端的 CUDA kernel 包装函数签名如下：
// 它接收所有的指针和维度参数。
unsafe extern "C" {
    pub fn flash_attn_gqa_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn flash_decoding_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn flash_decoding_cu_bf16(
        q_ptr: *const half::bf16,
        k_ptr: *const half::bf16,
        v_ptr: *const half::bf16,
        o_ptr: *mut half::bf16,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_cute_128x64x64_tile(
        q_ptr: *const half::bf16,
        k_ptr: *const half::bf16,
        v_ptr: *const half::bf16,
        o_ptr: *mut half::bf16,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_cute_dispatch(
        q_ptr: *const half::bf16,
        k_ptr: *const half::bf16,
        v_ptr: *const half::bf16,
        o_ptr: *mut half::bf16,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    // PagedAttention kernel with CSR-style kv_indices (FlashInfer design)
    pub fn flash_attn_gqa_paged_cu(
        q_ptr: *const half::bf16,
        k_ptr: *const half::bf16,
        v_ptr: *const half::bf16,
        o_ptr: *mut half::bf16,
        kv_indices: *const i32,      // [nnz_tokens] physical slot indices (CSR-style)
        kv_indptr: *const i32,       // [batch_size + 1] CSR row pointers
        batch_size: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        sm_scale: f32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Flash Attention GQA 的 CUDA 内核包装函数 (Prefill/Decode 模式)。
/// 
/// 该函数用于分发参数给底层的 CUDA Kernel，Kernel 在内部处理 K/V Cache 的索引和因果遮蔽。
/// 
/// # Arguments
/// * `input_q`: Query 张量, [Q_SeqLen, Q_HiddenDim]
/// * `input_k_cache`, `input_v_cache`: K/V Cache 完整内存, [Max_SeqLen, KV_HiddenDim]
/// * `output_o`: 输出张量, [Q_SeqLen, Q_HiddenDim]
/// * `q_seq_len`: Q 的实际序列长度 (S_Q)
/// * `current_kv_len`: K/V Cache 的有效历史长度 (S_KV_history)
/// * `num_q_heads`, `num_kv_heads`, `head_dim`: Attention 结构参数
/// * `cuda_config`: 可选的 CUDA stream 配置。
///
/// # Safety
/// This function is unsafe because it accepts a raw pointer (`current_kv_len_gpu`)
/// which must be a valid pointer to device memory containing the KV cache length.
/// The caller must ensure that:
/// - The pointer points to valid, initialized device memory
/// - The memory remains valid for the duration of the function call
/// - The pointer is properly aligned for i32 access
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_gqa(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    q_seq_len: usize,
    current_kv_len_gpu: *const i32,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // --- 1. 数据类型校验 ---
    let dtype = input_q.dtype();
    
    if input_q.dtype() != input_k_cache.dtype() || input_q.dtype() != input_v_cache.dtype() || input_q.dtype() != output_o.dtype() {
        return Err(Error::InvalidArgument(format!(
            "All tensors must have the same data type for flash_attn_gqa. Q: {:?}, K: {:?}, V: {:?}, O: {:?}",
            input_q.dtype(), input_k_cache.dtype(), input_v_cache.dtype(), output_o.dtype()
        )).into());
    }

    // --- 2. 维度检查和转换 ---
    
    // 维度转换为 i32 (假设所有维度都不超过 i32 的范围，这是 LLM 中的标准假设)
    let q_seq_len_i32 = q_seq_len as i32;
    let num_q_heads_i32 = num_q_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;
    
    if head_dim_i32 % 4 != 0 {
        return Err(Error::InvalidArgument("SGEMV float4 kernel requires the inner dimension (N) to be a multiple of 4.".into()).into());
    }

    // --- 3. 获取 CUDA stream ---
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // --- 4. 根据数据类型分发 ---
    match dtype {
        crate::base::DataType::F32 => {
            let q_ptr = input_q.as_f32()?.buffer().as_ptr() as *const f32;
            let k_ptr = input_k_cache.as_f32()?.buffer().as_ptr() as *const f32;
            let v_ptr = input_v_cache.as_f32()?.buffer().as_ptr() as *const f32;
            let o_ptr = output_o.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe {
                if q_seq_len == 1 {
                    flash_decoding_cu(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        head_dim_i32,
                        stream,
                    );
                } else {
                    flash_attn_gqa_cu(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        head_dim_i32,
                        stream,
                    );
                }
            }
        }
        crate::base::DataType::BF16 => {
            let q_ptr = input_q.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let k_ptr = input_k_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let v_ptr = input_v_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let o_ptr = output_o.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

            unsafe {
                if q_seq_len == 1 {
                    flash_decoding_cu_bf16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        head_dim_i32,
                        stream,
                    );
                } else {
                    // Use dispatch function to select appropriate kernel based on head_dim
                    launch_flash_attn_cute_dispatch(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        head_dim_i32,
                        stream,
                    );
                }
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!("Unsupported dtype {:?} for flash_attn_gqa", dtype)).into());
        }
    }
    
    Ok(())
}

/// Flash Attention GQA with PagedAttention support (CSR-style, following FlashInfer design)
///
/// This function implements batch decode with paged KV cache using CSR (Compressed Sparse Row)
/// indexing for efficient memory access. Each request's KV tokens are indexed via:
/// - kv_indices: [nnz_tokens] physical slot indices for all tokens
/// - kv_indptr: [batch_size + 1] CSR row pointers (kv_len[i] = kv_indptr[i+1] - kv_indptr[i])
///
/// # Arguments
/// * `input_q`: Query tensor, [batch_size, num_q_heads, head_dim] (decode mode: q_seq_len=1)
/// * `input_k_cache`: K cache, [total_slots, num_kv_heads, head_dim]
/// * `input_v_cache`: V cache, [total_slots, num_kv_heads, head_dim]
/// * `output_o`: Output tensor, [batch_size, num_q_heads, head_dim]
/// * `kv_indices_gpu`: Device pointer to [nnz_tokens] physical slot indices
/// * `kv_indptr_gpu`: Device pointer to [batch_size + 1] CSR row pointers
/// * `batch_size`: Number of sequences in the batch
/// * `num_q_heads`, `num_kv_heads`, `head_dim`: Attention parameters
/// * `cuda_config`: Optional CUDA stream configuration
///
/// # Safety
/// The caller must ensure:
/// - `kv_indices_gpu` points to valid device memory of size [nnz_tokens]
/// - `kv_indptr_gpu` points to valid device memory of size [batch_size + 1]
/// - All tensor pointers are valid and properly aligned
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_gqa_paged(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    kv_indices_gpu: *const i32,
    kv_indptr_gpu: *const i32,
    batch_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // --- 1. 数据类型校验 ---
    let dtype = input_q.dtype();

    if input_q.dtype() != input_k_cache.dtype() || input_q.dtype() != input_v_cache.dtype() || input_q.dtype() != output_o.dtype() {
        return Err(Error::InvalidArgument(format!(
            "All tensors must have the same data type for flash_attn_gqa_paged. Q: {:?}, K: {:?}, V: {:?}, O: {:?}",
            input_q.dtype(), input_k_cache.dtype(), input_v_cache.dtype(), output_o.dtype()
        )).into());
    }

    // --- 2. 维度检查和转换 ---
    let batch_size_i32 = batch_size as i32;
    let num_q_heads_i32 = num_q_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;

    // Softmax scale: 1/sqrt(head_dim)
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    if head_dim_i32 % 8 != 0 {
        return Err(Error::InvalidArgument("FlashAttention requires head dimension to be a multiple of 8 for vectorized copy.".into()).into());
    }

    // --- 3. 获取 CUDA stream ---
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // --- 4. 根据数据类型分发 ---
    match dtype {
        crate::base::DataType::BF16 => {
            let q_ptr = input_q.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let k_ptr = input_k_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let v_ptr = input_v_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let o_ptr = output_o.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

            unsafe {
                flash_attn_gqa_paged_cu(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    o_ptr,
                    kv_indices_gpu,
                    kv_indptr_gpu,
                    batch_size_i32,
                    num_q_heads_i32,
                    num_kv_heads_i32,
                    head_dim_i32,
                    sm_scale,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!("Unsupported dtype {:?} for flash_attn_gqa_paged, only BF16 is supported", dtype)).into());
        }
    }

    Ok(())
}