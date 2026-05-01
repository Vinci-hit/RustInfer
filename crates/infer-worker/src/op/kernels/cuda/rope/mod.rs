use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};

// --- FFI 声明 ---
unsafe extern "C" {
    // F32 RoPE —— 保留旧 pos_offset+seq_idx 语义（z_image 等用）
    pub fn rope_kernel_cu(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        input_q: *mut f32,
        input_k: *mut f32,
        input_pos: *const i32,
        seq_len: i32,
        sin_cache: *const f32,
        cos_cache: *const f32,
        stream: cuda::ffi::cudaStream_t,
    );

    // BF16 / FP16 RoPE —— **唯一** API，per-row pos 语义
    //   positions[i] 是第 i 行的绝对位置，长度 == seq_len
    pub fn rope_kernel_cu_bf16(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        input_q: *mut half::bf16,
        input_k: *mut half::bf16,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const half::bf16,
        cos_cache: *const half::bf16,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn rope_kernel_cu_fp16(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        input_q: *mut half::f16,
        input_k: *mut half::f16,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const half::f16,
        cos_cache: *const half::f16,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Rotary Positional Embedding 的 CUDA kernel 包装函数。
///
/// **唯一**入口：永远传一个位置数组。caller 永远负责把当前这批 token 的
/// 绝对位置写到 `positions`（长度 = `input_q.shape()[0]`）。
/// - decode 单步：positions = `[p]`
/// - batch decode：positions = `[p_0, p_1, ..., p_{B-1}]`
/// - prefill 一段：positions = `[start, start+1, ..., start+seq_len-1]`
///
/// # Arguments
/// * `input_q` / `input_k` — in-place 修改，shape `[seq_len, num_q_heads*head_size]` /
///   `[seq_len, num_kv_heads*head_size]`（连续内存）
/// * `positions` — I32 device tensor, shape `[seq_len]`
#[allow(clippy::too_many_arguments)]
pub fn rope(
    dim: usize,
    kv_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    positions: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let seq_len = input_q.shape()[0];
    rope_strided(
        dim, kv_dim, head_size,
        input_q, input_k,
        dim, kv_dim, 0, 0,    // 连续内存，row_stride = inner_dim，col_offset = 0
        positions,
        sin_cache, cos_cache,
        seq_len,
        cuda_config,
    )
}

/// RoPE 低层接口：允许传任意 row_stride / col_offset（元素单位），
/// 使 q / k 可以指向 fused qkv tensor 的对应段，省去上游 split。
#[allow(clippy::too_many_arguments)]
pub fn rope_strided(
    dim: usize,
    kv_dim: usize,
    head_size: usize,
    q_tensor: &mut Tensor,
    k_tensor: &mut Tensor,
    q_row_stride: usize,
    k_row_stride: usize,
    q_col_offset: usize,
    k_col_offset: usize,
    positions: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
    seq_len: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = q_tensor.dtype();
    if positions.shape()[0] < seq_len {
        return Err(Error::InvalidArgument(format!(
            "rope_strided: positions.len ({}) < seq_len ({})",
            positions.shape()[0], seq_len
        )).into());
    }

    let pos_ptr = positions.as_i32()?.buffer().as_ptr() as *const i32;
    let dim_i32 = dim as i32;
    let kv_dim_i32 = kv_dim as i32;
    let head_size_i32 = head_size as i32;
    let q_stride_i32 = q_row_stride as i32;
    let k_stride_i32 = k_row_stride as i32;
    let seq_len_i32 = seq_len as i32;
    let stream = CudaConfig::resolve_stream(cuda_config);

    match dtype {
        crate::base::DataType::BF16 => {
            let q_base = q_tensor.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let k_base = k_tensor.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let q_ptr = unsafe { q_base.add(q_col_offset) };
            let k_ptr = unsafe { k_base.add(k_col_offset) };
            let sin_ptr = sin_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let cos_ptr = cos_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            unsafe {
                rope_kernel_cu_bf16(
                    dim_i32, kv_dim_i32, head_size_i32,
                    q_ptr, k_ptr, pos_ptr, seq_len_i32,
                    q_stride_i32, k_stride_i32,
                    sin_ptr, cos_ptr, stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            let q_base = q_tensor.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            let k_base = k_tensor.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            let q_ptr = unsafe { q_base.add(q_col_offset) };
            let k_ptr = unsafe { k_base.add(k_col_offset) };
            let sin_ptr = sin_cache.as_f16()?.buffer().as_ptr() as *const half::f16;
            let cos_ptr = cos_cache.as_f16()?.buffer().as_ptr() as *const half::f16;
            unsafe {
                rope_kernel_cu_fp16(
                    dim_i32, kv_dim_i32, head_size_i32,
                    q_ptr, k_ptr, pos_ptr, seq_len_i32,
                    q_stride_i32, k_stride_i32,
                    sin_ptr, cos_ptr, stream,
                );
            }
        }
        crate::base::DataType::F32 => {
            // F32 还用旧的 pos_offset+seq_idx 语义，限制 positions 必须是长度 1 的 "起始 pos"。
            // （F32 RoPE 目前只有 z_image 的 non-LLM 路径在用。）
            if positions.shape()[0] != 1 {
                return Err(Error::InvalidArgument(
                    "F32 RoPE kernel 仍采用 'start_pos + seq_idx' 语义，positions 必须长度 1".into()
                ).into());
            }
            let q_base = q_tensor.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            let k_base = k_tensor.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            let q_ptr = unsafe { q_base.add(q_col_offset) };
            let k_ptr = unsafe { k_base.add(k_col_offset) };
            let sin_ptr = sin_cache.as_f32()?.buffer().as_ptr() as *const f32;
            let cos_ptr = cos_cache.as_f32()?.buffer().as_ptr() as *const f32;
            unsafe {
                rope_kernel_cu(
                    dim_i32, kv_dim_i32, head_size_i32,
                    q_ptr, k_ptr, pos_ptr, seq_len_i32,
                    sin_ptr, cos_ptr, stream,
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

// ─────────────────── sin/cos cache 计算（不受 rope 合并影响）───────────────────

unsafe extern "C" {
    pub fn sin_cos_cache_calc_cu(
        head_size: i32,
        max_seq_len: i32,
        rope_theta: f32,
        sin_cache: *mut f32,
        cos_cache: *mut f32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn sin_cos_cache_calc_cu_bf16(
        head_size: i32,
        max_seq_len: i32,
        rope_theta: f32,
        sin_cache: *mut half::bf16,
        cos_cache: *mut half::bf16,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_pos_emb: f32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn sin_cos_cache_calc_cu_fp16(
        head_size: i32,
        max_seq_len: i32,
        rope_theta: f32,
        sin_cache: *mut half::f16,
        cos_cache: *mut half::f16,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_pos_emb: f32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// 计算并填充 RoPE 的 sin/cos 缓存。
#[allow(clippy::too_many_arguments)]
pub fn sin_cos_cache_calc_cuda(
    head_size: usize,
    max_seq_len: usize,
    rope_theta: f32,
    sin_cache: &mut Tensor,
    cos_cache: &mut Tensor,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_pos_emb: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = sin_cache.dtype();
    let head_size_i32 = head_size as i32;
    let max_seq_len_i32 = max_seq_len as i32;
    let stream = CudaConfig::resolve_stream(cuda_config);

    match dtype {
        crate::base::DataType::F32 => {
            let sin_ptr = sin_cache.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            let cos_ptr = cos_cache.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe {
                sin_cos_cache_calc_cu(head_size_i32, max_seq_len_i32, rope_theta, sin_ptr, cos_ptr, stream);
            }
        }
        crate::base::DataType::BF16 => {
            let sin_ptr = sin_cache.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let cos_ptr = cos_cache.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe {
                sin_cos_cache_calc_cu_bf16(
                    head_size_i32, max_seq_len_i32, rope_theta, sin_ptr, cos_ptr,
                    factor, low_freq_factor, high_freq_factor, original_max_pos_emb, stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            let sin_ptr = sin_cache.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            let cos_ptr = cos_cache.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe {
                sin_cos_cache_calc_cu_fp16(
                    head_size_i32, max_seq_len_i32, rope_theta, sin_ptr, cos_ptr,
                    factor, low_freq_factor, high_freq_factor, original_max_pos_emb, stream,
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
