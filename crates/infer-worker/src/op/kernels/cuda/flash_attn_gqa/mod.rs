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
        is_causal: i32,
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
        workspace: *mut f32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn flash_decoding_cu_fp16(
        q_ptr: *const half::f16,
        k_ptr: *const half::f16,
        v_ptr: *const half::f16,
        o_ptr: *mut half::f16,
        workspace: *mut f32,
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
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_cute_128x64x64_tile_fp16(
        q_ptr: *const half::f16,
        k_ptr: *const half::f16,
        v_ptr: *const half::f16,
        o_ptr: *mut half::f16,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn flash_decoding_cu_bf16_hdim128(
        q_ptr: *const half::bf16,
        k_ptr: *const half::bf16,
        v_ptr: *const half::bf16,
        o_ptr: *mut half::bf16,
        workspace: *mut f32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn flash_decoding_cu_fp16_hdim128(
        q_ptr: *const half::f16,
        k_ptr: *const half::f16,
        v_ptr: *const half::f16,
        o_ptr: *mut half::f16,
        workspace: *mut f32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_cute_bf16_hdim128(
        q_ptr: *const half::bf16,
        k_ptr: *const half::bf16,
        v_ptr: *const half::bf16,
        o_ptr: *mut half::bf16,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn launch_flash_attn_cute_fp16_hdim128(
        q_ptr: *const half::f16,
        k_ptr: *const half::f16,
        v_ptr: *const half::f16,
        o_ptr: *mut half::f16,
        q_seq_len: i32,
        kv_seq_len: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    // Batched flash-decoding (BF16, head_dim = 64)
    pub fn flash_decoding_cu_bf16_batch(
        q_flat: *const half::bf16,
        k_ptrs_dev: *const *const half::bf16,
        v_ptrs_dev: *const *const half::bf16,
        o_flat: *mut half::bf16,
        workspace: *mut f32,
        seq_lens_dev: *const i32,
        batch_size: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        q_row_stride: i32,
        o_row_stride: i32,
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
    is_causal: bool,
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
    let stream = CudaConfig::resolve_stream(cuda_config);
    let is_causal_i32: i32 = if is_causal { 1 } else { 0 };

    // --- 4. 根据数据类型分发 ---
    match dtype {
        crate::base::DataType::F32 => {
            let q_ptr = input_q.as_f32()?.data_ptr();
            let k_ptr = input_k_cache.as_f32()?.data_ptr();
            let v_ptr = input_v_cache.as_f32()?.data_ptr();
            let o_ptr = output_o.as_f32_mut()?.data_ptr_mut();
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
                        is_causal_i32,
                        stream,
                    );
                }
            }
        }
        crate::base::DataType::BF16 => {
            let q_ptr = input_q.as_bf16()?.data_ptr();
            let k_ptr = input_k_cache.as_bf16()?.data_ptr();
            let v_ptr = input_v_cache.as_bf16()?.data_ptr();
            let o_ptr = output_o.as_bf16_mut()?.data_ptr_mut();

            unsafe {
                if q_seq_len == 1 {
                    // Decode 路径走 split-K。调用方 CudaConfig 必须已经用
                    // `.with_flash_decode(num_q_heads, head_dim, max_batch_size)` 分配了 workspace。
                    let cfg = cuda_config.ok_or_else(|| Error::InvalidArgument(
                        "flash_attn_gqa BF16 decode path requires CudaConfig".into()
                    ))?;
                    if cfg.flash_decode_workspace.is_null() {
                        return Err(Error::InvalidArgument(
                            "CudaConfig.flash_decode_workspace not initialized; \
                             construct the config as \
                             `CudaConfig::new()?.with_flash_decode(num_q_heads, head_dim, max_batch_size)?`".into()
                        ).into());
                    }
                    let workspace_ptr = cfg.flash_decode_workspace as *mut f32;

                    if head_dim_i32 <= 64 {
                        flash_decoding_cu_bf16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            o_ptr,
                            workspace_ptr,
                            current_kv_len_gpu,
                            num_q_heads_i32,
                            num_kv_heads_i32,
                            head_dim_i32,
                            stream,
                        );
                    } else {
                        flash_decoding_cu_bf16_hdim128(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            o_ptr,
                            workspace_ptr,
                            current_kv_len_gpu,
                            num_q_heads_i32,
                            num_kv_heads_i32,
                            head_dim_i32,
                            stream,
                        );
                    }
                } else if head_dim_i32 <= 64 {
                    launch_flash_attn_cute_128x64x64_tile(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        is_causal_i32,
                        stream,
                    );
                } else {
                    launch_flash_attn_cute_bf16_hdim128(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        is_causal_i32,
                        stream,
                    );
                }
            }
        }
        crate::base::DataType::F16 => {
            let q_ptr = input_q.as_f16()?.data_ptr();
            let k_ptr = input_k_cache.as_f16()?.data_ptr();
            let v_ptr = input_v_cache.as_f16()?.data_ptr();
            let o_ptr = output_o.as_f16_mut()?.data_ptr_mut();

            unsafe {
                if q_seq_len == 1 {
                    // Decode 路径走 split-K（与 BF16 同结构）。调用方 CudaConfig 必须已经用
                    // `.with_flash_decode(num_q_heads, head_dim, max_batch_size)` 分配了 workspace。
                    let cfg = cuda_config.ok_or_else(|| Error::InvalidArgument(
                        "flash_attn_gqa F16 decode path requires CudaConfig".into()
                    ))?;
                    if cfg.flash_decode_workspace.is_null() {
                        return Err(Error::InvalidArgument(
                            "CudaConfig.flash_decode_workspace not initialized; \
                             construct the config as \
                             `CudaConfig::new()?.with_flash_decode(num_q_heads, head_dim, max_batch_size)?`".into()
                        ).into());
                    }
                    let workspace_ptr = cfg.flash_decode_workspace as *mut f32;

                    if head_dim_i32 <= 64 {
                        flash_decoding_cu_fp16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            o_ptr,
                            workspace_ptr,
                            current_kv_len_gpu,
                            num_q_heads_i32,
                            num_kv_heads_i32,
                            head_dim_i32,
                            stream,
                        );
                    } else {
                        flash_decoding_cu_fp16_hdim128(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            o_ptr,
                            workspace_ptr,
                            current_kv_len_gpu,
                            num_q_heads_i32,
                            num_kv_heads_i32,
                            head_dim_i32,
                            stream,
                        );
                    }
                } else if head_dim_i32 <= 64 {
                    launch_flash_attn_cute_128x64x64_tile_fp16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        is_causal_i32,
                        stream,
                    );
                } else {
                    launch_flash_attn_cute_fp16_hdim128(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        q_seq_len_i32,
                        current_kv_len_gpu,
                        num_q_heads_i32,
                        num_kv_heads_i32,
                        is_causal_i32,
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
/// Batched flash-decoding (BF16, head_dim=64):
/// - `q`  : [B, num_q_heads, head_dim]  CUDA tensor
/// - `o`  : [B, num_q_heads, head_dim]  CUDA tensor (输出)
/// - `k_caches` / `v_caches` : B 个独立 cache (每个 `[max_seq_len, num_kv_heads, head_dim]`)
/// - `kv_lens_dev` : [B] i32 device tensor，每 seq 的有效 kv 长度 - 1（kernel 内部会 +1）
/// - `k_ptrs_dev` / `v_ptrs_dev` : 预分配的 device 指针数组 buffer（容量 ≥ B*8 bytes）
///
/// `cuda_config` 必须已用 `with_flash_decode(num_q_heads, head_dim, max_batch_size)`
/// 预分配了足够覆盖 B 段的 workspace；否则本函数会返回 workspace too small 错误。
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_decoding_batch_bf16(
    q: &Tensor,
    k_caches: &[&Tensor],
    v_caches: &[&Tensor],
    o: &mut Tensor,
    kv_lens_dev: &Tensor,
    k_ptrs_dev: *mut u64,
    v_ptrs_dev: *mut u64,
    cuda_config: &CudaConfig,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let batch_size = k_caches.len();
    assert_eq!(v_caches.len(), batch_size);
    if batch_size == 0 { return Ok(()); }
    if head_dim != 64 {
        return Err(Error::InvalidArgument(format!(
            "flash_decoding_batch_bf16 currently only supports head_dim=64, got {}", head_dim
        )).into());
    }

    let stream = CudaConfig::resolve_stream(Some(cuda_config));

    // 收集 host 指针数组
    let mut k_host: Vec<u64> = Vec::with_capacity(batch_size);
    let mut v_host: Vec<u64> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        k_host.push(k_caches[i].as_bf16()?.data_ptr() as u64);
        v_host.push(v_caches[i].as_bf16()?.data_ptr() as u64);
    }
    let bytes = batch_size * std::mem::size_of::<u64>();
    unsafe {
        crate::cuda_check!(cuda::ffi::cudaMemcpyAsync(
            k_ptrs_dev as *mut _,
            k_host.as_ptr() as *const _,
            bytes,
            cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream,
        ))?;
        crate::cuda_check!(cuda::ffi::cudaMemcpyAsync(
            v_ptrs_dev as *mut _,
            v_host.as_ptr() as *const _,
            bytes,
            cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream,
        ))?;
        crate::cuda_check!(cuda::ffi::cudaStreamSynchronize(stream))?;
    }
    drop(k_host);
    drop(v_host);

    // 检查 workspace 足够大
    let needed_per_seq = num_q_heads * crate::cuda::FLASH_DECODE_N_SPLIT * (2 + head_dim);
    let needed_bytes = batch_size * needed_per_seq * std::mem::size_of::<f32>();
    if cuda_config.flash_decode_workspace_size < needed_bytes {
        return Err(Error::InvalidArgument(format!(
            "flash_decoding_batch_bf16: workspace too small ({} < {})",
            cuda_config.flash_decode_workspace_size, needed_bytes
        )).into());
    }

    let q_ptr = q.as_bf16()?.data_ptr();
    let o_ptr = o.as_bf16_mut()?.data_ptr_mut();
    let kv_lens_ptr = kv_lens_dev.as_i32()?.data_ptr();
    let workspace_ptr = cuda_config.flash_decode_workspace as *mut f32;
    let qo_stride = (num_q_heads * head_dim) as i32;

    unsafe {
        flash_decoding_cu_bf16_batch(
            q_ptr,
            k_ptrs_dev as *const *const half::bf16,
            v_ptrs_dev as *const *const half::bf16,
            o_ptr,
            workspace_ptr,
            kv_lens_ptr,
            batch_size as i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            qo_stride, qo_stride,
            stream,
        );
    }
    Ok(())
}

/// Batched flash-decoding (BF16, hdim=64) 的 "launch-ready" 版：
/// `k_ptrs_dev` / `v_ptrs_dev` 必须已经包含 B 个 cache 的正确指针，
/// 不做 H2D copy / stream sync → 可 CUDA Graph 捕获。
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_decoding_batch_bf16_launch_ready(
    q: &Tensor,
    o: &mut Tensor,
    kv_lens_dev: &Tensor,
    k_ptrs_dev: *mut u64,
    v_ptrs_dev: *mut u64,
    cuda_config: &CudaConfig,
    batch_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_row_stride: usize,
    o_row_stride: usize,
    q_col_offset: usize,
    o_col_offset: usize,
) -> Result<()> {
    if batch_size == 0 { return Ok(()); }
    if head_dim != 64 {
        return Err(Error::InvalidArgument(format!(
            "flash_decoding_batch_bf16_launch_ready only supports head_dim=64, got {}", head_dim
        )).into());
    }

    let needed_per_seq = num_q_heads * crate::cuda::FLASH_DECODE_N_SPLIT * (2 + head_dim);
    let needed_bytes = batch_size * needed_per_seq * std::mem::size_of::<f32>();
    if cuda_config.flash_decode_workspace_size < needed_bytes {
        return Err(Error::InvalidArgument(format!(
            "flash_decoding_batch_bf16_launch_ready: workspace too small ({} < {})",
            cuda_config.flash_decode_workspace_size, needed_bytes
        )).into());
    }

    let stream = CudaConfig::resolve_stream(Some(cuda_config));
    let q_base = q.as_bf16()?.data_ptr();
    let o_base = o.as_bf16_mut()?.data_ptr_mut();
    let q_ptr = unsafe { q_base.add(q_col_offset) };
    let o_ptr = unsafe { o_base.add(o_col_offset) };
    let kv_lens_ptr = kv_lens_dev.as_i32()?.data_ptr();
    let workspace_ptr = cuda_config.flash_decode_workspace as *mut f32;

    unsafe {
        flash_decoding_cu_bf16_batch(
            q_ptr,
            k_ptrs_dev as *const *const half::bf16,
            v_ptrs_dev as *const *const half::bf16,
            o_ptr,
            workspace_ptr,
            kv_lens_ptr,
            batch_size as i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q_row_stride as i32,
            o_row_stride as i32,
            stream,
        );
    }
    Ok(())
}
