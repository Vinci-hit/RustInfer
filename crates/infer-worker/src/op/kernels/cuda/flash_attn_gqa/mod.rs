use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};

// ============================================================================
// FFI into the CUDA kernel library.
//
// Only two kernels are exposed:
//   * launch_flash_attn_prefill_{bf16,fp16}          – prefill (q_len > 1)
//   * launch_flash_attn_batched_decode_{bf16,fp16}   – batched decode (q_len=1)
// F32 attention runs on CPU; there is no CUDA F32 path.
// ============================================================================
unsafe extern "C" {
    // ---- Stride-aware prefill ----
    pub fn launch_flash_attn_prefill_bf16(
        q: *const half::bf16, qsb: i64, qss: i64, qsh: i64,
        k: *const half::bf16, ksb: i64, kss: i64, ksh: i64,
        v: *const half::bf16, vsb: i64, vss: i64, vsh: i64,
        o: *mut   half::bf16, osb: i64, oss: i64, osh: i64,
        batch: i32, q_len: i32, kv_len: i32,
        num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32, is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_prefill_fp16(
        q: *const half::f16, qsb: i64, qss: i64, qsh: i64,
        k: *const half::f16, ksb: i64, kss: i64, ksh: i64,
        v: *const half::f16, vsb: i64, vss: i64, vsh: i64,
        o: *mut   half::f16, osb: i64, oss: i64, osh: i64,
        batch: i32, q_len: i32, kv_len: i32,
        num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32, is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    // ---- Batched decode (split-KV, per-request KV cache, graph-friendly) ----
    pub fn flash_attn_batched_decode_workspace_bytes(
        batch: i32, num_q_heads: i32, head_dim: i32,
    ) -> i64;

    pub fn launch_flash_attn_batched_decode_bf16(
        q: *const half::bf16, qsb: i64, qsh: i64,
        k_cache_ptrs: *const *const half::bf16,
        v_cache_ptrs: *const *const half::bf16,
        kv_stride_s: i64, kv_stride_h: i64,
        o: *mut   half::bf16, osb: i64, osh: i64,
        req_to_slot: *const i32,
        kv_lens: *const i32,
        workspace: *mut f32,
        batch: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_batched_decode_fp16(
        q: *const half::f16, qsb: i64, qsh: i64,
        k_cache_ptrs: *const *const half::f16,
        v_cache_ptrs: *const *const half::f16,
        kv_stride_s: i64, kv_stride_h: i64,
        o: *mut   half::f16, osb: i64, osh: i64,
        req_to_slot: *const i32,
        kv_lens: *const i32,
        workspace: *mut f32,
        batch: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn launch_flash_attn_paged_decode_bf16(
        q: *const half::bf16, qsb: i64, qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16, osb: i64, osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        batch: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn launch_flash_attn_paged_decode_fp16(
        q: *const half::f16, qsb: i64, qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16, osb: i64, osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        batch: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn launch_flash_attn_paged_ragged_bf16(
        q: *const half::bf16, qss: i64, qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16, oss: i64, osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        batch: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    pub fn launch_flash_attn_paged_ragged_fp16(
        q: *const half::f16, qss: i64, qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16, oss: i64, osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        batch: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    // ---- Ragged attention (variable q_len / kv_len per request) ----
    pub fn launch_flash_attn_ragged_bf16(
        q: *const half::bf16, qss: i64, qsh: i64,
        k_cache_ptrs: *const *const half::bf16,
        v_cache_ptrs: *const *const half::bf16,
        kv_stride_s: i64, kv_stride_h: i64,
        o: *mut   half::bf16, oss: i64, osh: i64,
        req_to_slot: *const i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        block2req: *const i32,
        block2tile: *const i32,
        total_q_tiles: i32,
        num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32, is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    pub fn launch_flash_attn_ragged_fp16(
        q: *const half::f16, qss: i64, qsh: i64,
        k_cache_ptrs: *const *const half::f16,
        v_cache_ptrs: *const *const half::f16,
        kv_stride_s: i64, kv_stride_h: i64,
        o: *mut   half::f16, oss: i64, osh: i64,
        req_to_slot: *const i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        block2req: *const i32,
        block2tile: *const i32,
        total_q_tiles: i32,
        num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32, is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

// ============================================================================
// Prefill wrapper
// ============================================================================

/// Extract stride / length arguments for the stride-aware prefill kernel.
///
/// Input tensors are expected to be 2-D row-major:
///   Q, O : [q_seq_len,    num_q_heads  * head_dim]
///   K, V : [max_kv_seq_len, num_kv_heads * head_dim]
///
/// `current_kv_len_host` is the "already-cached" KV length (past history,
/// not including the new prefill tokens).  `kv_len_total = past + new`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
unsafe fn prefill_stride_args(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    q_seq_len: usize,
    current_kv_len_host: i32,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(i64, i64, i64,
             i64, i64, i64,
             i64, i64, i64,
             i64, i64, i64,
             i32, i32)> {
    if q.shape().len() != 2 || k.shape().len() != 2 || v.shape().len() != 2 || o.shape().len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "flash_attn_gqa prefill expects 2-D tensors (got shapes \
             Q={:?}, K={:?}, V={:?}, O={:?})",
            q.shape(), k.shape(), v.shape(), o.shape()
        )).into());
    }

    let qss = q.strides()[0] as i64;
    let kss = k.strides()[0] as i64;
    let vss = v.strides()[0] as i64;
    let oss = o.strides()[0] as i64;
    let qsh = head_dim as i64;
    let ksh = head_dim as i64;
    let vsh = head_dim as i64;
    let osh = head_dim as i64;
    let qsb = (q_seq_len as i64) * qss;
    let ksb = (k.shape()[0] as i64) * kss;
    let vsb = (v.shape()[0] as i64) * vss;
    let osb = (o.shape()[0] as i64) * oss;

    let kv_len_total = current_kv_len_host as i64 + q_seq_len as i64;

    if q.shape()[1] != num_q_heads * head_dim {
        return Err(Error::InvalidArgument(format!(
            "Q last-dim mismatch: {} vs num_q_heads*head_dim={}",
            q.shape()[1], num_q_heads * head_dim
        )).into());
    }
    if k.shape()[1] != num_kv_heads * head_dim || v.shape()[1] != num_kv_heads * head_dim {
        return Err(Error::InvalidArgument(format!(
            "K/V last-dim mismatch: K={}, V={}, expected num_kv_heads*head_dim={}",
            k.shape()[1], v.shape()[1], num_kv_heads * head_dim
        )).into());
    }
    if (kv_len_total as usize) > k.shape()[0] {
        return Err(Error::InvalidArgument(format!(
            "kv_len_total={} exceeds K.shape[0]={}", kv_len_total, k.shape()[0]
        )).into());
    }

    Ok((qsb, qss, qsh,
        ksb, kss, ksh,
        vsb, vss, vsh,
        osb, oss, osh,
        q_seq_len as i32, kv_len_total as i32))
}

/// Prefill-only CUDA entry point. BF16 / FP16 supported.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_gqa_prefill(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    q_seq_len: usize,
    current_kv_len_host: i32,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_causal: bool,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = input_q.dtype();
    if dtype != input_k_cache.dtype() || dtype != input_v_cache.dtype() || dtype != output_o.dtype() {
        return Err(Error::InvalidArgument(
            "flash_attn_gqa_prefill: Q/K/V/O must share dtype".into()
        ).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let is_causal_i32 = if is_causal { 1 } else { 0 };
    let (qsb, qss, qsh,
         ksb, kss, ksh,
         vsb, vss, vsh,
         osb, oss, osh,
         q_len, kv_len_total) = unsafe { prefill_stride_args(
        input_q, input_k_cache, input_v_cache, output_o,
        q_seq_len, current_kv_len_host,
        num_q_heads, num_kv_heads, head_dim,
    )? };
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    match dtype {
        crate::base::DataType::BF16 => unsafe {
            launch_flash_attn_prefill_bf16(
                input_q.as_bf16()?.data_ptr(),         qsb, qss, qsh,
                input_k_cache.as_bf16()?.data_ptr(),   ksb, kss, ksh,
                input_v_cache.as_bf16()?.data_ptr(),   vsb, vss, vsh,
                output_o.as_bf16_mut()?.data_ptr_mut(),osb, oss, osh,
                1, q_len, kv_len_total,
                num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale, is_causal_i32,
                stream,
            );
        },
        crate::base::DataType::F16 => unsafe {
            launch_flash_attn_prefill_fp16(
                input_q.as_f16()?.data_ptr(),          qsb, qss, qsh,
                input_k_cache.as_f16()?.data_ptr(),    ksb, kss, ksh,
                input_v_cache.as_f16()?.data_ptr(),    vsb, vss, vsh,
                output_o.as_f16_mut()?.data_ptr_mut(), osb, oss, osh,
                1, q_len, kv_len_total,
                num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale, is_causal_i32,
                stream,
            );
        },
        other => {
            return Err(Error::InvalidArgument(format!(
                "flash_attn_gqa_prefill: unsupported dtype {:?} \
                 (only BF16 / F16 supported)", other
            )).into());
        }
    }
    Ok(())
}

// ============================================================================
// Batched decode wrapper
// ============================================================================

/// Query how many bytes of `float` workspace the batched decode kernel needs
/// for a given (batch, num_q_heads, head_dim).  Must be called on host.
pub fn batched_decode_workspace_bytes(batch: usize, num_q_heads: usize, head_dim: usize) -> usize {
    unsafe {
        flash_attn_batched_decode_workspace_bytes(
            batch as i32, num_q_heads as i32, head_dim as i32,
        ) as usize
    }
}

/// Batched Flash-Decoding (q_len = 1).
///
/// The caller supplies:
///   * `q` / `o`                 – `[batch, num_q_heads, head_dim]` contiguous device tensors
///   * `k_cache_ptrs_dev`        – device array `[*const Elem; max_slots]`, filled with
///                                 each KV-cache slot's base pointer (stable across steps)
///   * `v_cache_ptrs_dev`        – same for V
///   * `kv_stride_s/h`           – per-token / per-kv-head stride within a single cache (elements)
///   * `req_to_slot_dev`         – device `[i32; batch]`, which slot each request currently uses
///   * `kv_lens_dev`             – device `[i32; batch]`, per-request KV length
///   * `workspace`               – `[f32]` with at least `batched_decode_workspace_bytes(...)` bytes
///
/// All device arrays must have stable addresses (graph-capture safe).
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_batched_decode(
    q: &Tensor,
    k_cache_ptrs_dev: *const *const std::ffi::c_void,
    v_cache_ptrs_dev: *const *const std::ffi::c_void,
    kv_stride_s: i64,
    kv_stride_h: i64,
    o: &mut Tensor,
    req_to_slot_dev: *const i32,
    kv_lens_dev: *const i32,
    workspace: *mut f32,
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = q.dtype();
    if dtype != o.dtype() {
        return Err(Error::InvalidArgument(
            "flash_attn_batched_decode: q and o must share dtype".into()
        ).into());
    }
    if q.shape().len() != 3 || o.shape().len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "flash_attn_batched_decode: q/o must be 3-D [batch, num_q_heads, head_dim] \
             (got q={:?}, o={:?})", q.shape(), o.shape()
        )).into());
    }
    if q.shape()[0] != batch || o.shape()[0] != batch {
        return Err(Error::InvalidArgument(format!(
            "batch mismatch: q.shape[0]={}, o.shape[0]={}, batch={}",
            q.shape()[0], o.shape()[0], batch,
        )).into());
    }
    if q.shape()[1] != num_q_heads || q.shape()[2] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "q shape mismatch: got {:?}, expected [_, {}, {}]",
            q.shape(), num_q_heads, head_dim,
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let qsb = q.strides()[0] as i64;
    let qsh = q.strides()[1] as i64;
    let osb = o.strides()[0] as i64;
    let osh = o.strides()[1] as i64;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    match dtype {
        crate::base::DataType::BF16 => unsafe {
            launch_flash_attn_batched_decode_bf16(
                q.as_bf16()?.data_ptr(), qsb, qsh,
                k_cache_ptrs_dev as *const *const half::bf16,
                v_cache_ptrs_dev as *const *const half::bf16,
                kv_stride_s, kv_stride_h,
                o.as_bf16_mut()?.data_ptr_mut(), osb, osh,
                req_to_slot_dev, kv_lens_dev,
                workspace,
                batch as i32, num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale,
                stream,
            );
        },
        crate::base::DataType::F16 => unsafe {
            launch_flash_attn_batched_decode_fp16(
                q.as_f16()?.data_ptr(), qsb, qsh,
                k_cache_ptrs_dev as *const *const half::f16,
                v_cache_ptrs_dev as *const *const half::f16,
                kv_stride_s, kv_stride_h,
                o.as_f16_mut()?.data_ptr_mut(), osb, osh,
                req_to_slot_dev, kv_lens_dev,
                workspace,
                batch as i32, num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale,
                stream,
            );
        },
        other => {
            return Err(Error::InvalidArgument(format!(
                "flash_attn_batched_decode: unsupported dtype {:?} \
                 (only BF16 / F16 supported)", other
            )).into());
        }
    }
    Ok(())
}

// ============================================================================
// Ragged attention wrapper
// ============================================================================

/// Ragged attention over packed Q / O + per-request KV caches.
///
/// Shapes:
///   q, o         : [total_q_tokens, num_q_heads, head_dim]
///   KV per slot  : caller's layout, described by (kv_stride_s, kv_stride_h)
///
/// All device pointers must have stable addresses across graph replays.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_ragged(
    q: &Tensor,
    k_cache_ptrs_dev: *const *const std::ffi::c_void,
    v_cache_ptrs_dev: *const *const std::ffi::c_void,
    kv_stride_s: i64,
    kv_stride_h: i64,
    o: &mut Tensor,
    req_to_slot_dev: *const i32,
    kv_lens_dev: *const i32,
    cu_q_lens_dev: *const i32,
    block2req_dev: *const i32,
    block2tile_dev: *const i32,
    total_q_tiles: i32,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_causal: bool,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = q.dtype();
    if dtype != o.dtype() {
        return Err(Error::InvalidArgument(
            "flash_attn_ragged: q and o must share dtype".into()
        ).into());
    }
    if q.shape().len() != 3 || o.shape().len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "flash_attn_ragged: q/o must be 3-D [total_q, Hq, HD] (got q={:?}, o={:?})",
            q.shape(), o.shape()
        )).into());
    }
    if q.shape()[1] != num_q_heads || q.shape()[2] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "q shape {:?} incompatible with [_, {}, {}]",
            q.shape(), num_q_heads, head_dim
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let qss = q.strides()[0] as i64;
    let qsh = q.strides()[1] as i64;
    let oss = o.strides()[0] as i64;
    let osh = o.strides()[1] as i64;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let is_causal_i32 = if is_causal { 1 } else { 0 };

    match dtype {
        crate::base::DataType::BF16 => unsafe {
            launch_flash_attn_ragged_bf16(
                q.as_bf16()?.data_ptr(), qss, qsh,
                k_cache_ptrs_dev as *const *const half::bf16,
                v_cache_ptrs_dev as *const *const half::bf16,
                kv_stride_s, kv_stride_h,
                o.as_bf16_mut()?.data_ptr_mut(), oss, osh,
                req_to_slot_dev, kv_lens_dev, cu_q_lens_dev,
                block2req_dev, block2tile_dev, total_q_tiles,
                num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale, is_causal_i32,
                stream,
            );
        },
        crate::base::DataType::F16 => unsafe {
            launch_flash_attn_ragged_fp16(
                q.as_f16()?.data_ptr(), qss, qsh,
                k_cache_ptrs_dev as *const *const half::f16,
                v_cache_ptrs_dev as *const *const half::f16,
                kv_stride_s, kv_stride_h,
                o.as_f16_mut()?.data_ptr_mut(), oss, osh,
                req_to_slot_dev, kv_lens_dev, cu_q_lens_dev,
                block2req_dev, block2tile_dev, total_q_tiles,
                num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale, is_causal_i32,
                stream,
            );
        },
        other => {
            return Err(Error::InvalidArgument(format!(
                "flash_attn_ragged: unsupported dtype {:?} (only BF16 / F16)", other
            )).into());
        }
    }
    Ok(())
}

/// Paged Flash-Decoding (q_len = 1) over a global KV pool.
///
/// K/V pool layout is `[num_blocks, block_size, num_kv_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged_decode(
    q: &Tensor,
    k_pool: *const std::ffi::c_void,
    v_pool: *const std::ffi::c_void,
    o: &mut Tensor,
    block_tables_dev: *const u32,
    max_blocks_per_seq: usize,
    block_size: usize,
    kv_lens_dev: *const i32,
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = q.dtype();
    if dtype != o.dtype() {
        return Err(Error::InvalidArgument(
            "flash_attn_paged_decode: q and o must share dtype".into()
        ).into());
    }
    if q.shape().len() != 3 || o.shape().len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "flash_attn_paged_decode: q/o must be 3-D [batch, num_q_heads, head_dim] \
             (got q={:?}, o={:?})", q.shape(), o.shape()
        )).into());
    }
    if q.shape()[0] != batch || o.shape()[0] != batch {
        return Err(Error::InvalidArgument(format!(
            "paged decode batch mismatch: q.shape[0]={}, o.shape[0]={}, batch={}",
            q.shape()[0], o.shape()[0], batch,
        )).into());
    }
    if q.shape()[1] != num_q_heads || q.shape()[2] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "paged decode q shape mismatch: got {:?}, expected [_, {}, {}]",
            q.shape(), num_q_heads, head_dim,
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let qsb = q.strides()[0] as i64;
    let qsh = q.strides()[1] as i64;
    let osb = o.strides()[0] as i64;
    let osh = o.strides()[1] as i64;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    match dtype {
        crate::base::DataType::BF16 => unsafe {
            launch_flash_attn_paged_decode_bf16(
                q.as_bf16()?.data_ptr(), qsb, qsh,
                k_pool as *const half::bf16,
                v_pool as *const half::bf16,
                o.as_bf16_mut()?.data_ptr_mut(), osb, osh,
                block_tables_dev,
                max_blocks_per_seq as i32,
                block_size as i32,
                kv_lens_dev,
                batch as i32, num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale,
                stream,
            );
        },
        crate::base::DataType::F16 => unsafe {
            launch_flash_attn_paged_decode_fp16(
                q.as_f16()?.data_ptr(), qsb, qsh,
                k_pool as *const half::f16,
                v_pool as *const half::f16,
                o.as_f16_mut()?.data_ptr_mut(), osb, osh,
                block_tables_dev,
                max_blocks_per_seq as i32,
                block_size as i32,
                kv_lens_dev,
                batch as i32, num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale,
                stream,
            );
        },
        other => {
            return Err(Error::InvalidArgument(format!(
                "flash_attn_paged_decode: unsupported dtype {:?} (only BF16 / F16 supported)", other
            )).into());
        }
    }
    Ok(())
}

/// Paged ragged/prefill attention over a global KV pool.
///
/// Q/O layout: `[total_q_tokens, num_q_heads, head_dim]`.
/// K/V pool layout: `[num_blocks, block_size, num_kv_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged_ragged(
    q: &Tensor,
    k_pool: *const std::ffi::c_void,
    v_pool: *const std::ffi::c_void,
    o: &mut Tensor,
    block_tables_dev: *const u32,
    max_blocks_per_seq: usize,
    block_size: usize,
    kv_lens_dev: *const i32,
    cu_q_lens_dev: *const i32,
    batch: usize,
    total_q_tokens: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_causal: bool,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dtype = q.dtype();
    if dtype != o.dtype() {
        return Err(Error::InvalidArgument(
            "flash_attn_paged_ragged: q and o must share dtype".into()
        ).into());
    }
    if q.shape().len() != 3 || o.shape().len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "flash_attn_paged_ragged: q/o must be 3-D [total_q_tokens, num_q_heads, head_dim] \
             (got q={:?}, o={:?})", q.shape(), o.shape()
        )).into());
    }
    if q.shape()[0] != total_q_tokens || o.shape()[0] != total_q_tokens {
        return Err(Error::InvalidArgument(format!(
            "paged ragged token mismatch: q.shape[0]={}, o.shape[0]={}, total_q_tokens={}",
            q.shape()[0], o.shape()[0], total_q_tokens,
        )).into());
    }
    if q.shape()[1] != num_q_heads || q.shape()[2] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "paged ragged q shape mismatch: got {:?}, expected [_, {}, {}]",
            q.shape(), num_q_heads, head_dim,
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let qss = q.strides()[0] as i64;
    let qsh = q.strides()[1] as i64;
    let oss = o.strides()[0] as i64;
    let osh = o.strides()[1] as i64;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal_i = if is_causal { 1 } else { 0 };

    match dtype {
        crate::base::DataType::BF16 => unsafe {
            launch_flash_attn_paged_ragged_bf16(
                q.as_bf16()?.data_ptr(), qss, qsh,
                k_pool as *const half::bf16,
                v_pool as *const half::bf16,
                o.as_bf16_mut()?.data_ptr_mut(), oss, osh,
                block_tables_dev,
                max_blocks_per_seq as i32,
                block_size as i32,
                kv_lens_dev,
                cu_q_lens_dev,
                batch as i32,
                total_q_tokens as i32,
                num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale,
                causal_i,
                stream,
            );
        },
        crate::base::DataType::F16 => unsafe {
            launch_flash_attn_paged_ragged_fp16(
                q.as_f16()?.data_ptr(), qss, qsh,
                k_pool as *const half::f16,
                v_pool as *const half::f16,
                o.as_f16_mut()?.data_ptr_mut(), oss, osh,
                block_tables_dev,
                max_blocks_per_seq as i32,
                block_size as i32,
                kv_lens_dev,
                cu_q_lens_dev,
                batch as i32,
                total_q_tokens as i32,
                num_q_heads as i32, num_kv_heads as i32, head_dim as i32,
                scale,
                causal_i,
                stream,
            );
        },
        other => {
            return Err(Error::InvalidArgument(format!(
                "flash_attn_paged_ragged: unsupported dtype {:?} (only BF16 / F16 supported)", other
            )).into());
        }
    }
    Ok(())
}
