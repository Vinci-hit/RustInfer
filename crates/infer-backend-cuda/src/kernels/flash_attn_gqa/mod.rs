//! Paged ragged / paged decode attention wrapper.
//!
//! - `BatchKind::DecodeOnly` → cuDNN frontend SDPA, then Flash fallback
//! - `BatchKind::Ragged`     → `launch_flash_attn_paged_ragged_cute_*`
//!
//! Both kernels read K/V from a single `[num_blocks, block_size, kv_dim]`
//! pool per layer; per-seq routing comes from the device `block_tables`
//! tensor (`[batch, max_blocks_per_seq]` u32).

use infer_core::kv::KvIndexTensors;
use infer_core::plan;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};
use crate::Cuda;
use crate::ffi::{cudaStream_t, cudnnHandle_t};
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_flash_attn_paged_decode_bf16(
        q: *const half::bf16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        workspace: *mut f32,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    );
    fn launch_flash_attn_paged_decode_fp16(
        q: *const half::f16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        workspace: *mut f32,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    );
    fn flash_attn_batched_decode_workspace_bytes(
        batch: i32,
        num_q_heads: i32,
        head_dim: i32,
    ) -> i64;
    fn launch_cudnn_paged_decode_bf16(
        handle: cudnnHandle_t,
        q: *const half::bf16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        q_lens: *const i32,
        kv_lens: *const i32,
        num_blocks: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    fn launch_cudnn_paged_decode_fp16(
        handle: cudnnHandle_t,
        q: *const half::f16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        q_lens: *const i32,
        kv_lens: *const i32,
        num_blocks: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> i32;

    fn launch_flash_attn_paged_ragged_cute_bf16(
        q: *const half::bf16,
        qss: i64,
        qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16,
        oss: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        block2req: *const i32,
        block2tile: *const i32,
        total_q_tiles: i32,
        batch: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: cudaStream_t,
    );
    fn launch_flash_attn_paged_ragged_cute_fp16(
        q: *const half::f16,
        qss: i64,
        qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16,
        oss: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        block2req: *const i32,
        block2tile: *const i32,
        total_q_tiles: i32,
        batch: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: cudaStream_t,
    );
}

const DISABLE_CUDNN_ATTENTION_ENV: &str = "RUSTINFER_DISABLE_CUDNN_ATTENTION";
const STRICT_CUDNN_ATTENTION_ENV: &str = "RUSTINFER_STRICT_CUDNN_ATTENTION";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagedAttentionKind {
    DecodeOnly,
    Ragged,
}

#[derive(Clone, Copy)]
pub struct PagedAttentionPlan<'a> {
    pub kind: PagedAttentionKind,
    pub num_tokens: usize,
    pub batch: usize,
    /// Per-request query lengths (host), in row order. The fused mixed batch is
    /// laid out decode-first, so the leading run of `1`s is the decode prefix and
    /// the remainder are prefill chunks. Used to split a `Ragged` batch's
    /// attention by row type (decode → cuDNN decode SDPA; prefill → cuDNN
    /// bottom-right-causal SDPA), keeping q=1 rows off the CuTe ragged kernel.
    pub q_lens: &'a [i32],
    pub block_tables: &'a Tensor<i32, Cuda>,
    pub cu_q_lens: &'a Tensor<i32, Cuda>,
    pub kv_lens: &'a Tensor<i32, Cuda>,
    pub seq_lens_step: &'a Tensor<i32, Cuda>,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub block2req: &'a Tensor<i32, Cuda>,
    pub block2tile: &'a Tensor<i32, Cuda>,
    pub total_q_tiles: i32,
}

impl<'a> PagedAttentionPlan<'a> {
    pub fn from_v2(plan: &'a plan::BatchPlan, index: &'a KvIndexTensors<Cuda>) -> Self {
        let kind = match plan.kind {
            plan::BatchKind::DecodeOnly => PagedAttentionKind::DecodeOnly,
            plan::BatchKind::Ragged | plan::BatchKind::Spec { .. } => PagedAttentionKind::Ragged,
        };
        Self {
            kind,
            num_tokens: plan.num_tokens,
            batch: plan.batch,
            q_lens: &plan.q_lens,
            block_tables: &index.block_tables,
            cu_q_lens: &index.cu_q_lens,
            kv_lens: &index.kv_lens,
            seq_lens_step: &index.seq_lens_step,
            max_blocks_per_seq: plan.max_blocks_per_seq,
            block_size: plan.block_size,
            block2req: &index.block2req,
            block2tile: &index.block2tile,
            total_q_tiles: plan.total_q_tiles,
        }
    }
}

/// f32 element count needed by the paged-decode flash attention kernel for
/// a worst-case `(batch, num_q_heads, head_dim)`. Callers (`ForwardWorkspace`)
/// use this to size the long-lived attention scratch.
pub fn flash_decode_workspace_capacity_f32(
    batch: usize,
    num_q_heads: usize,
    head_dim: usize,
) -> usize {
    let bytes = unsafe {
        flash_attn_batched_decode_workspace_bytes(batch as i32, num_q_heads as i32, head_dim as i32)
    } as usize;
    (bytes + 3) / 4
}

/// Launch the CuTe paged ragged Flash kernel over a (sub)set of q tiles. The
/// kernel indexes q / kv_lens / block_tables by the *absolute* request id held
/// in `block2req[tile]` (and `cu_q_lens[req]`), so a fused batch's prefill
/// suffix runs by passing the full base pointers but with `block2req` /
/// `block2tile` shifted past the leading decode tiles and `total_q_tiles`
/// reduced to match. Always causal (chunked-prefill bottom-right via the
/// kernel's `kv_len - q_len` mask shift).
#[allow(clippy::too_many_arguments)]
unsafe fn launch_cute_ragged<T: Dtype>(
    q_ptr: *const T,
    q_stride_seq: i64,
    q_stride_head: i64,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    o_ptr: *mut T,
    o_stride_seq: i64,
    o_stride_head: i64,
    block_tables_ptr: *const u32,
    max_blocks_per_seq: i32,
    block_size: i32,
    kv_lens_ptr: *const i32,
    cu_q_lens_ptr: *const i32,
    block2req_ptr: *const i32,
    block2tile_ptr: *const i32,
    total_q_tiles: i32,
    batch: i32,
    num_tokens: i32,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
    stream: cudaStream_t,
) -> OpResult<()> {
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => launch_flash_attn_paged_ragged_cute_bf16(
                q_ptr as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                o_ptr as _,
                o_stride_seq,
                o_stride_head,
                block_tables_ptr,
                max_blocks_per_seq,
                block_size,
                kv_lens_ptr,
                cu_q_lens_ptr,
                block2req_ptr,
                block2tile_ptr,
                total_q_tiles,
                batch,
                num_tokens,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                1,
                stream,
            ),
            DataType::F16 => launch_flash_attn_paged_ragged_cute_fp16(
                q_ptr as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                o_ptr as _,
                o_stride_seq,
                o_stride_head,
                block_tables_ptr,
                max_blocks_per_seq,
                block_size,
                kv_lens_ptr,
                cu_q_lens_ptr,
                block2req_ptr,
                block2tile_ptr,
                total_q_tiles,
                batch,
                num_tokens,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                1,
                stream,
            ),
            _ => {
                return Err(OpError::Kernel(format!(
                    "attention_paged Ragged: dtype {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn try_cudnn_paged_decode<T: Dtype>(
    device: &Cuda,
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: PagedAttentionPlan<'_>,
    q_stride_seq: i64,
    q_stride_head: i64,
    o_stride_seq: i64,
    o_stride_head: i64,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
    batch: i32,
    stream: cudaStream_t,
) -> Option<i32> {
    if std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_some() {
        return None;
    }

    let num_blocks = k_pool
        .shape()
        .as_slice()
        .first()
        .copied()
        .unwrap_or_default() as i32;
    let status = match T::DATA_TYPE {
        DataType::BF16 => unsafe {
            launch_cudnn_paged_decode_bf16(
                device.config.cudnn_handle,
                q.data_ptr() as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                output.data_ptr_mut() as _,
                o_stride_seq,
                o_stride_head,
                plan.block_tables.data_ptr() as *const u32,
                plan.max_blocks_per_seq as i32,
                plan.block_size as i32,
                plan.seq_lens_step.data_ptr(),
                plan.kv_lens.data_ptr(),
                num_blocks,
                device.config.workspace,
                device.config.workspace_size,
                batch,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                stream,
            )
        },
        DataType::F16 => unsafe {
            launch_cudnn_paged_decode_fp16(
                device.config.cudnn_handle,
                q.data_ptr() as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                output.data_ptr_mut() as _,
                o_stride_seq,
                o_stride_head,
                plan.block_tables.data_ptr() as *const u32,
                plan.max_blocks_per_seq as i32,
                plan.block_size as i32,
                plan.seq_lens_step.data_ptr(),
                plan.kv_lens.data_ptr(),
                num_blocks,
                device.config.workspace,
                device.config.workspace_size,
                batch,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                stream,
            )
        },
        _ => return None,
    };
    Some(status)
}

pub fn attention_paged<T: Dtype>(
    stream: cudaStream_t,
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: PagedAttentionPlan<'_>,
    workspace: &mut Tensor<f32, Cuda>,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<()> {
    let device = q.device();
    let batch = plan.batch as i32;

    // Q / O layout: [num_tokens, num_q_heads * head_dim]; q may be a
    // strided view (zero-copy slice of a fused QKV buffer), so read its
    // row stride directly. O is always contiguous (workspace-owned).
    //   stride seq  = q's row stride (== num_q_heads * head_dim if contig)
    //   stride head = head_dim
    let q_stride_seq = q.strides().as_slice()[0] as i64;
    let q_stride_head = head_dim as i64;
    let o_stride_seq = (head_num * head_dim) as i64;
    let o_stride_head = head_dim as i64;

    // Pool layout: [num_blocks, block_size, kv_dim]; the kernels handle
    // their own indexing through block_tables + block_size, so we only pass
    // the base pointers.

    match plan.kind {
        PagedAttentionKind::DecodeOnly => {
            // Decode attention uses cuDNN SDPA by default (matches commit
            // 96f7b4e: correct + ~3.3x faster than the custom flash-decode
            // kernel). cuDNN IS CUDA-graph-capturable here — warmup builds &
            // caches the SDPA plan and capture reuses the cached plan (see
            // cudnn_paged_attention.cu), so warmup and the captured graph run the
            // SAME cuDNN kernel. Opt out (custom kernel) via env.
            let use_cudnn = std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_none();
            if use_cudnn {
            if let Some(cudnn_status) = unsafe {
                try_cudnn_paged_decode(
                    device,
                    q,
                    k_pool,
                    v_pool,
                    output,
                    plan,
                    q_stride_seq,
                    q_stride_head,
                    o_stride_seq,
                    o_stride_head,
                    head_num,
                    kv_head_num,
                    head_dim,
                    scale,
                    batch,
                    stream,
                )
            } {
                if cudnn_status == 0 {
                    return Ok(());
                }
                if std::env::var_os(STRICT_CUDNN_ATTENTION_ENV).is_some() {
                    return Err(OpError::Kernel(format!(
                        "cuDNN paged decode attention failed with status {}",
                        cudnn_status
                    )));
                }
            }
            } // end if use_cudnn

            // The caller pre-allocated `workspace` with at least
            // `flash_attn_batched_decode_workspace_bytes(cap_batch, head_num, head_dim)`
            // bytes. We sanity-check at debug time, but trust the caller in release.
            #[cfg(debug_assertions)]
            {
                let need = unsafe {
                    flash_attn_batched_decode_workspace_bytes(
                        batch,
                        head_num as i32,
                        head_dim as i32,
                    )
                } as usize;
                let have = workspace.numel() * std::mem::size_of::<f32>();
                debug_assert!(
                    have >= need,
                    "attention_paged workspace too small: have {} bytes, need {}",
                    have,
                    need
                );
            }

            unsafe {
                match T::DATA_TYPE {
                    DataType::BF16 => launch_flash_attn_paged_decode_bf16(
                        q.data_ptr() as _,
                        q_stride_seq,
                        q_stride_head,
                        k_pool.data_ptr() as _,
                        v_pool.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        o_stride_seq,
                        o_stride_head,
                        plan.block_tables.data_ptr() as *const u32,
                        plan.max_blocks_per_seq as i32,
                        plan.block_size as i32,
                        plan.kv_lens.data_ptr(),
                        workspace.data_ptr_mut(),
                        batch,
                        head_num as i32,
                        kv_head_num as i32,
                        head_dim as i32,
                        scale,
                        stream,
                    ),
                    DataType::F16 => launch_flash_attn_paged_decode_fp16(
                        q.data_ptr() as _,
                        q_stride_seq,
                        q_stride_head,
                        k_pool.data_ptr() as _,
                        v_pool.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        o_stride_seq,
                        o_stride_head,
                        plan.block_tables.data_ptr() as *const u32,
                        plan.max_blocks_per_seq as i32,
                        plan.block_size as i32,
                        plan.kv_lens.data_ptr(),
                        workspace.data_ptr_mut(),
                        batch,
                        head_num as i32,
                        kv_head_num as i32,
                        head_dim as i32,
                        scale,
                        stream,
                    ),
                    _ => {
                        return Err(OpError::Kernel(format!(
                            "attention_paged DecodeOnly: dtype {:?}",
                            T::DATA_TYPE
                        )));
                    }
                }
            }
        }
        PagedAttentionKind::Ragged => {
            // Fused mixed batch is laid out decode-first, so the leading run of
            // q==1 rows is the decode prefix. Rescue it onto cuDNN's fast decode
            // SDPA and run the q>1 prefill suffix on the CuTe ragged kernel
            // (efficient at q>1; its bottom-right causal mask works on any cuDNN
            // backend, unlike paged-SDPA causal which needs backend >= 9.21).
            //
            // The split is pure pointer arithmetic: each decode row is one tile,
            // so the first `decode_count` tiles are the decode rows. Shifting
            // block2req/block2tile past them and reducing total_q_tiles selects
            // exactly the prefill tiles, and the CuTe kernel's absolute
            // block2req[tile] / cu_q_lens[req] indexing keeps every lookup
            // correct against the full base pointers. Decode output rows are
            // never touched by the suffix launch.
            let cudnn_enabled = std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_none();
            let decode_count = plan.q_lens.iter().take_while(|&&q| q == 1).count();

            if cudnn_enabled && decode_count > 0 && decode_count < plan.batch {
                let dec = unsafe {
                    try_cudnn_paged_decode(
                        device,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        plan,
                        q_stride_seq,
                        q_stride_head,
                        o_stride_seq,
                        o_stride_head,
                        head_num,
                        kv_head_num,
                        head_dim,
                        scale,
                        decode_count as i32,
                        stream,
                    )
                };
                if dec == Some(0) {
                    // decode prefix done on cuDNN → prefill suffix on CuTe.
                    let suffix_tiles = plan.total_q_tiles - decode_count as i32;
                    if suffix_tiles > 0 {
                        unsafe {
                            launch_cute_ragged(
                                q.data_ptr(),
                                q_stride_seq,
                                q_stride_head,
                                k_pool,
                                v_pool,
                                output.data_ptr_mut(),
                                o_stride_seq,
                                o_stride_head,
                                plan.block_tables.data_ptr() as *const u32,
                                plan.max_blocks_per_seq as i32,
                                plan.block_size as i32,
                                plan.kv_lens.data_ptr(),
                                plan.cu_q_lens.data_ptr(),
                                plan.block2req.data_ptr().add(decode_count),
                                plan.block2tile.data_ptr().add(decode_count),
                                suffix_tiles,
                                batch,
                                plan.num_tokens as i32,
                                head_num,
                                kv_head_num,
                                head_dim,
                                scale,
                                stream,
                            )?;
                        }
                    }
                    return Ok(());
                }
                if dec.is_some() && std::env::var_os(STRICT_CUDNN_ATTENTION_ENV).is_some() {
                    return Err(OpError::Kernel(
                        "cuDNN fused decode-prefix attention failed".into(),
                    ));
                }
                // else: decode rescue unavailable → full-batch CuTe ragged below.
            }

            // Full-batch CuTe ragged: no decode prefix, cuDNN disabled, or the
            // decode rescue failed (non-strict). Correct for any composition.
            unsafe {
                launch_cute_ragged(
                    q.data_ptr(),
                    q_stride_seq,
                    q_stride_head,
                    k_pool,
                    v_pool,
                    output.data_ptr_mut(),
                    o_stride_seq,
                    o_stride_head,
                    plan.block_tables.data_ptr() as *const u32,
                    plan.max_blocks_per_seq as i32,
                    plan.block_size as i32,
                    plan.kv_lens.data_ptr(),
                    plan.cu_q_lens.data_ptr(),
                    plan.block2req.data_ptr(),
                    plan.block2tile.data_ptr(),
                    plan.total_q_tiles,
                    batch,
                    plan.num_tokens as i32,
                    head_num,
                    kv_head_num,
                    head_dim,
                    scale,
                    stream,
                )?;
            }
        }
    }
    Ok(())
}
