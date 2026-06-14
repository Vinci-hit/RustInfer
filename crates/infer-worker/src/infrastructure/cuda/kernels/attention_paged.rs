//! Paged ragged / paged decode attention wrapper.
//!
//! - `BatchKind::DecodeOnly` → `launch_flash_attn_paged_decode_*`
//! - `BatchKind::Ragged`     → `launch_flash_attn_paged_ragged_cute_*`
//!
//! Both kernels read K/V from a single `[num_blocks, block_size, kv_dim]`
//! pool per layer; per-seq routing comes from the device `block_tables`
//! tensor (`[batch, max_blocks_per_seq]` u32).

use crate::domain::batch::{BatchKind, BatchPlan};
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

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

pub fn attention_paged<T: Dtype>(
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: &BatchPlan<Cuda>,
    workspace: &mut Tensor<f32, Cuda>,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<()> {
    let device = q.device();
    let stream = device.config.stream;
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
        BatchKind::DecodeOnly => {
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
        BatchKind::Ragged => unsafe {
            match T::DATA_TYPE {
                DataType::BF16 => launch_flash_attn_paged_ragged_cute_bf16(
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
                    plan.cu_q_lens.data_ptr(),
                    plan.block2req.data_ptr(),
                    plan.block2tile.data_ptr(),
                    plan.total_q_tiles,
                    batch,
                    plan.num_tokens as i32,
                    head_num as i32,
                    kv_head_num as i32,
                    head_dim as i32,
                    scale,
                    1,
                    stream,
                ),
                DataType::F16 => launch_flash_attn_paged_ragged_cute_fp16(
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
                    plan.cu_q_lens.data_ptr(),
                    plan.block2req.data_ptr(),
                    plan.block2tile.data_ptr(),
                    plan.total_q_tiles,
                    batch,
                    plan.num_tokens as i32,
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
        },
    }
    Ok(())
}
