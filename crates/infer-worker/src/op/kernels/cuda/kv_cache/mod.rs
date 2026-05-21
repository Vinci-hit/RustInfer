//! KV cache CUDA kernels.

use std::ffi::c_void;

use crate::OpConfig;
use crate::base::DataType;
use crate::base::error::{Error, Result};
use crate::cuda;
use crate::cuda::CudaConfig;
use crate::tensor::Tensor;

unsafe extern "C" {
    fn scatter_kv_batch_bf16(
        k_src: *const c_void,
        v_src: *const c_void,
        k_cache_ptrs: *const *mut c_void,
        v_cache_ptrs: *const *mut c_void,
        layer_idx: i32,
        max_slots: i32,
        slot_indices: *const i32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        batch: i32,
        kv_dim: i32,
        k_src_row_stride_elems: i32,
        v_src_row_stride_elems: i32,
        dst_row_stride_elems: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_batch_fp16(
        k_src: *const c_void,
        v_src: *const c_void,
        k_cache_ptrs: *const *mut c_void,
        v_cache_ptrs: *const *mut c_void,
        layer_idx: i32,
        max_slots: i32,
        slot_indices: *const i32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        batch: i32,
        kv_dim: i32,
        k_src_row_stride_elems: i32,
        v_src_row_stride_elems: i32,
        dst_row_stride_elems: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_batch_f32(
        k_src: *const c_void,
        v_src: *const c_void,
        k_cache_ptrs: *const *mut c_void,
        v_cache_ptrs: *const *mut c_void,
        layer_idx: i32,
        max_slots: i32,
        slot_indices: *const i32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        batch: i32,
        kv_dim: i32,
        k_src_row_stride_elems: i32,
        v_src_row_stride_elems: i32,
        dst_row_stride_elems: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_paged_bf16(
        k_src: *const c_void,
        v_src: *const c_void,
        k_pool: *mut c_void,
        v_pool: *mut c_void,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        batch: i32,
        kv_dim: i32,
        k_src_row_stride_elems: i32,
        v_src_row_stride_elems: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_paged_fp16(
        k_src: *const c_void,
        v_src: *const c_void,
        k_pool: *mut c_void,
        v_pool: *mut c_void,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        batch: i32,
        kv_dim: i32,
        k_src_row_stride_elems: i32,
        v_src_row_stride_elems: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_paged_f32(
        k_src: *const c_void,
        v_src: *const c_void,
        k_pool: *mut c_void,
        v_pool: *mut c_void,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        batch: i32,
        kv_dim: i32,
        k_src_row_stride_elems: i32,
        v_src_row_stride_elems: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Batched K/V scatter —— 一次 launch。
///
/// 所有指针都在 device 上：
/// - `*_cache_ptrs`：`[layer_num, max_slots]` u64 base 表，runner 管填；
/// - `slot_indices / seq_positions / seq_starts / seq_lens`：`[B]` i32 小数组，
///   runner 在 step 入口一次性填，所有层共用。
///
/// kernel 自己 `cache_ptrs[layer_idx * max_slots + slot] + pos * dst_row_stride`
/// 算起点。op 内 **零 malloc / 零 sync / 零 per-step H2D**。
///
/// # Safety
/// 所有裸指针必须在本次 launch 期间有效且地址稳定。
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_scatter_batched(
    k_src: &Tensor,
    v_src: &Tensor,
    k_cache_ptrs_dev: *const *mut c_void,
    v_cache_ptrs_dev: *const *mut c_void,
    layer_idx: usize,
    max_slots: usize,
    slot_indices_dev: *const i32,
    seq_positions_dev: *const i32,
    seq_starts_dev: *const i32,
    seq_lens_dev: *const i32,
    batch: usize,
    kv_dim: usize,
    k_src_row_stride: usize,
    v_src_row_stride: usize,
    dst_row_stride: usize,
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    if batch == 0 {
        return Ok(());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let layer_i = layer_idx as i32;
    let slots_i = max_slots as i32;
    let batch_i = batch as i32;
    let kv_dim_i = kv_dim as i32;
    let ksr = k_src_row_stride as i32;
    let vsr = v_src_row_stride as i32;
    let dsr = dst_row_stride as i32;

    match k_src.dtype() {
        DataType::BF16 => {
            let k_ptr = k_src.as_bf16()?.data_ptr() as *const c_void;
            let v_ptr = v_src.as_bf16()?.data_ptr() as *const c_void;
            unsafe {
                scatter_kv_batch_bf16(
                    k_ptr, v_ptr,
                    k_cache_ptrs_dev, v_cache_ptrs_dev,
                    layer_i, slots_i,
                    slot_indices_dev, seq_positions_dev,
                    seq_starts_dev, seq_lens_dev,
                    batch_i, kv_dim_i,
                    ksr, vsr, dsr,
                    stream,
                );
            }
        }
        DataType::F16 => {
            let k_ptr = k_src.as_f16()?.data_ptr() as *const c_void;
            let v_ptr = v_src.as_f16()?.data_ptr() as *const c_void;
            unsafe {
                scatter_kv_batch_fp16(
                    k_ptr, v_ptr,
                    k_cache_ptrs_dev, v_cache_ptrs_dev,
                    layer_i, slots_i,
                    slot_indices_dev, seq_positions_dev,
                    seq_starts_dev, seq_lens_dev,
                    batch_i, kv_dim_i,
                    ksr, vsr, dsr,
                    stream,
                );
            }
        }
        DataType::F32 => {
            let k_ptr = k_src.as_f32()?.data_ptr() as *const c_void;
            let v_ptr = v_src.as_f32()?.data_ptr() as *const c_void;
            unsafe {
                scatter_kv_batch_f32(
                    k_ptr, v_ptr,
                    k_cache_ptrs_dev, v_cache_ptrs_dev,
                    layer_i, slots_i,
                    slot_indices_dev, seq_positions_dev,
                    seq_starts_dev, seq_lens_dev,
                    batch_i, kv_dim_i,
                    ksr, vsr, dsr,
                    stream,
                );
            }
        }
        dt => {
            return Err(Error::InvalidArgument(format!(
                "scatter_kv_batch: unsupported dtype {:?}", dt
            )).into());
        }
    }
    Ok(())
}

/// Paged K/V scatter into global KV pool [num_blocks, block_size, kv_dim].
///
/// # Safety
/// All raw device pointers must be valid for this launch. `block_tables_dev`
/// points to `[batch, max_blocks_per_seq]` physical block ids.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_scatter_paged(
    k_src: &Tensor,
    v_src: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    block_tables_dev: *const u32,
    max_blocks_per_seq: usize,
    block_size: usize,
    seq_positions_dev: *const i32,
    seq_starts_dev: *const i32,
    seq_lens_dev: *const i32,
    batch: usize,
    kv_dim: usize,
    k_src_row_stride: usize,
    v_src_row_stride: usize,
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    if batch == 0 {
        return Ok(());
    }
    if k_pool.dtype() != k_src.dtype() || v_pool.dtype() != v_src.dtype() {
        return Err(Error::InvalidArgument(format!(
            "scatter_kv_paged dtype mismatch: k_src={:?} k_pool={:?} v_pool={:?}",
            k_src.dtype(), k_pool.dtype(), v_pool.dtype(),
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let max_blocks_i = max_blocks_per_seq as i32;
    let block_size_i = block_size as i32;
    let batch_i = batch as i32;
    let kv_dim_i = kv_dim as i32;
    let ksr = k_src_row_stride as i32;
    let vsr = v_src_row_stride as i32;

    match k_src.dtype() {
        DataType::BF16 => {
            let k_ptr = k_src.as_bf16()?.data_ptr() as *const c_void;
            let v_ptr = v_src.as_bf16()?.data_ptr() as *const c_void;
            let k_pool_ptr = k_pool.as_bf16()?.data_ptr() as *mut c_void;
            let v_pool_ptr = v_pool.as_bf16()?.data_ptr() as *mut c_void;
            unsafe {
                scatter_kv_paged_bf16(
                    k_ptr, v_ptr, k_pool_ptr, v_pool_ptr,
                    block_tables_dev, max_blocks_i, block_size_i,
                    seq_positions_dev, seq_starts_dev, seq_lens_dev,
                    batch_i, kv_dim_i, ksr, vsr, stream,
                );
            }
        }
        DataType::F16 => {
            let k_ptr = k_src.as_f16()?.data_ptr() as *const c_void;
            let v_ptr = v_src.as_f16()?.data_ptr() as *const c_void;
            let k_pool_ptr = k_pool.as_f16()?.data_ptr() as *mut c_void;
            let v_pool_ptr = v_pool.as_f16()?.data_ptr() as *mut c_void;
            unsafe {
                scatter_kv_paged_fp16(
                    k_ptr, v_ptr, k_pool_ptr, v_pool_ptr,
                    block_tables_dev, max_blocks_i, block_size_i,
                    seq_positions_dev, seq_starts_dev, seq_lens_dev,
                    batch_i, kv_dim_i, ksr, vsr, stream,
                );
            }
        }
        DataType::F32 => {
            let k_ptr = k_src.as_f32()?.data_ptr() as *const c_void;
            let v_ptr = v_src.as_f32()?.data_ptr() as *const c_void;
            let k_pool_ptr = k_pool.as_f32()?.data_ptr() as *mut c_void;
            let v_pool_ptr = v_pool.as_f32()?.data_ptr() as *mut c_void;
            unsafe {
                scatter_kv_paged_f32(
                    k_ptr, v_ptr, k_pool_ptr, v_pool_ptr,
                    block_tables_dev, max_blocks_i, block_size_i,
                    seq_positions_dev, seq_starts_dev, seq_lens_dev,
                    batch_i, kv_dim_i, ksr, vsr, stream,
                );
            }
        }
        dt => {
            return Err(Error::InvalidArgument(format!(
                "scatter_kv_paged: unsupported dtype {:?}", dt
            )).into());
        }
    }
    Ok(())
}
