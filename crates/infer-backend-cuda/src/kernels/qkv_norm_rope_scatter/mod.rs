//! Fused Qwen3 Q/K RMSNorm + RoPE + paged K/V scatter CUDA wrapper.

use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};
use crate::Cuda;
use crate::ffi::cudaStream_t;

unsafe extern "C" {
    fn qkv_norm_rope_scatter_bf16(
        q: *mut half::bf16,
        k: *mut half::bf16,
        v: *const half::bf16,
        q_weight: *const half::bf16,
        k_weight: *const half::bf16,
        sin_cache: *const half::bf16,
        cos_cache: *const half::bf16,
        positions: *const i32,
        k_pool: *mut half::bf16,
        v_pool: *mut half::bf16,
        block_tables: *const u32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        num_tokens: i32,
        batch: i32,
        head_num: i32,
        kv_head_num: i32,
        head_dim: i32,
        kv_dim: i32,
        q_row_stride: i64,
        k_row_stride: i64,
        v_row_stride: i64,
        max_blocks_per_seq: i32,
        block_size: i32,
        q_eps: f32,
        k_eps: f32,
        stream: cudaStream_t,
    );
}

pub fn qkv_norm_rope_scatter<T: Dtype>(
    stream: cudaStream_t,
    q: &mut Tensor<T, Cuda>,
    k: &mut Tensor<T, Cuda>,
    v: &Tensor<T, Cuda>,
    q_weight: Option<&Tensor<T, Cuda>>,
    k_weight: Option<&Tensor<T, Cuda>>,
    q_eps: f32,
    k_eps: f32,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions: &Tensor<i32, Cuda>,
    k_pool: &mut Tensor<T, Cuda>,
    v_pool: &mut Tensor<T, Cuda>,
    block_tables: &Tensor<i32, Cuda>,
    seq_positions: &Tensor<i32, Cuda>,
    cu_q_lens: &Tensor<i32, Cuda>,
    seq_lens_step: &Tensor<i32, Cuda>,
    max_blocks_per_seq: usize,
    block_size: usize,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    kv_dim: usize,
) -> OpResult<()> {
    let Some(q_weight) = q_weight else {
        return Err(OpError::Kernel(
            "qkv_norm_rope_scatter: missing q_norm weight".into(),
        ));
    };
    let Some(k_weight) = k_weight else {
        return Err(OpError::Kernel(
            "qkv_norm_rope_scatter: missing k_norm weight".into(),
        ));
    };
    let batch = seq_positions.shape().as_slice()[0];
    if batch == 0 || positions.numel() == 0 {
        return Ok(());
    }
    if T::DATA_TYPE != DataType::BF16 {
        return Err(OpError::Kernel(format!(
            "qkv_norm_rope_scatter: unsupported dtype {:?}",
            T::DATA_TYPE
        )));
    }

    let q_row_stride = q.strides().as_slice()[0] as i64;
    let k_row_stride = k.strides().as_slice()[0] as i64;
    let v_row_stride = v.strides().as_slice()[0] as i64;

    unsafe {
        qkv_norm_rope_scatter_bf16(
            q.data_ptr_mut() as _,
            k.data_ptr_mut() as _,
            v.data_ptr() as _,
            q_weight.data_ptr() as _,
            k_weight.data_ptr() as _,
            sin.data_ptr() as _,
            cos.data_ptr() as _,
            positions.data_ptr(),
            k_pool.data_ptr_mut() as _,
            v_pool.data_ptr_mut() as _,
            block_tables.data_ptr() as *const u32,
            seq_positions.data_ptr(),
            cu_q_lens.data_ptr(),
            seq_lens_step.data_ptr(),
            positions.numel() as i32,
            batch as i32,
            head_num as i32,
            kv_head_num as i32,
            head_dim as i32,
            kv_dim as i32,
            q_row_stride,
            k_row_stride,
            v_row_stride,
            max_blocks_per_seq as i32,
            block_size as i32,
            q_eps,
            k_eps,
            stream,
        );
    }
    Ok(())
}
