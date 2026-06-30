//! Scatter K/V into a layer's paged KV pool — one CUDA launch.
//!
//! Wraps `scatter_kv_paged_{bf16,fp16,f32}`. The kernel addresses the pool
//! through `block_tables[seq][token / block_size]` and writes
//! `pool[block_id * block_size + (token % block_size)]`.

use std::ffi::c_void;

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
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
        stream: cudaStream_t,
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
        stream: cudaStream_t,
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
        stream: cudaStream_t,
    );
}

pub fn scatter_kv_paged<T: Dtype>(
    stream: cudaStream_t,
    k_src: &Tensor<T, Cuda>,
    v_src: &Tensor<T, Cuda>,
    k_pool: &mut Tensor<T, Cuda>,
    v_pool: &mut Tensor<T, Cuda>,
    block_tables: &Tensor<i32, Cuda>,
    seq_positions: &Tensor<i32, Cuda>,
    cu_q_lens: &Tensor<i32, Cuda>,
    seq_lens_step: &Tensor<i32, Cuda>,
    max_blocks_per_seq: usize,
    block_size: usize,
    kv_dim: usize,
) -> OpResult<()> {
    let batch = seq_positions.shape().as_slice()[0];
    if batch == 0 {
        return Ok(());
    }

    // Read row stride from the tensors directly so strided views (e.g.
    // zero-copy slices of a fused QKV buffer) are handled correctly.
    let k_stride = k_src.strides().as_slice()[0] as i32;
    let v_stride = v_src.strides().as_slice()[0] as i32;

    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => scatter_kv_paged_bf16(
                k_src.data_ptr() as *const c_void,
                v_src.data_ptr() as *const c_void,
                k_pool.data_ptr_mut() as *mut c_void,
                v_pool.data_ptr_mut() as *mut c_void,
                block_tables.data_ptr() as *const u32,
                max_blocks_per_seq as i32,
                block_size as i32,
                seq_positions.data_ptr(),
                cu_q_lens.data_ptr(),
                seq_lens_step.data_ptr(),
                batch as i32,
                kv_dim as i32,
                k_stride,
                v_stride,
                stream,
            ),
            DataType::F16 => scatter_kv_paged_fp16(
                k_src.data_ptr() as *const c_void,
                v_src.data_ptr() as *const c_void,
                k_pool.data_ptr_mut() as *mut c_void,
                v_pool.data_ptr_mut() as *mut c_void,
                block_tables.data_ptr() as *const u32,
                max_blocks_per_seq as i32,
                block_size as i32,
                seq_positions.data_ptr(),
                cu_q_lens.data_ptr(),
                seq_lens_step.data_ptr(),
                batch as i32,
                kv_dim as i32,
                k_stride,
                v_stride,
                stream,
            ),
            DataType::F32 => scatter_kv_paged_f32(
                k_src.data_ptr() as *const c_void,
                v_src.data_ptr() as *const c_void,
                k_pool.data_ptr_mut() as *mut c_void,
                v_pool.data_ptr_mut() as *mut c_void,
                block_tables.data_ptr() as *const u32,
                max_blocks_per_seq as i32,
                block_size as i32,
                seq_positions.data_ptr(),
                cu_q_lens.data_ptr(),
                seq_lens_step.data_ptr(),
                batch as i32,
                kv_dim as i32,
                k_stride,
                v_stride,
                stream,
            ),
            _ => {
                return Err(OpError::Kernel(format!(
                    "scatter_kv_paged: dtype {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}
