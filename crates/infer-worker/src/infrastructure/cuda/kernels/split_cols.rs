//! Split columns CUDA kernel — extracts a sub-range of columns from a matrix.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn split_cols_bf16(src: *const half::bf16, dst: *mut half::bf16, rows: i32, total_cols: i32, col_offset: i32, dst_cols: i32, stream: cudaStream_t);
    fn split_cols_fp16(src: *const half::f16, dst: *mut half::f16, rows: i32, total_cols: i32, col_offset: i32, dst_cols: i32, stream: cudaStream_t);
    fn split_cols_f32(src: *const f32, dst: *mut f32, rows: i32, total_cols: i32, col_offset: i32, dst_cols: i32, stream: cudaStream_t);
}

/// Split columns [col_offset..col_offset+dst_cols) from src [rows, total_cols] into dst [rows, dst_cols].
pub fn split_cols<T: Dtype>(
    src: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
    rows: i32,
    total_cols: i32,
    col_offset: i32,
    dst_cols: i32,
) -> OpResult<()> {
    let stream = src.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => split_cols_bf16(src.data_ptr() as _, dst.data_ptr_mut() as _, rows, total_cols, col_offset, dst_cols, stream),
            DataType::F16 => split_cols_fp16(src.data_ptr() as _, dst.data_ptr_mut() as _, rows, total_cols, col_offset, dst_cols, stream),
            DataType::F32 => split_cols_f32(src.data_ptr() as _, dst.data_ptr_mut() as _, rows, total_cols, col_offset, dst_cols, stream),
            _ => return Err(OpError::Kernel(format!("split_cols: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
