use crate::base::error::Result;
use crate::cuda;
use crate::tensor::Tensor;
use std::ffi::c_void;

unsafe extern "C" {
    fn split_cols_bf16(
        src: *const c_void,
        dst: *mut c_void,
        rows: i32,
        total_cols: i32,
        col_offset: i32,
        dst_cols: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn split_cols_fp16(
        src: *const c_void,
        dst: *mut c_void,
        rows: i32,
        total_cols: i32,
        col_offset: i32,
        dst_cols: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Split columns from a fused [rows, total_cols] BF16 tensor.
/// Copies columns [col_offset, col_offset + dst_cols) into dst [rows, dst_cols].
pub fn split_cols_bf16_tensor(
    src: &Tensor,
    dst: &mut Tensor,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
    stream: cuda::ffi::cudaStream_t,
) -> Result<()> {
    let src_ptr = src.as_bf16()?.buffer().as_ptr() as *const c_void;
    let dst_ptr = dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut c_void;
    unsafe {
        split_cols_bf16(
            src_ptr,
            dst_ptr,
            rows as i32,
            total_cols as i32,
            col_offset as i32,
            dst_cols as i32,
            stream,
        );
    }
    Ok(())
}

pub fn split_cols_fp16_tensor(
    src: &Tensor,
    dst: &mut Tensor,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
    stream: cuda::ffi::cudaStream_t,
) -> Result<()> {
    let src_ptr = src.as_f16()?.buffer().as_ptr() as *const c_void;
    let dst_ptr = dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut c_void;
    unsafe {
        split_cols_fp16(
            src_ptr,
            dst_ptr,
            rows as i32,
            total_cols as i32,
            col_offset as i32,
            dst_cols as i32,
            stream,
        );
    }
    Ok(())
}
