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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::Shape;
    use half::bf16;

    #[test]
    fn split_cols_f32_extracts_middle_block() {
        let cuda = Cuda::new(0).expect("cuda init");
        let rows = 3usize;
        let total_cols = 6usize;
        // src laid out as rows × total_cols, values = row * 100 + col.
        let src_host: Vec<f32> = (0..rows).flat_map(|r| (0..total_cols).map(move |c| (r * 100 + c) as f32)).collect();
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &src_host, Shape::from_slice(&[rows, total_cols]), &cuda,
        ).unwrap();
        // Slice cols [2..5).
        let dst_cols = 3usize;
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([rows, dst_cols], &cuda).unwrap();
        split_cols(&src, &mut dst, rows as i32, total_cols as i32, 2, dst_cols as i32).unwrap();
        let got = dst.to_host_vec().unwrap();
        let expected: Vec<f32> = (0..rows).flat_map(|r| (2..5).map(move |c| (r * 100 + c) as f32)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn split_cols_bf16_qkv_split() {
        let cuda = Cuda::new(0).expect("cuda init");
        let rows = 4usize;
        let dim = 8usize;
        let total_cols = 3 * dim; // q | k | v
        let src_host: Vec<bf16> = (0..rows * total_cols).map(|i| bf16::from_f32(i as f32)).collect();
        let src: Tensor<bf16, Cuda> = Tensor::from_host_slice(
            &src_host, Shape::from_slice(&[rows, total_cols]), &cuda,
        ).unwrap();
        let mut q: Tensor<bf16, Cuda> = Tensor::zeros([rows, dim], &cuda).unwrap();
        let mut k: Tensor<bf16, Cuda> = Tensor::zeros([rows, dim], &cuda).unwrap();
        let mut v: Tensor<bf16, Cuda> = Tensor::zeros([rows, dim], &cuda).unwrap();
        split_cols(&src, &mut q, rows as i32, total_cols as i32, 0, dim as i32).unwrap();
        split_cols(&src, &mut k, rows as i32, total_cols as i32, dim as i32, dim as i32).unwrap();
        split_cols(&src, &mut v, rows as i32, total_cols as i32, 2 * dim as i32, dim as i32).unwrap();

        let q_host: Vec<f32> = q.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        let k_host: Vec<f32> = k.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        let v_host: Vec<f32> = v.to_host_vec().unwrap().iter().map(|v| v.to_f32()).collect();

        for r in 0..rows {
            for c in 0..dim {
                let row_base = r * total_cols;
                assert_eq!(q_host[r * dim + c], src_host[row_base + c].to_f32());
                assert_eq!(k_host[r * dim + c], src_host[row_base + dim + c].to_f32());
                assert_eq!(v_host[r * dim + c], src_host[row_base + 2 * dim + c].to_f32());
            }
        }
    }
}
