use crate::base::error::Result;
use crate::tensor::Tensor;
use ndarray::{ArrayView1, ArrayViewMut1};
use ndarray_linalg::Norm;
use rayon::prelude::*;
const RMS_NORM_EPSILON: f32 = 1e-6;

/// RMSNorm 的 CPU 内核实现，使用 ndarray 库进行高性能计算。
pub fn rmsnorm(input: &Tensor, weight: &Tensor, output: &mut Tensor) -> Result<()> {
    // --- 1. 获取 Rust slice ---
    // 类型和设备检查已在上层完成，这里可以直接 unwrap
    let input_typed = input.as_f32().unwrap();
    let weight_typed = weight.as_f32().unwrap();
    let output_typed = output.as_f32_mut().unwrap();

    let input_slice = input_typed.as_slice().unwrap();
    let weight_slice = weight_typed.as_slice().unwrap();
    let output_slice = output_typed.as_slice_mut().unwrap();
    
    let dim = weight_slice.len();
    let dim_sqrt = (dim as f32).sqrt();
    // --- 2. 将 slice 转换为 ndarray 视图 (零拷贝) ---
    let weight_view = ArrayView1::from(weight_slice);
    
    output_slice
        .par_chunks_mut(dim)
        .zip(input_slice.par_chunks(dim))
        .for_each(|(output_chunk, input_chunk)| {
            let input_row_view = ArrayView1::from(input_chunk);
            let mut output_row_view = ArrayViewMut1::from(output_chunk);
            let norm = input_row_view.norm_l2();
            let rms = norm / dim_sqrt;
            let rsqrt = (rms * rms + RMS_NORM_EPSILON).sqrt().recip();
            
            output_row_view.assign(&(&weight_view * (&input_row_view * rsqrt)));
        });
    
    Ok(())
}