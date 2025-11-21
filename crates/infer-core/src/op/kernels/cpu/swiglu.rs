use crate::base::error::Result;
use crate::tensor::Tensor;
// 引入 rayon 进行并行计算
use rayon::prelude::*;

/// Sigmoid 函数的数值稳定实现
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// (原地版本) SwiGLU CPU 内核。
///
/// 计算 `x = (x * SiLU(x)) * y`，并将结果写回 `x`。
pub fn swiglu(
    input_y: &Tensor,       // 只读的 y 张量
    input_output_x: &mut Tensor, // 可读写的 x 张量
) -> Result<()> {
    // --- 1. 获取数据 slice ---
    // x 需要可变切片，y 只需要不可变切片
    let x_slice = input_output_x.as_f32_mut()?.as_slice_mut()?;
    let y_slice = input_y.as_f32()?.as_slice()?;
    
    // --- 2. 执行并行的原地计算 ---
    // 我们并行地遍历 x 切片 (可变的)
    x_slice
        .par_iter_mut()
        .zip(y_slice.par_iter()) // 并行地 zip 只读的 y
        .for_each(|(x_val, &y_val)| {
            // `x_val` 是一个 `&mut f32`，指向 `x_slice` 中的一个元素
            
            // a. 读取 x 的原始值
            let x_orig = *x_val;
            
            // b. 计算 SiLU(x)
            let silu_x = x_orig * sigmoid(x_orig);
            
            // c. 计算最终结果
            let final_val = silu_x * y_val;
            
            // d. 将结果写回 x 的原始位置，覆盖掉旧值
            *x_val = final_val;
        });

    Ok(())
}