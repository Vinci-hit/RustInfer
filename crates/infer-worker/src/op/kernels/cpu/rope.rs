use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;
use ndarray::{ArrayViewMut2, Axis};
use rayon::prelude::*; // 导入 rayon 的并行迭代器 trait

pub fn sin_cos_cache_calc_bf16(
    head_size: usize,
    max_seq_len: usize,
    rope_base: f32,
    sin_cache: &mut Tensor,
    cos_cache: &mut Tensor,
) -> Result<()> {
    // --- 1. 形状检查和 ndarray 视图创建 (针对 BF16) ---
    let shape = (max_seq_len, head_size);
    let expected_len = max_seq_len * head_size;

    let sin_typed = sin_cache.as_bf16_mut()?;
    let cos_typed = cos_cache.as_bf16_mut()?;
    
    // b. 获取底层的可变 bf16 切片
    let sin_slice = sin_typed.as_slice_mut()?;
    let cos_slice = cos_typed.as_slice_mut()?;

    if sin_slice.len() != expected_len || cos_slice.len() != expected_len {
        return Err(Error::InvalidArgument(format!(
            "Cache size mismatch. Expected {}, got sin {} and cos {}", 
            expected_len, sin_slice.len(), cos_slice.len()
        )).into());
    }

    // c. 将 bf16 切片包装成 ndarray 的可变视图
    let mut sin_view: ArrayViewMut2<bf16> = ArrayViewMut2::from_shape(shape, sin_slice)
        .map_err(|e| Error::InvalidArgument(format!("sin_cache view creation failed: {}", e)))?;
    let mut cos_view: ArrayViewMut2<bf16> = ArrayViewMut2::from_shape(shape, cos_slice)
        .map_err(|e| Error::InvalidArgument(format!("cos_cache view creation failed: {}", e)))?;


    // --- 2. 预先计算频率 freqs (总是使用 f32 计算) ---
    let freqs: Vec<f32> = (0..head_size).map(|head_dim| {
        let exponent = head_dim as f32 / head_size as f32;
        1.0f32 / rope_base.powf(exponent)
    }).collect();
    
    // --- 3. 核心多线程并行计算并填充 BF16 缓存 ---
    sin_view.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(cos_view.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .for_each(|(pos, (mut sin_row, mut cos_row))| {
            let pos_f = pos as f32;

            for head_dim in 0..head_size {
                let freq = freqs[head_dim];
                let val = pos_f * freq;
                
                // a. 使用 f32 进行三角函数计算
                let fcr_f32 = val.cos(); // 余弦 (f32)
                let fci_f32 = val.sin(); // 正弦 (f32)
                
                // b. 将 f32 结果转换为 bf16
                let fcr_bf16 = bf16::from_f32(fcr_f32);
                let fci_bf16 = bf16::from_f32(fci_f32);
                
                // c. 将 bf16 值写入 ndarray 视图
                sin_row[head_dim] = fci_bf16;
                cos_row[head_dim] = fcr_bf16;
            }
        });

    Ok(())
}

/// BF16版本的rope_kernel_batch实现
/// 支持位置数组（每个token一个位置）而不是单一起始位置
pub fn rope_kernel_batch_bf16(
    kv_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    pos_array: &Tensor,  // 位置数组 [seq_len]，每个token一个位置
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    if input_q.shape().len() != 2 || input_k.shape().len() != 2 {
        return Err(Error::InvalidArgument("Input Q and K for batch RoPE must be 2D.".to_string()).into());
    }

    let seq_len = input_q.shape()[0];
    let dim = input_q.shape()[1];

    // 获取BF16切片
    let q_slice_bf16 = input_q.as_bf16_mut()?.as_slice_mut()?;
    let k_slice_bf16 = input_k.as_bf16_mut()?.as_slice_mut()?;
    let sin_slice = sin_cache.as_bf16()?.as_slice()?;
    let cos_slice = cos_cache.as_bf16()?.as_slice()?;
    let pos_slice = pos_array.as_i32()?.as_slice()?;

    if pos_slice.len() != seq_len {
        return Err(Error::InvalidArgument(
            format!("Position array length ({}) must match sequence length ({})", pos_slice.len(), seq_len)
        ).into());
    }

    let head_dim = head_size; // 每个头的维度（64）

    // ========== 核心逻辑：遍历每个token（行） ==========
    for i in 0..seq_len {
        // 从位置数组获取当前token的位置
        let pos = pos_slice[i] as usize;

        // ========== 处理Q的旋转：逐token+逐成对维度 ==========
        let q_row_start = i * dim;
        let k_row_start = i * kv_dim;

        for j in (0..dim).step_by(head_dim) {
            for k in 0..head_dim/2 {
                // 从sin/cos缓存获取当前位置和头维度的值
                let sin_val = sin_slice[pos * head_dim + k*2];
                let cos_val = cos_slice[pos * head_dim + k*2];

                let q_idx_j = q_row_start + j + k;
                let q_idx_j1 = q_row_start + j + k + head_dim / 2;

                // 从BF16转换为F32进行计算
                let v0_q = q_slice_bf16[q_idx_j];
                let v1_q = q_slice_bf16[q_idx_j1];

                // F32计算
                let result0 = v0_q * cos_val - v1_q * sin_val;
                let result1 = v0_q * sin_val + v1_q * cos_val;

                // 将结果转换回BF16并存储
                q_slice_bf16[q_idx_j] = result0;
                q_slice_bf16[q_idx_j1] = result1;

                if j < kv_dim {
                    let k_idx_j = k_row_start + j + k;
                    let k_idx_j1 = k_row_start + j + k + head_dim / 2;

                    // 从BF16转换为F32进行计算
                    let v0_k = k_slice_bf16[k_idx_j];
                    let v1_k = k_slice_bf16[k_idx_j1];

                    // F32计算
                    let result0 = v0_k * cos_val - v1_k * sin_val;
                    let result1 = v0_k * sin_val + v1_k * cos_val;

                    // 将结果转换回BF16并存储
                    k_slice_bf16[k_idx_j] = result0;
                    k_slice_bf16[k_idx_j1] = result1;
                }
            }
        }
    }

    Ok(())
}

/// 计算并填充正弦和余弦旋转嵌入 (RoPE) 的缓存。
/// 
/// 该函数根据输入张量的数据类型自动分发到对应的实现，并使用 `rayon` 库进行多线程并行计算，
/// 极大地加速了缓存的生成。缓存的形状为 `[max_seq_len, head_size]`。
/// 
/// # Arguments
/// * `head_size`: 旋转维度的大小 (K)。
/// * `max_seq_len`: 序列的最大长度 (M)。
/// * `sin_cache`: 正弦值输出张量, 形状 [max_seq_len, head_size]。
/// * `cos_cache`: 余弦值输出张量, 形状 [max_seq_len, head_size]。
pub fn sin_cos_cache_calc(
    head_size: usize,
    max_seq_len: usize,
    rope_base: f32,
    sin_cache: &mut Tensor,
    cos_cache: &mut Tensor,
) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match sin_cache.dtype() {
        crate::base::DataType::F32 => {
            sin_cos_cache_calc_f32(head_size, max_seq_len, rope_base, sin_cache, cos_cache)
        }
        crate::base::DataType::BF16 => {
            sin_cos_cache_calc_bf16(head_size, max_seq_len, rope_base, sin_cache, cos_cache)
        }
        _ => {
            Err(Error::InvalidArgument(format!(
                "Unsupported data type for sin_cos_cache_calc: {:?}", sin_cache.dtype()
            )).into())
        }
    }
}

/// F32版本的sin_cos_cache_calc实现
fn sin_cos_cache_calc_f32(
    head_size: usize,
    max_seq_len: usize,
    rope_base: f32,
    sin_cache: &mut Tensor,
    cos_cache: &mut Tensor,
) -> Result<()> {
    // ... (1. 形状检查和 ndarray 视图创建 - 保持不变)
    let shape = (max_seq_len, head_size);
    let expected_len = max_seq_len * head_size;

    let sin_typed = sin_cache.as_f32_mut()?;
    let cos_typed = cos_cache.as_f32_mut()?;
    
    let sin_slice = sin_typed.as_slice_mut()
        .map_err(|_| Error::InvalidArgument("sin_cache is not contiguous".to_string()))?;
    let cos_slice = cos_typed.as_slice_mut()
        .map_err(|_| Error::InvalidArgument("cos_cache is not contiguous".to_string()))?;

    if sin_slice.len() != expected_len || cos_slice.len() != expected_len {
         return Err(Error::InvalidArgument(format!(
            "Cache size mismatch. Expected {}, got sin {} and cos {}", 
            expected_len, sin_slice.len(), cos_slice.len()
        )).into());
    }

    let mut sin_view: ArrayViewMut2<f32> = ArrayViewMut2::from_shape(shape, sin_slice)
        .map_err(|e| Error::InvalidArgument(format!("sin_cache view creation failed: {}", e)))?;
    let mut cos_view: ArrayViewMut2<f32> = ArrayViewMut2::from_shape(shape, cos_slice)
        .map_err(|e| Error::InvalidArgument(format!("cos_cache view creation failed: {}", e)))?;


    // ... (2. 预先计算频率 freqs - 保持不变)
    let mut freqs = Vec::with_capacity(head_size);
    let head_size_f = head_size as f32;

    for head_dim in 0..head_size {
        let head_dim_f = head_dim as f32;
        let exponent = head_dim_f / head_size_f;
        let power_val = rope_base.powf(exponent);
        let freq = 1.0f32 / power_val;
        freqs.push(freq);
    }
    
    let freqs_ref: &[_] = freqs.as_slice();


    // 3. 核心多线程并行计算并填充缓存
    
    // 使用 .axis_iter_mut(Axis(0)) 获取行 (pos) 的可变迭代器
    // 然后调用 .into_par_iter() 转换为 rayon 的并行迭代器。
    // 注意：需要导入 `ndarray::RemoveAxis` trait。
    
    sin_view.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(cos_view.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate() // 引入 pos 索引
        .for_each(|(pos, (mut sin_row, mut cos_row))| {
            // sin_row 和 cos_row 现在是 ArrayViewMut1<f32> (行切片)
            let pos_f = pos as f32;

            // 迭代当前行 (head_dim)
            for head_dim in 0..head_size {
                let freq = freqs_ref[head_dim]; 
                
                // val = pos * freq
                let val = pos_f * freq;
                
                // 计算 cos(val) 和 sin(val)
                let fcr = val.cos(); // 余弦
                let fci = val.sin(); // 正弦
                
                // 写入当前行对应 head_dim 的元素
                // sin_row[head_dim] 和 cos_row[head_dim] 是 ndarray 视图的索引操作
                sin_row[head_dim] = fci;
                cos_row[head_dim] = fcr;
            }
        });

    Ok(())
}
pub fn rope_kernel_batch(
    // dim, kv_dim, head_size: 这些通常是 RoPEOp 算子的成员，可以从那里获取
    kv_dim: usize,
    head_size: usize,
    // 输入张量
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    // pos 张量现在告诉我们这个批次的起始位置
    start_pos_tensor: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match input_q.dtype() {
        crate::base::DataType::F32 => {
            rope_kernel_batch_f32(kv_dim, head_size, input_q, input_k, start_pos_tensor, sin_cache, cos_cache)
        }
        crate::base::DataType::BF16 => {
            rope_kernel_batch_bf16(kv_dim, head_size, input_q, input_k, start_pos_tensor, sin_cache, cos_cache)
        }
        _ => {
            Err(Error::InvalidArgument(format!(
                "Unsupported data type for rope_kernel_batch: {:?}", input_q.dtype()
            )).into())
        }
    }
}

fn rope_kernel_batch_f32(
    // dim, kv_dim, head_size: 这些通常是 RoPEOp 算子的成员，可以从那里获取
    kv_dim: usize,
    head_size: usize,
    // 输入张量
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    // pos 张量现在告诉我们这个批次的起始位置
    start_pos_tensor: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    if input_q.shape().len() != 2 || input_k.shape().len() != 2 {
        return Err(Error::InvalidArgument("Input Q and K for batch RoPE must be 2D.".to_string()).into());
    }
    let seq_len = input_q.shape()[0];
    let dim = input_q.shape()[1];
    let q_slice = input_q.as_f32_mut()?.as_slice_mut()?;
    let k_slice = input_k.as_f32_mut()?.as_slice_mut()?;
    let sin_slice = sin_cache.as_f32()?.as_slice()?;
    let cos_slice = cos_cache.as_f32()?.as_slice()?;
    let start_pos_slice = start_pos_tensor.as_i32()?.as_slice()?;
    if start_pos_slice.is_empty() {
        return Err(Error::InvalidArgument("start_pos_tensor is empty".to_string()).into());
    }
    let start_pos = start_pos_slice[0] as usize;
    // --- 3. 维度和边界检查 ---
    let max_pos = start_pos + seq_len-1;
    let max_cache_index = max_pos * head_size;
    if max_cache_index > sin_slice.len() || max_cache_index > cos_slice.len() {
        return Err(Error::IndexOutOfBounds(
            format!("RoPE cache index out of bounds. Max pos={}, head_size={}, cache_len={}",
                    max_pos, head_size, sin_slice.len())
        ).into());
    }
    // --- 4. 将切片包装成二维 ndarray 视图以方便按行迭代 ---
    
    let head_dim = head_size; // 每个头的维度（64）

    // ========== 核心顺序逻辑：遍历每个token（行） ==========
    for i in 0..seq_len {
        // 计算当前token的绝对位置
        let pos = start_pos + i;
        // ========== 处理Q的旋转：逐token+逐成对维度 ==========
        // Q当前行的起始索引：i * dim
        let q_row_start = i * dim;
        let k_row_start = i * kv_dim;
        for j in (0..dim).step_by(head_dim) {
            for k in 0..head_dim/2 {
                let sin_val = sin_slice[pos * head_dim + k*2];
                let cos_val = cos_slice[pos * head_dim + k*2];
                let q_idx_j = q_row_start + j + k;
                let q_idx_j1 = q_row_start + j + k + head_dim / 2;
                let v0_q = q_slice[q_idx_j];
                let v1_q = q_slice[q_idx_j1];
                q_slice[q_idx_j] = v0_q * cos_val - v1_q * sin_val;
                q_slice[q_idx_j1] = v0_q * sin_val + v1_q * cos_val;
                if j < kv_dim {
                    let k_idx_j = k_row_start + j + k;
                    let k_idx_j1 = k_row_start + j + k + head_dim / 2;
                    let v0_k = k_slice[k_idx_j];
                    let v1_k = k_slice[k_idx_j1];
                    k_slice[k_idx_j] = v0_k * cos_val - v1_k * sin_val;
                    k_slice[k_idx_j1] = v0_k * sin_val + v1_k * cos_val;
                }
            }
        }
    }
    Ok(())
}