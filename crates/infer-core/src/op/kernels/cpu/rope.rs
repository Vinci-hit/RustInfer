use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;
use ndarray::{ArrayViewMut2, Axis};
use rayon::prelude::*; // 导入 rayon 的并行迭代器 trait

pub fn sin_cos_cache_calc_bf16(
    head_size: usize,
    max_seq_len: usize,
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
        1.0f32 / 10000.0f32.powf(exponent)
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

/// 计算并填充正弦和余弦旋转嵌入 (RoPE) 的缓存。
/// 
/// 该函数使用 `rayon` 库进行多线程并行计算，极大地加速了缓存的生成。
/// 缓存的形状为 `[max_seq_len, head_size]`。
/// 
/// # Arguments
/// * `head_size`: 旋转维度的大小 (K)。
/// * `max_seq_len`: 序列的最大长度 (M)。
/// * `sin_cache`: 正弦值输出张量, 形状 [max_seq_len, head_size]。
/// * `cos_cache`: 余弦值输出张量, 形状 [max_seq_len, head_size]。
pub fn sin_cos_cache_calc(
    head_size: usize,
    max_seq_len: usize,
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
    let base_f = 500000.0f32;

    for head_dim in 0..head_size {
        let head_dim_f = head_dim as f32;
        let exponent = head_dim_f / head_size_f;
        let power_val = base_f.powf(exponent);
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
// ndarray 在这里没有直接用到视图，因为操作是逐元素的，但我们保留了风格和 Result 的使用。

/// 应用 Rotary Positional Embedding (RoPE) 旋转核。
/// 这是一个就地 (in-place) 操作，会修改 input_q 和 input_k 的内容。
/// 
/// # Arguments
/// * `dim`: Q 和 K 向量的总旋转维度 (通常是模型维度)。
/// * `kv_dim`: K 向量旋转的维度。如果 `dim > kv_dim`，则 Q 在剩余维度上单独旋转。
/// * `head_size`: Attention Head 的大小，用于 RoPE 缓存的索引计算。
/// * `input_q`: Query 张量，将被就地修改。
/// * `input_k`: Key 张量，将被就地修改 (直到 kv_dim)。
/// * `input_pos`: 包含当前位置索引的张量 (通常是 [1] 形状的 i32 张量)。
/// * `sin_cache`: 正弦缓存张量。
/// * `cos_cache`: 余弦缓存张量。
pub fn rope_kernel(
    dim: usize,
    kv_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    input_pos: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    // --- 1. 获取类型化的数据切片 ---
    // Q 和 K 需要可变切片
    let q_slice = input_q.as_f32_mut()?.as_slice_mut()?;
    let k_slice = input_k.as_f32_mut()?.as_slice_mut()?;

    // Cache 需要不可变切片
    let sin_slice = sin_cache.as_f32()?.as_slice()?;
    let cos_slice = cos_cache.as_f32()?.as_slice()?;
    
    // Position 需要 i32 切片
    let pos_slice = input_pos.as_i32()?.as_slice()?;

    // --- 2. 获取当前位置 `pos` 并进行基本检查 ---
    if pos_slice.is_empty() {
        return Err(Error::InvalidArgument("input_pos tensor is empty".to_string()).into());
    }
    
    let pos = pos_slice[0] as usize;

    // --- 3. 维度检查 (保障后续索引安全) ---

    if kv_dim > dim {
         return Err(Error::InvalidArgument(
            format!("kv_dim ({}) cannot be greater than dim ({}).", kv_dim, dim)
        ).into());
    }
    // 缓存索引 pos * head_size + head_size - 1 必须小于缓存长度
    // 假设缓存是 [max_seq_len, head_size]
    let max_cache_index = (pos + 1) * head_size;
    if max_cache_index > sin_slice.len() || max_cache_index > cos_slice.len() {
        return Err(Error::IndexOutOfBounds(
            format!("RoPE cache index out of bounds. pos={}, head_size={}, cache_len={}",
                    pos, head_size, sin_slice.len())
        ).into());
    }
    

    // --- 4. 核心 RoPE 旋转计算 ---
    // 循环从 0 到 dim，步长为 2 (处理 (i, i+1) 对)
    for i in (0..dim).step_by(2) {
        // 计算缓存的索引
        let head_dim = i % head_size;
        let cache_idx = pos * head_size + head_dim;

        // 加载 sin 和 cos 值
        // 由于我们在 3 中已经检查了边界，这里可以直接使用 [cache_idx]
        let fci = sin_slice[cache_idx]; // sin(val)
        let fcr = cos_slice[cache_idx]; // cos(val)

        // **旋转 Query (Q) 向量**
        // 获取 Q 的 (v0, v1)
        let v0_q = q_slice[i];
        let v1_q = q_slice[i + 1];
        
        // 应用旋转: v0' = v0 * cos - v1 * sin, v1' = v0 * sin + v1 * cos
        q_slice[i]     = v0_q * fcr - v1_q * fci;
        q_slice[i + 1] = v0_q * fci + v1_q * fcr;
        
        // **旋转 Key (K) 向量**

        if i < kv_dim {
            // 获取 K 的 (v0, v1)
            let v0_k = k_slice[i];
            let v1_k = k_slice[i + 1];

            // 应用旋转
            k_slice[i]     = v0_k * fcr - v1_k * fci;
            k_slice[i + 1] = v0_k * fci + v1_k * fcr;
        }
    }

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
    // --- 1. 获取类型化的数据切片 ---
    // (与之前类似，但现在我们将它们包装成二维 ndarray 视图)
    // --- 2. 获取维度和起始位置 ---
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
    
    let mut q_view: ArrayViewMut2<f32> = ArrayViewMut2::from_shape((seq_len, dim), q_slice).map_err(|e| 
        Error::InvalidArgument(format!("Q view creation failed: {}", e))
    )?;
    let mut k_view: ArrayViewMut2<f32> = ArrayViewMut2::from_shape((seq_len, kv_dim), k_slice).map_err(|e| 
        Error::InvalidArgument(format!("K view creation failed: {}", e))
    )?;
    
    // --- 5. 核心 RoPE 旋转计算 (并行化) ---
    // a) 并行地迭代 q_view 和 k_view 的每一行
    q_view.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(k_view.axis_iter_mut(Axis(0)).into_par_iter())
        // b) 使用 .enumerate() 来获取当前行在批次中的相对索引 `i`
        .enumerate()
        .for_each(|(i, (mut q_row, mut k_row))| {
            // c) 计算当前 token 在整个序列中的绝对位置 `pos`
            let pos = start_pos + i;

            // d) 迭代当前行的每个维度对 (j, j+1)
            for j in (0..dim).step_by(2) {
                // 计算在 sin/cos 缓存中查找的索引
                let head_dim = j % head_size;
                let cache_idx = pos * head_size + head_dim;

                // 加载 sin 和 cos 值 (边界已检查，这里是安全的)
                let fci = sin_slice[cache_idx];
                let fcr = cos_slice[cache_idx];

                // **旋转 Query (Q) 行向量**
                let v0_q = q_row[j];
                let v1_q = q_row[j + 1];
                q_row[j]     = v0_q * fcr - v1_q * fci;
                q_row[j + 1] = v0_q * fci + v1_q * fcr;

                // **旋转 Key (K) 行向量**
                // 只有当维度 j 在 kv_dim 范围内时才旋转
                if j < kv_dim {
                    let v0_k = k_row[j];
                    let v1_k = k_row[j + 1];
                    k_row[j]     = v0_k * fcr - v1_k * fci;
                    k_row[j + 1] = v0_k * fci + v1_k * fcr;
                }
            }
        });
    
    Ok(())
}