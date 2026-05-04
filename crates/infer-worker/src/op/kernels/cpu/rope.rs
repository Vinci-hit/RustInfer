use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;
use ndarray::{ArrayViewMut2, Axis};
use rayon::prelude::*; // 导入 rayon 的并行迭代器 trait

pub fn sin_cos_cache_calc_bf16(
    head_size: usize,
    max_seq_len: usize,
    rope_theta: f32,
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
        1.0f32 / rope_theta.powf(exponent)
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
    rope_theta: f32,
    sin_cache: &mut Tensor,
    cos_cache: &mut Tensor,
) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match sin_cache.dtype() {
        crate::base::DataType::F32 => {
            sin_cos_cache_calc_f32(head_size, max_seq_len, rope_theta, sin_cache, cos_cache)
        }
        crate::base::DataType::BF16 => {
            sin_cos_cache_calc_bf16(head_size, max_seq_len, rope_theta, sin_cache, cos_cache)
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
    rope_theta: f32,
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
    let base_f = rope_theta;

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
// ============================================================================
// RoPE (LLM, BF16 / F32)：**唯一**语义 —— positions 长度 = seq_len，
// 每行使用 positions[i] 作为绝对位置。
// 所有 caller（prefill 传 [p, p+1, ...]，decode 传 [p] 或 [p0, p1, ...]）
// 都走这条路径。
//
// Stride-aware：Q / K tensor 允许是 strided view（例如 `qkv.narrow(1, ...)`
// 切出来的列段），kernel 通过 `strides()[0]` + `data_ptr()` 按行步长访问，
// 不再要求整块连续。sin_cache / cos_cache 仍然假定是连续的 `[max_seq, head_size]`。
// ============================================================================

/// 核心 stride-aware 内核：以 2D row-major 视图访问 `[seq_len, inner_dim]`，
/// 但行步长由 `row_stride` 参数给出（元素单位）。
///
/// SAFETY: `base` 必须指向 tensor 当前 view 的第 0 元素（含 storage_offset），
/// 访问范围为 `[base + row * row_stride + col]`，0 ≤ row < seq_len,
/// 0 ≤ col < inner_dim。
///
/// RoPE 只在 inner 维度前 `inner_dim` 个元素内做旋转；inner 内部必是连续的
/// （`narrow(1, ...)` 切的列段天然满足：列维 stride == 1）。
#[inline(always)]
fn rope_rotate_bf16(
    q_base: *mut bf16,
    q_row_stride: usize,
    q_inner_dim: usize,
    k_base: *mut bf16,
    k_row_stride: usize,
    k_inner_dim: usize,
    head_size: usize,
    seq_len: usize,
    positions: &[i32],
    sin_cache: &[bf16],
    cos_cache: &[bf16],
) {
    let half = head_size / 2;
    for i in 0..seq_len {
        let pos = positions[i] as usize;
        // 行内偏移用指针加法，不再走 flat index（后者要求整块连续）
        let q_row = unsafe { q_base.add(i * q_row_stride) };
        let k_row = unsafe { k_base.add(i * k_row_stride) };
        let sin_row = &sin_cache[pos * head_size..pos * head_size + head_size];
        let cos_row = &cos_cache[pos * head_size..pos * head_size + head_size];

        let mut j = 0usize;
        while j < q_inner_dim {
            for k in 0..half {
                let sin_val = sin_row[k * 2];
                let cos_val = cos_row[k * 2];
                unsafe {
                    let p0 = q_row.add(j + k);
                    let p1 = q_row.add(j + k + half);
                    let v0 = *p0;
                    let v1 = *p1;
                    *p0 = v0 * cos_val - v1 * sin_val;
                    *p1 = v0 * sin_val + v1 * cos_val;
                    // K 只在它自己的 inner 范围内做
                    if j + head_size <= k_inner_dim {
                        let kp0 = k_row.add(j + k);
                        let kp1 = k_row.add(j + k + half);
                        let kv0 = *kp0;
                        let kv1 = *kp1;
                        *kp0 = kv0 * cos_val - kv1 * sin_val;
                        *kp1 = kv0 * sin_val + kv1 * cos_val;
                    }
                }
            }
            j += head_size;
        }
    }
}

#[inline(always)]
fn rope_rotate_f32(
    q_base: *mut f32,
    q_row_stride: usize,
    q_inner_dim: usize,
    k_base: *mut f32,
    k_row_stride: usize,
    k_inner_dim: usize,
    head_size: usize,
    seq_len: usize,
    positions: &[i32],
    sin_cache: &[f32],
    cos_cache: &[f32],
) {
    let half = head_size / 2;
    for i in 0..seq_len {
        let pos = positions[i] as usize;
        let q_row = unsafe { q_base.add(i * q_row_stride) };
        let k_row = unsafe { k_base.add(i * k_row_stride) };
        let sin_row = &sin_cache[pos * head_size..pos * head_size + head_size];
        let cos_row = &cos_cache[pos * head_size..pos * head_size + head_size];

        let mut j = 0usize;
        while j < q_inner_dim {
            for k in 0..half {
                let sin_val = sin_row[k * 2];
                let cos_val = cos_row[k * 2];
                unsafe {
                    let p0 = q_row.add(j + k);
                    let p1 = q_row.add(j + k + half);
                    let v0 = *p0;
                    let v1 = *p1;
                    *p0 = v0 * cos_val - v1 * sin_val;
                    *p1 = v0 * sin_val + v1 * cos_val;
                    if j + head_size <= k_inner_dim {
                        let kp0 = k_row.add(j + k);
                        let kp1 = k_row.add(j + k + half);
                        let kv0 = *kp0;
                        let kv1 = *kp1;
                        *kp0 = kv0 * cos_val - kv1 * sin_val;
                        *kp1 = kv0 * sin_val + kv1 * cos_val;
                    }
                }
            }
            j += head_size;
        }
    }
}

/// 取 2D tensor 的 row stride（元素单位）。若是连续 storage，等于列数。
#[inline]
fn row_stride_2d(t: &Tensor) -> Result<usize> {
    let s = t.strides();
    if s.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "rope: expected 2D tensor, got shape {:?}", t.shape()
        )).into());
    }
    // 列维 stride 必须 == 1（即最后一维在内存里连续），否则 RoPE 的 head 级访问会错。
    if s[1] != 1 {
        return Err(Error::InvalidArgument(format!(
            "rope: inner-dim stride must be 1, got strides={:?}", s
        )).into());
    }
    Ok(s[0])
}

fn rope_kernel_batch_bf16(
    kv_inner_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    positions: &Tensor, // [B] i32
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    if input_q.shape().len() != 2 || input_k.shape().len() != 2 {
        return Err(Error::InvalidArgument(
            "rope: Q and K must be 2D [seq_len, inner_dim].".into(),
        ).into());
    }
    let seq_len = input_q.shape()[0];
    if input_k.shape()[0] != seq_len {
        return Err(Error::InvalidArgument(format!(
            "rope: Q and K seq_len mismatch: {} vs {}", seq_len, input_k.shape()[0]
        )).into());
    }
    let q_inner_dim = input_q.shape()[1];
    if kv_inner_dim != input_k.shape()[1] {
        return Err(Error::InvalidArgument(format!(
            "rope: kv_dim arg ({}) mismatches K.shape[1] ({})",
            kv_inner_dim, input_k.shape()[1]
        )).into());
    }

    let q_row_stride = row_stride_2d(input_q)?;
    let k_row_stride = row_stride_2d(input_k)?;

    let pos_slice = positions.as_i32()?.as_slice()?;
    if pos_slice.len() < seq_len {
        return Err(Error::InvalidArgument(format!(
            "rope: positions.len ({}) < seq_len ({})", pos_slice.len(), seq_len
        )).into());
    }
    let sin_slice = sin_cache.as_bf16()?.as_slice()?;
    let cos_slice = cos_cache.as_bf16()?.as_slice()?;

    let q_base = input_q.as_bf16_mut()?.data_ptr_mut();
    let k_base = input_k.as_bf16_mut()?.data_ptr_mut();

    rope_rotate_bf16(
        q_base, q_row_stride, q_inner_dim,
        k_base, k_row_stride, kv_inner_dim,
        head_size, seq_len,
        pos_slice, sin_slice, cos_slice,
    );
    Ok(())
}

fn rope_kernel_batch_f32(
    kv_inner_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    positions: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    if input_q.shape().len() != 2 || input_k.shape().len() != 2 {
        return Err(Error::InvalidArgument(
            "rope: Q and K must be 2D [seq_len, inner_dim].".into(),
        ).into());
    }
    let seq_len = input_q.shape()[0];
    if input_k.shape()[0] != seq_len {
        return Err(Error::InvalidArgument(format!(
            "rope: Q and K seq_len mismatch: {} vs {}", seq_len, input_k.shape()[0]
        )).into());
    }
    let q_inner_dim = input_q.shape()[1];
    if kv_inner_dim != input_k.shape()[1] {
        return Err(Error::InvalidArgument(format!(
            "rope: kv_dim arg ({}) mismatches K.shape[1] ({})",
            kv_inner_dim, input_k.shape()[1]
        )).into());
    }

    let q_row_stride = row_stride_2d(input_q)?;
    let k_row_stride = row_stride_2d(input_k)?;

    let pos_slice = positions.as_i32()?.as_slice()?;
    if pos_slice.len() < seq_len {
        return Err(Error::InvalidArgument(format!(
            "rope: positions.len ({}) < seq_len ({})", pos_slice.len(), seq_len
        )).into());
    }
    let sin_slice = sin_cache.as_f32()?.as_slice()?;
    let cos_slice = cos_cache.as_f32()?.as_slice()?;

    let q_base = input_q.as_f32_mut()?.data_ptr_mut();
    let k_base = input_k.as_f32_mut()?.data_ptr_mut();

    rope_rotate_f32(
        q_base, q_row_stride, q_inner_dim,
        k_base, k_row_stride, kv_inner_dim,
        head_size, seq_len,
        pos_slice, sin_slice, cos_slice,
    );
    Ok(())
}

pub fn rope_kernel_batch(
    kv_dim: usize,
    head_size: usize,
    input_q: &mut Tensor,
    input_k: &mut Tensor,
    positions: &Tensor,
    sin_cache: &Tensor,
    cos_cache: &Tensor,
) -> Result<()> {
    match input_q.dtype() {
        crate::base::DataType::F32 => rope_kernel_batch_f32(
            kv_dim, head_size, input_q, input_k, positions, sin_cache, cos_cache,
        ),
        crate::base::DataType::BF16 => rope_kernel_batch_bf16(
            kv_dim, head_size, input_q, input_k, positions, sin_cache, cos_cache,
        ),
        dt => Err(Error::InvalidArgument(format!(
            "Unsupported data type for rope_kernel_batch: {:?}", dt
        )).into()),
    }
}
