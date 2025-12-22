use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;
use ndarray::{s, ArrayView2, ArrayViewMut3, Axis}; 
use rayon::prelude::*; 
use std::f32;

/// Flash Attention V2 的 CPU 核心实现 (Batch=1, K/V Cache 模式)。
/// 
/// **警告：Softmax 采用非数值稳定实现，仅用作测试的黄金标准。**
/// 
/// # Arguments
/// * `input_q`: Query 张量，形状为 [Q_SeqLen, Q_HiddenDim]
/// * `input_k_cache`, `input_v_cache`: K, V Cache 的完整内存，形状为 [Max_SeqLen, KV_HiddenDim]
/// * `output_o`: 输出张量，形状为 [Q_SeqLen, Q_HiddenDim]
/// * `q_seq_len`: Query 的序列长度 (S_Q)。
/// * `current_kv_len`: K/V Cache 的有效历史长度 (S_KV_history)。
pub fn flash_attn_gqa(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    q_seq_len: usize,
    current_kv_len: usize, // 之前已有的 KV 长度
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match (input_q.dtype(), input_k_cache.dtype(), input_v_cache.dtype(), output_o.dtype()) {
        (
            crate::base::DataType::F32,
            crate::base::DataType::F32,
            crate::base::DataType::F32,
            crate::base::DataType::F32,
        ) => {
            flash_attn_gqa_f32(
                input_q,
                input_k_cache,
                input_v_cache,
                output_o,
                q_seq_len,
                current_kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
            )
        }
        (
            crate::base::DataType::BF16,
            crate::base::DataType::BF16,
            crate::base::DataType::BF16,
            crate::base::DataType::BF16,
        ) => {
            flash_attn_gqa_bf16(
                input_q,
                input_k_cache,
                input_v_cache,
                output_o,
                q_seq_len,
                current_kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
            )
        }
        _ => Err(Error::InvalidArgument(
            "Unsupported data type combination for flash_attn_gqa".to_string(),
        ).into()),
    }
}

/// F32版本的Flash Attention GQA实现
fn flash_attn_gqa_f32(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    q_seq_len: usize,
    current_kv_len: usize, // 之前已有的 KV 长度
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    // --- 1. 获取数据切片 ---
    let q_slice = input_q.as_f32()?;
    let k_slice = input_k_cache.as_f32()?;
    let v_slice = input_v_cache.as_f32()?;
    let o_slice_mut = output_o.as_f32_mut()?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let groups = num_q_heads / num_kv_heads;
    let max_kv_seq_len = input_k_cache.shape()[0];
    let total_seq_len = q_seq_len + current_kv_len;

    let q_hidden_dim = num_q_heads * head_dim;
    let kv_hidden_dim = num_kv_heads * head_dim;

    // --- 2. 创建视图 ---
    
    // Q: [Seq, Num_Q_Heads * Head_Dim] -> 逻辑上看作 [Seq, Num_Q_Heads, Head_Dim]
    // 为了方便切片，我们保持 2D 视图，在循环中手动切分列
    let q_view = ArrayView2::from_shape((q_seq_len, q_hidden_dim), q_slice.as_slice()?)
        .map_err(|e| Error::InvalidArgument(format!("Q view failed: {}", e)))?;

    // K, V: [Max_Seq, Num_KV_Heads * Head_Dim]
    let k_full_view = ArrayView2::from_shape((max_kv_seq_len, kv_hidden_dim), k_slice.as_slice()?)
        .map_err(|e| Error::InvalidArgument(format!("K view failed: {}", e)))?;
    let v_full_view = ArrayView2::from_shape((max_kv_seq_len, kv_hidden_dim), v_slice.as_slice()?)
        .map_err(|e| Error::InvalidArgument(format!("V view failed: {}", e)))?;

    // Output: 为了安全并行，我们将视图重塑为 3D [Seq, Num_Heads, Head_Dim]
    // 这样我们可以沿着 Axis(1) (Heads) 进行可变迭代，Rayon 就能安全地分发任务
    let mut o_view_3d = ArrayViewMut3::from_shape(
        (q_seq_len, num_q_heads, head_dim),
        o_slice_mut.as_slice_mut()?,
    ).map_err(|e| Error::InvalidArgument(format!("O view failed: {}", e)))?;

    // --- 3. 并行计算 (按 Query Head 维度) ---

    // 将 Output 按 Head 切分成独立的视图，收集后并行迭代
    // 这一步消除了 unsafe，因为编译器知道这些切片是互不重叠的
    let o_heads: Vec<_> = o_view_3d.axis_iter_mut(Axis(1)).collect();

    o_heads.into_par_iter().enumerate().for_each(|(hq, mut o_head_view)| {
        // 计算当前 Query Head 对应的 KV Head 索引
        let hkv = hq / groups;

        // A. 切片获取 (Zero-Copy)
        // Q_head: [Seq_Q, Head_Dim]
        let q_head = q_view.slice(s![.., hq * head_dim..(hq + 1) * head_dim]);
        
        // K_head, V_head: [Seq_Total, Head_Dim] (只取有效长度)
        let k_head = k_full_view.slice(s![0..total_seq_len, hkv * head_dim..(hkv + 1) * head_dim]);
        let v_head = v_full_view.slice(s![0..total_seq_len, hkv * head_dim..(hkv + 1) * head_dim]);

        // B. Attention Scores 计算: S = Q · K^T
        // shape: [Seq_Q, Seq_Total]
        let mut scores = q_head.dot(&k_head.t());
        
        // C. Scale
        scores *= scale;

        // D. Causal Masking (因果掩码)
        // 逻辑: 对于 Q 的第 i 行 (绝对位置 current_kv_len + i)，
        // 只能看 K 的前 current_kv_len + i + 1 个元素。
        // 将 j > current_kv_len + i 的位置设为 -inf
        for (i, mut row) in scores.outer_iter_mut().enumerate() {
            let valid_len = current_kv_len + i + 1;
            if valid_len < total_seq_len {
                row.slice_mut(s![valid_len..]).fill(f32::NEG_INFINITY);
            }
        }

        // E. Standard Softmax (Vectorized)
        // 1. Find Max (for numerical stability)
        // max_per_row: [Seq_Q, 1]
        let max_vals = scores.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        
        // 2. Exp(x - max)
        // 利用广播机制减去最大值
        let scores_exps = (&scores - &max_vals.insert_axis(Axis(1))).mapv(f32::exp);

        // 3. Sum Exp
        let sum_exps = scores_exps.sum_axis(Axis(1));

        // 4. Divide -> Probabilities
        // 注意处理除以 0 的情况 (虽然有 mask 且 max 减法通常不会全 0，但为了稳健)
        let probs = &scores_exps / &sum_exps.insert_axis(Axis(1));

        // F. Output Calculation: O = Probs · V
        // shape: [Seq_Q, Head_Dim]
        let output_result = probs.dot(&v_head);

        // G. 写入结果
        // o_head_view 是 Output 张量对应 Head 的可变视图，直接 assign 即可
        o_head_view.assign(&output_result);
    });

    Ok(())
}

/// BF16版本的Flash Attention GQA实现
/// 为了数值稳定性，内部计算仍然使用F32，输入输出使用BF16
fn flash_attn_gqa_bf16(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    q_seq_len: usize,
    current_kv_len: usize, // 之前已有的 KV 长度
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    // --- 1. 获取数据切片 ---
    let q_slice = input_q.as_bf16()?;
    let k_slice = input_k_cache.as_bf16()?;
    let v_slice = input_v_cache.as_bf16()?;
    let o_slice_mut = output_o.as_bf16_mut()?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let groups = num_q_heads / num_kv_heads;
    let max_kv_seq_len = input_k_cache.shape()[0];
    let total_seq_len = q_seq_len + current_kv_len;

    let q_hidden_dim = num_q_heads * head_dim;
    let kv_hidden_dim = num_kv_heads * head_dim;

    // --- 2. 转换为F32进行计算 ---
    // 将BF16数据转换为F32以保证数值稳定性
    let q_f32: Vec<f32> = q_slice.as_slice()?.iter().map(|&x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_slice.as_slice()?.iter().map(|&x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_slice.as_slice()?.iter().map(|&x| x.to_f32()).collect();
    let mut o_f32 = vec![0.0f32; o_slice_mut.as_slice()?.len()];

    // --- 3. 创建视图 ---
    
    // Q: [Seq, Num_Q_Heads * Head_Dim] -> 逻辑上看作 [Seq, Num_Q_Heads, Head_Dim]
    let q_view = ArrayView2::from_shape((q_seq_len, q_hidden_dim), q_f32.as_slice())
        .map_err(|e| Error::InvalidArgument(format!("Q view failed: {}", e)))?;

    // K, V: [Max_Seq, Num_KV_Heads * Head_Dim]
    let k_full_view = ArrayView2::from_shape((max_kv_seq_len, kv_hidden_dim), k_f32.as_slice())
        .map_err(|e| Error::InvalidArgument(format!("K view failed: {}", e)))?;
    let v_full_view = ArrayView2::from_shape((max_kv_seq_len, kv_hidden_dim), v_f32.as_slice())
        .map_err(|e| Error::InvalidArgument(format!("V view failed: {}", e)))?;

    // Output: 为了安全并行，我们将视图重塑为 3D [Seq, Num_Heads, Head_Dim]
    let mut o_view_3d = ArrayViewMut3::from_shape(
        (q_seq_len, num_q_heads, head_dim),
        o_f32.as_mut_slice(),
    ).map_err(|e| Error::InvalidArgument(format!("O view failed: {}", e)))?;

    // --- 4. 并行计算 (按 Query Head 维度) ---

    // 将 Output 按 Head 切分成独立的视图，收集后并行迭代
    let o_heads: Vec<_> = o_view_3d.axis_iter_mut(Axis(1)).collect();

    o_heads.into_par_iter().enumerate().for_each(|(hq, mut o_head_view)| {
        // 计算当前 Query Head 对应的 KV Head 索引
        let hkv = hq / groups;

        // A. 切片获取 (Zero-Copy)
        // Q_head: [Seq_Q, Head_Dim]
        let q_head = q_view.slice(s![.., hq * head_dim..(hq + 1) * head_dim]);
        
        // K_head, V_head: [Seq_Total, Head_Dim] (只取有效长度)
        let k_head = k_full_view.slice(s![0..total_seq_len, hkv * head_dim..(hkv + 1) * head_dim]);
        let v_head = v_full_view.slice(s![0..total_seq_len, hkv * head_dim..(hkv + 1) * head_dim]);

        // B. Attention Scores 计算: S = Q · K^T
        // shape: [Seq_Q, Seq_Total]
        let mut scores = q_head.dot(&k_head.t());
        
        // C. Scale
        scores *= scale;

        // D. Causal Masking (因果掩码)
        // 逻辑: 对于 Q 的第 i 行 (绝对位置 current_kv_len + i)，
        // 只能看 K 的前 current_kv_len + i + 1 个元素。
        // 将 j > current_kv_len + i 的位置设为 -inf
        for (i, mut row) in scores.outer_iter_mut().enumerate() {
            let valid_len = current_kv_len + i + 1;
            if valid_len < total_seq_len {
                row.slice_mut(s![valid_len..]).fill(f32::NEG_INFINITY);
            }
        }

        // E. Standard Softmax (Vectorized)
        // 1. Find Max (for numerical stability)
        // max_per_row: [Seq_Q, 1]
        let max_vals = scores.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        
        // 2. Exp(x - max)
        // 利用广播机制减去最大值
        let scores_exps = (&scores - &max_vals.insert_axis(Axis(1))).mapv(f32::exp);

        // 3. Sum Exp
        let sum_exps = scores_exps.sum_axis(Axis(1));

        // 4. Divide -> Probabilities
        // 注意处理除以 0 的情况 (虽然有 mask 且 max 减法通常不会全 0，但为了稳健)
        let probs = &scores_exps / &sum_exps.insert_axis(Axis(1));

        // F. Output Calculation: O = Probs · V
        // shape: [Seq_Q, Head_Dim]
        let output_result = probs.dot(&v_head);

        // G. 写入结果
        // o_head_view 是 Output 张量对应 Head 的可变视图，直接 assign 即可
        o_head_view.assign(&output_result);
    });

    // --- 5. 将结果从F32转换回BF16 ---
    let o_slice = o_slice_mut.as_slice_mut()?;
    for (i, &val_f32) in o_f32.iter().enumerate() {
        o_slice[i] = bf16::from_f32(val_f32);
    }

    Ok(())
}