use crate::base::error::{Result, Error};
use crate::tensor::Tensor;
use rayon::prelude::*; // 引入 rayon 的并行迭代器 trait

/// Embedding 的 CPU 高性能内核实现 (并行化)
pub fn embedding(input_tokens: &Tensor, weight: &Tensor, output: &mut Tensor) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match (weight.dtype(), output.dtype()) {
        (crate::base::DataType::F32, crate::base::DataType::F32) => {
            embedding_f32(input_tokens, weight, output)
        }
        (crate::base::DataType::BF16, crate::base::DataType::BF16) => {
            embedding_bf16(input_tokens, weight, output)
        }
        _ => {
            // 如果输入和输出的数据类型不匹配，则返回错误
            Err(Error::InvalidArgument(format!(
                "Unsupported data type combination for embedding: weight={:?}, output={:?}",
                weight.dtype(), output.dtype()
            )).into())
        }
    }
}

/// F32版本的Embedding实现
fn embedding_f32(input_tokens: &Tensor, weight: &Tensor, output: &mut Tensor) -> Result<()> {
    // --- 1. 获取数据 slice (与之前版本相同) ---
    let tokens_typed = input_tokens.as_i32()?;
    let weight_typed = weight.as_f32()?;
    let output_typed = output.as_f32_mut()?;

    let tokens_slice = tokens_typed.as_slice()?;
    let weight_slice = weight_typed.as_slice()?;
    let output_slice = output_typed.as_slice_mut()?;
    let vocab_size = weight.shape()[0];
    let dim = weight.shape()[1];

    // --- 2. 执行并行查表 (内存拷贝) ---

    // a) 将输出 slice 按照 `dim` 的大小，切分成多个可变的行 (chunks)
    //    .par_chunks_mut(dim) 会返回一个并行迭代器，每个元素是一个 `&mut [f32]`
    output_slice
        .par_chunks_mut(dim)
        // b) 将输出的每一行，与输入的 token ID 并行地 zip 在一起
        .zip(tokens_slice.par_iter())
        // c) 对每一对 (输出行, token_id) 并行地执行操作
        .for_each(|(output_row, &token_id)| {
            // 安全检查：检查 token_id 是否有效
            // 我们可以在这里选择 panic (因为这是不可恢复的逻辑错误) 或在 Result 中处理
            // 在内核中，panic 更直接
            let token_id = token_id as usize;
            assert!(
                token_id < vocab_size,
                "Token ID {} is out of bounds for vocab size {}",
                token_id, vocab_size
            );

            // 计算在权重矩阵中，该 token 对应的行的起始位置
            let weight_row_start = token_id * dim;
            let weight_row_end = weight_row_start + dim;
            
            // 从权重 slice 中切出对应的行
            let weight_row = &weight_slice[weight_row_start..weight_row_end];
            
            // 将权重行拷贝到输出行的 chunk 中
            output_row.copy_from_slice(weight_row);
        });
    Ok(())
}

/// BF16版本的Embedding实现
fn embedding_bf16(input_tokens: &Tensor, weight: &Tensor, output: &mut Tensor) -> Result<()> {
    // --- 1. 获取数据 slice ---
    let tokens_typed = input_tokens.as_i32()?;
    let weight_typed = weight.as_bf16()?;
    let output_typed = output.as_bf16_mut()?;

    let tokens_slice = tokens_typed.as_slice()?;
    let weight_slice = weight_typed.as_slice()?;
    let output_slice = output_typed.as_slice_mut()?;
    let vocab_size = weight.shape()[0];
    let dim = weight.shape()[1];

    // --- 2. 执行并行查表 (内存拷贝) ---

    // a) 将输出 slice 按照 `dim` 的大小，切分成多个可变的行 (chunks)
    //    .par_chunks_mut(dim) 会返回一个并行迭代器，每个元素是一个 `&mut [bf16]`
    output_slice
        .par_chunks_mut(dim)
        // b) 将输出的每一行，与输入的 token ID 并行地 zip 在一起
        .zip(tokens_slice.par_iter())
        // c) 对每一对 (输出行, token_id) 并行地执行操作
        .for_each(|(output_row, &token_id)| {
            // 安全检查：检查 token_id 是否有效
            let token_id = token_id as usize;
            assert!(
                token_id < vocab_size,
                "Token ID {} is out of bounds for vocab size {}",
                token_id, vocab_size
            );

            // 计算在权重矩阵中，该 token 对应的行的起始位置
            let weight_row_start = token_id * dim;
            let weight_row_end = weight_row_start + dim;
            
            // 从权重 slice 中切出对应的行
            let weight_row = &weight_slice[weight_row_start..weight_row_end];
            
            // 将权重行拷贝到输出行的 chunk 中
            output_row.copy_from_slice(weight_row);
        });
    Ok(())
}