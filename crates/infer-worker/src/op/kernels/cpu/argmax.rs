use crate::base::error::{Result, Error};
use crate::base::DataType;
use crate::tensor::Tensor;

/// 在 CPU 上对 1D 张量执行 argmax 操作。
pub fn argmax(logits: &Tensor,output_token:&mut Tensor) -> Result<()> {
    // 根据数据类型分派到具体的实现
    let max_idx = match logits.dtype() {
        DataType::BF16 => {
            let slice = logits.as_bf16()?.as_slice()?;
            slice.iter()
                .enumerate()
                // 使用 partial_cmp 来处理浮点数的比较
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                .map(|(index, _)| index)
                // 如果张量为空，默认返回 token 0
                .unwrap_or(0)
        }
        DataType::F32 => {
            let slice = logits.as_f32()?.as_slice()?;
            slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                .map(|(index, _)| index)
                .unwrap_or(0)
        }
        unsupported_dtype => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported dtype {:?} for CPU argmax kernel.",
                unsupported_dtype
            )).into());
        }
    };
    let output_slice = output_token.as_i32_mut()?.as_slice_mut()?;
    output_slice[0] = max_idx as i32;
    Ok(())
}

/// Batch argmax on CPU
///
/// Input: logits [batch_size, vocab_size]
/// Output: token_ids [batch_size], each element is the argmax of corresponding row
pub fn argmax_batch(logits: &Tensor, output_tokens: &mut Tensor) -> Result<()> {
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(
            format!("Expected 2D logits [batch_size, vocab_size], got shape {:?}", shape)
        ).into());
    }

    let batch_size = shape[0];
    let vocab_size = shape[1];

    let output_slice = output_tokens.as_i32_mut()?.as_slice_mut()?;
    if output_slice.len() != batch_size {
        return Err(Error::InvalidArgument(
            format!("Output size {} != batch_size {}", output_slice.len(), batch_size)
        ).into());
    }

    match logits.dtype() {
        DataType::BF16 => {
            let logits_data = logits.as_bf16()?.as_slice()?;
            for batch_idx in 0..batch_size {
                let row_start = batch_idx * vocab_size;
                let row_end = row_start + vocab_size;
                let row = &logits_data[row_start..row_end];

                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(index, _)| index)
                    .unwrap_or(0);

                output_slice[batch_idx] = max_idx as i32;
            }
        }
        DataType::F32 => {
            let logits_data = logits.as_f32()?.as_slice()?;
            for batch_idx in 0..batch_size {
                let row_start = batch_idx * vocab_size;
                let row_end = row_start + vocab_size;
                let row = &logits_data[row_start..row_end];

                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(index, _)| index)
                    .unwrap_or(0);

                output_slice[batch_idx] = max_idx as i32;
            }
        }
        unsupported_dtype => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported dtype {:?} for CPU argmax_batch kernel.",
                unsupported_dtype
            )).into());
        }
    }

    Ok(())
}
