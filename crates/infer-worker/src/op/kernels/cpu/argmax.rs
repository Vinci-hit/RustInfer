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