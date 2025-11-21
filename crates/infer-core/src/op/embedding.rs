use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::op::{kernels, Op, OpContext};
use crate::tensor::Tensor;

/// Embedding 算子，根据输入的 token ID 从权重矩阵中查找嵌入向量。
pub struct Embedding {
    /// 权重矩阵 (查询表), 形状为 [vocab_size, dim]。
    pub weight: Tensor,
    
    // 配置信息
    vocab_size: usize,
    dim: usize,
}

impl Embedding {
    /// 创建一个新的 Embedding 算子。
    ///
    /// # Arguments
    /// * `vocab_size` - 词汇表的大小。
    /// * `dim` - 嵌入向量的维度。
    /// * `dtype` - 权重的数据类型 (通常是 F32)。
    /// * `device` - 参数所在的设备。
    pub fn new(vocab_size: usize, dim: usize, dtype: DataType, device: DeviceType) -> Result<Self> {
        let weight = Tensor::new(&[vocab_size, dim], dtype, device)?;
        Ok(Self { weight, vocab_size, dim })
    }
    pub fn from(weight: Tensor) -> Self {
        let shape = weight.shape();
        let vocab_size = shape[0];
        let dim = shape[1];
        Self { weight, vocab_size, dim }
    }
}

impl Op for Embedding {
    fn name(&self) -> &'static str {
        "Embedding"
    }

    /// 执行 Embedding 的前向计算。
    ///
    /// # Context
    /// * `ctx.inputs[0]`: 输入的 token ID 张量，1D，数据类型必须是 I32。
    /// * `ctx.outputs[0]`: 输出的嵌入向量张量，2D，形状为 [token_len, dim]。
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        // ==================== 1. 检查逻辑 ====================

        if ctx.inputs.len() != 1 || ctx.outputs.len() != 1 {
            return Err(Error::InvalidArgument(
                "Embedding operator expects 1 input and 1 output".into(),
            ).into());
        }

        let input_tokens = &ctx.inputs[0];
        let output = &mut ctx.outputs[0];
        let weight = &self.weight;

        // --- a. 检查设备和数据类型 ---
        if output.device() != weight.device() || weight.device() != input_tokens.device() {
            return Err(Error::InvalidArgument("Output and weight tensors must be on the same device".into()).into());
        }
        // 输入的 token ID 可以在 CPU 上，而权重和输出在 GPU 上
        if input_tokens.dtype() != DataType::I32 {
            return Err(Error::InvalidArgument(format!("Input tokens for Embedding must be of type I32")).into());
        }
        if output.dtype() != weight.dtype() {
            return Err(Error::InvalidArgument(format!("Output and weight must have the same data type, Output:{:?}, weight{:?}",output.dtype(),weight.dtype())).into());
        }

        // --- b. 检查形状 ---
        if input_tokens.shape().len() != 1 {
            return Err(Error::InvalidArgument("Input tokens must be a 1D tensor.".into()).into());
        }
        let token_len = input_tokens.shape()[0];

        if weight.shape() != [self.vocab_size, self.dim] {
            return Err(Error::InvalidArgument(format!(
                "Weight shape is incorrect. Expected [{}, {}], but got {:?}",
                self.vocab_size, self.dim, weight.shape()
            )).into());
        }
        
        if output.shape() != [token_len, self.dim] {
            return Err(Error::InvalidArgument(format!(
                "Output shape is incorrect. Expected [{}, {}], but got {:?}",
                token_len, self.dim, output.shape()
            )).into());
        }

        // ==================== 2. 分派到内核 ====================
        // Embedding 的内核通常需要在与权重相同的设备上执行
        match weight.device() {
            DeviceType::Cpu => {
                // --- CPU 路径 ---
                // 在 CPU 路径内部，我们再根据 dtype 调用不同的内核
                match weight.dtype() {
                    DataType::F32 => {
                        // 调用 f32 版本的 CPU 内核
                        kernels::cpu::embedding(input_tokens, weight, output)?
                    }
                    unsupported_dtype => {
                        return Err(Error::InvalidArgument(format!(
                            "Unsupported weight dtype '{:?}' for CPU embedding.", unsupported_dtype
                        )).into());
                    }
                }
            }
            
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // --- CUDA 路径 ---
                // CUDA 内核通常被设计为可以通过模板或运行时参数处理多种类型，
                // 但如果您的设计是为每种类型提供一个 FFI 函数，这里的逻辑也类似。
                //
                // 假设您的 kernels::cuda::embedding 内部已经处理了 f32/bf16 的分发。
                // 如果没有，这里的结构将和 CPU 路径完全一样。
                kernels::cuda::embedding(input_tokens, weight, output, ctx.cuda_config)?
            }
            
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Embedding {
    /// Move embedding weights to CUDA device
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.weight = self.weight.to_cuda(device_id)?;
        Ok(())
    }
}

// ============================================================================
//  单元测试 (Unit Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::base::DeviceType;
    use crate::base::error::Result;

    /// 核心测试逻辑：在指定设备上验证 Embedding 实现
    ///
    /// # Arguments
    /// * `device` - 将在其上执行计算的设备 (CPU 或 CUDA)。
    /// * `use_stream` - (仅限 CUDA) 是否使用自定义 stream。
    fn run_embedding_test(device: DeviceType, use_stream: bool) -> Result<()> {
        let cpu_device = DeviceType::Cpu;
        let dtype_weights = DataType::F32;
        let dtype_tokens = DataType::I32;

        // --- 1. 定义维度和要查找的 token ---
        let vocab_size = 4;
        let dim = 512;
        // 测试一个包含多个 token 的序列
        let tokens_to_lookup: Vec<i32> = vec![0, 3, 1, 2];
        let token_len = tokens_to_lookup.len();
        
        // --- 2. 在 CPU 上准备黄金数据 ---
        // 准备权重数据
        let weight_data: Vec<f32> = (0..(vocab_size * dim)).map(|i| i as f32).collect();
        // 准备输入 token ID 数据
        let tokens_data = tokens_to_lookup.clone();

        // --- 3. 创建在目标设备 (`device`) 上的算子和张量 ---
        
        // a) 创建 Embedding 算子，其权重在目标设备上
        let mut embedding_op = Embedding::new(vocab_size, dim, dtype_weights, device)?;
        // 将 CPU 上的权重数据拷贝到算子的权重张量中
        embedding_op.weight.as_f32_mut()?.buffer_mut().copy_from_host(&weight_data)?;

        // b) 创建输入 token ID 张量。Token ID 通常在 CPU 上。
        let mut input_tokens = Tensor::new(&[token_len], dtype_tokens, cpu_device)?;
        input_tokens.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&tokens_data);
        match device{
            DeviceType::Cuda(id) => input_tokens = input_tokens.to_cuda(id)?,
            _ => ()
        }
        // c) 创建输出张量，它必须和算子在同一个设备上
        let mut output = Tensor::new(&[token_len, dim], dtype_weights, device)?;
        
        // --- 4. 创建上下文并执行 forward ---
        let cuda_config = if use_stream { Some(crate::cuda::CudaConfig::new()?) } else { None };

        embedding_op.forward(&mut OpContext::new(&[&input_tokens], &mut [&mut output], cuda_config.as_ref()))?;

        // --- 5. 将结果拷贝回 CPU 进行验证 ---
        let result_tensor = output.to_cpu()?;
        let result_slice = result_tensor.as_f32()?.as_slice()?;

        // --- 6. 计算期望结果 ---
        let mut expected_data = Vec::with_capacity(token_len * dim);
        for &token_id in &tokens_to_lookup {
            let start = (token_id as usize) * dim;
            let end = start + dim;
            expected_data.extend_from_slice(&weight_data[start..end]);
        }
        
        assert_eq!(result_slice, expected_data.as_slice());

        Ok(())
    }

    // --- 调用测试辅助函数 ---

    #[test]
    fn test_embedding_cpu() -> Result<()> {
        run_embedding_test(DeviceType::Cpu, false)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_embedding_cuda_no_stream() -> Result<()> {
        run_embedding_test(DeviceType::Cuda(0), false)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_embedding_cuda_with_stream() -> Result<()> {
        run_embedding_test(DeviceType::Cuda(0), true)
    }
}