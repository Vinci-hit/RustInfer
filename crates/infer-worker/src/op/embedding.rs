use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::config::CudaConfig;

use super::kernels;

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

impl Embedding {
    /// 执行 Embedding 前向计算: 根据 token ID 从权重表中查找嵌入向量
    pub fn forward(
        &self,
        input_tokens: &Tensor,
        output: &mut Tensor,
        #[cfg(feature = "cuda")] cuda_config: Option<&CudaConfig>,
    ) -> Result<()> {
        let weight = &self.weight;

        match weight.device() {
            DeviceType::Cpu => {
                kernels::cpu::embedding(input_tokens, weight, output)?
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::embedding(input_tokens, weight, output, cuda_config)?
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
        if let DeviceType::Cuda(id) = device {
            input_tokens = input_tokens.to_cuda(id)?;
        }
        // c) 创建输出张量，它必须和算子在同一个设备上
        let mut output = Tensor::new(&[token_len, dim], dtype_weights, device)?;
        
        // --- 4. 创建上下文并执行 forward ---
        let cuda_config = if use_stream { Some(crate::cuda::CudaConfig::new()?) } else { None };

        embedding_op.forward(&input_tokens, &mut output, cuda_config.as_ref())?;

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

    // ========================================================================
    // BF16 Comprehensive Batch Tests
    // ========================================================================

    /// Helper to assert BF16 results are close
    fn assert_bf16_close(a: &[half::bf16], b: &[half::bf16], tol: f32) {
        assert_eq!(a.len(), b.len(), "BF16 slices have different lengths");
        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (val_a.to_f32() - val_b.to_f32()).abs();
            assert!(
                diff < tol,
                "BF16 mismatch at index {}: a = {}, b = {}, diff = {}",
                i, val_a.to_f32(), val_b.to_f32(), diff
            );
        }
    }

    #[test]
    fn test_embedding_bf16_cpu_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        let vocab_size = 128;
        let dim = 256;

        // Test multiple batch sizes (number of tokens)
        for token_len in [1, 4, 8, 16] {
            // Create embedding operator
            let mut embedding_op = Embedding::new(vocab_size, dim, dtype, device)?;

            // Initialize weight with known pattern
            let weight_data: Vec<bf16> = (0..(vocab_size * dim))
                .map(|i| bf16::from_f32((i as f32) * 0.001))
                .collect();
            embedding_op.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

            // Create input tokens (various IDs)
            let mut input_tokens = Tensor::new(&[token_len], DataType::I32, device)?;
            let token_ids: Vec<i32> = (0..token_len)
                .map(|i| ((i * 7) % vocab_size) as i32) // Varied token IDs
                .collect();
            input_tokens.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&token_ids);

            // Create output tensor
            let mut output = Tensor::new(&[token_len, dim], dtype, device)?;

            // Compute embedding
            embedding_op.forward(&input_tokens, &mut output, None)?;

            // Verify results
            let result = output.as_bf16()?.as_slice()?;
            for (i, &token_id) in token_ids.iter().enumerate() {
                let start = (token_id as usize) * dim;
                let end = start + dim;
                let expected = &weight_data[start..end];
                let actual = &result[(i * dim)..((i + 1) * dim)];
                assert_bf16_close(actual, expected, 1e-3);
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_embedding_bf16_cuda_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;

        let vocab_size = 256;
        let dim = 512;

        // Test multiple batch sizes
        for token_len in [1, 8, 16, 32] {
            // Create embedding operator
            let mut embedding_op = Embedding::new(vocab_size, dim, dtype, device)?;

            // Initialize weight
            let weight_data: Vec<bf16> = (0..(vocab_size * dim))
                .map(|i| bf16::from_f32(((i % 1000) as f32) * 0.01))
                .collect();
            embedding_op.weight.as_bf16_mut()?.buffer_mut().copy_from_host(&weight_data)?;

            // Create input tokens on CPU first
            let mut input_tokens = Tensor::new(&[token_len], DataType::I32, DeviceType::Cpu)?;
            let token_ids: Vec<i32> = (0..token_len)
                .map(|i| ((i * 11) % vocab_size) as i32)
                .collect();
            input_tokens.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&token_ids);

            // Move to GPU
            let input_tokens_gpu = input_tokens.to_cuda(0)?;

            // Create output tensor on GPU
            let mut output = Tensor::new(&[token_len, dim], dtype, device)?;

            // Compute embedding with CUDA
            let cuda_config = crate::cuda::CudaConfig::new()?;
            embedding_op.forward(&input_tokens_gpu, &mut output, Some(&cuda_config))?;

            // Copy result back and verify
            let result_tensor = output.to_cpu()?;
            let result = result_tensor.as_bf16()?.as_slice()?;

            for (i, &token_id) in token_ids.iter().enumerate() {
                let start = (token_id as usize) * dim;
                let end = start + dim;
                let expected = &weight_data[start..end];
                let actual = &result[(i * dim)..((i + 1) * dim)];
                assert_bf16_close(actual, expected, 1e-2);
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_embedding_bf16_cpu_vs_cuda() -> Result<()> {
        use half::bf16;
        let dtype = DataType::BF16;

        let vocab_size = 200;
        let dim = 384;

        // Test CPU vs CUDA for various token lengths
        for token_len in [1, 4, 16] {
            // Prepare weight data
            let weight_data: Vec<bf16> = (0..(vocab_size * dim))
                .map(|i| bf16::from_f32(((i * 3) % 500) as f32 * 0.01))
                .collect();

            // CPU computation
            let mut embedding_op_cpu = Embedding::new(vocab_size, dim, dtype, DeviceType::Cpu)?;
            embedding_op_cpu.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

            let mut input_tokens_cpu = Tensor::new(&[token_len], DataType::I32, DeviceType::Cpu)?;
            let token_ids: Vec<i32> = (0..token_len)
                .map(|i| ((i * 13) % vocab_size) as i32)
                .collect();
            input_tokens_cpu.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&token_ids);

            let mut output_cpu = Tensor::new(&[token_len, dim], dtype, DeviceType::Cpu)?;
            embedding_op_cpu.forward(&input_tokens_cpu, &mut output_cpu, None)?;
            let cpu_result = output_cpu.as_bf16()?.as_slice()?.to_vec();

            // GPU computation
            let mut embedding_op_gpu = Embedding::new(vocab_size, dim, dtype, DeviceType::Cuda(0))?;
            embedding_op_gpu.weight.as_bf16_mut()?.buffer_mut().copy_from_host(&weight_data)?;

            let input_tokens_gpu = input_tokens_cpu.to_cuda(0)?;
            let mut output_gpu = Tensor::new(&[token_len, dim], dtype, DeviceType::Cuda(0))?;

            let cuda_config = crate::cuda::CudaConfig::new()?;
            embedding_op_gpu.forward(&input_tokens_gpu, &mut output_gpu, Some(&cuda_config))?;

            let gpu_result_tensor = output_gpu.to_cpu()?;
            let gpu_result = gpu_result_tensor.as_bf16()?.as_slice()?;

            // CPU and GPU results should match
            assert_bf16_close(&cpu_result, gpu_result, 1e-2);
        }

        Ok(())
    }

    #[test]
    fn test_embedding_bf16_large_vocab() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        // Test with realistic large vocabulary (like LLaMA)
        let vocab_size = 32000;
        let dim = 512;
        let token_len = 8;

        let mut embedding_op = Embedding::new(vocab_size, dim, dtype, device)?;

        // Initialize weight (use modulo to avoid huge data)
        let weight_data: Vec<bf16> = (0..(vocab_size * dim))
            .map(|i| bf16::from_f32(((i % 10000) as f32) * 0.0001))
            .collect();
        embedding_op.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

        // Create tokens spanning the vocabulary
        let mut input_tokens = Tensor::new(&[token_len], DataType::I32, device)?;
        let token_ids: Vec<i32> = vec![0, 100, 1000, 10000, 20000, 30000, 31999, 15000];
        input_tokens.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&token_ids);

        let mut output = Tensor::new(&[token_len, dim], dtype, device)?;

        embedding_op.forward(&input_tokens, &mut output, None)?;

        // Verify some specific tokens
        let result = output.as_bf16()?.as_slice()?;
        for (i, &token_id) in token_ids.iter().enumerate() {
            let start = (token_id as usize) * dim;
            let expected_first = weight_data[start];
            let actual_first = result[i * dim];
            assert!(
                (expected_first.to_f32() - actual_first.to_f32()).abs() < 1e-3,
                "Mismatch for token {} at position {}", token_id, i
            );
        }

        Ok(())
    }
}