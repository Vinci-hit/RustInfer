use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::op::{Op, OpContext};
use crate::tensor::Tensor;

// 引入我们即将创建的内核模块
use super::kernels;

/// RMSNorm 算子结构体，包含其配置和权重
pub struct RMSNorm {
    pub weight: Tensor,
    dim: usize,
}

impl RMSNorm {
    /// 创建一个新的 RMSNorm 算子
    pub fn new(dim: usize, dtype: DataType, device: DeviceType) -> Result<Self> {
        let weight = Tensor::new(&[dim], dtype, device)?;
        Ok(Self { weight, dim })
    }
    pub fn from(weight: Tensor) -> Self {
        let dim = weight.shape()[0];
        Self { weight, dim }
    }
}

impl Op for RMSNorm {
    fn name(&self) -> &'static str {
        "RMSNorm"
    }

    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
         // --- 1. 检查输入输出数量 (基础契约) ---
        if ctx.inputs.len() != 1 || ctx.outputs.len() != 1 {
            return Err(Error::InvalidArgument(
                "RMSNorm expects 1 input and 1 output".into(),
            ).into());
        }

        let input = &ctx.inputs[0];
        let output = &mut ctx.outputs[0];
        let weight = &self.weight; // 从 self 中获取权重

        // --- 2. 检查设备和数据类型的一致性 ---
        let expected_device = input.device();
        let expected_dtype = input.dtype();

        if output.device() != expected_device || weight.device() != expected_device {
            // 提供更详细的错误信息
            return Err(Error::InvalidArgument(format!(
                "Device mismatch in RMSNorm: input is on {:?}, output on {:?}, weight on {:?}",
                expected_device, output.device(), weight.device()
            )).into());
        }

        if output.dtype() != expected_dtype || weight.dtype() != expected_dtype {
             return Err(Error::InvalidArgument(format!(
                "Data type mismatch in RMSNorm: input is {:?}, output is {:?}, weight is {:?}",
                expected_dtype, output.dtype(), weight.dtype()
            )).into());
        }

        // --- 3. 检查形状 (Shape) 的正确性 ---
        
        // a) 输入和输出的形状必须完全匹配
        if input.shape() != output.shape() {
            return Err(Error::InvalidArgument(format!(
                "Input and output shapes must match in RMSNorm, but got {:?} and {:?}",
                input.shape(), output.shape()
            )).into());
        }

        // b) 权重的形状必须是 [dim]
        if weight.shape() != [self.dim] {
             return Err(Error::InvalidArgument(format!(
                "Weight shape mismatch in RMSNorm: expected [{:?}], but got {:?}",
                self.dim, weight.shape()
            )).into());
        }

        // c) 输入的最后一维必须等于 self.dim
        //    我们使用 `if let` 来安全地获取最后一维
        if let Some(last_dim) = input.shape().last() {
            if *last_dim != self.dim {
                return Err(Error::InvalidArgument(format!(
                    "Input's last dimension ({}) does not match RMSNorm's normalization dim ({})",
                    last_dim, self.dim
                )).into());
            }
        } else {
            // 处理输入是一维张量的情况（如果支持的话），或者零维张量
            if input.shape().len() == 1 && input.shape()[0] != self.dim {
                 return Err(Error::InvalidArgument(format!(
                    "1D Input's dimension ({}) does not match RMSNorm's normalization dim ({})",
                    input.shape()[0], self.dim
                )).into());
            }
            // 如果输入是空 shape `[]`，可能需要特殊处理或报错
            if input.shape().is_empty() {
                 return Err(Error::InvalidArgument("RMSNorm does not support scalar (0-dim) input".into()).into());
            }
        }

        // ==================== 检查逻辑结束 ====================
        // --- 4. 分派到具体的内核实现 ---
        match input.device() {
            DeviceType::Cpu => kernels::cpu::rmsnorm(input, weight, output),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => kernels::cuda::rmsnorm(input, weight, output, ctx.cuda_config),
        }
    }
}

#[cfg(feature = "cuda")]
impl RMSNorm {
    /// Move RMSNorm weight to CUDA device
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.weight = self.weight.to_cuda(device_id)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*; // 导入父模块的 RMSNorm
    use crate::op::{Op, OpContext};
    use crate::tensor::Tensor;
    use crate::base::{DataType, DeviceType};
    use rand::distr::Uniform;
    use rand::prelude::*;

    /// 辅助函数，用于比较两个 f32 slice 是否在误差范围内相等
    fn assert_close(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(a.len(), b.len(), "Slices have different lengths");
        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < epsilon,
                "Mismatch at index {}: a = {}, b = {}",
                i,
                val_a,
                val_b
            );
        }
    }

    /// 辅助函数，用于创建一个 RMSNorm Op 和它的 OpContext
    fn setup(
        shape: &[usize],
        device: DeviceType,
    ) -> Result<(RMSNorm, Tensor, Tensor)> {
        let dim = *shape.last().unwrap();
        let op = RMSNorm::new(dim, DataType::F32, device)?;
        let input = Tensor::new(shape, DataType::F32, device)?;
        let output = Tensor::new(shape, DataType::F32, device)?;
        Ok((op, input, output))
    }

    // ========================================================================
    // TEST 1: test_rmsnorm_cu, rmsnorm_nostream
    // 验证多维输入 (2D) 的情况
    // ========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_rmsnorm_2d_sync() -> Result<()> {
        let rows = 32;
        let dim = 16;
        let shape = &[rows, dim];
        
        // --- 1. CPU (Golden) 计算 ---
        let (mut cpu_op, mut cpu_input, mut cpu_output) = setup(shape, DeviceType::Cpu)?;
        // 用随机数填充 CPU 输入和权重
        let mut rng = rand::rng();
        let dist = Uniform::new(0.0f32, 1.0).unwrap();
        cpu_input.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.sample(dist));
        cpu_op.weight.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.sample(dist));
        
        let mut cpu_ctx = OpContext {
            inputs: &[&cpu_input],
            outputs: &mut [&mut cpu_output],
            cuda_config: None,
        };
        cpu_op.forward(&mut cpu_ctx)?;
        let cpu_result = cpu_ctx.outputs[0].as_f32()?.as_slice()?.to_vec();

        // --- 2. CUDA 计算 ---
        let (mut gpu_op, mut gpu_input, mut gpu_output) = setup(shape, DeviceType::Cuda(0))?;
        // 将相同的输入和权重数据拷贝到 GPU
        gpu_input.as_f32_mut()?.buffer_mut().copy_from_host(cpu_input.as_f32()?.as_slice()?)?;
        gpu_op.weight.as_f32_mut()?.buffer_mut().copy_from_host(cpu_op.weight.as_f32()?.as_slice()?)?;

        let mut gpu_ctx = OpContext {
            inputs: &[&gpu_input],
            outputs: &mut [&mut gpu_output],
            cuda_config: None,
        };
        gpu_op.forward(&mut gpu_ctx)?;
        // 将结果拷贝回 CPU
        let gpu_result_tensor = gpu_ctx.outputs[0].to_cpu()?;
        let gpu_result = gpu_result_tensor.as_f32()?.as_slice()?;

        // --- 3. 对比结果 ---
        assert_close(&cpu_result, gpu_result, 1e-3);
        Ok(())
    }

    // ========================================================================
    // TEST 2: test_rmsnorm_cu, rmsnorm_stream
    // 验证 1D 输入和自定义 stream
    // ========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_rmsnorm_1d_sync() -> Result<()> {
        let dim = 32 * 151 * 15; // 一个大的一维张量
        let shape = &[dim];

        // --- 1. CPU (Golden) 计算 ---
        let (mut cpu_op, mut cpu_input, mut cpu_output) = setup(shape, DeviceType::Cpu)?;
        // (填充随机数据，同上)
        let mut rng = rand::rng();
        let dist = Uniform::new(0.0f32, 1.0).unwrap();
        cpu_input.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.sample(dist));
        cpu_op.weight.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.sample(dist));
        
        let mut cpu_ctx = OpContext {
            inputs: &[&cpu_input],
            outputs: &mut [&mut cpu_output],
            cuda_config: None,
        };
        cpu_op.forward(&mut cpu_ctx)?;
        let cpu_result = cpu_ctx.outputs[0].as_f32()?.as_slice()?.to_vec();
        
        let (mut gpu_op, mut gpu_input, mut gpu_output) = setup(shape, DeviceType::Cuda(0))?;
        // (拷贝数据到 GPU，同上)
        gpu_input.as_f32_mut()?.buffer_mut().copy_from_host(cpu_input.as_f32()?.as_slice()?)?;
        gpu_op.weight.as_f32_mut()?.buffer_mut().copy_from_host(cpu_op.weight.as_f32()?.as_slice()?)?;
        
        let mut gpu_ctx = OpContext {
            inputs: &[&gpu_input],
            outputs: &mut [&mut gpu_output],
            cuda_config: None,
        };
        gpu_op.forward(&mut gpu_ctx)?;
        // 将结果拷贝回 CPU
        let gpu_result_tensor = gpu_ctx.outputs[0].to_cpu()?;
        // to_cpu 内部的 cudaMemcpy 默认是同步的，所以这里已经保证了计算完成
        let gpu_result = gpu_result_tensor.as_f32()?.as_slice()?;

        // --- 3. 对比结果 ---
        assert_close(&cpu_result, gpu_result, 1e-2);
        Ok(())
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
    fn test_rmsnorm_bf16_cpu_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        let dim = 256; // Must be multiple of 16 for alignment

        // Test multiple batch sizes
        for batch in [1, 2, 4, 8] {
            let seq_len = 16;
            let shape = &[batch, seq_len, dim];

            // Create operator and tensors
            let mut rmsnorm_op = RMSNorm::new(dim, dtype, device)?;
            let mut input = Tensor::new(shape, dtype, device)?;
            let mut output = Tensor::new(shape, dtype, device)?;

            // Initialize with test data
            let input_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i % 100) as f32) * 0.01 + 1.0))
                .collect();
            input.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&input_data);

            let weight_data: Vec<bf16> = (0..dim)
                .map(|_| bf16::from_f32(1.0))
                .collect();
            rmsnorm_op.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

            // Compute
            rmsnorm_op.forward(&mut OpContext::new(&[&input], &mut [&mut output], None))?;

            // Verify output is valid (not NaN or Inf)
            let result = output.as_bf16()?.as_slice()?;
            for &val in result {
                assert!(val.to_f32().is_finite(), "Output contains non-finite value");
            }

            // Verify normalization property: mean of squares should be ~1
            for b in 0..batch {
                for s in 0..seq_len {
                    let start = (b * seq_len + s) * dim;
                    let end = start + dim;
                    let slice = &result[start..end];

                    let mean_sq: f32 = slice.iter()
                        .map(|&x| {
                            let val = x.to_f32();
                            val * val
                        })
                        .sum::<f32>() / (dim as f32);

                    // After RMSNorm with weight=1.0, mean of squares should be close to 1
                    assert!(
                        (mean_sq - 1.0).abs() < 0.15,
                        "Batch {}, Seq {}: Mean of squares = {} (expected ~1.0)",
                        b, s, mean_sq
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_rmsnorm_bf16_cuda_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;

        let dim = 512; // Larger dimension for CUDA test

        // Test multiple batch sizes
        for batch in [1, 2, 4, 8, 16] {
            let seq_len = 32;
            let shape = &[batch, seq_len, dim];

            // Create operator and tensors
            let mut rmsnorm_op = RMSNorm::new(dim, dtype, device)?;
            let mut input = Tensor::new(shape, dtype, device)?;
            let mut output = Tensor::new(shape, dtype, device)?;

            // Prepare data on CPU
            let input_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i % 200) as f32) * 0.005 + 0.5))
                .collect();
            let weight_data: Vec<bf16> = (0..dim)
                .map(|_| bf16::from_f32(1.0))
                .collect();

            // Copy to GPU
            input.as_bf16_mut()?.buffer_mut().copy_from_host(&input_data)?;
            rmsnorm_op.weight.as_bf16_mut()?.buffer_mut().copy_from_host(&weight_data)?;

            // Compute with CUDA
            let cuda_config = crate::cuda::CudaConfig::new()?;
            rmsnorm_op.forward(&mut OpContext::new(&[&input], &mut [&mut output], Some(&cuda_config)))?;

            // Copy result back
            let result_tensor = output.to_cpu()?;
            let result = result_tensor.as_bf16()?.as_slice()?;

            // Verify output is valid
            for &val in result {
                assert!(val.to_f32().is_finite(), "Output contains non-finite value");
            }

            // Spot check normalization
            let start = (batch / 2) * seq_len * dim + (seq_len / 2) * dim;
            let end = start + dim;
            let slice = &result[start..end];

            let mean_sq: f32 = slice.iter()
                .map(|&x| {
                    let val = x.to_f32();
                    val * val
                })
                .sum::<f32>() / (dim as f32);

            assert!(
                (mean_sq - 1.0).abs() < 0.15,
                "Batch {}: Mean of squares = {} (expected ~1.0)",
                batch, mean_sq
            );
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_rmsnorm_bf16_cpu_vs_cuda() -> Result<()> {
        use half::bf16;
        let dtype = DataType::BF16;

        let dim = 256;

        // Test CPU vs CUDA for various batch sizes
        for batch in [1, 4, 8] {
            let seq_len = 20;
            let shape = &[batch, seq_len, dim];

            // Prepare input data
            let input_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i * 7) % 100) as f32 * 0.01 + 1.5))
                .collect();
            let weight_data: Vec<bf16> = (0..dim)
                .map(|i| bf16::from_f32(((i * 3) % 50) as f32 * 0.02 + 0.9))
                .collect();

            // CPU computation
            let mut rmsnorm_op_cpu = RMSNorm::new(dim, dtype, DeviceType::Cpu)?;
            let mut input_cpu = Tensor::new(shape, dtype, DeviceType::Cpu)?;
            let mut output_cpu = Tensor::new(shape, dtype, DeviceType::Cpu)?;

            input_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&input_data);
            rmsnorm_op_cpu.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

            rmsnorm_op_cpu.forward(&mut OpContext::new(&[&input_cpu], &mut [&mut output_cpu], None))?;
            let cpu_result = output_cpu.as_bf16()?.as_slice()?.to_vec();

            // GPU computation
            let mut rmsnorm_op_gpu = RMSNorm::new(dim, dtype, DeviceType::Cuda(0))?;
            let mut input_gpu = Tensor::new(shape, dtype, DeviceType::Cuda(0))?;
            let mut output_gpu = Tensor::new(shape, dtype, DeviceType::Cuda(0))?;

            input_gpu.as_bf16_mut()?.buffer_mut().copy_from_host(&input_data)?;
            rmsnorm_op_gpu.weight.as_bf16_mut()?.buffer_mut().copy_from_host(&weight_data)?;

            let cuda_config = crate::cuda::CudaConfig::new()?;
            rmsnorm_op_gpu.forward(&mut OpContext::new(&[&input_gpu], &mut [&mut output_gpu], Some(&cuda_config)))?;

            let gpu_result_tensor = output_gpu.to_cpu()?;
            let gpu_result = gpu_result_tensor.as_bf16()?.as_slice()?;

            // CPU and GPU results should match (with BF16 tolerance)
            assert_bf16_close(&cpu_result, gpu_result, 5e-2);
        }

        Ok(())
    }

    #[test]
    fn test_rmsnorm_bf16_1d_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        // Test 1D input (single vector) with multiple sizes
        for dim in [128, 256, 512] {
            let shape = &[dim];

            let mut rmsnorm_op = RMSNorm::new(dim, dtype, device)?;
            let mut input = Tensor::new(shape, dtype, device)?;
            let mut output = Tensor::new(shape, dtype, device)?;

            // Initialize
            let input_data: Vec<bf16> = (0..dim)
                .map(|i| bf16::from_f32((i as f32) * 0.01))
                .collect();
            let weight_data: Vec<bf16> = (0..dim)
                .map(|_| bf16::from_f32(1.0))
                .collect();

            input.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&input_data);
            rmsnorm_op.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

            // Compute
            rmsnorm_op.forward(&mut OpContext::new(&[&input], &mut [&mut output], None))?;

            // Verify
            let result = output.as_bf16()?.as_slice()?;
            let mean_sq: f32 = result.iter()
                .map(|&x| {
                    let val = x.to_f32();
                    val * val
                })
                .sum::<f32>() / (dim as f32);

            assert!(
                (mean_sq - 1.0).abs() < 0.1,
                "1D RMSNorm dim {}: Mean of squares = {} (expected ~1.0)",
                dim, mean_sq
            );
        }

        Ok(())
    }
}