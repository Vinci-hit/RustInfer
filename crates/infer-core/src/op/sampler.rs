// In src/model/sampler.rs

use crate::tensor::Tensor;
use crate::base::error::{Result, Error};
use crate::base::DeviceType;
use crate::op::{kernels, Op, OpContext}; // 引入内核模块和 OpContext
use crate::cuda::CudaConfig; // 引入 CudaConfig

/// Sampler trait 定义了所有采样策略的通用接口。
pub trait Sampler: Send + Sync {
    /// 接收 logits 张量并分发到合适的内核来执行采样。
    fn sample(&self, logits: &Tensor, output_token: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()>;
}

// ------------------- Argmax Sampler (分发器) -------------------

pub struct ArgmaxSampler {
    device_type: DeviceType,
}

impl ArgmaxSampler {
    pub fn new(device_type: DeviceType) -> Self {
        Self { device_type }
    }
}

impl Sampler for ArgmaxSampler {
    /// `sample` 方法现在是一个分发器。
    fn sample(&self, logits: &Tensor,output_token: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
        // ---- 1. 执行前置检查 ----
        if logits.shape().len() != 1 {
            return Err(Error::InvalidArgument(format!(
                "Input logits must be a 1D vector [vocab_size], but got shape {:?}",
                logits.shape()
            )).into());
        }
        if logits.device() != self.device_type {
            return Err(Error::DeviceMismatch {
                expected: self.device_type,
                actual: logits.device(),
                in_method: "ArgmaxSampler::sample".to_string(),
            }.into());
        }
        
        // ---- 2. 根据设备类型分发到不同的内核 ----
        match self.device_type {
            DeviceType::Cpu => {
                // 调用 CPU 内核函数
                kernels::cpu::argmax(logits, output_token)?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // 调用 CUDA 内核函数 (使用 CudaConfig 中的预分配 buffer)
                kernels::cuda::argmax(logits, output_token,cuda_config)?;
            }
        }
        Ok(())
    }
}

/// SamplerOp 算子，作为一个包装器，用于在 Op 调用链中执行采样逻辑。
///
/// 它本身不包含参数，而是持有一个实现了 Sampler trait 的具体采样器实例。
pub struct SamplerOp {
    /// 内部持有的具体采样器，使用 Trait 对象实现动态分发。
    sampler: Box<dyn Sampler>,
}

impl SamplerOp {
    /// 创建一个新的 SamplerOp 算子。
    ///
    /// # Arguments
    /// * `sampler` - 一个已经创建好的、实现了 Sampler trait 的采样器实例。
    pub fn new(sampler: Box<dyn Sampler>) -> Self {
        Self { sampler }
    }
}

impl Op for SamplerOp {
    fn name(&self) -> &'static str {
        "SamplerOp"
    }

    /// 执行 SamplerOp 的前向计算。
    ///
    /// 这个方法会调用内部采样器的 `sample` 方法，并将结果写入输出张量。
    /// - 输入: `logits` 张量，形状为 `[vocab_size]`。
    /// - 输出: 标量张量（形状为 `[1]`），包含采样出的 `token_id`。
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        // ==================== 1. 检查逻辑 ====================

        if ctx.inputs.len() != 1 || ctx.outputs.len() != 1 {
            return Err(Error::InvalidArgument("SamplerOp expects 1 input (logits) and 1 output (token_id)".into()).into());
        }

        let logits = &ctx.inputs[0];
        let output_token_id = &mut ctx.outputs[0];

        if logits.shape().len() != 1 {
            return Err(Error::InvalidArgument(format!(
                "Input logits must be a 1D vector [vocab_size], but got shape {:?}",
                logits.shape()
            )).into());
        }
        if output_token_id.shape() != [1] {
             return Err(Error::InvalidArgument(format!(
                "Output must be a scalar tensor with shape [1], but got shape {:?}",
                output_token_id.shape()
            )).into());
        }
        if output_token_id.dtype() != crate::base::DataType::I32 {
             return Err(Error::InvalidArgument(format!(
                "Output tensor must have dtype I32, but got {:?}",
                output_token_id.dtype()
            )).into());
        }

        // ==================== 2. 分派到具体的 Sampler 实现 ====================
        
        self.sampler.sample(logits, output_token_id, ctx.cuda_config)?;

        Ok(())
    }
}

// 未来可以添加其他采样器，它们也只负责分发
// pub struct TopKSampler { ... }
// impl Sampler for TopKSampler {
//     fn sample(&self, logits: &Tensor) -> Result<i32> {
//         match self.device_type {
//             DeviceType::Cpu => kernels::cpu::top_k(logits, self.k),
//             DeviceType::Cuda(_) => kernels::cuda::top_k(logits, self.k),
//         }
//     }
// }

/// ============================================================================
//  单元测试 (Unit Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*; // 导入父模块 (sampler.rs) 的所有公共项
    use crate::tensor::Tensor;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;
    use half::bf16;

    // ------------------------------------------------------------------------
    // 测试 SamplerOp + ArgmaxSampler 在 CPU 上的行为
    // ------------------------------------------------------------------------
    #[test]
    fn test_sampler_op_with_argmax_cpu() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;
        let vocab_size = 32000;
        let max_index = 12345;

        // --- 1. 准备输入数据和算子 ---
        // 创建 logits 张量
        let mut logits = Tensor::new(&[vocab_size], dtype, device)?;
        // 填充数据，让最大值在我们期望的索引上
        let logits_data: Vec<bf16> = (0..vocab_size)
            .map(|i| if i == max_index { bf16::from_f32(100.0) } else { bf16::from_f32(i as f32 / 1000.0) })
            .collect();
        logits.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&logits_data);

        // 创建 ArgmaxSampler (策略) 和 SamplerOp (上下文/算子)
        let argmax_sampler = Box::new(ArgmaxSampler::new(device));
        let sampler_op = SamplerOp::new(argmax_sampler);

        // --- 2. 准备输出张量并执行 forward ---
        // 输出张量必须在 CPU 上，并且是 i32 类型
        let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        let mut ctx = OpContext {
            inputs: &[&logits],
            outputs: &mut [&mut output],
            cuda_config: None,
        };
        sampler_op.forward(&mut ctx)?;

        // --- 3. 验证结果 ---
        let result_slice = ctx.outputs[0].as_i32()?.as_slice()?;
        assert_eq!(result_slice, &[max_index as i32]);

        Ok(())
    }

    // ========================================================================
    // BF16 Comprehensive Batch Tests
    // ========================================================================

    #[test]
    fn test_sampler_bf16_cpu_batch() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        // Test multiple vocab sizes
        for (vocab_size, max_index) in [(128, 50), (1024, 512), (4096, 2048), (32000, 15000)] {
            // Create logits tensor with known maximum
            let mut logits = Tensor::new(&[vocab_size], dtype, device)?;
            let logits_data: Vec<bf16> = (0..vocab_size)
                .map(|i| {
                    if i == max_index {
                        bf16::from_f32(100.0) // Maximum value
                    } else {
                        bf16::from_f32(((i * 7) % 100) as f32 * 0.1) // Values range 0.0 to 9.9
                    }
                })
                .collect();
            logits.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&logits_data);

            // Create sampler and operator
            let argmax_sampler = Box::new(ArgmaxSampler::new(device));
            let sampler_op = SamplerOp::new(argmax_sampler);

            // Execute sampling
            let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            sampler_op.forward(&mut OpContext::new(&[&logits], &mut [&mut output], None))?;

            // Verify result
            let result = output.as_i32()?.as_slice()?;
            assert_eq!(
                result[0], max_index as i32,
                "Vocab size {}: Expected max_index {}, got {}",
                vocab_size, max_index, result[0]
            );
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sampler_bf16_cuda_batch() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;

        // Test multiple vocab sizes
        for (vocab_size, max_index) in [(256, 100), (2048, 1024), (8192, 4096), (32000, 20000)] {
            // Prepare logits data on CPU
            let logits_data: Vec<bf16> = (0..vocab_size)
                .map(|i| {
                    if i == max_index {
                        bf16::from_f32(100.0)
                    } else {
                        bf16::from_f32(((i * 13) % 100) as f32 * 0.1) // Values range 0.0 to 9.9
                    }
                })
                .collect();

            // Create logits tensor on GPU
            let mut logits = Tensor::new(&[vocab_size], dtype, device)?;
            logits.as_bf16_mut()?.buffer_mut().copy_from_host(&logits_data)?;

            // Create sampler and operator
            let argmax_sampler = Box::new(ArgmaxSampler::new(device));
            let sampler_op = SamplerOp::new(argmax_sampler);

            // Execute sampling
            let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cuda(0))?;
            let cuda_config = crate::cuda::CudaConfig::new()?;
            sampler_op.forward(&mut OpContext::new(&[&logits], &mut [&mut output], Some(&cuda_config)))?;


            assert_eq!(
                output.to_cpu()?.as_i32()?.as_slice()?[0], max_index as i32,
                "CUDA vocab size {}: Expected max_index {}, got {}",
                vocab_size, max_index, output.to_cpu()?.as_i32()?.as_slice()?[0]
            );
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sampler_bf16_cpu_vs_cuda() -> Result<()> {
        let dtype = DataType::BF16;

        // Test CPU vs CUDA for various vocab sizes
        for (vocab_size, max_index) in [(512, 256), (4096, 2048), (16000, 8000)] {
            // Prepare logits data
            let logits_data: Vec<bf16> = (0..vocab_size)
                .map(|i| {
                    if i == max_index {
                        bf16::from_f32(50.0)
                    } else {
                        bf16::from_f32(((i * 7) % 100) as f32 * 0.1)
                    }
                })
                .collect();

            // CPU computation
            let mut logits_cpu = Tensor::new(&[vocab_size], dtype, DeviceType::Cpu)?;
            logits_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&logits_data);

            let argmax_sampler_cpu = Box::new(ArgmaxSampler::new(DeviceType::Cpu));
            let sampler_op_cpu = SamplerOp::new(argmax_sampler_cpu);

            let mut output_cpu = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            sampler_op_cpu.forward(&mut OpContext::new(&[&logits_cpu], &mut [&mut output_cpu], None))?;
            let cpu_result = output_cpu.as_i32()?.as_slice()?[0];

            // GPU computation
            let mut logits_gpu = Tensor::new(&[vocab_size], dtype, DeviceType::Cuda(0))?;
            logits_gpu.as_bf16_mut()?.buffer_mut().copy_from_host(&logits_data)?;

            let argmax_sampler_gpu = Box::new(ArgmaxSampler::new(DeviceType::Cuda(0)));
            let sampler_op_gpu = SamplerOp::new(argmax_sampler_gpu);

            let mut output_gpu = Tensor::new(&[1], DataType::I32, DeviceType::Cuda(0))?;
            let cuda_config = crate::cuda::CudaConfig::new()?;
            sampler_op_gpu.forward(&mut OpContext::new(&[&logits_gpu], &mut [&mut output_gpu], Some(&cuda_config)))?;
            let gpu_result = output_gpu.to_cpu()?.as_i32()?.as_slice()?[0];

            // CPU and GPU results should match
            assert_eq!(
                cpu_result, gpu_result,
                "Vocab size {}: CPU result {} != GPU result {}",
                vocab_size, cpu_result, gpu_result
            );
            assert_eq!(
                cpu_result, max_index as i32,
                "Vocab size {}: Both should find max_index {}",
                vocab_size, max_index
            );
        }

        Ok(())
    }

}