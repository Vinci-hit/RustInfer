// In src/model/sampler.rs

use crate::tensor::Tensor;
use crate::base::error::{Result, Error};
use crate::base::DeviceType;
use crate::op::{kernels, Op, OpContext}; // 引入内核模块

/// Sampler trait 定义了所有采样策略的通用接口。
pub trait Sampler: Send + Sync {
    /// 接收 logits 张量并分发到合适的内核来执行采样。
    fn sample(&self, logits: &Tensor) -> Result<i32>;
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
    fn sample(&self, logits: &Tensor) -> Result<i32> {
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
                kernels::cpu::argmax(logits)
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // 调用 CUDA 内核函数
                kernels::cuda::argmax(logits, None)
            }
        }
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
        if output_token_id.device() != crate::base::DeviceType::Cpu {
             return Err(Error::InvalidArgument(format!(
                "Output tensor for token_id must be on the CPU, but got {:?}",
                output_token_id.device()
            )).into());
        }

        // ==================== 2. 分派到具体的 Sampler 实现 ====================
        
        let sampled_id = self.sampler.sample(logits)?;

        // ==================== 3. 将结果写入输出张量 ====================
        
        let output_slice = output_token_id.as_i32_mut()?.as_slice_mut()?;
        output_slice[0] = sampled_id;

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

    // ------------------------------------------------------------------------
    // 测试 SamplerOp + ArgmaxSampler 在 CUDA 上的行为
    // ------------------------------------------------------------------------
    #[test]
    #[cfg(feature = "cuda")]
    fn test_sampler_op_with_argmax_cuda() -> Result<()> {
        use crate::cuda::CudaConfig;

        let cpu_device = DeviceType::Cpu;
        let gpu_device = DeviceType::Cuda(0);
        let dtype = DataType::F32;
        let vocab_size = 50000;
        let max_index = 43210;

        // --- 1. 在 CPU 上准备数据 ---
        let mut logits_cpu = Tensor::new(&[vocab_size], dtype, cpu_device)?;
        let mut logits_data_cpu = vec![0.0f32; vocab_size];
        // 设置一个独特的最大值
        logits_data_cpu[max_index] = 999.9;
        logits_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&logits_data_cpu);
        
        // --- 2. 将输入移动到 GPU ---
        let logits_gpu = logits_cpu.to_cuda(0)?;

        // --- 3. 准备算子，注意 Sampler 需要知道它在 GPU 上工作 ---
        let argmax_sampler = Box::new(ArgmaxSampler::new(gpu_device));
        let sampler_op = SamplerOp::new(argmax_sampler);

        // --- 4. 准备输出张量 (必须在 CPU) 并执行 forward ---
        let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        let cuda_config = CudaConfig::new()?;
        let mut ctx = OpContext {
            inputs: &[&logits_gpu],
            outputs: &mut [&mut output],
            cuda_config: Some(&cuda_config),
        };
        sampler_op.forward(&mut ctx)?;

        // --- 5. 验证结果 ---
        // 因为输出张量已经在 CPU 上，并且我们的内核是隐式同步的，
        // 所以 forward 返回后，结果就已经可用了。
        let result_slice = ctx.outputs[0].as_i32()?.as_slice()?;
        assert_eq!(result_slice, &[max_index as i32]);
        
        Ok(())
    }
}