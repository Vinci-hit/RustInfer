use crate::base::error::{Error, Result};
use crate::base::{DeviceType,DataType};
use crate::op::{kernels, Op, OpContext};

/// SwiGLU 算子，执行 Output = (Input1 * sigmoid(Input1)) ⊙ Input2 (按元素)
/// 这是一个无状态的算子。
#[derive(Debug, Clone, Copy)]
pub struct SwiGLU;

impl SwiGLU {
    /// 创建一个新的 SwiGLU 算子实例。
    pub fn new() -> Self {
        SwiGLU
    }
}

impl Default for SwiGLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for SwiGLU {
    fn name(&self) -> &'static str {
        "SwiGLU"
    }

    /// (原地版本) 执行 SwiGLU 的前向计算。
    ///
    /// 计算 `x = (x * SiLU(x)) * y`
    ///
    /// # Context
    /// * `ctx.inputs[0]` (Y): 只读的第二个输入张量。
    /// * `ctx.outputs[0]` (X): 可读写的、同时作为第一个输入和输出的张量。
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        // ==================== 1. 检查逻辑 (适配原地操作) ====================

        // --- a. 检查输入输出数量 ---
        // **核心修改**: 现在我们期望 1 个输入和 1 个输出
        if ctx.inputs.len() != 1 || ctx.outputs.len() != 1 {
            return Err(Error::InvalidArgument(
                "SwiGLU (in-place) operator expects 1 input (Y) and 1 output (X)".into(),
            ).into());
        }

        // --- b. 重新解释输入和输出 ---
        let input_y = ctx.inputs[0];
        let input_output_x = &mut ctx.outputs[0];

        // --- c. 检查设备和数据类型 ---
        let device = input_y.device();
        let dtype = input_y.dtype();

        // `x` 和 `y` 的设备和类型必须一致
        if input_output_x.device() != device {
            return Err(Error::InvalidArgument("All tensors must be on the same device for SwiGLU.".into()).into());
        }
        if input_output_x.dtype() != dtype {
            return Err(Error::InvalidArgument("All tensors must have the same data type for SwiGLU.".into()).into());
        }
        
        // --- d. 检查形状 ---
        if input_y.shape() != input_output_x.shape() {
            return Err(Error::InvalidArgument(format!(
                "Tensor shapes must be identical for SwiGLU. Got Y: {:?}, X: {:?}",
                input_y.shape(), input_output_x.shape()
            )).into());
        }

        // ==================== 2. 分派到原地内核 ====================
        match device {
            DeviceType::Cpu => {
                kernels::cpu::swiglu(input_y, input_output_x)
            }
            
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // **核心修改**: 调用我们之前实现的原地 CUDA 内核
                kernels::cuda::swiglu(input_y, input_output_x, ctx.cuda_config)
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl SwiGLU {
    /// Stateless operator; nothing to move to CUDA.
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
//  单元测试 (Unit Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*; // 导入父模块 (swiglu.rs) 的所有公共项
    use crate::tensor::Tensor;
    use crate::base::{DeviceType,DataType};
    use crate::base::error::Result;
    use rand::distr::Uniform;
    use rand::prelude::*;

    /// 辅助函数，用于断言两个 float slice 是否足够接近
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "Slices have different lengths");
        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tol,
                "Mismatch at index {}: a = {}, b = {}, diff = {}",
                i, val_a, val_b, (val_a - val_b).abs()
            );
        }
    }

    /// 核心测试逻辑：对比 SwiGLU 的 CPU 和 GPU 实现
    #[cfg(feature = "cuda")]
    fn run_swiglu_cpu_vs_gpu_test(use_stream: bool) -> Result<()> {
        let cpu_device = DeviceType::Cpu;
        let dtype = DataType::F32;
        let size = 32 * 152; // 使用 152，确保可以被 4 整除 (float4 优化)

        // --- 1. 在 CPU 上准备随机输入数据 ---
        let mut rng = rand::rng();
        let dist = Uniform::new(0.0f32, 1.0f32).unwrap();
        
        let mut input_x_cpu = Tensor::new(&[16,size], dtype, cpu_device)?;
        let mut input_x_gpu = Tensor::new(&[16,size], dtype, cpu_device)?;
        input_x_cpu.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.sample(dist));

        let mut input_y_cpu = Tensor::new(&[16,size], dtype, cpu_device)?;
        input_y_cpu.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.sample(dist));
        input_x_gpu.copy_from(&input_x_cpu)?;
        input_x_gpu = input_x_gpu.to_cuda(0)?;
        // --- 2. 在 CPU 上计算黄金结果 ---
        let swiglu_op = SwiGLU::new();
        let mut ctx_cpu = OpContext {
            inputs: &[&input_y_cpu],
            outputs: &mut [&mut input_x_cpu],
            cuda_config:None,
        };
        swiglu_op.forward(&mut ctx_cpu)?;
        let cpu_result_slice = input_x_cpu.as_f32()?.as_slice()?;
        
        // --- 3. 在 GPU 上执行相同计算 ---
        input_x_cpu.to_cuda(0)?;
        let input_y_gpu = input_y_cpu.to_cuda(0)?;

        let cuda_config = if use_stream { Some(crate::cuda::CudaConfig::new()?) } else { None };
        // Option<&T>
        let cuda_config_ref = cuda_config.as_ref();
        
        let mut ctx_gpu = OpContext {
            inputs: &[&input_y_gpu],
            outputs: &mut [&mut input_x_gpu],
            cuda_config: cuda_config_ref,
        };
        swiglu_op.forward(&mut ctx_gpu)?;

        // --- 4. 将 GPU 结果拷回并对比 ---
        // to_cpu() 内部的 cudaMemcpy 是同步的，它会等待 stream 上的计算完成
        let gpu_result_tensor = ctx_gpu.outputs[0].to_cpu()?;
        let gpu_result_slice = gpu_result_tensor.as_f32()?.as_slice()?;

        println!("Running SwiGLU test with stream: {}", use_stream);
        // println!("CPU Result: {:?}", &cpu_result_slice[..5]);
        // println!("GPU Result: {:?}", &gpu_result_slice[..5]);
        assert_close(cpu_result_slice, gpu_result_slice, 1e-5);
        
        Ok(())
    }

    // C++ TEST(test_swiglu_cu, swiglu_nostream)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_swiglu_compare_cpu_vs_gpu_no_stream() -> Result<()> {
        run_swiglu_cpu_vs_gpu_test(false)
    }

    // C++ TEST(test_swiglu_cu, swiglu_stream)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_swiglu_compare_cpu_vs_gpu_with_stream() -> Result<()> {
        run_swiglu_cpu_vs_gpu_test(true)
    }
}