use crate::base::error::{Error, Result};
use crate::base::DeviceType;
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
            #[cfg(feature = "cuda")]
            attn_backend: None,
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
            #[cfg(feature = "cuda")]
            attn_backend: None,
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
    fn test_swiglu_bf16_cpu_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        // Test multiple batch sizes
        for batch in [1, 2, 4, 8] {
            let seq_len = 16;
            let dim = 256;
            let shape = &[batch, seq_len, dim];

            // Create tensors (x and y)
            let mut input_x = Tensor::new(shape, dtype, device)?;
            let mut input_y = Tensor::new(shape, dtype, device)?;

            // Initialize with test data
            let x_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i % 100) as f32) * 0.01))
                .collect();
            let y_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i % 50) as f32) * 0.02))
                .collect();

            input_x.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&x_data);
            input_y.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&y_data);

            // Compute: x = silu(x) * x * y
            let swiglu_op = SwiGLU::new();
            swiglu_op.forward(&mut OpContext::new(&[&input_y], &mut [&mut input_x], None))?;

            // Verify result is finite
            let result = input_x.as_bf16()?.as_slice()?;
            for &val in result {
                assert!(val.to_f32().is_finite(), "Output contains non-finite value");
            }

            // Verify basic properties (output magnitude should be reasonable)
            let mean_abs: f32 = result.iter()
                .map(|&x| x.to_f32().abs())
                .sum::<f32>() / result.len() as f32;
            assert!(mean_abs < 10.0, "Output values too large: mean_abs = {}", mean_abs);
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_swiglu_bf16_cuda_batch() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;

        // Test multiple batch sizes
        for batch in [1, 2, 4, 8, 16] {
            let seq_len = 32;
            let dim = 512;
            let shape = &[batch, seq_len, dim];

            // Create tensors on GPU
            let mut input_x = Tensor::new(shape, dtype, device)?;
            let mut input_y = Tensor::new(shape, dtype, device)?;

            // Prepare data
            let x_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i % 200) as f32) * 0.005 + 0.5))
                .collect();
            let y_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i % 150) as f32) * 0.01))
                .collect();

            // Copy to GPU
            input_x.as_bf16_mut()?.buffer_mut().copy_from_host(&x_data)?;
            input_y.as_bf16_mut()?.buffer_mut().copy_from_host(&y_data)?;

            // Compute
            let swiglu_op = SwiGLU::new();
            let cuda_config = crate::cuda::CudaConfig::new()?;
            swiglu_op.forward(&mut OpContext::new(&[&input_y], &mut [&mut input_x], Some(&cuda_config)))?;

            // Copy result back
            let result_tensor = input_x.to_cpu()?;
            let result = result_tensor.as_bf16()?.as_slice()?;

            // Verify result is finite
            for &val in result {
                assert!(val.to_f32().is_finite(), "Output contains non-finite value");
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_swiglu_bf16_cpu_vs_cuda() -> Result<()> {
        use half::bf16;
        let dtype = DataType::BF16;

        // Test CPU vs CUDA for various batch sizes
        for batch in [1, 4, 8] {
            let seq_len = 20;
            let dim = 256;
            let shape = &[batch, seq_len, dim];

            // Prepare data
            let x_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i * 7) % 100) as f32 * 0.01))
                .collect();
            let y_data: Vec<bf16> = (0..(batch * seq_len * dim))
                .map(|i| bf16::from_f32(((i * 11) % 80) as f32 * 0.01))
                .collect();

            // CPU computation
            let mut input_x_cpu = Tensor::new(shape, dtype, DeviceType::Cpu)?;
            let mut input_y_cpu = Tensor::new(shape, dtype, DeviceType::Cpu)?;

            input_x_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&x_data);
            input_y_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&y_data);

            let swiglu_op = SwiGLU::new();
            swiglu_op.forward(&mut OpContext::new(&[&input_y_cpu], &mut [&mut input_x_cpu], None))?;
            let cpu_result = input_x_cpu.as_bf16()?.as_slice()?.to_vec();

            // GPU computation
            let mut input_x_gpu = Tensor::new(shape, dtype, DeviceType::Cuda(0))?;
            let mut input_y_gpu = Tensor::new(shape, dtype, DeviceType::Cuda(0))?;

            input_x_gpu.as_bf16_mut()?.buffer_mut().copy_from_host(&x_data)?;
            input_y_gpu.as_bf16_mut()?.buffer_mut().copy_from_host(&y_data)?;

            let cuda_config = crate::cuda::CudaConfig::new()?;
            swiglu_op.forward(&mut OpContext::new(&[&input_y_gpu], &mut [&mut input_x_gpu], Some(&cuda_config)))?;

            let gpu_result_tensor = input_x_gpu.to_cpu()?;
            let gpu_result = gpu_result_tensor.as_bf16()?.as_slice()?;

            // CPU and GPU results should match
            assert_bf16_close(&cpu_result, gpu_result, 1e-2);
        }

        Ok(())
    }

    #[test]
    fn test_swiglu_bf16_numerical_correctness() -> Result<()> {
        use half::bf16;
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;

        // Test with known values to verify correctness
        let size = 4;
        let shape = &[1, size];

        let mut input_x = Tensor::new(shape, dtype, device)?;
        let mut input_y = Tensor::new(shape, dtype, device)?;

        // Use simple values for verification
        let x_data = vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(-1.0), bf16::from_f32(0.5)];
        let y_data = vec![bf16::from_f32(2.0), bf16::from_f32(1.0), bf16::from_f32(3.0), bf16::from_f32(4.0)];

        input_x.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&x_data);
        input_y.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&y_data);

        // Compute SwiGLU
        let swiglu_op = SwiGLU::new();
        swiglu_op.forward(&mut OpContext::new(&[&input_y], &mut [&mut input_x], None))?;

        let result = input_x.as_bf16()?.as_slice()?;

        // Verify results are reasonable
        // SwiGLU(x, y) = silu(x) * y where silu(x) = x / (1 + exp(-x))
        // Note: The operation is inplace, so we compute silu(x) * y
        for (i, &val) in result.iter().enumerate() {
            let x = x_data[i].to_f32();
            let y = y_data[i].to_f32();
            let silu_x = x / (1.0 + (-x).exp());
            let expected = silu_x * y;
            let actual = val.to_f32();

            // Allow generous tolerance for BF16
            let rel_error = if expected.abs() > 1e-6 {
                (expected - actual).abs() / expected.abs()
            } else {
                (expected - actual).abs()
            };

            assert!(
                (expected - actual).abs() < 0.1 || rel_error < 0.15,
                "Index {}: expected ~{}, got {}, rel_error={}", i, expected, actual, rel_error
            );
        }

        Ok(())
    }
}