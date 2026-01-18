use crate::base::error::Result;
use crate::base::DeviceType;
use crate::op::{kernels, Op, OpContext};

/// AddInplace 算子，执行原地加法 A = A + B。
/// 第一个输入张量会被原地修改，第二个输入张量保持不变。
/// 这是一个无状态的算子，因此它是一个零大小的结构体。
#[derive(Debug, Clone, Copy)]
pub struct AddInplace;

impl AddInplace {
    /// 创建一个新的 AddInplace 算子实例。
    pub fn new() -> Self {
        AddInplace
    }
}

impl Default for AddInplace {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for AddInplace {
    fn name(&self) -> &'static str {
        "AddInplace"
    }

    /// 执行 AddInplace 的前向计算。
    ///
    /// # Context
    /// * `ctx.inputs[0]` (B): 要加到输出张量上的输入张量。
    /// * `ctx.outputs[0]` (A): 输入输出张量，会被原地修改为 A = A + B。
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        let input_b = ctx.inputs[0];
        let output_a = &mut ctx.outputs[0];

        // --- 检查 ---
        if input_b.device() != output_a.device() {
            return Err(anyhow::anyhow!("Device mismatch for AddInplace"));
        }
        if input_b.dtype() != output_a.dtype() {
            return Err(anyhow::anyhow!("DType mismatch for AddInplace"));
        }
        if input_b.shape() != output_a.shape() {
            return Err(anyhow::anyhow!("Shape mismatch for AddInplace"));
        }

        // --- 分派到内核 ---
        match input_b.device() {
            DeviceType::Cpu => {
                // 调用 CPU 原地加法内核
                kernels::cpu::add_inplace(output_a, input_b)?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // 调用 CUDA 原地加法内核
                kernels::cuda::add_inplace(output_a, input_b, ctx.cuda_config)?;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl AddInplace {
    /// AddInplace is stateless; nothing to move to CUDA.
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
//  单元测试 (Unit Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;
    use crate::tensor::Tensor;

    /// 辅助函数，用于断言两个 float slice 是否足够接近
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "Slices have different lengths");
        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tol,
                "Mismatch at index {}: a = {}, b = {}",
                i, val_a, val_b
            );
        }
    }

    // ------------------------------------------------------------------------
    // CPU 测试
    // ------------------------------------------------------------------------
    #[test]
    fn test_add_inplace_cpu_basic() -> Result<()> {
        let device = DeviceType::Cpu;
        let size = 100;

        // 创建输入张量
        let host_a = vec![2.0f32; size];
        let host_b = vec![3.0f32; size];

        let mut t_a = Tensor::new(&[size], DataType::F32, device)?;
        t_a.as_f32_mut()?.buffer_mut().copy_from_host(&host_a)?;

        let mut t_b = Tensor::new(&[size], DataType::F32, device)?;
        t_b.as_f32_mut()?.buffer_mut().copy_from_host(&host_b)?;

        // 执行原地加法
        let add_inplace_op = AddInplace::new();
        add_inplace_op.forward(&mut OpContext::new(&[&t_b], &mut [&mut t_a], None))?;

        // 验证结果
        let result_slice = t_a.as_f32()?.as_slice()?;
        let expected_result = vec![5.0; size];
        assert_close(result_slice, &expected_result, 1e-6);

        Ok(())
    }

    // ------------------------------------------------------------------------
    // CUDA 测试
    // ------------------------------------------------------------------------
    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_inplace_cuda_basic() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let size = 32 * 151;

        // 1. 在 CPU 上准备输入数据
        let host_a = vec![2.0f32; size];
        let host_b = vec![3.0f32; size];

        // 2. 创建 GPU 张量并拷贝数据
        let mut t_a = Tensor::new(&[size], DataType::F32, device)?;
        t_a.as_f32_mut()?.buffer_mut().copy_from_host(&host_a)?;

        let mut t_b = Tensor::new(&[size], DataType::F32, device)?;
        t_b.as_f32_mut()?.buffer_mut().copy_from_host(&host_b)?;

        // 3. 执行原地加法
        let add_inplace_op = AddInplace::new();
        add_inplace_op.forward(&mut OpContext::new(&[&t_b], &mut [&mut t_a], None))?;

        // 4. 将结果拷贝回 CPU 进行验证
        let result_tensor = t_a.to_cpu()?;
        let result_slice = result_tensor.as_f32()?.as_slice()?;

        // 5. 断言
        let expected_result = vec![5.0; size];
        assert_close(result_slice, &expected_result, 1e-6);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_inplace_cuda_with_stream() -> Result<()> {
        use crate::cuda::CudaConfig;

        let device = DeviceType::Cuda(0);
        let size = 32 * 151;

        // 1. 准备数据和张量
        let host_a = vec![2.0f32; size];
        let host_b = vec![3.0f32; size];

        let mut t_a = Tensor::new(&[size], DataType::F32, device)?;
        t_a.as_f32_mut()?.buffer_mut().copy_from_host(&host_a)?;

        let mut t_b = Tensor::new(&[size], DataType::F32, device)?;
        t_b.as_f32_mut()?.buffer_mut().copy_from_host(&host_b)?;

        // 2. 创建一个自定义的 CudaConfig (包含一个新的 stream)
        let cuda_config = CudaConfig::new()?;

        // 3. 执行原地加法，传入 cuda_config
        let add_inplace_op = AddInplace::new();
        add_inplace_op.forward(&mut OpContext::new(
            &[&t_b],
            &mut [&mut t_a],
            Some(&cuda_config),
        ))?;

        // 4. 显式同步设备以确保计算完成
        unsafe {
            crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?;
        }

        // 5. 将结果拷贝回 CPU 并验证
        let result_tensor = t_a.to_cpu()?;
        let result_slice = result_tensor.as_f32()?.as_slice()?;

        let expected_result = vec![5.0; size];
        assert_close(result_slice, &expected_result, 1e-6);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_inplace_cuda_bf16() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let size = 32 * 151 * 8; // 确保对齐

        // 1. 准备 BF16 数据
        let host_a = vec![half::bf16::from_f32(2.0); size];
        let host_b = vec![half::bf16::from_f32(3.0); size];

        // 2. 创建 GPU 张量
        let mut t_a = Tensor::new(&[size], DataType::BF16, device)?;
        t_a.as_bf16_mut()?.buffer_mut().copy_from_host(&host_a)?;

        let mut t_b = Tensor::new(&[size], DataType::BF16, device)?;
        t_b.as_bf16_mut()?.buffer_mut().copy_from_host(&host_b)?;

        // 3. 执行原地加法
        let add_inplace_op = AddInplace::new();
        add_inplace_op.forward(&mut OpContext::new(&[&t_b], &mut [&mut t_a], None))?;

        // 4. 将结果拷贝回 CPU 并转换为 f32 进行验证
        let result_tensor = t_a.to_cpu()?;
        let result_slice_bf16 = result_tensor.as_bf16()?.as_slice()?;
        let result_slice_f32: Vec<f32> = result_slice_bf16
            .iter()
            .map(|&x| x.to_f32())
            .collect();

        // 5. 断言 (BF16 精度较低)
        let expected_result = vec![5.0; size];
        assert_close(&result_slice_f32, &expected_result, 0.01);

        Ok(())
    }
}