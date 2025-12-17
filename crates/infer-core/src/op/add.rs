use crate::base::error::Result;
use crate::base::DeviceType; // 引入 DeviceType
use crate::op::{kernels, Op, OpContext};

/// Add 算子，执行 C = A + B (按元素相加)。
/// 这是一个无状态的算子，因此它是一个零大小的结构体。
#[derive(Debug, Clone, Copy)]
pub struct Add;

impl Add {
    /// 创建一个新的 Add 算子实例。
    pub fn new() -> Self {
        Add
    }
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for Add {
    fn name(&self) -> &'static str {
        "Add"
    }

    /// 执行 Add 的前向计算。
    ///
    /// # Context
    /// * `ctx.inputs[0]` (A): 第一个输入张量。
    /// * `ctx.inputs[1]` (B): 第二个输入张量。
    /// * `ctx.outputs[0]` (C): 输出张量。
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
    let output = &mut ctx.outputs[0];

    // --- 核心修改：根据输入数量选择不同的逻辑路径 ---
    if ctx.inputs.len() == 2 {
        // ==================== 路径 A: 非原地加法 (output = a + b) ====================
        let input_a = ctx.inputs[0];
        let input_b = ctx.inputs[1];

        // --- a. 检查 ---
        if input_a.device() != input_b.device() || input_a.device() != output.device() {
            return Err(anyhow::anyhow!("Device mismatch for non-inplace Add"));
        }
        if input_a.dtype() != input_b.dtype() || input_a.dtype() != output.dtype() {
            return Err(anyhow::anyhow!("DType mismatch for non-inplace Add"));
        }
        if input_a.shape() != input_b.shape() || input_a.shape() != output.shape() {
            return Err(anyhow::anyhow!("Shape mismatch for non-inplace Add"));
        }

        // --- b. 分派到内核 ---
        // (这里的内核可以是 bf16 适配版本)
        match input_a.device() {
            DeviceType::Cpu => {
                // 调用期望 `output` 是独立缓冲区的内核
                kernels::cpu::add(input_a, input_b, output)?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::add(input_a, input_b, output, ctx.cuda_config)?;
            }
        }
        
    } else if ctx.inputs.len() == 1 {
        // ==================== 路径 B: 原地加法 (output = output + a) ====================
        let input_a = ctx.inputs[0];

        // --- a. 检查 ---
        // 此时，output 同时扮演了“输入B”和“输出”的角色
        if input_a.device() != output.device() {
            return Err(anyhow::anyhow!("Device mismatch for inplace Add"));
        }
        if input_a.dtype() != output.dtype() {
            return Err(anyhow::anyhow!("DType mismatch for inplace Add"));
        }
        if input_a.shape() != output.shape() {
            return Err(anyhow::anyhow!("Shape mismatch for inplace Add"));
        }

        // --- b. 分派到内核 ---
        // **关键**: 我们将 `output` 同时作为第二个输入和输出传递给内核！
        match input_a.device() {
            DeviceType::Cpu => {
                kernels::cpu::add_inplace(output, input_a)?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::add_inplace( output, input_a, ctx.cuda_config)?;
            }
        }

    } else {
        // --- 错误路径 ---
        return Err(anyhow::anyhow!(
            "Add operator expects 1 (inplace) or 2 (out-of-place) inputs, but got {}",
            ctx.inputs.len()
        ));
    }

    Ok(())
}
}

#[cfg(feature = "cuda")]
impl Add {
    /// Add is stateless; nothing to move to CUDA.
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
//  单元测试 (Unit Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*; // 导入父模块 (add.rs) 的所有公共项
    use crate::tensor::Tensor;
    use crate::base::{DeviceType,DataType};
    use crate::base::error::Result;
    

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
    // C++ TEST(test_add_cu, add1_nostream) and TEST(test_add_cu, add_align1)
    // ------------------------------------------------------------------------
    // 这两个测试逻辑几乎一样，只是数据和 size 不同，我们可以用一个函数来合并
    fn run_add_test(size: usize, val1: f32, val2: f32, expected_sum: f32, tol: f32) -> Result<()> {
        let device = DeviceType::Cuda(0);

        // 1. 在 CPU 上准备输入数据
        let host_a = vec![val1; size];
        let host_b = vec![val2; size];

        // 2. 创建 GPU 张量并拷贝数据
        let mut t1 = Tensor::new(&[size], DataType::F32, device)?;
        t1.as_f32_mut()?.buffer_mut().copy_from_host(&host_a)?;

        let mut t2 = Tensor::new(&[size], DataType::F32, device)?;
        t2.as_f32_mut()?.buffer_mut().copy_from_host(&host_b)?;
        
        let mut output = Tensor::new(&[size], DataType::F32, device)?;
        
        // 3. 创建算子和上下文，并执行 forward
        let add_op = Add::new();
        add_op.forward(&mut OpContext::new(&[&t1, &t2], &mut [&mut output], None))?;

        // 4. 将结果拷贝回 CPU 进行验证
        let result_tensor = output.to_cpu()?;
        let result_slice = result_tensor.as_f32()?.as_slice()?;

        // 5. 断言
        let expected_result = vec![expected_sum; size];
        assert_close(result_slice, &expected_result, tol);
        
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_cuda_basic() -> Result<()> {
        run_add_test(32 * 151, 2.0, 3.0, 5.0, 1e-6)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_cuda_large_and_float() -> Result<()> {
        run_add_test(32 * 151 * 13, 2.1, 3.3, 5.4, 0.001)
    }

    // ------------------------------------------------------------------------
    // C++ TEST(test_add_cu, add1_stream)
    // ------------------------------------------------------------------------
    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_cuda_with_stream() -> Result<()> {
        use crate::cuda::CudaConfig;
        
        let device = DeviceType::Cuda(0);
        let size = 32 * 151;

        // 1. 准备数据和张量 (同上)
        let host_a = vec![2.0f32; size];
        let host_b = vec![3.0f32; size];
        let mut t1 = Tensor::new(&[size], DataType::F32, device)?;
        t1.as_f32_mut()?.buffer_mut().copy_from_host(&host_a)?;
        let mut t2 = Tensor::new(&[size], DataType::F32, device)?;
        t2.as_f32_mut()?.buffer_mut().copy_from_host(&host_b)?;
        let mut output = Tensor::new(&[size], DataType::F32, device)?;

        // 2. 创建一个自定义的 CudaConfig (包含一个新的 stream)
        let cuda_config = CudaConfig::new()?;
        
        // 3. 创建算子和上下文，这次传入 cuda_config
        let add_op = Add::new();
        add_op.forward(&mut OpContext::new(&[&t1, &t2], &mut [&mut output], Some(&cuda_config)))?;

        // 4. 显式同步设备以确保计算完成
        //    在实际应用中，你可能会同步 stream，但同步整个设备对于测试来说更简单可靠
        unsafe { crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?; }
        
        // 5. 将结果拷贝回 CPU 并验证
        let result_tensor = output.to_cpu()?;
        let result_slice = result_tensor.as_f32()?.as_slice()?;

        let expected_result = vec![5.0; size];
        assert_close(result_slice, &expected_result, 1e-6);

        // `cuda_config` 离开作用域时，其 Drop impl 会自动调用 cudaStreamDestroy
        Ok(())
    }
}