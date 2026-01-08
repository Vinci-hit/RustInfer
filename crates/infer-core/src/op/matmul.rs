use crate::base::error::{Result,Error} ;
use crate::base::{DataType, DeviceType};
use crate::op::{kernels, Op, OpContext};
use crate::tensor::Tensor;

/// Matmul 算子，执行 Y = W@X^T + B
///
/// 其中：
/// - X 是输入张量
/// - W 是权重矩阵
/// - B 是可选的偏置向量
pub struct Matmul {
    /// 权重矩阵 W，形状通常为 [out_features, in_features]
    pub weight: Tensor,
    
    /// 可选的偏置向量 B，形状通常为 [out_features]
    pub bias: Option<Tensor>,
}

impl Matmul {
    /// 创建一个新的 Matmul 算子。
    ///
    /// # Arguments
    /// * `in_features` - 输入特征维度。
    /// * `out_features` - 输出特征维度。
    /// * `has_bias` - 是否创建偏置项。
    /// * `dtype` - 参数（权重和偏置）的数据类型。
    /// * `device` - 参数所在的设备。
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        dtype: DataType,
        device: DeviceType,
    ) -> Result<Self> {
        // 创建权重张量 W，形状为 [out_features, in_features]
        let weight = Tensor::new(&[out_features, in_features], dtype, device)?;
        
        // 如果需要，创建偏置张量 B
        let bias = if has_bias {
            let b = Tensor::new(&[out_features], dtype, device)?;
            Some(b)
        } else {
            None
        };

        Ok(Self { weight, bias })
    }
    pub fn from(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Op for Matmul {
    fn name(&self) -> &'static str {
        "Matmul"
    }

    /// 执行 Matmul 的前向计算： Y = W@X + B
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        // ==================== 1. 检查逻辑 (对应 C++ 的 check()) ====================

        // --- a. 检查输入输出数量 ---
        if ctx.inputs.len() != 1 || ctx.outputs.len() != 1 {
            return Err(Error::InvalidArgument("Matmul expects 1 input and 1 output".into()).into());
        }

        let input = &ctx.inputs[0];
        let output = &mut ctx.outputs[0];
        let weight = &self.weight;
        // --- b. 检查设备和数据类型 ---
        let device = input.device();
        let dtype = input.dtype();

        if output.device() != device || weight.device() != device {
            return Err(Error::InvalidArgument("All tensors must be on the same device".into()).into());
        }
        if output.dtype() != dtype || weight.dtype() != dtype {
            return Err(Error::InvalidArgument("All tensors must have the same data type".into()).into());
        }
        if let Some(bias) = &self.bias
            && (bias.device() != device || bias.dtype() != dtype)
        {
            return Err(Error::InvalidArgument("Bias tensor has mismatched device or dtype".into()).into());
        }
        
        let weight_shape = weight.shape();

        // 权重 W 必须是 2D 矩阵 [K, M]
        if weight_shape.len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "Weight must be a 2D matrix [out_features, in_features], but got shape {:?}",
                weight_shape
            )).into());
        }
        
        // ==================== 2. 分派到内核 (对应 C++ 的 forward() 主体) ====================
        match device {
            DeviceType::Cpu => {
                // 调用 CPU 内核函数Y = W@X^T + B
                kernels::cpu::matmul(input,weight,output)?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                if input.dtype() == DataType::BF16{
                    kernels::cuda::hgemm_bf16(input, weight, output, ctx.cuda_config)?;
                }else{
                    if input.shape()[0] == 1{
                        kernels::cuda::sgemv(input, weight, output, ctx.cuda_config)?;
                    }else{
                        kernels::cuda::sgemm(input, weight, output, ctx.cuda_config)?;
                    }
                }
            }
        }

        // --- b. 如果有偏置，执行加法 ---
        if let Some(bias) = &self.bias {
            match device {
                DeviceType::Cpu => {
                    // 调用 CPU 加法内核 (原地 add)
                    kernels::cpu::add_inplace(output, bias)?;
                }
                #[cfg(feature = "cuda")]
                DeviceType::Cuda(_) => {
                    // 调用 CUDA 加法内核 (原地 add)
                    kernels::cuda::add_inplace(output,bias,None)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Matmul {
    /// Move Matmul internal parameters to CUDA device
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        // move weight
        self.weight = self.weight.to_cuda(device_id)?;
        // move bias if exists
        if let Some(b) = &self.bias {
            self.bias = Some(b.to_cuda(device_id)?);
        }
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
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;

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
    
    // ... (保留其他测试用例) ...

    // ------------------------------------------------------------------------
    // 测试多维矩阵乘法 (CPU vs GPU)
    // 输入: [B, M]
    // 权重: [M, N]
    // 输出: [B, N]
    // ------------------------------------------------------------------------
    #[test]
    #[cfg(feature = "cuda")]
    fn test_matmul_multidim_compare_cpu_vs_gpu() -> Result<()> {
        use crate::cuda::CudaConfig;
        
        let cpu_device = DeviceType::Cpu;
        let gpu_device = DeviceType::Cuda(0);
        let dtype = DataType::F32;

        // 1. 定义多维矩阵的维度
        const B:usize = 16; // Batch size
        const M:usize = 64; // Input feature size
        const N:usize = 64; // Output feature size

        // --- 2. 准备 CPU 数据和计算 ---
        println!("Preparing CPU data...");
        
        // 输入张量: [B, M]
        let input_shape = &[B, M];
        let mut input_cpu = Tensor::new(input_shape, dtype, cpu_device)?;
        let input_data: Vec<f32> = (0..(B*M)).map(|i| i as f32).collect(); // 简单的线性数据
        input_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&input_data);
        println!("CPU Input:\n{:?}", input_data.chunks_exact(M).collect::<Vec<_>>());

        // 权重张量: [M, N]
        let mut matmul_op_cpu = Matmul::new(M, N, false, dtype, cpu_device)?; // weight is [M, N], no transpose
        let weight_data: Vec<f32> = (0..(M*N)).map(|i| i as f32).collect();
        matmul_op_cpu.weight.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);
        println!("CPU Weight:\n{:?}", weight_data.chunks_exact(N).collect::<Vec<_>>());
        
        // 在 CPU 上计算黄金结果
        let output_shape = &[B, N];
        let mut output_cpu = Tensor::new(output_shape, dtype, cpu_device)?;
        matmul_op_cpu.forward(&mut OpContext::new(&[&input_cpu], &mut [&mut output_cpu], None))?;
        let cpu_result_slice = output_cpu.as_f32()?.as_slice()?;
        println!("CPU Result:\n{:?}", cpu_result_slice.chunks_exact(N).collect::<Vec<_>>());

        // --- 3. 准备 GPU 数据和计算 ---
        println!("\nPreparing GPU data...");
        
        // 将输入和权重拷贝到 GPU
        let input_gpu = input_cpu.to_cuda(0)?;
        
        let mut matmul_op_gpu = Matmul::new(M, N, false, dtype, gpu_device)?;
        matmul_op_gpu.weight = matmul_op_cpu.weight.to_cuda(0)?;

        // 在 GPU 上分配输出张量
        let mut output_gpu = Tensor::new(output_shape, dtype, gpu_device)?;

        // 执行 GPU 计算
        let cuda_config = CudaConfig::new()?;
        matmul_op_gpu.forward(&mut OpContext::new(&[&input_gpu], &mut [&mut output_gpu], Some(&cuda_config)))?;
        
        // 等待 GPU 计算完成
        unsafe { crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?; }
        println!("GPU computation finished.");

        // --- 4. 结果对比 ---
        println!("\nComparing results...");
        
        // 将 GPU 结果拷贝回 CPU
        let gpu_result_tensor = output_gpu.to_cpu()?;
        let gpu_result_slice = gpu_result_tensor.as_f32()?.as_slice()?;
        println!("GPU Result:\n{:?}", gpu_result_slice.chunks_exact(N).collect::<Vec<_>>());

        // 断言两个结果是否接近
        assert_close(cpu_result_slice, gpu_result_slice, 1e-5);
        
        println!("\nTest passed! CPU and GPU results match within tolerance.");
        Ok(())
    }
}