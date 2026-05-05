use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;
use crate::OpConfig;

use super::kernels;

/// RMSNorm 算子结构体，包含其配置和权重
pub struct RMSNorm {
    pub weight: Tensor,
    #[allow(dead_code)] // 仅用于调试/未来扩展
    dim: usize,
    eps: f32,
}

impl RMSNorm {
    /// 创建一个新的 RMSNorm 算子
    pub fn new(dim: usize, dtype: DataType, device: DeviceType, eps: f32) -> Result<Self> {
        let weight = Tensor::new(&[dim], dtype, device)?;
        Ok(Self { weight, dim, eps })
    }
    pub fn from(weight: Tensor, eps: f32) -> Self {
        let dim = weight.shape()[0];
        Self { weight, dim, eps }
    }

    /// Expose the RMSNorm epsilon for callers that need it (e.g. to
    /// parameterize a fused kernel such as `fused_add_rmsnorm`).
    #[inline]
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl RMSNorm {
    /// 执行 RMSNorm 前向计算: output = input * weight / rms(input)
    pub fn forward(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let weight = &self.weight;

        match input.device() {
            DeviceType::Cpu => { let _ = cuda_config; kernels::cpu::rmsnorm(input, weight, output, self.eps) }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => kernels::cuda::rmsnorm(input, weight, output, self.eps, cuda_config),
        }
    }

    /// In-place RMSNorm: `x = rmsnorm(x, self.weight, self.eps)`.
    ///
    /// Avoids the scratch buffer and the two extra copies that the
    /// out-of-place [`Self::forward`] otherwise forces at call sites
    /// that don't already have a distinct destination slot. The
    /// underlying CUDA kernels are in-place safe (per-element
    /// register-loaded read-before-write, see the kernel source);
    /// the CPU path processes each row chunk exclusively inside a
    /// single rayon task so the in-place update is data-race free.
    pub fn forward_inplace(
        &self,
        x: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let weight = &self.weight;
        match x.device() {
            DeviceType::Cpu => { let _ = cuda_config; kernels::cpu::rmsnorm_inplace(x, weight, self.eps) }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => kernels::cuda::rmsnorm_inplace(x, weight, self.eps, cuda_config),
        }
    }

    /// Fused `residual += input; out = rmsnorm(residual, self.weight, self.eps)`.
    ///
    /// 等价于 [`crate::op::fused_add_rmsnorm::fused_add_rmsnorm`]，但由
    /// `RMSNorm` 自身持有 weight / eps —— 调用方不再需要显式传 weight。
    /// 这就是在模型代码里替代 `fused_add_rmsnorm(out, x, a, &next_norm.weight, eps, cfg)`
    /// 的那一行的优雅形态：`next_norm.forward_with_residual(out, x, a, cfg)`。
    pub fn forward_with_residual(
        &self,
        out: &mut Tensor,
        residual: &mut Tensor,
        input: &Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
            out, residual, input, &self.weight, self.eps, cuda_config,
        )
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
    use super::*;
    
    use crate::tensor::Tensor;
use crate::base::DeviceType;
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

    /// 辅助函数，用于创建一个 RMSNorm Op
    fn setup(
        shape: &[usize],
        device: DeviceType,
    ) -> Result<(RMSNorm, Tensor, Tensor)> {
        let dim = *shape.last().unwrap();
        let op = RMSNorm::new(dim, DataType::F32, device, 1e-6)?;
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
        
        cpu_op.forward(&cpu_input, &mut cpu_output, None)?;
        let cpu_result = cpu_output.as_f32()?.as_slice()?.to_vec();

        // --- 2. CUDA 计算 ---
        let (mut gpu_op, mut gpu_input, mut gpu_output) = setup(shape, DeviceType::Cuda(0))?;
        // 将相同的输入和权重数据拷贝到 GPU
        gpu_input.as_f32_mut()?.buffer_mut().copy_from_host(cpu_input.as_f32()?.as_slice()?)?;
        gpu_op.weight.as_f32_mut()?.buffer_mut().copy_from_host(cpu_op.weight.as_f32()?.as_slice()?)?;

        gpu_op.forward(&gpu_input, &mut gpu_output, None)?;
        // 将结果拷贝回 CPU
        let gpu_result_tensor = gpu_output.to_cpu()?;
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
        
        cpu_op.forward(&cpu_input, &mut cpu_output, None)?;
        let cpu_result = cpu_output.as_f32()?.as_slice()?.to_vec();
        
        let (mut gpu_op, mut gpu_input, mut gpu_output) = setup(shape, DeviceType::Cuda(0))?;
        // (拷贝数据到 GPU，同上)
        gpu_input.as_f32_mut()?.buffer_mut().copy_from_host(cpu_input.as_f32()?.as_slice()?)?;
        gpu_op.weight.as_f32_mut()?.buffer_mut().copy_from_host(cpu_op.weight.as_f32()?.as_slice()?)?;
        
        gpu_op.forward(&gpu_input, &mut gpu_output, None)?;
        // 将结果拷贝回 CPU
        let gpu_result_tensor = gpu_output.to_cpu()?;
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
            let mut rmsnorm_op = RMSNorm::new(dim, dtype, device, 1e-6)?;
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
            rmsnorm_op.forward(&input, &mut output, None)?;

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
            let mut rmsnorm_op = RMSNorm::new(dim, dtype, device, 1e-6)?;
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
            rmsnorm_op.forward(&input, &mut output, Some(&cuda_config))?;

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
            let mut rmsnorm_op_cpu = RMSNorm::new(dim, dtype, DeviceType::Cpu, 1e-6)?;
            let mut input_cpu = Tensor::new(shape, dtype, DeviceType::Cpu)?;
            let mut output_cpu = Tensor::new(shape, dtype, DeviceType::Cpu)?;

            input_cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&input_data);
            rmsnorm_op_cpu.weight.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&weight_data);

            rmsnorm_op_cpu.forward(&input_cpu, &mut output_cpu, None)?;
            let cpu_result = output_cpu.as_bf16()?.as_slice()?.to_vec();

            // GPU computation
            let mut rmsnorm_op_gpu = RMSNorm::new(dim, dtype, DeviceType::Cuda(0), 1e-6)?;
            let mut input_gpu = Tensor::new(shape, dtype, DeviceType::Cuda(0))?;
            let mut output_gpu = Tensor::new(shape, dtype, DeviceType::Cuda(0))?;

            input_gpu.as_bf16_mut()?.buffer_mut().copy_from_host(&input_data)?;
            rmsnorm_op_gpu.weight.as_bf16_mut()?.buffer_mut().copy_from_host(&weight_data)?;

            let cuda_config = crate::cuda::CudaConfig::new()?;
            rmsnorm_op_gpu.forward(&input_gpu, &mut output_gpu, Some(&cuda_config))?;

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

            let mut rmsnorm_op = RMSNorm::new(dim, dtype, device, 1e-6)?;
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
            rmsnorm_op.forward(&input, &mut output, None)?;

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

    // ========================================================================
    // strided view 回归：从一个 dense [rows, q_dim+2*kv_dim] tensor narrow 出
    // [rows, q_dim] 的列视图（行 stride > q_dim），验证：
    //   - CUDA 路径直接接受 strided view（不再要求 contiguous）
    //   - in-place norm 结果与"先 contiguous 再 dense norm"完全一致
    // ========================================================================

    /// 通用的 strided view 检查：构造 fused QKV 形态的 dense base，narrow 出 Q
    /// 列做 in-place norm；然后 base 再 narrow 一份相同区域、先 `contiguous()`
    /// 物化、走 dense norm 作为 reference。两条路径应当输出完全相同。
    fn check_strided_inplace_matches_dense(device: DeviceType) -> Result<()> {
        use half::bf16;
        let dtype = DataType::BF16;
        let rows = 16usize;
        let head_dim = 64usize;            // RMSNorm dim 必须 % 8 = 0
        let q_heads = 4usize;
        let kv_heads = 1usize;
        let q_dim = q_heads * head_dim;       // 256
        let kv_dim = kv_heads * head_dim;     // 64
        let cols = q_dim + 2 * kv_dim;        // 384

        // base: [rows, cols] dense bf16，填一些非零数据
        let mut base = Tensor::new(&[rows, cols], dtype, device)?;
        let raw: Vec<bf16> = (0..(rows * cols))
            .map(|i| bf16::from_f32(((i * 13) % 97) as f32 * 0.01 - 0.5))
            .collect();
        // 先在 CPU 上构造数据再上传，避免 cuda 端 raw 写入复杂
        if device.is_cuda() {
            let mut cpu = Tensor::new(&[rows, cols], dtype, DeviceType::Cpu)?;
            cpu.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&raw);
            base = cpu.to_cuda(0)?;
        } else {
            base.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&raw);
        }

        // QK-norm weight: [head_dim]
        let w_data: Vec<bf16> = (0..head_dim)
            .map(|i| bf16::from_f32(0.5 + (i as f32) * 0.001))
            .collect();
        let q_norm = {
            let mut w = Tensor::new(&[head_dim], dtype, DeviceType::Cpu)?;
            w.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&w_data);
            let w = if device.is_cuda() { w.to_cuda(0)? } else { w };
            RMSNorm::from(w, 1e-6)
        };

        // ── reference：先拷出 Q 列、reshape 成 [rows*head_num, head_dim] dense，
        //    走原来的 dense 路径做 norm ──
        let ref_q_dense = base.narrow(1, 0, q_dim)?.contiguous()?;
        // dense reshape 是零拷贝 view 改 strides
        let mut ref_q_2d = ref_q_dense
            .reshape(&[rows * q_heads, head_dim])?
            .to_owned()?;
        q_norm.forward_inplace(&mut ref_q_2d, None)?;
        let ref_cpu = ref_q_2d.to_cpu()?;
        let ref_slice: Vec<f32> = ref_cpu.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();

        // ── strided in-place：base2 的 Q 列 unflatten 为 3-D
        //    [T, head_num, head_dim]（strides=[cols, head_dim, 1]）→ 直接喂 RMSNorm ──
        let base2 = base.contiguous()?.to_owned()?;
        let mut q_view_3d = base2
            .narrow(1, 0, q_dim)?
            .unflatten(1, &[q_heads, head_dim])?;
        // sanity: 3-D strided view，stride0 = cols, stride1 = head_dim
        assert_eq!(q_view_3d.shape(), &[rows, q_heads, head_dim]);
        assert_eq!(q_view_3d.strides()[0], cols);
        assert_eq!(q_view_3d.strides()[1], head_dim);
        assert_eq!(q_view_3d.strides()[2], 1);
        assert!(!q_view_3d.is_contiguous());

        q_norm.forward_inplace(&mut q_view_3d, None)?;

        // 取出 base2 的 Q 列（每行前 q_dim 个 elem）跟 reference 比对
        let strided_cpu = base2.to_cpu()?;
        let strided_slice = strided_cpu.as_bf16()?.as_slice()?;
        let mut got: Vec<f32> = Vec::with_capacity(rows * q_dim);
        for r in 0..rows {
            let start = r * cols;
            for c in 0..q_dim {
                got.push(strided_slice[start + c].to_f32());
            }
        }
        assert_close(&got, &ref_slice, 5e-3);

        // K/V 区（base2 的 [q_dim..]）必须未被触碰
        let raw_f32: Vec<f32> = raw.iter().map(|v| v.to_f32()).collect();
        for r in 0..rows {
            let row_start = r * cols;
            for c in q_dim..cols {
                let got_v = strided_slice[row_start + c].to_f32();
                let want_v = raw_f32[row_start + c];
                assert!(
                    (got_v - want_v).abs() < 1e-6,
                    "K/V region row {} col {} was overwritten: got {} want {}",
                    r, c, got_v, want_v
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_rmsnorm_strided_inplace_cpu() -> Result<()> {
        check_strided_inplace_matches_dense(DeviceType::Cpu)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_rmsnorm_strided_inplace_cuda() -> Result<()> {
        check_strided_inplace_matches_dense(DeviceType::Cuda(0))
    }
}
