// src/op/rope.rs

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::op::{kernels, Op, OpContext};
use crate::tensor::Tensor;

/// Rotary Positional Embedding (RoPE) 算子。
/// 
/// 这是一个**无参数**的就地 (in-place) 算子，用于旋转 Query (Q) 和 Key (K) 张量。
/// 它依赖于外部提供的 sin/cos 缓存。
pub struct RoPEOp {
    /// 旋转维度 D
    pub dim: usize,
    /// Key/Value 旋转维度 K (通常 K <= D)
    pub kv_dim: usize,
    /// Attention Head 的大小
    pub head_size: usize,
}

impl RoPEOp {
    /// 创建一个新的 RoPEOp 算子。
    ///
    /// # Arguments
    /// * `dim` - Q 和 K 向量的总旋转维度。
    /// * `kv_dim` - K 向量旋转的维度。
    /// * `head_size` - Attention Head 的大小。
    pub fn new(dim: usize, kv_dim: usize, head_size: usize) -> Result<Self> {
        if kv_dim > dim {
             return Err(Error::InvalidArgument(format!(
                "RoPEOp: kv_dim ({}) cannot be greater than dim ({}).", kv_dim, dim
            )).into());
        }
        Ok(Self { dim, kv_dim, head_size })
    }
}

impl Op for RoPEOp {
    fn name(&self) -> &'static str {
        "RoPEOp"
    }

    /// 执行 RoPE 的前向计算：就地旋转 Q 和 K。
    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        // ==================== 1. 检查逻辑 ====================

        // 预期输入: [input_pos, sin_cache, cos_cache] (3个)
        // 预期输出: [input_q, input_k] (2个，将被就地修改)
        if ctx.inputs.len() != 3 || ctx.outputs.len() != 2 {
            return Err(Error::InvalidArgument(
                "RoPEOp expects 3 inputs (Pos, Sin, Cos) and 2 outputs (Q, K for in-place modification)".into()
            ).into());
        }

        // 获取不可变的 Pos, Sin, Cos
        let input_pos = &ctx.inputs[0]; 
        let sin_cache = &ctx.inputs[1];
        let cos_cache = &ctx.inputs[2];
        
        // --- 核心修复: 使用 split_at_mut 安全地获取两个不重叠的可变引用 ---
        // 将 outputs 切片拆分为前两个元素 ([Q, K]) 和其余部分 (空)
        let q_k_tensor_slice = ctx.outputs
            .get_mut(0..2)
            .ok_or_else(|| Error::InvalidArgument("Outputs must contain Q and K tensors (length >= 2)".to_string()))?;
        
        // 使用 split_at_mut(1) 将 [Q, K] 切片分成 [Q] 和 [K]
        let (q_slice, k_slice) = q_k_tensor_slice.split_at_mut(1);
        
        // 安全地获取 Q 和 K 的可变引用
        let input_q: &mut Tensor = q_slice[0];
        let input_k: &mut Tensor = k_slice[0]; 
        let seq_len  = input_q.shape()[0];
        // --- c. 检查设备和数据类型 (使用 input_q/k 而非 inputs[0]/1) ---
        let device = input_q.device();
        let dtype = input_q.dtype();
        
        // 所有张量必须在同一个设备上
        if input_k.device() != device || sin_cache.device() != device || cos_cache.device() != device {
            return Err(Error::InvalidArgument(format!("RoPE: All tensors must be on the same device,input_q:{:?}, input_k:{:?}, sin_cache:{:?}, cos_cache:{:?}",device,input_k.device(),sin_cache.device(),cos_cache.device())).into());
        }
        
        if input_k.dtype() != dtype || sin_cache.dtype() != dtype || cos_cache.dtype() != dtype {
                return Err(Error::InvalidArgument(format!("RoPE: All tensors must have the same data type. input_q:{:?}, input_k:{:?}, sin_cache:{:?}, cos_cache:{:?}", dtype, input_k.dtype(), sin_cache.dtype(), cos_cache.dtype())).into());
        }

        // Pos 必须是 i32
        if input_pos.dtype() != DataType::I32 {
            return Err(Error::InvalidArgument("Pos tensor must be I32 type".into()).into());
        }

        // --- c. 检查形状 ---
        // Q 和 K 向量长度必须至少为 self.dim 和 self.kv_dim
        if input_q.shape()[1] < self.dim {
            return Err(Error::InvalidArgument(format!(
                "Input Q length ({}) must be at least RoPE dim ({})", input_q.shape()[0], self.dim
            )).into());
        }
        if input_k.shape()[1] < self.kv_dim {
            return Err(Error::InvalidArgument(format!(
                "Input K length ({}) must be at least RoPE kv_dim ({})", input_k.shape()[0], self.kv_dim
            )).into());
        }
        
        // Pos 必须是 [1] 形状或长度为 1
        if input_pos.shape()[0] != 1 {
            return Err(Error::InvalidArgument("Input Pos must be a single element tensor".into()).into());
        }


        // ==================== 2. 分派到内核 (对应 C++ 的 forward() 主体) ====================
        // RoPE 是就地操作，需要可变访问 Q 和 K 张量。
        let input_q_mut: &mut Tensor = unsafe { 
            // 假设我们有某种方式从上下文获取可变的 Q 和 K
            let q_ptr = input_q as *const Tensor as *mut Tensor;
            &mut *q_ptr
        };
        let input_k_mut: &mut Tensor = unsafe { 
            let k_ptr = input_k as *const Tensor as *mut Tensor;
            &mut *k_ptr
        };
        
        match device {
            DeviceType::Cpu => {
                // 调用 CPU 内核函数 (F32版本)
                kernels::cpu::rope_kernel_batch(
                    self.kv_dim,
                    self.head_size,
                    input_q_mut, // 可变
                    input_k_mut, // 可变
                    input_pos,
                    sin_cache,
                    cos_cache,
                )?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::rope(
                    self.dim,
                    self.kv_dim,
                    self.head_size,
                    input_q_mut, // 可变
                    input_k_mut, // 可变
                    input_pos,
                    seq_len as i32,
                    sin_cache,
                    cos_cache,
                    ctx.cuda_config
                )?;
            }
            // _ => {
            //     return Err(Error::InvalidArgument(format!(
            //         "Unsupported device/data type combination: {:?}/{:?}", device, dtype
            //     )).into());
            // }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl RoPEOp {
    /// RoPEOp is stateless w.r.t. owned tensors; nothing to move.
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*; // 导入 RoPEOp
    use crate::tensor::Tensor;
    use crate::base::{DataType, DeviceType};
    use crate::op::{Op, OpContext};
    use crate::base::error::Result;
    
    // 引入 rand 相关的 trait
    use rand::Rng;
    
    // ------------------------------------------------------------------------
    // 辅助函数: 断言两个 float slice 是否足够接近
    // ------------------------------------------------------------------------
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
    
    // ------------------------------------------------------------------------
    // TEST 2: test_rope_bf16_equivalence
    // 验证 BF16 和 F32 计算结果的等价性
    // ------------------------------------------------------------------------
    #[test]
    fn test_rope_bf16_equivalence() -> Result<()> {
        let dim = 256;
        let head_size = 64;
        let kv_dim = 128;
        let pos = 3;
        let seq_len = 4;
        
        // --- 1. 准备 F32 数据 ---
        let dtype_f32 = DataType::F32;
        let pos_dtype = DataType::I32;
        let max_seq_len = pos as usize + 512;
        
        let mut rng = rand::rng();
        
        // F32 输入张量
        let mut input_q_f32 = Tensor::new(&[seq_len, dim], dtype_f32, DeviceType::Cpu)?;
        input_q_f32.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.random_range(0.0f32..1.0f32));
        let mut input_k_f32 = Tensor::new(&[seq_len, kv_dim], dtype_f32, DeviceType::Cpu)?;
        input_k_f32.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.random_range(0.0f32..1.0f32));
        
        // Pos 张量
        let mut input_pos = Tensor::new(&[1], pos_dtype, DeviceType::Cpu)?;
        input_pos.as_i32_mut()?.as_slice_mut()?[0] = pos;
        
        // Sin/Cos 缓存 (F32)
        let mut sin_cache_f32 = Tensor::new(&[max_seq_len, head_size], dtype_f32, DeviceType::Cpu)?;
        let mut cos_cache_f32 = Tensor::new(&[max_seq_len, head_size], dtype_f32, DeviceType::Cpu)?;
        kernels::cpu::rope_sin_cos_cache_calc(head_size, max_seq_len, &mut sin_cache_f32, &mut cos_cache_f32)?;
        
        // --- 2. 准备 BF16 数据 (从 F32 转换) ---
        let dtype_bf16 = DataType::BF16;
        
        // BF16 输入张量 (从 F32 转换)
        let mut input_q_bf16 = Tensor::new(&[seq_len, dim], dtype_bf16, DeviceType::Cpu)?;
        let mut input_k_bf16 = Tensor::new(&[seq_len, kv_dim], dtype_bf16, DeviceType::Cpu)?;
        
        // 将 F32 数据转换为 BF16
        for i in 0..(seq_len * dim) {
            let val = input_q_f32.as_f32()?.as_slice()?[i];
            input_q_bf16.as_bf16_mut()?.as_slice_mut()?[i] = half::bf16::from_f32(val);
        }
        for i in 0..(seq_len * kv_dim) {
            let val = input_k_f32.as_f32()?.as_slice()?[i];
            input_k_bf16.as_bf16_mut()?.as_slice_mut()?[i] = half::bf16::from_f32(val);
        }
        
        // Sin/Cos 缓存 (BF16 版本使用 F32 缓存，因为计算是在 F32 中进行的)
        let mut sin_cache_bf16 = Tensor::new(&[max_seq_len, head_size], dtype_f32, DeviceType::Cpu)?;
        let mut cos_cache_bf16 = Tensor::new(&[max_seq_len, head_size], dtype_f32, DeviceType::Cpu)?;
        kernels::cpu::rope_sin_cos_cache_calc(head_size, max_seq_len, &mut sin_cache_bf16, &mut cos_cache_bf16)?;
        
        // --- 3. F32 计算 ---
        let op_f32 = RoPEOp::new(dim, kv_dim, head_size)?;
        let mut ctx_f32 = OpContext {
            inputs: &[&input_pos, &sin_cache_f32, &cos_cache_f32],
            outputs: &mut [&mut input_q_f32, &mut input_k_f32],
            cuda_config: None,
        };
        op_f32.forward(&mut ctx_f32)?;
        let q_result_f32 = ctx_f32.outputs[0].as_f32()?.as_slice()?.to_vec();
        let k_result_f32 = ctx_f32.outputs[1].as_f32()?.as_slice()?.to_vec();
        
        // --- 4. BF16 计算 ---
        let op_bf16 = RoPEOp::new(dim, kv_dim, head_size)?;
        let mut ctx_bf16 = OpContext {
            inputs: &[&input_pos, &sin_cache_bf16, &cos_cache_bf16],
            outputs: &mut [&mut input_q_bf16, &mut input_k_bf16],
            cuda_config: None,
        };
        op_bf16.forward(&mut ctx_bf16)?;
        
        // 将 BF16 结果转换为 F32 用于比较
        let q_result_bf16: Vec<f32> = ctx_bf16.outputs[0].as_bf16()?.as_slice()?.iter().map(|&x| x.to_f32()).collect();
        let k_result_bf16: Vec<f32> = ctx_bf16.outputs[1].as_bf16()?.as_slice()?.iter().map(|&x| x.to_f32()).collect();
        
        // --- 5. 对比结果 (容忍 BF16 精度损失) ---
        assert_close(&q_result_f32, &q_result_bf16, 1e-2); // BF16 精度约为 3-4 位小数
        assert_close(&k_result_f32, &k_result_bf16, 1e-2);
        
        Ok(())
    }
    
    // ------------------------------------------------------------------------
    // 辅助函数: 设置所有输入张量
    // ------------------------------------------------------------------------
    fn setup_tensors(
        dim: usize, 
        head_size: usize, 
        pos: i32, 
        device: DeviceType
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dtype = DataType::F32;
        let pos_dtype = DataType::I32;
        let max_seq_len = pos as usize + 512; // 确保 max_seq_len 足够大
        
        // --- 1. Q, K 张量 (用随机数填充) ---
        let mut rng = rand::rng();
        let mut input_q = Tensor::new(&[4,dim], dtype, DeviceType::Cpu)?; // as slice方法暂时只支持在cpu上
        input_q.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.random_range(0.0f32..1.0f32));
        let mut input_k = Tensor::new(&[4,dim/2], dtype, DeviceType::Cpu)?;
        input_k.as_f32_mut()?.as_slice_mut()?.iter_mut().for_each(|x| *x = rng.random_range(0.0f32..1.0f32));

        // --- 2. Pos 张量 (设置 pos 值) ---
        let mut input_pos = Tensor::new(&[1], pos_dtype, DeviceType::Cpu)?;
        input_pos.as_i32_mut()?.as_slice_mut()?[0] = pos;


        let mut sin_cache = Tensor::new(&[max_seq_len, head_size], dtype, device)?;
        let mut cos_cache = Tensor::new(&[max_seq_len, head_size], dtype, device)?;
        
        // 使用高效的 CPU 实现来计算缓存 (确保 CPU 结果是正确的)
        match device{
            DeviceType::Cpu => kernels::cpu::rope_sin_cos_cache_calc(head_size, max_seq_len, &mut sin_cache, &mut cos_cache)?,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(device_id) => {
                input_q = input_q.to_cuda(device_id)?;
                input_k = input_k.to_cuda(device_id)?;
                kernels::cuda::rope_sin_cos_cache_calc_cuda(head_size, max_seq_len, &mut sin_cache, &mut cos_cache,None)?
            }
        }

        Ok((input_q, input_k, input_pos, sin_cache, cos_cache))
    }

    // ------------------------------------------------------------------------
    // TEST 1: test_rope_cu, rope_nostream (Case 1 对应)
    // 验证 CPU 和 CUDA 的结果一致性 (无 stream)
    // ------------------------------------------------------------------------
    #[test]
    fn test_rope_nostream_case1() -> Result<()> {
        let dim = 256;
        let head_size = 64;
        let kv_dim = 128;
        let pos = 3;
        
        // --- 1. CPU (Golden) 计算 ---
        // Q 和 K 是被就地修改的，所以需要克隆一份作为输入
        let (mut q_cpu_in, mut k_cpu_in, pos_cpu, sin_cpu, cos_cpu) = setup_tensors(dim, head_size, pos, DeviceType::Cpu)?;
        let (mut q_gpu_result, mut k_gpu_result, pos_gpu, sin_gpu, cos_gpu) = setup_tensors(dim, head_size, pos, DeviceType::Cuda(0))?;
        assert_close(sin_cpu.as_f32()?.as_slice()?,sin_gpu.to_cpu()?.as_f32()?.as_slice()?,1e-3);
        assert_close(cos_cpu.as_f32()?.as_slice()?,cos_gpu.to_cpu()?.as_f32()?.as_slice()?,1e-3);
        q_gpu_result.as_f32_mut()?.buffer_mut().copy_from_host(q_cpu_in.as_f32()?.as_slice()?)?; // Q 的 GPU 结果，将被修改
        k_gpu_result.as_f32_mut()?.buffer_mut().copy_from_host(k_cpu_in.as_f32()?.as_slice()?)?;
        // RoPE Op 算子
        let op_cpu = RoPEOp::new(dim, kv_dim, head_size)?;
        
        // RoPE 算子的 Context (输入 Pos, Sin, Cos; 输出 Q, K)
        let mut cpu_ctx = OpContext {
            inputs: &[&pos_cpu, &sin_cpu, &cos_cpu],
            outputs: &mut [&mut q_cpu_in, &mut k_cpu_in],
            cuda_config: None,
        };
        op_cpu.forward(&mut cpu_ctx)?;
        // 从 context 获取最终结果（被修改后的 Q 和 K）
        let q_gold_result = cpu_ctx.outputs[0].as_f32()?.as_slice()?.to_vec();
        let k_gold_result = cpu_ctx.outputs[1].as_f32()?.as_slice()?.to_vec();
        // --- 2. CUDA 计算 ---
        // 使用相同的输入数据在 GPU 上执行
        let op_gpu = RoPEOp::new(dim, kv_dim, head_size)?;
        
        let mut gpu_ctx = OpContext {
            inputs: &[&pos_gpu, &sin_gpu, &cos_gpu],
            outputs: &mut [&mut q_gpu_result, &mut k_gpu_result],
            cuda_config: None, // 使用默认 stream (nullptr)
        };
        op_gpu.forward(&mut gpu_ctx)?;
        // 将结果拷贝回 CPU
        let q_gpu_cpu_tensor = gpu_ctx.outputs[0].to_cpu()?;
        let k_gpu_cpu_tensor = gpu_ctx.outputs[1].to_cpu()?;
        let q_gpu_result = q_gpu_cpu_tensor.as_f32()?.as_slice()?;
        let k_gpu_result = k_gpu_cpu_tensor.as_f32()?.as_slice()?;
        
        // --- 3. 对比结果 ---
        assert_close(&q_gold_result, q_gpu_result, 1e-3);
        assert_close(&k_gold_result, k_gpu_result, 1e-3);
        
        Ok(())
    }
    
}