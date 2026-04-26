use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

/// Scaled Dot-Product Attention (通用任意 head_dim)
///
/// 标准公式: O = softmax(Q @ K^T / sqrt(head_dim)) @ V
///
/// 输入:
///   q: [B, num_heads, S_q, head_dim]
///   k: [B, num_heads, S_kv, head_dim]
///   v: [B, num_heads, S_kv, head_dim]
/// 输出:
///   output: [B, num_heads, S_q, head_dim]
///
/// 支持任意 head_dim，无 causal mask。
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    q: &Tensor,       // [B, num_heads, S_q, head_dim]
    k: &Tensor,       // [B, num_heads, S_kv, head_dim]
    v: &Tensor,       // [B, num_heads, S_kv, head_dim]
    output: &mut Tensor, // [B, num_heads, S_q, head_dim]
    cuda_config: Option<&crate::OpConfig>,
) -> Result<()> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "sdpa: expected 4D input, got shape {:?}", q_shape
        )).into());
    }
    let batch = q_shape[0];
    let num_heads = q_shape[1];
    let s_q = q_shape[2];
    let head_dim = q_shape[3];
    let s_kv = k.shape()[2];

    let scale = 1.0 / (head_dim as f32).sqrt();

    match q.device() {
        DeviceType::Cpu => { let _ = cuda_config; sdpa_cpu(q, k, v, output, batch, num_heads, s_q, s_kv, head_dim, scale) }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => sdpa_cuda(q, k, v, output, batch, num_heads, s_q, s_kv, head_dim, scale, cuda_config),
    }
}

/// CPU 实现: 显式 QK^T → scale → softmax → @V
fn sdpa_cpu(
    q: &Tensor, k: &Tensor, v: &Tensor, output: &mut Tensor,
    batch: usize, num_heads: usize, s_q: usize, s_kv: usize, head_dim: usize, scale: f32,
) -> Result<()> {
    match q.dtype() {
        DataType::F32 => {
            let q_data = q.as_f32()?.as_slice()?;
            let k_data = k.as_f32()?.as_slice()?;
            let v_data = v.as_f32()?.as_slice()?;
            let o_data = output.as_f32_mut()?.as_slice_mut()?;

            let head_stride_q = s_q * head_dim;
            let head_stride_kv = s_kv * head_dim;

            for b in 0..batch {
                for h in 0..num_heads {
                    let q_off = (b * num_heads + h) * head_stride_q;
                    let k_off = (b * num_heads + h) * head_stride_kv;
                    let v_off = k_off;
                    let o_off = q_off;

                    let mut scores = vec![0.0f32; s_q * s_kv];
                    for i in 0..s_q {
                        for j in 0..s_kv {
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_data[q_off + i * head_dim + d]
                                    * k_data[k_off + j * head_dim + d];
                            }
                            scores[i * s_kv + j] = dot * scale;
                        }
                    }

                    for i in 0..s_q {
                        let row = &mut scores[i * s_kv..(i + 1) * s_kv];
                        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let mut sum = 0.0f32;
                        for v in row.iter_mut() {
                            *v = (*v - max_val).exp();
                            sum += *v;
                        }
                        let inv = 1.0 / sum;
                        for v in row.iter_mut() {
                            *v *= inv;
                        }
                    }

                    for i in 0..s_q {
                        for d in 0..head_dim {
                            let mut acc = 0.0f32;
                            for j in 0..s_kv {
                                acc += scores[i * s_kv + j]
                                    * v_data[v_off + j * head_dim + d];
                            }
                            o_data[o_off + i * head_dim + d] = acc;
                        }
                    }
                }
            }
            Ok(())
        }
        _ => Err(Error::InvalidArgument(format!(
            "sdpa CPU: unsupported dtype {:?}", q.dtype()
        )).into()),
    }
}

/// CUDA 实现: 使用矩阵乘 + softmax 组合现有算子，不依赖自定义 CUDA 内核。
///
/// 步骤:
///   1. scores = Q_i @ K_i^T        (sgemm/hgemm_bf16: output = input @ weight^T)
///   2. scaled_scores = scores * scale  (scalar_mul)
///   3. attn = softmax(scaled_scores)   (softmax kernel)
///   4. out_i = attn @ V_i              (sgemm/hgemm_bf16 with transposed V)
#[cfg(feature = "cuda")]
fn sdpa_cuda(
    q: &Tensor, k: &Tensor, v: &Tensor, output: &mut Tensor,
    batch: usize, num_heads: usize, s_q: usize, s_kv: usize, head_dim: usize, scale: f32,
    cuda_config: Option<&crate::OpConfig>,
) -> Result<()> {
    use crate::op::kernels::cuda::{sgemm, hgemm_bf16};
    use crate::op::tensor_utils::permute_nd;

    let _stream = crate::cuda::get_current_cuda_stream();
    let dtype = q.dtype();
    let bh = batch * num_heads;

    // Q/K/V reshape → 3D [bh, seq, dim]
    let q_3d = q.view(&[bh, s_q, head_dim])?;
    let k_3d = k.view(&[bh, s_kv, head_dim])?;
    let v_3d = v.view(&[bh, s_kv, head_dim])?;

    // V 转置: [bh, s_kv, head_dim] → [bh, head_dim, s_kv]
    // sgemm 执行 A @ B^T, 所以 weight=[head_dim, s_kv] → weight^T=[s_kv, head_dim]
    // 因此 attn @ weight^T = attn @ V_i ✓
    let v_t = permute_nd(&v_3d, &[0, 2, 1])?;

    // 预分配临时张量（循环外，减少分配开销）
    let mut scores = Tensor::new(&[s_q, s_kv], dtype, q.device())?;
    let mut scaled_scores = Tensor::new(&[s_q, s_kv], dtype, q.device())?;
    let mut attn = Tensor::new(&[s_q, s_kv], dtype, q.device())?;
    let mut out_i = Tensor::new(&[s_q, head_dim], dtype, q.device())?;

    // output 也 reshape 为 3D 便于按 head 写入
    let output_3d = output.view(&[bh, s_q, head_dim])?;

    for i in 0..bh {
        // 用 slice 取出第 i 个 head 的 2D 切片 (零拷贝)
        let q_i = q_3d.slice(&[i, 0, 0], &[1, s_q, head_dim])?.view(&[s_q, head_dim])?;
        let k_i = k_3d.slice(&[i, 0, 0], &[1, s_kv, head_dim])?.view(&[s_kv, head_dim])?;
        let v_t_i = v_t.slice(&[i, 0, 0], &[1, head_dim, s_kv])?.view(&[head_dim, s_kv])?;

        // ── Step 1: scores = Q_i @ K_i^T ──
        // sgemm(input=[M,K], weight=[N,K], output=[M,N]) computes output = input @ weight^T
        // Q_i=[s_q, head_dim], K_i=[s_kv, head_dim] → scores=[s_q, s_kv] ✓
        match dtype {
            DataType::F32  => sgemm(&q_i, &k_i, &mut scores, cuda_config)?,
            DataType::BF16 => hgemm_bf16(&q_i, &k_i, &mut scores, cuda_config)?,
            _ => unreachable!(),
        }

        // ── Step 2: scaled_scores = scores * scale ──
        crate::op::scalar::scalar_mul(&scores, &mut scaled_scores, scale)?;

        // ── Step 3: attn = softmax(scaled_scores) ──
        crate::op::softmax::softmax(&scaled_scores, &mut attn)?;

        // ── Step 4: out_i = attn @ V_i ──
        // sgemm(input=[M,K], weight=[N,K], output=[M,N]) computes output = input @ weight^T
        // attn=[s_q, s_kv], v_t_i=[head_dim, s_kv] → out_i=[s_q, head_dim]
        // 即 attn @ v_t_i^T = attn @ V_i ✓
        match dtype {
            DataType::F32  => sgemm(&attn, &v_t_i, &mut out_i, cuda_config)?,
            DataType::BF16 => hgemm_bf16(&attn, &v_t_i, &mut out_i, cuda_config)?,
            _ => unreachable!(),
        }

        // 写回 output 的第 i 个 head
        let mut out_slice = output_3d.slice(&[i, 0, 0], &[1, s_q, head_dim])?
            .view(&[s_q, head_dim])?;
        out_slice.copy_from(&out_i)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助函数: 断言两个 f32 slice 足够接近
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "Slices have different lengths: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "sdpa mismatch at index {}: cpu={} gpu={}, diff={}",
                i, x, y, (x - y).abs()
            );
        }
    }

    /// 纯 CPU 正确性测试: 验证 SDPA 输出有限且非零
    #[test]
    fn test_sdpa_cpu_f32_basic() -> Result<()> {
        let (b, h, s, d) = (1, 2, 4, 8);
        let q = Tensor::randn(&[b, h, s, d], DataType::F32, DeviceType::Cpu, Some(42))?;
        let k = Tensor::randn(&[b, h, s, d], DataType::F32, DeviceType::Cpu, Some(43))?;
        let v = Tensor::randn(&[b, h, s, d], DataType::F32, DeviceType::Cpu, Some(44))?;
        let mut out = Tensor::new(&[b, h, s, d], DataType::F32, DeviceType::Cpu)?;

        scaled_dot_product_attention(&q, &k, &v, &mut out, None)?;

        let data = out.as_f32()?.as_slice()?;
        assert!(data.iter().all(|x| x.is_finite()), "output contains non-finite values");
        assert!(data.iter().any(|x| *x != 0.0), "output is all zeros");
        Ok(())
    }

    /// CPU vs CUDA 对比测试 (F32)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_sdpa_cuda_f32_matches_cpu() -> Result<()> {
        // 测试多种尺寸
        for (b, h, s_q, s_kv, d) in [(1, 4, 8, 8, 16), (2, 2, 4, 6, 32), (1, 1, 16, 16, 64)] {
            let q_cpu = Tensor::randn(&[b, h, s_q, d], DataType::F32, DeviceType::Cpu, Some(42))?;
            let k_cpu = Tensor::randn(&[b, h, s_kv, d], DataType::F32, DeviceType::Cpu, Some(43))?;
            let v_cpu = Tensor::randn(&[b, h, s_kv, d], DataType::F32, DeviceType::Cpu, Some(44))?;

            // CPU 计算
            let mut out_cpu = Tensor::new(&[b, h, s_q, d], DataType::F32, DeviceType::Cpu)?;
            scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, &mut out_cpu, None)?;

            // CUDA 计算
            let q_gpu = q_cpu.to_cuda(0)?;
            let k_gpu = k_cpu.to_cuda(0)?;
            let v_gpu = v_cpu.to_cuda(0)?;
            let mut out_gpu = Tensor::new(&[b, h, s_q, d], DataType::F32, DeviceType::Cuda(0))?;

            let cuda_config = crate::cuda::CudaConfig::new()?;
            scaled_dot_product_attention(&q_gpu, &k_gpu, &v_gpu, &mut out_gpu, Some(&cuda_config))?;
            let out_gpu_cpu = out_gpu.to_cpu()?;

            let a = out_cpu.as_f32()?.as_slice()?;
            let b_ = out_gpu_cpu.as_f32()?.as_slice()?;
            assert_close(a, b_, 1e-2);
        }
        Ok(())
    }
}
