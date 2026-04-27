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
        out_slice.copy_from_on_current_stream(&out_i)?;
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// DiT-style self-attention (SHD layout, non-causal, single batch)
// ═══════════════════════════════════════════════════════════════════════════
//
// The Z-Image DiT blocks all use the same attention signature:
//
//   - Q/K/V in `[seq, n_heads, head_dim]` (SHD) — same memory as the
//     `[seq, dim]` QKV projection outputs, just reinterpreted.
//   - Output in `[seq, n_heads, head_dim]` (SHD) — same memory as the
//     `[seq, dim]` input to the `to_out` projection.
//   - Self-attention (S_q == S_kv == seq), no causal mask, no KV cache.
//
// For the fast BF16 + head_dim=128 + seq%128==0 combination we dispatch to
// the CUTLASS flash-attention prefill kernel
// (`launch_flash_attn_cute_bf16_hdim128`) in a single launch that handles
// all heads at once — roughly a 5×+ speedup versus the per-head SGEMM loop
// in [`scaled_dot_product_attention`], plus zero data-movement since SHD is
// the layout the kernel expects natively.
//
// Anything that doesn't match the fast-path preconditions falls back to the
// existing per-head SDPA by internally permuting SHD → BHSD and back.

/// Self-attention for DiT-style inputs with native SHD layout.
///
/// # Arguments
/// - `q`, `k`, `v`: shape `[seq, n_heads, head_dim]`, same dtype/device.
/// - `output`: shape `[seq, n_heads, head_dim]`, pre-allocated.
/// - `n_heads`, `head_dim`: attention shape parameters.
/// - `cuda_config`: required on CUDA (for stream/workspace).
///
/// # Fast path
/// BF16 + CUDA + `head_dim == 128` + `seq % 128 == 0` → single-launch
/// CUTLASS prefill kernel. Zero intermediate copies.
///
/// # Fallback
/// Delegates to [`scaled_dot_product_attention`] with an internal SHD→BHSD
/// permute and back. Correct but slower.
pub fn dit_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    n_heads: usize,
    head_dim: usize,
    cuda_config: Option<&crate::OpConfig>,
) -> Result<()> {
    let q_shape = q.shape();
    if q_shape.len() != 3 || q_shape[1] != n_heads || q_shape[2] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "dit_sdpa: q must be [seq, {}, {}], got {:?}",
            n_heads, head_dim, q_shape,
        )).into());
    }
    let seq = q_shape[0];
    if k.shape() != q_shape || v.shape() != q_shape || output.shape() != q_shape {
        return Err(Error::InvalidArgument(format!(
            "dit_sdpa: all tensors must share shape [seq, n_heads, head_dim], got q={:?} k={:?} v={:?} out={:?}",
            q_shape, k.shape(), v.shape(), output.shape(),
        )).into());
    }

    #[cfg(feature = "cuda")]
    {
        // `DIT_SDPA_FALLBACK=1` forces the slow per-head path — used by
        // parity tests to diff flash-attn against the reference SDPA.
        let force_fallback = std::env::var_os("DIT_SDPA_FALLBACK").is_some();
        if !force_fallback
            && q.device().is_cuda()
            && q.dtype() == DataType::BF16
            && head_dim == 128
            && seq % 128 == 0
            && seq > 0
        {
            return dit_sdpa_cuda_flash_bf16_hdim128(q, k, v, output, seq, n_heads, cuda_config);
        }
    }

    // Fallback: shuttle through BHSD and call the generic per-head SDPA.
    dit_sdpa_fallback(q, k, v, output, seq, n_heads, head_dim, cuda_config)
}

#[cfg(feature = "cuda")]
fn dit_sdpa_cuda_flash_bf16_hdim128(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    seq: usize,
    n_heads: usize,
    _cuda_config: Option<&crate::OpConfig>,
) -> Result<()> {
    use crate::cuda::CudaConfig;

    unsafe extern "C" {
        fn launch_flash_attn_cute_bf16_hdim128(
            q_ptr: *const half::bf16,
            k_ptr: *const half::bf16,
            v_ptr: *const half::bf16,
            o_ptr: *mut half::bf16,
            q_seq_len: i32,
            kv_len_ptr: *const i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            is_causal: i32,
            stream: crate::cuda::ffi::cudaStream_t,
        );
    }

    let stream = CudaConfig::resolve_stream(_cuda_config);

    // Self-attention (no KV cache): the kernel computes kv_len = *kv_len_ptr
    // + seq, which it does on the HOST side (see the kernel's launch
    // function). So kv_len_ptr must be a host pointer; a stack i32 of 0
    // satisfies that and makes `kv_len == seq`.
    let kv_len_host: i32 = 0;

    let q_ptr = q.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let k_ptr = k.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let v_ptr = v.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let o_ptr = output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

    unsafe {
        launch_flash_attn_cute_bf16_hdim128(
            q_ptr, k_ptr, v_ptr, o_ptr,
            seq as i32,
            &kv_len_host as *const i32,
            n_heads as i32,
            n_heads as i32, // self-attn: num_kv_heads == num_q_heads
            /*is_causal=*/ 0,
            stream,
        );
    }
    Ok(())
}

fn dit_sdpa_fallback(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    cuda_config: Option<&crate::OpConfig>,
) -> Result<()> {
    use crate::op::tensor_utils::{materialize, permute_nd};

    // SHD [seq, H, D] → BHSD [1, H, seq, D] for the generic kernel.
    let to_bhsd = |t: &Tensor| -> Result<Tensor> {
        let hsd = permute_nd(&t.view(&[seq, n_heads, head_dim])?, &[1, 0, 2])?;
        let bhsd_view = hsd.view(&[1, n_heads, seq, head_dim])?;
        materialize(&bhsd_view)
    };
    let q_bhsd = to_bhsd(q)?;
    let k_bhsd = to_bhsd(k)?;
    let v_bhsd = to_bhsd(v)?;

    let mut out_bhsd = Tensor::new(&[1, n_heads, seq, head_dim], q.dtype(), q.device())?;
    scaled_dot_product_attention(&q_bhsd, &k_bhsd, &v_bhsd, &mut out_bhsd, cuda_config)?;

    // BHSD [1, H, seq, D] → SHD [seq, H, D] for the output.
    let out_hsd = out_bhsd.view(&[n_heads, seq, head_dim])?;
    let out_shd = permute_nd(&out_hsd, &[1, 0, 2])?;
    output.copy_from_on_current_stream(&out_shd.view(&[seq, n_heads, head_dim])?)?;
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

    // ── dit_sdpa: flash-attn fast path vs BHSD fallback parity ──
    //
    // Exercises exactly the config used by Z-Image DiT main layers:
    // BF16, seq multiple of 128, n_heads=30, head_dim=128 — which hits
    // `launch_flash_attn_cute_bf16_hdim128`. Compares tensor output
    // against the SHD-wrapped generic SDPA (BHSD permute + per-head
    // sgemm/softmax loop) and requires a small BF16 tolerance.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_dit_sdpa_flash_matches_fallback_bf16_hdim128() -> Result<()> {
        use half::bf16;
        let seq = 128;
        let n_heads = 30;
        let head_dim = 128;

        // Build random SHD tensors on CPU (F32), then quantize to BF16 and
        // ship to GPU — both paths consume the identical BF16 bytes, so any
        // delta is purely due to the different reduction orders.
        let q_f32 = Tensor::randn(&[seq, n_heads, head_dim], DataType::F32, DeviceType::Cpu, Some(1))?;
        let k_f32 = Tensor::randn(&[seq, n_heads, head_dim], DataType::F32, DeviceType::Cpu, Some(2))?;
        let v_f32 = Tensor::randn(&[seq, n_heads, head_dim], DataType::F32, DeviceType::Cpu, Some(3))?;

        let to_bf16_cuda = |t: &Tensor| -> Result<Tensor> {
            let bf = t.to_dtype(DataType::BF16)?;
            bf.to_cuda(0)
        };
        let q_gpu = to_bf16_cuda(&q_f32)?;
        let k_gpu = to_bf16_cuda(&k_f32)?;
        let v_gpu = to_bf16_cuda(&v_f32)?;

        let cuda_config = crate::cuda::CudaConfig::new()?;

        // Fast path (flash-attn)
        let mut out_flash = Tensor::new(&[seq, n_heads, head_dim], DataType::BF16, DeviceType::Cuda(0))?;
        dit_sdpa_cuda_flash_bf16_hdim128(&q_gpu, &k_gpu, &v_gpu, &mut out_flash, seq, n_heads, Some(&cuda_config))?;

        // Fallback (per-head sgemm/softmax)
        let mut out_ref = Tensor::new(&[seq, n_heads, head_dim], DataType::BF16, DeviceType::Cuda(0))?;
        dit_sdpa_fallback(&q_gpu, &k_gpu, &v_gpu, &mut out_ref, seq, n_heads, head_dim, Some(&cuda_config))?;

        // Copy back + convert to f32 for comparison.
        let a = out_flash.to_cpu()?;
        let b = out_ref.to_cpu()?;
        let a_slice = a.as_bf16()?.as_slice()?;
        let b_slice = b.as_bf16()?.as_slice()?;
        let mut max_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (x, y) in a_slice.iter().zip(b_slice.iter()) {
            let xf = x.to_f32();
            let yf = y.to_f32();
            let d = (xf - yf).abs();
            if d > max_diff { max_diff = d; }
            sum_sq_diff += (d as f64) * (d as f64);
            sum_sq_ref += (yf as f64) * (yf as f64);
        }
        let rel_rms = (sum_sq_diff / sum_sq_ref.max(1e-12)).sqrt();
        eprintln!(
            "dit_sdpa parity: max_abs_diff={:.4}, rel_rms_diff={:.4}",
            max_diff, rel_rms
        );
        // Two independent BF16 reductions of ~128 summands differ by a few
        // ULPs per output — we allow a generous 5% relative RMS and 0.5
        // absolute for peak element divergence.
        assert!(rel_rms < 0.05, "relative RMS diff too large: {}", rel_rms);
        assert!(max_diff < 0.5, "max abs diff too large: {}", max_diff);
        let _ = bf16::from_f32(0.0); // silence unused import if any
        Ok(())
    }
}
