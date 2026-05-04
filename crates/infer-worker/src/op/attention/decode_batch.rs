//! Batched Flash-Decoding operator.
//!
//! One kernel launch handles `B` independent decode requests, each with its
//! own KV cache buffer.  The kernel auto-splits long KV sequences into
//! chunks (see `flash_attn_batched_decode.cu`) so batch-size + SM-count
//! mismatches are hidden even for single-request / long-context cases.
//!
//! **Caller-managed device arrays.**  To keep the forward call
//! CUDA-Graph-capturable, every input pointer to the kernel is expected to
//! live in a device buffer whose address is stable across replays:
//!
//! - `k_cache_ptrs_dev` / `v_cache_ptrs_dev` – arrays of KV-cache base
//!   pointers (one per "slot").  Cardinality ≥ max number of live slots.
//! - `req_to_slot_dev` – `[B]` i32 mapping each batch slot to a KV-cache
//!   slot.  The Rust layer only re-fills *contents* of these buffers per
//!   step; addresses stay constant.
//! - `kv_lens_dev`     – `[B]` i32 current KV length per request.
//! - `workspace`       – `f32` scratch, at least [`FlashAttnDecodeBatch::workspace_bytes`].
//!
//! Only BF16 / FP16 on CUDA.  F32 / CPU are intentionally not supported.

use std::ffi::c_void;

use crate::OpConfig;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

use crate::op::kernels;

/// Batched Flash-Decoding (q_len = 1 per request, many requests in one launch).
pub struct FlashAttnDecodeBatch {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl FlashAttnDecodeBatch {
    pub fn new(num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        if num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(Error::InvalidArgument(format!(
                "FlashAttnDecodeBatch requires num_q_heads ({}) to be a nonzero multiple \
                 of num_kv_heads ({})",
                num_q_heads, num_kv_heads
            )).into());
        }
        Ok(Self { num_q_heads, num_kv_heads, head_dim })
    }

    /// Size (in bytes) of the `f32` workspace required for a given batch/shape.
    /// Queryable statically so callers can pre-allocate before capturing a graph.
    pub fn workspace_bytes(
        batch: usize,
        num_q_heads: usize,
        head_dim: usize,
    ) -> usize {
        kernels::cuda::batched_decode_workspace_bytes(batch, num_q_heads, head_dim)
    }

    /// Batched decode forward.
    ///
    /// `q` / `o` must be `[batch, num_q_heads, head_dim]` CUDA tensors.
    /// Every other pointer is a raw device pointer owned by the caller
    /// (so the caller can freely use CUDA Graph capture).
    ///
    /// # Safety
    /// All raw pointers must be valid device pointers for the duration of the
    /// launch, with stable addresses across graph replays.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        q: &Tensor,
        k_cache_ptrs_dev: *const *const c_void,
        v_cache_ptrs_dev: *const *const c_void,
        kv_stride_s: i64,
        kv_stride_h: i64,
        req_to_slot_dev: *const i32,
        kv_lens_dev: *const i32,
        workspace: *mut f32,
        o: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        if q.shape().len() != 3 || o.shape().len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "FlashAttnDecodeBatch expects q/o of shape [batch, num_q_heads, head_dim] \
                 (got q={:?}, o={:?})", q.shape(), o.shape()
            )).into());
        }
        let batch = q.shape()[0];
        if o.shape()[0] != batch {
            return Err(Error::InvalidArgument(format!(
                "q/o batch mismatch: {} vs {}", batch, o.shape()[0]
            )).into());
        }
        if q.shape()[1] != self.num_q_heads || q.shape()[2] != self.head_dim {
            return Err(Error::InvalidArgument(format!(
                "q shape {:?} incompatible with operator [_, {}, {}]",
                q.shape(), self.num_q_heads, self.head_dim,
            )).into());
        }
        unsafe {
            kernels::cuda::flash_attn_batched_decode(
                q,
                k_cache_ptrs_dev,
                v_cache_ptrs_dev,
                kv_stride_s,
                kv_stride_h,
                o,
                req_to_slot_dev,
                kv_lens_dev,
                workspace,
                batch,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                cuda_config,
            )
        }
    }
}

// ============================================================================
// Tests
//   CUDA-vs-naive-f32-reference.  The reference runs on-device in pure f32
//   so we do not depend on any CPU attention op.
// ============================================================================
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::cuda::CudaConfig;
    use crate::tensor::Tensor;

    // ---- small deterministic RNG for reproducible inputs ----
    fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| {
            let k = ((i * 2654435761usize).wrapping_add(seed)) & 0xFFFF;
            (k as f32 / 65535.0) * 2.0 - 1.0
        }).collect()
    }

    /// Naive f32 reference: one-request decode, performed entirely on CPU via
    /// vectors (for correctness, not speed).  Inputs are already in f32.
    fn naive_decode_ref_f32(
        q: &[f32],             // [num_q_heads * head_dim]
        k: &[f32],             // [kv_len * num_kv_heads * head_dim]
        v: &[f32],             // same
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let groups = num_q_heads / num_kv_heads;
        let mut out = vec![0.0f32; num_q_heads * head_dim];
        for qh in 0..num_q_heads {
            let kvh = qh / groups;
            let q_row = &q[qh * head_dim .. (qh + 1) * head_dim];

            // pass 1: max
            let mut m = f32::NEG_INFINITY;
            for t in 0..kv_len {
                let k_row_start = (t * num_kv_heads + kvh) * head_dim;
                let k_row = &k[k_row_start .. k_row_start + head_dim];
                let mut s = 0.0f32;
                for d in 0..head_dim { s += q_row[d] * k_row[d]; }
                s *= scale;
                if s > m { m = s; }
            }

            // pass 2: Σ exp(s-m) * v, Σ exp(s-m)
            let mut denom = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];
            for t in 0..kv_len {
                let k_row_start = (t * num_kv_heads + kvh) * head_dim;
                let v_row_start = k_row_start;
                let k_row = &k[k_row_start .. k_row_start + head_dim];
                let v_row = &v[v_row_start .. v_row_start + head_dim];
                let mut s = 0.0f32;
                for d in 0..head_dim { s += q_row[d] * k_row[d]; }
                let e = (s * scale - m).exp();
                denom += e;
                for d in 0..head_dim { acc[d] += e * v_row[d]; }
            }
            let inv = if denom == 0.0 { 1.0 } else { 1.0 / denom };
            for d in 0..head_dim {
                out[qh * head_dim + d] = acc[d] * inv;
            }
        }
        out
    }

    /// Helper: convert f32 vec to a fresh CUDA bf16 Tensor of shape `shape`.
    fn upload_bf16(shape: &[usize], data_f32: &[f32]) -> Result<Tensor> {
        let device = DeviceType::Cuda(0);
        let mut t = Tensor::new(shape, DataType::BF16, device)?;
        let bf: Vec<half::bf16> = data_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        t.as_bf16_mut()?.buffer_mut().copy_from_host(&bf)?;
        Ok(t)
    }
    fn upload_fp16(shape: &[usize], data_f32: &[f32]) -> Result<Tensor> {
        let device = DeviceType::Cuda(0);
        let mut t = Tensor::new(shape, DataType::F16, device)?;
        let h: Vec<half::f16> = data_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        t.as_f16_mut()?.buffer_mut().copy_from_host(&h)?;
        Ok(t)
    }

    #[derive(Copy, Clone)]
    enum DT { Bf16, Fp16 }

    fn run_case(
        dt: DT,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_lens: &[i32],
    ) -> Result<()> {
        let device = DeviceType::Cuda(0);
        let batch = kv_lens.len();
        let max_kv = *kv_lens.iter().max().unwrap() as usize;
        let q_dim  = num_q_heads  * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scale  = 1.0f32 / (head_dim as f32).sqrt();

        // ---- inputs ----
        // Q: [batch, num_q_heads, head_dim]
        let q_f = rand_f32(batch * q_dim, 0x11 + head_dim);
        // B independent KV caches: [max_kv, num_kv_heads, head_dim]
        let k_f: Vec<Vec<f32>> = (0..batch)
            .map(|b| rand_f32(max_kv * kv_dim, 0x22 + b * 101 + head_dim))
            .collect();
        let v_f: Vec<Vec<f32>> = (0..batch)
            .map(|b| rand_f32(max_kv * kv_dim, 0x33 + b * 103 + head_dim))
            .collect();

        // ---- device tensors ----
        let q_tensor = match dt {
            DT::Bf16 => upload_bf16(&[batch, num_q_heads, head_dim], &q_f)?,
            DT::Fp16 => upload_fp16(&[batch, num_q_heads, head_dim], &q_f)?,
        };
        // B independent KV-cache tensors (one [max_kv, kv_dim] each).
        let k_caches: Vec<Tensor> = (0..batch).map(|b| -> Result<Tensor> {
            match dt {
                DT::Bf16 => upload_bf16(&[max_kv, kv_dim], &k_f[b]),
                DT::Fp16 => upload_fp16(&[max_kv, kv_dim], &k_f[b]),
            }
        }).collect::<Result<_>>()?;
        let v_caches: Vec<Tensor> = (0..batch).map(|b| -> Result<Tensor> {
            match dt {
                DT::Bf16 => upload_bf16(&[max_kv, kv_dim], &v_f[b]),
                DT::Fp16 => upload_fp16(&[max_kv, kv_dim], &v_f[b]),
            }
        }).collect::<Result<_>>()?;

        // device pointer arrays (u64 packed in I32[2] per slot).
        let mut k_ptrs_dev = Tensor::new(&[2 * batch], DataType::I32, device)?;
        let mut v_ptrs_dev = Tensor::new(&[2 * batch], DataType::I32, device)?;
        let k_ptrs_host: Vec<u64> = (0..batch).map(|b| k_caches[b].data_ptr() as u64).collect();
        let v_ptrs_host: Vec<u64> = (0..batch).map(|b| v_caches[b].data_ptr() as u64).collect();
        k_ptrs_dev.buffer_mut().copy_from_host(&k_ptrs_host)?;
        v_ptrs_dev.buffer_mut().copy_from_host(&v_ptrs_host)?;

        // req_to_slot + kv_lens
        let mut slot_dev  = Tensor::new(&[batch], DataType::I32, device)?;
        let mut kvlen_dev = Tensor::new(&[batch], DataType::I32, device)?;
        let slot_host: Vec<i32> = (0..batch as i32).collect();
        slot_dev.write_from_i32_host(&slot_host, batch)?;
        kvlen_dev.write_from_i32_host(kv_lens, batch)?;

        // workspace
        let ws_bytes = FlashAttnDecodeBatch::workspace_bytes(batch, num_q_heads, head_dim);
        let ws_len = ws_bytes.div_ceil(std::mem::size_of::<f32>());
        let workspace = Tensor::new(&[ws_len], DataType::F32, device)?;

        // output
        let out_dtype = match dt { DT::Bf16 => DataType::BF16, DT::Fp16 => DataType::F16 };
        let mut o_tensor = Tensor::new(&[batch, num_q_heads, head_dim], out_dtype, device)?;

        // ---- launch ----
        let op = FlashAttnDecodeBatch::new(num_q_heads, num_kv_heads, head_dim)?;
        let cfg = CudaConfig::new()?;
        unsafe {
            op.forward(
                &q_tensor,
                k_ptrs_dev.data_ptr() as *const *const c_void,
                v_ptrs_dev.data_ptr() as *const *const c_void,
                (num_kv_heads * head_dim) as i64,  // kv_stride_s
                head_dim as i64,                    // kv_stride_h
                slot_dev.as_i32()?.data_ptr(),
                kvlen_dev.as_i32()?.data_ptr(),
                workspace.as_f32()?.data_ptr() as *mut f32,
                &mut o_tensor,
                Some(&cfg),
            )?;
        }
        unsafe { crate::cuda_check!(crate::cuda::ffi::cudaStreamSynchronize(cfg.stream))?; }

        // ---- reference on CPU ----
        let o_cpu = o_tensor.to_cpu()?;
        let (atol, rtol) = match dt {
            DT::Bf16 => (7e-2f32, 1e-2f32),
            DT::Fp16 => (1e-2f32, 5e-3f32),
        };

        let mut max_err: f32 = 0.0;
        let mut bad = 0usize;
        let mut n = 0usize;
        for b in 0..batch {
            let kv_len_b = kv_lens[b] as usize;
            let q_slice = &q_f[b * q_dim .. (b + 1) * q_dim];
            let k_slice = &k_f[b][.. kv_len_b * kv_dim];
            let v_slice = &v_f[b][.. kv_len_b * kv_dim];
            let ref_out = naive_decode_ref_f32(
                q_slice, k_slice, v_slice,
                num_q_heads, num_kv_heads, head_dim, kv_len_b, scale,
            );
            for qh in 0..num_q_heads {
                for d in 0..head_dim {
                    let ridx = qh * head_dim + d;
                    let gidx = (b * num_q_heads + qh) * head_dim + d;
                    let got = match dt {
                        DT::Bf16 => o_cpu.as_bf16()?.as_slice()?[gidx].to_f32(),
                        DT::Fp16 => o_cpu.as_f16()?.as_slice()?[gidx].to_f32(),
                    };
                    let r = ref_out[ridx];
                    let e = (got - r).abs();
                    let tol = atol + rtol * r.abs();
                    if e > tol { bad += 1; }
                    if e > max_err { max_err = e; }
                    n += 1;
                }
            }
        }
        println!(
            "decode-batch  dt={}  Hq={} Hkv={} HD={} kv_lens={:?}  max_err={:.4e} bad={}/{}",
            match dt { DT::Bf16 => "bf16", DT::Fp16 => "fp16" },
            num_q_heads, num_kv_heads, head_dim, kv_lens, max_err, bad, n,
        );
        assert!(bad == 0,
            "decode-batch mismatch: max_err={:.4e}, bad={}/{}", max_err, bad, n);
        Ok(())
    }

    #[test]
    fn test_flash_attn_decode_batch() -> Result<()> {
        // ---- head_dim = 64 ----
        run_case(DT::Bf16, 8,  2,  64, &[100])?;
        run_case(DT::Bf16, 4,  2,  64, &[10, 20, 30])?;
        run_case(DT::Bf16, 8,  2,  64, &[99, 199, 299])?;
        run_case(DT::Fp16, 4,  2,  64, &[10, 20, 30])?;
        // ---- head_dim = 128 ----
        run_case(DT::Bf16, 8,  2, 128, &[50, 50, 50, 50])?;
        run_case(DT::Bf16, 16, 4, 128, &[100, 2048, 500, 99])?;
        run_case(DT::Bf16, 8,  2, 128, &[4096, 8192])?;
        run_case(DT::Fp16, 8,  2, 128, &[4096, 8192])?;
        // ---- head_dim = 192 ----
        run_case(DT::Bf16, 8,  2, 192, &[1234])?;
        // ---- head_dim = 256 ----
        run_case(DT::Bf16, 8,  2, 256, &[1024])?;
        Ok(())
    }

    #[test]
    fn test_new_validates_head_counts() {
        assert!(FlashAttnDecodeBatch::new(8, 2, 64).is_ok());
        assert!(FlashAttnDecodeBatch::new(7, 2, 64).is_err()); // not divisible
        assert!(FlashAttnDecodeBatch::new(8, 0, 64).is_err()); // zero kv heads
    }
}
