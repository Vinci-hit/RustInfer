//! Ragged-batch attention: variable `q_len_i` + `kv_len_i` per request, one
//! kernel launch. Uses Q-tile parallelism (no KV split, no cross-block
//! reduction) so each CTA owns a complete `BlockM × HD` output tile.
//!
//! Mirrors [`FlashAttnDecodeBatch`] in the "caller-managed device arrays"
//! philosophy so both ops can share a single BatchPlanner at the call site
//! and both are CUDA-Graph-capturable.
//!
//! Workflow from the caller:
//!   1. Pack Q tokens from all requests in request order:
//!        Q : [Σ q_len_i, num_q_heads, head_dim]
//!   2. Precompute (on host, then upload or keep in pinned buffers):
//!        cu_q_lens  [B+1]  = prefix sum of q_len_i
//!        num_tiles  [B]    = ceil(q_len_i / 128)
//!        block2req  [T]    = for each flattened q-tile, which request
//!        block2tile [T]    = ...and which tile inside that request
//!      where T = Σ num_tiles[i].
//!   3. Launch `forward(...)`.
//!   4. Unpack O back to per-request chunks via cu_q_lens.
//!
//! KV cache tile size inside the kernel: BlockM=128, BlockN=64. Head dim
//! is statically dispatched to {64, 128, 192, 256}.

use std::ffi::c_void;

use crate::OpConfig;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

use crate::op::kernels;

/// Q-tile size used by the ragged kernel.  Must match `kBlockM` in
/// `flash_attn_gqa_prefill.cu`.
pub const RAGGED_Q_TILE: usize = 128;

/// Given per-request `q_lens`, compute the scheduler tables required by
/// [`FlashAttnRagged::forward`]:
///
///   cu_q_lens  : [B+1]
///   block2req  : [T]    flattened (req, q_tile) → req
///   block2tile : [T]                         → tile_in_req
///
/// Host-side O(B + T); callers typically run this once per step and copy
/// the three arrays into stable device buffers.
pub fn plan_ragged_tiles(q_lens: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let b = q_lens.len();
    let mut cu = Vec::with_capacity(b + 1);
    cu.push(0i32);
    let mut acc = 0i32;
    for &q in q_lens {
        acc = acc.saturating_add(q);
        cu.push(acc);
    }
    let mut block2req  = Vec::new();
    let mut block2tile = Vec::new();
    for (req, &q) in q_lens.iter().enumerate() {
        let n_tiles = (q as usize).div_ceil(RAGGED_Q_TILE);
        for t in 0..n_tiles {
            block2req.push(req as i32);
            block2tile.push(t as i32);
        }
    }
    (cu, block2req, block2tile)
}

/// Ragged-batch attention operator.
pub struct FlashAttnRagged {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
}

impl FlashAttnRagged {
    pub fn new(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<Self> {
        if num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(Error::InvalidArgument(format!(
                "FlashAttnRagged requires num_q_heads ({}) to be a nonzero multiple \
                 of num_kv_heads ({})",
                num_q_heads, num_kv_heads
            )).into());
        }
        Ok(Self { num_q_heads, num_kv_heads, head_dim, causal })
    }

    /// Ragged attention forward.
    ///
    /// Inputs (all device-resident; addresses stable for graph capture):
    ///   - `q` / `o` :  `[total_q_tokens, num_q_heads, head_dim]` packed tensors
    ///   - `k_cache_ptrs_dev`, `v_cache_ptrs_dev` :
    ///     device arrays of KV-cache base pointers, indexed by slot
    ///   - `kv_stride_s`, `kv_stride_h` : per-token / per-kv-head stride
    ///     (in elements) within a single KV cache buffer
    ///   - `req_to_slot_dev` : `[B]`  request i → slot id
    ///   - `kv_lens_dev`     : `[B]`  per-request current KV length
    ///   - `cu_q_lens_dev`   : `[B+1]`  prefix sum of `q_len_i`
    ///   - `block2req_dev`   : `[total_q_tiles]`
    ///   - `block2tile_dev`  : `[total_q_tiles]`
    ///   - `total_q_tiles`   : `Σ ceil(q_len_i / RAGGED_Q_TILE)`
    ///
    /// # Safety
    /// All raw pointers must be valid device pointers for the kernel's
    /// lifetime.  The control arrays must be internally consistent (
    /// `plan_ragged_tiles` computes them correctly from `q_lens`).
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
        cu_q_lens_dev: *const i32,
        block2req_dev: *const i32,
        block2tile_dev: *const i32,
        total_q_tiles: i32,
        o: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        unsafe {
            kernels::cuda::flash_attn_ragged(
                q,
                k_cache_ptrs_dev,
                v_cache_ptrs_dev,
                kv_stride_s,
                kv_stride_h,
                o,
                req_to_slot_dev,
                kv_lens_dev,
                cu_q_lens_dev,
                block2req_dev,
                block2tile_dev,
                total_q_tiles,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.causal,
                cuda_config,
            )
        }
    }

    /// Paged ragged/prefill attention forward.
    ///
    /// # Safety
    /// Raw pool pointers and block table pointers must be valid device pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_paged(
        &self,
        q: &Tensor,
        k_pool: *const c_void,
        v_pool: *const c_void,
        block_tables_dev: *const u32,
        max_blocks_per_seq: usize,
        block_size: usize,
        kv_lens_dev: *const i32,
        cu_q_lens_dev: *const i32,
        block2req_dev: *const i32,
        block2tile_dev: *const i32,
        total_q_tiles: i32,
        batch: usize,
        o: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let total_q_tokens = q.shape()[0];
        unsafe {
            kernels::cuda::flash_attn_paged_ragged(
                q,
                k_pool,
                v_pool,
                o,
                block_tables_dev,
                max_blocks_per_seq,
                block_size,
                kv_lens_dev,
                cu_q_lens_dev,
                block2req_dev,
                block2tile_dev,
                total_q_tiles,
                batch,
                total_q_tokens,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.causal,
                cuda_config,
            )
        }
    }
}

// ============================================================================
// Tests — CUDA vs naive f32 reference.
// ============================================================================
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::cuda::CudaConfig;
    use crate::tensor::Tensor;

    fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| {
            let k = ((i * 2654435761usize).wrapping_add(seed)) & 0xFFFF;
            (k as f32 / 65535.0) * 2.0 - 1.0
        }).collect()
    }

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

    /// Per-request naive f32 attention reference.
    /// Q : [q_len, Hq, HD], K / V : [kv_len, Hkv, HD].
    fn naive_attn_ref_f32(
        q: &[f32], k: &[f32], v: &[f32],
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        q_len: usize, kv_len: usize,
        causal: bool, scale: f32,
    ) -> Vec<f32> {
        let groups = num_q_heads / num_kv_heads;
        let causal_shift = kv_len as i32 - q_len as i32;
        let mut out = vec![0.0f32; q_len * num_q_heads * head_dim];
        for qh in 0..num_q_heads {
            let kvh = qh / groups;
            for i in 0..q_len {
                let q_row = &q[(i * num_q_heads + qh) * head_dim ..
                                (i * num_q_heads + qh + 1) * head_dim];
                let k_upper = if causal {
                    ((i as i32 + 1 + causal_shift).max(0) as usize).min(kv_len)
                } else {
                    kv_len
                };
                if k_upper == 0 {
                    continue;
                }
                let mut m = f32::NEG_INFINITY;
                for t in 0..k_upper {
                    let k_row = &k[(t * num_kv_heads + kvh) * head_dim ..
                                    (t * num_kv_heads + kvh + 1) * head_dim];
                    let mut s = 0.0f32;
                    for d in 0..head_dim { s += q_row[d] * k_row[d]; }
                    s *= scale;
                    if s > m { m = s; }
                }
                let mut denom = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for t in 0..k_upper {
                    let k_row = &k[(t * num_kv_heads + kvh) * head_dim ..
                                    (t * num_kv_heads + kvh + 1) * head_dim];
                    let v_row = &v[(t * num_kv_heads + kvh) * head_dim ..
                                    (t * num_kv_heads + kvh + 1) * head_dim];
                    let mut s = 0.0f32;
                    for d in 0..head_dim { s += q_row[d] * k_row[d]; }
                    let e = (s * scale - m).exp();
                    denom += e;
                    for d in 0..head_dim { acc[d] += e * v_row[d]; }
                }
                let inv = if denom == 0.0 { 1.0 } else { 1.0 / denom };
                let out_off = (i * num_q_heads + qh) * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] = acc[d] * inv;
                }
            }
        }
        out
    }

    #[derive(Copy, Clone)]
    enum DT { Bf16, Fp16 }

    fn run_case(
        dt: DT,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_lens: &[i32],
        kv_lens: &[i32],
        causal: bool,
    ) -> Result<()> {
        assert_eq!(q_lens.len(), kv_lens.len(), "q_lens / kv_lens must share B");
        let device = DeviceType::Cuda(0);
        let batch = q_lens.len();
        let q_dim  = num_q_heads  * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scale  = 1.0f32 / (head_dim as f32).sqrt();

        let total_q: i32 = q_lens.iter().sum();
        let total_q = total_q as usize;

        // Inputs.
        // Q packed: [total_q, num_q_heads, head_dim]
        let q_f = rand_f32(total_q * q_dim, 0x11 + head_dim + (causal as usize) * 9);
        // Per-request independent KV cache: [kv_lens[i], num_kv_heads, head_dim]
        let k_f: Vec<Vec<f32>> = (0..batch).map(|b| {
            rand_f32((kv_lens[b] as usize) * kv_dim, 0x22 + b * 101 + head_dim)
        }).collect();
        let v_f: Vec<Vec<f32>> = (0..batch).map(|b| {
            rand_f32((kv_lens[b] as usize) * kv_dim, 0x33 + b * 103 + head_dim)
        }).collect();

        // Upload.
        let q_tensor = match dt {
            DT::Bf16 => upload_bf16(&[total_q, num_q_heads, head_dim], &q_f)?,
            DT::Fp16 => upload_fp16(&[total_q, num_q_heads, head_dim], &q_f)?,
        };
        let k_caches: Vec<Tensor> = (0..batch).map(|b| -> Result<Tensor> {
            let shape = [kv_lens[b] as usize, kv_dim];
            match dt {
                DT::Bf16 => upload_bf16(&shape, &k_f[b]),
                DT::Fp16 => upload_fp16(&shape, &k_f[b]),
            }
        }).collect::<Result<_>>()?;
        let v_caches: Vec<Tensor> = (0..batch).map(|b| -> Result<Tensor> {
            let shape = [kv_lens[b] as usize, kv_dim];
            match dt {
                DT::Bf16 => upload_bf16(&shape, &v_f[b]),
                DT::Fp16 => upload_fp16(&shape, &v_f[b]),
            }
        }).collect::<Result<_>>()?;

        // Device pointer arrays (u64-per-slot, stored inside an I32[2*B] tensor).
        let mut k_ptrs_dev = Tensor::new(&[2 * batch], DataType::I32, device)?;
        let mut v_ptrs_dev = Tensor::new(&[2 * batch], DataType::I32, device)?;
        let k_ptrs_host: Vec<u64> = (0..batch).map(|b| k_caches[b].data_ptr() as u64).collect();
        let v_ptrs_host: Vec<u64> = (0..batch).map(|b| v_caches[b].data_ptr() as u64).collect();
        k_ptrs_dev.buffer_mut().copy_from_host(&k_ptrs_host)?;
        v_ptrs_dev.buffer_mut().copy_from_host(&v_ptrs_host)?;

        // Control arrays.
        let (cu_q_lens_host, block2req_host, block2tile_host) = plan_ragged_tiles(q_lens);
        let total_tiles = block2req_host.len() as i32;

        let mut slot_dev      = Tensor::new(&[batch],               DataType::I32, device)?;
        let mut kvlens_dev    = Tensor::new(&[batch],               DataType::I32, device)?;
        let mut cu_q_lens_dev = Tensor::new(&[cu_q_lens_host.len()],DataType::I32, device)?;
        let mut b2req_dev     = Tensor::new(&[block2req_host.len().max(1)], DataType::I32, device)?;
        let mut b2tile_dev    = Tensor::new(&[block2tile_host.len().max(1)], DataType::I32, device)?;
        let slot_host: Vec<i32> = (0..batch as i32).collect();
        slot_dev     .write_from_i32_host(&slot_host,      batch)?;
        kvlens_dev   .write_from_i32_host(kv_lens,         batch)?;
        cu_q_lens_dev.write_from_i32_host(&cu_q_lens_host, cu_q_lens_host.len())?;
        if !block2req_host.is_empty() {
            b2req_dev .write_from_i32_host(&block2req_host,  block2req_host.len())?;
            b2tile_dev.write_from_i32_host(&block2tile_host, block2tile_host.len())?;
        }

        // Output.
        let out_dtype = match dt { DT::Bf16 => DataType::BF16, DT::Fp16 => DataType::F16 };
        let mut o_tensor = Tensor::new(&[total_q, num_q_heads, head_dim], out_dtype, device)?;

        // Launch.
        let op = FlashAttnRagged::new(num_q_heads, num_kv_heads, head_dim, causal)?;
        let cfg = CudaConfig::new()?;
        unsafe {
            op.forward(
                &q_tensor,
                k_ptrs_dev.data_ptr() as *const *const c_void,
                v_ptrs_dev.data_ptr() as *const *const c_void,
                (num_kv_heads * head_dim) as i64,
                head_dim as i64,
                slot_dev.as_i32()?.data_ptr(),
                kvlens_dev.as_i32()?.data_ptr(),
                cu_q_lens_dev.as_i32()?.data_ptr(),
                b2req_dev.as_i32()?.data_ptr(),
                b2tile_dev.as_i32()?.data_ptr(),
                total_tiles,
                &mut o_tensor,
                Some(&cfg),
            )?;
        }
        unsafe { crate::cuda_check!(crate::cuda::ffi::cudaStreamSynchronize(cfg.stream))?; }

        // Reference + compare (per request, then concatenate).
        let o_cpu = o_tensor.to_cpu()?;
        let (atol, rtol) = match dt {
            DT::Bf16 => (7e-2f32, 1e-2f32),
            DT::Fp16 => (1e-2f32, 5e-3f32),
        };

        let mut max_err = 0.0f32;
        let mut bad = 0usize;
        let mut n = 0usize;
        let mut q_off = 0usize;
        for b in 0..batch {
            let q_len_b = q_lens[b] as usize;
            let kv_len_b = kv_lens[b] as usize;
            if q_len_b == 0 { continue; }
            let q_slice = &q_f[q_off * q_dim .. (q_off + q_len_b) * q_dim];
            let ref_out = naive_attn_ref_f32(
                q_slice, &k_f[b], &v_f[b],
                num_q_heads, num_kv_heads, head_dim,
                q_len_b, kv_len_b, causal, scale,
            );
            for i in 0..q_len_b {
                for qh in 0..num_q_heads {
                    for d in 0..head_dim {
                        let rel_ref  = (i * num_q_heads + qh) * head_dim + d;
                        let rel_got  = ((q_off + i) * num_q_heads + qh) * head_dim + d;
                        let got = match dt {
                            DT::Bf16 => o_cpu.as_bf16()?.as_slice()?[rel_got].to_f32(),
                            DT::Fp16 => o_cpu.as_f16()?.as_slice()?[rel_got].to_f32(),
                        };
                        let r = ref_out[rel_ref];
                        let e = (got - r).abs();
                        let tol = atol + rtol * r.abs();
                        if e > tol { bad += 1; }
                        if e > max_err { max_err = e; }
                        n += 1;
                    }
                }
            }
            q_off += q_len_b;
        }
        println!(
            "ragged  dt={}  Hq={} Hkv={} HD={} q_lens={:?} kv_lens={:?} causal={} \
             max_err={:.4e} bad={}/{}",
            match dt { DT::Bf16 => "bf16", DT::Fp16 => "fp16" },
            num_q_heads, num_kv_heads, head_dim, q_lens, kv_lens, causal,
            max_err, bad, n,
        );
        // Allow a tiny fraction of outliers (bf16 online softmax over long KV
        // occasionally produces a single element past the nominal tol).
        let allowed_bad = (n as f64 * 1e-4).ceil() as usize; // 0.01%
        assert!(
            bad <= allowed_bad,
            "ragged mismatch: max_err={:.4e}, bad={}/{} (allowed ≤ {})",
            max_err, bad, n, allowed_bad,
        );
        Ok(())
    }

    fn run_paged_case(
        dt: DT,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_lens: &[i32],
        kv_lens: &[i32],
        causal: bool,
        block_size: usize,
    ) -> Result<()> {
        assert_eq!(q_lens.len(), kv_lens.len(), "q_lens / kv_lens must share B");
        let device = DeviceType::Cuda(0);
        let batch = q_lens.len();
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let total_q = q_lens.iter().sum::<i32>() as usize;
        let max_kv = *kv_lens.iter().max().unwrap() as usize;
        let max_blocks_per_seq = max_kv.div_ceil(block_size).max(1);
        let num_blocks = batch * max_blocks_per_seq;

        let q_f = rand_f32(total_q * q_dim, 0x411 + head_dim + (causal as usize) * 9);
        let k_pool_f = rand_f32(num_blocks * block_size * kv_dim, 0x422 + head_dim);
        let v_pool_f = rand_f32(num_blocks * block_size * kv_dim, 0x433 + head_dim);

        let mut block_tables = vec![0i32; batch * max_blocks_per_seq];
        for b in 0..batch {
            for logical in 0..max_blocks_per_seq {
                block_tables[b * max_blocks_per_seq + logical] =
                    (b * max_blocks_per_seq + (max_blocks_per_seq - 1 - logical)) as i32;
            }
        }

        let q_tensor = match dt {
            DT::Bf16 => upload_bf16(&[total_q, num_q_heads, head_dim], &q_f)?,
            DT::Fp16 => upload_fp16(&[total_q, num_q_heads, head_dim], &q_f)?,
        };
        let k_pool = match dt {
            DT::Bf16 => upload_bf16(&[num_blocks, block_size, num_kv_heads, head_dim], &k_pool_f)?,
            DT::Fp16 => upload_fp16(&[num_blocks, block_size, num_kv_heads, head_dim], &k_pool_f)?,
        };
        let v_pool = match dt {
            DT::Bf16 => upload_bf16(&[num_blocks, block_size, num_kv_heads, head_dim], &v_pool_f)?,
            DT::Fp16 => upload_fp16(&[num_blocks, block_size, num_kv_heads, head_dim], &v_pool_f)?,
        };

        let (cu_q_lens_host, block2req_host, block2tile_host) = plan_ragged_tiles(q_lens);
        let total_tiles = block2req_host.len() as i32;
        let mut block_dev     = Tensor::new(&[batch * max_blocks_per_seq], DataType::I32, device)?;
        let mut kvlens_dev    = Tensor::new(&[batch], DataType::I32, device)?;
        let mut cu_q_lens_dev = Tensor::new(&[cu_q_lens_host.len()], DataType::I32, device)?;
        let mut b2req_dev     = Tensor::new(&[block2req_host.len().max(1)], DataType::I32, device)?;
        let mut b2tile_dev    = Tensor::new(&[block2tile_host.len().max(1)], DataType::I32, device)?;
        block_dev.write_from_i32_host(&block_tables, block_tables.len())?;
        kvlens_dev.write_from_i32_host(kv_lens, batch)?;
        cu_q_lens_dev.write_from_i32_host(&cu_q_lens_host, cu_q_lens_host.len())?;
        if !block2req_host.is_empty() {
            b2req_dev.write_from_i32_host(&block2req_host, block2req_host.len())?;
            b2tile_dev.write_from_i32_host(&block2tile_host, block2tile_host.len())?;
        }

        let out_dtype = match dt { DT::Bf16 => DataType::BF16, DT::Fp16 => DataType::F16 };
        let mut o_tensor = Tensor::new(&[total_q, num_q_heads, head_dim], out_dtype, device)?;
        let op = FlashAttnRagged::new(num_q_heads, num_kv_heads, head_dim, causal)?;
        let cfg = CudaConfig::new()?;
        unsafe {
            op.forward_paged(
                &q_tensor,
                k_pool.data_ptr() as *const c_void,
                v_pool.data_ptr() as *const c_void,
                block_dev.as_i32()?.data_ptr() as *const u32,
                max_blocks_per_seq,
                block_size,
                kvlens_dev.as_i32()?.data_ptr(),
                cu_q_lens_dev.as_i32()?.data_ptr(),
                b2req_dev.as_i32()?.data_ptr(),
                b2tile_dev.as_i32()?.data_ptr(),
                total_tiles,
                batch,
                &mut o_tensor,
                Some(&cfg),
            )?;
            crate::cuda_check!(crate::cuda::ffi::cudaStreamSynchronize(cfg.stream))?;
        }

        let o_cpu = o_tensor.to_cpu()?;
        let (atol, rtol) = match dt {
            DT::Bf16 => (7e-2f32, 1e-2f32),
            DT::Fp16 => (1e-2f32, 5e-3f32),
        };
        let mut max_err = 0.0f32;
        let mut bad = 0usize;
        let mut n = 0usize;
        let mut q_off = 0usize;
        for b in 0..batch {
            let q_len_b = q_lens[b] as usize;
            let kv_len_b = kv_lens[b] as usize;
            let mut k_seq = vec![0.0f32; kv_len_b * kv_dim];
            let mut v_seq = vec![0.0f32; kv_len_b * kv_dim];
            for t in 0..kv_len_b {
                let logical = t / block_size;
                let off = t % block_size;
                let phys = block_tables[b * max_blocks_per_seq + logical] as usize;
                let src = (phys * block_size + off) * kv_dim;
                let dst = t * kv_dim;
                k_seq[dst..dst + kv_dim].copy_from_slice(&k_pool_f[src..src + kv_dim]);
                v_seq[dst..dst + kv_dim].copy_from_slice(&v_pool_f[src..src + kv_dim]);
            }
            let q_slice = &q_f[q_off * q_dim .. (q_off + q_len_b) * q_dim];
            let ref_out = naive_attn_ref_f32(
                q_slice, &k_seq, &v_seq,
                num_q_heads, num_kv_heads, head_dim,
                q_len_b, kv_len_b, causal, scale,
            );
            for i in 0..q_len_b {
                for qh in 0..num_q_heads {
                    for d in 0..head_dim {
                        let rel_ref = (i * num_q_heads + qh) * head_dim + d;
                        let rel_got = ((q_off + i) * num_q_heads + qh) * head_dim + d;
                        let got = match dt {
                            DT::Bf16 => o_cpu.as_bf16()?.as_slice()?[rel_got].to_f32(),
                            DT::Fp16 => o_cpu.as_f16()?.as_slice()?[rel_got].to_f32(),
                        };
                        let r = ref_out[rel_ref];
                        let e = (got - r).abs();
                        let tol = atol + rtol * r.abs();
                        if e > tol { bad += 1; }
                        if e > max_err { max_err = e; }
                        n += 1;
                    }
                }
            }
            q_off += q_len_b;
        }
        println!(
            "paged-ragged dt={} Hq={} Hkv={} HD={} block={} q_lens={:?} kv_lens={:?} causal={} max_err={:.4e} bad={}/{}",
            match dt { DT::Bf16 => "bf16", DT::Fp16 => "fp16" },
            num_q_heads, num_kv_heads, head_dim, block_size, q_lens, kv_lens, causal, max_err, bad, n,
        );
        let allowed_bad = (n as f64 * 1e-4).ceil() as usize;
        assert!(
            bad <= allowed_bad,
            "paged-ragged mismatch: max_err={:.4e}, bad={}/{} (allowed ≤ {})",
            max_err, bad, n, allowed_bad,
        );
        Ok(())
    }

    #[test]
    fn test_flash_attn_paged_ragged_prefill() -> Result<()> {
        run_paged_case(DT::Bf16, 8, 2, 64, &[128, 65], &[128, 129], true, 16)?;
        run_paged_case(DT::Fp16, 4, 2, 64, &[64, 1], &[96, 33], true, 16)?;
        run_paged_case(DT::Bf16, 8, 2, 128, &[100, 32], &[160, 80], false, 32)?;
        Ok(())
    }

    #[test]
    fn test_flash_attn_ragged_prefill_only() -> Result<()> {
        // Pure prefill batch: q_len == kv_len.
        run_case(DT::Bf16, 8,  2,  64, &[128], &[128], true)?;
        run_case(DT::Bf16, 8,  2,  64, &[128, 160], &[128, 160], true)?;
        run_case(DT::Bf16, 16, 4, 128, &[256, 130], &[256, 130], true)?;
        run_case(DT::Fp16, 8,  2, 128, &[128, 192], &[128, 192], true)?;
        Ok(())
    }

    #[test]
    fn test_flash_attn_ragged_incremental_prefill() -> Result<()> {
        // Chunked prefill: q_len < kv_len.
        run_case(DT::Bf16, 16, 4, 128, &[64, 128], &[512, 128 + 256], true)?;
        run_case(DT::Bf16, 8,  2, 128, &[256], &[1024], true)?;
        Ok(())
    }

    #[test]
    fn test_flash_attn_ragged_mixed_batch() -> Result<()> {
        // Mixed prefill (128 + kv 128) + decode-like (q=1 + kv 2048).
        run_case(DT::Bf16, 16, 4, 128, &[128, 1, 1, 64], &[128, 2048, 500, 64], true)?;
        run_case(DT::Fp16, 16, 4, 128, &[128, 1], &[128, 2048], true)?;
        Ok(())
    }

    #[test]
    fn test_flash_attn_ragged_noncausal() -> Result<()> {
        run_case(DT::Bf16, 8,  2,  64, &[128, 64],  &[256, 130], false)?;
        run_case(DT::Bf16, 8,  2, 128, &[200, 100], &[400, 500], false)?;
        Ok(())
    }

    #[test]
    fn test_flash_attn_ragged_speculative_like() -> Result<()> {
        // Speculative-like: q_len ∈ 2..=16.
        run_case(DT::Bf16, 16, 4, 128, &[4, 8, 2, 16], &[512, 1024, 2048, 200], true)?;
        Ok(())
    }

    #[test]
    fn test_flash_attn_ragged_various_head_dims() -> Result<()> {
        run_case(DT::Bf16, 8,  2, 192, &[128, 64], &[256, 200], true)?;
        run_case(DT::Bf16, 8,  2, 256, &[128], &[256], true)?;
        Ok(())
    }

    #[test]
    fn new_validates_head_counts() {
        assert!(FlashAttnRagged::new(8, 2, 64, true).is_ok());
        assert!(FlashAttnRagged::new(7, 2, 64, true).is_err());
        assert!(FlashAttnRagged::new(8, 0, 64, true).is_err());
    }

    #[test]
    fn plan_ragged_tiles_basic() {
        let (cu, b2r, b2t) = plan_ragged_tiles(&[130, 1, 256]);
        // cu = [0, 130, 131, 387]
        assert_eq!(cu, vec![0, 130, 131, 387]);
        // num_tiles = [2, 1, 2]  → total tiles = 5
        assert_eq!(b2r, vec![0, 0, 1, 2, 2]);
        assert_eq!(b2t, vec![0, 1, 0, 0, 1]);
    }
}
