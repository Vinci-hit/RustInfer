//! Scaled Dot-Product Attention for DiT (no KV cache, fixed seq).
//!
//! Implementation strategy:
//!   1. SHD → HSD permute Q, K, V (one kernel each).
//!   2. Replicate K, V along head axis if `n_heads > n_kv_heads` (GQA).
//!   3. Strided-batched GEMM `scores = Q @ K^T`.
//!   4. Scale + softmax (row-wise over last axis = seq_kv).
//!   5. Strided-batched GEMM `out = attn @ V`.
//!   6. HSD → SHD permute back.
//!
//! All matmuls go through cuBLAS `cublasGemmStridedBatchedEx` (extern wrappers
//! `gemm_strided_batched_{bf16,f32}_{axbt,axb}` in `matmul.cu`). No host
//! round-trips, no per-row D2D launches.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype, Shape};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::{
    cudaError_cudaSuccess, cudaMemcpyAsync, cudaMemcpyKind, cudaStream_t,
};

unsafe extern "C" {
    fn gemm_strided_batched_bf16_axbt(
        a: *const half::bf16,
        b: *const half::bf16,
        c: *mut half::bf16,
        m: i32,
        n: i32,
        k: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch: i32,
        stream: cudaStream_t,
    );
    fn gemm_strided_batched_f32_axbt(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch: i32,
        stream: cudaStream_t,
    );

    fn permute_bf16_forward(
        dst: *mut half::bf16,
        src: *const half::bf16,
        ndim: i32,
        new_shape: *const i64,
        new_strides: *const i64,
        old_strides: *const i64,
        perm: *const i32,
        num_elements: i64,
        stream: cudaStream_t,
    );
    fn permute_f32_forward(
        dst: *mut f32,
        src: *const f32,
        ndim: i32,
        new_shape: *const i64,
        new_strides: *const i64,
        old_strides: *const i64,
        perm: *const i32,
        num_elements: i64,
        stream: cudaStream_t,
    );
}

/// Permute a 3D tensor with arbitrary `perm` of its 3 axes.
/// `src` is contiguous `[a0, a1, a2]` (row-major).
/// `dst` must be pre-allocated with shape `[new0, new1, new2]` where
/// `new_i = src.shape[perm[i]]`.
fn permute_3d<T: Dtype>(
    src: *const T,
    dst: *mut T,
    src_shape: [i64; 3],
    perm: [i32; 3],
    stream: cudaStream_t,
) -> OpResult<()> {
    let new_shape: [i64; 3] = [
        src_shape[perm[0] as usize],
        src_shape[perm[1] as usize],
        src_shape[perm[2] as usize],
    ];
    let new_strides: [i64; 3] = [new_shape[1] * new_shape[2], new_shape[2], 1];
    let old_strides: [i64; 3] = [src_shape[1] * src_shape[2], src_shape[2], 1];
    let numel = src_shape[0] * src_shape[1] * src_shape[2];
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => permute_bf16_forward(
                dst as *mut half::bf16,
                src as *const half::bf16,
                3,
                new_shape.as_ptr(),
                new_strides.as_ptr(),
                old_strides.as_ptr(),
                perm.as_ptr(),
                numel,
                stream,
            ),
            DataType::F32 => permute_f32_forward(
                dst as *mut f32,
                src as *const f32,
                3,
                new_shape.as_ptr(),
                new_strides.as_ptr(),
                old_strides.as_ptr(),
                perm.as_ptr(),
                numel,
                stream,
            ),
            other => {
                return Err(OpError::Kernel(format!(
                    "permute_3d: unsupported dtype {:?}",
                    other
                )));
            }
        }
    }
    Ok(())
}

/// SDPA on SHD-layout `[seq, n_heads, head_dim]` (Q) and
/// `[seq, n_kv_heads, head_dim]` (K, V). GQA: `n_heads % n_kv_heads == 0`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa<T: Dtype>(
    stream: cudaStream_t,
    q: &Tensor<T, Cuda>,
    k: &Tensor<T, Cuda>,
    v: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<()> {
    if num_heads % num_kv_heads != 0 {
        return Err(OpError::Kernel(format!(
            "sdpa: n_heads ({}) must be divisible by n_kv_heads ({})",
            num_heads, num_kv_heads,
        )));
    }
    let group = num_heads / num_kv_heads;
    if !matches!(T::DATA_TYPE, DataType::F32 | DataType::BF16) {
        return Err(OpError::Kernel(format!(
            "sdpa: unsupported dtype {:?}",
            T::DATA_TYPE
        )));
    }
    let qs = q.shape().as_slice();
    let ks = k.shape().as_slice();
    let vs = v.shape().as_slice();
    let os = output.shape().as_slice();
    if qs.len() != 3 {
        return Err(OpError::Shape(format!(
            "sdpa: Q expected SHD, got {:?}",
            qs
        )));
    }
    let (seq, h_q, d) = (qs[0], qs[1], qs[2]);
    if h_q != num_heads || d != head_dim {
        return Err(OpError::Shape(format!(
            "sdpa: Q shape {:?} doesn't match (n_heads={}, head_dim={})",
            qs, num_heads, head_dim,
        )));
    }
    if ks != [seq, num_kv_heads, head_dim] || vs != [seq, num_kv_heads, head_dim] {
        return Err(OpError::Shape(format!(
            "sdpa: KV shape mismatch k={:?} v={:?} expected=[{}, {}, {}]",
            ks, vs, seq, num_kv_heads, head_dim,
        )));
    }
    if os != qs {
        return Err(OpError::Shape(format!(
            "sdpa: output shape {:?} != Q shape {:?}",
            os, qs
        )));
    }

    let dev = q.device().clone();

    // ── 1. Permute Q [S, H, D] → [H, S, D] ──
    let q_hsd: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
    permute_3d::<T>(
        q.data_ptr(),
        q_hsd.data_ptr_mut(),
        [seq as i64, num_heads as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;

    // ── 2. Permute K, V [S, Hkv, D] → [Hkv, S, D] ──
    let k_hsd_kv: Tensor<T, Cuda> = Tensor::zeros([num_kv_heads, seq, head_dim], &dev)?;
    let v_hsd_kv: Tensor<T, Cuda> = Tensor::zeros([num_kv_heads, seq, head_dim], &dev)?;
    permute_3d::<T>(
        k.data_ptr(),
        k_hsd_kv.data_ptr_mut(),
        [seq as i64, num_kv_heads as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;
    permute_3d::<T>(
        v.data_ptr(),
        v_hsd_kv.data_ptr_mut(),
        [seq as i64, num_kv_heads as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;

    // ── 3. GQA: replicate KV from [Hkv, S, D] to [H, S, D] by copying each
    //         kv head `group` times. One D2D per kv head (Hkv launches, max 32).
    let (k_hsd, v_hsd) = if group == 1 {
        (k_hsd_kv, v_hsd_kv)
    } else {
        let k_full: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
        let v_full: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
        let head_bytes = seq * head_dim * T::SIZE_BYTES;
        unsafe {
            for kv_hi in 0..num_kv_heads {
                for g in 0..group {
                    let dst_hi = kv_hi * group + g;
                    let src_off = kv_hi * seq * head_dim * T::SIZE_BYTES;
                    let dst_off = dst_hi * seq * head_dim * T::SIZE_BYTES;
                    let ksrc = (k_hsd_kv.data_ptr() as *const u8).add(src_off);
                    let kdst = (k_full.data_ptr_mut() as *mut u8).add(dst_off);
                    let vsrc = (v_hsd_kv.data_ptr() as *const u8).add(src_off);
                    let vdst = (v_full.data_ptr_mut() as *mut u8).add(dst_off);
                    let r1 = cudaMemcpyAsync(
                        kdst as *mut _,
                        ksrc as *const _,
                        head_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream,
                    );
                    if r1 != cudaError_cudaSuccess {
                        return Err(OpError::Kernel(format!("sdpa K replicate: {:?}", r1)));
                    }
                    let r2 = cudaMemcpyAsync(
                        vdst as *mut _,
                        vsrc as *const _,
                        head_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream,
                    );
                    if r2 != cudaError_cudaSuccess {
                        return Err(OpError::Kernel(format!("sdpa V replicate: {:?}", r2)));
                    }
                }
            }
        }
        (k_full, v_full)
    };

    // ── 4. scores = Q @ K^T, shape [H, S, S].
    let mut scores: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, seq], &dev)?;
    let stride_qkv = (seq * head_dim) as i64;
    let stride_scores = (seq * seq) as i64;
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => gemm_strided_batched_bf16_axbt(
                q_hsd.data_ptr() as *const half::bf16,
                k_hsd.data_ptr() as *const half::bf16,
                scores.data_ptr_mut() as *mut half::bf16,
                seq as i32,
                seq as i32,
                head_dim as i32,
                stride_qkv,
                stride_qkv,
                stride_scores,
                num_heads as i32,
                stream,
            ),
            DataType::F32 => gemm_strided_batched_f32_axbt(
                q_hsd.data_ptr() as *const f32,
                k_hsd.data_ptr() as *const f32,
                scores.data_ptr_mut() as *mut f32,
                seq as i32,
                seq as i32,
                head_dim as i32,
                stride_qkv,
                stride_qkv,
                stride_scores,
                num_heads as i32,
                stream,
            ),
            _ => unreachable!(),
        }
    }

    // ── 5. Scale + softmax (over last axis).
    super::scalar::scalar_mul_inplace(stream, &mut scores, scale as f64)?;
    let mut attn: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, seq], &dev)?;
    super::softmax::softmax(stream, &scores, &mut attn)?;

    // ── 6. out_hsd = attn[H,S,S] @ V[H,S,D].
    //   We want axb(attn, V) → [H, S, D], but axb has subtle row/col-major
    //   issues with cuBLAS. Instead, transpose V to [H, D, S] and use axbt:
    //     out[h, sq, d] = sum_sk attn[h, sq, sk] * V[h, sk, d]
    //                   = sum_sk attn[h, sq, sk] * V_t[h, d, sk]
    //                   = (attn[h] @ V_t[h]^T)[sq, d]   ✓
    //   Permute v_hsd [H, S, D] → v_hds [H, D, S].
    let v_hds: Tensor<T, Cuda> = Tensor::zeros([num_heads, head_dim, seq], &dev)?;
    permute_3d::<T>(
        v_hsd.data_ptr(),
        v_hds.data_ptr_mut(),
        [num_heads as i64, seq as i64, head_dim as i64],
        [0, 2, 1],
        stream,
    )?;
    let out_hsd: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
    let stride_attn = (seq * seq) as i64;
    let stride_v_hds = (head_dim * seq) as i64;
    let stride_out = (seq * head_dim) as i64;
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => gemm_strided_batched_bf16_axbt(
                attn.data_ptr() as *const half::bf16,
                v_hds.data_ptr() as *const half::bf16,
                out_hsd.data_ptr_mut() as *mut half::bf16,
                seq as i32,
                head_dim as i32,
                seq as i32,
                stride_attn,
                stride_v_hds,
                stride_out,
                num_heads as i32,
                stream,
            ),
            DataType::F32 => gemm_strided_batched_f32_axbt(
                attn.data_ptr() as *const f32,
                v_hds.data_ptr() as *const f32,
                out_hsd.data_ptr_mut() as *mut f32,
                seq as i32,
                head_dim as i32,
                seq as i32,
                stride_attn,
                stride_v_hds,
                stride_out,
                num_heads as i32,
                stream,
            ),
            _ => unreachable!(),
        }
    }

    // ── 7. Permute [H, S, D] → [S, H, D] back into output.
    permute_3d::<T>(
        out_hsd.data_ptr(),
        output.data_ptr_mut(),
        [num_heads as i64, seq as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;

    let _ = Shape::from_slice(&[seq, num_heads, head_dim]);
    Ok(())
}

/// SDPA with an additive attention mask broadcast across all heads.
/// `mask` is `[seq, seq]` (T-dtype). Same Q/K/V/output layout as `sdpa`.
/// Mask is added to scaled scores before softmax (use `0.0` for attended
/// positions and a large negative number such as `-3.39e38` for masked).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_masked<T: Dtype>(
    stream: cudaStream_t,
    q: &Tensor<T, Cuda>,
    k: &Tensor<T, Cuda>,
    v: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    mask: &Tensor<T, Cuda>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<()> {
    if num_heads % num_kv_heads != 0 {
        return Err(OpError::Kernel(format!(
            "sdpa_masked: n_heads ({}) must be divisible by n_kv_heads ({})",
            num_heads, num_kv_heads,
        )));
    }
    let group = num_heads / num_kv_heads;
    if !matches!(T::DATA_TYPE, DataType::F32 | DataType::BF16) {
        return Err(OpError::Kernel(format!(
            "sdpa_masked: unsupported dtype {:?}",
            T::DATA_TYPE
        )));
    }
    let qs = q.shape().as_slice();
    let ks = k.shape().as_slice();
    let vs = v.shape().as_slice();
    let os = output.shape().as_slice();
    let ms = mask.shape().as_slice();
    if qs.len() != 3 {
        return Err(OpError::Shape(format!(
            "sdpa_masked: Q expected SHD, got {:?}",
            qs
        )));
    }
    let (seq, h_q, d) = (qs[0], qs[1], qs[2]);
    if h_q != num_heads || d != head_dim {
        return Err(OpError::Shape(format!(
            "sdpa_masked: Q shape {:?} doesn't match (n_heads={}, head_dim={})",
            qs, num_heads, head_dim,
        )));
    }
    if ks != [seq, num_kv_heads, head_dim] || vs != [seq, num_kv_heads, head_dim] {
        return Err(OpError::Shape(format!(
            "sdpa_masked: KV shape mismatch k={:?} v={:?} expected=[{}, {}, {}]",
            ks, vs, seq, num_kv_heads, head_dim,
        )));
    }
    if os != qs {
        return Err(OpError::Shape(format!(
            "sdpa_masked: output shape {:?} != Q shape {:?}",
            os, qs
        )));
    }
    if ms != [seq, seq] {
        return Err(OpError::Shape(format!(
            "sdpa_masked: mask shape {:?} expected [{}, {}]",
            ms, seq, seq,
        )));
    }

    let dev = q.device().clone();

    // ── 1. Permute Q [S, H, D] → [H, S, D] ──
    let q_hsd: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
    permute_3d::<T>(
        q.data_ptr(),
        q_hsd.data_ptr_mut(),
        [seq as i64, num_heads as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;
    let k_hsd_kv: Tensor<T, Cuda> = Tensor::zeros([num_kv_heads, seq, head_dim], &dev)?;
    let v_hsd_kv: Tensor<T, Cuda> = Tensor::zeros([num_kv_heads, seq, head_dim], &dev)?;
    permute_3d::<T>(
        k.data_ptr(),
        k_hsd_kv.data_ptr_mut(),
        [seq as i64, num_kv_heads as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;
    permute_3d::<T>(
        v.data_ptr(),
        v_hsd_kv.data_ptr_mut(),
        [seq as i64, num_kv_heads as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;

    // GQA replicate.
    let (k_hsd, v_hsd) = if group == 1 {
        (k_hsd_kv, v_hsd_kv)
    } else {
        let k_full: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
        let v_full: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
        let head_bytes = seq * head_dim * T::SIZE_BYTES;
        unsafe {
            for kv_hi in 0..num_kv_heads {
                for g in 0..group {
                    let dst_hi = kv_hi * group + g;
                    let src_off = kv_hi * seq * head_dim * T::SIZE_BYTES;
                    let dst_off = dst_hi * seq * head_dim * T::SIZE_BYTES;
                    let ksrc = (k_hsd_kv.data_ptr() as *const u8).add(src_off);
                    let kdst = (k_full.data_ptr_mut() as *mut u8).add(dst_off);
                    let vsrc = (v_hsd_kv.data_ptr() as *const u8).add(src_off);
                    let vdst = (v_full.data_ptr_mut() as *mut u8).add(dst_off);
                    let r1 = cudaMemcpyAsync(
                        kdst as *mut _,
                        ksrc as *const _,
                        head_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream,
                    );
                    if r1 != cudaError_cudaSuccess {
                        return Err(OpError::Kernel(format!(
                            "sdpa_masked K replicate: {:?}",
                            r1
                        )));
                    }
                    let r2 = cudaMemcpyAsync(
                        vdst as *mut _,
                        vsrc as *const _,
                        head_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream,
                    );
                    if r2 != cudaError_cudaSuccess {
                        return Err(OpError::Kernel(format!(
                            "sdpa_masked V replicate: {:?}",
                            r2
                        )));
                    }
                }
            }
        }
        (k_full, v_full)
    };

    // scores = Q @ K^T  [H, S, S]
    let mut scores: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, seq], &dev)?;
    let stride_qkv = (seq * head_dim) as i64;
    let stride_scores = (seq * seq) as i64;
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => gemm_strided_batched_bf16_axbt(
                q_hsd.data_ptr() as *const half::bf16,
                k_hsd.data_ptr() as *const half::bf16,
                scores.data_ptr_mut() as *mut half::bf16,
                seq as i32,
                seq as i32,
                head_dim as i32,
                stride_qkv,
                stride_qkv,
                stride_scores,
                num_heads as i32,
                stream,
            ),
            DataType::F32 => gemm_strided_batched_f32_axbt(
                q_hsd.data_ptr() as *const f32,
                k_hsd.data_ptr() as *const f32,
                scores.data_ptr_mut() as *mut f32,
                seq as i32,
                seq as i32,
                head_dim as i32,
                stride_qkv,
                stride_qkv,
                stride_scores,
                num_heads as i32,
                stream,
            ),
            _ => unreachable!(),
        }
    }

    // Scale, then add mask (broadcast across heads), then softmax.
    super::scalar::scalar_mul_inplace(stream, &mut scores, scale as f64)?;
    // scores is [H, S, S] viewed as [H, S*S] with bias [S*S]. Use the
    // existing broadcast_add_inplace which adds bias[j] to x[i, j].
    super::broadcast_mul::broadcast_add_inplace(stream, &mut scores, mask)?;
    let mut attn: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, seq], &dev)?;
    super::softmax::softmax(stream, &scores, &mut attn)?;

    // out = attn @ V (via V permute to [H, D, S] + axbt)
    let v_hds: Tensor<T, Cuda> = Tensor::zeros([num_heads, head_dim, seq], &dev)?;
    permute_3d::<T>(
        v_hsd.data_ptr(),
        v_hds.data_ptr_mut(),
        [num_heads as i64, seq as i64, head_dim as i64],
        [0, 2, 1],
        stream,
    )?;
    let out_hsd: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;
    let stride_attn = (seq * seq) as i64;
    let stride_v_hds = (head_dim * seq) as i64;
    let stride_out = (seq * head_dim) as i64;
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => gemm_strided_batched_bf16_axbt(
                attn.data_ptr() as *const half::bf16,
                v_hds.data_ptr() as *const half::bf16,
                out_hsd.data_ptr_mut() as *mut half::bf16,
                seq as i32,
                head_dim as i32,
                seq as i32,
                stride_attn,
                stride_v_hds,
                stride_out,
                num_heads as i32,
                stream,
            ),
            DataType::F32 => gemm_strided_batched_f32_axbt(
                attn.data_ptr() as *const f32,
                v_hds.data_ptr() as *const f32,
                out_hsd.data_ptr_mut() as *mut f32,
                seq as i32,
                head_dim as i32,
                seq as i32,
                stride_attn,
                stride_v_hds,
                stride_out,
                num_heads as i32,
                stream,
            ),
            _ => unreachable!(),
        }
    }
    permute_3d::<T>(
        out_hsd.data_ptr(),
        output.data_ptr_mut(),
        [num_heads as i64, seq as i64, head_dim as i64],
        [1, 0, 2],
        stream,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    /// CPU reference SDPA on SHD layout (Q) and SHD-kv layout (K, V), F32.
    fn sdpa_cpu_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq: usize,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        scale: f32,
    ) {
        let group = n_heads / n_kv;
        for h in 0..n_heads {
            let kv_h = h / group;
            let mut scores = vec![0.0f32; seq * seq];
            for sq in 0..seq {
                for sk in 0..seq {
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q[(sq * n_heads + h) * head_dim + d]
                            * k[(sk * n_kv + kv_h) * head_dim + d];
                    }
                    scores[sq * seq + sk] = acc * scale;
                }
            }
            for sq in 0..seq {
                let row = &mut scores[sq * seq..(sq + 1) * seq];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - mx).exp();
                    s += *x;
                }
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            for sq in 0..seq {
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for sk in 0..seq {
                        acc += scores[sq * seq + sk] * v[(sk * n_kv + kv_h) * head_dim + d];
                    }
                    out[(sq * n_heads + h) * head_dim + d] = acc;
                }
            }
        }
    }

    #[test]
    fn sdpa_f32_matches_cpu_reference_mha() {
        let cuda = Cuda::new(0).expect("cuda init");
        let (seq, h, d) = (4usize, 2usize, 8usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q_host: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.013).sin()).collect();
        let k_host: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.017).cos()).collect();
        let v_host: Vec<f32> = (0..seq * h * d).map(|i| i as f32 * 0.07 - 0.5).collect();

        let mut ref_out = vec![0.0f32; seq * h * d];
        sdpa_cpu_f32(&q_host, &k_host, &v_host, &mut ref_out, seq, h, h, d, scale);

        let q: Tensor<f32, Cuda> = Tensor::from_host_slice(&q_host, [seq, h, d], &cuda).unwrap();
        let k: Tensor<f32, Cuda> = Tensor::from_host_slice(&k_host, [seq, h, d], &cuda).unwrap();
        let v: Tensor<f32, Cuda> = Tensor::from_host_slice(&v_host, [seq, h, d], &cuda).unwrap();
        let mut out: Tensor<f32, Cuda> = Tensor::zeros([seq, h, d], &cuda).unwrap();
        sdpa(cuda.config.stream, &q, &k, &v, &mut out, h, h, d, scale).unwrap();
        let got = out.to_host_vec().unwrap();
        for (i, (a, b)) in ref_out.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "mha sdpa mismatch at {}: cpu={} gpu={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn sdpa_f32_matches_cpu_reference_gqa() {
        let cuda = Cuda::new(0).unwrap();
        // GQA: 4 query heads, 2 kv heads, group=2.
        let (seq, n_h, n_kv, d) = (4usize, 4usize, 2usize, 8usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q_host: Vec<f32> = (0..seq * n_h * d)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        let k_host: Vec<f32> = (0..seq * n_kv * d)
            .map(|i| (i as f32 * 0.019).cos())
            .collect();
        let v_host: Vec<f32> = (0..seq * n_kv * d).map(|i| i as f32 * 0.05 - 0.3).collect();

        let mut ref_out = vec![0.0f32; seq * n_h * d];
        sdpa_cpu_f32(
            &q_host,
            &k_host,
            &v_host,
            &mut ref_out,
            seq,
            n_h,
            n_kv,
            d,
            scale,
        );

        let q: Tensor<f32, Cuda> = Tensor::from_host_slice(&q_host, [seq, n_h, d], &cuda).unwrap();
        let k: Tensor<f32, Cuda> = Tensor::from_host_slice(&k_host, [seq, n_kv, d], &cuda).unwrap();
        let v: Tensor<f32, Cuda> = Tensor::from_host_slice(&v_host, [seq, n_kv, d], &cuda).unwrap();
        let mut out: Tensor<f32, Cuda> = Tensor::zeros([seq, n_h, d], &cuda).unwrap();
        sdpa(
            cuda.config.stream,
            &q,
            &k,
            &v,
            &mut out,
            n_h,
            n_kv,
            d,
            scale,
        )
        .unwrap();
        let got = out.to_host_vec().unwrap();
        for (i, (a, b)) in ref_out.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "gqa sdpa mismatch at {}: cpu={} gpu={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn sdpa_bf16_matches_cpu_reference() {
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (4usize, 2usize, 8usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q_f32: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.021).sin()).collect();
        let k_f32: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.013).cos()).collect();
        let v_f32: Vec<f32> = (0..seq * h * d).map(|i| i as f32 * 0.05 - 0.3).collect();
        let q_bf16: Vec<bf16> = q_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let k_bf16: Vec<bf16> = k_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let v_bf16: Vec<bf16> = v_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let q_rt: Vec<f32> = q_bf16.iter().map(|x| x.to_f32()).collect();
        let k_rt: Vec<f32> = k_bf16.iter().map(|x| x.to_f32()).collect();
        let v_rt: Vec<f32> = v_bf16.iter().map(|x| x.to_f32()).collect();

        let mut ref_out = vec![0.0f32; seq * h * d];
        sdpa_cpu_f32(&q_rt, &k_rt, &v_rt, &mut ref_out, seq, h, h, d, scale);

        let q: Tensor<bf16, Cuda> = Tensor::from_host_slice(&q_bf16, [seq, h, d], &cuda).unwrap();
        let k: Tensor<bf16, Cuda> = Tensor::from_host_slice(&k_bf16, [seq, h, d], &cuda).unwrap();
        let v: Tensor<bf16, Cuda> = Tensor::from_host_slice(&v_bf16, [seq, h, d], &cuda).unwrap();
        let mut out: Tensor<bf16, Cuda> = Tensor::zeros([seq, h, d], &cuda).unwrap();
        sdpa(cuda.config.stream, &q, &k, &v, &mut out, h, h, d, scale).unwrap();
        let got: Vec<f32> = out
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        for (i, (a, b)) in ref_out.iter().zip(got.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-3);
            assert!(
                abs < 0.1 || rel < 0.05,
                "bf16 sdpa mismatch at {}: cpu={} gpu={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn sdpa_masked_f32_causal_matches_cpu() {
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (8usize, 2usize, 8usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q_host: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.013).sin()).collect();
        let k_host: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.017).cos()).collect();
        let v_host: Vec<f32> = (0..seq * h * d).map(|i| i as f32 * 0.07 - 0.5).collect();

        // Build causal mask host: 0 for j<=i, -INF for j>i.
        let neg = -3.3895313892515355e+38_f32;
        let mut mask_host = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    mask_host[i * seq + j] = neg;
                }
            }
        }

        // CPU reference: scaled scores + mask, then softmax.
        let mut ref_out = vec![0.0f32; seq * h * d];
        for hi in 0..h {
            let mut scores = vec![0.0f32; seq * seq];
            for sq in 0..seq {
                for sk in 0..seq {
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += q_host[(sq * h + hi) * d + dd] * k_host[(sk * h + hi) * d + dd];
                    }
                    scores[sq * seq + sk] = acc * scale + mask_host[sq * seq + sk];
                }
            }
            for sq in 0..seq {
                let row = &mut scores[sq * seq..(sq + 1) * seq];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - mx).exp();
                    s += *x;
                }
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            for sq in 0..seq {
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for sk in 0..seq {
                        acc += scores[sq * seq + sk] * v_host[(sk * h + hi) * d + dd];
                    }
                    ref_out[(sq * h + hi) * d + dd] = acc;
                }
            }
        }

        let q: Tensor<f32, Cuda> = Tensor::from_host_slice(&q_host, [seq, h, d], &cuda).unwrap();
        let k: Tensor<f32, Cuda> = Tensor::from_host_slice(&k_host, [seq, h, d], &cuda).unwrap();
        let v: Tensor<f32, Cuda> = Tensor::from_host_slice(&v_host, [seq, h, d], &cuda).unwrap();
        let m: Tensor<f32, Cuda> = Tensor::from_host_slice(&mask_host, [seq, seq], &cuda).unwrap();
        let mut out: Tensor<f32, Cuda> = Tensor::zeros([seq, h, d], &cuda).unwrap();
        sdpa_masked(cuda.config.stream, &q, &k, &v, &mut out, &m, h, h, d, scale).unwrap();
        let got = out.to_host_vec().unwrap();
        for (i, (a, b)) in ref_out.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 5e-3,
                "masked sdpa f32 mismatch at {}: cpu={} gpu={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn sdpa_masked_bf16_causal_padding_matches_cpu() {
        // Verify causal+padding mask. seq=8 with last 3 positions padded.
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (8usize, 4usize, 16usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q_f32: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.011).sin()).collect();
        let k_f32: Vec<f32> = (0..seq * h * d).map(|i| (i as f32 * 0.019).cos()).collect();
        let v_f32: Vec<f32> = (0..seq * h * d).map(|i| i as f32 * 0.05 - 0.3).collect();
        let neg = -3.3895313892515355e+38_f32;

        // mask: causal AND attention_mask[j]==1 for j in 0..5
        let am = [1, 1, 1, 1, 1, 0, 0, 0];
        let mut mask_host = vec![neg; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j <= i && am[j] == 1 {
                    mask_host[i * seq + j] = 0.0;
                }
            }
        }
        let q_bf: Vec<half::bf16> = q_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let k_bf: Vec<half::bf16> = k_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let v_bf: Vec<half::bf16> = v_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let mask_bf: Vec<half::bf16> = mask_host.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let q_rt: Vec<f32> = q_bf.iter().map(|x| x.to_f32()).collect();
        let k_rt: Vec<f32> = k_bf.iter().map(|x| x.to_f32()).collect();
        let v_rt: Vec<f32> = v_bf.iter().map(|x| x.to_f32()).collect();

        // CPU reference.
        let mut ref_out = vec![0.0f32; seq * h * d];
        for hi in 0..h {
            let mut scores = vec![0.0f32; seq * seq];
            for sq in 0..seq {
                for sk in 0..seq {
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += q_rt[(sq * h + hi) * d + dd] * k_rt[(sk * h + hi) * d + dd];
                    }
                    scores[sq * seq + sk] = acc * scale + mask_host[sq * seq + sk];
                }
            }
            for sq in 0..seq {
                let row = &mut scores[sq * seq..(sq + 1) * seq];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - mx).exp();
                    s += *x;
                }
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            for sq in 0..seq {
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for sk in 0..seq {
                        acc += scores[sq * seq + sk] * v_rt[(sk * h + hi) * d + dd];
                    }
                    ref_out[(sq * h + hi) * d + dd] = acc;
                }
            }
        }

        let q: Tensor<half::bf16, Cuda> =
            Tensor::from_host_slice(&q_bf, [seq, h, d], &cuda).unwrap();
        let k: Tensor<half::bf16, Cuda> =
            Tensor::from_host_slice(&k_bf, [seq, h, d], &cuda).unwrap();
        let v: Tensor<half::bf16, Cuda> =
            Tensor::from_host_slice(&v_bf, [seq, h, d], &cuda).unwrap();
        let m: Tensor<half::bf16, Cuda> =
            Tensor::from_host_slice(&mask_bf, [seq, seq], &cuda).unwrap();
        let mut out: Tensor<half::bf16, Cuda> = Tensor::zeros([seq, h, d], &cuda).unwrap();
        sdpa_masked(cuda.config.stream, &q, &k, &v, &mut out, &m, h, h, d, scale).unwrap();
        let got: Vec<f32> = out
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        for (i, (a, b)) in ref_out.iter().zip(got.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-3);
            assert!(
                abs < 0.1 || rel < 0.06,
                "masked bf16 sdpa mismatch at {}: cpu={} gpu={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn batched_axbt_bf16_isolated() {
        // Test the BF16 strided batched gemm A @ B^T directly with a known
        // input/output. batch=2, M=3, N=4, K=5.
        let cuda = Cuda::new(0).unwrap();
        let (batch, m, n, k) = (2usize, 8usize, 16usize, 8usize);
        let a_host: Vec<bf16> = (0..batch * m * k)
            .map(|i| bf16::from_f32(i as f32 * 0.01))
            .collect();
        let b_host: Vec<bf16> = (0..batch * n * k)
            .map(|i| bf16::from_f32(i as f32 * 0.02))
            .collect();
        let a: Tensor<bf16, Cuda> = Tensor::from_host_slice(&a_host, [batch, m, k], &cuda).unwrap();
        let b: Tensor<bf16, Cuda> = Tensor::from_host_slice(&b_host, [batch, n, k], &cuda).unwrap();
        let c: Tensor<bf16, Cuda> = Tensor::zeros([batch, m, n], &cuda).unwrap();
        let stride_a = (m * k) as i64;
        let stride_b = (n * k) as i64;
        let stride_c = (m * n) as i64;
        unsafe {
            super::gemm_strided_batched_bf16_axbt(
                a.data_ptr(),
                b.data_ptr(),
                c.data_ptr_mut(),
                m as i32,
                n as i32,
                k as i32,
                stride_a,
                stride_b,
                stride_c,
                batch as i32,
                cuda.config.stream,
            );
        }
        let got: Vec<f32> = c
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        // CPU ref: c[b, i, j] = sum_k a[b, i, k] * b[b, j, k]
        let mut ref_c = vec![0.0_f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0_f32;
                    for kk in 0..k {
                        acc += a_host[(bi * m + i) * k + kk].to_f32()
                            * b_host[(bi * n + j) * k + kk].to_f32();
                    }
                    ref_c[(bi * m + i) * n + j] = acc;
                }
            }
        }
        for (i, (a, b)) in ref_c.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.5,
                "axbt mismatch at {}: cpu={} gpu={}",
                i,
                a,
                b
            );
        }
    }
}
