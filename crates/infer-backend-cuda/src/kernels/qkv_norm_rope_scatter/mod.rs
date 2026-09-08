//! Fused Qwen3 Q/K RMSNorm + RoPE + paged K/V scatter CUDA wrapper.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
    fn qkv_norm_rope_scatter_bf16(
        q: *mut half::bf16,
        k: *mut half::bf16,
        v: *const half::bf16,
        q_weight: *const half::bf16,
        k_weight: *const half::bf16,
        sin_cache: *const half::bf16,
        cos_cache: *const half::bf16,
        positions: *const i32,
        k_pool: *mut half::bf16,
        v_pool: *mut half::bf16,
        block_tables: *const u32,
        seq_positions: *const i32,
        seq_starts: *const i32,
        seq_lens: *const i32,
        num_tokens: i32,
        batch: i32,
        head_num: i32,
        kv_head_num: i32,
        head_dim: i32,
        rotary_dim: i32,
        kv_dim: i32,
        q_row_stride: i64,
        k_row_stride: i64,
        v_row_stride: i64,
        max_blocks_per_seq: i32,
        block_size: i32,
        q_eps: f32,
        k_eps: f32,
        stream: cudaStream_t,
    );
}

pub fn qkv_norm_rope_scatter<T: Dtype>(
    stream: cudaStream_t,
    q: &mut Tensor<T, Cuda>,
    k: &mut Tensor<T, Cuda>,
    v: &Tensor<T, Cuda>,
    q_weight: Option<&Tensor<T, Cuda>>,
    k_weight: Option<&Tensor<T, Cuda>>,
    q_eps: f32,
    k_eps: f32,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions: &Tensor<i32, Cuda>,
    k_pool: &mut Tensor<T, Cuda>,
    v_pool: &mut Tensor<T, Cuda>,
    block_tables: &Tensor<i32, Cuda>,
    seq_positions: &Tensor<i32, Cuda>,
    cu_q_lens: &Tensor<i32, Cuda>,
    seq_lens_step: &Tensor<i32, Cuda>,
    max_blocks_per_seq: usize,
    block_size: usize,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    rotary_dim: usize,
    kv_dim: usize,
) -> OpResult<()> {
    let Some(q_weight) = q_weight else {
        return Err(OpError::Kernel(
            "qkv_norm_rope_scatter: missing q_norm weight".into(),
        ));
    };
    let Some(k_weight) = k_weight else {
        return Err(OpError::Kernel(
            "qkv_norm_rope_scatter: missing k_norm weight".into(),
        ));
    };
    let seq_positions_shape = seq_positions.shape().as_slice();
    if seq_positions_shape.len() != 1 || !seq_positions.is_contiguous() {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: seq_positions must be contiguous rank 1, got {seq_positions_shape:?}"
        )));
    }
    let batch = seq_positions_shape[0];
    if batch == 0 || positions.numel() == 0 {
        return Ok(());
    }
    if T::DATA_TYPE != DataType::BF16 {
        return Err(OpError::Kernel(format!(
            "qkv_norm_rope_scatter: unsupported dtype {:?}",
            T::DATA_TYPE
        )));
    }
    if head_num == 0
        || kv_head_num == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim / 2 > 1024
        || head_num.max(kv_head_num) > 32
    {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: invalid head_num={head_num} kv_head_num={kv_head_num} head_dim={head_dim} rotary_dim={rotary_dim}"
        )));
    }
    let q_dim = head_num
        .checked_mul(head_dim)
        .ok_or_else(|| OpError::Shape("qkv_norm_rope_scatter: q width overflows".into()))?;
    let expected_kv_dim = kv_head_num
        .checked_mul(head_dim)
        .ok_or_else(|| OpError::Shape("qkv_norm_rope_scatter: kv width overflows".into()))?;
    if kv_dim != expected_kv_dim {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: kv_dim {kv_dim} != kv_head_num * head_dim {expected_kv_dim}"
        )));
    }
    let q_shape = q.shape().as_slice();
    if q_shape.len() != 2 || q_shape[0] == 0 || q_shape[1] != q_dim {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: q must be non-empty [num_tokens, {q_dim}], got {q_shape:?}"
        )));
    }
    let num_tokens_usize = q_shape[0];
    for (name, shape, strides, width) in [
        ("k", k.shape(), k.strides(), kv_dim),
        ("v", v.shape(), v.strides(), kv_dim),
    ] {
        if shape.as_slice() != [num_tokens_usize, width] || strides.as_slice()[1] != 1 {
            return Err(OpError::Shape(format!(
                "qkv_norm_rope_scatter: {name} must be [{num_tokens_usize}, {width}] with packed columns, got shape {:?}, strides {:?}",
                shape.as_slice(),
                strides.as_slice()
            )));
        }
    }
    if q.strides().as_slice()[1] != 1 {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: q columns must be packed, got strides {:?}",
            q.strides().as_slice()
        )));
    }
    if q_weight.shape().as_slice() != [head_dim]
        || k_weight.shape().as_slice() != [head_dim]
        || !q_weight.is_contiguous()
        || !k_weight.is_contiguous()
    {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: q/k norm weights must be contiguous [{head_dim}], got {:?}/{:?}",
            q_weight.shape().as_slice(),
            k_weight.shape().as_slice()
        )));
    }
    if sin.shape().as_slice() != cos.shape().as_slice()
        || sin.ndim() != 2
        || sin.shape().as_slice()[0] == 0
        || sin.shape().as_slice()[1] != rotary_dim / 2
        || !sin.is_contiguous()
        || !cos.is_contiguous()
    {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: sin/cos must be matching contiguous [max_position, {}], got {:?}/{:?}",
            rotary_dim / 2,
            sin.shape().as_slice(),
            cos.shape().as_slice()
        )));
    }
    if !positions.is_contiguous() || positions.numel() < num_tokens_usize {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: positions must have at least {num_tokens_usize} contiguous entries, got {:?}",
            positions.shape().as_slice()
        )));
    }
    if !q_eps.is_finite() || q_eps < 0.0 || !k_eps.is_finite() || k_eps < 0.0 {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: eps must be finite and non-negative, got q={q_eps}, k={k_eps}"
        )));
    }
    if max_blocks_per_seq == 0 || block_size == 0 {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: max_blocks_per_seq/block_size must be non-zero, got {max_blocks_per_seq}/{block_size}"
        )));
    }
    let block_shape = block_tables.shape().as_slice();
    if block_shape.len() != 2
        || block_shape[0] < batch
        || block_shape[1] < max_blocks_per_seq
        || !block_tables.is_contiguous()
    {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: block_tables must cover [{batch}, {max_blocks_per_seq}], got shape {block_shape:?}"
        )));
    }
    for (name, tensor) in [("seq_starts", cu_q_lens), ("seq_lens", seq_lens_step)] {
        if tensor.ndim() != 1 || tensor.numel() < batch || !tensor.is_contiguous() {
            return Err(OpError::Shape(format!(
                "qkv_norm_rope_scatter: {name} must be contiguous rank 1 with at least {batch} entries, got {:?}",
                tensor.shape().as_slice()
            )));
        }
    }
    let pool_shape = k_pool.shape().as_slice();
    if pool_shape.len() != 3
        || pool_shape != v_pool.shape().as_slice()
        || pool_shape[0] == 0
        || pool_shape[1] != block_size
        || pool_shape[2] != kv_dim
        || !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
    {
        return Err(OpError::Shape(format!(
            "qkv_norm_rope_scatter: k/v pools must be matching contiguous [num_blocks, {block_size}, {kv_dim}], got {:?}/{:?}",
            k_pool.shape().as_slice(),
            v_pool.shape().as_slice()
        )));
    }

    let q_row_stride = i64::try_from(q.strides().as_slice()[0])
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: q row stride exceeds i64".into()))?;
    let k_row_stride = i64::try_from(k.strides().as_slice()[0])
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: k row stride exceeds i64".into()))?;
    let v_row_stride = i64::try_from(v.strides().as_slice()[0])
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: v row stride exceeds i64".into()))?;

    // Actual token-row count = rows of the q/k/v buffers. The kernel's
    // `row < num_tokens` bounds guard MUST use this, NOT `positions.numel()`:
    // `rope_positions` is allocated at capacity (`cap_num_tokens`) and only
    // prefix-filled, so its numel() is the worst-case (e.g. 8192), which makes
    // the guard ineffective and lets the kernel read past q/k/v -> OOB reads,
    // corrupted Q, and non-deterministic garbage for batch > 1.
    let num_tokens = i32::try_from(num_tokens_usize)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: num_tokens exceeds i32".into()))?;
    let batch = i32::try_from(batch)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: batch exceeds i32".into()))?;
    let head_num = i32::try_from(head_num)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: head_num exceeds i32".into()))?;
    let kv_head_num = i32::try_from(kv_head_num)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: kv_head_num exceeds i32".into()))?;
    let head_dim = i32::try_from(head_dim)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: head_dim exceeds i32".into()))?;
    let rotary_dim = i32::try_from(rotary_dim)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: rotary_dim exceeds i32".into()))?;
    let kv_dim = i32::try_from(kv_dim)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: kv_dim exceeds i32".into()))?;
    let max_blocks_per_seq = i32::try_from(max_blocks_per_seq).map_err(|_| {
        OpError::Shape("qkv_norm_rope_scatter: max_blocks_per_seq exceeds i32".into())
    })?;
    let block_size = i32::try_from(block_size)
        .map_err(|_| OpError::Shape("qkv_norm_rope_scatter: block_size exceeds i32".into()))?;

    unsafe {
        qkv_norm_rope_scatter_bf16(
            q.data_ptr_mut() as _,
            k.data_ptr_mut() as _,
            v.data_ptr() as _,
            q_weight.data_ptr() as _,
            k_weight.data_ptr() as _,
            sin.data_ptr() as _,
            cos.data_ptr() as _,
            positions.data_ptr(),
            k_pool.data_ptr_mut() as _,
            v_pool.data_ptr_mut() as _,
            block_tables.data_ptr() as *const u32,
            seq_positions.data_ptr(),
            cu_q_lens.data_ptr(),
            seq_lens_step.data_ptr(),
            num_tokens,
            batch,
            head_num,
            kv_head_num,
            head_dim,
            rotary_dim,
            kv_dim,
            q_row_stride,
            k_row_stride,
            v_row_stride,
            max_blocks_per_seq,
            block_size,
            q_eps,
            k_eps,
            stream,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn norm_rope_head(
        input: &[bf16],
        weight: &[bf16],
        sin: &[bf16],
        cos: &[bf16],
        position: usize,
        head_dim: usize,
        rotary_dim: usize,
        eps: f32,
    ) -> Vec<bf16> {
        let mut square_sum = 0.0f32;
        for &value in input {
            let value = value.to_f32();
            square_sum = value.mul_add(value, square_sum);
        }
        let inv_rms = (square_sum / head_dim as f32 + eps).sqrt().recip();
        let mut output = input
            .iter()
            .zip(weight)
            .map(|(&value, &weight)| bf16::from_f32(value.to_f32() * weight.to_f32() * inv_rms))
            .collect::<Vec<_>>();
        let half_rotary = rotary_dim / 2;
        for pair in 0..half_rotary {
            let left = output[pair].to_f32();
            let right = output[pair + half_rotary].to_f32();
            let cache = position * half_rotary + pair;
            let sin_value = sin[cache].to_f32();
            let cos_value = cos[cache].to_f32();
            output[pair] = bf16::from_f32(left * cos_value - right * sin_value);
            output[pair + half_rotary] = bf16::from_f32(left * sin_value + right * cos_value);
        }
        output
    }

    #[test]
    fn bf16_qwen35_256x64_partial_rope_normalizes_tail_and_scatters_gqa_kv() {
        let cuda = Cuda::new(0).expect("cuda init");
        let (tokens, batch) = (3usize, 2usize);
        let (head_num, kv_head_num, head_dim, rotary_dim) = (16usize, 4usize, 256usize, 64usize);
        let (q_dim, kv_dim) = (head_num * head_dim, kv_head_num * head_dim);
        let (prefix, suffix) = (3usize, 5usize);
        let row_pitch = prefix + q_dim + 2 * kv_dim + suffix;
        let sentinel = bf16::from_f32(-9.0);

        let mut packed = vec![sentinel; tokens * row_pitch];
        let mut q_input = vec![bf16::from_f32(0.0); tokens * q_dim];
        let mut k_input = vec![bf16::from_f32(0.0); tokens * kv_dim];
        let mut v_input = vec![bf16::from_f32(0.0); tokens * kv_dim];
        for token in 0..tokens {
            for col in 0..q_dim {
                let logical = token * q_dim + col;
                let value = bf16::from_f32(
                    ((logical as f32 * 0.007).sin() + 0.21 * (logical as f32 * 0.013).cos()) * 1.2,
                );
                q_input[logical] = value;
                packed[token * row_pitch + prefix + col] = value;
            }
            for col in 0..kv_dim {
                let logical = token * kv_dim + col;
                let key = bf16::from_f32(
                    ((logical as f32 * 0.011).cos() - 0.17 * (logical as f32 * 0.019).sin()) * 0.9,
                );
                let value = bf16::from_f32(((logical as f32 * 0.017).sin() - 0.3) * 0.8);
                k_input[logical] = key;
                v_input[logical] = value;
                packed[token * row_pitch + prefix + q_dim + col] = key;
                packed[token * row_pitch + prefix + q_dim + kv_dim + col] = value;
            }
        }
        let q_weight = (0..head_dim)
            .map(|index| bf16::from_f32(0.75 + 0.3 * (index as f32 * 0.031).cos()))
            .collect::<Vec<_>>();
        let k_weight = (0..head_dim)
            .map(|index| bf16::from_f32(0.8 + 0.25 * (index as f32 * 0.027).sin()))
            .collect::<Vec<_>>();
        let max_position = 6usize;
        let half_rotary = rotary_dim / 2;
        let mut sin = vec![bf16::from_f32(0.0); max_position * half_rotary];
        let mut cos = vec![bf16::from_f32(0.0); max_position * half_rotary];
        for position in 0..max_position {
            for pair in 0..half_rotary {
                let angle = position as f32 * (0.009 + pair as f32 * 0.0017);
                sin[position * half_rotary + pair] = bf16::from_f32(angle.sin());
                cos[position * half_rotary + pair] = bf16::from_f32(angle.cos());
            }
        }
        let positions = vec![1i32, 2, 5];
        let q_eps = 1e-6;
        let k_eps = 2e-6;

        let mut expected_q = vec![bf16::from_f32(0.0); tokens * q_dim];
        let mut expected_k = vec![bf16::from_f32(0.0); tokens * kv_dim];
        for token in 0..tokens {
            for head in 0..head_num {
                let base = token * q_dim + head * head_dim;
                let expected = norm_rope_head(
                    &q_input[base..base + head_dim],
                    &q_weight,
                    &sin,
                    &cos,
                    positions[token] as usize,
                    head_dim,
                    rotary_dim,
                    q_eps,
                );
                expected_q[base..base + head_dim].copy_from_slice(&expected);
            }
            for head in 0..kv_head_num {
                let base = token * kv_dim + head * head_dim;
                let expected = norm_rope_head(
                    &k_input[base..base + head_dim],
                    &k_weight,
                    &sin,
                    &cos,
                    positions[token] as usize,
                    head_dim,
                    rotary_dim,
                    k_eps,
                );
                expected_k[base..base + head_dim].copy_from_slice(&expected);
            }
        }

        let packed_dev = Tensor::from_host_slice(&packed, [tokens, row_pitch], &cuda).unwrap();
        let mut q = packed_dev.narrow(1, prefix, q_dim).unwrap();
        let mut k = packed_dev.narrow(1, prefix + q_dim, kv_dim).unwrap();
        let v = packed_dev
            .narrow(1, prefix + q_dim + kv_dim, kv_dim)
            .unwrap();
        let q_weight_dev = Tensor::from_host_slice(&q_weight, [head_dim], &cuda).unwrap();
        let k_weight_dev = Tensor::from_host_slice(&k_weight, [head_dim], &cuda).unwrap();
        let sin_dev = Tensor::from_host_slice(&sin, [max_position, half_rotary], &cuda).unwrap();
        let cos_dev = Tensor::from_host_slice(&cos, [max_position, half_rotary], &cuda).unwrap();
        let positions_dev = Tensor::from_host_slice(&positions, [tokens], &cuda).unwrap();

        let block_size = 4usize;
        let max_blocks_per_seq = 2usize;
        let num_blocks = 2usize;
        let block_tables =
            Tensor::from_host_slice(&[1i32, 0, 1, 0], [batch, max_blocks_per_seq], &cuda).unwrap();
        let seq_positions = Tensor::from_host_slice(&[0i32, 4], [batch], &cuda).unwrap();
        let cu_q_lens = Tensor::from_host_slice(&[0i32, 2], [batch], &cuda).unwrap();
        let seq_lens = Tensor::from_host_slice(&[2i32, 1], [batch], &cuda).unwrap();
        let mut k_pool = Tensor::from_host_slice(
            &vec![sentinel; num_blocks * block_size * kv_dim],
            [num_blocks, block_size, kv_dim],
            &cuda,
        )
        .unwrap();
        let mut v_pool = Tensor::from_host_slice(
            &vec![sentinel; num_blocks * block_size * kv_dim],
            [num_blocks, block_size, kv_dim],
            &cuda,
        )
        .unwrap();

        qkv_norm_rope_scatter(
            cuda.config.stream,
            &mut q,
            &mut k,
            &v,
            Some(&q_weight_dev),
            Some(&k_weight_dev),
            q_eps,
            k_eps,
            &sin_dev,
            &cos_dev,
            &positions_dev,
            &mut k_pool,
            &mut v_pool,
            &block_tables,
            &seq_positions,
            &cu_q_lens,
            &seq_lens,
            max_blocks_per_seq,
            block_size,
            head_num,
            kv_head_num,
            head_dim,
            rotary_dim,
            kv_dim,
        )
        .unwrap();

        let packed_got = packed_dev.to_host_vec().unwrap();
        for token in 0..tokens {
            for col in 0..q_dim {
                let got = packed_got[token * row_pitch + prefix + col].to_f32();
                let expected = expected_q[token * q_dim + col].to_f32();
                assert!(
                    (got - expected).abs() < 0.035,
                    "q mismatch token={token} col={col}: got={got}, expected={expected}"
                );
            }
            // The fused path reads K and scatters its transformed value without
            // mutating the zero-copy K projection view.
            for col in 0..kv_dim {
                assert_eq!(
                    packed_got[token * row_pitch + prefix + q_dim + col].to_bits(),
                    k_input[token * kv_dim + col].to_bits()
                );
            }
            for col in 0..prefix {
                assert_eq!(
                    packed_got[token * row_pitch + col].to_bits(),
                    sentinel.to_bits()
                );
            }
            for col in row_pitch - suffix..row_pitch {
                assert_eq!(
                    packed_got[token * row_pitch + col].to_bits(),
                    sentinel.to_bits()
                );
            }
        }

        let k_pool = k_pool.to_host_vec().unwrap();
        let v_pool = v_pool.to_host_vec().unwrap();
        // token -> physical pool slot: seq0 rows use block 1 offsets 0/1;
        // seq1 starts at logical position 4 (block-table entry 1 -> block 0).
        let pool_slots = [4usize, 5usize, 0usize];
        for token in 0..tokens {
            let pool_base = pool_slots[token] * kv_dim;
            for col in 0..kv_dim {
                let got_k = k_pool[pool_base + col].to_f32();
                let expected = expected_k[token * kv_dim + col].to_f32();
                assert!(
                    (got_k - expected).abs() < 0.035,
                    "k pool mismatch token={token} col={col}: got={got_k}, expected={expected}"
                );
                assert_eq!(
                    v_pool[pool_base + col].to_bits(),
                    v_input[token * kv_dim + col].to_bits(),
                    "v pool mismatch token={token} col={col}"
                );
            }
        }
        for slot in [1usize, 2, 3, 6, 7] {
            let range = slot * kv_dim..(slot + 1) * kv_dim;
            assert!(
                k_pool[range.clone()]
                    .iter()
                    .all(|value| value.to_bits() == sentinel.to_bits())
            );
            assert!(
                v_pool[range]
                    .iter()
                    .all(|value| value.to_bits() == sentinel.to_bits())
            );
        }
    }

    #[test]
    fn bf16_full_rope_head128_preserves_existing_optimized_path() {
        let cuda = Cuda::new(0).expect("cuda init");
        let (head_num, kv_head_num, head_dim, rotary_dim) = (2usize, 1usize, 128usize, 128usize);
        let (q_dim, kv_dim) = (head_num * head_dim, kv_head_num * head_dim);
        let q_input = (0..q_dim)
            .map(|index| bf16::from_f32(((index as f32 * 0.031).sin() - 0.2) * 1.1))
            .collect::<Vec<_>>();
        let k_input = (0..kv_dim)
            .map(|index| bf16::from_f32(((index as f32 * 0.043).cos() + 0.1) * 0.8))
            .collect::<Vec<_>>();
        let v_input = (0..kv_dim)
            .map(|index| bf16::from_f32((index as f32 * 0.021).sin()))
            .collect::<Vec<_>>();
        let q_weight = (0..head_dim)
            .map(|index| bf16::from_f32(0.8 + index as f32 * 0.001))
            .collect::<Vec<_>>();
        let k_weight = (0..head_dim)
            .map(|index| bf16::from_f32(1.05 - index as f32 * 0.0015))
            .collect::<Vec<_>>();
        let mut sin = vec![bf16::from_f32(0.0); rotary_dim / 2];
        let mut cos = vec![bf16::from_f32(0.0); rotary_dim / 2];
        for pair in 0..rotary_dim / 2 {
            let angle = 0.005 * (pair + 1) as f32;
            sin[pair] = bf16::from_f32(angle.sin());
            cos[pair] = bf16::from_f32(angle.cos());
        }
        let q_eps = 1e-6;
        let k_eps = 1e-6;
        let mut expected_q = vec![bf16::from_f32(0.0); q_dim];
        for head in 0..head_num {
            let base = head * head_dim;
            expected_q[base..base + head_dim].copy_from_slice(&norm_rope_head(
                &q_input[base..base + head_dim],
                &q_weight,
                &sin,
                &cos,
                0,
                head_dim,
                rotary_dim,
                q_eps,
            ));
        }
        let expected_k = norm_rope_head(
            &k_input, &k_weight, &sin, &cos, 0, head_dim, rotary_dim, k_eps,
        );

        let mut q = Tensor::from_host_slice(&q_input, [1, q_dim], &cuda).unwrap();
        let mut k = Tensor::from_host_slice(&k_input, [1, kv_dim], &cuda).unwrap();
        let v = Tensor::from_host_slice(&v_input, [1, kv_dim], &cuda).unwrap();
        let q_weight = Tensor::from_host_slice(&q_weight, [head_dim], &cuda).unwrap();
        let k_weight = Tensor::from_host_slice(&k_weight, [head_dim], &cuda).unwrap();
        let sin = Tensor::from_host_slice(&sin, [1, rotary_dim / 2], &cuda).unwrap();
        let cos = Tensor::from_host_slice(&cos, [1, rotary_dim / 2], &cuda).unwrap();
        let positions = Tensor::from_host_slice(&[0i32], [1], &cuda).unwrap();
        let block_tables = Tensor::from_host_slice(&[0i32], [1, 1], &cuda).unwrap();
        let seq_positions = Tensor::from_host_slice(&[0i32], [1], &cuda).unwrap();
        let seq_starts = Tensor::from_host_slice(&[0i32], [1], &cuda).unwrap();
        let seq_lens = Tensor::from_host_slice(&[1i32], [1], &cuda).unwrap();
        let mut k_pool = Tensor::<bf16, Cuda>::zeros([1, 1, kv_dim], &cuda).unwrap();
        let mut v_pool = Tensor::<bf16, Cuda>::zeros([1, 1, kv_dim], &cuda).unwrap();

        qkv_norm_rope_scatter(
            cuda.config.stream,
            &mut q,
            &mut k,
            &v,
            Some(&q_weight),
            Some(&k_weight),
            q_eps,
            k_eps,
            &sin,
            &cos,
            &positions,
            &mut k_pool,
            &mut v_pool,
            &block_tables,
            &seq_positions,
            &seq_starts,
            &seq_lens,
            1,
            1,
            head_num,
            kv_head_num,
            head_dim,
            rotary_dim,
            kv_dim,
        )
        .unwrap();

        for (index, (&got, &expected)) in
            q.to_host_vec().unwrap().iter().zip(&expected_q).enumerate()
        {
            assert!(
                (got.to_f32() - expected.to_f32()).abs() < 0.025,
                "q mismatch at {index}: got={}, expected={}",
                got.to_f32(),
                expected.to_f32()
            );
        }
        for (index, (&got, &expected)) in k_pool
            .to_host_vec()
            .unwrap()
            .iter()
            .zip(&expected_k)
            .enumerate()
        {
            assert!(
                (got.to_f32() - expected.to_f32()).abs() < 0.025,
                "k mismatch at {index}: got={}, expected={}",
                got.to_f32(),
                expected.to_f32()
            );
        }
        assert_eq!(
            v_pool
                .to_host_vec()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            v_input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
