//! Sequential Gated DeltaNet recurrence CUDA launch wrapper.
//!
//! The operator owns no cache. Its fp32 recurrent state is supplied by the
//! caller and updated in place. The v1 kernel deliberately loops over tokens
//! sequentially inside one block per `(sequence, value_head)`; this mirrors the
//! recurrence directly and provides the correctness baseline for a later
//! chunked-scan optimization.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn gated_delta_rule_f32_forward(
        query: *const f32,
        key: *const f32,
        value: *const f32,
        a: *const f32,
        b: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        recurrent_state: *mut f32,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut f32,
        num_tokens: i32,
        batch: i32,
        num_key_heads: i32,
        num_value_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
        num_slots: i32,
        query_row_stride: i64,
        key_row_stride: i64,
        value_row_stride: i64,
        a_row_stride: i64,
        b_row_stride: i64,
        stream: cudaStream_t,
    );
    fn gated_delta_rule_bf16_forward(
        query: *const half::bf16,
        key: *const half::bf16,
        value: *const half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        a_log: *const f32,
        dt_bias: *const half::bf16,
        recurrent_state: *mut f32,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut half::bf16,
        num_tokens: i32,
        batch: i32,
        num_key_heads: i32,
        num_value_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
        num_slots: i32,
        query_row_stride: i64,
        key_row_stride: i64,
        value_row_stride: i64,
        a_row_stride: i64,
        b_row_stride: i64,
        stream: cudaStream_t,
    );
    fn gated_delta_rule_f16_forward(
        query: *const half::f16,
        key: *const half::f16,
        value: *const half::f16,
        a: *const half::f16,
        b: *const half::f16,
        a_log: *const f32,
        dt_bias: *const half::f16,
        recurrent_state: *mut f32,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut half::f16,
        num_tokens: i32,
        batch: i32,
        num_key_heads: i32,
        num_value_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
        num_slots: i32,
        query_row_stride: i64,
        key_row_stride: i64,
        value_row_stride: i64,
        a_row_stride: i64,
        b_row_stride: i64,
        stream: cudaStream_t,
    );
}

/// Element types with a sequential gated-delta CUDA kernel.
///
/// # Safety
/// All pointers and strides must describe the validated device tensors,
/// `state_slots` must contain distinct in-range values, and `cu_seqlens` must
/// be monotonic from zero to `num_tokens`.
pub trait GatedDeltaRuleKernel: CudaFloat {
    #[allow(clippy::too_many_arguments)]
    unsafe fn gated_delta_rule(
        query: *const Self,
        key: *const Self,
        value: *const Self,
        a: *const Self,
        b: *const Self,
        a_log: *const f32,
        dt_bias: *const Self,
        recurrent_state: *mut f32,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut Self,
        num_tokens: i32,
        batch: i32,
        num_key_heads: i32,
        num_value_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
        num_slots: i32,
        query_row_stride: i64,
        key_row_stride: i64,
        value_row_stride: i64,
        a_row_stride: i64,
        b_row_stride: i64,
        stream: cudaStream_t,
    );
}

macro_rules! impl_gated_delta_kernel {
    ($ty:ty, $entry:ident) => {
        impl GatedDeltaRuleKernel for $ty {
            #[inline]
            unsafe fn gated_delta_rule(
                query: *const Self,
                key: *const Self,
                value: *const Self,
                a: *const Self,
                b: *const Self,
                a_log: *const f32,
                dt_bias: *const Self,
                recurrent_state: *mut f32,
                state_slots: *const i32,
                cu_seqlens: *const i32,
                output: *mut Self,
                num_tokens: i32,
                batch: i32,
                num_key_heads: i32,
                num_value_heads: i32,
                key_head_dim: i32,
                value_head_dim: i32,
                num_slots: i32,
                query_row_stride: i64,
                key_row_stride: i64,
                value_row_stride: i64,
                a_row_stride: i64,
                b_row_stride: i64,
                stream: cudaStream_t,
            ) {
                unsafe {
                    $entry(
                        query,
                        key,
                        value,
                        a,
                        b,
                        a_log,
                        dt_bias,
                        recurrent_state,
                        state_slots,
                        cu_seqlens,
                        output,
                        num_tokens,
                        batch,
                        num_key_heads,
                        num_value_heads,
                        key_head_dim,
                        value_head_dim,
                        num_slots,
                        query_row_stride,
                        key_row_stride,
                        value_row_stride,
                        a_row_stride,
                        b_row_stride,
                        stream,
                    )
                }
            }
        }
    };
}

impl_gated_delta_kernel!(f32, gated_delta_rule_f32_forward);
impl_gated_delta_kernel!(half::bf16, gated_delta_rule_bf16_forward);
impl_gated_delta_kernel!(half::f16, gated_delta_rule_f16_forward);

fn checked_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value)
        .map_err(|_| OpError::Shape(format!("gated_delta_rule: {name} exceeds i32")))
}

fn packed_row_stride<T: infer_core::dtype::Dtype>(
    tensor: &Tensor<T, Cuda>,
    rows: usize,
    cols: usize,
    name: &str,
) -> OpResult<i64> {
    if tensor.shape().as_slice() != [rows, cols] {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: {name} shape {:?} != [{rows}, {cols}]",
            tensor.shape().as_slice()
        )));
    }
    let strides = tensor.strides().as_slice();
    if strides.len() != 2 || strides[1] != 1 || (rows > 1 && strides[0] < cols) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: {name} must have packed columns, got strides {:?}",
            strides
        )));
    }
    i64::try_from(strides[0])
        .map_err(|_| OpError::Shape(format!("gated_delta_rule: {name} row stride exceeds i64")))
}

/// Run the sequential Gated DeltaNet recurrence over a flattened ragged tape.
///
/// Q/K L2 normalization, query scaling, GQA repeat mapping, beta sigmoid, and
/// log-decay construction from `a`, `a_log`, and `dt_bias` are fused into the
/// launch. The kernel allocates no memory and keeps no state of its own.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule<T: GatedDeltaRuleKernel>(
    stream: cudaStream_t,
    query: &Tensor<T, Cuda>,
    key: &Tensor<T, Cuda>,
    value: &Tensor<T, Cuda>,
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    a_log: &Tensor<f32, Cuda>,
    dt_bias: &Tensor<T, Cuda>,
    recurrent_state: &mut Tensor<f32, Cuda>,
    state_slots: &Tensor<i32, Cuda>,
    cu_seqlens: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let query_shape = query.shape().as_slice();
    if query_shape.len() != 2 || query_shape[0] == 0 || query_shape[1] == 0 {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: query must be non-empty [num_tokens, key_width], got {:?}",
            query_shape
        )));
    }
    let (num_tokens, key_width) = (query_shape[0], query_shape[1]);

    let state_shape = recurrent_state.shape().as_slice();
    if state_shape.len() != 4
        || state_shape[0] == 0
        || state_shape[1] == 0
        || state_shape[2] == 0
        || state_shape[3] == 0
    {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: recurrent_state must be non-empty [num_slots, num_value_heads, key_head_dim, value_head_dim], got {:?}",
            state_shape
        )));
    }
    let (num_slots, num_value_heads, key_head_dim, value_head_dim) = (
        state_shape[0],
        state_shape[1],
        state_shape[2],
        state_shape[3],
    );
    if key_head_dim > 1024 {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: key_head_dim {key_head_dim} exceeds CUDA v1 limit 1024"
        )));
    }
    if !key_width.is_multiple_of(key_head_dim) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: key width {key_width} is not divisible by key_head_dim {key_head_dim}"
        )));
    }
    let num_key_heads = key_width / key_head_dim;
    if num_key_heads == 0 || !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: num_value_heads {num_value_heads} must be divisible by num_key_heads {num_key_heads}"
        )));
    }
    let value_width = num_value_heads
        .checked_mul(value_head_dim)
        .ok_or_else(|| OpError::Shape("gated_delta_rule: value width overflows".into()))?;

    let query_row_stride = packed_row_stride(query, num_tokens, key_width, "query")?;
    let key_row_stride = packed_row_stride(key, num_tokens, key_width, "key")?;
    let value_row_stride = packed_row_stride(value, num_tokens, value_width, "value")?;
    let a_row_stride = packed_row_stride(a, num_tokens, num_value_heads, "a")?;
    let b_row_stride = packed_row_stride(b, num_tokens, num_value_heads, "b")?;
    if output.shape().as_slice() != [num_tokens, value_width] || !output.is_contiguous() {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: output must be contiguous [{num_tokens}, {value_width}], got shape {:?}, strides {:?}",
            output.shape().as_slice(),
            output.strides().as_slice()
        )));
    }
    if a_log.shape().as_slice() != [num_value_heads]
        || dt_bias.shape().as_slice() != [num_value_heads]
    {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: a_log/dt_bias must both be [{num_value_heads}], got {:?}/{:?}",
            a_log.shape().as_slice(),
            dt_bias.shape().as_slice()
        )));
    }

    let slots_shape = state_slots.shape().as_slice();
    if slots_shape.len() != 1 || slots_shape[0] == 0 {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: state_slots must be non-empty rank 1, got {:?}",
            slots_shape
        )));
    }
    let batch = slots_shape[0];
    let cu_len = batch
        .checked_add(1)
        .ok_or_else(|| OpError::Shape("gated_delta_rule: batch size overflows".into()))?;
    if cu_seqlens.shape().as_slice() != [cu_len] {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: cu_seqlens shape {:?} != [{cu_len}]",
            cu_seqlens.shape().as_slice()
        )));
    }

    for (shape, contiguous) in [
        (*a_log.shape(), a_log.is_contiguous()),
        (*dt_bias.shape(), dt_bias.is_contiguous()),
        (*recurrent_state.shape(), recurrent_state.is_contiguous()),
        (*state_slots.shape(), state_slots.is_contiguous()),
        (*cu_seqlens.shape(), cu_seqlens.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::NotContiguous(shape));
        }
    }

    let _blocks = checked_i32(
        batch
            .checked_mul(num_value_heads)
            .ok_or_else(|| OpError::Shape("gated_delta_rule: launch grid overflows".into()))?,
        "launch blocks",
    )?;
    let num_tokens = checked_i32(num_tokens, "num_tokens")?;
    let batch = checked_i32(batch, "batch")?;
    let num_key_heads = checked_i32(num_key_heads, "num_key_heads")?;
    let num_value_heads = checked_i32(num_value_heads, "num_value_heads")?;
    let key_head_dim = checked_i32(key_head_dim, "key_head_dim")?;
    let value_head_dim = checked_i32(value_head_dim, "value_head_dim")?;
    let num_slots = checked_i32(num_slots, "num_slots")?;
    unsafe {
        T::gated_delta_rule(
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
            a_log.data_ptr(),
            dt_bias.data_ptr(),
            recurrent_state.data_ptr_mut(),
            state_slots.data_ptr(),
            cu_seqlens.data_ptr(),
            output.data_ptr_mut(),
            num_tokens,
            batch,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            num_slots,
            query_row_stride,
            key_row_stride,
            value_row_stride,
            a_row_stride,
            b_row_stride,
            stream,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaScope;
    use half::bf16;
    use infer_core::ports::FusedOps;

    #[allow(clippy::too_many_arguments)]
    fn reference_f32(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        a: &[f32],
        b: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        mut state: Vec<f32>,
        slots: &[i32],
        cu_seqlens: &[i32],
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        round_beta: impl Fn(f32) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        fn sigmoid(x: f32) -> f32 {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        }
        fn softplus(x: f32) -> f32 {
            if x > 20.0 { x } else { x.exp().ln_1p() }
        }

        let key_width = num_key_heads * key_head_dim;
        let value_width = num_value_heads * value_head_dim;
        let value_heads_per_key = num_value_heads / num_key_heads;
        let mut output = vec![0.0; value.len()];
        for (sequence, &raw_slot) in slots.iter().enumerate() {
            let start = cu_seqlens[sequence] as usize;
            let end = cu_seqlens[sequence + 1] as usize;
            let slot = raw_slot as usize;
            for value_head in 0..num_value_heads {
                let key_head = value_head / value_heads_per_key;
                let state_base =
                    ((slot * num_value_heads + value_head) * key_head_dim) * value_head_dim;
                let neg_a = -a_log[value_head].exp();
                for token in start..end {
                    let q_base = token * key_width + key_head * key_head_dim;
                    let k_base = q_base;
                    let mut q_norm_sq = 0.0f32;
                    let mut k_norm_sq = 0.0f32;
                    for dim in 0..key_head_dim {
                        q_norm_sq = query[q_base + dim].mul_add(query[q_base + dim], q_norm_sq);
                        k_norm_sq = key[k_base + dim].mul_add(key[k_base + dim], k_norm_sq);
                    }
                    let q_scale = 1.0 / ((q_norm_sq + 1e-6).sqrt() * (key_head_dim as f32).sqrt());
                    let k_scale = 1.0 / (k_norm_sq + 1e-6).sqrt();
                    let head_offset = token * num_value_heads + value_head;
                    let beta = round_beta(sigmoid(b[head_offset]));
                    let decay = (neg_a * softplus(a[head_offset] + dt_bias[value_head])).exp();
                    let value_base = token * value_width + value_head * value_head_dim;

                    for value_dim in 0..value_head_dim {
                        let mut memory = 0.0f32;
                        for key_dim in 0..key_head_dim {
                            let state_index = state_base + key_dim * value_head_dim + value_dim;
                            let decayed = state[state_index] * decay;
                            state[state_index] = decayed;
                            memory = decayed.mul_add(key[k_base + key_dim] * k_scale, memory);
                        }
                        let delta = (value[value_base + value_dim] - memory) * beta;
                        let mut out = 0.0f32;
                        for key_dim in 0..key_head_dim {
                            let state_index = state_base + key_dim * value_head_dim + value_dim;
                            let updated = (key[k_base + key_dim] * k_scale)
                                .mul_add(delta, state[state_index]);
                            state[state_index] = updated;
                            out = updated.mul_add(query[q_base + key_dim] * q_scale, out);
                        }
                        output[value_base + value_dim] = out;
                    }
                }
            }
        }
        (output, state)
    }

    fn deterministic_values(len: usize, phase: f32, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                ((index as f32 * phase).sin() + 0.31 * (index as f32 * 0.07).cos()) * scale
            })
            .collect()
    }

    #[test]
    fn f32_ragged_strided_qkv_matches_reference_and_external_slots() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (tokens, num_key_heads, num_value_heads) = (5usize, 2usize, 4usize);
        let (key_head_dim, value_head_dim, num_slots) = (3usize, 5usize, 3usize);
        let key_width = num_key_heads * key_head_dim;
        let value_width = num_value_heads * value_head_dim;
        let packed_width = 2 * key_width + value_width;

        let query = deterministic_values(tokens * key_width, 0.17, 0.7);
        let key = deterministic_values(tokens * key_width, 0.23, 0.6);
        let value = deterministic_values(tokens * value_width, 0.11, 0.8);
        let mut packed = vec![0.0f32; tokens * packed_width];
        for token in 0..tokens {
            packed[token * packed_width..token * packed_width + key_width]
                .copy_from_slice(&query[token * key_width..(token + 1) * key_width]);
            packed[token * packed_width + key_width..token * packed_width + 2 * key_width]
                .copy_from_slice(&key[token * key_width..(token + 1) * key_width]);
            packed[token * packed_width + 2 * key_width..(token + 1) * packed_width]
                .copy_from_slice(&value[token * value_width..(token + 1) * value_width]);
        }
        let a = deterministic_values(tokens * num_value_heads, 0.19, 0.4);
        let b = deterministic_values(tokens * num_value_heads, 0.13, 0.9);
        let a_log = vec![-0.7, -0.2, 0.1, 0.35];
        let dt_bias = vec![0.3, -0.1, 0.5, -0.4];
        let state = deterministic_values(
            num_slots * num_value_heads * key_head_dim * value_head_dim,
            0.037,
            0.2,
        );
        let slots = vec![2, 1, 0];
        let cu_seqlens = vec![0, 3, 3, 5];
        let (expected_output, expected_state) = reference_f32(
            &query,
            &key,
            &value,
            &a,
            &b,
            &a_log,
            &dt_bias,
            state.clone(),
            &slots,
            &cu_seqlens,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            |beta| beta,
        );

        let packed_dev = Tensor::from_host_slice(&packed, [tokens, packed_width], &cuda).unwrap();
        let query_dev = packed_dev.narrow(1, 0, key_width).unwrap();
        let key_dev = packed_dev.narrow(1, key_width, key_width).unwrap();
        let value_dev = packed_dev.narrow(1, 2 * key_width, value_width).unwrap();
        assert!(!query_dev.is_contiguous());
        let a_dev = Tensor::from_host_slice(&a, [tokens, num_value_heads], &cuda).unwrap();
        let b_dev = Tensor::from_host_slice(&b, [tokens, num_value_heads], &cuda).unwrap();
        let a_log_dev = Tensor::from_host_slice(&a_log, [num_value_heads], &cuda).unwrap();
        let dt_bias_dev = Tensor::from_host_slice(&dt_bias, [num_value_heads], &cuda).unwrap();
        let mut state_dev = Tensor::from_host_slice(
            &state,
            [num_slots, num_value_heads, key_head_dim, value_head_dim],
            &cuda,
        )
        .unwrap();
        let slots_dev = Tensor::from_host_slice(&slots, [slots.len()], &cuda).unwrap();
        let cu_dev = Tensor::from_host_slice(&cu_seqlens, [cu_seqlens.len()], &cuda).unwrap();
        let mut output_dev = Tensor::<f32, Cuda>::zeros([tokens, value_width], &cuda).unwrap();

        <Cuda as FusedOps>::gated_delta_rule(
            &scope,
            &query_dev,
            &key_dev,
            &value_dev,
            &a_dev,
            &b_dev,
            &a_log_dev,
            &dt_bias_dev,
            &mut state_dev,
            &slots_dev,
            &cu_dev,
            &mut output_dev,
        )
        .unwrap();

        let output = output_dev.to_host_vec().unwrap();
        let got_state = state_dev.to_host_vec().unwrap();
        for (index, (&got, &expected)) in output.iter().zip(&expected_output).enumerate() {
            assert!(
                (got - expected).abs() < 8e-5,
                "output mismatch at {index}: got={got}, expected={expected}"
            );
        }
        for (index, (&got, &expected)) in got_state.iter().zip(&expected_state).enumerate() {
            assert!(
                (got - expected).abs() < 1.5e-4,
                "state mismatch at {index}: got={got}, expected={expected}"
            );
        }

        let slot_size = num_value_heads * key_head_dim * value_head_dim;
        assert_eq!(
            &got_state[slot_size..2 * slot_size],
            &state[slot_size..2 * slot_size]
        );
    }

    #[test]
    fn bf16_split_calls_equal_one_shot_and_match_reference() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (tokens, num_key_heads, num_value_heads) = (7usize, 2usize, 4usize);
        let (key_head_dim, value_head_dim, num_slots) = (4usize, 3usize, 2usize);
        let key_width = num_key_heads * key_head_dim;
        let value_width = num_value_heads * value_head_dim;

        let to_bf16 = |values: Vec<f32>| values.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
        let query = to_bf16(deterministic_values(tokens * key_width, 0.09, 0.8));
        let key = to_bf16(deterministic_values(tokens * key_width, 0.14, 0.65));
        let value = to_bf16(deterministic_values(tokens * value_width, 0.12, 0.75));
        let a = to_bf16(deterministic_values(tokens * num_value_heads, 0.18, 0.35));
        let b = to_bf16(deterministic_values(tokens * num_value_heads, 0.21, 0.8));
        let a_log = vec![-0.5, -0.1, 0.2, 0.45];
        let dt_bias = to_bf16(vec![0.2, -0.3, 0.45, 0.05]);
        let initial_state = deterministic_values(
            num_slots * num_value_heads * key_head_dim * value_head_dim,
            0.031,
            0.1,
        );
        let as_f32 = |values: &[bf16]| values.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let (expected_output, expected_state) = reference_f32(
            &as_f32(&query),
            &as_f32(&key),
            &as_f32(&value),
            &as_f32(&a),
            &as_f32(&b),
            &a_log,
            &as_f32(&dt_bias),
            initial_state.clone(),
            &[1],
            &[0, tokens as i32],
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            |beta| bf16::from_f32(beta).to_f32(),
        );

        let query_all = Tensor::from_host_slice(&query, [tokens, key_width], &cuda).unwrap();
        let key_all = Tensor::from_host_slice(&key, [tokens, key_width], &cuda).unwrap();
        let value_all = Tensor::from_host_slice(&value, [tokens, value_width], &cuda).unwrap();
        let a_all = Tensor::from_host_slice(&a, [tokens, num_value_heads], &cuda).unwrap();
        let b_all = Tensor::from_host_slice(&b, [tokens, num_value_heads], &cuda).unwrap();
        let a_log_dev = Tensor::from_host_slice(&a_log, [num_value_heads], &cuda).unwrap();
        let dt_bias_dev = Tensor::from_host_slice(&dt_bias, [num_value_heads], &cuda).unwrap();
        let slots_dev = Tensor::from_host_slice(&[1i32], [1], &cuda).unwrap();
        let cu_all = Tensor::from_host_slice(&[0i32, tokens as i32], [2], &cuda).unwrap();
        let mut state_all = Tensor::from_host_slice(
            &initial_state,
            [num_slots, num_value_heads, key_head_dim, value_head_dim],
            &cuda,
        )
        .unwrap();
        let mut output_all = Tensor::<bf16, Cuda>::zeros([tokens, value_width], &cuda).unwrap();
        <Cuda as FusedOps>::gated_delta_rule(
            &scope,
            &query_all,
            &key_all,
            &value_all,
            &a_all,
            &b_all,
            &a_log_dev,
            &dt_bias_dev,
            &mut state_all,
            &slots_dev,
            &cu_all,
            &mut output_all,
        )
        .unwrap();

        let one_shot = output_all.to_host_vec().unwrap();
        for (index, (&got, &expected)) in one_shot.iter().zip(&expected_output).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() < 0.02,
                "BF16 output mismatch at {index}: got={}, expected={expected}",
                got.to_f32()
            );
        }
        let final_all = state_all.to_host_vec().unwrap();
        for (index, (&got, &expected)) in final_all.iter().zip(&expected_state).enumerate() {
            assert!(
                (got - expected).abs() < 2e-4,
                "BF16 state mismatch at {index}: got={got}, expected={expected}"
            );
        }

        let split = 3usize;
        let make_chunk = |data: &[bf16], width: usize, start: usize, end: usize| {
            Tensor::from_host_slice(
                &data[start * width..end * width],
                [end - start, width],
                &cuda,
            )
            .unwrap()
        };
        let query_a = make_chunk(&query, key_width, 0, split);
        let query_b = make_chunk(&query, key_width, split, tokens);
        let key_a = make_chunk(&key, key_width, 0, split);
        let key_b = make_chunk(&key, key_width, split, tokens);
        let value_a = make_chunk(&value, value_width, 0, split);
        let value_b = make_chunk(&value, value_width, split, tokens);
        let a_a = make_chunk(&a, num_value_heads, 0, split);
        let a_b = make_chunk(&a, num_value_heads, split, tokens);
        let b_a = make_chunk(&b, num_value_heads, 0, split);
        let b_b = make_chunk(&b, num_value_heads, split, tokens);
        let cu_a = Tensor::from_host_slice(&[0i32, split as i32], [2], &cuda).unwrap();
        let cu_b = Tensor::from_host_slice(&[0i32, (tokens - split) as i32], [2], &cuda).unwrap();
        let mut state_split = Tensor::from_host_slice(
            &initial_state,
            [num_slots, num_value_heads, key_head_dim, value_head_dim],
            &cuda,
        )
        .unwrap();
        let mut output_a = Tensor::<bf16, Cuda>::zeros([split, value_width], &cuda).unwrap();
        let mut output_b =
            Tensor::<bf16, Cuda>::zeros([tokens - split, value_width], &cuda).unwrap();
        <Cuda as FusedOps>::gated_delta_rule(
            &scope,
            &query_a,
            &key_a,
            &value_a,
            &a_a,
            &b_a,
            &a_log_dev,
            &dt_bias_dev,
            &mut state_split,
            &slots_dev,
            &cu_a,
            &mut output_a,
        )
        .unwrap();
        <Cuda as FusedOps>::gated_delta_rule(
            &scope,
            &query_b,
            &key_b,
            &value_b,
            &a_b,
            &b_b,
            &a_log_dev,
            &dt_bias_dev,
            &mut state_split,
            &slots_dev,
            &cu_b,
            &mut output_b,
        )
        .unwrap();

        let mut chunked = output_a.to_host_vec().unwrap();
        chunked.extend(output_b.to_host_vec().unwrap());
        assert_eq!(
            one_shot.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            chunked.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(state_split.to_host_vec().unwrap(), final_all);
        let slot_size = num_value_heads * key_head_dim * value_head_dim;
        assert_eq!(&final_all[..slot_size], &initial_state[..slot_size]);
    }

    #[test]
    fn bf16_qwen35_decode_dimensions_match_closed_form_from_zero_state() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (num_key_heads, num_value_heads) = (16usize, 32usize);
        let (key_head_dim, value_head_dim) = (128usize, 128usize);
        let key_width = num_key_heads * key_head_dim;
        let value_width = num_value_heads * value_head_dim;
        let to_bf16 = |values: Vec<f32>| values.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
        let query = to_bf16(deterministic_values(key_width, 0.013, 0.7));
        let key = to_bf16(deterministic_values(key_width, 0.017, 0.6));
        let value = to_bf16(deterministic_values(value_width, 0.011, 0.8));
        let a = to_bf16(deterministic_values(num_value_heads, 0.09, 0.3));
        let b = to_bf16(deterministic_values(num_value_heads, 0.12, 0.7));
        let a_log = vec![0.0f32; num_value_heads];
        let dt_bias = vec![bf16::from_f32(0.2); num_value_heads];

        let query_dev = Tensor::from_host_slice(&query, [1, key_width], &cuda).unwrap();
        let key_dev = Tensor::from_host_slice(&key, [1, key_width], &cuda).unwrap();
        let value_dev = Tensor::from_host_slice(&value, [1, value_width], &cuda).unwrap();
        let a_dev = Tensor::from_host_slice(&a, [1, num_value_heads], &cuda).unwrap();
        let b_dev = Tensor::from_host_slice(&b, [1, num_value_heads], &cuda).unwrap();
        let a_log_dev = Tensor::from_host_slice(&a_log, [num_value_heads], &cuda).unwrap();
        let dt_bias_dev = Tensor::from_host_slice(&dt_bias, [num_value_heads], &cuda).unwrap();
        let mut state_dev =
            Tensor::<f32, Cuda>::zeros([1, num_value_heads, key_head_dim, value_head_dim], &cuda)
                .unwrap();
        let slots_dev = Tensor::from_host_slice(&[0i32], [1], &cuda).unwrap();
        let cu_dev = Tensor::from_host_slice(&[0i32, 1], [2], &cuda).unwrap();
        let mut output_dev = Tensor::<bf16, Cuda>::zeros([1, value_width], &cuda).unwrap();
        <Cuda as FusedOps>::gated_delta_rule(
            &scope,
            &query_dev,
            &key_dev,
            &value_dev,
            &a_dev,
            &b_dev,
            &a_log_dev,
            &dt_bias_dev,
            &mut state_dev,
            &slots_dev,
            &cu_dev,
            &mut output_dev,
        )
        .unwrap();

        let output = output_dev.to_host_vec().unwrap();
        let state = state_dev.to_host_vec().unwrap();
        for (value_head, raw_b_value) in b.iter().enumerate() {
            let key_head = value_head / 2;
            let q_base = key_head * key_head_dim;
            let mut q_norm_sq = 0.0f32;
            let mut k_norm_sq = 0.0f32;
            for dim in 0..key_head_dim {
                let q = query[q_base + dim].to_f32();
                let k = key[q_base + dim].to_f32();
                q_norm_sq = q.mul_add(q, q_norm_sq);
                k_norm_sq = k.mul_add(k, k_norm_sq);
            }
            let q_scale = 1.0 / ((q_norm_sq + 1e-6).sqrt() * (key_head_dim as f32).sqrt());
            let k_scale = 1.0 / (k_norm_sq + 1e-6).sqrt();
            let mut qk = 0.0f32;
            for dim in 0..key_head_dim {
                qk = (query[q_base + dim].to_f32() * q_scale)
                    .mul_add(key[q_base + dim].to_f32() * k_scale, qk);
            }
            let raw_b = raw_b_value.to_f32();
            let beta = bf16::from_f32(1.0 / (1.0 + (-raw_b).exp())).to_f32();
            let value_base = value_head * value_head_dim;
            for value_dim in 0..value_head_dim {
                let delta = value[value_base + value_dim].to_f32() * beta;
                let expected = qk * delta;
                let got = output[value_base + value_dim].to_f32();
                assert!(
                    (got - expected).abs() < 0.02,
                    "real-shape output mismatch at head={value_head}, dim={value_dim}: got={got}, expected={expected}"
                );
            }

            for &(key_dim, value_dim) in &[(0usize, 0usize), (37, 61), (127, 127)] {
                let expected = key[q_base + key_dim].to_f32()
                    * k_scale
                    * value[value_base + value_dim].to_f32()
                    * beta;
                let index = (value_head * key_head_dim + key_dim) * value_head_dim + value_dim;
                assert!(
                    (state[index] - expected).abs() < 2e-5,
                    "real-shape state mismatch at head={value_head}, key={key_dim}, value={value_dim}: got={}, expected={expected}",
                    state[index]
                );
            }
        }
    }
}
