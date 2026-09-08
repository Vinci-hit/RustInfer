//! Causal depthwise Conv1d + SiLU CUDA launch wrapper.
//!
//! The operator itself owns no cache. `conv_state` is a caller-owned mutable
//! tensor, while `state_slots` maps each ragged sequence to the row it may
//! update. Dtype dispatch follows the same trait-per-kernel structure as the
//! other CUDA operators in this crate.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn causal_conv1d_silu_f32_forward(
        input: *const f32,
        weight: *const f32,
        conv_state: *mut f32,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut f32,
        num_tokens: i32,
        batch: i32,
        channels: i32,
        kernel_size: i32,
        num_slots: i32,
        stream: cudaStream_t,
    );
    fn causal_conv1d_silu_bf16_forward(
        input: *const half::bf16,
        weight: *const half::bf16,
        conv_state: *mut half::bf16,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut half::bf16,
        num_tokens: i32,
        batch: i32,
        channels: i32,
        kernel_size: i32,
        num_slots: i32,
        stream: cudaStream_t,
    );
    fn causal_conv1d_silu_f16_forward(
        input: *const half::f16,
        weight: *const half::f16,
        conv_state: *mut half::f16,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut half::f16,
        num_tokens: i32,
        batch: i32,
        channels: i32,
        kernel_size: i32,
        num_slots: i32,
        stream: cudaStream_t,
    );
}

/// Element types with a causal-conv CUDA kernel.
///
/// # Safety
/// All pointers must address device tensors matching the documented contiguous
/// layouts, `state_slots` must contain distinct in-range values, and
/// `cu_seqlens` must be monotonic from zero to `num_tokens`.
pub trait CausalConv1dKernel: CudaFloat {
    #[allow(clippy::too_many_arguments)]
    unsafe fn causal_conv1d_silu(
        input: *const Self,
        weight: *const Self,
        conv_state: *mut Self,
        state_slots: *const i32,
        cu_seqlens: *const i32,
        output: *mut Self,
        num_tokens: i32,
        batch: i32,
        channels: i32,
        kernel_size: i32,
        num_slots: i32,
        stream: cudaStream_t,
    );
}

macro_rules! impl_causal_conv_kernel {
    ($ty:ty, $entry:ident) => {
        impl CausalConv1dKernel for $ty {
            #[inline]
            unsafe fn causal_conv1d_silu(
                input: *const Self,
                weight: *const Self,
                conv_state: *mut Self,
                state_slots: *const i32,
                cu_seqlens: *const i32,
                output: *mut Self,
                num_tokens: i32,
                batch: i32,
                channels: i32,
                kernel_size: i32,
                num_slots: i32,
                stream: cudaStream_t,
            ) {
                unsafe {
                    $entry(
                        input,
                        weight,
                        conv_state,
                        state_slots,
                        cu_seqlens,
                        output,
                        num_tokens,
                        batch,
                        channels,
                        kernel_size,
                        num_slots,
                        stream,
                    )
                }
            }
        }
    };
}

impl_causal_conv_kernel!(f32, causal_conv1d_silu_f32_forward);
impl_causal_conv_kernel!(half::bf16, causal_conv1d_silu_bf16_forward);
impl_causal_conv_kernel!(half::f16, causal_conv1d_silu_f16_forward);

fn checked_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value)
        .map_err(|_| OpError::Shape(format!("causal_conv1d_silu: {name} exceeds i32")))
}

/// Run causal depthwise Conv1d + SiLU over a flattened ragged token tape.
///
/// `conv_state` is both input and output. The CUDA wrapper stores no state and
/// allocates no memory, so the launch is safe to capture in a CUDA graph when
/// the caller keeps all tensor addresses stable.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_silu<T: CausalConv1dKernel>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    conv_state: &mut Tensor<T, Cuda>,
    state_slots: &Tensor<i32, Cuda>,
    cu_seqlens: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let input_shape = input.shape().as_slice();
    if input_shape.len() != 2 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: input must be [num_tokens, channels], got {:?}",
            input_shape
        )));
    }
    let (num_tokens, channels) = (input_shape[0], input_shape[1]);
    if num_tokens == 0 || channels == 0 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: input dimensions must be non-zero, got {:?}",
            input_shape
        )));
    }
    if output.shape().as_slice() != input_shape {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: output shape {:?} != input shape {:?}",
            output.shape().as_slice(),
            input_shape
        )));
    }

    let weight_shape = weight.shape().as_slice();
    if weight_shape.len() != 3 || weight_shape[0] != channels || weight_shape[1] != 1 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: weight must be [{channels}, 1, kernel_size], got {:?}",
            weight_shape
        )));
    }
    let kernel_size = weight_shape[2];
    if kernel_size == 0 {
        return Err(OpError::Shape(
            "causal_conv1d_silu: kernel_size must be non-zero".into(),
        ));
    }

    let state_shape = conv_state.shape().as_slice();
    if state_shape.len() != 3
        || state_shape[0] == 0
        || state_shape[1] != channels
        || state_shape[2] != kernel_size
    {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: conv_state must be [num_slots, {channels}, {kernel_size}], got {:?}",
            state_shape
        )));
    }
    let num_slots = state_shape[0];

    let slot_shape = state_slots.shape().as_slice();
    if slot_shape.len() != 1 || slot_shape[0] == 0 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: state_slots must be non-empty rank 1, got {:?}",
            slot_shape
        )));
    }
    let batch = slot_shape[0];
    let cu_len = batch
        .checked_add(1)
        .ok_or_else(|| OpError::Shape("causal_conv1d_silu: batch size overflows".into()))?;
    if cu_seqlens.shape().as_slice() != [cu_len] {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: cu_seqlens shape {:?} != [{}]",
            cu_seqlens.shape().as_slice(),
            cu_len
        )));
    }

    for (shape, contiguous) in [
        (*input.shape(), input.is_contiguous()),
        (*weight.shape(), weight.is_contiguous()),
        (*conv_state.shape(), conv_state.is_contiguous()),
        (*state_slots.shape(), state_slots.is_contiguous()),
        (*cu_seqlens.shape(), cu_seqlens.is_contiguous()),
        (*output.shape(), output.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::NotContiguous(shape));
        }
    }

    let num_tokens = checked_i32(num_tokens, "num_tokens")?;
    let batch = checked_i32(batch, "batch")?;
    let channels = checked_i32(channels, "channels")?;
    let kernel_size = checked_i32(kernel_size, "kernel_size")?;
    let num_slots = checked_i32(num_slots, "num_slots")?;
    unsafe {
        T::causal_conv1d_silu(
            input.data_ptr(),
            weight.data_ptr(),
            conv_state.data_ptr_mut(),
            state_slots.data_ptr(),
            cu_seqlens.data_ptr(),
            output.data_ptr_mut(),
            num_tokens,
            batch,
            channels,
            kernel_size,
            num_slots,
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

    fn reference_f32(
        input: &[f32],
        weight: &[f32],
        mut state: Vec<f32>,
        slots: &[i32],
        cu_seqlens: &[i32],
        channels: usize,
        kernel_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut output = vec![0.0; input.len()];
        for (sequence, &raw_slot) in slots.iter().enumerate() {
            let start = cu_seqlens[sequence] as usize;
            let end = cu_seqlens[sequence + 1] as usize;
            let sequence_length = end - start;
            let slot = raw_slot as usize;
            for channel in 0..channels {
                let state_base = (slot * channels + channel) * kernel_size;
                let weight_base = channel * kernel_size;
                for token in 0..sequence_length {
                    let mut sum = 0.0f32;
                    for tap in 0..kernel_size {
                        let relative = token as isize + tap as isize + 1 - kernel_size as isize;
                        let value = if relative >= 0 {
                            input[(start + relative as usize) * channels + channel]
                        } else {
                            state[state_base + (kernel_size as isize + relative) as usize]
                        };
                        sum = value.mul_add(weight[weight_base + tap], sum);
                    }
                    output[(start + token) * channels + channel] = sum / (1.0 + (-sum).exp());
                }
                for state_col in 0..kernel_size {
                    let concat_col = sequence_length + state_col;
                    state[state_base + state_col] = if concat_col < kernel_size {
                        state[state_base + concat_col]
                    } else {
                        input[(start + concat_col - kernel_size) * channels + channel]
                    };
                }
            }
        }
        (output, state)
    }

    #[test]
    fn f32_ragged_port_matches_reference_and_updates_only_selected_slots() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (channels, kernel_size, num_slots) = (5usize, 4usize, 3usize);
        // Includes an empty middle row, matching a padded ragged/graph row.
        let cu_seqlens = vec![0, 3, 3, 5];
        let slots = vec![2, 1, 0];
        let input: Vec<f32> = (0..5 * channels)
            .map(|i| ((i as f32 * 0.31).sin() - 0.2) * 1.7)
            .collect();
        let weight: Vec<f32> = (0..channels * kernel_size)
            .map(|i| ((i as f32 * 0.17).cos() - 0.35) * 0.6)
            .collect();
        let state: Vec<f32> = (0..num_slots * channels * kernel_size)
            .map(|i| (i as f32 * 0.09).sin() * 0.8)
            .collect();
        let (expected_output, expected_state) = reference_f32(
            &input,
            &weight,
            state.clone(),
            &slots,
            &cu_seqlens,
            channels,
            kernel_size,
        );

        let input_dev = Tensor::from_host_slice(&input, [5, channels], &cuda).unwrap();
        let weight_dev =
            Tensor::from_host_slice(&weight, [channels, 1, kernel_size], &cuda).unwrap();
        let mut state_dev =
            Tensor::from_host_slice(&state, [num_slots, channels, kernel_size], &cuda).unwrap();
        let slots_dev = Tensor::from_host_slice(&slots, [slots.len()], &cuda).unwrap();
        let cu_dev = Tensor::from_host_slice(&cu_seqlens, [cu_seqlens.len()], &cuda).unwrap();
        let mut output_dev = Tensor::<f32, Cuda>::zeros([5, channels], &cuda).unwrap();

        <Cuda as FusedOps>::causal_conv1d_silu(
            &scope,
            &input_dev,
            &weight_dev,
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
                (got - expected).abs() < 2e-6,
                "output mismatch at {index}: got={got}, expected={expected}"
            );
        }
        for (index, (&got, &expected)) in got_state.iter().zip(&expected_state).enumerate() {
            assert!(
                (got - expected).abs() < 1e-7,
                "state mismatch at {index}: got={got}, expected={expected}"
            );
        }

        // Sequence 1 had q_len=0, so its externally-owned slot is untouched.
        let untouched_start = channels * kernel_size;
        let untouched_end = 2 * channels * kernel_size;
        assert_eq!(
            &got_state[untouched_start..untouched_end],
            &state[untouched_start..untouched_end]
        );
    }

    #[test]
    fn bf16_split_calls_equal_one_shot_and_preserve_external_state() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (tokens, channels, kernel_size, num_slots) = (7usize, 7usize, 4usize, 2usize);
        let input: Vec<bf16> = (0..tokens * channels)
            .map(|i| bf16::from_f32(((i as f32 * 0.23).sin() - 0.1) * 1.3))
            .collect();
        let weight: Vec<bf16> = (0..channels * kernel_size)
            .map(|i| bf16::from_f32(((i as f32 * 0.11).cos() - 0.4) * 0.7))
            .collect();
        let weight_dev =
            Tensor::from_host_slice(&weight, [channels, 1, kernel_size], &cuda).unwrap();
        let slots_dev = Tensor::from_host_slice(&[1i32], [1], &cuda).unwrap();

        let input_all = Tensor::from_host_slice(&input, [tokens, channels], &cuda).unwrap();
        let cu_all = Tensor::from_host_slice(&[0i32, tokens as i32], [2], &cuda).unwrap();
        let mut state_all =
            Tensor::<bf16, Cuda>::zeros([num_slots, channels, kernel_size], &cuda).unwrap();
        let mut output_all = Tensor::<bf16, Cuda>::zeros([tokens, channels], &cuda).unwrap();
        <Cuda as FusedOps>::causal_conv1d_silu(
            &scope,
            &input_all,
            &weight_dev,
            &mut state_all,
            &slots_dev,
            &cu_all,
            &mut output_all,
        )
        .unwrap();

        let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let weight_f32 = weight
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let (expected_output, expected_state) = reference_f32(
            &input_f32,
            &weight_f32,
            vec![0.0; num_slots * channels * kernel_size],
            &[1],
            &[0, tokens as i32],
            channels,
            kernel_size,
        );
        let one_shot = output_all.to_host_vec().unwrap();
        for (index, (&got, &expected)) in one_shot.iter().zip(&expected_output).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() < 0.02,
                "BF16 output mismatch at {index}: got={}, expected={expected}",
                got.to_f32()
            );
        }
        let final_all = state_all.to_host_vec().unwrap();
        assert_eq!(
            final_all.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_state
                .iter()
                .map(|&x| bf16::from_f32(x).to_bits())
                .collect::<Vec<_>>()
        );

        let split = 3usize;
        let input_a =
            Tensor::from_host_slice(&input[..split * channels], [split, channels], &cuda).unwrap();
        let input_b = Tensor::from_host_slice(
            &input[split * channels..],
            [tokens - split, channels],
            &cuda,
        )
        .unwrap();
        let cu_a = Tensor::from_host_slice(&[0i32, split as i32], [2], &cuda).unwrap();
        let cu_b = Tensor::from_host_slice(&[0i32, (tokens - split) as i32], [2], &cuda).unwrap();
        let mut state_split =
            Tensor::<bf16, Cuda>::zeros([num_slots, channels, kernel_size], &cuda).unwrap();
        let mut output_a = Tensor::<bf16, Cuda>::zeros([split, channels], &cuda).unwrap();
        let mut output_b = Tensor::<bf16, Cuda>::zeros([tokens - split, channels], &cuda).unwrap();
        <Cuda as FusedOps>::causal_conv1d_silu(
            &scope,
            &input_a,
            &weight_dev,
            &mut state_split,
            &slots_dev,
            &cu_a,
            &mut output_a,
        )
        .unwrap();
        <Cuda as FusedOps>::causal_conv1d_silu(
            &scope,
            &input_b,
            &weight_dev,
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

        let final_split = state_split.to_host_vec().unwrap();
        assert_eq!(
            final_all.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            final_split.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        // Slot 0 was never selected by either call and remains zero.
        assert!(
            final_split[..channels * kernel_size]
                .iter()
                .all(|value| value.to_bits() == 0)
        );
    }
}
