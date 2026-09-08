//! Qwen3.5 gated RMSNorm CUDA launch wrapper.
//!
//! This is a stateless fused operator: all inputs, including the fp32 norm
//! weight, arrive as tensors on every invocation. It accepts contiguous
//! `[..., head_dim]` tensors and rank-2/rank-3 views whose last dimension is
//! packed.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::Dtype;

unsafe extern "C" {
    fn gated_rmsnorm_f32_forward(
        input: *const f32,
        gate: *const f32,
        weight: *const f32,
        output: *mut f32,
        outer0: i32,
        outer1: i32,
        head_dim: i32,
        input_stride0: i64,
        input_stride1: i64,
        gate_stride0: i64,
        gate_stride1: i64,
        output_stride0: i64,
        output_stride1: i64,
        eps: f32,
        stream: cudaStream_t,
    );
    fn gated_rmsnorm_bf16_forward(
        input: *const half::bf16,
        gate: *const half::bf16,
        weight: *const f32,
        output: *mut half::bf16,
        outer0: i32,
        outer1: i32,
        head_dim: i32,
        input_stride0: i64,
        input_stride1: i64,
        gate_stride0: i64,
        gate_stride1: i64,
        output_stride0: i64,
        output_stride1: i64,
        eps: f32,
        stream: cudaStream_t,
    );
    fn gated_rmsnorm_f16_forward(
        input: *const half::f16,
        gate: *const half::f16,
        weight: *const f32,
        output: *mut half::f16,
        outer0: i32,
        outer1: i32,
        head_dim: i32,
        input_stride0: i64,
        input_stride1: i64,
        gate_stride0: i64,
        gate_stride1: i64,
        output_stride0: i64,
        output_stride1: i64,
        eps: f32,
        stream: cudaStream_t,
    );
}

#[derive(Clone, Copy)]
pub struct Layout {
    outer0: i32,
    outer1: i32,
    head_dim: i32,
    stride0: i64,
    stride1: i64,
}

/// Element types with a gated RMSNorm CUDA kernel.
///
/// # Safety
/// Pointers and layouts must describe validated tensors on `stream`. The
/// input, gate, and output may alias, but the fp32 weight must remain readable
/// for the entire launch.
pub trait GatedRmsNormKernel: CudaFloat {
    #[allow(clippy::too_many_arguments)]
    unsafe fn gated_rmsnorm(
        input: *const Self,
        gate: *const Self,
        weight: *const f32,
        output: *mut Self,
        input_layout: Layout,
        gate_layout: Layout,
        output_layout: Layout,
        eps: f32,
        stream: cudaStream_t,
    );
}

macro_rules! impl_gated_rmsnorm_kernel {
    ($ty:ty, $entry:ident) => {
        impl GatedRmsNormKernel for $ty {
            #[inline]
            unsafe fn gated_rmsnorm(
                input: *const Self,
                gate: *const Self,
                weight: *const f32,
                output: *mut Self,
                input_layout: Layout,
                gate_layout: Layout,
                output_layout: Layout,
                eps: f32,
                stream: cudaStream_t,
            ) {
                unsafe {
                    $entry(
                        input,
                        gate,
                        weight,
                        output,
                        input_layout.outer0,
                        input_layout.outer1,
                        input_layout.head_dim,
                        input_layout.stride0,
                        input_layout.stride1,
                        gate_layout.stride0,
                        gate_layout.stride1,
                        output_layout.stride0,
                        output_layout.stride1,
                        eps,
                        stream,
                    )
                }
            }
        }
    };
}

impl_gated_rmsnorm_kernel!(f32, gated_rmsnorm_f32_forward);
impl_gated_rmsnorm_kernel!(half::bf16, gated_rmsnorm_bf16_forward);
impl_gated_rmsnorm_kernel!(half::f16, gated_rmsnorm_f16_forward);

fn checked_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value).map_err(|_| OpError::Shape(format!("gated_rmsnorm: {name} exceeds i32")))
}

fn checked_i64(value: usize, name: &str) -> OpResult<i64> {
    i64::try_from(value).map_err(|_| OpError::Shape(format!("gated_rmsnorm: {name} exceeds i64")))
}

fn derive_layout<T: Dtype>(tensor: &Tensor<T, Cuda>, head_dim: usize) -> OpResult<Layout> {
    let shape = tensor.shape().as_slice();
    let strides = tensor.strides().as_slice();
    if shape.is_empty() || shape.last().copied() != Some(head_dim) || strides.last() != Some(&1) {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: bad layout shape={shape:?} strides={strides:?} head_dim={head_dim}"
        )));
    }
    let head_dim_i32 = checked_i32(head_dim, "head_dim")?;
    match tensor.ndim() {
        1 => Ok(Layout {
            outer0: 1,
            outer1: 1,
            head_dim: head_dim_i32,
            stride0: 0,
            stride1: 0,
        }),
        2 => Ok(Layout {
            outer0: checked_i32(shape[0], "outer0")?,
            outer1: 1,
            head_dim: head_dim_i32,
            stride0: checked_i64(strides[0], "stride0")?,
            stride1: 0,
        }),
        3 => {
            let rows = shape[0]
                .checked_mul(shape[1])
                .ok_or_else(|| OpError::Shape("gated_rmsnorm: row count overflows".into()))?;
            let _ = checked_i32(rows, "row count")?;
            Ok(Layout {
                outer0: checked_i32(shape[0], "outer0")?,
                outer1: checked_i32(shape[1], "outer1")?,
                head_dim: head_dim_i32,
                stride0: checked_i64(strides[0], "stride0")?,
                stride1: checked_i64(strides[1], "stride1")?,
            })
        }
        _ if tensor.is_contiguous() => Ok(Layout {
            outer0: checked_i32(tensor.numel() / head_dim, "row count")?,
            outer1: 1,
            head_dim: head_dim_i32,
            stride0: checked_i64(head_dim, "contiguous row stride")?,
            stride1: 0,
        }),
        _ => Err(OpError::Shape(
            "gated_rmsnorm: strided tensors must have rank 2 or 3".into(),
        )),
    }
}

/// Compute `RMSNorm(input, weight, eps) * SiLU(gate)` over the last dimension.
///
/// The launch is allocation-free and owns no state, making it CUDA-graph
/// capturable when the caller supplies stable tensor addresses.
pub fn gated_rmsnorm<T: GatedRmsNormKernel>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    gate: &Tensor<T, Cuda>,
    weight: &Tensor<f32, Cuda>,
    output: &mut Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let shape = input.shape().as_slice();
    if shape.is_empty() || input.numel() == 0 {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: input must be non-empty [..., head_dim], got {shape:?}"
        )));
    }
    if gate.shape().as_slice() != shape || output.shape().as_slice() != shape {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: input/gate/output shapes must match, got {:?}/{:?}/{:?}",
            shape,
            gate.shape().as_slice(),
            output.shape().as_slice()
        )));
    }
    let head_dim = *shape.last().expect("non-empty shape checked above");
    if head_dim == 0 || weight.shape().as_slice() != [head_dim] || !weight.is_contiguous() {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: weight must be contiguous fp32 [{head_dim}], got shape {:?}, strides {:?}",
            weight.shape().as_slice(),
            weight.strides().as_slice()
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: eps must be finite and non-negative, got {eps}"
        )));
    }

    let input_layout = derive_layout(input, head_dim)?;
    let gate_layout = derive_layout(gate, head_dim)?;
    let output_layout = derive_layout(output, head_dim)?;
    if input_layout.outer0 != gate_layout.outer0
        || input_layout.outer1 != gate_layout.outer1
        || input_layout.outer0 != output_layout.outer0
        || input_layout.outer1 != output_layout.outer1
    {
        return Err(OpError::Shape(
            "gated_rmsnorm: input/gate/output row layouts are incompatible".into(),
        ));
    }

    unsafe {
        T::gated_rmsnorm(
            input.data_ptr(),
            gate.data_ptr(),
            weight.data_ptr(),
            output.data_ptr_mut(),
            input_layout,
            gate_layout,
            output_layout,
            eps,
            stream,
        )
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
        gate: &[f32],
        weight: &[f32],
        head_dim: usize,
        eps: f32,
        round_normalized: impl Fn(f32) -> f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for row in 0..input.len() / head_dim {
            let base = row * head_dim;
            let mut square_sum = 0.0f32;
            for col in 0..head_dim {
                let value = input[base + col];
                square_sum = value.mul_add(value, square_sum);
            }
            let inv_rms = (square_sum / head_dim as f32 + eps).sqrt().recip();
            for col in 0..head_dim {
                let normalized = round_normalized(input[base + col] * inv_rms);
                let gate_value = gate[base + col];
                let silu_gate = gate_value / (1.0 + (-gate_value).exp());
                output[base + col] = normalized * weight[col] * silu_gate;
            }
        }
        output
    }

    #[test]
    fn f32_rank3_views_match_reference_without_touching_padding() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (outer0, outer1, head_dim) = (2usize, 3usize, 7usize);
        let rows = outer0 * outer1;
        let (input_pitch, gate_pitch, output_pitch) = (11usize, 13usize, 12usize);
        let (input_start, gate_start, output_start) = (1usize, 2usize, 3usize);
        let sentinel = -91.25f32;

        let mut input_storage = vec![77.0f32; rows * input_pitch];
        let mut gate_storage = vec![-66.0f32; rows * gate_pitch];
        let mut logical_input = vec![0.0f32; rows * head_dim];
        let mut logical_gate = vec![0.0f32; rows * head_dim];
        for row in 0..rows {
            for col in 0..head_dim {
                let logical = row * head_dim + col;
                let input_value = ((logical as f32 * 0.37).sin() - 0.2) * 1.4;
                let gate_value = ((logical as f32 * 0.19).cos() - 0.1) * 1.7;
                logical_input[logical] = input_value;
                logical_gate[logical] = gate_value;
                input_storage[row * input_pitch + input_start + col] = input_value;
                gate_storage[row * gate_pitch + gate_start + col] = gate_value;
            }
        }
        let weight: Vec<f32> = (0..head_dim).map(|col| 0.45 + col as f32 * 0.13).collect();
        let eps = 1e-5;
        let expected = reference_f32(
            &logical_input,
            &logical_gate,
            &weight,
            head_dim,
            eps,
            |value| value,
        );

        let input_backing =
            Tensor::from_host_slice(&input_storage, [outer0, outer1, input_pitch], &cuda).unwrap();
        let gate_backing =
            Tensor::from_host_slice(&gate_storage, [outer0, outer1, gate_pitch], &cuda).unwrap();
        let output_backing = Tensor::from_host_slice(
            &vec![sentinel; rows * output_pitch],
            [outer0, outer1, output_pitch],
            &cuda,
        )
        .unwrap();
        let input = input_backing.narrow(2, input_start, head_dim).unwrap();
        let gate = gate_backing.narrow(2, gate_start, head_dim).unwrap();
        let mut output = output_backing.narrow(2, output_start, head_dim).unwrap();
        let weight = Tensor::from_host_slice(&weight, [head_dim], &cuda).unwrap();

        <Cuda as FusedOps>::gated_rmsnorm(&scope, &input, &gate, &weight, &mut output, eps)
            .unwrap();

        let got_storage = output_backing.to_host_vec().unwrap();
        for row in 0..rows {
            for col in 0..output_pitch {
                let got = got_storage[row * output_pitch + col];
                if (output_start..output_start + head_dim).contains(&col) {
                    let expected_value = expected[row * head_dim + col - output_start];
                    assert!(
                        (got - expected_value).abs() < 3e-6,
                        "row={row} col={col}: got={got}, expected={expected_value}"
                    );
                } else {
                    assert_eq!(got, sentinel, "padding changed at row={row}, col={col}");
                }
            }
        }

        // A packed output/gate may be paired with a strided input. Rank-3
        // tensors retain the same logical row decomposition in either case.
        let packed_gate =
            Tensor::from_host_slice(&logical_gate, [outer0, outer1, head_dim], &cuda).unwrap();
        let mut packed_output =
            Tensor::<f32, Cuda>::zeros([outer0, outer1, head_dim], &cuda).unwrap();
        <Cuda as FusedOps>::gated_rmsnorm(
            &scope,
            &input,
            &packed_gate,
            &weight,
            &mut packed_output,
            eps,
        )
        .unwrap();
        for (index, (&got, &expected)) in packed_output
            .to_host_vec()
            .unwrap()
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert!(
                (got - expected).abs() < 3e-6,
                "packed output mismatch at {index}: got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn bf16_qwen_value_head_shape_matches_official_order() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (tokens, value_heads, head_dim) = (3usize, 32usize, 128usize);
        let rows = tokens * value_heads;
        let input: Vec<bf16> = (0..rows * head_dim)
            .map(|index| {
                bf16::from_f32(
                    ((index as f32 * 0.017).sin() + 0.23 * (index as f32 * 0.031).cos()) * 1.3,
                )
            })
            .collect();
        let gate: Vec<bf16> = (0..rows * head_dim)
            .map(|index| bf16::from_f32(((index as f32 * 0.023).cos() - 0.15) * 1.8))
            .collect();
        let weight: Vec<f32> = (0..head_dim)
            .map(|index| 0.55 + 0.65 * (index as f32 * 0.041).cos())
            .collect();
        let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let gate_f32 = gate.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let eps = 1e-6;
        let expected = reference_f32(&input_f32, &gate_f32, &weight, head_dim, eps, |value| {
            bf16::from_f32(value).to_f32()
        });

        let input =
            Tensor::from_host_slice(&input, [tokens, value_heads, head_dim], &cuda).unwrap();
        let gate = Tensor::from_host_slice(&gate, [tokens, value_heads, head_dim], &cuda).unwrap();
        let weight = Tensor::from_host_slice(&weight, [head_dim], &cuda).unwrap();
        let mut output =
            Tensor::<bf16, Cuda>::zeros([tokens, value_heads, head_dim], &cuda).unwrap();
        <Cuda as FusedOps>::gated_rmsnorm(&scope, &input, &gate, &weight, &mut output, eps)
            .unwrap();

        let got = output.to_host_vec().unwrap();
        for (index, (&got, &expected)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() < 0.025,
                "BF16 mismatch at {index}: got={}, expected={expected}",
                got.to_f32()
            );
        }
    }

    #[test]
    fn bf16_rounds_normalized_value_before_fp32_weight_and_gate() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (value_heads, head_dim) = (32usize, 128usize);
        let mut input = vec![bf16::from_f32(0.0); value_heads * head_dim];
        let mut gate = vec![bf16::from_f32(0.0); value_heads * head_dim];
        for head in 0..value_heads {
            input[head * head_dim] = bf16::from_f32(1.0);
            gate[head * head_dim] = bf16::from_f32(1.0);
        }

        // Chosen so the final BF16 result falls on opposite sides of a
        // rounding midpoint depending on whether sqrt(128) is first rounded
        // to BF16, making the official intermediate cast observable.
        let mut weight = vec![1.0f32; head_dim];
        weight[0] = 0.121_380_13;
        let normalized_fp32 = (head_dim as f32).sqrt();
        let normalized_bf16 = bf16::from_f32(normalized_fp32).to_f32();
        let silu_one = 1.0 / (1.0 + (-1.0f32).exp());
        let official = bf16::from_f32(normalized_bf16 * weight[0] * silu_one);
        let without_intermediate_round = bf16::from_f32(normalized_fp32 * weight[0] * silu_one);
        assert_ne!(
            official.to_bits(),
            without_intermediate_round.to_bits(),
            "test values must distinguish the two numerical orders"
        );

        let input = Tensor::from_host_slice(&input, [1, value_heads, head_dim], &cuda).unwrap();
        let gate = Tensor::from_host_slice(&gate, [1, value_heads, head_dim], &cuda).unwrap();
        let weight = Tensor::from_host_slice(&weight, [head_dim], &cuda).unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([1, value_heads, head_dim], &cuda).unwrap();
        <Cuda as FusedOps>::gated_rmsnorm(&scope, &input, &gate, &weight, &mut output, 0.0)
            .unwrap();

        let got = output.to_host_vec().unwrap();
        for head in 0..value_heads {
            assert_eq!(
                got[head * head_dim].to_bits(),
                official.to_bits(),
                "wrong intermediate rounding at head {head}"
            );
            assert!(
                got[head * head_dim + 1..(head + 1) * head_dim]
                    .iter()
                    .all(|value| value.to_bits() == 0)
            );
        }
    }
}
