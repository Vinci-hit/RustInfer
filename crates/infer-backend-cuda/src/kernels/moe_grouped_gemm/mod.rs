//! Correctness-first BF16 grouped GEMM for dense expert weights.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use half::bf16;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn grouped_expert_gemm_bf16(
        input: *const bf16,
        weights: *const bf16,
        output: *mut bf16,
        expert_offsets: *const i32,
        rows: i32,
        experts: i32,
        out_features: i32,
        in_features: i32,
        stream: cudaStream_t,
    );
}

pub fn grouped_gemm_bf16(
    stream: cudaStream_t,
    input: &Tensor<bf16, Cuda>,
    weights: &Tensor<bf16, Cuda>,
    output: &mut Tensor<bf16, Cuda>,
    expert_offsets: &Tensor<i32, Cuda>,
) -> OpResult<()> {
    for (name, contiguous) in [
        ("input", input.is_contiguous()),
        ("weights", weights.is_contiguous()),
        ("output", output.is_contiguous()),
        ("expert_offsets", expert_offsets.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::Shape(format!(
                "grouped_expert_gemm {} must be contiguous",
                name
            )));
        }
    }

    let input_shape = input.shape().as_slice();
    let weight_shape = weights.shape().as_slice();
    let output_shape = output.shape().as_slice();
    if input_shape.len() != 2 || input_shape.contains(&0) {
        return Err(OpError::Shape(format!(
            "grouped_expert_gemm input must be non-empty [rows,in], got {:?}",
            input_shape
        )));
    }
    if weight_shape.len() != 3 || weight_shape.contains(&0) {
        return Err(OpError::Shape(format!(
            "grouped_expert_gemm weights must be non-empty [experts,out,in], got {:?}",
            weight_shape
        )));
    }
    let rows = input_shape[0];
    let in_features = input_shape[1];
    let experts = weight_shape[0];
    let out_features = weight_shape[1];
    if weight_shape[2] != in_features || output_shape != [rows, out_features] {
        return Err(OpError::Shape(format!(
            "grouped_expert_gemm incompatible input {:?}, weights {:?}, output {:?}",
            input_shape, weight_shape, output_shape
        )));
    }
    if expert_offsets.shape().as_slice() != [experts + 1] {
        return Err(OpError::Shape(format!(
            "grouped_expert_gemm offsets must be [{}], got {:?}",
            experts + 1,
            expert_offsets.shape().as_slice()
        )));
    }
    let output_elements = rows
        .checked_mul(out_features)
        .ok_or_else(|| OpError::Shape("grouped_expert_gemm output size overflows".into()))?;
    const THREADS: usize = 256;
    let blocks = output_elements.div_ceil(THREADS);
    let _ = to_i32(blocks, "grid block count")?;

    let rows = to_i32(rows, "row count")?;
    let experts = to_i32(experts, "expert count")?;
    let out_features = to_i32(out_features, "output feature count")?;
    let in_features = to_i32(in_features, "input feature count")?;

    unsafe {
        grouped_expert_gemm_bf16(
            input.data_ptr(),
            weights.data_ptr(),
            output.data_ptr_mut(),
            expert_offsets.data_ptr(),
            rows,
            experts,
            out_features,
            in_features,
            stream,
        );
    }
    Ok(())
}

fn to_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value)
        .map_err(|_| OpError::Shape(format!("grouped_expert_gemm {} exceeds i32", name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16s(values: &[f32]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    #[test]
    fn grouped_gemm_uses_offsets_and_skips_empty_experts() {
        let cuda = Cuda::new(0).unwrap();
        let input = Tensor::from_host_slice(
            &bf16s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            [4, 2],
            &cuda,
        )
        .unwrap();
        let weights = Tensor::from_host_slice(
            &bf16s(&[
                1.0, 0.0, 0.0, 1.0, // expert 0: identity
                9.0, 9.0, 9.0, 9.0, // expert 1: empty
                1.0, 1.0, 1.0, -1.0, // expert 2: sum/difference
            ]),
            [3, 2, 2],
            &cuda,
        )
        .unwrap();
        let offsets = Tensor::from_host_slice(&[0i32, 2, 2, 4], [4], &cuda).unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([4, 2], &cuda).unwrap();

        grouped_gemm_bf16(cuda.config.stream, &input, &weights, &mut output, &offsets).unwrap();

        let got = output
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 11.0, -1.0, 15.0, -1.0]);
    }
}
