//! Optional Triton RMSNorm integration coverage, using an independent CPU oracle.
//!
//! Run with a GPU matching the build's CUDA_ARCH:
//! `cargo test -p infer-backend-cuda --features triton --test triton_rmsnorm -- --ignored --test-threads=1`

#![cfg(feature = "triton")]

use half::{bf16, f16};
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::dtype::{DTypeId, Dtype};
use infer_core::exec::ExecScope;
use infer_core::ports::MathOps;
use infer_core::tensor::Tensor;

const EPS: f32 = 1e-5;

fn scope() -> CudaScope {
    let scope = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .expect("create CUDA device")
    .scope();
    assert!(
        scope.device().config.triton_available(),
        "these tests require Triton kernels built for this GPU's CUDA_ARCH"
    );
    scope
}

fn data<T: Dtype>(n: usize, mut seed: u32) -> Vec<T> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            T::write_f64((seed % 65536) as f64 / 16384.0 - 2.0)
        })
        .collect()
}

fn round<T: Dtype>(x: f64) -> f64 {
    T::read_f64(&T::write_f64(x))
}

fn oracle<T: Dtype>(input: &[T], weight: &[T]) -> Vec<T> {
    let dim = weight.len();
    input
        .chunks_exact(dim)
        .flat_map(|row| {
            let sum = row.iter().map(|x| T::read_f64(x).powi(2)).sum::<f64>();
            let inverse = (sum / dim as f64 + EPS as f64).sqrt().recip();
            row.iter().zip(weight).map(move |(x, w)| {
                let (x, w) = (T::read_f64(x), T::read_f64(w));
                // CUDA's half kernels round the inverse RMS to the storage
                // dtype, round x * inverse, then round the weighted result.
                let normalized = if T::ID == DTypeId::F32 {
                    x * inverse
                } else {
                    round::<T>(x * round::<T>(inverse))
                };
                T::write_f64(normalized * w)
            })
        })
        .collect()
}

fn assert_close<T: Dtype>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    let (atol, rtol) = match T::ID {
        DTypeId::F32 => (3e-6, 3e-6),
        DTypeId::F16 => (2e-3, 2e-3),
        DTypeId::BF16 => (1.6e-2, 1.6e-2),
        _ => panic!("unexpected test dtype"),
    };
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let (actual, expected) = (T::read_f64(actual), T::read_f64(expected));
        let tolerance = atol + rtol * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{:?} element {i}: got {actual}, expected {expected}, tolerance {tolerance}",
            T::ID
        );
    }
}

fn check_contiguous<T: Dtype>(scope: &CudaScope, dim: usize, inplace: bool) {
    let host = data::<T>(6 * dim, 17);
    let weights = data::<T>(dim, 113);
    let expected = oracle(&host, &weights);
    let mut input = Tensor::from_host_slice(&host, [2, 3, dim], scope.device()).unwrap();
    let weight = Tensor::from_host_slice(&weights, [dim], scope.device()).unwrap();
    if inplace {
        Cuda::rmsnorm_inplace(scope, &mut input, &weight, EPS).unwrap();
        assert_close(&input.to_host_vec().unwrap(), &expected);
    } else {
        let mut output = Tensor::zeros([2, 3, dim], scope.device()).unwrap();
        Cuda::rmsnorm(scope, &input, &weight, &mut output, EPS).unwrap();
        assert_close(&output.to_host_vec().unwrap(), &expected);
        assert_eq!(
            input
                .to_host_vec()
                .unwrap()
                .iter()
                .map(T::read_f64)
                .collect::<Vec<_>>(),
            host.iter().map(T::read_f64).collect::<Vec<_>>()
        );
    }
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn contiguous_f32_matches_cpu_oracle() {
    let scope = scope();
    for dim in [1, 37, 256, 1025, 16384] {
        check_contiguous::<f32>(&scope, dim, false);
    }
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn contiguous_f16_matches_cpu_oracle() {
    let scope = scope();
    for dim in [1, 37, 256, 1025, 16384] {
        check_contiguous::<f16>(&scope, dim, false);
    }
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn contiguous_bf16_matches_cpu_oracle() {
    let scope = scope();
    for dim in [1, 37, 256, 1025, 16384] {
        check_contiguous::<bf16>(&scope, dim, false);
    }
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn inplace_matches_cpu_oracle() {
    let scope = scope();
    for dim in [37, 256, 1025] {
        check_contiguous::<f32>(&scope, dim, true);
        check_contiguous::<f16>(&scope, dim, true);
        check_contiguous::<bf16>(&scope, dim, true);
    }
}

fn check_half_rounding<T: Dtype>(scope: &CudaScope, value: f64) {
    let host = vec![T::write_f64(value); 37];
    let weights: Vec<_> = (0..37)
        .map(|i| T::write_f64([1.0, 1.3125, 0.625][i % 3]))
        .collect();
    let input = Tensor::from_host_slice(&host, [37], scope.device()).unwrap();
    let weight = Tensor::from_host_slice(&weights, [37], scope.device()).unwrap();
    let mut output = Tensor::zeros([37], scope.device()).unwrap();
    Cuda::rmsnorm(scope, &input, &weight, &mut output, EPS).unwrap();
    // These constants are away from inverse-RMS rounding ties, and distinguish
    // dtype-rounded scale/normalization from a single final conversion.
    assert_eq!(
        output
            .to_host_vec()
            .unwrap()
            .iter()
            .map(T::read_f64)
            .collect::<Vec<_>>(),
        oracle(&host, &weights)
            .iter()
            .map(T::read_f64)
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn half_rounding_preserves_cuda_arithmetic_boundaries() {
    let scope = scope();
    check_half_rounding::<f16>(&scope, 0.25439453125);
    check_half_rounding::<bf16>(&scope, 0.29296875);
}

fn check_strided<T: Dtype>(
    scope: &CudaScope,
    strided_input: bool,
    strided_output: bool,
    rank_three: bool,
) {
    let (rows, dim) = (6, 32);
    let input_offset = if strided_input { 8 } else { 0 };
    let input_stride = if strided_input { dim + 16 } else { dim };
    let output_offset = if strided_output { 16 } else { 0 };
    let output_stride = if strided_output { dim + 24 } else { dim };
    let host = data::<T>(rows * input_stride, 173);
    let weights = data::<T>(dim, 239);
    let logical: Vec<_> = host
        .chunks_exact(input_stride)
        .flat_map(|row| row[input_offset..input_offset + dim].iter().copied())
        .collect();
    let expected = oracle(&logical, &weights);
    let input_storage = if rank_three {
        Tensor::from_host_slice(&host, [2, 3, input_stride], scope.device())
    } else {
        Tensor::from_host_slice(&host, [rows, input_stride], scope.device())
    }
    .unwrap();
    let last_axis = if rank_three { 2 } else { 1 };
    let input = input_storage.narrow(last_axis, input_offset, dim).unwrap();
    let sentinel = T::write_f64(-17.0);
    let output_host = vec![sentinel; rows * output_stride];
    let output_storage = if rank_three {
        Tensor::from_host_slice(&output_host, [2, 3, output_stride], scope.device())
    } else {
        Tensor::from_host_slice(&output_host, [rows, output_stride], scope.device())
    }
    .unwrap();
    let mut output = output_storage
        .narrow(last_axis, output_offset, dim)
        .unwrap();
    let weight = Tensor::from_host_slice(&weights, [dim], scope.device()).unwrap();
    assert_eq!(!input.is_contiguous(), strided_input);
    assert_eq!(!output.is_contiguous(), strided_output);
    Cuda::rmsnorm(scope, &input, &weight, &mut output, EPS).unwrap();
    let actual = output_storage.to_host_vec().unwrap();
    for (row, expected) in actual
        .chunks_exact(output_stride)
        .zip(expected.chunks_exact(dim))
    {
        assert_close(&row[output_offset..output_offset + dim], expected);
        for untouched in row[..output_offset]
            .iter()
            .chain(&row[output_offset + dim..])
        {
            assert_eq!(T::read_f64(untouched), -17.0, "overwrote output padding");
        }
    }
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn strided_layouts_and_oversized_rows_use_cuda_fallback() {
    let scope = scope();
    for rank_three in [false, true] {
        for (input, output) in [(true, false), (false, true), (true, true)] {
            check_strided::<f32>(&scope, input, output, rank_three);
            check_strided::<f16>(&scope, input, output, rank_three);
            check_strided::<bf16>(&scope, input, output, rank_three);
        }
    }
    // Beyond Triton's maximum block size, but valid for CUDA's vector widths.
    check_contiguous::<f32>(&scope, 16392, false);
    check_contiguous::<f16>(&scope, 16392, false);
    check_contiguous::<bf16>(&scope, 16392, false);
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn first_invocation_is_capture_safe_and_replay_reads_updated_input() {
    let scope = scope();
    let (rows, dim) = (3, 37);
    let initial = data::<bf16>(rows * dim, 11);
    let weights = data::<bf16>(dim, 23);
    let mut input = Tensor::from_host_slice(&initial, [rows, dim], scope.device()).unwrap();
    let weight = Tensor::from_host_slice(&weights, [dim], scope.device()).unwrap();
    let mut output = Tensor::<bf16, Cuda>::zeros([rows, dim], scope.device()).unwrap();
    scope.synchronize().unwrap();

    // Deliberately omit an eager RMSNorm warmup: modules/functions must already
    // be loaded by CudaConfig initialization before the first capture starts.
    scope.graph_capture_begin().unwrap();
    Cuda::rmsnorm(&scope, &input, &weight, &mut output, EPS).unwrap();
    scope.graph_capture_end(37).unwrap();
    assert!(scope.graph_ready(37));
    scope.graph_launch(37).unwrap();
    scope.synchronize().unwrap();
    assert_close(&output.to_host_vec().unwrap(), &oracle(&initial, &weights));

    let changed = data::<bf16>(rows * dim, 991);
    input.upload_from_host(&changed).unwrap();
    scope.graph_launch(37).unwrap();
    scope.synchronize().unwrap();
    assert_close(&output.to_host_vec().unwrap(), &oracle(&changed, &weights));
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn independent_configs_keep_their_loaded_kernels_alive() {
    let first = scope();
    check_contiguous::<f32>(&first, 37, false);
    {
        let second = scope();
        check_contiguous::<bf16>(&second, 1025, false);
        let input = Tensor::<f32, Cuda>::zeros([2, 32], first.device()).unwrap();
        let weight = Tensor::<f32, Cuda>::zeros([32], first.device()).unwrap();
        let mut output = Tensor::<f32, Cuda>::zeros([2, 32], first.device()).unwrap();
        let mut foreign_input = Tensor::<f32, Cuda>::zeros([2, 32], second.device()).unwrap();
        let foreign_weight = Tensor::<f32, Cuda>::zeros([32], second.device()).unwrap();
        let mut foreign_output = Tensor::<f32, Cuda>::zeros([2, 32], second.device()).unwrap();
        assert!(Cuda::rmsnorm(&first, &foreign_input, &weight, &mut output, EPS).is_err());
        assert!(Cuda::rmsnorm(&first, &input, &foreign_weight, &mut output, EPS).is_err());
        assert!(Cuda::rmsnorm(&first, &input, &weight, &mut foreign_output, EPS).is_err());
        assert!(Cuda::rmsnorm_inplace(&first, &mut foreign_input, &weight, EPS).is_err());
        second.synchronize().unwrap();
    }
    check_contiguous::<f16>(&first, 37, false);
    first.synchronize().unwrap();
}

#[test]
#[ignore = "requires a matching CUDA GPU and the Triton build toolchain"]
fn invalid_contracts_fail_before_launch() {
    let scope = scope();
    let input = Tensor::from_host_slice(&data::<f32>(64, 17), [2, 32], scope.device()).unwrap();
    let weight = Tensor::from_host_slice(&data::<f32>(32, 23), [32], scope.device()).unwrap();
    let mut output = Tensor::<f32, Cuda>::zeros([2, 32], scope.device()).unwrap();
    for eps in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(Cuda::rmsnorm(&scope, &input, &weight, &mut output, eps).is_err());
    }
    let mut wrong_shape = Tensor::<f32, Cuda>::zeros([4, 16], scope.device()).unwrap();
    assert!(Cuda::rmsnorm(&scope, &input, &weight, &mut wrong_shape, EPS).is_err());

    let strided_weight = Tensor::from_host_slice(&data::<f32>(64, 71), [32, 2], scope.device())
        .unwrap()
        .narrow(1, 0, 1)
        .unwrap();
    assert!(!strided_weight.is_contiguous());
    assert!(Cuda::rmsnorm(&scope, &input, &strided_weight, &mut output, EPS).is_err());

    let overlapping_weight = output.narrow(0, 0, 1).unwrap();
    assert!(Cuda::rmsnorm(&scope, &input, &overlapping_weight, &mut output, EPS).is_err());
    assert!(Cuda::rmsnorm_inplace(&scope, &mut output, &overlapping_weight, EPS).is_err());
    let overlap_storage = Tensor::<f32, Cuda>::zeros([3, 32], scope.device()).unwrap();
    let overlap_input = overlap_storage.narrow(0, 0, 2).unwrap();
    let mut overlap_output = overlap_storage.narrow(0, 1, 2).unwrap();
    assert!(Cuda::rmsnorm(&scope, &overlap_input, &weight, &mut overlap_output, EPS).is_err());

    let empty = Tensor::<f32, Cuda>::zeros([2, 0], scope.device()).unwrap();
    let empty_weight = Tensor::<f32, Cuda>::zeros([0], scope.device()).unwrap();
    let mut empty_output = Tensor::<f32, Cuda>::zeros([2, 0], scope.device()).unwrap();
    assert!(Cuda::rmsnorm(&scope, &empty, &empty_weight, &mut empty_output, EPS).is_err());
    scope.synchronize().unwrap();
}
