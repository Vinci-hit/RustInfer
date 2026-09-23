//! Run with --features cute-dsl -- --ignored --test-threads=1.
#![cfg(feature = "cute-dsl")]
use infer_backend_cpu::{Cpu, block_quant::decode_row};
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::{
    dtype::{Dtype, quant::BlockQuantFormat as F},
    exec::{ExecScope, HostScope},
    ports::{MathOps, OpError},
    quantized::{BlockQuantLayout, BlockQuantView, BlockQuantWeight},
    tensor::Tensor,
    types::Strides,
};

fn scope() -> CudaScope {
    let s = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
    .scope();
    assert!(s.device().config.cute_dsl_available());
    s
}
fn fixture(f: F) -> Vec<u8> {
    let file = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../infer-backend-cpu/tests/fixtures/block_quant/{f:?}.bin"
        )),
    )
    .unwrap();
    let b = f.layout();
    (0..64)
        .flat_map(|i| {
            file[4 + i * (b.bytes + 4 * b.elements)..4 + i * (b.bytes + 4 * b.elements) + b.bytes]
                .iter()
                .copied()
        })
        .collect()
}
fn check<T: Dtype>(roundoff: f64) {
    let s = scope();
    let cpu = HostScope::new(Cpu);
    for &f in F::ALL {
        let bytes = fixture(f);
        for blocks in [1, 2, 8, 64] {
            let k = blocks * f.layout().elements;
            let n = 64 / blocks;
            let l = BlockQuantLayout::new(f, n, k).unwrap();
            let view = BlockQuantView::new(l, &bytes).unwrap();
            let mut padded = vec![0xa5];
            padded.extend_from_slice(&bytes);
            let storage = Tensor::from_host_slice(&padded, [padded.len()], s.device()).unwrap();
            let parent =
                BlockQuantWeight::try_new(l, storage.narrow(0, 1, bytes.len()).unwrap()).unwrap();
            for sliced in [false, true] {
                let start = if sliced && n > 1 { 1 } else { 0 };
                let w = parent.slice_rows(start..n).unwrap();
                let v = view.slice_rows(start..n).unwrap();
                let n = n - start;
                let x: Vec<T> = (0..k)
                    .map(|i| T::write_f64(((i * 17 % 13) as f64 - 6.0) / 65536.0))
                    .collect();
                let bias: Vec<T> = (0..n)
                    .map(|i| T::write_f64((i as f64 - 7.0) / 16.0))
                    .collect();
                let mut padded_x = vec![T::write_f64(123.0); k * 2 + 1];
                for i in 0..k {
                    padded_x[1 + i * 2] = x[i];
                }
                let x_storage =
                    Tensor::from_host_slice(&padded_x, [padded_x.len()], s.device()).unwrap();
                let input =
                    x_storage.view_raw([1, k].into(), Strides::from_slice(&[0, 2]), 1, false);
                let mut padded_b = vec![T::write_f64(123.0); n * 3 + 1];
                for i in 0..n {
                    padded_b[1 + i * 3] = bias[i];
                }
                let b_storage =
                    Tensor::from_host_slice(&padded_b, [padded_b.len()], s.device()).unwrap();
                let b = b_storage.view_raw([n].into(), Strides::from_slice(&[3]), 1, false);
                for biased in [false, true] {
                    let storage = Tensor::from_host_slice(
                        &vec![T::write_f64(123.0); n * 2 + 2],
                        [n * 2 + 2],
                        s.device(),
                    )
                    .unwrap();
                    let mut out =
                        storage.view_raw([1, n].into(), Strides::from_slice(&[0, 2]), 1, false);
                    Cuda::matmul_block_quant(&s, &input, &w, biased.then_some(&b), &mut out)
                        .unwrap();
                    s.synchronize().unwrap();
                    let got = storage.to_host_vec().unwrap();
                    let cx = Tensor::from_host_slice(&x, [1, k], &Cpu).unwrap();
                    let cb = Tensor::from_host_slice(&bias, [n], &Cpu).unwrap();
                    let cw = BlockQuantWeight::from_host(v, &Cpu).unwrap();
                    let mut cy = Tensor::<T, Cpu>::zeros([1, n], &Cpu).unwrap();
                    Cpu::matmul_block_quant(&cpu, &cx, &cw, biased.then_some(&cb), &mut cy)
                        .unwrap();
                    let cpu_values = cy.to_host_vec().unwrap();
                    let mut row = vec![0.0; k];
                    for (j, value) in got.iter().enumerate() {
                        let actual = T::read_f64(value);
                        if j % 2 == 0 || j >= 2 * n {
                            assert_eq!(actual, 123.0);
                            continue;
                        }
                        let r = (j - 1) / 2;
                        decode_row(v, r, &mut row).unwrap();
                        let products = row.iter().zip(&x).map(|(&w, x)| w as f64 * T::read_f64(x));
                        let l1: f64 = products.clone().map(f64::abs).sum();
                        let exact = products.sum::<f64>()
                            + if biased { T::read_f64(&bias[r]) } else { 0.0 };
                        let rounded = T::read_f64(&T::write_f64(exact));
                        let reference = T::read_f64(&cpu_values[r]);
                        if rounded.is_infinite() {
                            assert_eq!(actual, rounded);
                            continue;
                        }
                        // FP32 reduction reorders addition; include a bound scaled
                        // by sum(abs(products)), plus one output rounding unit.
                        let tolerance = 4e-6 * l1 + roundoff * exact.abs() + 1e-6;
                        assert!(
                            (actual - exact).abs() <= tolerance,
                            "{f:?}/{:?} blocks={blocks} sliced={sliced} bias={biased} row={r}: {actual} != {exact}, tol={tolerance}",
                            T::ID
                        );
                        if reference.is_finite() {
                            assert!((actual - reference).abs() <= 2.0 * tolerance);
                        }
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "requires CUDA"]
fn f32_all_formats() {
    check::<f32>(2e-7);
}
#[test]
#[ignore = "requires CUDA"]
fn f16_all_formats() {
    check::<half::f16>(0.001);
}
#[test]
#[ignore = "requires CUDA"]
fn bf16_all_formats() {
    check::<half::bf16>(0.008);
}

#[test]
#[ignore = "requires CUDA"]
fn validation_capture_broadcast_and_nonfinite() {
    let s = scope();
    let bytes = fixture(F::Q8_0);
    let v = BlockQuantView::new(BlockQuantLayout::new(F::Q8_0, 64, 32).unwrap(), &bytes).unwrap();
    let w = BlockQuantWeight::from_host(v, s.device()).unwrap();
    let x = Tensor::from_host_slice(&[1.0f32; 32], [1, 32], s.device()).unwrap();
    let bias = Tensor::from_host_slice(&[0.5f32; 64], [64], s.device()).unwrap();
    let mut out = Tensor::from_host_slice(&[123.0f32; 64], [1, 64], s.device()).unwrap();
    let wrong = x.narrow(1, 0, 31).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &wrong, &w, None, &mut out).is_err());
    let bad_bias = bias.narrow(0, 0, 63).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &x, &w, Some(&bad_bias), &mut out).is_err());
    let mut overlapping = out.view_raw([1, 64].into(), Strides::from_slice(&[0, 0]), 0, false);
    assert!(Cuda::matmul_block_quant(&s, &x, &w, None, &mut overlapping).is_err());
    let alias = out.narrow(1, 0, 32).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &alias, &w, None, &mut out).is_err());
    let alias = out.view_contiguous([64].into()).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &x, &w, Some(&alias), &mut out).is_err());
    let mut alias = Tensor::<f32, Cuda>::from_raw_parts(
        std::sync::Arc::clone(w.bytes().storage()),
        [1, 64].into(),
        Strides::from_slice(&[64, 1]),
        0,
        true,
    );
    assert!(Cuda::matmul_block_quant(&s, &x, &w, None, &mut alias).is_err());
    let multi = Tensor::<f32, Cuda>::zeros([2, 32], s.device()).unwrap();
    let mut multi_out = Tensor::<f32, Cuda>::zeros([2, 64], s.device()).unwrap();
    Cuda::matmul_block_quant(&s, &multi, &w, None, &mut multi_out).unwrap();
    assert_eq!(multi_out.to_host_vec().unwrap(), vec![0.0; 128]);
    let integer = Tensor::<i32, Cuda>::zeros([1, 32], s.device()).unwrap();
    let mut integer_out = Tensor::<i32, Cuda>::zeros([1, 64], s.device()).unwrap();
    assert!(matches!(
        Cuda::matmul_block_quant(&s, &integer, &w, None, &mut integer_out),
        Err(OpError::Unsupported { .. })
    ));
    {
        let other = scope();
        let foreign_x = Tensor::<f32, Cuda>::zeros([1, 32], other.device()).unwrap();
        let foreign_b = Tensor::<f32, Cuda>::zeros([64], other.device()).unwrap();
        let foreign_w = BlockQuantWeight::from_host(v, other.device()).unwrap();
        let mut foreign_out = Tensor::<f32, Cuda>::zeros([1, 64], other.device()).unwrap();
        assert!(Cuda::matmul_block_quant(&s, &foreign_x, &w, None, &mut out).is_err());
        assert!(Cuda::matmul_block_quant(&s, &x, &w, Some(&foreign_b), &mut out).is_err());
        assert!(Cuda::matmul_block_quant(&s, &x, &foreign_w, None, &mut out).is_err());
        assert!(Cuda::matmul_block_quant(&s, &x, &w, None, &mut foreign_out).is_err());
    }
    assert_eq!(out.to_host_vec().unwrap(), [123.0; 64]);
    let broadcast_x = x.view_raw([1, 32].into(), Strides::from_slice(&[0, 0]), 0, false);
    let broadcast_b = bias.view_raw([64].into(), Strides::from_slice(&[0]), 0, false);
    s.graph_capture_begin().unwrap();
    Cuda::matmul_block_quant(&s, &broadcast_x, &w, Some(&broadcast_b), &mut out).unwrap();
    s.graph_capture_end(571).unwrap();
    for _ in 0..2 {
        s.graph_launch(571).unwrap();
        s.synchronize().unwrap();
        let got = out.to_host_vec().unwrap();
        for (r, &got) in got.iter().enumerate() {
            let mut row = vec![0.0; 32];
            decode_row(v, r, &mut row).unwrap();
            let exact = row.iter().map(|&v| v as f64).sum::<f64>() + 0.5;
            let tol = row.iter().map(|v| v.abs() as f64).sum::<f64>() * 2e-6 + 1e-6;
            assert!((got as f64 - exact).abs() <= tol);
        }
    }
    let empty = Tensor::<f32, Cuda>::zeros([0, 32], s.device()).unwrap();
    let mut empty_out = Tensor::<f32, Cuda>::zeros([0, 64], s.device()).unwrap();
    Cuda::matmul_block_quant(&s, &empty, &w, None, &mut empty_out).unwrap();
    // Every Q8 value is 1; special activations must propagate through reduction.
    let mut bytes = vec![0u8; 34];
    bytes[0..2].copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
    bytes[2..].fill(1);
    let w = BlockQuantWeight::from_host(
        BlockQuantView::new(BlockQuantLayout::new(F::Q8_0, 1, 32).unwrap(), &bytes).unwrap(),
        s.device(),
    )
    .unwrap();
    for special in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = Tensor::from_host_slice(&[special; 32], [1, 32], s.device()).unwrap();
        let mut y = Tensor::<f32, Cuda>::zeros([1, 1], s.device()).unwrap();
        Cuda::matmul_block_quant(&s, &x, &w, None, &mut y).unwrap();
        let got = y.to_host_vec().unwrap()[0];
        if special.is_nan() {
            assert!(got.is_nan());
        } else {
            assert_eq!(got, special);
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn bias_is_added_before_low_precision_rounding() {
    fn check<T: Dtype>(s: &CudaScope, d: f32, bias: f32) {
        let mut bytes = vec![1u8; 34];
        bytes[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let view =
            BlockQuantView::new(BlockQuantLayout::new(F::Q8_0, 1, 32).unwrap(), &bytes).unwrap();
        let w = BlockQuantWeight::from_host(view, s.device()).unwrap();
        let mut x = vec![T::write_f64(0.0); 32];
        x[0] = T::write_f64(d as f64);
        let x = Tensor::from_host_slice(&x, [1, 32], s.device()).unwrap();
        let b = Tensor::from_host_slice(&[T::write_f64(bias as f64)], [1], s.device()).unwrap();
        let mut y = Tensor::<T, Cuda>::zeros([1, 1], s.device()).unwrap();
        Cuda::matmul_block_quant(s, &x, &w, Some(&b), &mut y).unwrap();
        let expected = T::read_f64(&T::write_f64((d * d + bias) as f64));
        assert!(expected > 0.0);
        assert_eq!(T::read_f64(&y.to_host_vec().unwrap()[0]), expected);
    }
    let s = scope();
    // Premature FP16/BF16 product rounding would make both results zero.
    check::<half::f16>(&s, 1.0009766, -1.0019531);
    check::<half::bf16>(&s, 1.0078125, -1.015625);
}
