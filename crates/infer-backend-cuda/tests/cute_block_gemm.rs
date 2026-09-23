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
        // Tile boundaries, partial tiles on both axes, and non-power-of-two K.
        for (case, (m, blocks)) in [(2, 1), (7, 2), (8, 8), (9, 3), (17, 16), (33, 64)]
            .into_iter()
            .enumerate()
        {
            let k = blocks * f.layout().elements;
            let parent_n = 64 / blocks;
            let layout = BlockQuantLayout::new(f, parent_n, k).unwrap();
            let bytes = &bytes[..layout.byte_len()];
            let mut odd = vec![0xa5];
            odd.extend_from_slice(bytes);
            let storage = Tensor::from_host_slice(&odd, [odd.len()], s.device()).unwrap();
            let parent =
                BlockQuantWeight::try_new(layout, storage.narrow(0, 1, bytes.len()).unwrap())
                    .unwrap();
            let start = usize::from(case % 2 == 1 && parent_n > 1);
            let w = parent.slice_rows(start..parent_n).unwrap();
            let view = BlockQuantView::new(layout, bytes)
                .unwrap()
                .slice_rows(start..parent_n)
                .unwrap();
            let n = parent_n - start;
            let data: Vec<T> = (0..m * k)
                .map(|i| T::write_f64(((i * 17 + i / k * 3) % 13) as f64 / 65536.0 - 6.0 / 65536.0))
                .collect();
            let bias: Vec<T> = (0..n)
                .map(|i| T::write_f64((i as f64 - 7.0) / 16.0))
                .collect();
            // Alternate padded and transposed layouts, with nonzero offsets.
            let (xs0, xs1, ys0, ys1) = if case % 2 == 0 {
                (2 * k + 3, 2, 2 * n + 3, 2)
            } else {
                (1, m + 1, 1, m + 2)
            };
            let xlen = 2 + (m - 1) * xs0 + (k - 1) * xs1;
            let ylen = 2 + (m - 1) * ys0 + (n - 1) * ys1;
            let mut padded = vec![T::write_f64(123.0); xlen];
            for t in 0..m {
                for col in 0..k {
                    padded[1 + t * xs0 + col * xs1] = data[t * k + col];
                }
            }
            let xstorage = Tensor::from_host_slice(&padded, [xlen], s.device()).unwrap();
            let x = xstorage.view_raw([m, k].into(), Strides::from_slice(&[xs0, xs1]), 1, false);
            let mut padded_bias = vec![T::write_f64(123.0); n * 3 + 1];
            for (r, &b) in bias.iter().enumerate() {
                padded_bias[1 + r * 3] = b;
            }
            let bstorage =
                Tensor::from_host_slice(&padded_bias, [padded_bias.len()], s.device()).unwrap();
            let b = bstorage.view_raw([n].into(), Strides::from_slice(&[3]), 1, false);
            let cw = BlockQuantWeight::from_host(view, &Cpu).unwrap();
            let cx = Tensor::from_host_slice(&data, [m, k], &Cpu).unwrap();
            let cb = Tensor::from_host_slice(&bias, [n], &Cpu).unwrap();
            for biased in [false, true] {
                let storage =
                    Tensor::from_host_slice(&vec![T::write_f64(123.0); ylen], [ylen], s.device())
                        .unwrap();
                let mut y =
                    storage.view_raw([m, n].into(), Strides::from_slice(&[ys0, ys1]), 1, false);
                Cuda::matmul_block_quant(&s, &x, &w, biased.then_some(&b), &mut y).unwrap();
                let actual = storage.to_host_vec().unwrap();
                let mut cy = Tensor::<T, Cpu>::zeros([m, n], &Cpu).unwrap();
                Cpu::matmul_block_quant(&cpu, &cx, &cw, biased.then_some(&cb), &mut cy).unwrap();
                let reference = cy.to_host_vec().unwrap();
                let mut touched = vec![false; ylen];
                let mut row = vec![0.0; k];
                for r in 0..n {
                    decode_row(view, r, &mut row).unwrap();
                    for t in 0..m {
                        let address = 1 + t * ys0 + r * ys1;
                        touched[address] = true;
                        let got = T::read_f64(&actual[address]);
                        let products = row
                            .iter()
                            .zip(&data[t * k..(t + 1) * k])
                            .map(|(&w, x)| w as f64 * T::read_f64(x));
                        let norm = products.clone().map(f64::abs).sum::<f64>();
                        let exact = products.sum::<f64>()
                            + if biased { T::read_f64(&bias[r]) } else { 0.0 };
                        let rounded = T::read_f64(&T::write_f64(exact));
                        if rounded.is_infinite() {
                            assert_eq!(got, rounded);
                            continue;
                        }
                        let tol = 4e-6 * norm + roundoff * exact.abs() + 1e-6;
                        assert!(
                            (got - exact).abs() <= tol,
                            "{f:?}/{:?} M={m},N={n},K={k},t={t},r={r},bias={biased}: {got} != {exact} tol={tol}",
                            T::ID
                        );
                        let cpu = T::read_f64(&reference[t * n + r]);
                        if cpu.is_finite() {
                            assert!((got - cpu).abs() <= 2.0 * tol);
                        }
                    }
                }
                for (i, value) in actual.iter().enumerate() {
                    if !touched[i] {
                        assert_eq!(T::read_f64(value), 123.0);
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
fn invalid_contracts_and_capture_replay_with_changed_input() {
    let s = scope();
    let bytes = fixture(F::Q8_0);
    let view =
        BlockQuantView::new(BlockQuantLayout::new(F::Q8_0, 64, 32).unwrap(), &bytes).unwrap();
    let w = BlockQuantWeight::from_host(view, s.device()).unwrap();
    let m = 9;
    let mut x = Tensor::from_host_slice(&vec![1.0f32; m * 32], [m, 32], s.device()).unwrap();
    let bias = Tensor::from_host_slice(&[0.5f32; 64], [64], s.device()).unwrap();
    let mut y = Tensor::from_host_slice(&vec![123.0f32; m * 64], [m, 64], s.device()).unwrap();
    for strides in [[0, 1], [64, 0], [32, 1], [2, 4]] {
        let mut invalid = y.view_raw([m, 64].into(), Strides::from_slice(&strides), 0, false);
        assert!(matches!(
            Cuda::matmul_block_quant(&s, &x, &w, None, &mut invalid),
            Err(OpError::Shape(_))
        ));
    }
    let wrong = x.narrow(1, 0, 31).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &wrong, &w, None, &mut y).is_err());
    let wrong_bias = bias.narrow(0, 0, 63).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &x, &w, Some(&wrong_bias), &mut y).is_err());
    let alias = y.narrow(1, 0, 32).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &alias, &w, None, &mut y).is_err());
    let alias = y
        .view_contiguous([m * 64].into())
        .unwrap()
        .narrow(0, 0, 64)
        .unwrap();
    assert!(Cuda::matmul_block_quant(&s, &x, &w, Some(&alias), &mut y).is_err());
    let mut alias = Tensor::<f32, Cuda>::from_raw_parts(
        std::sync::Arc::clone(w.bytes().storage()),
        [2, 64].into(),
        Strides::from_slice(&[64, 1]),
        0,
        true,
    );
    let two = x.narrow(0, 0, 2).unwrap();
    assert!(Cuda::matmul_block_quant(&s, &two, &w, None, &mut alias).is_err());
    let ints = Tensor::<i32, Cuda>::zeros([m, 32], s.device()).unwrap();
    let mut ints_out = Tensor::<i32, Cuda>::zeros([m, 64], s.device()).unwrap();
    assert!(matches!(
        Cuda::matmul_block_quant(&s, &ints, &w, None, &mut ints_out),
        Err(OpError::Unsupported { .. })
    ));
    {
        let foreign = scope();
        let fx = Tensor::<f32, Cuda>::zeros([m, 32], foreign.device()).unwrap();
        let fw = BlockQuantWeight::from_host(view, foreign.device()).unwrap();
        let fb = Tensor::<f32, Cuda>::zeros([64], foreign.device()).unwrap();
        let mut fy = Tensor::<f32, Cuda>::zeros([m, 64], foreign.device()).unwrap();
        assert!(Cuda::matmul_block_quant(&s, &fx, &w, None, &mut y).is_err());
        assert!(Cuda::matmul_block_quant(&s, &x, &fw, None, &mut y).is_err());
        assert!(Cuda::matmul_block_quant(&s, &x, &w, Some(&fb), &mut y).is_err());
        assert!(Cuda::matmul_block_quant(&s, &x, &w, None, &mut fy).is_err());
    }
    assert_eq!(y.to_host_vec().unwrap(), vec![123.0; m * 64]);
    // First GEMM invocation is captured, without eager warmup.
    s.graph_capture_begin().unwrap();
    Cuda::matmul_block_quant(&s, &x, &w, Some(&bias), &mut y).unwrap();
    s.graph_capture_end(572).unwrap();
    for replay in 0..2 {
        let data: Vec<f32> = (0..m * 32)
            .map(|i| ((i * 7 + replay * 5) % 13) as f32 / 32.0)
            .collect();
        x.upload_from_host(&data).unwrap();
        s.graph_launch(572).unwrap();
        s.synchronize().unwrap();
        let actual = y.to_host_vec().unwrap();
        for r in 0..64 {
            let mut row = vec![0.0; 32];
            decode_row(view, r, &mut row).unwrap();
            for t in 0..m {
                let prod = row
                    .iter()
                    .zip(&data[t * 32..(t + 1) * 32])
                    .map(|(&a, &b)| a as f64 * b as f64);
                let tol = prod.clone().map(f64::abs).sum::<f64>() * 4e-6 + 1e-6;
                let exact = prod.sum::<f64>() + 0.5;
                assert!((actual[t * 64 + r] as f64 - exact).abs() <= tol);
            }
        }
    }
    // Broadcast both activation dimensions and bias; transpose output.
    let bx = x.view_raw([m, 32].into(), Strides::from_slice(&[0, 0]), 0, false);
    let bb = bias.view_raw([64].into(), Strides::from_slice(&[0]), 0, false);
    let mut by = y.view_raw([m, 64].into(), Strides::from_slice(&[1, m]), 0, false);
    Cuda::matmul_block_quant(&s, &bx, &w, Some(&bb), &mut by).unwrap();
    let actual = y.to_host_vec().unwrap();
    let scalar = x.to_host_vec().unwrap()[0] as f64;
    for r in 0..64 {
        let mut row = vec![0.0; 32];
        decode_row(view, r, &mut row).unwrap();
        let exact = row.iter().map(|&a| a as f64 * scalar).sum::<f64>() + 0.5;
        let tol = row.iter().map(|&a| (a as f64 * scalar).abs()).sum::<f64>() * 4e-6 + 1e-6;
        for t in 0..m {
            assert!((actual[r * m + t] as f64 - exact).abs() <= tol);
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn rounding_and_nonfinite_do_not_leak_between_tokens() {
    fn rounding<T: Dtype>(s: &CudaScope, d: f32, bias: f32) {
        let mut bytes = vec![1u8; 34];
        bytes[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let w = BlockQuantWeight::from_host(
            BlockQuantView::new(BlockQuantLayout::new(F::Q8_0, 1, 32).unwrap(), &bytes).unwrap(),
            s.device(),
        )
        .unwrap();
        let mut data = vec![T::write_f64(0.0); 9 * 32];
        for t in 0..9 {
            data[t * 32] = T::write_f64(d as f64);
        }
        let x = Tensor::from_host_slice(&data, [9, 32], s.device()).unwrap();
        let b = Tensor::from_host_slice(&[T::write_f64(bias as f64)], [1], s.device()).unwrap();
        let mut y = Tensor::<T, Cuda>::zeros([9, 1], s.device()).unwrap();
        Cuda::matmul_block_quant(s, &x, &w, Some(&b), &mut y).unwrap();
        let expected = T::read_f64(&T::write_f64((d * d + bias) as f64));
        assert!(expected > 0.0);
        for v in y.to_host_vec().unwrap() {
            assert_eq!(T::read_f64(&v), expected);
        }
    }
    let s = scope();
    rounding::<half::f16>(&s, 1.0009766, -1.0019531);
    rounding::<half::bf16>(&s, 1.0078125, -1.015625);
    let mut bytes = vec![1u8; 34];
    bytes[..2].copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
    let w = BlockQuantWeight::from_host(
        BlockQuantView::new(BlockQuantLayout::new(F::Q8_0, 1, 32).unwrap(), &bytes).unwrap(),
        s.device(),
    )
    .unwrap();
    let data: Vec<f32> = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        0.0,
        2.0,
        -1.0,
        0.5,
        3.0,
    ]
    .into_iter()
    .flat_map(|x| [x; 32])
    .collect();
    let x = Tensor::from_host_slice(&data, [9, 32], s.device()).unwrap();
    let mut y = Tensor::<f32, Cuda>::zeros([9, 1], s.device()).unwrap();
    Cuda::matmul_block_quant(&s, &x, &w, None, &mut y).unwrap();
    let actual = y.to_host_vec().unwrap();
    assert!(actual[0].is_nan());
    assert_eq!(
        &actual[1..],
        &[
            f32::INFINITY,
            f32::NEG_INFINITY,
            32.0,
            0.0,
            64.0,
            -32.0,
            16.0,
            96.0
        ]
    );
}
