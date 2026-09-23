use super::*;
use infer_backend_cpu::block_quant::decode_row;
use infer_backend_cuda::{Cuda, CudaMemoryPlan};
use infer_core::{dtype::Dtype, exec::ExecScope, ports::MathOps, types::Strides};

fn check_real<T: Dtype>(scope: &infer_backend_cuda::CudaScope, reader: &GgufReader, rounding: f64) {
    // Real rows of every format, using their real K rather than fixture widths.
    let mut seen = std::collections::HashSet::new();
    for info in reader.tensors() {
        let Some(f) = info.ggml_type().block_quant_format() else {
            continue;
        };
        if !seen.insert(f) {
            continue;
        }
        let view = block_quant_view(reader.read_view(info.name()).unwrap()).unwrap();
        let [n, k] = view.layout().shape();
        let view = view.slice_rows(n - 3..n).unwrap();
        let w = BlockQuantWeight::from_host(view, scope.device()).unwrap();
        let m = 9;
        let data: Vec<T> = (0..m * k)
            .map(|i| T::write_f64((((i * 17 + i / k * 3) % 23) as f64 - 11.0) / 16.0))
            .collect();
        let x = Tensor::from_host_slice(&data, [m, k], scope.device()).unwrap();
        let mut out = Tensor::<T, Cuda>::zeros([m, 3], scope.device()).unwrap();
        Cuda::matmul_block_quant(scope, &x, &w, None, &mut out).unwrap();
        let actual = out.to_host_vec().unwrap();
        let mut row = vec![0.0; k];
        for (r, actual) in actual.iter().enumerate() {
            decode_row(view, r % 3, &mut row).unwrap();
            let products = row
                .iter()
                .zip(&data[(r / 3) * k..(r / 3 + 1) * k])
                .map(|(&w, x)| w as f64 * T::read_f64(x));
            let norm = products.clone().map(f64::abs).sum::<f64>();
            let exact = products.sum::<f64>();
            assert!(
                (T::read_f64(actual) - exact).abs() <= 4e-6 * norm + rounding * exact.abs() + 1e-6,
                "{} {:?} row={r}",
                info.name(),
                T::ID
            );
        }
    }
    assert_eq!(seen.len(), 13);
    eprintln!("Real GGUF GEMM M=9: 13 formats / {:?} passed", T::ID);
}

#[test]
#[ignore = "requires CUDA and RUSTINFER_GGUF_MODEL"]
fn block_quant_local_cuda_gemm() {
    let reader =
        GgufReader::open(std::env::var_os("RUSTINFER_GGUF_MODEL").expect("RUSTINFER_GGUF_MODEL"))
            .unwrap();
    let scope = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
    .scope();
    assert!(scope.device().config.cute_dsl_available());
    check_real::<f32>(&scope, &reader, 2e-7);
    check_real::<half::f16>(&scope, &reader, 0.001);
    check_real::<half::bf16>(&scope, &reader, 0.008);
    let mut gpu = vec![];
    let mut views = vec![];
    let mut bytes = 0;
    for name in [
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
    ] {
        let v = block_quant_view(reader.read_view(name).unwrap()).unwrap();
        bytes += v.bytes().len();
        gpu.push(BlockQuantWeight::from_host(v, scope.device()).unwrap());
        views.push(v);
    }
    let projection = BlockQuantProjection::try_new(gpu).unwrap();
    let [n, k] = projection.shape();
    assert_eq!([n, k], [14336, 5120]);
    let bias_data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
    let bias = Tensor::from_host_slice(&bias_data, [n], scope.device()).unwrap();
    let linear = Linear::from_block_quant(projection, Some(bias)).unwrap();
    let m = 17;
    let data: Vec<f32> = (0..m * k)
        .map(|i| (((i * 17 + i / k * 3) % 23) as f32 - 11.0) / 16.0)
        .collect();
    let x = Tensor::from_host_slice(&data, [m, k], scope.device()).unwrap();
    let pitch = n * 2 + 3;
    let storage =
        Tensor::from_host_slice(&vec![123.0f32; m * pitch], [m, pitch], scope.device()).unwrap();
    let mut y = storage.view_raw([m, n].into(), Strides::from_slice(&[pitch, 2]), 1, false);
    let mut p = plan();
    p.kind = BatchKind::Ragged;
    p.num_tokens = m;
    p.batch = 1;
    p.q_lens = vec![m as i32];
    p.kv_lens = vec![m as i32];
    p.seq_positions = vec![0];
    p.rope_positions = (0..m as i32).collect();
    let ctx = StepCtx::new(&scope, &p);
    // Mixed projection itself must also be capture-safe.
    scope.graph_capture_begin().unwrap();
    linear.forward(&x, &mut y, &ctx).unwrap();
    scope.graph_capture_end(573).unwrap();
    scope.graph_launch(573).unwrap();
    scope.synchronize().unwrap();
    let actual = storage.to_host_vec().unwrap();
    let single = Tensor::<f32, Cuda>::zeros([m, n], scope.device()).unwrap();
    for t in 0..m {
        let tx = x.narrow(0, t, 1).unwrap();
        let mut ty = single.narrow(0, t, 1).unwrap();
        linear.forward(&tx, &mut ty, &ctx).unwrap();
    }
    let gemv = single.to_host_vec().unwrap();
    for t in 0..m {
        for j in 0..pitch {
            if j >= 1 && j < 1 + 2 * n && j % 2 == 1 {
                assert_eq!(
                    actual[t * pitch + j],
                    gemv[t * n + (j - 1) / 2],
                    "token {t}, output {}",
                    (j - 1) / 2
                );
            } else {
                assert_eq!(actual[t * pitch + j], 123.0);
            }
        }
    }
    // Independent CPU Linear and FP64 reference at first/last/tile-boundary/
    // middle channels of every projection, for every token. The complete GPU
    // output was checked above against the independently validated GEMV path.
    let mut cpu_parts = vec![];
    let mut sampled = vec![];
    let mut sampled_bias = vec![];
    let mut dense = vec![];
    let mut base = 0;
    for v in views {
        let rows = v.layout().shape()[0];
        for r in [0, 3, 4, rows / 2, rows - 1] {
            cpu_parts
                .push(BlockQuantWeight::from_host(v.slice_rows(r..r + 1).unwrap(), &Cpu).unwrap());
            sampled.push(base + r);
            sampled_bias.push(bias_data[base + r]);
            let mut row = vec![0.0; k];
            decode_row(v, r, &mut row).unwrap();
            dense.push(row);
        }
        base += rows;
    }
    let cpu_scope = HostScope::new(Cpu);
    let cpu_ctx = StepCtx::new(&cpu_scope, &p);
    let cpu_linear = Linear::from_block_quant(
        BlockQuantProjection::try_new(cpu_parts).unwrap(),
        Some(Tensor::from_host_slice(&sampled_bias, [sampled.len()], &Cpu).unwrap()),
    )
    .unwrap();
    let cx = Tensor::from_host_slice(&data, [m, k], &Cpu).unwrap();
    let mut cy = Tensor::<f32, Cpu>::zeros([m, sampled.len()], &Cpu).unwrap();
    cpu_linear.forward(&cx, &mut cy, &cpu_ctx).unwrap();
    let cpu_values = cy.to_host_vec().unwrap();
    let mut max_error = 0.0f64;
    let mut max_cpu_error = 0.0f64;
    for t in 0..m {
        for (j, &r) in sampled.iter().enumerate() {
            let prod = dense[j]
                .iter()
                .zip(&data[t * k..(t + 1) * k])
                .map(|(&w, &x)| w as f64 * x as f64);
            let norm = prod.clone().map(f64::abs).sum::<f64>();
            let exact = prod.sum::<f64>() + bias_data[r] as f64;
            let got = actual[t * pitch + 1 + 2 * r] as f64;
            let error = (got - exact).abs();
            max_error = max_error.max(error);
            let cpu_error = (got - cpu_values[t * sampled.len() + j] as f64).abs();
            max_cpu_error = max_cpu_error.max(cpu_error);
            assert!(error <= 4e-6 * norm + 1e-6);
            assert!(cpu_error <= 2e-5 * norm + 2e-6);
        }
    }
    eprintln!(
        "Full QKV GEMM [{m},{k}] x [{n},{k}]: {bytes} compressed bytes, all outputs equal GEMV; {} CPU/FP64 sampled outputs, max FP64 error={max_error:.8}, max CPU error={max_cpu_error:.8}",
        m * sampled.len()
    );
    // Compare warmed batches against issuing one GEMV per token. Event times
    // include host launch gaps, and must not be interpreted as model tokens/s.
    for m in [2, 8, 17, 64] {
        let x = Tensor::from_host_slice(&vec![0.125f32; m * k], [m, k], scope.device()).unwrap();
        let mut y = Tensor::<f32, Cuda>::zeros([m, n], scope.device()).unwrap();
        for _ in 0..3 {
            linear.forward(&x, &mut y, &ctx).unwrap();
        }
        scope.synchronize().unwrap();
        let mut timer = scope.create_timer().unwrap().unwrap();
        timer.start().unwrap();
        for _ in 0..10 {
            linear.forward(&x, &mut y, &ctx).unwrap();
        }
        timer.stop().unwrap();
        scope.synchronize().unwrap();
        let gemm_ms = timer.elapsed_ms().unwrap().unwrap() / 10.0;
        timer.start().unwrap();
        for _ in 0..10 {
            for t in 0..m {
                linear
                    .forward(
                        &x.narrow(0, t, 1).unwrap(),
                        &mut y.narrow(0, t, 1).unwrap(),
                        &ctx,
                    )
                    .unwrap();
            }
        }
        timer.stop().unwrap();
        scope.synchronize().unwrap();
        let gemv_ms = timer.elapsed_ms().unwrap().unwrap() / 10.0;
        eprintln!(
            "Mixed QKV FP32 M={m}: GEMM={gemm_ms:.3} ms, repeated GEMV={gemv_ms:.3} ms, ratio={:.2}x",
            gemv_ms / gemm_ms
        );
    }
}
