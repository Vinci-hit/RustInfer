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
        let data: Vec<T> = (0..k)
            .map(|i| T::write_f64(((i * 17 % 23) as f64 - 11.0) / 16.0))
            .collect();
        let x = Tensor::from_host_slice(&data, [1, k], scope.device()).unwrap();
        let mut out = Tensor::<T, Cuda>::zeros([1, 3], scope.device()).unwrap();
        Cuda::matmul_block_quant(scope, &x, &w, None, &mut out).unwrap();
        let actual = out.to_host_vec().unwrap();
        let mut row = vec![0.0; k];
        for (r, actual) in actual.iter().enumerate() {
            decode_row(view, r, &mut row).unwrap();
            let products = row
                .iter()
                .zip(&data)
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
    eprintln!("Real GGUF GEMV: 13 formats / {:?} passed", T::ID);
}

#[test]
#[ignore = "requires CUDA and RUSTINFER_GGUF_MODEL"]
fn block_quant_local_cuda_gemv() {
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
    // Full layer 3 projections, including Q gate: IQ4_NL + Q4_K + Q5_K.
    let mut gpu = vec![];
    let mut cpu = vec![];
    let mut views = vec![];
    let mut bytes = 0;
    for name in [
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
    ] {
        let view = block_quant_view(reader.read_view(name).unwrap()).unwrap();
        bytes += view.bytes().len();
        gpu.push(BlockQuantWeight::from_host(view, scope.device()).unwrap());
        cpu.push(BlockQuantWeight::from_host(view, &Cpu).unwrap());
        views.push(view);
    }
    let projection = BlockQuantProjection::try_new(gpu).unwrap();
    let [n, k] = projection.shape();
    assert_eq!([n, k], [14336, 5120]);
    let bias_data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
    let bias = Tensor::from_host_slice(&bias_data, [n], scope.device()).unwrap();
    let linear = Linear::from_block_quant(projection, Some(bias)).unwrap();
    let data: Vec<f32> = (0..k)
        .map(|i| ((i * 17 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let x = Tensor::from_host_slice(&data, [1, k], scope.device()).unwrap();
    let storage =
        Tensor::from_host_slice(&vec![123.0f32; n * 2 + 2], [n * 2 + 2], scope.device()).unwrap();
    let mut y = storage.view_raw([1, n].into(), Strides::from_slice(&[0, 2]), 1, false);
    let mut p = plan();
    p.num_tokens = 1;
    p.batch = 1;
    p.q_lens = vec![1];
    p.kv_lens = vec![1];
    p.seq_positions = vec![0];
    p.rope_positions = vec![0];
    let ctx = StepCtx::new(&scope, &p);
    linear.forward(&x, &mut y, &ctx).unwrap();
    scope.synchronize().unwrap();
    let actual = storage.to_host_vec().unwrap();
    // Compare the complete projection with the native CPU Linear, then FP64
    // dot products from CPU-decoded rows. No dense GPU matrix is allocated.
    let cpu_scope = HostScope::new(Cpu);
    let cpu_ctx = StepCtx::new(&cpu_scope, &p);
    let cpu_linear = Linear::from_block_quant(
        BlockQuantProjection::try_new(cpu).unwrap(),
        Some(Tensor::from_host_slice(&bias_data, [n], &Cpu).unwrap()),
    )
    .unwrap();
    let cpu_x = Tensor::from_host_slice(&data, [1, k], &Cpu).unwrap();
    let mut cpu_y = Tensor::<f32, Cpu>::zeros([1, n], &Cpu).unwrap();
    cpu_linear.forward(&cpu_x, &mut cpu_y, &cpu_ctx).unwrap();
    let cpu_values = cpu_y.to_host_vec().unwrap();
    let mut base = 0;
    let mut max_error = 0.0f64;
    let mut max_cpu_error = 0.0f64;
    for view in views {
        let mut row = vec![0.0; k];
        for r in 0..view.layout().shape()[0] {
            decode_row(view, r, &mut row).unwrap();
            let products = row.iter().zip(&data).map(|(&w, &x)| w as f64 * x as f64);
            let norm = products.clone().map(f64::abs).sum::<f64>();
            let exact = products.sum::<f64>() + bias_data[base + r] as f64;
            let got = actual[1 + (base + r) * 2] as f64;
            let error = (got - exact).abs();
            max_error = max_error.max(error);
            let cpu_error = (got - cpu_values[base + r] as f64).abs();
            max_cpu_error = max_cpu_error.max(cpu_error);
            assert!(
                error <= norm * 4e-6 + 1e-6,
                "row {}: {got} != {exact}",
                base + r
            );
            assert!(cpu_error <= norm * 2e-5 + 2e-6);
        }
        base += view.layout().shape()[0];
    }
    for (i, &v) in actual.iter().enumerate() {
        if i % 2 == 0 || i >= n * 2 {
            assert_eq!(v, 123.0);
        }
    }
    // Event timing covers 30 warmed full mixed projections, including launch
    // gaps; this is a baseline, not end-to-end token throughput.
    for _ in 0..3 {
        linear.forward(&x, &mut y, &ctx).unwrap();
    }
    scope.synchronize().unwrap();
    let mut timer = scope.create_timer().unwrap().unwrap();
    timer.start().unwrap();
    for _ in 0..30 {
        linear.forward(&x, &mut y, &ctx).unwrap();
    }
    timer.stop().unwrap();
    scope.synchronize().unwrap();
    let ms = timer.elapsed_ms().unwrap().unwrap() / 30.0;
    eprintln!(
        "Full layer 3 mixed QKV GEMV [{n},{k}]: compressed={bytes} bytes, max FP64 error={max_error:.8}, max CPU error={max_cpu_error:.8}, event mean={ms:.3} ms (30 iterations)"
    );
}
