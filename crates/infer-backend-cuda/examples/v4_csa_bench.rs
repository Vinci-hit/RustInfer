//! CUDA-event graph microbenchmarks, excluding projection GEMMs and transfers.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;

fn measure(
    s: &CudaScope,
    key: u64,
    calls_per_graph: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut timer = s.create_timer()?.ok_or("CUDA timer unavailable")?;
    for _ in 0..5 {
        s.graph_launch(key)?;
    }
    s.synchronize()?;
    let before = s.device().config.pool_stats();
    let mut samples = Vec::new();
    for _ in 0..7 {
        timer.start()?;
        for _ in 0..5 {
            s.graph_launch(key)?;
        }
        timer.stop()?;
        s.synchronize()?;
        samples.push(
            timer.elapsed_ms()?.ok_or("timer incomplete")? * 1000.0 / (5 * calls_per_graph) as f32,
        );
    }
    assert_eq!(s.device().config.pool_stats(), before);
    samples.sort_by(f32::total_cmp);
    Ok(samples[3])
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 10007) as f32 * 0.137).sin()).collect()
}
fn rope_table(capacity: usize) -> Vec<f32> {
    (0..capacity)
        .flat_map(|b| {
            (0..32).flat_map(move |d| {
                let angle = (b * 4) as f32 / 10000f32.powf(d as f32 / 32.0);
                [angle.cos(), angle.sin()]
            })
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )?
    .scope();
    let _active = s.enter();
    let dev = s.device();
    let ape = Tensor::from_host_slice(&values(4 * 1024), [4, 1024], dev)?;
    let norm = Tensor::from_host_slice(&[1.0f32; 512], [512], dev)?;
    for n in [128, 512, 1024, 4096] {
        let cap = n / 4 + 1;
        let v = Tensor::from_host_slice(&values(n * 1024), [n, 1024], dev)?;
        let g = Tensor::from_host_slice(
            &values(n * 1024)
                .into_iter()
                .map(|v| v * 3.0)
                .collect::<Vec<_>>(),
            [n, 1024],
            dev,
        )?;
        let rope = Tensor::from_host_slice(&rope_table(cap), [cap, 32, 2], dev)?;
        let positions = Tensor::from_host_slice(&(0..n as i32).collect::<Vec<_>>(), [n], dev)?;
        let start = positions.narrow(0, 0, 1)?;
        let mut state = Tensor::zeros([3, 3, 512], dev)?;
        let mut pool = Tensor::<bf16, _>::zeros([cap, 512], dev)?;
        let mut serial_state = Tensor::zeros([3, 3, 512], dev)?;
        let mut serial_pool = Tensor::<bf16, _>::zeros([cap, 512], dev)?;
        s.synchronize()?;
        // Bundle short calls to reduce idle gaps between graph submissions.
        // start=0 makes each prefill independently repeatable.
        s.graph_capture_begin()?;
        for _ in 0..32 {
            Cuda::v4_csa_compress(
                &s, &v, &g, &ape, &norm, &rope, &start, &mut state, &mut pool, 1e-6,
            )?;
        }
        s.graph_capture_end(30)?;
        s.graph_capture_begin()?;
        for t in 0..n {
            Cuda::v4_csa_compress(
                &s,
                &v.narrow(0, t, 1)?,
                &g.narrow(0, t, 1)?,
                &ape,
                &norm,
                &rope,
                &positions.narrow(0, t, 1)?,
                &mut serial_state,
                &mut serial_pool,
                1e-6,
            )?;
        }
        s.graph_capture_end(31)?;
        let prefill = measure(&s, 30, 32)?;
        let serial = measure(&s, 31, 1)?;
        let a = pool.to_host_vec()?;
        let b = serial_pool.to_host_vec()?;
        assert!(a[..n / 4 * 512].iter().all(|v| v.is_finite()));
        assert_eq!(a, b);
        assert_eq!(state.to_host_vec()?, serial_state.to_host_vec()?);
        println!(
            "{{\"kind\":\"compress_prefill\",\"tokens\":{n},\"prefill_us\":{prefill:.3},\"serial_decode_us\":{serial:.3},\"speedup\":{:.2},\"bitwise_equal\":true}}",
            serial / prefill
        );
        dev.config.invalidate_all_graphs();
    }
    // A repeating four-token block at positions 4..7. After warmup the previous
    // and current block have the same first-half projections, so each cycle is
    // a valid fixed snapshot including one overlapping output emission.
    let v = Tensor::from_host_slice(&values(4 * 1024), [4, 1024], dev)?;
    let g = Tensor::from_host_slice(&values(4 * 1024), [4, 1024], dev)?;
    let rope = Tensor::from_host_slice(&rope_table(2), [2, 32, 2], dev)?;
    let positions = Tensor::from_host_slice(&[0, 4, 5, 6, 7], [5], dev)?;
    let mut state = Tensor::zeros([3, 3, 512], dev)?;
    let mut pool = Tensor::<bf16, _>::zeros([2, 512], dev)?;
    Cuda::v4_csa_compress(
        &s,
        &v,
        &g,
        &ape,
        &norm,
        &rope,
        &positions.narrow(0, 0, 1)?,
        &mut state,
        &mut pool,
        1e-6,
    )?;
    s.synchronize()?;
    s.graph_capture_begin()?;
    for _ in 0..32 {
        Cuda::v4_csa_compress(
            &s,
            &v.narrow(0, 0, 1)?,
            &g.narrow(0, 0, 1)?,
            &ape,
            &norm,
            &rope,
            &positions.narrow(0, 1, 1)?,
            &mut state,
            &mut pool,
            1e-6,
        )?;
    }
    s.graph_capture_end(32)?;
    s.graph_capture_begin()?;
    for _ in 0..32 {
        for t in 0..4 {
            Cuda::v4_csa_compress(
                &s,
                &v.narrow(0, t, 1)?,
                &g.narrow(0, t, 1)?,
                &ape,
                &norm,
                &rope,
                &positions.narrow(0, t + 1, 1)?,
                &mut state,
                &mut pool,
                1e-6,
            )?;
        }
    }
    s.graph_capture_end(33)?;
    let no_emit = measure(&s, 32, 32)?;
    let average = measure(&s, 33, 128)?;
    let output = pool.to_host_vec()?;
    assert!(output.iter().all(|v| v.is_finite()));
    // Check this repeated snapshot against a single full prefill over two
    // identical input blocks, outside the timed region.
    let mut repeated = values(4 * 1024);
    repeated.extend_from_within(..);
    let all_v = Tensor::from_host_slice(&repeated, [8, 1024], dev)?;
    let all_g = Tensor::from_host_slice(&repeated, [8, 1024], dev)?;
    Cuda::v4_csa_compress(
        &s,
        &all_v,
        &all_g,
        &ape,
        &norm,
        &rope,
        &positions.narrow(0, 0, 1)?,
        &mut state,
        &mut pool,
        1e-6,
    )?;
    s.synchronize()?;
    assert_eq!(output, pool.to_host_vec()?);
    println!(
        "{{\"kind\":\"compress_decode\",\"no_emit_us\":{no_emit:.3},\"average_with_one_emit_per_four_tokens_us\":{average:.3},\"state_bytes\":18432}}"
    );
    dev.config.invalidate_all_graphs();
    Ok(())
}
