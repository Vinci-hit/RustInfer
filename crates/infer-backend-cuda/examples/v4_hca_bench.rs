//! CUDA-event graph microbenchmarks, excluding projection GEMMs and transfers.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

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
                let angle = (b * 128) as f32 / 10000f32.powf(d as f32 / 32.0);
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
    for n in [128, 512, 1024] {
        let c = n / 128 + 1;
        let v = Tensor::from_host_slice(&values(n * 512), [n, 512], dev)?;
        let g = Tensor::from_host_slice(&values(n * 512), [n, 512], dev)?;
        let ape = Tensor::from_host_slice(&values(128 * 512), [128, 512], dev)?;
        let norm = Tensor::from_host_slice(&[1.0f32; 512], [512], dev)?;
        let rope = Tensor::from_host_slice(&rope_table(c), [c, 32, 2], dev)?;
        let positions = Tensor::from_host_slice(&(0..n as i32).collect::<Vec<_>>(), [n], dev)?;
        let start = positions.narrow(0, 0, 1)?;
        let q = Tensor::from_host_slice(
            &values(n * 64 * 512)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [n, 64, 512],
            dev,
        )?;
        let kv = Tensor::from_host_slice(
            &values(n * 512)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [n, 512],
            dev,
        )?;
        let sink = Tensor::from_host_slice(&[0.0f32; 64], [64], dev)?;
        let mut state = Tensor::zeros([3, 512], dev)?;
        let mut pool = Tensor::zeros([c, 512], dev)?;
        let mut ring = Tensor::zeros([128, 512], dev)?;
        let mut out = Tensor::zeros([n, 64, 512], dev)?;
        let mut serial_state = Tensor::zeros([3, 512], dev)?;
        let mut serial_pool = Tensor::zeros([c, 512], dev)?;
        let mut serial_ring = Tensor::zeros([128, 512], dev)?;
        let serial_out = Tensor::zeros([n, 64, 512], dev)?;
        s.synchronize()?;
        s.graph_capture_begin()?;
        Cuda::v4_hca_compress(
            &s, &v, &g, &ape, &norm, &rope, &start, &mut state, &mut pool, 1e-6,
        )?;
        s.graph_capture_end(20)?;
        s.graph_capture_begin()?;
        Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut out)?;
        s.graph_capture_end(21)?;
        s.graph_capture_begin()?;
        Cuda::v4_hca_compress(
            &s, &v, &g, &ape, &norm, &rope, &start, &mut state, &mut pool, 1e-6,
        )?;
        Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut out)?;
        s.graph_capture_end(22)?;
        s.graph_capture_begin()?;
        for t in 0..n {
            let vt = v.narrow(0, t, 1)?;
            let gt = g.narrow(0, t, 1)?;
            let pt = positions.narrow(0, t, 1)?;
            Cuda::v4_hca_compress(
                &s,
                &vt,
                &gt,
                &ape,
                &norm,
                &rope,
                &pt,
                &mut serial_state,
                &mut serial_pool,
                1e-6,
            )?;
            Cuda::v4_hca_decode(
                &s,
                &q.narrow(0, t, 1)?
                    .view_contiguous(Shape::from_slice(&[64, 512]))?,
                &kv.narrow(0, t, 1)?
                    .view_contiguous(Shape::from_slice(&[512]))?,
                &sink,
                &pt,
                &serial_pool,
                &mut serial_ring,
                &mut serial_out
                    .narrow(0, t, 1)?
                    .view_contiguous(Shape::from_slice(&[64, 512]))?,
            )?;
        }
        s.graph_capture_end(23)?;
        let compress = measure(&s, 20, 1)?;
        let attention = measure(&s, 21, 1)?;
        let pipeline = measure(&s, 22, 1)?;
        let serial = measure(&s, 23, 1)?;
        let mut error = 0.0f32;
        for (a, b) in out.to_host_vec()?.iter().zip(serial_out.to_host_vec()?) {
            assert!(a.is_finite() && b.is_finite());
            error = error.max((a.to_f32() - b.to_f32()).abs());
        }
        assert!(error <= 0.008, "prefill/decode max difference {error}");
        println!(
            "{{\"kind\":\"prefill\",\"tokens\":{n},\"heads\":64,\"compress_us\":{compress:.3},\"attention_us\":{attention:.3},\"pipeline_us\":{pipeline:.3},\"serial_pipeline_us\":{serial:.3},\"speedup\":{:.2},\"max_abs_difference\":{error}}}",
            serial / pipeline
        );
        dev.config.invalidate_all_graphs();
    }
    // Fixed snapshots at a block START: repeated compression ignores old state;
    // constant local KV makes repeated attention append idempotent. No uploads
    // inside timing. This does not measure the every-128th-token emit cost.
    for history in [4096usize, 32768, 131072] {
        let c = history / 128 + 1;
        let v = Tensor::from_host_slice(&values(512), [1, 512], dev)?;
        let g = Tensor::zeros([1, 512], dev)?;
        let ape = Tensor::zeros([128, 512], dev)?;
        let norm = Tensor::from_host_slice(&[1.0f32; 512], [512], dev)?;
        let rope = Tensor::from_host_slice(&rope_table(c), [c, 32, 2], dev)?;
        let start = Tensor::from_host_slice(&[history as i32], [1], dev)?;
        let mut state = Tensor::zeros([3, 512], dev)?;
        let mut pool = Tensor::from_host_slice(&vec![bf16::ONE; c * 512], [c, 512], dev)?;
        let q = Tensor::from_host_slice(
            &values(64 * 512)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [64, 512],
            dev,
        )?;
        let kv = Tensor::from_host_slice(&[bf16::ONE; 512], [512], dev)?;
        let sink = Tensor::from_host_slice(&[0.0f32; 64], [64], dev)?;
        let mut ring = Tensor::from_host_slice(&vec![bf16::ONE; 128 * 512], [128, 512], dev)?;
        let mut out = Tensor::zeros([64, 512], dev)?;
        s.synchronize()?;
        s.graph_capture_begin()?;
        for _ in 0..32 {
            Cuda::v4_hca_compress(
                &s, &v, &g, &ape, &norm, &rope, &start, &mut state, &mut pool, 1e-6,
            )?;
        }
        s.graph_capture_end(24)?;
        s.graph_capture_begin()?;
        for _ in 0..32 {
            Cuda::v4_hca_decode(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut out)?;
        }
        s.graph_capture_end(25)?;
        let compress = measure(&s, 24, 32)?;
        let attention = measure(&s, 25, 32)?;
        assert!(out.to_host_vec()?.iter().all(|v| v.is_finite()));
        println!(
            "{{\"kind\":\"decode\",\"history_tokens\":{history},\"compressed_rows\":{},\"heads\":64,\"compress_no_emit_us\":{compress:.3},\"attention_us\":{attention:.3}}}",
            history / 128
        );
        dev.config.invalidate_all_graphs();
    }
    Ok(())
}
