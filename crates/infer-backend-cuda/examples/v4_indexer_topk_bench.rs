//! CUDA-event timings for top-k and the prepared-Q/K scoring + top-k pipeline.
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
    let (h, d, k) = (64, 128, 512);
    for (n, start, c) in [
        (128, 0, 32),
        (1024, 0, 256),
        (128, 32768, 8224),
        (1, 127, 32),
        (1, 1023, 256),
        (1, 32767, 8192),
        (1, 131071, 32768),
        (1, 1048575, 262144),
    ] {
        let q = Tensor::from_host_slice(
            &values(n * h * d)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [n, h, d],
            dev,
        )?;
        let keys = Tensor::from_host_slice(
            &values(c * d)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [c, d],
            dev,
        )?;
        let weights = Tensor::from_host_slice(
            &values(n * h)
                .into_iter()
                .map(|v| v / ((h * d) as f32).sqrt())
                .collect::<Vec<_>>(),
            [n, h],
            dev,
        )?;
        let pos = Tensor::from_host_slice(&[start], [1], dev)?;
        let mut scores = Tensor::<f32, _>::zeros([n, c], dev)?;
        let words = Cuda::v4_indexer_topk_workspace_words(n, c, k)?;
        let mut workspace = Tensor::<i32, _>::zeros([words], dev)?;
        let mut ids = Tensor::<i32, _>::zeros([n, k], dev)?;
        Cuda::v4_indexer_scores(&s, &q, &keys, &weights, &pos, &mut scores)?;
        s.synchronize()?;
        let repeat = if n == 1 { 32 } else { 8 };
        s.graph_capture_begin()?;
        for _ in 0..repeat {
            Cuda::v4_indexer_scores(&s, &q, &keys, &weights, &pos, &mut scores)?;
        }
        s.graph_capture_end(50)?;
        s.graph_capture_begin()?;
        for _ in 0..repeat {
            Cuda::v4_indexer_topk(&s, &scores, &pos, &mut workspace, &mut ids)?;
        }
        s.graph_capture_end(51)?;
        s.graph_capture_begin()?;
        for _ in 0..repeat {
            Cuda::v4_indexer_scores(&s, &q, &keys, &weights, &pos, &mut scores)?;
            Cuda::v4_indexer_topk(&s, &scores, &pos, &mut workspace, &mut ids)?;
        }
        s.graph_capture_end(52)?;
        let scoring = measure(&s, 50, repeat)?;
        let selection = measure(&s, 51, repeat)?;
        let pipeline = measure(&s, 52, repeat)?;
        let host_scores = scores.to_host_vec()?;
        let host_ids = ids.to_host_vec()?;
        for t in 0..n {
            let row = &host_scores[t * c..(t + 1) * c];
            let mut expected: Vec<_> = (0..(start as usize + t + 1) / 4)
                .filter(|&j| row[j] > f32::NEG_INFINITY)
                .collect();
            expected.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            let expected: Vec<_> = expected
                .into_iter()
                .map(|v| v as i32)
                .chain(std::iter::repeat(-1))
                .take(k)
                .collect();
            assert_eq!(&host_ids[t * k..(t + 1) * k], &expected);
        }
        println!(
            "{{\"tokens\":{n},\"start\":{start},\"capacity\":{c},\"k\":{k},\"score_us\":{scoring:.3},\"topk_us\":{selection:.3},\"pipeline_us\":{pipeline:.3},\"workspace_bytes\":{},\"exact_cpu_sort_match\":true}}",
            words * 4
        );
        dev.config.invalidate_all_graphs();
    }
    Ok(())
}
