//! CUDA-event timings for top-k and the prepared-Q/K scoring + top-k pipeline.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::{ExecScope, StepCtx};
use infer_core::plan::{BatchKind, BatchPlan};
use infer_core::ports::FusedOps;
use infer_core::ports::fused_ops::v4_indexer_score_capacity;
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
// Existing sampler/Beam path includes full sort, exp/CDF and logprobs. Report
// its complete reuse cost, not a fictional isolated radix-sort measurement.
fn beam_baseline(
    s: &CudaScope,
    scores: &Tensor<f32, Cuda>,
    expected: &[i32],
    repeat: usize,
) -> Result<(f32, usize), Box<dyn std::error::Error>> {
    let width = scores.numel();
    let row = scores
        .clone()
        .view_contiguous(Shape::from_slice(&[width]))?;
    let words = Cuda::sampling_workspace_words(width)?;
    let work = Tensor::<f32, _>::zeros([words], s.device())?;
    let mut ids = Tensor::<i32, _>::zeros([expected.len()], s.device())?;
    let mut probs = Tensor::<f32, _>::zeros([expected.len()], s.device())?;
    let plan = BatchPlan {
        kind: BatchKind::DecodeOnly,
        num_tokens: 1,
        batch: 1,
        q_lens: vec![1],
        kv_lens: vec![1],
        seq_positions: vec![0],
        rope_positions: vec![0],
        max_blocks_per_seq: 0,
        block_size: 128,
        total_q_tiles: 1,
    };
    let ctx = StepCtx::new(s, &plan);
    s.synchronize()?;
    s.graph_capture_begin()?;
    for _ in 0..repeat {
        assert!(Cuda::beam_candidates_into(
            &ctx, &row, &mut ids, &mut probs, &work
        )?);
    }
    s.graph_capture_end(53)?;
    let elapsed = measure(s, 53, repeat)?;
    assert_eq!(ids.to_host_vec()?, expected);
    // Graphs reference work/ids/probs; destroy before their allocations drop.
    s.device().config.invalidate_all_graphs();
    Ok((elapsed, words * 4))
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
        (1, 127, 8192),
        (1, 127, 262144),
        (1, 32767, 262144),
        (128, 0, 262144),
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
        let bucket = v4_indexer_score_capacity(start as usize, n, c)?;
        let mut previous_ids = None;
        for b in if bucket == c {
            vec![c]
        } else {
            vec![c, bucket]
        } {
            let mut scores = Tensor::<f32, _>::zeros([n, b], dev)?;
            let words = Cuda::v4_indexer_topk_workspace_words(n, b, k)?;
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
                let row = &host_scores[t * b..(t + 1) * b];
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
            if let Some(previous) = &previous_ids {
                assert_eq!(&host_ids, previous);
            }
            previous_ids = Some(host_ids.clone());
            // At least K finite visible candidates: no mismatch between Beam's
            // vocabulary semantics and the Indexer's -1 padding/exclusion contract.
            let (beam_us, beam_bytes) = if n == 1 && (start as usize + 1) / 4 >= k {
                let (us, bytes) = beam_baseline(&s, &scores, &host_ids, repeat)?;
                (format!("{us:.3}"), bytes)
            } else {
                ("null".to_owned(), 0)
            };
            println!(
                "{{\"tokens\":{n},\"start\":{start},\"capacity\":{c},\"score_capacity\":{b},\"k\":{k},\"score_us\":{scoring:.3},\"topk_us\":{selection:.3},\"pipeline_us\":{pipeline:.3},\"workspace_bytes\":{},\"score_bytes\":{},\"legacy_beam_us\":{beam_us},\"legacy_beam_workspace_bytes\":{beam_bytes},\"exact_cpu_sort_match\":true}}",
                words * 4,
                n * b * 4
            );
            dev.config.invalidate_all_graphs();
        }
    }
    Ok(())
}
