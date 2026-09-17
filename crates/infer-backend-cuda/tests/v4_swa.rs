//! Real-shape SWA tests. Run with --ignored --test-threads=1 on a CUDA GPU.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

const D: usize = 512;
const W: usize = 128;

fn scope() -> CudaScope {
    Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
    .scope()
}

fn data(n: usize, seed: u32) -> Vec<bf16> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bf16::from_f32((state % 65536) as f32 / 16384.0 - 2.0)
        })
        .collect()
}

struct Case {
    q: Tensor<bf16, Cuda>,
    kv: Tensor<bf16, Cuda>,
    sink: Tensor<f32, Cuda>,
    pos: Tensor<i32, Cuda>,
    cache: Tensor<bf16, Cuda>,
    out: Tensor<bf16, Cuda>,
}

impl Case {
    fn new(s: &CudaScope, h: usize) -> Self {
        let dev = s.device();
        Self {
            q: Tensor::from_host_slice(&data(h * D, 93), [h, D], dev).unwrap(),
            kv: Tensor::from_host_slice(&data(D, 17), [D], dev).unwrap(),
            sink: Tensor::from_host_slice(&vec![0.0; h], [h], dev).unwrap(),
            pos: Tensor::from_host_slice(&[0], [1], dev).unwrap(),
            // Poison unused slots: accidental reads must fail numerically.
            cache: Tensor::from_host_slice(&vec![bf16::NAN; W * D], [W, D], dev).unwrap(),
            out: Tensor::zeros([h, D], dev).unwrap(),
        }
    }
    fn run(&mut self, s: &CudaScope) {
        Cuda::v4_swa_decode(
            s,
            &self.q,
            &self.kv,
            &self.sink,
            &self.pos,
            &mut self.cache,
            &mut self.out,
        )
        .unwrap();
    }
}

// Independent dense FP64 oracle in chronological order. No ring addressing,
// warp decomposition or shared implementation with the CUDA kernel.
fn reference(q: &[bf16], history: &[Vec<bf16>], sinks: &[f32]) -> Vec<f64> {
    let visible = &history[history.len().saturating_sub(W)..];
    q.chunks_exact(D)
        .zip(sinks)
        .flat_map(|(q, &sink)| {
            let scores: Vec<f64> = visible
                .iter()
                .map(|kv| {
                    q.iter()
                        .zip(kv)
                        .map(|(q, k)| q.to_f64() * k.to_f64())
                        .sum::<f64>()
                        / (D as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::from(sink), f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom = weights.iter().sum::<f64>() + (f64::from(sink) - max).exp();
            (0..D)
                .map(|d| {
                    visible
                        .iter()
                        .zip(&weights)
                        .map(|(kv, p)| kv[d].to_f64() * p)
                        .sum::<f64>()
                        / denom
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn assert_close(actual: &[bf16], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let error = (a.to_f64() - e).abs();
        // BF16 rounding is <= 0.390625% of the unrounded finite result.
        assert!(
            a.is_finite() && error <= 1e-5 + 0.004 * e.abs(),
            "element {i}: got {a}, reference {e}, error {error}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sequential_wraps_match_dense_reference_and_graph_reads_updated_inputs() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 64);
    let mut other = Case::new(&s, 3);
    let sinks: Vec<f32> = (0..64).map(|h| h as f32 / 8.0 - 4.0).collect();
    c.sink.upload_from_host(&sinks).unwrap();
    s.synchronize().unwrap();
    s.graph_capture_begin().unwrap();
    c.run(&s);
    s.graph_capture_end(901).unwrap();
    let before = s.device().config.pool_stats();
    let mut history = Vec::new();
    for pos in 0..260 {
        let q = data(64 * D, pos as u32 * 7 + 3);
        let kv = data(D, pos as u32 * 97 + 29);
        c.q.upload_from_host(&q).unwrap();
        c.kv.upload_from_host(&kv).unwrap();
        c.pos.upload_from_host(&[pos]).unwrap();
        history.push(kv);
        // Exercise both launch modes against one cache across two wraparounds.
        if pos % 2 == 0 {
            s.graph_launch(901).unwrap();
        } else {
            c.run(&s);
        }
        if pos % 17 == 0 {
            other.run(&s);
        }
        s.synchronize().unwrap();
        if [0, 1, 7, 126, 127, 128, 129, 254, 255, 256, 259].contains(&pos) {
            assert_close(
                &c.out.to_host_vec().unwrap(),
                &reference(&q, &history, &sinks),
            );
        }
    }
    assert_eq!(
        s.device().config.pool_stats(),
        before,
        "decode must allocate nothing"
    );
    let cache = c.cache.to_host_vec().unwrap();
    for (pos, expected) in history.iter().enumerate().skip(history.len() - W) {
        assert_eq!(&cache[(pos % W) * D..(pos % W + 1) * D], expected);
    }
    // A separate request repeatedly restarted at zero did not touch c's history.
    let other_cache = other.cache.to_host_vec().unwrap();
    assert_eq!(&other_cache[..D], &data(D, 17));
    assert!(other_cache[D..].iter().all(|v| v.is_nan()));
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sink_normalization_extremes_and_large_absolute_positions() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 4);
    c.q.upload_from_host(&vec![bf16::ZERO; 4 * D]).unwrap();
    c.kv.upload_from_host(&vec![bf16::ONE; D]).unwrap();
    let sinks = [f32::NEG_INFINITY, 0.0, 10000.0, -10000.0];
    c.sink.upload_from_host(&sinks).unwrap();
    // With zero scores and constant KV=1, output is n/(n+exp(sink)).
    for pos in [0, 1, 127, 128, 1_048_576, i32::MAX] {
        c.cache.upload_from_host(&vec![bf16::ONE; W * D]).unwrap();
        c.pos.upload_from_host(&[pos]).unwrap();
        c.run(&s);
        s.synchronize().unwrap();
        let n = f64::from(pos.min(127) + 1);
        let expected: Vec<f64> = sinks
            .iter()
            .flat_map(|&sink| vec![n / (n + f64::from(sink).exp()); D])
            .collect();
        assert_close(&c.out.to_host_vec().unwrap(), &expected);
    }
    // Strong finite logits need a stable softmax, even with the sink disabled.
    c.q.upload_from_host(&vec![bf16::from_f32(100.0); 4 * D])
        .unwrap();
    c.sink.upload_from_host(&[f32::NEG_INFINITY; 4]).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    assert_eq!(c.out.to_host_vec().unwrap(), vec![bf16::ONE; 4 * D]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn invalid_metadata_and_negative_position_do_not_mutate_cache() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 2);
    let initial = data(W * D, 29);
    c.cache.upload_from_host(&initial).unwrap();
    let bad = Tensor::zeros([2, 256], s.device()).unwrap();
    assert!(
        Cuda::v4_swa_decode(&s, &bad, &c.kv, &c.sink, &c.pos, &mut c.cache, &mut c.out).is_err()
    );
    let strided = Tensor::<bf16, _>::zeros([2, D + 1], s.device())
        .unwrap()
        .narrow(1, 0, D)
        .unwrap();
    assert!(
        Cuda::v4_swa_decode(
            &s,
            &strided,
            &c.kv,
            &c.sink,
            &c.pos,
            &mut c.cache,
            &mut c.out
        )
        .is_err()
    );
    let unaligned = Tensor::<bf16, _>::zeros([D + 1], s.device())
        .unwrap()
        .narrow(0, 1, D)
        .unwrap();
    assert!(
        Cuda::v4_swa_decode(
            &s,
            &c.q,
            &unaligned,
            &c.sink,
            &c.pos,
            &mut c.cache,
            &mut c.out
        )
        .is_err()
    );
    let mut alias = c.q.clone();
    assert!(
        Cuda::v4_swa_decode(&s, &c.q, &c.kv, &c.sink, &c.pos, &mut c.cache, &mut alias).is_err()
    );
    let cache_alias = c
        .cache
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[D]))
        .unwrap();
    assert!(
        Cuda::v4_swa_decode(
            &s,
            &c.q,
            &cache_alias,
            &c.sink,
            &c.pos,
            &mut c.cache,
            &mut c.out
        )
        .is_err()
    );
    c.pos.upload_from_host(&[-1]).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    assert_eq!(c.cache.to_host_vec().unwrap(), initial);
    assert!(c.out.to_host_vec().unwrap().iter().all(|v| v.is_nan()));
}

struct Prefill {
    q: Tensor<bf16, Cuda>,
    kv: Tensor<bf16, Cuda>,
    sink: Tensor<f32, Cuda>,
    start: Tensor<i32, Cuda>,
    cache: Tensor<bf16, Cuda>,
    out: Tensor<bf16, Cuda>,
}

impl Prefill {
    fn new(s: &CudaScope, n: usize, h: usize) -> Self {
        let dev = s.device();
        Self {
            q: Tensor::from_host_slice(&data(n * h * D, 73), [n, h, D], dev).unwrap(),
            kv: Tensor::from_host_slice(&data(n * D, 109), [n, D], dev).unwrap(),
            sink: Tensor::from_host_slice(&vec![0.0f32; h], [h], dev).unwrap(),
            start: Tensor::from_host_slice(&[0i32], [1], dev).unwrap(),
            cache: Tensor::from_host_slice(&vec![bf16::NAN; W * D], [W, D], dev).unwrap(),
            out: Tensor::zeros([n, h, D], dev).unwrap(),
        }
    }
    fn run(&mut self, s: &CudaScope) {
        Cuda::v4_swa_prefill(
            s,
            &self.q,
            &self.kv,
            &self.sink,
            &self.start,
            &mut self.cache,
            &mut self.out,
        )
        .unwrap();
    }
}

fn assert_ring(cache: &Tensor<bf16, Cuda>, history: &[Vec<bf16>]) {
    let actual = cache.to_host_vec().unwrap();
    for (pos, expected) in history
        .iter()
        .enumerate()
        .skip(history.len().saturating_sub(W))
    {
        assert_eq!(&actual[pos % W * D..(pos % W + 1) * D], expected);
    }
    if history.len() < W {
        assert!(actual[history.len() * D..].iter().all(|v| v.is_nan()));
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn prefill_full_and_chunks_match_reference_then_decode_continues() {
    let s = scope();
    let _active = s.enter();
    // A head tail, a token tail, >2 windows, and chunks crossing both edges.
    let (n, h) = (273, 17);
    let mut full = Prefill::new(&s, n, h);
    let mut chunks = Prefill::new(&s, n, h);
    let q = data(n * h * D, 73);
    let history: Vec<Vec<bf16>> = data(n * D, 109)
        .chunks_exact(D)
        .map(<[bf16]>::to_vec)
        .collect();
    let sinks: Vec<f32> = (0..h).map(|i| i as f32 * 0.37 - 2.0).collect();
    full.sink.upload_from_host(&sinks).unwrap();
    chunks.sink.upload_from_host(&sinks).unwrap();
    let before = s.device().config.pool_stats();
    full.run(&s);
    let mut start = 0;
    for count in [3, 125, 1, 7, 128, 9] {
        chunks.start.upload_from_host(&[start as i32]).unwrap();
        let qi = chunks.q.narrow(0, start, count).unwrap();
        let ki = chunks.kv.narrow(0, start, count).unwrap();
        let mut oi = chunks.out.narrow(0, start, count).unwrap();
        Cuda::v4_swa_prefill(
            &s,
            &qi,
            &ki,
            &chunks.sink,
            &chunks.start,
            &mut chunks.cache,
            &mut oi,
        )
        .unwrap();
        start += count;
    }
    assert_eq!(start, n);
    s.synchronize().unwrap();
    assert_eq!(s.device().config.pool_stats(), before);
    let output = full.out.to_host_vec().unwrap();
    let chunk_output = chunks.out.to_host_vec().unwrap();
    for (i, (a, b)) in output.iter().zip(&chunk_output).enumerate() {
        assert_eq!(a, b, "full/chunk mismatch at element {i}");
    }
    for t in [0, 1, 126, 127, 128, 129, 135, 136, 255, 256, 272] {
        let range = t * h * D..(t + 1) * h * D;
        assert_close(
            &output[range.clone()],
            &reference(&q[range], &history[..=t], &sinks),
        );
    }
    assert_ring(&full.cache, &history);
    assert_ring(&chunks.cache, &history);
    let mut next = Case::new(&s, h);
    next.cache = chunks.cache;
    next.pos.upload_from_host(&[n as i32]).unwrap();
    next.sink.upload_from_host(&sinks).unwrap();
    next.run(&s);
    s.synchronize().unwrap();
    let mut extended = history;
    extended.push(data(D, 17));
    assert_close(
        &next.out.to_host_vec().unwrap(),
        &reference(&data(h * D, 93), &extended, &sinks),
    );
    assert_ring(&next.cache, &extended);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn prefill_graph_replay_updates_start_and_inputs_with_real_head_count() {
    let s = scope();
    let _active = s.enter();
    let (n, h) = (17, 64);
    let mut p = Prefill::new(&s, n, h);
    s.synchronize().unwrap();
    s.graph_capture_begin().unwrap();
    p.run(&s);
    s.graph_capture_end(902).unwrap();
    let before = s.device().config.pool_stats();
    let mut history = Vec::new();
    for chunk in 0..9 {
        let q = data(n * h * D, 133 + chunk as u32 * 31);
        let kv = data(n * D, 99 + chunk as u32 * 87);
        p.q.upload_from_host(&q).unwrap();
        p.kv.upload_from_host(&kv).unwrap();
        p.start.upload_from_host(&[(chunk * n) as i32]).unwrap();
        history.extend(kv.chunks_exact(D).map(<[bf16]>::to_vec));
        s.graph_launch(902).unwrap();
        s.synchronize().unwrap();
        let output = p.out.to_host_vec().unwrap();
        for t in [0, n - 1] {
            let r = t * h * D..(t + 1) * h * D;
            assert_close(
                &output[r.clone()],
                &reference(&q[r], &history[..chunk * n + t + 1], &vec![0.0; h]),
            );
        }
        assert_ring(&p.cache, &history);
    }
    assert_eq!(s.device().config.pool_stats(), before);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn prefill_is_causal_and_stable_with_sink_and_validates_metadata() {
    let s = scope();
    let _active = s.enter();
    let (n, h) = (129, 3);
    let mut p = Prefill::new(&s, n, h);
    p.q.upload_from_host(&vec![bf16::ZERO; n * h * D]).unwrap();
    let kv: Vec<bf16> = (0..n)
        .flat_map(|t| vec![bf16::from_f32(t as f32); D])
        .collect();
    p.kv.upload_from_host(&kv).unwrap();
    p.sink
        .upload_from_host(&[f32::NEG_INFINITY, 0.0, 10000.0])
        .unwrap();
    p.run(&s);
    s.synchronize().unwrap();
    let output = p.out.to_host_vec().unwrap();
    for t in 0..n {
        let lo = (t + 1).saturating_sub(W);
        let count = (t + 1 - lo) as f64;
        let mean = (lo + t) as f64 / 2.0;
        let expected: Vec<f64> = [mean, mean * count / (count + 1.0), 0.0]
            .into_iter()
            .flat_map(|v| vec![v; D])
            .collect();
        assert_close(&output[t * h * D..(t + 1) * h * D], &expected);
    }
    // Large finite scores, including sink disabled, must not overflow exp().
    p.q.upload_from_host(&vec![bf16::from_f32(100.0); n * h * D])
        .unwrap();
    p.kv.upload_from_host(&vec![bf16::ONE; n * D]).unwrap();
    p.sink.upload_from_host(&[f32::NEG_INFINITY; 3]).unwrap();
    p.run(&s);
    s.synchronize().unwrap();
    assert_close(&p.out.to_host_vec().unwrap(), &vec![1.0; n * h * D]);

    let before = p.cache.to_host_vec().unwrap();
    for start in [-1, i32::MAX - n as i32 + 2] {
        p.start.upload_from_host(&[start]).unwrap();
        p.run(&s);
        s.synchronize().unwrap();
        assert_eq!(p.cache.to_host_vec().unwrap(), before);
        assert!(p.out.to_host_vec().unwrap().iter().all(|v| v.is_nan()));
    }
    p.start
        .upload_from_host(&[i32::MAX - n as i32 + 1])
        .unwrap();
    p.run(&s);
    s.synchronize().unwrap();
    assert_close(&p.out.to_host_vec().unwrap(), &vec![1.0; n * h * D]);

    let mut alias = p.q.clone();
    assert!(
        Cuda::v4_swa_prefill(&s, &p.q, &p.kv, &p.sink, &p.start, &mut p.cache, &mut alias).is_err()
    );
    let strided = Tensor::<bf16, _>::zeros([n, h, D + 1], s.device())
        .unwrap()
        .narrow(2, 0, D)
        .unwrap();
    assert!(
        Cuda::v4_swa_prefill(
            &s,
            &strided,
            &p.kv,
            &p.sink,
            &p.start,
            &mut p.cache,
            &mut p.out
        )
        .is_err()
    );
    let empty = p.q.narrow(0, 0, 0).unwrap();
    assert!(
        Cuda::v4_swa_prefill(
            &s,
            &empty,
            &p.kv,
            &p.sink,
            &p.start,
            &mut p.cache,
            &mut p.out
        )
        .is_err()
    );
    assert_eq!(p.cache.to_host_vec().unwrap(), before);
}
