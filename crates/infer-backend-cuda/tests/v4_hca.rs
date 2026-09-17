//! Independent numeric oracles; run with --ignored --test-threads=1 on CUDA.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

const D: usize = 512;
const R: usize = 128;
const EPS: f32 = 1e-6;

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
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 65536) as f32 / 16384.0 - 2.0
        })
        .collect()
}
fn bf(n: usize, seed: u32) -> Vec<bf16> {
    data(n, seed).into_iter().map(bf16::from_f32).collect()
}
fn close(actual: &[bf16], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a.to_f64() - b).abs() <= 2e-5 + 0.004 * b.abs(),
            "element {i}: actual {a}, expected {b}"
        );
    }
}

struct Compress {
    values: Tensor<f32, Cuda>,
    gates: Tensor<f32, Cuda>,
    ape: Tensor<f32, Cuda>,
    norm: Tensor<f32, Cuda>,
    rope: Tensor<f32, Cuda>,
    start: Tensor<i32, Cuda>,
    state: Tensor<f32, Cuda>,
    pool: Tensor<bf16, Cuda>,
}
impl Compress {
    fn new(s: &CudaScope, n: usize, capacity: usize) -> Self {
        let dev = s.device();
        let gates: Vec<_> = data(n * D, 37).into_iter().map(|v| v * 8.0).collect();
        let norm: Vec<_> = data(D, 29).into_iter().map(|v| 1.0 + v * 0.1).collect();
        let rope: Vec<_> = (0..capacity)
            .flat_map(|b| {
                (0..32).flat_map(move |d| {
                    let angle = (b * R) as f32 / 10000f32.powf(d as f32 / 32.0);
                    [angle.cos(), angle.sin()]
                })
            })
            .collect();
        Self {
            values: Tensor::from_host_slice(&data(n * D, 71), [n, D], dev).unwrap(),
            gates: Tensor::from_host_slice(&gates, [n, D], dev).unwrap(),
            ape: Tensor::from_host_slice(&data(R * D, 47), [R, D], dev).unwrap(),
            norm: Tensor::from_host_slice(&norm, [D], dev).unwrap(),
            rope: Tensor::from_host_slice(&rope, [capacity, 32, 2], dev).unwrap(),
            start: Tensor::from_host_slice(&[0], [1], dev).unwrap(),
            state: Tensor::from_host_slice(&vec![f32::NAN; 3 * D], [3, D], dev).unwrap(),
            pool: Tensor::from_host_slice(&vec![bf16::NAN; capacity * D], [capacity, D], dev)
                .unwrap(),
        }
    }
    fn run(&mut self, s: &CudaScope, offset: usize, n: usize) {
        let v = self.values.narrow(0, offset, n).unwrap();
        let g = self.gates.narrow(0, offset, n).unwrap();
        Cuda::v4_hca_compress(
            s,
            &v,
            &g,
            &self.ape,
            &self.norm,
            &self.rope,
            &self.start,
            &mut self.state,
            &mut self.pool,
            EPS,
        )
        .unwrap();
    }
}

// Dense two-pass FP64 softmax, independent of the online CUDA accumulator.
fn compression_reference(c: &Compress, n: usize) -> Vec<bf16> {
    let v = c.values.to_host_vec().unwrap();
    let g = c.gates.to_host_vec().unwrap();
    let ape = c.ape.to_host_vec().unwrap();
    let norm = c.norm.to_host_vec().unwrap();
    let rope = c.rope.to_host_vec().unwrap();
    let mut out = Vec::new();
    for block in 0..n / R {
        let mut pooled = Vec::new();
        for d in 0..D {
            let scores: Vec<_> = (0..R)
                .map(|j| f64::from(g[(block * R + j) * D + d] + ape[j * D + d]))
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|x| (x - max).exp()).collect();
            let z = weights.iter().sum::<f64>();
            let p = (0..R)
                .map(|j| f64::from(v[(block * R + j) * D + d]) * weights[j])
                .sum::<f64>()
                / z;
            pooled.push(bf16::from_f64(p).to_f64());
        }
        let inv = (pooled.iter().map(|v| v * v).sum::<f64>() / D as f64 + f64::from(EPS))
            .sqrt()
            .recip();
        let normalized: Vec<_> = pooled
            .iter()
            .zip(&norm)
            .map(|(p, w)| bf16::from_f64(p * inv * f64::from(*w)))
            .collect();
        let mut row = normalized.clone();
        for pair in 0..32 {
            let d = D - 64 + 2 * pair;
            let a = normalized[d].to_f32();
            let b = normalized[d + 1].to_f32();
            let cos = rope[block * 64 + 2 * pair];
            let sin = rope[block * 64 + 2 * pair + 1];
            row[d] = bf16::from_f32(a * cos - b * sin);
            row[d + 1] = bf16::from_f32(a * sin + b * cos);
        }
        out.extend(row);
    }
    out
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn compression_full_chunks_and_incremental_graph_match_dense_pooling() {
    let s = scope();
    let _active = s.enter();
    let n = 513;
    let mut full = Compress::new(&s, n, 6);
    let mut chunk = Compress::new(&s, n, 6);
    let mut serial = Compress::new(&s, 1, 6);
    full.run(&s, 0, n);
    let mut offset = 0;
    for count in [7, 120, 1, 2, 127, 128, 4, 124] {
        chunk.start.upload_from_host(&[offset as i32]).unwrap();
        chunk.run(&s, offset, count);
        offset += count;
    }
    assert_eq!(offset, n);
    s.synchronize().unwrap();
    let expected = compression_reference(&full, n);
    let a = full.pool.to_host_vec().unwrap();
    // Boundaries include two intermediate BF16 casts and a rotation; usually
    // exact, allow one BF16 step around a rounding boundary in the oracle.
    for (i, (a, b)) in a.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && (a.to_f32() - b.to_f32()).abs() <= 0.001 + 0.008 * b.to_f32().abs(),
            "compressed {i}: {a} vs {b}"
        );
    }
    assert!(a[4 * D..].iter().all(|v| v.is_nan()));
    assert_eq!(&a[..4 * D], &chunk.pool.to_host_vec().unwrap()[..4 * D]);
    let values = full.values.to_host_vec().unwrap();
    let gates = full.gates.to_host_vec().unwrap();
    s.graph_capture_begin().unwrap();
    serial.run(&s, 0, 1);
    s.graph_capture_end(1901).unwrap();
    let before = s.device().config.pool_stats();
    for t in 0..n {
        serial
            .values
            .upload_from_host(&values[t * D..(t + 1) * D])
            .unwrap();
        serial
            .gates
            .upload_from_host(&gates[t * D..(t + 1) * D])
            .unwrap();
        serial.start.upload_from_host(&[t as i32]).unwrap();
        s.graph_launch(1901).unwrap();
        s.synchronize().unwrap();
        if t == 126 {
            assert!(
                serial
                    .pool
                    .to_host_vec()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
        if t == 127 {
            assert!(
                serial.pool.to_host_vec().unwrap()[..D]
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
    }
    assert_eq!(s.device().config.pool_stats(), before);
    assert_eq!(&a[..4 * D], &serial.pool.to_host_vec().unwrap()[..4 * D]);
    assert_eq!(
        full.state.to_host_vec().unwrap(),
        chunk.state.to_host_vec().unwrap()
    );
    assert_eq!(
        full.state.to_host_vec().unwrap(),
        serial.state.to_host_vec().unwrap()
    );
    s.device().config.invalidate_all_graphs();
}

fn attention_reference(
    q: &[bf16],
    raw: &[bf16],
    pool: &[bf16],
    pos: usize,
    sinks: &[f32],
) -> Vec<f64> {
    let rows: Vec<_> = raw[(pos + 1).saturating_sub(R) * D..(pos + 1) * D]
        .chunks_exact(D)
        .chain(pool[..((pos + 1) / R) * D].chunks_exact(D))
        .collect();
    q.chunks_exact(D)
        .zip(sinks)
        .flat_map(|(q, sink)| {
            let scores: Vec<f64> = rows
                .iter()
                .map(|kv| {
                    q.iter()
                        .zip(*kv)
                        .map(|(q, k)| q.to_f64() * k.to_f64())
                        .sum::<f64>()
                        / (D as f64).sqrt()
                })
                .collect();
            let m = scores.iter().copied().fold(f64::from(*sink), f64::max);
            let p: Vec<_> = scores.iter().map(|x| (x - m).exp()).collect();
            let z = p.iter().sum::<f64>() + (f64::from(*sink) - m).exp();
            (0..D)
                .map(|d| {
                    rows.iter()
                        .zip(&p)
                        .map(|(r, p)| r[d].to_f64() * p)
                        .sum::<f64>()
                        / z
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn full_and_chunked_pipeline_are_causal_and_decode_continues() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let n = 273;
    let h = 17;
    let mut full = Compress::new(&s, n + 1, 4);
    let mut chunk = Compress::new(&s, n + 1, 4);
    let queries = bf((n + 1) * h * D, 93);
    let raw = bf((n + 1) * D, 17);
    let sinks: Vec<_> = (0..h).map(|i| i as f32 / 4.0 - 2.0).collect();
    let q = Tensor::from_host_slice(&queries, [n + 1, h, D], dev).unwrap();
    let kv = Tensor::from_host_slice(&raw, [n + 1, D], dev).unwrap();
    let sink = Tensor::from_host_slice(&sinks, [h], dev).unwrap();
    let mut ring = Tensor::from_host_slice(&vec![bf16::NAN; R * D], [R, D], dev).unwrap();
    let mut ring2 = Tensor::from_host_slice(&vec![bf16::NAN; R * D], [R, D], dev).unwrap();
    let out = Tensor::zeros([n + 1, h, D], dev).unwrap();
    let out2 = Tensor::zeros([n + 1, h, D], dev).unwrap();
    let before = dev.config.pool_stats();
    full.run(&s, 0, n);
    Cuda::v4_hca_prefill(
        &s,
        &q.narrow(0, 0, n).unwrap(),
        &kv.narrow(0, 0, n).unwrap(),
        &sink,
        &full.start,
        &full.pool,
        &mut ring,
        &mut out.narrow(0, 0, n).unwrap(),
    )
    .unwrap();
    let mut offset = 0;
    for count in [3, 125, 1, 7, 128, 9] {
        chunk.start.upload_from_host(&[offset as i32]).unwrap();
        chunk.run(&s, offset, count);
        Cuda::v4_hca_prefill(
            &s,
            &q.narrow(0, offset, count).unwrap(),
            &kv.narrow(0, offset, count).unwrap(),
            &sink,
            &chunk.start,
            &chunk.pool,
            &mut ring2,
            &mut out2.narrow(0, offset, count).unwrap(),
        )
        .unwrap();
        offset += count;
    }
    for (c, cache, output) in [
        (&mut full, &mut ring, &out),
        (&mut chunk, &mut ring2, &out2),
    ] {
        c.start.upload_from_host(&[n as i32]).unwrap();
        c.run(&s, n, 1);
        Cuda::v4_hca_decode(
            &s,
            &q.narrow(0, n, 1)
                .unwrap()
                .view_contiguous(Shape::from_slice(&[h, D]))
                .unwrap(),
            &kv.narrow(0, n, 1)
                .unwrap()
                .view_contiguous(Shape::from_slice(&[D]))
                .unwrap(),
            &sink,
            &c.start,
            &c.pool,
            cache,
            &mut output
                .narrow(0, n, 1)
                .unwrap()
                .view_contiguous(Shape::from_slice(&[h, D]))
                .unwrap(),
        )
        .unwrap();
    }
    s.synchronize().unwrap();
    assert_eq!(dev.config.pool_stats(), before);
    let a = out.to_host_vec().unwrap();
    assert_eq!(a, out2.to_host_vec().unwrap());
    let pool = full.pool.to_host_vec().unwrap();
    for t in [0, 1, 126, 127, 128, 254, 255, 256, 272, 273] {
        close(
            &a[t * h * D..(t + 1) * h * D],
            &attention_reference(&queries[t * h * D..(t + 1) * h * D], &raw, &pool, t, &sinks),
        );
    }
    let cache = ring.to_host_vec().unwrap();
    assert_eq!(cache, ring2.to_host_vec().unwrap());
    for t in n + 1 - R..=n {
        assert_eq!(&cache[t % R * D..(t % R + 1) * D], &raw[t * D..(t + 1) * D]);
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn long_history_graphs_merge_multiple_tiles_with_64_heads() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let h = 64;
    let n = 17;
    let capacity = 260;
    let mut q = Tensor::zeros([n, h, D], dev).unwrap();
    let mut kv = Tensor::zeros([n, D], dev).unwrap();
    let sinks: Vec<_> = (0..h)
        .map(|i| {
            if i == 0 {
                f32::NEG_INFINITY
            } else {
                i as f32 / 8.0 - 4.0
            }
        })
        .collect();
    let sink = Tensor::from_host_slice(&sinks, [h], dev).unwrap();
    let mut start = Tensor::from_host_slice(&[32750], [1], dev).unwrap();
    let mut pool_data = bf(capacity * D, 47);
    // Late compressed tiles have much larger logits for the positive query.
    // This forces online rescaling of both previous numerator and denominator.
    pool_data[256 * D..257 * D].fill(bf16::from_f32(8.0));
    pool_data[258 * D..].fill(bf16::NAN);
    let pool = Tensor::from_host_slice(&pool_data, [capacity, D], dev).unwrap();
    let mut ring = Tensor::zeros([R, D], dev).unwrap();
    let mut output = Tensor::zeros([n, h, D], dev).unwrap();
    let qd = q
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[h, D]))
        .unwrap();
    let kd = kv
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[D]))
        .unwrap();
    let mut od = output
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[h, D]))
        .unwrap();
    s.synchronize().unwrap();
    s.graph_capture_begin().unwrap();
    Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut output).unwrap();
    s.graph_capture_end(1902).unwrap();
    s.graph_capture_begin().unwrap();
    Cuda::v4_hca_decode(&s, &qd, &kd, &sink, &start, &pool, &mut ring, &mut od).unwrap();
    s.graph_capture_end(1903).unwrap();
    let before = dev.config.pool_stats();
    for (run, pos) in [32750usize, 32767, 32880].into_iter().enumerate() {
        let mut queries = bf(n * h * D, 93 + run as u32);
        if run == 2 {
            queries.fill(bf16::from_f32(2.0));
        }
        let raw = bf((pos + n) * D, 109 + run as u32);
        let mut cache = vec![bf16::NAN; R * D];
        for t in pos - R..pos {
            cache[t % R * D..(t % R + 1) * D].copy_from_slice(&raw[t * D..(t + 1) * D]);
        }
        q.upload_from_host(&queries).unwrap();
        kv.upload_from_host(&raw[pos * D..]).unwrap();
        start.upload_from_host(&[pos as i32]).unwrap();
        ring.upload_from_host(&cache).unwrap();
        s.graph_launch(1902).unwrap();
        s.synchronize().unwrap();
        let a = output.to_host_vec().unwrap();
        for t in [0, n - 1] {
            close(
                &a[t * h * D..(t + 1) * h * D],
                &attention_reference(
                    &queries[t * h * D..(t + 1) * h * D],
                    &raw,
                    &pool_data,
                    pos + t,
                    &sinks,
                ),
            );
        }
        ring.upload_from_host(&cache).unwrap();
        s.graph_launch(1903).unwrap();
        s.synchronize().unwrap();
        close(
            &od.to_host_vec().unwrap(),
            &attention_reference(&queries[..h * D], &raw, &pool_data, pos, &sinks),
        );
    }
    assert_eq!(dev.config.pool_stats(), before);
    dev.config.invalidate_all_graphs();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn boundary_sink_and_invalid_inputs_preserve_state_and_cache() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let n = 257;
    let h = 3;
    let mut q = Tensor::zeros([n, h, D], dev).unwrap();
    let kv = Tensor::from_host_slice(&vec![bf16::ONE; n * D], [n, D], dev).unwrap();
    let sink = Tensor::from_host_slice(&[f32::NEG_INFINITY, 0.0, 10000.0], [h], dev).unwrap();
    let mut start = Tensor::from_host_slice(&[0], [1], dev).unwrap();
    let pool = Tensor::from_host_slice(&vec![bf16::from_f32(3.0); 3 * D], [3, D], dev).unwrap();
    let mut ring = Tensor::from_host_slice(&vec![bf16::NAN; R * D], [R, D], dev).unwrap();
    let mut out = Tensor::zeros([n, h, D], dev).unwrap();
    Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut out).unwrap();
    s.synchronize().unwrap();
    let a = out.to_host_vec().unwrap();
    for t in [0, 126, 127, 128, 254, 255, 256] {
        let local = (t + 1).min(R) as f64;
        let blocks = ((t + 1) / R) as f64;
        let expected: Vec<_> = [
            (local + 3.0 * blocks) / (local + blocks),
            (local + 3.0 * blocks) / (local + blocks + 1.0),
            0.0,
        ]
        .into_iter()
        .flat_map(|v| vec![v; D])
        .collect();
        close(&a[t * h * D..(t + 1) * h * D], &expected);
    }
    q.upload_from_host(&vec![bf16::from_f32(100.0); n * h * D])
        .unwrap();
    Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut out).unwrap();
    s.synchronize().unwrap();
    assert_eq!(
        &out.to_host_vec().unwrap()[256 * h * D..256 * h * D + 2 * D],
        &vec![bf16::from_f32(3.0); 2 * D]
    );
    let initial = ring.to_host_vec().unwrap();
    let mut c = Compress::new(&s, n, 3);
    c.state.upload_from_host(&vec![7.0; 3 * D]).unwrap();
    c.pool.upload_from_host(&vec![bf16::ONE; 3 * D]).unwrap();
    for bad in [-1, i32::MAX, 256] {
        start.upload_from_host(&[bad]).unwrap();
        c.start.upload_from_host(&[bad]).unwrap();
        Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut out).unwrap();
        c.run(&s, 0, n);
        s.synchronize().unwrap();
        assert_eq!(ring.to_host_vec().unwrap(), initial);
        assert!(out.to_host_vec().unwrap().iter().all(|v| v.is_nan()));
        assert_eq!(c.state.to_host_vec().unwrap(), vec![7.0; 3 * D]);
        assert_eq!(c.pool.to_host_vec().unwrap(), vec![bf16::ONE; 3 * D]);
    }
    // Decode rejects invalid device positions/capacity without a ring write.
    let qd = q
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[h, D]))
        .unwrap();
    let kd = kv
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[D]))
        .unwrap();
    let mut od = out
        .narrow(0, 0, 1)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[h, D]))
        .unwrap();
    for bad in [-1, i32::MAX, 511] {
        start.upload_from_host(&[bad]).unwrap();
        c.start.upload_from_host(&[bad]).unwrap();
        Cuda::v4_hca_decode(&s, &qd, &kd, &sink, &start, &pool, &mut ring, &mut od).unwrap();
        c.run(&s, 0, 1);
        s.synchronize().unwrap();
        assert_eq!(ring.to_host_vec().unwrap(), initial);
        assert!(od.to_host_vec().unwrap().iter().all(|v| v.is_nan()));
        assert_eq!(c.state.to_host_vec().unwrap(), vec![7.0; 3 * D]);
        assert_eq!(c.pool.to_host_vec().unwrap(), vec![bf16::ONE; 3 * D]);
    }
    let mut alias = q.clone();
    assert!(
        Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &pool, &mut ring, &mut alias).is_err()
    );
    let ring_alias = ring.clone();
    assert!(
        Cuda::v4_hca_prefill(&s, &q, &kv, &sink, &start, &ring_alias, &mut ring, &mut out).is_err()
    );
    let strided = Tensor::<bf16, _>::zeros([n, h, D + 1], dev)
        .unwrap()
        .narrow(2, 0, D)
        .unwrap();
    assert!(
        Cuda::v4_hca_prefill(&s, &strided, &kv, &sink, &start, &pool, &mut ring, &mut out).is_err()
    );
    let empty = q.narrow(0, 0, 0).unwrap();
    assert!(
        Cuda::v4_hca_prefill(&s, &empty, &kv, &sink, &start, &pool, &mut ring, &mut out).is_err()
    );
    let mut state_alias = c.ape.narrow(0, 0, 3).unwrap();
    assert!(
        Cuda::v4_hca_compress(
            &s,
            &c.values,
            &c.gates,
            &c.ape,
            &c.norm,
            &c.rope,
            &c.start,
            &mut state_alias,
            &mut c.pool,
            EPS
        )
        .is_err()
    );
    assert!(
        Cuda::v4_hca_compress(
            &s,
            &c.values,
            &c.gates,
            &c.ape,
            &c.norm,
            &c.rope,
            &c.start,
            &mut c.state,
            &mut c.pool,
            0.0
        )
        .is_err()
    );
    let wrong_rope = Tensor::zeros([3, 64], dev).unwrap();
    assert!(
        Cuda::v4_hca_compress(
            &s,
            &c.values,
            &c.gates,
            &c.ape,
            &c.norm,
            &wrong_rope,
            &c.start,
            &mut c.state,
            &mut c.pool,
            EPS
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn compression_extreme_per_channel_gates_and_block_start_rope() {
    let s = scope();
    let _active = s.enter();
    let mut c = Compress::new(&s, 256, 3);
    let gates: Vec<_> = (0..256)
        .flat_map(|t| (0..D).map(move |d| if t % R == d % R { 10000.0 } else { -10000.0 }))
        .collect();
    c.gates.upload_from_host(&gates).unwrap();
    c.run(&s, 0, 256);
    s.synchronize().unwrap();
    let expected = compression_reference(&c, 256);
    let actual = c.pool.to_host_vec().unwrap();
    // One overwhelmingly preferred token per channel; dense oracle checks that
    // token reduction does not accidentally normalize over the feature axis.
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && (a.to_f32() - b.to_f32()).abs() <= 0.001 + 0.008 * b.to_f32().abs(),
            "extreme gate element {i}: {a} vs {b}"
        );
    }
    assert!(actual[2 * D..].iter().all(|v| v.is_nan()));
    let state = c.state.to_host_vec().unwrap();
    assert_eq!(&state[..D], &[f32::NEG_INFINITY; D]);
    assert_eq!(&state[D..], &[0.0; 2 * D]);
}
