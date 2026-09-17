//! Independent numeric oracles; run with --ignored --test-threads=1 on CUDA.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

const D: usize = 512;
const R: usize = 4;
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
        let gates: Vec<_> = data(n * 2 * D, 37).into_iter().map(|v| v * 8.0).collect();
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
            values: Tensor::from_host_slice(&data(n * 2 * D, 71), [n, 2 * D], dev).unwrap(),
            gates: Tensor::from_host_slice(&gates, [n, 2 * D], dev).unwrap(),
            ape: Tensor::from_host_slice(&data(R * 2 * D, 47), [R, 2 * D], dev).unwrap(),
            norm: Tensor::from_host_slice(&norm, [D], dev).unwrap(),
            rope: Tensor::from_host_slice(&rope, [capacity, 32, 2], dev).unwrap(),
            start: Tensor::from_host_slice(&[0], [1], dev).unwrap(),
            state: Tensor::from_host_slice(&vec![f32::NAN; 9 * D], [3, 3, D], dev).unwrap(),
            pool: Tensor::from_host_slice(&vec![bf16::NAN; capacity * D], [capacity, D], dev)
                .unwrap(),
        }
    }
    fn run(&mut self, s: &CudaScope, offset: usize, n: usize) {
        let v = self.values.narrow(0, offset, n).unwrap();
        let g = self.gates.narrow(0, offset, n).unwrap();
        Cuda::v4_csa_compress(
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

// Two-pass FP64 softmax over the actual eight rows, without online statistics
// or merging two independently normalized halves. Round at BF16 boundaries.
fn reference(c: &Compress, n: usize) -> Vec<(bf16, f32)> {
    let v = c.values.to_host_vec().unwrap();
    let g = c.gates.to_host_vec().unwrap();
    let ape = c.ape.to_host_vec().unwrap();
    let norm = c.norm.to_host_vec().unwrap();
    let rope = c.rope.to_host_vec().unwrap();
    let mut out = Vec::new();
    for block in 0..n / R {
        let mut pooled = Vec::new();
        for d in 0..D {
            let rows: Vec<_> = ((block * R).saturating_sub(R)..(block + 1) * R)
                .map(|t| {
                    let half = usize::from(t >= block * R);
                    let src = t * 2 * D + half * D + d;
                    (
                        f64::from(v[src]),
                        f64::from(g[src] + ape[t % R * 2 * D + half * D + d]),
                    )
                })
                .collect();
            let max = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = rows.iter().map(|r| (r.1 - max).exp()).sum();
            let value: f64 = rows.iter().map(|r| r.0 * (r.1 - max).exp()).sum::<f64>() / z;
            pooled.push(bf16::from_f64(value).to_f64());
        }
        let inv = (pooled.iter().map(|v| v * v).sum::<f64>() / D as f64 + f64::from(EPS))
            .sqrt()
            .recip();
        let normalized: Vec<_> = pooled
            .iter()
            .zip(&norm)
            .map(|(v, w)| bf16::from_f64(v * inv * f64::from(*w)))
            .collect();
        let mut row = normalized.clone();
        let mut bounds: Vec<_> = normalized
            .iter()
            .map(|v| 0.001 + 0.008 * v.to_f32().abs())
            .collect();
        for pair in 0..32 {
            let d = D - 64 + 2 * pair;
            let a = normalized[d].to_f32();
            let b = normalized[d + 1].to_f32();
            let cos = rope[block * 64 + 2 * pair];
            let sin = rope[block * 64 + 2 * pair + 1];
            row[d] = bf16::from_f32(a * cos - b * sin);
            row[d + 1] = bf16::from_f32(a * sin + b * cos);
            // FP32 vs FP64 pooling may straddle an intermediate BF16 rounding
            // boundary. Propagate that error through both rotation terms:
            // relative error on the final difference is meaningless near zero.
            bounds[d] = 0.001 + 0.008 * ((a * cos).abs() + (b * sin).abs());
            bounds[d + 1] = 0.001 + 0.008 * ((a * sin).abs() + (b * cos).abs());
        }
        out.extend(row.into_iter().zip(bounds));
    }
    out
}
fn close(actual: &[bf16], expected: &[(bf16, f32)]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, (b, bound))) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a.to_f32() - b.to_f32()).abs() <= *bound,
            "compressed element {i}: {a} vs {b}"
        );
    }
}
fn check_state(c: &Compress, n: usize) {
    let v = c.values.to_host_vec().unwrap();
    let g = c.gates.to_host_vec().unwrap();
    let ape = c.ape.to_host_vec().unwrap();
    let state = c.state.to_host_vec().unwrap();
    let cutoff = n / R * R;
    for group in 0..3 {
        let (begin, end, half) = if group == 0 {
            (cutoff.saturating_sub(R), cutoff, 0)
        } else {
            (cutoff, n, group - 1)
        };
        for d in 0..D {
            let scores: Vec<_> = (begin..end)
                .map(|t| f64::from(g[t * 2 * D + half * D + d] + ape[t % R * 2 * D + half * D + d]))
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = scores.iter().map(|x| (x - max).exp()).sum();
            let u: f64 = (begin..end)
                .zip(&scores)
                .map(|(t, x)| f64::from(v[t * 2 * D + half * D + d]) * (x - max).exp())
                .sum();
            for (plane, b) in [max, z, u].into_iter().enumerate() {
                let a = f64::from(state[(group * 3 + plane) * D + d]);
                assert!(
                    a == b || (a - b).abs() <= 2e-6 + 2e-6 * b.abs(),
                    "state group {group} plane {plane} channel {d}: {a} vs {b}"
                );
            }
        }
    }
}
fn equal(c: &Compress, other: &Compress, n: usize) {
    let a = c.pool.to_host_vec().unwrap();
    let b = other.pool.to_host_vec().unwrap();
    assert_eq!(&a[..n / R * D], &b[..n / R * D]);
    assert!(b[n / R * D..].iter().all(|v| v.is_nan()));
    assert_eq!(
        c.state.to_host_vec().unwrap(),
        other.state.to_host_vec().unwrap()
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn full_chunks_and_dynamic_decode_graph_match_dense_overlap() {
    let s = scope();
    let _active = s.enter();
    let n = 259;
    let cap = n / R + 2;
    let mut full = Compress::new(&s, n, cap);
    let mut chunk = Compress::new(&s, n, cap);
    let mut serial = Compress::new(&s, 1, cap);
    full.run(&s, 0, n);
    let mut offset = 0;
    for count in [1, 2, 1, 7, 3, 4, 63, 128, 50] {
        chunk.start.upload_from_host(&[offset as i32]).unwrap();
        chunk.run(&s, offset, count);
        offset += count;
    }
    assert_eq!(offset, n);
    s.synchronize().unwrap();
    close(
        &full.pool.to_host_vec().unwrap()[..n / R * D],
        &reference(&full, n),
    );
    check_state(&full, n);
    equal(&full, &chunk, n);
    let values = full.values.to_host_vec().unwrap();
    let gates = full.gates.to_host_vec().unwrap();
    s.graph_capture_begin().unwrap();
    serial.run(&s, 0, 1);
    s.graph_capture_end(2901).unwrap();
    let before = s.device().config.pool_stats();
    for t in 0..n {
        serial
            .values
            .upload_from_host(&values[t * 2 * D..(t + 1) * 2 * D])
            .unwrap();
        serial
            .gates
            .upload_from_host(&gates[t * 2 * D..(t + 1) * 2 * D])
            .unwrap();
        serial.start.upload_from_host(&[t as i32]).unwrap();
        s.graph_launch(2901).unwrap();
        s.synchronize().unwrap();
        if t < 8 {
            let pool = serial.pool.to_host_vec().unwrap();
            assert!(pool[..(t + 1) / R * D].iter().all(|v| v.is_finite()));
            assert!(pool[(t + 1) / R * D..].iter().all(|v| v.is_nan()));
        }
    }
    assert_eq!(s.device().config.pool_stats(), before);
    equal(&full, &serial, n);
    s.device().config.invalidate_all_graphs();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn every_prefix_and_chunk_boundary_preserves_overlap_state() {
    let s = scope();
    let _active = s.enter();
    // Every start slot, empty history, no emission, one emission and multiple
    // emissions, including a chunk that ends exactly on a block boundary.
    for prefix in 0..12 {
        for count in 1..=12 {
            let n = prefix + count;
            let mut full = Compress::new(&s, n, n / R + 1);
            let mut chunk = Compress::new(&s, n, n / R + 1);
            full.run(&s, 0, n);
            if prefix > 0 {
                chunk.run(&s, 0, prefix);
            }
            chunk.start.upload_from_host(&[prefix as i32]).unwrap();
            chunk.run(&s, prefix, count);
            s.synchronize().unwrap();
            equal(&full, &chunk, n);
            check_state(&chunk, n);
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn prefill_graph_updates_inputs_positions_and_restarts_request() {
    let s = scope();
    let _active = s.enter();
    let n = 70;
    let cap = n / R + 1;
    let mut full = Compress::new(&s, n, cap);
    let mut chunk = Compress::new(&s, 7, cap);
    full.run(&s, 0, n);
    s.synchronize().unwrap();
    let values = full.values.to_host_vec().unwrap();
    let gates = full.gates.to_host_vec().unwrap();
    s.graph_capture_begin().unwrap();
    chunk.run(&s, 0, 7);
    s.graph_capture_end(2902).unwrap();
    let before = s.device().config.pool_stats();
    for offset in (0..n).step_by(7) {
        chunk
            .values
            .upload_from_host(&values[offset * 2 * D..(offset + 7) * 2 * D])
            .unwrap();
        chunk
            .gates
            .upload_from_host(&gates[offset * 2 * D..(offset + 7) * 2 * D])
            .unwrap();
        chunk.start.upload_from_host(&[offset as i32]).unwrap();
        s.graph_launch(2902).unwrap();
    }
    s.synchronize().unwrap();
    equal(&full, &chunk, n);
    // A new request at zero discards old state, without touching future cache.
    chunk.values.upload_from_host(&values[..7 * 2 * D]).unwrap();
    chunk.gates.upload_from_host(&gates[..7 * 2 * D]).unwrap();
    chunk.start.upload_from_host(&[0]).unwrap();
    let old_pool = chunk.pool.to_host_vec().unwrap();
    s.graph_launch(2902).unwrap();
    s.synchronize().unwrap();
    check_state(&chunk, 7);
    let new_pool = chunk.pool.to_host_vec().unwrap();
    assert_eq!(&new_pool[..n / R * D], &old_pool[..n / R * D]);
    assert_eq!(s.device().config.pool_stats(), before);
    s.device().config.invalidate_all_graphs();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn learned_bias_extreme_gates_and_previous_half_use_one_softmax() {
    let s = scope();
    let _active = s.enter();
    let n = 12;
    let mut c = Compress::new(&s, n, 4);
    let values = data(n * 2 * D, 193);
    let mut gates = vec![-10000.0; n * 2 * D];
    let mut ape = vec![0.0; R * 2 * D];
    // Different channels select previous/current halves and different token
    // slots. Bias picks the winning slot; block zero must mask absent history.
    for t in 0..n {
        for half in 0..2 {
            for d in 0..D {
                gates[t * 2 * D + half * D + d] = if d % 2 == half { 10000.0 } else { -10000.0 };
                ape[t % R * 2 * D + half * D + d] =
                    if t % R == (d / 2) % R { 100.0 } else { -100.0 };
            }
        }
    }
    c.values.upload_from_host(&values).unwrap();
    c.gates.upload_from_host(&gates).unwrap();
    c.ape.upload_from_host(&ape).unwrap();
    c.run(&s, 0, n);
    s.synchronize().unwrap();
    close(
        &c.pool.to_host_vec().unwrap()[..n / R * D],
        &reference(&c, n),
    );
    check_state(&c, n);
    // Ordinary unequal weights require one joint denominator, not an average
    // of two separately normalized vectors. Also exercises destructive sums.
    for t in 0..n {
        for d in 0..2 * D {
            gates[t * 2 * D + d] = if d < D { 1.25 } else { -0.75 };
        }
    }
    c.gates.upload_from_host(&gates).unwrap();
    c.ape.upload_from_host(&vec![0.0; R * 2 * D]).unwrap();
    c.run(&s, 0, n);
    s.synchronize().unwrap();
    close(
        &c.pool.to_host_vec().unwrap()[..n / R * D],
        &reference(&c, n),
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn first_half_influences_next_block_only_and_future_tokens_are_causal() {
    let s = scope();
    let _active = s.enter();
    let mut c = Compress::new(&s, 16, 5);
    c.run(&s, 0, 16);
    s.synchronize().unwrap();
    let base = c.pool.to_host_vec().unwrap();
    let mut values = c.values.to_host_vec().unwrap();
    for t in 0..R {
        for d in 0..D {
            values[t * 2 * D + d] = (d % 17) as f32 - 8.0;
        }
    }
    c.values.upload_from_host(&values).unwrap();
    c.run(&s, 0, 16);
    s.synchronize().unwrap();
    let a = c.pool.to_host_vec().unwrap();
    assert_eq!(&a[..D], &base[..D]);
    assert_ne!(&a[D..2 * D], &base[D..2 * D]);
    assert_eq!(&a[2 * D..4 * D], &base[2 * D..4 * D]);
    close(&a[..4 * D], &reference(&c, 16));
    // Changing future input cannot affect already completed rows.
    values[12 * 2 * D..].fill(20.0);
    c.values.upload_from_host(&values).unwrap();
    c.run(&s, 0, 16);
    s.synchronize().unwrap();
    assert_eq!(&c.pool.to_host_vec().unwrap()[..3 * D], &a[..3 * D]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn invalid_positions_capacity_shapes_and_aliases_do_not_write() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let mut c = Compress::new(&s, 12, 2);
    c.state.upload_from_host(&vec![7.0; 9 * D]).unwrap();
    c.pool.upload_from_host(&vec![bf16::ONE; 2 * D]).unwrap();
    for n in [1, 2, 12] {
        for bad in [-1, i32::MAX, 11] {
            c.start.upload_from_host(&[bad]).unwrap();
            c.run(&s, 0, n);
            s.synchronize().unwrap();
            assert_eq!(c.state.to_host_vec().unwrap(), vec![7.0; 9 * D]);
            assert_eq!(c.pool.to_host_vec().unwrap(), vec![bf16::ONE; 2 * D]);
        }
    }
    c.start.upload_from_host(&[0]).unwrap();
    // C is completed-block capacity: eleven tokens fit C=2, twelve do not.
    c.run(&s, 0, 11);
    s.synchronize().unwrap();
    close(&c.pool.to_host_vec().unwrap(), &reference(&c, 11));
    check_state(&c, 11);
    let state = c.state.to_host_vec().unwrap();
    let pool = c.pool.to_host_vec().unwrap();
    c.start.upload_from_host(&[11]).unwrap();
    c.run(&s, 11, 1);
    s.synchronize().unwrap();
    assert_eq!(c.state.to_host_vec().unwrap(), state);
    assert_eq!(c.pool.to_host_vec().unwrap(), pool);
    let mut call = |values: &Tensor<f32, Cuda>,
                    gates: &Tensor<f32, Cuda>,
                    ape: &Tensor<f32, Cuda>,
                    rope: &Tensor<f32, Cuda>,
                    state: &mut Tensor<f32, Cuda>,
                    eps| {
        Cuda::v4_csa_compress(
            &s,
            values,
            gates,
            ape,
            &c.norm,
            rope,
            &c.start,
            state,
            &mut c.pool,
            eps,
        )
    };
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(call(&c.values, &c.gates, &c.ape, &c.rope, &mut c.state, eps).is_err());
    }
    let empty = c.values.narrow(0, 0, 0).unwrap();
    assert!(call(&empty, &empty, &c.ape, &c.rope, &mut c.state, EPS).is_err());
    let wrong = Tensor::zeros([12, D], dev).unwrap();
    assert!(call(&wrong, &wrong, &c.ape, &c.rope, &mut c.state, EPS).is_err());
    let strided = Tensor::<f32, _>::zeros([12, 2 * D + 1], dev)
        .unwrap()
        .narrow(1, 0, 2 * D)
        .unwrap();
    assert!(call(&strided, &c.gates, &c.ape, &c.rope, &mut c.state, EPS).is_err());
    let wrong_rope = Tensor::zeros([2, 64, 2], dev).unwrap();
    assert!(call(&c.values, &c.gates, &c.ape, &wrong_rope, &mut c.state, EPS).is_err());
    let alias = c
        .state
        .clone()
        .view_contiguous(Shape::from_slice(&[9 * D]))
        .unwrap()
        .narrow(0, 0, 8 * D)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[R, 2 * D]))
        .unwrap();
    assert!(call(&c.values, &c.gates, &alias, &c.rope, &mut c.state, EPS).is_err());
    let mut wrong_state = Tensor::zeros([9, D], dev).unwrap();
    assert!(call(&c.values, &c.gates, &c.ape, &c.rope, &mut wrong_state, EPS).is_err());
}
