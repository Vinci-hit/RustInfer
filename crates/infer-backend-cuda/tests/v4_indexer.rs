//! Independent numeric oracles; run with --ignored --test-threads=1 on CUDA.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

const D: usize = 128;
const R: usize = 4;

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
struct Inputs {
    q: Tensor<bf16, Cuda>,
    k: Tensor<bf16, Cuda>,
    weights: Tensor<f32, Cuda>,
    start: Tensor<i32, Cuda>,
    output: Tensor<f32, Cuda>,
}
impl Inputs {
    fn new(s: &CudaScope, n: usize, h: usize, c: usize, start: i32) -> Self {
        let dev = s.device();
        Self {
            q: Tensor::from_host_slice(&bf(n * h * D, 19), [n, h, D], dev).unwrap(),
            k: Tensor::from_host_slice(&bf(c * D, 31), [c, D], dev).unwrap(),
            weights: Tensor::from_host_slice(
                &data(n * h, 47)
                    .into_iter()
                    .map(|v| v / ((D * h) as f32).sqrt())
                    .collect::<Vec<_>>(),
                [n, h],
                dev,
            )
            .unwrap(),
            start: Tensor::from_host_slice(&[start], [1], dev).unwrap(),
            output: Tensor::from_host_slice(&vec![f32::NAN; n * c], [n, c], dev).unwrap(),
        }
    }
    fn run(&mut self, s: &CudaScope) {
        Cuda::v4_indexer_scores(
            s,
            &self.q,
            &self.k,
            &self.weights,
            &self.start,
            &mut self.output,
        )
        .unwrap();
    }
}

// FP64 scalar dot products and head reduction, independent of WMMA tiling.
// Absolute-product sum bounds FP32 accumulation error even when heads cancel.
fn reference(q: &[bf16], k: &[bf16], weights: &[f32], h: usize, visible: usize) -> Vec<(f64, f64)> {
    k.chunks_exact(D)
        .enumerate()
        .map(|(j, key)| {
            if j >= visible {
                return (f64::NEG_INFINITY, 0.0);
            }
            let mut score = 0.0;
            let mut magnitude = 0.0;
            for head in 0..h {
                let mut dot = 0.0;
                let mut abs_products = 0.0;
                for d in 0..D {
                    let product = q[head * D + d].to_f64() * key[d].to_f64();
                    dot += product;
                    abs_products += product.abs();
                }
                let w = f64::from(weights[head]);
                score += dot.max(0.0) * w;
                magnitude += abs_products * w.abs();
            }
            (score, 1e-6 + 1e-6 * magnitude)
        })
        .collect()
}
fn close(actual: &[f32], expected: &[(f64, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for (j, (&a, &(b, tol))) in actual.iter().zip(expected).enumerate() {
        if b == f64::NEG_INFINITY {
            assert_eq!(a, f32::NEG_INFINITY, "masked column {j}");
        } else {
            assert!(
                a.is_finite() && (f64::from(a) - b).abs() <= tol,
                "score {j}: {a} vs {b}, tolerance {tol}"
            );
        }
    }
}
fn check(c: &Inputs) {
    let [n, h, _] = <[usize; 3]>::try_from(c.q.shape().as_slice()).unwrap();
    let cap = c.k.shape().as_slice()[0];
    let start = c.start.to_host_vec().unwrap()[0] as usize;
    let q = c.q.to_host_vec().unwrap();
    let k = c.k.to_host_vec().unwrap();
    let w = c.weights.to_host_vec().unwrap();
    let out = c.output.to_host_vec().unwrap();
    for t in 0..n {
        close(
            &out[t * cap..(t + 1) * cap],
            &reference(
                &q[t * h * D..(t + 1) * h * D],
                &k,
                &w[t * h..(t + 1) * h],
                h,
                (start + t + 1) / R,
            ),
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn real_head_count_full_chunks_and_decode_match_fp64() {
    let s = scope();
    let _active = s.enter();
    let (n, h, c) = (137, 64, 37);
    let mut full = Inputs::new(&s, n, h, c, 0);
    let chunks = Tensor::<f32, _>::zeros([n, c], s.device()).unwrap();
    let serial = Tensor::<f32, _>::zeros([n, c], s.device()).unwrap();
    full.run(&s);
    let mut offset = 0;
    let before = s.device().config.pool_stats();
    for count in [1, 2, 1, 7, 63, 1, 62] {
        full.start.upload_from_host(&[offset as i32]).unwrap();
        Cuda::v4_indexer_scores(
            &s,
            &full.q.narrow(0, offset, count).unwrap(),
            &full.k,
            &full.weights.narrow(0, offset, count).unwrap(),
            &full.start,
            &mut chunks.narrow(0, offset, count).unwrap(),
        )
        .unwrap();
        offset += count;
    }
    assert_eq!(offset, n);
    for t in 0..n {
        full.start.upload_from_host(&[t as i32]).unwrap();
        Cuda::v4_indexer_scores(
            &s,
            &full.q.narrow(0, t, 1).unwrap(),
            &full.k,
            &full.weights.narrow(0, t, 1).unwrap(),
            &full.start,
            &mut serial.narrow(0, t, 1).unwrap(),
        )
        .unwrap();
    }
    s.synchronize().unwrap();
    assert_eq!(s.device().config.pool_stats(), before);
    assert_eq!(
        full.output.to_host_vec().unwrap(),
        chunks.to_host_vec().unwrap()
    );
    assert_eq!(
        full.output.to_host_vec().unwrap(),
        serial.to_host_vec().unwrap()
    );
    full.start.upload_from_host(&[0]).unwrap();
    check(&full);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn partial_heads_and_key_tiles_mask_poisoned_future_rows() {
    let s = scope();
    let _active = s.enter();
    for h in [1, 7, 16, 17, 63, 64, 65, 128] {
        let mut c = Inputs::new(&s, 7, h, 67, 251);
        let mut keys = c.k.to_host_vec().unwrap();
        keys[64 * D..].fill(bf16::NAN);
        c.k.upload_from_host(&keys).unwrap();
        c.run(&s);
        s.synchronize().unwrap();
        check(&c);
    }
    // Guard the compact FP32 output around a partial final tile, including
    // pointer alignment weaker than 32 bytes (only shared WMMA needs 32).
    let mut c = Inputs::new(&s, 2, 17, 65, 258);
    let guard = Tensor::from_host_slice(&vec![-987.0f32; 132], [132], s.device()).unwrap();
    c.output = guard
        .narrow(0, 1, 130)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[2, 65]))
        .unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    check(&c);
    let host = guard.to_host_vec().unwrap();
    assert_eq!(host[0], -987.0);
    assert_eq!(host[131], -987.0);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn relu_is_per_head_weights_are_signed_and_mask_is_negative_infinity() {
    let s = scope();
    let _active = s.enter();
    let mut c = Inputs::new(&s, 1, 2, 3, 7);
    let mut q = vec![bf16::ONE; 2 * D];
    q[D..].fill(bf16::from_f32(-1.0));
    let mut k = vec![bf16::NAN; 3 * D];
    k[..D].fill(bf16::ONE);
    k[D..2 * D].fill(bf16::from_f32(-1.0));
    c.q.upload_from_host(&q).unwrap();
    c.k.upload_from_host(&k).unwrap();
    c.weights.upload_from_host(&[1.0, -2.0]).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    assert_eq!(
        c.output.to_host_vec().unwrap(),
        vec![128.0, -256.0, f32::NEG_INFINITY]
    );
    c.weights.upload_from_host(&[0.0, 0.0]).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    assert_eq!(
        c.output.to_host_vec().unwrap(),
        vec![0.0, 0.0, f32::NEG_INFINITY]
    );
    // No completed block at positions 0..2. No key row is initialized.
    c.k.upload_from_host(&vec![bf16::NAN; 3 * D]).unwrap();
    for pos in 0..3 {
        c.start.upload_from_host(&[pos]).unwrap();
        c.run(&s);
        s.synchronize().unwrap();
        assert_eq!(c.output.to_host_vec().unwrap(), vec![f32::NEG_INFINITY; 3]);
    }
    c.start.upload_from_host(&[3]).unwrap();
    c.k.upload_from_host(&k).unwrap();
    c.weights.upload_from_host(&[1.0, -2.0]).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    assert_eq!(
        c.output.to_host_vec().unwrap(),
        vec![128.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn graph_replay_uses_new_queries_weights_keys_and_positions_without_allocation() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let (n, h, cap) = (7, 64, 69);
    let mut c = Inputs::new(&s, n, h, cap, 0);
    s.synchronize().unwrap();
    s.graph_capture_begin().unwrap();
    c.run(&s);
    s.graph_capture_end(3901).unwrap();
    let before = dev.config.pool_stats();
    for (run, pos) in [0, 1, 3, 4, 251, 252, 263, 0].into_iter().enumerate() {
        c.q.upload_from_host(&bf(n * h * D, 97 + run as u32))
            .unwrap();
        c.weights
            .upload_from_host(
                &data(n * h, 111 + run as u32)
                    .into_iter()
                    .map(|v| v / ((D * h) as f32).sqrt())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut k = bf(cap * D, 177 + run as u32);
        k[((pos + n) / R) * D..].fill(bf16::NAN);
        c.k.upload_from_host(&k).unwrap();
        c.start.upload_from_host(&[pos as i32]).unwrap();
        s.graph_launch(3901).unwrap();
        s.synchronize().unwrap();
        check(&c);
    }
    assert_eq!(dev.config.pool_stats(), before);
    dev.config.invalidate_all_graphs();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn long_history_decode_and_extreme_finite_logits_stay_fp32() {
    let s = scope();
    let _active = s.enter();
    let mut c = Inputs::new(&s, 1, 64, 8193, 32767);
    let mut k = c.k.to_host_vec().unwrap();
    k[8192 * D..].fill(bf16::NAN);
    c.k.upload_from_host(&k).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    // Check every tile plus edge rows against the independent scalar oracle.
    let q = c.q.to_host_vec().unwrap();
    let w = c.weights.to_host_vec().unwrap();
    let out = c.output.to_host_vec().unwrap();
    for j in (0..8192).step_by(64).chain([8191]) {
        close(
            &out[j..j + 1],
            &reference(&q, &k[j * D..(j + 1) * D], &w, 64, 1),
        );
    }
    assert!(out[..8192].iter().all(|v| v.is_finite()));
    assert_eq!(out[8192], f32::NEG_INFINITY);
    c.q.upload_from_host(&vec![bf16::from_f32(1024.0); 64 * D])
        .unwrap();
    k[..8192 * D].fill(bf16::from_f32(1024.0));
    c.k.upload_from_host(&k).unwrap();
    c.weights.upload_from_host(&vec![1.0; 64]).unwrap();
    c.run(&s);
    s.synchronize().unwrap();
    let expected = (64 * 128 * 1024 * 1024u64) as f32;
    let out = c.output.to_host_vec().unwrap();
    assert!(out[..8192].iter().all(|v| *v == expected));
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn invalid_positions_metadata_and_aliases_are_rejected() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let mut c = Inputs::new(&s, 7, 17, 3, 0);
    let keys = c.k.to_host_vec().unwrap();
    let query = c.q.to_host_vec().unwrap();
    for bad in [-1, i32::MAX, 9] {
        c.start.upload_from_host(&[bad]).unwrap();
        c.run(&s);
        s.synchronize().unwrap();
        assert!(c.output.to_host_vec().unwrap().iter().all(|v| v.is_nan()));
        assert_eq!(c.start.to_host_vec().unwrap(), [bad]);
        assert_eq!(c.k.to_host_vec().unwrap(), keys);
        assert_eq!(c.q.to_host_vec().unwrap(), query);
    }
    let mut d = Inputs::new(&s, 1, 17, 3, 0);
    for bad in [-1, i32::MAX, 15] {
        d.start.upload_from_host(&[bad]).unwrap();
        d.run(&s);
        s.synchronize().unwrap();
        assert!(d.output.to_host_vec().unwrap().iter().all(|v| v.is_nan()));
    }
    let call = |q: &Tensor<bf16, Cuda>,
                k: &Tensor<bf16, Cuda>,
                w: &Tensor<f32, Cuda>,
                out: &mut Tensor<f32, Cuda>| {
        Cuda::v4_indexer_scores(&s, q, k, w, &c.start, out)
    };
    let empty = c.q.narrow(0, 0, 0).unwrap();
    assert!(call(&empty, &c.k, &c.weights, &mut c.output).is_err());
    let wrong_d = Tensor::zeros([7, 17, 64], dev).unwrap();
    assert!(call(&wrong_d, &c.k, &c.weights, &mut c.output).is_err());
    let wrong_h = Tensor::zeros([7, 129, D], dev).unwrap();
    assert!(call(&wrong_h, &c.k, &c.weights, &mut c.output).is_err());
    let wrong_k = Tensor::zeros([3, 64], dev).unwrap();
    assert!(call(&c.q, &wrong_k, &c.weights, &mut c.output).is_err());
    let zero_k = c.k.narrow(0, 0, 0).unwrap();
    assert!(call(&c.q, &zero_k, &c.weights, &mut c.output).is_err());
    let wrong_w = Tensor::zeros([7, 16], dev).unwrap();
    assert!(call(&c.q, &c.k, &wrong_w, &mut c.output).is_err());
    let mut wrong_out = Tensor::zeros([7, 4], dev).unwrap();
    assert!(call(&c.q, &c.k, &c.weights, &mut wrong_out).is_err());
    let strided = Tensor::<bf16, _>::zeros([7, 17, D + 1], dev)
        .unwrap()
        .narrow(2, 0, D)
        .unwrap();
    assert!(call(&strided, &c.k, &c.weights, &mut c.output).is_err());
    let misaligned = Tensor::<bf16, _>::zeros([7 * 17 * D + 1], dev)
        .unwrap()
        .narrow(0, 1, 7 * 17 * D)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[7, 17, D]))
        .unwrap();
    assert!(call(&misaligned, &c.k, &c.weights, &mut c.output).is_err());
    let mut alias = c
        .weights
        .clone()
        .view_contiguous(Shape::from_slice(&[7 * 17]))
        .unwrap()
        .narrow(0, 0, 7 * 3)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[7, 3]))
        .unwrap();
    assert!(call(&c.q, &c.k, &c.weights, &mut alias).is_err());
}
