//! Independent numeric oracles; run with --ignored --test-threads=1 on CUDA.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

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
fn reference(row: &[f32], visible: usize, k: usize) -> Vec<i32> {
    let mut ids: Vec<_> = (0..visible)
        .filter(|&j| row[j] > f32::NEG_INFINITY)
        .collect();
    ids.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    let mut out = vec![-1; k];
    for (slot, id) in out.iter_mut().zip(ids) {
        *slot = id as i32;
    }
    out
}
struct Select {
    scores: Tensor<f32, Cuda>,
    start: Tensor<i32, Cuda>,
    work: Tensor<i32, Cuda>,
    ids: Tensor<i32, Cuda>,
}
impl Select {
    fn new(s: &CudaScope, n: usize, c: usize, k: usize, start: i32) -> Self {
        let dev = s.device();
        Self {
            scores: Tensor::from_host_slice(&data(n * c, 31), [n, c], dev).unwrap(),
            start: Tensor::from_host_slice(&[start], [1], dev).unwrap(),
            work: Tensor::zeros(
                [Cuda::v4_indexer_topk_workspace_words(n, c, k).unwrap()],
                dev,
            )
            .unwrap(),
            ids: Tensor::from_host_slice(&vec![-777; n * k], [n, k], dev).unwrap(),
        }
    }
    fn run(&mut self, s: &CudaScope) {
        Cuda::v4_indexer_topk(s, &self.scores, &self.start, &mut self.work, &mut self.ids).unwrap();
    }
    fn check(&self) {
        let shape = self.scores.shape().as_slice();
        let (n, c) = (shape[0], shape[1]);
        let k = self.ids.shape().as_slice()[1];
        let start = self.start.to_host_vec().unwrap()[0] as usize;
        let scores = self.scores.to_host_vec().unwrap();
        let ids = self.ids.to_host_vec().unwrap();
        for t in 0..n {
            assert_eq!(
                &ids[t * k..(t + 1) * k],
                reference(&scores[t * c..(t + 1) * c], (start + t + 1) / 4, k),
                "row {t}, C={c}, K={k}, start={start}"
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn all_radix_sizes_and_odd_merge_trees_match_stable_cpu_sort() {
    let s = scope();
    let _active = s.enter();
    for c in [
        1, 17, 255, 256, 257, 511, 512, 513, 1024, 1025, 2047, 2048, 2049, 4096, 4097, 8193,
        131073, 262144,
    ] {
        for k in [1, 7, 511, 512] {
            let mut a = Select::new(&s, 3, c, k, (c * 4 - 2) as i32);
            a.run(&s);
            s.synchronize().unwrap();
            a.check();
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn ties_zeros_nonfinite_scores_and_future_entries_have_defined_order() {
    let s = scope();
    let _active = s.enter();
    let c = 4097;
    let mut a = Select::new(&s, 3, c, 512, (c * 4 - 2) as i32);
    let mut scores = vec![f32::NEG_INFINITY; 3 * c];
    for (j, v) in scores[..c].iter_mut().enumerate() {
        *v = if j % 2 == 0 { 0.0 } else { -0.0 };
    }
    scores[c..2 * c].fill(f32::NAN);
    for (j, v) in scores[2 * c..].iter_mut().enumerate() {
        *v = match j % 5 {
            0 => f32::NAN,
            1 => f32::NEG_INFINITY,
            2 => -1000.0,
            3 => f32::INFINITY,
            _ => -1.0,
        };
    }
    a.scores.upload_from_host(&scores).unwrap();
    a.run(&s);
    s.synchronize().unwrap();
    a.check();
    let out = a.ids.to_host_vec().unwrap();
    assert_eq!(&out[..512], &(0..512).collect::<Vec<i32>>());
    assert!(out[512..1024].iter().all(|&v| v == -1));
    let mut one = Select::new(&s, 1, c, 512, 0);
    for pos in [0, 1, 2, 3, 4, 7, 2047, 2048, 8191, 8192, 16386, 16387] {
        let visible = (pos + 1) / 4;
        let mut row = vec![f32::INFINITY; c];
        for (j, v) in row[..visible].iter_mut().enumerate() {
            *v = if j % 3 == 0 { f32::NAN } else { -(j as f32) };
        }
        one.scores.upload_from_host(&row).unwrap();
        one.start.upload_from_host(&[pos as i32]).unwrap();
        one.run(&s);
        s.synchronize().unwrap();
        one.check();
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn graph_replay_updates_scores_and_positions_and_overwrites_padding() {
    let s = scope();
    let _active = s.enter();
    let n = 7;
    let c = 8193;
    let k = 512;
    let mut a = Select::new(&s, n, c, k, 0);
    s.synchronize().unwrap();
    s.graph_capture_begin().unwrap();
    a.run(&s);
    s.graph_capture_end(4901).unwrap();
    let before = s.device().config.pool_stats();
    for (run, pos) in [0, 1, 3, 252, 8190, 32765, 0].into_iter().enumerate() {
        let mut scores = data(n * c, 177 + run as u32);
        if run % 2 == 0 {
            for v in &mut scores {
                *v = (*v * 8.0).round();
            }
        }
        a.scores.upload_from_host(&scores).unwrap();
        a.start.upload_from_host(&[pos]).unwrap();
        s.graph_launch(4901).unwrap();
        s.synchronize().unwrap();
        a.check();
    }
    assert_eq!(s.device().config.pool_stats(), before);
    s.device().config.invalidate_all_graphs();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn scoring_selection_pipeline_full_chunks_and_decode_agree_exactly() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let (n, h, c, k) = (19, 64, 2053, 512);
    let start = 8190;
    let mut q = vec![bf16::ZERO; n * h * 128];
    let mut keys = vec![bf16::ZERO; c * 128];
    let mut w = vec![0.0; n * h];
    for t in 0..n {
        for head in 0..h {
            q[(t * h + head) * 128] = bf16::from_f32(if t % 2 == 0 { 1.0 } else { -1.0 });
            w[t * h + head] = if t % 3 == 0 { -1.0 / 64.0 } else { 1.0 / 64.0 };
        }
    }
    for j in 0..c {
        keys[j * 128] = bf16::from_f32((j % 257) as f32 - 128.0);
    }
    let q = Tensor::from_host_slice(&q, [n, h, 128], dev).unwrap();
    let keys = Tensor::from_host_slice(&keys, [c, 128], dev).unwrap();
    let w = Tensor::from_host_slice(&w, [n, h], dev).unwrap();
    let mut full = Select::new(&s, n, c, k, start);
    Cuda::v4_indexer_scores(&s, &q, &keys, &w, &full.start, &mut full.scores).unwrap();
    full.run(&s);
    s.synchronize().unwrap();
    full.check();
    // Analytic score: the 64 identical heads with weights +/-1/64 cancel the
    // head count exactly, so no rank ambiguity from FP32 vs FP64 dot rounding.
    let actual = full.scores.to_host_vec().unwrap();
    for t in 0..n {
        for j in 0..c {
            let visible = (start as usize + t + 1) / 4;
            let dot = ((j % 257) as f32 - 128.0) * if t % 2 == 0 { 1.0 } else { -1.0 };
            let expected = if j >= visible {
                f32::NEG_INFINITY
            } else {
                dot.max(0.0) * if t % 3 == 0 { -1.0 } else { 1.0 }
            };
            assert_eq!(actual[t * c + j], expected);
        }
    }
    let chunk_scores = Tensor::<f32, _>::zeros([n, c], dev).unwrap();
    let chunk_ids = Tensor::<i32, _>::zeros([n, k], dev).unwrap();
    let decode_scores = Tensor::<f32, _>::zeros([n, c], dev).unwrap();
    let decode_ids = Tensor::<i32, _>::zeros([n, k], dev).unwrap();
    let before = dev.config.pool_stats();
    let mut offset = 0;
    for count in [1, 2, 7, 9] {
        full.start
            .upload_from_host(&[start + offset as i32])
            .unwrap();
        let mut out = chunk_scores.narrow(0, offset, count).unwrap();
        Cuda::v4_indexer_scores(
            &s,
            &q.narrow(0, offset, count).unwrap(),
            &keys,
            &w.narrow(0, offset, count).unwrap(),
            &full.start,
            &mut out,
        )
        .unwrap();
        Cuda::v4_indexer_topk(
            &s,
            &out,
            &full.start,
            &mut full.work,
            &mut chunk_ids.narrow(0, offset, count).unwrap(),
        )
        .unwrap();
        offset += count;
    }
    assert_eq!(offset, n);
    for t in 0..n {
        full.start.upload_from_host(&[start + t as i32]).unwrap();
        let mut out = decode_scores.narrow(0, t, 1).unwrap();
        Cuda::v4_indexer_scores(
            &s,
            &q.narrow(0, t, 1).unwrap(),
            &keys,
            &w.narrow(0, t, 1).unwrap(),
            &full.start,
            &mut out,
        )
        .unwrap();
        Cuda::v4_indexer_topk(
            &s,
            &out,
            &full.start,
            &mut full.work,
            &mut decode_ids.narrow(0, t, 1).unwrap(),
        )
        .unwrap();
    }
    s.synchronize().unwrap();
    assert_eq!(dev.config.pool_stats(), before);
    assert_eq!(actual, chunk_scores.to_host_vec().unwrap());
    assert_eq!(actual, decode_scores.to_host_vec().unwrap());
    assert_eq!(
        full.ids.to_host_vec().unwrap(),
        chunk_ids.to_host_vec().unwrap()
    );
    assert_eq!(
        full.ids.to_host_vec().unwrap(),
        decode_ids.to_host_vec().unwrap()
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn scratch_and_output_guards_hold_and_scores_remain_immutable() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    for c in [13, 4097, 8193] {
        let (n, k) = (3, 37);
        let mut a = Select::new(&s, n, c, k, (c * 4 - 2) as i32);
        let words = Cuda::v4_indexer_topk_workspace_words(n, c, k).unwrap();
        let work = Tensor::from_host_slice(&vec![-987; words + 2], [words + 2], dev).unwrap();
        let ids = Tensor::from_host_slice(&vec![-987; n * k + 2], [n * k + 2], dev).unwrap();
        a.work = work.narrow(0, 1, words).unwrap();
        a.ids = ids
            .narrow(0, 1, n * k)
            .unwrap()
            .view_contiguous(Shape::from_slice(&[n, k]))
            .unwrap();
        let input = a.scores.to_host_vec().unwrap();
        a.run(&s);
        s.synchronize().unwrap();
        a.check();
        assert_eq!(a.scores.to_host_vec().unwrap(), input);
        let w = work.to_host_vec().unwrap();
        let i = ids.to_host_vec().unwrap();
        assert_eq!(
            (w[0], w[words + 1], i[0], i[n * k + 1]),
            (-987, -987, -987, -987)
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn invalid_positions_shapes_workspace_and_aliases_fail_closed() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    for c in [4, 2049] {
        for n in [1, 7] {
            let mut a = Select::new(&s, n, c, 512, 0);
            for bad in [-1, i32::MAX, (c * 4 + 3) as i32] {
                a.start.upload_from_host(&[bad]).unwrap();
                a.run(&s);
                s.synchronize().unwrap();
                assert_eq!(a.ids.to_host_vec().unwrap(), vec![-1; n * 512]);
            }
        }
    }
    for (n, c, k) in [
        (0, 4, 1),
        (1, 0, 1),
        (1, 4, 0),
        (1, 4, 513),
        (1, usize::MAX, 1),
        (i32::MAX as usize, 4096, 1),
    ] {
        assert!(Cuda::v4_indexer_topk_workspace_words(n, c, k).is_err());
    }
    assert_eq!(
        Cuda::v4_indexer_topk_workspace_words(7, 2048, 512).unwrap(),
        1
    );
    assert_eq!(
        Cuda::v4_indexer_topk_workspace_words(7, 2049, 512).unwrap(),
        4 * 7 * 2 * 512
    );
    let mut a = Select::new(&s, 2, 2049, 512, 0);
    let short_work = Tensor::<i32, _>::zeros([a.work.numel() - 1], dev).unwrap();
    assert!(
        Cuda::v4_indexer_topk(&s, &a.scores, &a.start, &mut short_work.clone(), &mut a.ids)
            .is_err()
    );
    let mut shaped = a
        .work
        .clone()
        .view_contiguous(Shape::from_slice(&[2, a.work.numel() / 2]))
        .unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &a.scores, &a.start, &mut shaped, &mut a.ids).is_err());
    let strided = Tensor::<f32, _>::zeros([2, 2050], dev)
        .unwrap()
        .narrow(1, 0, 2049)
        .unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &strided, &a.start, &mut a.work, &mut a.ids).is_err());
    let mut bad_ids = Tensor::<i32, _>::zeros([2, 513], dev).unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &a.scores, &a.start, &mut a.work, &mut bad_ids).is_err());
    let mut out_stride = Tensor::<i32, _>::zeros([2, 513], dev)
        .unwrap()
        .narrow(1, 0, 512)
        .unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &a.scores, &a.start, &mut a.work, &mut out_stride).is_err());
    let mut alias = a
        .work
        .narrow(0, 0, 1024)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[2, 512]))
        .unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &a.scores, &a.start, &mut a.work, &mut alias).is_err());
    let start_alias = a.work.narrow(0, 0, 1).unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &a.scores, &start_alias, &mut a.work, &mut a.ids).is_err());
    let wrong_start = Tensor::<i32, _>::zeros([2], dev).unwrap();
    assert!(Cuda::v4_indexer_topk(&s, &a.scores, &wrong_start, &mut a.work, &mut a.ids).is_err());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn score_buckets_reuse_large_key_pool_and_fail_closed_on_graph_overflow() {
    use infer_core::ports::fused_ops::v4_indexer_score_capacity;
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let (n, h, c, k) = (7, 64, 8193, 512);
    let query = Tensor::from_host_slice(
        &data(n * h * 128, 5)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
        [n, h, 128],
        dev,
    )
    .unwrap();
    let keys = Tensor::from_host_slice(
        &data(c * 128, 9)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
        [c, 128],
        dev,
    )
    .unwrap();
    let weights = Tensor::from_host_slice(&data(n * h, 7), [n, h], dev).unwrap();
    let key_address = keys.data_ptr();
    let mut full = Select::new(&s, n, c, k, 0);
    for start in [0usize, 121, 127, 8185, 8190, 32765] {
        let b = v4_indexer_score_capacity(start, n, c).unwrap();
        let mut small = Select::new(&s, n, b, k, start as i32);
        s.synchronize().unwrap();
        s.graph_capture_begin().unwrap();
        Cuda::v4_indexer_scores(&s, &query, &keys, &weights, &small.start, &mut small.scores)
            .unwrap();
        small.run(&s);
        s.graph_capture_end(4990).unwrap();
        let allocations = dev.config.pool_stats();
        // Grow within a bucket, deliberately cross it, then restart the request.
        for pos in [
            start as i32,
            (b * 4 + 3 - n) as i32,
            (b * 4 + 4 - n) as i32,
            -1,
            0,
        ] {
            small.start.upload_from_host(&[pos]).unwrap();
            s.graph_launch(4990).unwrap();
            s.synchronize().unwrap();
            let got = small.ids.to_host_vec().unwrap();
            let scores = small.scores.to_host_vec().unwrap();
            if pos < 0 || (pos as usize + n) / 4 > b {
                assert!(got.iter().all(|&id| id == -1));
                assert!(scores.iter().all(|v| v.is_nan()));
            } else {
                full.start.upload_from_host(&[pos]).unwrap();
                Cuda::v4_indexer_scores(&s, &query, &keys, &weights, &full.start, &mut full.scores)
                    .unwrap();
                full.run(&s);
                s.synchronize().unwrap();
                assert_eq!(got, full.ids.to_host_vec().unwrap());
                let all = full.scores.to_host_vec().unwrap();
                for t in 0..n {
                    assert_eq!(&scores[t * b..(t + 1) * b], &all[t * c..t * c + b]);
                }
            }
        }
        assert_eq!(dev.config.pool_stats(), allocations);
        assert_eq!(keys.data_ptr(), key_address);
        dev.config.invalidate_all_graphs();
    }
    let mut too_wide = Tensor::zeros([n, c + 1], dev).unwrap();
    assert!(
        Cuda::v4_indexer_scores(&s, &query, &keys, &weights, &full.start, &mut too_wide).is_err()
    );
}
