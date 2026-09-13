use super::*;
use crate::{CudaMemoryPlan, CudaScope};
use half::bf16;
use infer_core::exec::ExecScope;

const HD: usize = 128;
const QH: usize = 4;
const KH: usize = 2;
const Q_STRIDE: usize = QH * HD + 32;

struct Case {
    plan: BatchPlan,
    index: KvIndexTensors<Cuda>,
    q: Tensor<bf16, Cuda>,
    k: Tensor<bf16, Cuda>,
    v: Tensor<bf16, Cuda>,
    output: Tensor<bf16, Cuda>,
    host_q: Vec<bf16>,
    host_k: Vec<bf16>,
    host_v: Vec<bf16>,
    tables: Vec<i32>,
}

impl Case {
    fn new(device: &Cuda, q_lens: &[i32], kv_lens: &[i32], block_size: usize) -> Self {
        let max_blocks = 2048 / block_size;
        let num_blocks = max_blocks * q_lens.len() + 3;
        let rows = q_lens.iter().sum::<i32>() as usize;
        let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], device).unwrap();
        let (cu, reqs, tiles) = BatchPlan::plan_ragged_tiles(q_lens);
        // Reverse physical pages, with separate regions for different requests.
        let tables: Vec<_> = (0..max_blocks * q_lens.len())
            .map(|page| (num_blocks - 1 - page) as i32)
            .collect();
        let plan = BatchPlan {
            kind: BatchKind::Ragged,
            num_tokens: rows,
            batch: q_lens.len(),
            q_lens: q_lens.to_vec(),
            kv_lens: kv_lens.to_vec(),
            seq_positions: q_lens.iter().zip(kv_lens).map(|(q, kv)| kv - q).collect(),
            rope_positions: vec![],
            max_blocks_per_seq: max_blocks,
            block_size,
            total_q_tiles: reqs.len() as i32,
        };
        let index = KvIndexTensors {
            decode_rows: allocate(device, 8, max_blocks).unwrap(),
            block_tables: Tensor::from_host_slice(&tables, [q_lens.len(), max_blocks], device)
                .unwrap(),
            cu_q_lens: ints(&cu),
            kv_lens: ints(kv_lens),
            seq_lens_step: ints(q_lens),
            seq_positions: ints(&plan.seq_positions),
            rope_positions: ints(&[0]),
            block2req: ints(&reqs),
            block2tile: ints(&tiles),
            valid_q_tiles: ints(&[reqs.len() as i32]),
            valid_suffix_q_tiles: ints(&[reqs.len() as i32]),
        };
        let host_q: Vec<_> = (0..rows * Q_STRIDE)
            .map(|i| bf16::from_f32((i as f32 * 0.13).sin() * 0.7))
            .collect();
        let host_k: Vec<_> = (0..num_blocks * block_size * KH * HD)
            .map(|i| bf16::from_f32((i as f32 * 0.071).cos() * 0.7))
            .collect();
        let host_v: Vec<_> = (0..host_k.len())
            .map(|i| bf16::from_f32((i as f32 * 0.017).sin() * 2.0))
            .collect();
        Self {
            q: Tensor::from_host_slice(&host_q, [rows, Q_STRIDE], device)
                .unwrap()
                .narrow(1, 0, QH * HD)
                .unwrap(),
            k: Tensor::from_host_slice(&host_k, [num_blocks, block_size, KH * HD], device).unwrap(),
            v: Tensor::from_host_slice(&host_v, [num_blocks, block_size, KH * HD], device).unwrap(),
            output: Tensor::zeros([rows, QH * HD], device).unwrap(),
            plan,
            index,
            host_q,
            host_k,
            host_v,
            tables,
        }
    }

    fn run(&mut self, scope: &CudaScope) {
        prepare(scope.stream().0, &self.plan, &mut self.index).unwrap();
        assert!(
            try_attention(
                scope.stream().0,
                &self.q,
                &self.k,
                &self.v,
                &mut self.output,
                PagedAttentionPlan::from_v2(&self.plan, &self.index),
                QH,
                KH,
                HD,
                1.0 / (HD as f32).sqrt(),
            )
            .unwrap(),
            "test must execute cuDNN short query attention, without fallback"
        );
    }

    fn pool_offset(&self, req: usize, token: usize, head: usize) -> usize {
        let page =
            self.tables[req * self.plan.max_blocks_per_seq + token / self.plan.block_size] as usize;
        ((page * self.plan.block_size + token % self.plan.block_size) * KH + head) * HD
    }

    fn check(&self, scope: &CudaScope) {
        scope.synchronize().unwrap();
        let actual = self.output.to_host_vec().unwrap();
        let lengths = self
            .index
            .decode_rows
            .as_ref()
            .unwrap()
            .kv_lens
            .to_host_vec()
            .unwrap();
        let mut row = 0;
        for (req, (&q_len, &kv_len)) in self.plan.q_lens.iter().zip(&self.plan.kv_lens).enumerate()
        {
            for local in 0..q_len {
                let visible = (kv_len - q_len + local + 1) as usize;
                assert_eq!(lengths[row], visible as i32);
                for head in 0..QH {
                    let logits: Vec<_> = (0..visible)
                        .map(|token| {
                            let offset = self.pool_offset(req, token, head / (QH / KH));
                            (0..HD)
                                .map(|d| {
                                    self.host_q[row * Q_STRIDE + head * HD + d].to_f32()
                                        * self.host_k[offset + d].to_f32()
                                })
                                .sum::<f32>()
                                / (HD as f32).sqrt()
                        })
                        .collect();
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights: Vec<_> = logits.iter().map(|l| (l - max).exp()).collect();
                    let sum: f32 = weights.iter().sum();
                    for d in 0..HD {
                        let expected = weights
                            .iter()
                            .enumerate()
                            .map(|(token, w)| {
                                w * self.host_v[self.pool_offset(req, token, head / (QH / KH)) + d]
                                    .to_f32()
                            })
                            .sum::<f32>()
                            / sum;
                        let got = actual[(row * QH + head) * HD + d].to_f32();
                        assert!(
                            got.is_finite() && (got - expected).abs() < 0.015,
                            "q={q_len} kv={kv_len} block={} row={row} head={head} d={d}: {got} vs {expected}",
                            self.plan.block_size
                        );
                    }
                }
                row += 1;
            }
        }
    }
}

fn scope() -> CudaScope {
    CudaScope::new(
        Cuda::with_memory_plan(
            0,
            CudaMemoryPlan {
                kernel_workspace_bytes: 64 * 1024 * 1024,
                graph_arena_bytes: 8 * 1024 * 1024,
                pool_retain_bytes: 64 * 1024 * 1024,
            },
        )
        .unwrap(),
    )
}

#[test]
#[ignore = "requires a visible CUDA GPU and cuDNN"]
fn short_queries_match_causal_reference_and_ignore_future_kv() {
    let scope = scope();
    let _guard = scope.enter();
    for block_size in [1, 16] {
        for q in [2, 3, 4, 8] {
            let mut case = Case::new(scope.device(), &[q], &[q], block_size);
            for kv in [q, 31, 32, 33, 127, 128, 129, 513, 2048] {
                case.plan.kv_lens[0] = kv;
                case.index.kv_lens.upload_from_host(&[kv]).unwrap();
                case.run(&scope);
                case.check(&scope);
            }
        }
    }
    let mut case = Case::new(scope.device(), &[4], &[33], 16);
    case.run(&scope);
    scope.synchronize().unwrap();
    let before = case.output.to_host_vec().unwrap();
    let offset = case.pool_offset(0, 32, 0);
    case.host_v[offset..offset + KH * HD].fill(bf16::from_f32(64.0));
    case.v.upload_from_host(&case.host_v).unwrap();
    case.run(&scope);
    case.check(&scope);
    let after = case.output.to_host_vec().unwrap();
    assert_eq!(&before[..3 * QH * HD], &after[..3 * QH * HD]);
    assert_ne!(&before[3 * QH * HD..], &after[3 * QH * HD..]);
    // Reusing the same buffers for a long query must clear the short-row view.
    case.plan.num_tokens = 9;
    case.plan.q_lens[0] = 9;
    prepare(scope.stream().0, &case.plan, &mut case.index).unwrap();
    assert_eq!(case.index.decode_rows.as_ref().unwrap().num_rows, 0);
}

#[test]
#[ignore = "requires a visible CUDA GPU and cuDNN"]
fn short_queries_graph_replay_reads_new_lengths_and_request_mapping() {
    let scope = scope();
    let _guard = scope.enter();
    let mut case = Case::new(scope.device(), &[2, 3, 1], &[31, 65, 129], 16);
    case.run(&scope); // Populate the cuDNN plan cache before capture.
    case.check(&scope);
    scope.graph_capture_begin().unwrap();
    case.run(&scope);
    scope.graph_capture_end(901).unwrap();
    for (queries, lengths) in [([1, 1, 4], [65, 33, 193]), ([3, 2, 1], [128, 129, 513])] {
        case.plan.q_lens = queries.to_vec();
        case.plan.kv_lens = lengths.to_vec();
        let (cu, _, _) = BatchPlan::plan_ragged_tiles(&queries);
        case.index.cu_q_lens.upload_from_host(&cu).unwrap();
        case.index.kv_lens.upload_from_host(&lengths).unwrap();
        scope.graph_launch(901).unwrap();
        case.check(&scope);
    }
}
