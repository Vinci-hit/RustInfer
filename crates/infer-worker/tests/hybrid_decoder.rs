//! Numerical tests of the mixed decoder, independent of Runtime's future slot
//! allocator. Every fixture explicitly owns the KV blocks and recurrent slots.

use std::collections::HashMap;

use infer_worker::application::runtime::Runtime;
use infer_worker::application::sampler_stack::GreedySampler;
use infer_worker::components::{
    Attention, DecoderBlock, DenseFfn, Embed, FullAttention, GatedDeltaNet, GdnWeights, Linear,
    LmHead, RmsNorm,
};
use infer_worker::domain::cache::{
    CacheLayout, LayerCacheId, LayerCacheSpec, LinearBatch, LinearDims, LinearLayerState,
    ModelCacheView,
};
use infer_worker::domain::component::{Hidden, LayerRange};
use infer_worker::domain::exec::{HostScope, StepCtx};
use infer_worker::domain::forward_scratch::ForwardScratch;
use infer_worker::domain::gdn_scratch::GdnScratch;
use infer_worker::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use infer_worker::domain::model::{DecoderModel, ModelDims};
use infer_worker::domain::plan::{BatchKind, BatchPlan, MaskMode, StepOutput, StepRequest};
use infer_worker::domain::ports::OpResult;
use infer_worker::domain::tensor::Tensor;
use infer_worker::infrastructure::cpu::Cpu;
use infer_worker::models::decoder::Decoder;

const DIM: usize = 8;
const INTER: usize = 16;
const VOCAB: usize = 16;
const HEAD_DIM: usize = 4;
const KV_DIM: usize = 4;
const MAX_SEQ: usize = 16;
const SLOTS: usize = 4;
const LINEAR: LinearDims = LinearDims {
    num_key_heads: 2,
    num_value_heads: 4,
    key_head_dim: 2,
    value_head_dim: 2,
    conv_kernel_dim: 4,
};

fn weights(rows: usize, cols: usize, seed: f32) -> Tensor<f32, Cpu> {
    Tensor::from_host_slice(
        &(0..rows * cols)
            .map(|i| ((i as f32 * 0.37 + seed).sin() + 0.2) * 0.22)
            .collect::<Vec<_>>(),
        [rows, cols],
        &Cpu,
    )
    .unwrap()
}

fn norm(dim: usize) -> RmsNorm<f32, Cpu> {
    RmsNorm {
        zero_centered: false,
        weight: Tensor::from_host_slice(&vec![1.0; dim], [dim], &Cpu).unwrap(),
        eps: 1e-6,
    }
}

fn linear(rows: usize, cols: usize, seed: f32) -> Linear<f32, Cpu> {
    Linear::new(weights(rows, cols, seed), None)
}

fn gdn_weights(seed: f32) -> GdnWeights<f32, Cpu> {
    GdnWeights {
        input_layernorm: norm(DIM),
        in_proj_qkv: linear(LINEAR.conv_dim(), DIM, seed),
        in_proj_a: linear(LINEAR.num_value_heads, DIM, seed + 0.4),
        in_proj_b: linear(LINEAR.num_value_heads, DIM, seed + 0.8),
        in_proj_z: linear(LINEAR.value_dim(), DIM, seed + 1.2),
        conv1d: weights(LINEAR.conv_dim(), 4, seed + 1.5)
            .view_contiguous([LINEAR.conv_dim(), 1, 4].into())
            .unwrap(),
        a_log: Tensor::from_host_slice(&[-0.2, 0.1, -0.4, 0.3], [4], &Cpu).unwrap(),
        dt_bias: Tensor::from_host_slice(&[0.1, -0.1, 0.2, 0.3], [4], &Cpu).unwrap(),
        norm_weight: Tensor::from_host_slice(&[0.8, 1.2], [2], &Cpu).unwrap(),
        norm_eps: 1e-6,
        out_proj: linear(DIM, LINEAR.value_dim(), seed + 1.8),
    }
}

fn gdn(seed: f32) -> GatedDeltaNet<f32, Cpu> {
    GatedDeltaNet::new(gdn_weights(seed), LINEAR).unwrap()
}

fn model(shared_scratch: bool) -> Decoder<f32, Cpu> {
    let mut blocks = Vec::new();
    for i in 0..8 {
        let seed = 0.3 + i as f32 * 0.31;
        let attention = if i % 4 == 3 {
            Attention::Full(FullAttention {
                input_layernorm: norm(DIM),
                qkv_proj: linear(DIM + 2 * KV_DIM, DIM, seed),
                o_proj: linear(DIM, DIM, seed + 0.4),
                q_norm: Some(norm(HEAD_DIM)),
                k_norm: Some(norm(HEAD_DIM)),
                sin: Tensor::from_host_slice(
                    &(0..MAX_SEQ * 2)
                        .map(|i| (i as f32 * 0.13).sin())
                        .collect::<Vec<_>>(),
                    [MAX_SEQ, 2],
                    &Cpu,
                )
                .unwrap(),
                cos: Tensor::from_host_slice(
                    &(0..MAX_SEQ * 2)
                        .map(|i| (i as f32 * 0.13).cos())
                        .collect::<Vec<_>>(),
                    [MAX_SEQ, 2],
                    &Cpu,
                )
                .unwrap(),
                head_num: 2,
                kv_head_num: 1,
                head_dim: HEAD_DIM,
                rotary_dim: HEAD_DIM,
                attn_output_gate: false,
                scale: 0.5,
                scratch: None,
            })
        } else {
            Attention::Linear(gdn(seed))
        };
        blocks.push(DecoderBlock {
            attention,
            ffn: DenseFfn {
                post_attention_layernorm: norm(DIM),
                gate_up_proj: linear(2 * INTER, DIM, seed + 2.1),
                down_proj: linear(DIM, INTER, seed + 2.4),
                scratch: None,
            },
        });
    }
    let dims = ModelDims {
        dim: DIM,
        q_dim: DIM,
        kv_dim: KV_DIM,
        qkv_dim: DIM + 2 * KV_DIM,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_num: 2,
        head_dim: HEAD_DIM,
        kv_head_num: 1,
        num_layers: blocks.len(),
        ..ModelDims::default()
    };
    let mut model = Decoder::new(
        Embed::new(weights(VOCAB, DIM, 0.1)),
        blocks,
        norm(DIM),
        LmHead {
            proj: linear(VOCAB, DIM, 1.0),
        },
        dims,
    )
    .unwrap();
    if shared_scratch {
        model.install_scratch(ForwardScratch::new(&Cpu, dims, 32, SLOTS).unwrap());
        model
            .install_gdn_scratch(GdnScratch::new(&Cpu, DIM, LINEAR, 32).unwrap())
            .unwrap();
    }
    model
}

fn ints(values: &[i32]) -> Tensor<i32, Cpu> {
    Tensor::from_host_slice(values, [values.len()], &Cpu).unwrap()
}

struct Fixture {
    kv: PagedKvPool<f32, Cpu>,
    linear: Vec<LinearLayerState<f32, Cpu>>,
    positions: [usize; SLOTS],
}

struct Step {
    plan: BatchPlan,
    index: KvIndexTensors<Cpu>,
    linear: LinearBatch<Cpu>,
    ids: Tensor<i32, Cpu>,
}

impl Fixture {
    fn new(model: &Decoder<f32, Cpu>) -> Self {
        let layout = model.cache_layout();
        Self {
            kv: PagedKvPool {
                layers: (0..layout.num_full_layers())
                    .map(|_| PagedKvLayer {
                        k: Tensor::zeros([SLOTS * MAX_SEQ, 1, KV_DIM], &Cpu).unwrap(),
                        v: Tensor::zeros([SLOTS * MAX_SEQ, 1, KV_DIM], &Cpu).unwrap(),
                    })
                    .collect(),
                num_blocks: SLOTS * MAX_SEQ,
                block_size: 1,
                kv_dim: KV_DIM,
                quant: KvQuantTier::None,
                seq_kv_len: HashMap::new(),
            },
            linear: layout
                .linear_dims()
                .iter()
                .map(|&dims| LinearLayerState::new(dims, SLOTS, &Cpu).unwrap())
                .collect(),
            positions: [0; SLOTS],
        }
    }

    fn step(&self, sequences: &[(usize, &[i32])]) -> Step {
        let slots: Vec<i32> = sequences.iter().map(|(slot, _)| *slot as i32).collect();
        let q_lens: Vec<i32> = sequences.iter().map(|(_, ids)| ids.len() as i32).collect();
        let positions: Vec<i32> = sequences
            .iter()
            .map(|(slot, _)| self.positions[*slot] as i32)
            .collect();
        let kv_lens: Vec<i32> = positions.iter().zip(&q_lens).map(|(p, q)| p + q).collect();
        assert!(kv_lens.iter().all(|&len| len as usize <= MAX_SEQ));
        let rope_positions: Vec<i32> = positions
            .iter()
            .zip(&q_lens)
            .flat_map(|(&p, &q)| p..p + q)
            .collect();
        let block_tables: Vec<i32> = sequences
            .iter()
            .flat_map(|(slot, _)| (slot * MAX_SEQ..(slot + 1) * MAX_SEQ).map(|b| b as i32))
            .collect();
        let (cu, block2req, block2tile) = BatchPlan::plan_ragged_tiles(&q_lens);
        let ids: Vec<i32> = sequences
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        Step {
            plan: BatchPlan {
                kind: if q_lens.iter().all(|&q| q == 1) {
                    BatchKind::DecodeOnly
                } else {
                    BatchKind::Ragged
                },
                num_tokens: ids.len(),
                batch: slots.len(),
                q_lens: q_lens.clone(),
                kv_lens: kv_lens.clone(),
                seq_positions: positions.clone(),
                rope_positions: rope_positions.clone(),
                max_blocks_per_seq: MAX_SEQ,
                block_size: 1,
                total_q_tiles: block2req.len() as i32,
            },
            index: KvIndexTensors {
                block_tables: Tensor::from_host_slice(&block_tables, [slots.len(), MAX_SEQ], &Cpu)
                    .unwrap(),
                cu_q_lens: ints(&cu),
                kv_lens: ints(&kv_lens),
                seq_positions: ints(&positions),
                seq_lens_step: ints(&q_lens),
                rope_positions: ints(&rope_positions),
                block2req: ints(&block2req),
                block2tile: ints(&block2tile),
                valid_q_tiles: ints(&[block2req.len() as i32]),
                valid_suffix_q_tiles: ints(&[block2req.len() as i32]),
            },
            linear: LinearBatch::new(&slots, &q_lens, SLOTS, &Cpu).unwrap(),
            ids: ints(&ids),
        }
    }

    fn run(
        &mut self,
        model: &Decoder<f32, Cpu>,
        sequences: &[(usize, &[i32])],
        ranges: &[LayerRange],
    ) -> Vec<Vec<f32>> {
        let step = self.step(sequences);
        let scope = HostScope::new(Cpu);
        let ctx = StepCtx::new(&scope, &step.plan);
        let mut hidden = Hidden {
            stream: Tensor::zeros([step.plan.num_tokens, DIM], &Cpu).unwrap(),
            pending: None,
        };
        let mut cache =
            ModelCacheView::hybrid(&mut self.kv, &step.index, &mut self.linear, &step.linear);
        model.embed(&step.ids, &mut hidden, &ctx).unwrap();
        for &range in ranges {
            model
                .decode_layers(range, &mut hidden, &mut cache, &ctx)
                .unwrap();
        }
        assert!(hidden.pending.is_none());
        let output = hidden.stream.to_host_vec().unwrap();
        let mut cursor = 0;
        sequences
            .iter()
            .map(|(slot, ids)| {
                let end = cursor + ids.len() * DIM;
                let values = output[cursor..end].to_vec();
                cursor = end;
                self.positions[*slot] += ids.len();
                values
            })
            .collect()
    }

    fn snapshot(&self) -> Vec<Vec<f32>> {
        self.linear
            .iter()
            .flat_map(|state| {
                [
                    state.conv().to_host_vec().unwrap(),
                    state.ssm().to_host_vec().unwrap(),
                ]
            })
            .chain(self.kv.layers.iter().flat_map(|layer| {
                [
                    layer.k.to_host_vec().unwrap(),
                    layer.v.to_host_vec().unwrap(),
                ]
            }))
            .collect()
    }
}

const ALL: [LayerRange; 1] = [LayerRange { start: 0, end: 8 }];

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5 + b.abs() * 2e-5,
            "index {i}: actual={a} expected={b}"
        );
    }
}

fn same_state(a: &Fixture, b: &Fixture) {
    for (a, b) in a.snapshot().iter().zip(b.snapshot()) {
        close(a, &b);
    }
}

#[test]
fn ragged_chunked_and_serial_execution_preserve_request_slots() {
    let model = model(false);
    let mut full = Fixture::new(&model);
    let mut chunked = Fixture::new(&model);
    let mut serial = Fixture::new(&model);
    let reference = full.run(&model, &[(3, &[1, 2, 3, 4, 5]), (1, &[6, 7, 8])], &ALL);
    let c0 = chunked.run(&model, &[(1, &[6]), (3, &[1, 2])], &ALL);
    let c1 = chunked.run(&model, &[(3, &[3]), (1, &[7, 8])], &ALL);
    let c2 = chunked.run(&model, &[(3, &[4, 5])], &ALL);
    close(
        &[c0[1].as_slice(), c1[0].as_slice(), c2[0].as_slice()].concat(),
        &reference[0],
    );
    close(
        &[c0[0].as_slice(), c1[1].as_slice()].concat(),
        &reference[1],
    );
    close(
        &serial.run(&model, &[(3, &[1, 2, 3, 4, 5])], &ALL)[0],
        &reference[0],
    );
    close(
        &serial.run(&model, &[(1, &[6, 7, 8])], &ALL)[0],
        &reference[1],
    );
    same_state(&full, &chunked);
    same_state(&full, &serial);
    // Slot 0/2 were never selected by either ordering.
    for state in &chunked.linear {
        for slot in [0, 2] {
            assert!(
                state
                    .conv()
                    .narrow(0, slot, 1)
                    .unwrap()
                    .to_host_vec()
                    .unwrap()
                    .iter()
                    .all(|&v| v == 0.0)
            );
            assert!(
                state
                    .ssm()
                    .narrow(0, slot, 1)
                    .unwrap()
                    .to_host_vec()
                    .unwrap()
                    .iter()
                    .all(|&v| v == 0.0)
            );
        }
    }
    let expected = full.run(&model, &[(1, &[9]), (3, &[10])], &ALL);
    let actual = chunked.run(&model, &[(3, &[10]), (1, &[9])], &ALL);
    close(&actual[1], &expected[0]);
    close(&actual[0], &expected[1]);
    same_state(&full, &chunked);
}

#[test]
fn layer_ranges_use_global_mapping_into_compact_caches() {
    let model = model(false);
    assert_eq!(
        model.cache_layout().layers(),
        &[
            LayerCacheId::Linear(0),
            LayerCacheId::Linear(1),
            LayerCacheId::Linear(2),
            LayerCacheId::Full(0),
            LayerCacheId::Linear(3),
            LayerCacheId::Linear(4),
            LayerCacheId::Linear(5),
            LayerCacheId::Full(1),
        ]
    );
    let mut all = Fixture::new(&model);
    let mut split = Fixture::new(&model);
    let ranges = [
        LayerRange { start: 0, end: 2 },
        LayerRange { start: 2, end: 5 },
        LayerRange { start: 5, end: 8 },
    ];
    for sequences in [
        &[(2, &[1, 5, 7][..]), (0, &[2, 8][..])][..],
        &[(0, &[6][..]), (2, &[3][..])][..],
    ] {
        let expected = all.run(&model, sequences, &ALL);
        let actual = split.run(&model, sequences, &ranges);
        for (a, b) in actual.iter().zip(expected) {
            close(a, &b);
        }
        same_state(&all, &split);
    }
}

#[test]
fn shared_scratch_preserves_projections_and_deferred_residuals() {
    let plain = model(false);
    let shared = model(true);
    let mut reference = Fixture::new(&plain);
    let mut actual = Fixture::new(&shared);
    for sequences in [
        &[(3, &[2, 4, 5][..]), (0, &[6, 7][..])][..],
        &[(0, &[8][..]), (3, &[9][..])][..],
        &[(3, &[1, 2, 3][..])][..],
    ] {
        let expected = reference.run(&plain, sequences, &ALL);
        let values = actual.run(&shared, sequences, &ALL);
        for (a, b) in values.iter().zip(expected) {
            close(a, &b);
        }
        same_state(&actual, &reference);
    }
}

#[test]
fn invalid_metadata_and_later_layer_geometry_fail_before_state_changes() {
    for (slots, lengths) in [
        (&[1, 1][..], &[1, 2][..]),
        (&[-1][..], &[1][..]),
        (&[4][..], &[1][..]),
        (&[1][..], &[-1][..]),
        (&[1][..], &[0][..]),
        (&[1, 2][..], &[i32::MAX, 1][..]),
        (&[1][..], &[1, 2][..]),
    ] {
        assert!(LinearBatch::new(slots, lengths, SLOTS, &Cpu).is_err());
    }

    let model = model(true);
    for fault in 0..20 {
        let mut fixture = Fixture::new(&model);
        let mut step = fixture.step(&[(1, &[1, 2])]);
        let mut range = ALL[0];
        match fault {
            0 => step.plan.q_lens = vec![1, 1],
            1 => {
                step.plan.kind = BatchKind::Spec {
                    mask: MaskMode::Tree,
                    mask_handle: None,
                }
            }
            2 => {
                *fixture.linear.last_mut().unwrap() = LinearLayerState::new(
                    LinearDims {
                        value_head_dim: 3,
                        ..LINEAR
                    },
                    SLOTS,
                    &Cpu,
                )
                .unwrap()
            }
            3 => {
                fixture.linear.pop();
            }
            4 => {
                range.end = 9;
            }
            5 => {
                range.start = 6;
                range.end = 2;
            }
            6 => step.plan.kv_lens.clear(),
            7 => step.plan.seq_positions.clear(),
            8 => step.plan.max_blocks_per_seq = 0,
            9 => step.index.cu_q_lens = ints(&[]),
            10 => step.index.block_tables = ints(&[]),
            11 => step.index.seq_positions = ints(&[]),
            12 => step.index.kv_lens = ints(&[]),
            13 => step.index.seq_lens_step = ints(&[]),
            14 => step.index.rope_positions = ints(&[0]),
            15 => step.index.block2req = ints(&[]),
            16 => step.plan.kv_lens[0] = MAX_SEQ as i32 + 1,
            17 => step.plan.kind = BatchKind::DecodeOnly,
            18 => step.plan.total_q_tiles = 0,
            19 => step.plan.seq_positions[0] = -1,
            _ => unreachable!(),
        }
        let before = fixture.snapshot();
        let mut hidden = Hidden {
            stream: Tensor::from_host_slice(&[0.25; 2 * DIM], [2, DIM], &Cpu).unwrap(),
            pending: Some(Tensor::from_host_slice(&[0.125; 2 * DIM], [2, DIM], &Cpu).unwrap()),
        };
        let scope = HostScope::new(Cpu);
        let ctx = StepCtx::new(&scope, &step.plan);
        let mut cache = ModelCacheView::hybrid(
            &mut fixture.kv,
            &step.index,
            &mut fixture.linear,
            &step.linear,
        );
        assert!(
            model
                .decode_layers(range, &mut hidden, &mut cache, &ctx)
                .is_err(),
            "fault {fault}"
        );
        assert_eq!(hidden.stream.to_host_vec().unwrap(), vec![0.25; 2 * DIM]);
        assert_eq!(
            hidden.pending.unwrap().to_host_vec().unwrap(),
            vec![0.125; 2 * DIM]
        );
        assert_eq!(
            fixture.snapshot(),
            before,
            "fault {fault} changed persistent state"
        );
    }
}

#[test]
fn gdn_rejects_unsupported_kernel_dimensions_during_construction() {
    // Dimension validation must precede even weight geometry validation, so a
    // model can never reach an in-place convolution with this unsupported size.
    let result = GatedDeltaNet::new(
        gdn_weights(0.1),
        LinearDims {
            key_head_dim: 1025,
            ..LINEAR
        },
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("unsupported GDN dimensions accepted"),
    };
    assert!(error.to_string().contains("exceeds supported limit 1024"));
}

#[test]
fn full_cache_accepts_bucket_plan_with_device_owned_lengths() {
    let model = model(false);
    let mut fixture = Fixture::new(&model);
    let mut step = fixture.step(&[(1, &[1, 2, 3, 4]), (3, &[5])]);
    // Full-only mixed graphs carry placeholder host lengths. The live sequence
    // lengths and cumulative token offsets are already in the device indices.
    step.plan.q_lens = vec![2, 0];
    step.plan.kv_lens = vec![0, 0];
    let layout = CacheLayout::new([LayerCacheSpec::Full { kv_dim: KV_DIM }; 2]).unwrap();
    ModelCacheView::full(&mut fixture.kv, &step.index)
        .validate(&layout, LayerRange::all(2), &step.plan)
        .unwrap();
}

fn hybrid_runtime(capacity: usize) -> Runtime<f32, Cpu, Decoder<f32, Cpu>> {
    Runtime::new(
        model(false),
        HostScope::new(Cpu),
        Box::new(GreedySampler),
        SLOTS * MAX_SEQ,
        1,
        MAX_SEQ,
        MAX_SEQ,
        32,
        capacity,
        vec![1, 2, 4],
    )
    .unwrap()
}

#[test]
fn hybrid_verification_restores_every_rejected_prefix_and_continues_greedy() {
    // Sweep K=0 and rejection at every possible draft position, including
    // full acceptance. Continuation checks both full KV and all recurrent layers.
    for k in 0..=3 {
        for accepted in 0..=k {
            let mut oracle = hybrid_runtime(1);
            let first = oracle
                .step(&request(&[(10, 0, 0, &[1, 2, 3])]))
                .unwrap()
                .tokens[0][0]
                .token_id;
            let mut tape = vec![first];
            for i in 0..k + 3 {
                let next = oracle
                    .step(&request(&[(10, 0, 3 + i, &[tape[i]])]))
                    .unwrap()
                    .tokens[0][0]
                    .token_id;
                tape.push(next);
            }
            let mut actual = hybrid_runtime(1);
            use infer_worker::application::execution::{ExecutionMetrics, Phase};
            actual.execution_metrics = ExecutionMetrics::new("test-target", 100);
            let mut reference = hybrid_runtime(1);
            let prefill = request(&[(10, 0, 0, &[1, 2, 3])]);
            actual.step(&prefill).unwrap();
            reference.step(&prefill).unwrap();
            let mut inputs = tape[..k + 1].to_vec();
            if accepted < k {
                inputs[accepted + 1] = (inputs[accepted + 1] + 1) % VOCAB as i32;
            }
            let mut verify = request(&[(10, 0, 3, &inputs)]);
            verify.draft_tokens = vec![inputs[1..].to_vec()];
            let result = actual.step_with_hidden(&verify).unwrap();
            assert_eq!(result.output.accepted_drafts, Some(vec![accepted as u32]));
            assert_eq!(actual.execution_metrics.snapshot(Phase::Verify).calls, 1);
            assert_eq!(actual.execution_metrics.snapshot(Phase::Snapshot).calls, 1);
            assert_eq!(
                actual.execution_metrics.snapshot(Phase::Restore).calls,
                u64::from(accepted < k)
            );
            assert_eq!(
                actual.execution_metrics.snapshot(Phase::Replay).tokens,
                if accepted < k {
                    (accepted + 1) as u64
                } else {
                    0
                }
            );
            assert_eq!(result.output.materialized_tokens, vec![accepted as u32 + 1]);
            assert_eq!(
                result.output.tokens[0]
                    .iter()
                    .map(|t| t.token_id)
                    .collect::<Vec<_>>(),
                tape[1..accepted + 2]
            );
            let expected = reference
                .step_with_hidden(&request(&[(10, 0, 3, &tape[..accepted + 1])]))
                .unwrap();
            close(
                &result.normalized_hidden.to_host_vec().unwrap(),
                &expected.normalized_hidden.to_host_vec().unwrap(),
            );
            let saved = result.normalized_hidden.to_host_vec().unwrap();
            for (i, &token) in tape.iter().enumerate().skip(accepted + 1).take(2) {
                let req = request(&[(10, 0, 3 + i, &[token])]);
                let a = actual.step(&req).unwrap();
                let b = reference.step(&req).unwrap();
                assert_runtime_outputs_match(&actual, &reference, &a, &b, 1);
            }
            assert_eq!(result.normalized_hidden.to_host_vec().unwrap(), saved);
        }
    }
}

#[test]
fn hybrid_verification_eos_and_reordered_ragged_requests_commit_only_retained_history() {
    let mut actual = hybrid_runtime(2);
    let mut reference = hybrid_runtime(2);
    let prefill = request(&[(10, 0, 0, &[1, 2]), (20, 1, 0, &[3])]);
    actual.step(&prefill).unwrap();
    reference.step(&prefill).unwrap();
    // The first sequence has K=0; the other has a longer speculative suffix.
    let mut req = request(&[(20, 1, 1, &[4]), (10, 0, 2, &[5, 6, 7])]);
    req.draft_tokens = vec![vec![], vec![6, 7]];
    // Make every possible first output an EOS, testing truncation independently
    // of this synthetic model's token predictions.
    req.stop.eos_ids = (0..VOCAB as i32).collect();
    req.stop.ignore_eos = vec![false, false];
    let out = actual.step_with_hidden(&req).unwrap();
    assert_eq!(out.output.materialized_tokens, [1, 1]);
    assert_eq!(out.output.finished, [true, true]);
    let retained = request(&[(20, 1, 1, &[4]), (10, 0, 2, &[5])]);
    let expected = reference.step_with_hidden(&retained).unwrap();
    close(
        &out.normalized_hidden.to_host_vec().unwrap(),
        &expected.normalized_hidden.to_host_vec().unwrap(),
    );
    let next = request(&[(10, 0, 3, &[8]), (20, 1, 2, &[9])]);
    let a = actual.step(&next).unwrap();
    let b = reference.step(&next).unwrap();
    assert_runtime_outputs_match(&actual, &reference, &a, &b, 2);
}

#[test]
fn failed_recurrent_replay_invalidates_the_request_until_reprefill() {
    use infer_worker::domain::model::{Logits, SampleRows};
    use infer_worker::domain::ports::OpError;
    use std::cell::Cell;
    struct FailReplay {
        decoder: Decoder<f32, Cpu>,
        calls: Cell<usize>,
        fail_at: Cell<usize>,
    }
    impl DecoderModel<f32, Cpu> for FailReplay {
        fn dims(&self) -> ModelDims {
            self.decoder.dims()
        }
        fn cache_layout(&self) -> &CacheLayout {
            self.decoder.cache_layout()
        }
        fn stages(&self) -> &[infer_worker::domain::component::StageKind] {
            self.decoder.stages()
        }
        fn install_scratch(&mut self, s: std::rc::Rc<ForwardScratch<f32, Cpu>>) {
            self.decoder.install_scratch(s);
        }
        fn install_gdn_scratch(&mut self, s: std::rc::Rc<GdnScratch<f32, Cpu>>) -> OpResult<()> {
            self.decoder.install_gdn_scratch(s)
        }
        fn embed(
            &self,
            ids: &Tensor<i32, Cpu>,
            h: &mut Hidden<f32, Cpu>,
            ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<()> {
            self.decoder.embed(ids, h, ctx)
        }
        fn decode_layers(
            &self,
            range: LayerRange,
            h: &mut Hidden<f32, Cpu>,
            cache: &mut ModelCacheView<'_, f32, Cpu>,
            ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<()> {
            self.calls.set(self.calls.get() + 1);
            self.decoder.decode_layers(range, h, cache, ctx)?;
            if self.calls.get() == self.fail_at.get() {
                return Err(OpError::Kernel("injected replay failure".into()));
            }
            Ok(())
        }
        fn finalize(
            &self,
            h: &Hidden<f32, Cpu>,
            rows: SampleRows<'_>,
            ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<Logits<f32, Cpu>> {
            self.decoder.finalize(h, rows, ctx)
        }
    }
    let model = FailReplay {
        decoder: model(false),
        calls: Cell::new(0),
        fail_at: Cell::new(3),
    };
    let mut runtime = Runtime::new(
        model,
        HostScope::new(Cpu),
        Box::new(GreedySampler),
        SLOTS * MAX_SEQ,
        1,
        MAX_SEQ,
        MAX_SEQ,
        32,
        2,
        vec![],
    )
    .unwrap();
    let mut reference = hybrid_runtime(2);
    let prefill = request(&[(10, 0, 0, &[1, 2]), (20, 1, 0, &[3])]);
    runtime.step(&prefill).unwrap();
    reference.step(&prefill).unwrap();
    let mut req = request(&[(10, 0, 2, &[4, 5, 6])]);
    req.draft_tokens = vec![vec![5, 6]];
    req.stop.eos_ids = (0..VOCAB as i32).collect();
    req.stop.ignore_eos = vec![false];
    assert!(
        runtime
            .step(&req)
            .unwrap_err()
            .to_string()
            .contains("injected replay failure")
    );
    assert!(runtime.step(&request(&[(10, 0, 3, &[7])])).is_err());
    let next = request(&[(20, 1, 1, &[8])]);
    let a = runtime.step(&next).unwrap();
    let b = reference.step(&next).unwrap();
    assert_eq!(a.tokens[0][0].token_id, b.tokens[0][0].token_id);
    close(
        &runtime.hidden.stream.to_host_vec().unwrap()[..DIM],
        &reference.hidden.stream.to_host_vec().unwrap()[..DIM],
    );
    // Same id can recover from zero after the failed transaction.
    reference.release_sequence(10);
    let replay = request(&[(10, 0, 0, &[1, 2, 4])]);
    let a = runtime.step(&replay).unwrap();
    let b = reference.step(&replay).unwrap();
    assert_eq!(a.tokens[0][0].token_id, b.tokens[0][0].token_id);
    close(
        &runtime.hidden.stream.to_host_vec().unwrap()[..3 * DIM],
        &reference.hidden.stream.to_host_vec().unwrap()[..3 * DIM],
    );
}

fn request(seqs: &[(u64, usize, usize, &[i32])]) -> infer_worker::domain::plan::StepRequest {
    use infer_worker::domain::plan::{SeqStep, StepRequest, StopCriteria};
    StepRequest {
        seqs: seqs
            .iter()
            .map(|&(id, slot, start, ids)| SeqStep {
                sequence_id: id,
                input_ids: ids.to_vec(),
                positions: (start as i32..(start + ids.len()) as i32).collect(),
                kv_write_start: start as i32,
                kv_len_after: (start + ids.len()) as i32,
                block_table: (slot * MAX_SEQ..(slot + 1) * MAX_SEQ)
                    .map(|b| b as u32)
                    .collect(),
            })
            .collect(),
        sampling: vec![Default::default(); seqs.len()],
        stop: StopCriteria {
            eos_ids: vec![],
            generated_counts: vec![0; seqs.len()],
            max_tokens: vec![100; seqs.len()],
            ignore_eos: vec![true; seqs.len()],
        },
        draft_tokens: vec![],
    }
}

fn assert_runtime_outputs_match(
    actual: &Runtime<f32, Cpu, Decoder<f32, Cpu>>,
    expected: &Runtime<f32, Cpu, Decoder<f32, Cpu>>,
    actual_out: &StepOutput,
    expected_out: &StepOutput,
    input_rows: usize,
) {
    let token_ids = |out: &StepOutput| {
        out.tokens
            .iter()
            .map(|row| row.iter().map(|token| token.token_id).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(token_ids(actual_out), token_ids(expected_out));
    assert_eq!(actual_out.finished, expected_out.finished);
    close(
        &actual.hidden.stream.to_host_vec().unwrap()[..input_rows * DIM],
        &expected.hidden.stream.to_host_vec().unwrap()[..input_rows * DIM],
    );
    let rows = actual_out.tokens.len();
    close(
        &actual
            .model
            .scratch
            .as_ref()
            .unwrap()
            .logits(rows)
            .to_host_vec()
            .unwrap(),
        &expected
            .model
            .scratch
            .as_ref()
            .unwrap()
            .logits(rows)
            .to_host_vec()
            .unwrap(),
    );
}

fn assert_finalize_failure_invalidates_history(
    new_tokens: &[i32],
    fail: impl FnOnce(&mut Runtime<f32, Cpu, Decoder<f32, Cpu>>, &StepRequest) -> OpResult<()>,
) {
    let mut runtime = hybrid_runtime(2);
    let mut reference = hybrid_runtime(2);
    let prefill = request(&[(10, 0, 0, &[1, 2])]);
    runtime.step(&prefill).unwrap();
    reference.step(&prefill).unwrap();
    let failed = request(&[(10, 0, 2, &[3]), (20, 1, 0, new_tokens)]);
    reference.step(&failed).unwrap();

    // Both the existing request and a newly assigned slot reach all decoder
    // layers before the lm_head's incompatible input width rejects finalize.
    let original = std::mem::replace(&mut runtime.model.lm_head.proj, linear(VOCAB, DIM + 1, 1.0));
    assert!(fail(&mut runtime, &failed).is_err());
    close(
        &runtime.hidden.stream.to_host_vec().unwrap()[..(1 + new_tokens.len()) * DIM],
        &reference.hidden.stream.to_host_vec().unwrap()[..(1 + new_tokens.len()) * DIM],
    );
    runtime.model.lm_head.proj = original;

    // Neither the pre-step nor the would-be committed position can continue:
    // kernels already changed persistent state, so the whole history is invalid.
    for (id, slot, position) in [(10, 0, 2), (10, 0, 3), (20, 1, new_tokens.len())] {
        let error = runtime
            .step(&request(&[(id, slot, position, &[6])]))
            .unwrap_err();
        assert!(error.to_string().contains("recurrent history"), "{error}");
    }

    let mut fresh = hybrid_runtime(2);
    for (req, rows) in [
        (request(&[(20, 1, 0, &[7, 8]), (10, 0, 0, &[9])]), 3),
        (request(&[(10, 0, 1, &[10]), (20, 1, 2, &[11])]), 2),
    ] {
        let actual = runtime.step(&req).unwrap();
        let expected = fresh.step(&req).unwrap();
        assert_runtime_outputs_match(&runtime, &fresh, &actual, &expected, rows);
    }
}

#[test]
fn runtime_hybrid_finalize_failure_invalidates_and_rebuilds_history() {
    assert_finalize_failure_invalidates_history(&[4, 5], |runtime, req| {
        runtime.step(req).map(|_| ())
    });
}

#[test]
fn runtime_hybrid_fused_finalize_failure_invalidates_and_rebuilds_history() {
    assert_finalize_failure_invalidates_history(&[4, 5], |runtime, req| {
        runtime.step_fused_eager(req).map(|_| ())
    });
}

#[test]
fn runtime_hybrid_mixed_issue_failure_invalidates_and_rebuilds_history() {
    use infer_worker::application::runtime::RaggedRowKind;
    assert_finalize_failure_invalidates_history(&[4, 5], |runtime, req| {
        runtime
            .issue_fused_abc(
                req,
                &[RaggedRowKind::Decode, RaggedRowKind::PrefillFinal],
                None,
            )
            .map(|_| ())
    });
}

#[test]
fn runtime_hybrid_decode_issue_failure_invalidates_and_rebuilds_history() {
    assert_finalize_failure_invalidates_history(&[4], |runtime, req| {
        runtime.issue_decode_abc(
            req,
            0,
            &[0, 0],
            &[100, 100],
            &[true, true],
            &[],
            None,
            false,
        )
    });
}

#[test]
fn startup_self_check_releases_hybrid_history_before_serving() {
    // With only one recurrent slot, a leaked bootstrap sequence prevents the
    // first real request from running. Compare its hidden state against a
    // fresh runtime as well, to catch synthetic history leaking into reuse.
    let mut checked = hybrid_runtime(1);
    let mut fresh = hybrid_runtime(1);
    checked.startup_self_check().unwrap();
    assert!(checked.kv_pool.seq_kv_len.is_empty());
    let req = request(&[(42, 0, 0, &[3, 4])]);
    let expected = fresh.step(&req).unwrap();
    let actual = checked.step(&req).unwrap();
    assert_eq!(actual.tokens[0][0].token_id, expected.tokens[0][0].token_id);
    close(
        &checked.hidden.stream.to_host_vec().unwrap()[..2 * DIM],
        &fresh.hidden.stream.to_host_vec().unwrap()[..2 * DIM],
    );
}

#[test]
fn startup_self_check_releases_hybrid_history_after_failure() {
    let mut runtime = hybrid_runtime(1);
    // Fail finalization after recurrent layers have already advanced history.
    let original = std::mem::replace(&mut runtime.model.lm_head.proj, linear(VOCAB, DIM + 1, 1.0));
    assert!(runtime.startup_self_check().is_err());
    assert!(runtime.kv_pool.seq_kv_len.is_empty());
    runtime.model.lm_head.proj = original;
    runtime.startup_self_check().unwrap();
    runtime.step(&request(&[(42, 0, 0, &[3, 4])])).unwrap();
}

#[test]
fn startup_self_check_respects_small_context_and_token_budgets() {
    let mut runtime = hybrid_runtime(1);
    runtime.cap_num_tokens = 1;
    runtime.max_seq_len = 2;
    runtime.startup_self_check().unwrap();
    assert!(runtime.kv_pool.seq_kv_len.is_empty());
    runtime.max_seq_len = 1;
    assert!(
        runtime
            .startup_self_check()
            .unwrap_err()
            .to_string()
            .contains("capacity for a prompt and a decode token")
    );
}

#[test]
fn runtime_hybrid_tracks_reordered_chunks_and_recycles_cancelled_slots() {
    let reference = model(false);
    let mut expected = Fixture::new(&reference);
    let mut runtime = hybrid_runtime(2);
    assert_eq!(runtime.kv_pool.layers.len(), 2);
    runtime.prime_graphs().unwrap();
    assert!(runtime.graph.is_none());
    assert_eq!(runtime.next_capture_slot(1), None);
    for (req, seqs) in [
        (
            request(&[(10, 0, 0, &[1, 2]), (20, 1, 0, &[3])]),
            vec![(0, vec![1, 2]), (1, vec![3])],
        ),
        (
            request(&[(20, 1, 1, &[4]), (10, 0, 2, &[5, 6])]),
            vec![(1, vec![4]), (0, vec![5, 6])],
        ),
    ] {
        let seqs: Vec<_> = seqs
            .iter()
            .map(|(slot, ids)| (*slot, ids.as_slice()))
            .collect();
        let values = expected.run(&reference, &seqs, &ALL).concat();
        runtime.step(&req).unwrap();
        close(
            &runtime.hidden.stream.to_host_vec().unwrap()[..values.len()],
            &values,
        );
    }
    // Missing history, duplicate ids, and exhaustion cannot mutate valid requests.
    assert!(runtime.step(&request(&[(30, 2, 2, &[1])])).is_err());
    assert!(
        runtime
            .step(&request(&[(10, 0, 4, &[1]), (10, 0, 4, &[1])]))
            .is_err()
    );
    assert!(runtime.step(&request(&[(30, 2, 0, &[1])])).is_err());
    runtime.release_sequence(10);
    runtime.release_sequence(10); // idempotent cancellation
    let mut fresh = Fixture::new(&reference);
    let values = fresh.run(&reference, &[(2, &[7, 8])], &ALL).concat();
    runtime.step(&request(&[(30, 2, 0, &[7, 8])])).unwrap();
    close(
        &runtime.hidden.stream.to_host_vec().unwrap()[..values.len()],
        &values,
    );
    // Omitted requests retain their histories; releasing another slot cannot affect them.
    let values = expected.run(&reference, &[(1, &[9])], &ALL).concat();
    runtime.step(&request(&[(20, 1, 2, &[9])])).unwrap();
    close(
        &runtime.hidden.stream.to_host_vec().unwrap()[..values.len()],
        &values,
    );
    runtime.retain_sequences([]);
    runtime.step(&request(&[(10, 0, 0, &[1])])).unwrap();
}

#[test]
fn runtime_hybrid_mixed_and_abc_use_the_same_request_history() {
    use infer_worker::application::runtime::RaggedRowKind;
    let mut runtime = hybrid_runtime(2);
    let mut ordinary = hybrid_runtime(2);
    let prefill = request(&[(10, 0, 0, &[1, 2]), (20, 1, 0, &[3])]);
    ordinary.step(&prefill).unwrap();
    runtime
        .step_fused_abc_eager(
            &prefill,
            &[RaggedRowKind::PrefillFinal, RaggedRowKind::PrefillFinal],
            None,
        )
        .unwrap();
    let decode = request(&[(20, 1, 1, &[4]), (10, 0, 2, &[5])]);
    let expected = ordinary.step(&decode).unwrap();
    runtime
        .issue_decode_abc(
            &decode,
            0,
            &[0, 0],
            &[100, 100],
            &[true, true],
            &[],
            None,
            false,
        )
        .unwrap();
    let out = runtime.finalize_decode_abc(2).unwrap();
    for (i, token) in out.active.iter().enumerate() {
        assert_eq!(token.token_id, expected.tokens[i][0].token_id);
    }
    close(
        &runtime.hidden.stream.to_host_vec().unwrap()[..2 * DIM],
        &ordinary.hidden.stream.to_host_vec().unwrap()[..2 * DIM],
    );
    let mut spec = request(&[(10, 0, 3, &[6, 7])]);
    // Stochastic verification remains unsupported and must fail before mutation.
    spec.draft_tokens = vec![vec![7]];
    spec.sampling[0].temperature = 1.0;
    let error = runtime.step(&spec).unwrap_err();
    assert!(error.to_string().contains("stochastic"), "{error}");
    spec.sampling[0].temperature = 0.0;
    spec.draft_tokens.clear();
    let actual = runtime.step(&spec).unwrap();
    let expected = ordinary.step(&spec).unwrap();
    assert_runtime_outputs_match(&runtime, &ordinary, &actual, &expected, 2);
}

#[test]
fn runtime_hybrid_overlapped_decode_and_mixed_keep_committed_history() {
    use infer_worker::application::runtime::RaggedRowKind;
    let mut runtime = hybrid_runtime(2);
    let mut ordinary = hybrid_runtime(2);
    assert!(runtime.mixed_eager_mode());
    let prefill = request(&[(10, 0, 0, &[1, 2])]);
    runtime.step(&prefill).unwrap();
    ordinary.step(&prefill).unwrap();

    let decode = request(&[(10, 0, 2, &[3])]);
    let expected_decode = ordinary.step(&decode).unwrap();
    let next_token = expected_decode.tokens[0][0].token_id;
    runtime
        .issue_decode_abc(&decode, 0, &[0], &[100], &[true], &[], None, false)
        .unwrap();

    let mut verify = request(&[(10, 0, 3, &[next_token])]);
    verify.draft_tokens = vec![vec![]];
    assert!(
        runtime
            .step(&verify)
            .unwrap_err()
            .to_string()
            .contains("in-flight")
    );

    let kinds = [RaggedRowKind::Decode, RaggedRowKind::PrefillFinal];
    let mixed = request(&[(10, 0, 3, &[next_token]), (20, 1, 0, &[4, 5])]);
    let expected_mixed = ordinary.step(&mixed).unwrap();
    let mut placeholders = mixed.clone();
    placeholders.seqs[0].input_ids[0] = (next_token + 1) % VOCAB as i32;
    // The earlier decode still owns the host mirrors. Its GPU argmax feeds the
    // mixed tape, while recurrent history must already expose its issued prefix.
    let ticket = runtime
        .issue_fused_abc_overlapped(&placeholders, &kinds, None, 1)
        .unwrap();
    let decoded = runtime.finalize_decode_abc(1).unwrap();
    assert_eq!(decoded.active.len(), 1);
    assert!(decoded.finished.is_empty());
    assert_eq!(decoded.active[0].src_row, 0);
    assert_eq!(decoded.active[0].token_id, next_token);
    let actual_mixed = runtime
        .finalize_fused_abc(ticket, &placeholders, &kinds)
        .unwrap();
    assert_runtime_outputs_match(&runtime, &ordinary, &actual_mixed, &expected_mixed, 3);

    // Finalizing the earlier ticket must not reset the later step's length or
    // discard a request admitted by that later step; reorder both on continuation.
    let continuation = request(&[(20, 1, 2, &[7]), (10, 0, 4, &[8])]);
    let actual = runtime.step(&continuation).unwrap();
    let expected = ordinary.step(&continuation).unwrap();
    assert_runtime_outputs_match(&runtime, &ordinary, &actual, &expected, 2);
}

#[test]
fn runtime_hybrid_decode_collection_failure_invalidates_only_its_owners() {
    use infer_worker::application::runtime::RaggedRowKind;
    let mut runtime = hybrid_runtime(2);
    let mut ordinary = hybrid_runtime(2);
    let prefill = request(&[(10, 0, 0, &[1, 2])]);
    runtime.step(&prefill).unwrap();
    ordinary.step(&prefill).unwrap();
    let decode = request(&[(10, 0, 2, &[3])]);
    let next_token = ordinary.step(&decode).unwrap().tokens[0][0].token_id;
    runtime
        .issue_decode_abc(&decode, 0, &[0], &[100], &[true], &[], None, false)
        .unwrap();

    let kinds = [RaggedRowKind::Decode, RaggedRowKind::PrefillFinal];
    let mixed = request(&[(10, 0, 3, &[next_token]), (20, 1, 0, &[4, 5])]);
    ordinary.step(&mixed).unwrap();
    let ticket = runtime
        .issue_fused_abc_overlapped(&mixed, &kinds, None, 1)
        .unwrap();
    // A late collection error belongs to the earlier decode's owner set,
    // even though another issue has since advanced shared/new histories.
    let error = runtime.finalize_decode_abc(0).unwrap_err();
    assert!(
        error.to_string().contains("compact counts invalid"),
        "{error}"
    );
    runtime.finalize_fused_abc(ticket, &mixed, &kinds).unwrap();
    let error = runtime.step(&request(&[(10, 0, 4, &[6])])).unwrap_err();
    assert!(error.to_string().contains("recurrent history"), "{error}");

    let continuation = request(&[(20, 1, 2, &[7])]);
    let actual = runtime.step(&continuation).unwrap();
    let expected = ordinary.step(&continuation).unwrap();
    assert_runtime_outputs_match(&runtime, &ordinary, &actual, &expected, 1);

    let mut fresh = hybrid_runtime(1);
    let rebuild = request(&[(10, 0, 0, &[8, 9])]);
    let actual = runtime.step(&rebuild).unwrap();
    let expected = fresh.step(&rebuild).unwrap();
    assert_runtime_outputs_match(&runtime, &fresh, &actual, &expected, 2);
}

/// Independent scalar oracle: per-head Q/gate packing, partial rotation,
/// causal GQA, then sigmoid gating before an identity output projection.
#[test]
fn full_attention_partial_rope_and_gate_match_scalar_reference() {
    use infer_worker::domain::component::Component;
    let input = [
        [0.2, 0.4, -0.3, 0.7, 0.9, -0.1, 0.5, 0.8],
        [0.8, -0.2, 0.6, 0.1, -0.4, 0.3, 0.7, -0.9],
        [-0.5, 0.9, 0.2, -0.6, 0.4, 0.8, -0.7, 0.3],
    ];
    let queries = [[0.3, -0.7, 0.8, 0.2], [-0.4, 0.9, 0.1, -0.6]];
    let gates: [[f32; 4]; 2] = [[-1.0, 0.0, 1.0, 2.0], [0.7, -0.8, 1.3, -2.0]];
    let normalized: Vec<Vec<f32>> = input
        .iter()
        .map(|row| {
            let scale = 1.0 / (row.iter().map(|x| x * x).sum::<f32>() / DIM as f32 + 1e-6).sqrt();
            row.iter().map(|x| x * scale).collect()
        })
        .collect();
    for qk_norm in [false, true] {
        let normalize_head = |head: &[f32]| -> Vec<f32> {
            let scale = if qk_norm {
                1.0 / (head.iter().map(|x| x * x).sum::<f32>() / 4.0 + 1e-6).sqrt()
            } else {
                1.0
            };
            head.iter().map(|x| x * scale).collect()
        };
        let rotate = |mut head: Vec<f32>, pos: usize| {
            let (sin, cos) = (pos as f32 * 0.37).sin_cos();
            let (a, b) = (head[0], head[1]);
            head[0] = a * cos - b * sin;
            head[1] = b * cos + a * sin;
            head
        };
        let keys: Vec<_> = normalized
            .iter()
            .enumerate()
            .map(|(p, row)| rotate(normalize_head(&row[..4]), p))
            .collect();
        let mut expected = Vec::new();
        for pos in 0..3 {
            for head in 0..2 {
                let query = rotate(normalize_head(&queries[head]), pos);
                let scores: Vec<f32> = keys[..=pos]
                    .iter()
                    .map(|k| query.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() * 0.5)
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probs: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = probs.iter().sum();
                for col in 0..4 {
                    let value = probs
                        .iter()
                        .enumerate()
                        .map(|(p, prob)| prob / total * normalized[p][4 + col])
                        .sum::<f32>();
                    expected.push(value / (1.0 + (-gates[head][col]).exp()));
                }
            }
        }
        for shared in [false, true] {
            let model = model(false);
            let mut fixture = Fixture::new(&model);
            let step = fixture.step(&[(0, &[1, 2, 3])]);
            let mut projection = vec![0.0; 24 * DIM];
            for col in 0..8 {
                projection[(16 + col) * DIM + col] = 1.0;
            }
            let mut bias = Vec::new();
            for head in 0..2 {
                bias.extend(queries[head]);
                bias.extend(gates[head]);
            }
            bias.extend([0.0; 8]);
            let mut identity = vec![0.0; DIM * DIM];
            for col in 0..DIM {
                identity[col * DIM + col] = 1.0;
            }
            let attention = FullAttention {
                input_layernorm: norm(DIM),
                qkv_proj: Linear::new(
                    Tensor::from_host_slice(&projection, [24, DIM], &Cpu).unwrap(),
                    Some(Tensor::from_host_slice(&bias, [24], &Cpu).unwrap()),
                ),
                o_proj: Linear::new(
                    Tensor::from_host_slice(&identity, [DIM, DIM], &Cpu).unwrap(),
                    None,
                ),
                q_norm: qk_norm.then(|| norm(4)),
                k_norm: qk_norm.then(|| norm(4)),
                sin: Tensor::from_host_slice(
                    &(0..MAX_SEQ)
                        .map(|p| (p as f32 * 0.37).sin())
                        .collect::<Vec<_>>(),
                    [MAX_SEQ, 1],
                    &Cpu,
                )
                .unwrap(),
                cos: Tensor::from_host_slice(
                    &(0..MAX_SEQ)
                        .map(|p| (p as f32 * 0.37).cos())
                        .collect::<Vec<_>>(),
                    [MAX_SEQ, 1],
                    &Cpu,
                )
                .unwrap(),
                head_num: 2,
                kv_head_num: 1,
                head_dim: 4,
                rotary_dim: 2,
                scale: 0.5,
                attn_output_gate: true,
                scratch: shared.then(|| ForwardScratch::new(&Cpu, model.dims(), 8, SLOTS).unwrap()),
            };
            let scope = HostScope::new(Cpu);
            let ctx = StepCtx::new(&scope, &step.plan);
            let run = |attention: &FullAttention<f32, Cpu>, fixture: &mut Fixture| {
                let mut hidden = Hidden {
                    stream: Tensor::from_host_slice(&input.concat(), [3, DIM], &Cpu).unwrap(),
                    pending: None,
                };
                let mut kv = fixture
                    .kv
                    .view(LayerRange { start: 0, end: 1 }, &step.index);
                attention.run(&mut hidden, Some(&mut kv), &ctx).unwrap();
                close(&hidden.pending.unwrap().to_host_vec().unwrap(), &expected);
                close(&hidden.stream.to_host_vec().unwrap(), &input.concat());
            };
            run(&attention, &mut fixture);
            // Repeat with the same scratch to expose lifetime/aliasing errors.
            run(&attention, &mut fixture);
            let stored_keys = fixture.kv.layers[0].k.to_host_vec().unwrap();
            close(&stored_keys[..12], &keys.concat());
            // Incremental decode must agree with the causal prefill oracle.
            let mut decode = Fixture::new(&model);
            for pos in 0..3 {
                let step = decode.step(&[(0, &[1])]);
                let ctx = StepCtx::new(&scope, &step.plan);
                let mut hidden = Hidden {
                    stream: Tensor::from_host_slice(&input[pos], [1, DIM], &Cpu).unwrap(),
                    pending: None,
                };
                let mut kv = decode.kv.view(LayerRange { start: 0, end: 1 }, &step.index);
                attention.run(&mut hidden, Some(&mut kv), &ctx).unwrap();
                close(
                    &hidden.pending.unwrap().to_host_vec().unwrap(),
                    &expected[pos * DIM..(pos + 1) * DIM],
                );
                decode.positions[0] += 1;
            }
        }
    }
}

#[test]
fn beam_search_cached_gdn_forks_match_recomputed_prefixes_and_reuse() {
    use infer_worker::application::beam_search::{BeamSearchConfig, BeamSession};
    let prompt = [1, 2, 3, 4, 5];
    let reference_model = model(false);
    let mut oracle = vec![(Vec::<i32>::new(), 0.0f64)];
    for _ in 0..3 {
        let mut expansions = Vec::new();
        for (suffix, score) in &oracle {
            let mut input = prompt.to_vec();
            input.extend_from_slice(suffix);
            let mut fixture = Fixture::new(&reference_model);
            let step = fixture.step(&[(0, &input)]);
            let scope = HostScope::new(Cpu);
            let ctx = StepCtx::new(&scope, &step.plan);
            let mut hidden = Hidden {
                stream: Tensor::zeros([input.len(), DIM], &Cpu).unwrap(),
                pending: None,
            };
            let mut cache = ModelCacheView::hybrid(
                &mut fixture.kv,
                &step.index,
                &mut fixture.linear,
                &step.linear,
            );
            let logits = reference_model
                .forward(
                    &step.ids,
                    &mut hidden,
                    &mut cache,
                    infer_worker::domain::model::SampleRows::LastPerSeq,
                    &ctx,
                )
                .unwrap()
                .0
                .to_host_vec()
                .unwrap();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let logsum = logits
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln();
            for (id, &x) in logits.iter().enumerate() {
                let mut next = suffix.clone();
                next.push(id as i32);
                expansions.push((next, score + x as f64 - max - logsum));
            }
        }
        expansions.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        expansions.truncate(3);
        oracle = expansions;
    }
    let mut session = BeamSession::new(
        model(false),
        HostScope::new(Cpu),
        BeamSearchConfig {
            width: 3,
            max_context: MAX_SEQ,
            max_step_tokens: 3,
            max_new_tokens: 3,
            length_penalty: 0.0,
            eos_ids: vec![],
        },
    )
    .unwrap();
    for _ in 0..2 {
        let actual = session.generate(&prompt).unwrap();
        for (actual, expected) in actual.iter().zip(&oracle) {
            assert_eq!(actual.token_ids, expected.0);
            assert!(
                (actual.logprob - expected.1).abs() < 2e-5,
                "{} vs {}",
                actual.logprob,
                expected.1
            );
        }
    }
    assert!(session.generate(&[VOCAB as i32]).is_err());
}

#[test]
fn beam_width_one_matches_hybrid_greedy() {
    use infer_worker::application::beam_search::{BeamSearchConfig, BeamSession};
    let mut greedy = hybrid_runtime(1);
    let mut tokens = vec![1, 2, 3];
    let mut expected = Vec::new();
    for i in 0..4 {
        let start = if i == 0 { 0 } else { tokens.len() - 1 };
        let out = greedy
            .step(&request(&[(0, 0, start, &tokens[start..])]))
            .unwrap();
        let id = out.tokens[0][0].token_id;
        expected.push(id);
        tokens.push(id);
    }
    let mut beam = BeamSession::new(
        model(false),
        HostScope::new(Cpu),
        BeamSearchConfig {
            width: 1,
            max_context: MAX_SEQ,
            max_step_tokens: 3,
            max_new_tokens: 4,
            length_penalty: 1.0,
            eos_ids: vec![],
        },
    )
    .unwrap();
    assert_eq!(beam.generate(&[1, 2, 3]).unwrap()[0].token_ids, expected);
}
