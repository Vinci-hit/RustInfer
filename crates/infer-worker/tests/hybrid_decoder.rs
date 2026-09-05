//! Numerical tests of the mixed decoder, independent of Runtime's future slot
//! allocator. Every fixture explicitly owns the KV blocks and recurrent slots.

use std::collections::HashMap;

use infer_worker::application::runtime::Runtime;
use infer_worker::application::sampler_stack::GreedySampler;
use infer_worker::components::{
    Attention, DecoderBlock, DenseFfn, Embed, GatedDeltaNet, GdnWeights, Linear, LmHead, Mixer,
    RmsNorm,
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
use infer_worker::domain::plan::{BatchKind, BatchPlan, MaskMode};
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
        let mixer = if i % 4 == 3 {
            Mixer::Full(Attention {
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
                scale: 0.5,
                scratch: None,
            })
        } else {
            Mixer::Linear(gdn(seed))
        };
        blocks.push(DecoderBlock {
            mixer,
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

#[test]
fn runtime_rejects_hybrid_before_attempting_allocation() {
    let result = Runtime::new(
        model(false),
        HostScope::new(Cpu),
        Box::new(GreedySampler),
        usize::MAX,
        1,
        MAX_SEQ,
        MAX_SEQ,
        usize::MAX,
        SLOTS,
        Vec::new(),
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("hybrid Runtime unexpectedly accepted"),
    };
    assert!(error.to_string().contains("recurrent state lifecycle"));
}
