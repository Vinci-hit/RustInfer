//! Official Python DFlash reference plus incremental cache and shared verifier tests.
use infer_worker::application::speculative::{
    BlockProposer, DraftProposer, ProposerSession, SpeculativeLimits,
};
use infer_worker::components::dflash::DFlashDraftHead;
use infer_worker::domain::{
    cache::ModelCacheView,
    draft::BlockDraft,
    exec::{HostScope, StepCtx},
    kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool},
    model::DecoderModel,
    plan::{BatchKind, BatchPlan, MaskMode},
    tensor::Tensor,
};
use infer_worker::infrastructure::{cpu::Cpu, io::safetensors::SafetensorsReader};
use infer_worker::models::{
    loader::{LoadConfig, WeightLoader},
    qwen3::{
        self,
        dflash::{DFlashConfig, build},
    },
};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dflash")
        .join(name)
}

fn golden() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(fixture("golden.json")).unwrap()).unwrap()
}

fn matrix(g: &serde_json::Value, name: &str) -> Tensor<f32, Cpu> {
    let a = g[name].as_array().unwrap();
    let cols = a[0].as_array().unwrap().len();
    let values: Vec<f32> = a
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    Tensor::from_host_slice(&values, [a.len(), cols], &Cpu).unwrap()
}

fn ints(g: &serde_json::Value, name: &str) -> Vec<i32> {
    g[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}

fn close(actual: &Tensor<f32, Cpu>, expected: &Tensor<f32, Cpu>) {
    assert_eq!(actual.shape(), expected.shape());
    let a = actual.to_host_vec().unwrap();
    let b = expected.to_host_vec().unwrap();
    let error = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(error < 2e-5, "max absolute error {error}");
}

fn target_config() -> LoadConfig {
    LoadConfig {
        dim: 8,
        intermediate_size: 16,
        layer_num: 6,
        head_num: 2,
        kv_head_num: 1,
        head_dim: 4,
        vocab_size: 16,
        seq_len: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_scaling: None,
        mlp_quant: None,
        fp8_block: None,
        rotary_dim: 4,
        attn_output_gate: false,
        linear_attn: None,
        num_experts: 0,
        experts_per_tok: 0,
        moe_intermediate_size: 0,
        norm_topk_prob: false,
        decoder_sparse_step: 1,
    }
}

fn target() -> qwen3::Qwen3Model<f32, Cpu> {
    let reader = SafetensorsReader::open(fixture("../eagle3/target.safetensors")).unwrap();
    qwen3::build(&WeightLoader::new(&reader), &target_config(), &Cpu).unwrap()
}

fn index(start: usize, n: usize) -> (BatchPlan, KvIndexTensors<Cpu>) {
    let positions: Vec<i32> = (start as i32..(start + n) as i32).collect();
    let (cu, requests, tiles) = BatchPlan::plan_ragged_tiles(&[n as i32]);
    let plan = BatchPlan {
        kind: if n == 1 {
            BatchKind::DecodeOnly
        } else {
            BatchKind::Ragged
        },
        num_tokens: n,
        batch: 1,
        q_lens: vec![n as i32],
        kv_lens: vec![(start + n) as i32],
        seq_positions: vec![start as i32],
        rope_positions: positions.clone(),
        max_blocks_per_seq: 64,
        block_size: 1,
        total_q_tiles: requests.len() as i32,
    };
    let ints = |x: &[i32]| Tensor::from_host_slice(x, [x.len()], &Cpu).unwrap();
    let idx = KvIndexTensors {
        decode_rows: None,
        block_tables: Tensor::from_host_slice(&(0..64).collect::<Vec<i32>>(), [1, 64], &Cpu)
            .unwrap(),
        cu_q_lens: ints(&cu),
        kv_lens: ints(&plan.kv_lens),
        seq_positions: ints(&[start as i32]),
        seq_lens_step: ints(&[n as i32]),
        rope_positions: ints(&positions),
        block2req: ints(&requests),
        block2tile: ints(&tiles),
        valid_q_tiles: ints(&[requests.len() as i32]),
        valid_suffix_q_tiles: ints(&[requests.len() as i32]),
    };
    (plan, idx)
}

fn pool(layers: usize) -> PagedKvPool<f32, Cpu> {
    PagedKvPool {
        layers: (0..layers)
            .map(|_| PagedKvLayer {
                k: Tensor::zeros([64, 1, 4], &Cpu).unwrap(),
                v: Tensor::zeros([64, 1, 4], &Cpu).unwrap(),
            })
            .collect(),
        num_blocks: 64,
        block_size: 1,
        kv_dim: 4,
        quant: KvQuantTier::None,
        seq_kv_len: Default::default(),
    }
}

fn config() -> DFlashConfig {
    DFlashConfig::parse(&std::fs::read(fixture("config.json")).unwrap()).unwrap()
}
fn head() -> DFlashDraftHead<f32, Cpu> {
    let reader = SafetensorsReader::open(fixture("draft.safetensors")).unwrap();
    let target = target();
    build(
        &WeightLoader::new(&reader),
        &config(),
        target.dims(),
        &target.embed.table,
        target.lm_head.proj.weight.as_dense().unwrap(),
        64,
        &Cpu,
    )
    .unwrap()
}

#[test]
fn block_logits_and_context_cache_match_official_python() {
    let g = golden();
    let scope = HostScope::new(Cpu);
    let mut head = head();
    head.prepare(16).unwrap();
    let mut kv = pool(3);
    let (plan, ix) = index(0, 5);
    head.cache_features(
        &matrix(&g, "features").narrow(0, 0, 5).unwrap(),
        &mut ModelCacheView::full(&mut kv, &ix),
        &StepCtx::new(&scope, &plan),
    )
    .unwrap();
    for (i, layer) in kv.layers.iter().enumerate() {
        for (actual, key) in [(&layer.k, "context_k"), (&layer.v, "context_v")] {
            let expected = serde_json::json!({"matrix": g[key][i]});
            close(
                &actual
                    .narrow(0, 0, 5)
                    .unwrap()
                    .view_contiguous([5, 4].into())
                    .unwrap(),
                &matrix(&expected, "matrix"),
            );
        }
    }
    let mut ids = vec![15; 8];
    ids[0] = ints(&g, "trajectory")[5];
    let ids = Tensor::from_host_slice(&ids, [8], &Cpu).unwrap();
    let mut logits = Tensor::zeros([7, 16], &Cpu).unwrap();
    let (mut plan, ix) = index(5, 8);
    assert!(
        head.forward_block(
            &ids,
            &mut ModelCacheView::full(&mut kv, &ix),
            &mut logits,
            &StepCtx::new(&scope, &plan)
        )
        .is_err()
    );
    plan.kind = BatchKind::Spec {
        mask: MaskMode::Full,
        mask_handle: None,
    };
    head.forward_block(
        &ids,
        &mut ModelCacheView::full(&mut kv, &ix),
        &mut logits,
        &StepCtx::new(&scope, &plan),
    )
    .unwrap();
    close(&logits, &matrix(&g, "logits"));
}

#[test]
fn confirmed_features_replace_noise_after_every_retained_prefix_and_chunking() {
    let g = golden();
    let ids = ints(&g, "trajectory");
    let features = matrix(&g, "features");
    let scope = HostScope::new(Cpu);
    for kept in 1..=8 {
        for chunk in [1, 3, 5] {
            let mut p = BlockProposer::new(head(), 64, 16, &Cpu).unwrap();
            let mut start = 0;
            for tokens in ids[..5].chunks(chunk) {
                p.observe(
                    tokens,
                    start,
                    &features.narrow(0, start, tokens.len()).unwrap(),
                    &scope,
                )
                .unwrap();
                start += tokens.len();
            }
            let (drafts, tape) = p.draft_with_device(ids[5], 7, &scope).unwrap();
            assert_eq!(tape.to_host_vec().unwrap(), [vec![ids[5]], drafts].concat());
            assert_eq!(p.context_len(), 5);
            // Even a completely accepted block must be replaced by target features.
            p.observe(
                &ids[5..5 + kept],
                5,
                &features.narrow(0, 5, kept).unwrap(),
                &scope,
            )
            .unwrap();
            let mut fresh = BlockProposer::new(head(), 64, 16, &Cpu).unwrap();
            fresh
                .observe(
                    &ids[..5 + kept],
                    0,
                    &features.narrow(0, 0, 5 + kept).unwrap(),
                    &scope,
                )
                .unwrap();
            for width in [7, 1, 3, 0, 7] {
                let a = p.draft_with_device(ids[5 + kept], width, &scope).unwrap().0;
                let b = fresh
                    .draft_with_device(ids[5 + kept], width, &scope)
                    .unwrap()
                    .0;
                assert_eq!(a, b, "kept={kept} chunk={chunk} K={width}");
                assert_eq!(p.context_len(), 5 + kept);
            }
            p.reset();
            assert_eq!(p.context_len(), 0);
            assert!(p.draft_with_device(ids[0], 1, &scope).is_err());
        }
    }
}

#[test]
fn shared_session_matches_target_greedy_and_handles_budget_eos_reset() {
    let g = golden();
    let ids = ints(&g, "trajectory");
    for (prompt_len, k, count, context, eos) in [
        (5, 7, 1, 64, vec![]),
        (5, 7, 2, 64, vec![]),
        (5, 7, 17, 64, vec![]),
        (17, 3, 10, 64, vec![]),
        (5, 3, 10, 6, vec![]),
        (5, 7, 17, 64, vec![ids[5]]),
        (5, 7, 17, 64, vec![ids[6]]),
    ] {
        let cap = context.min(8);
        let target = target();
        let feature_spec = config().feature_spec(target.dims()).unwrap();
        let proposer = BlockProposer::new(head(), context, cap, &Cpu).unwrap();
        let mut session = ProposerSession::from_proposer(
            target,
            proposer,
            HostScope::new(Cpu),
            SpeculativeLimits {
                max_context: context,
                max_step_tokens: cap,
                max_output_tokens: count,
                draft_tokens: k,
                eos_ids: eos.clone(),
            },
            feature_spec,
        )
        .unwrap();
        for _ in 0..2 {
            let mut step = session.prefill(&ids[..prompt_len]).unwrap();
            let mut output = step.tokens;
            while !step.finished {
                step = session.decode().unwrap();
                output.extend_from_slice(&step.tokens);
                assert_eq!(session.cached_tokens(), session.draft_cached_tokens());
            }
            assert_eq!(&output, &ids[prompt_len..prompt_len + output.len()]);
            let expected_count = (count as usize).min(context - prompt_len + 1);
            if let Some(pos) = ids[prompt_len..prompt_len + expected_count]
                .iter()
                .position(|id| eos.contains(id))
            {
                assert_eq!(output.len(), pos + 1);
            } else {
                assert_eq!(output.len(), expected_count);
            }
            assert!(session.decode().is_err());
            session.reset().unwrap();
        }
    }
}

#[test]
fn loader_rejects_wrong_architecture_geometry_mask_and_feature_layers() {
    let valid = config();
    let dims = target().dims();
    valid.validate(dims, 64, 7).unwrap();
    assert!(valid.validate(dims, 64, 8).is_err());
    assert!(valid.validate(dims, 65, 7).is_err());
    for change in [
        "architecture",
        "mask",
        "features",
        "layers",
        "vocab",
        "rope",
        "heads",
    ] {
        let mut cfg = valid.clone();
        match change {
            "architecture" => cfg.architectures = vec!["Qwen3DSparkModel".into()],
            "mask" => cfg.dflash_config.mask_token_id = 16,
            "features" => cfg.dflash_config.target_layer_ids = vec![1, 1, 3],
            "layers" => cfg.num_target_layers += 1,
            "vocab" => cfg.vocab_size += 1,
            "rope" => cfg.rope_scaling = Some(serde_json::json!({"type":"linear"})),
            "heads" => cfg.num_key_value_heads = 3,
            _ => unreachable!(),
        }
        assert!(cfg.validate(dims, 64, 7).is_err(), "{change}");
    }
}
