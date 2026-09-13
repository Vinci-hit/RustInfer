//! Independent PyTorch fixtures plus cache-reconstruction and decoding oracles.
use infer_worker::application::speculative::{
    ConditionedProposer, SpeculativeLimits, SpeculativeSession,
};
use infer_worker::components::eagle3::Eagle3DraftHead;
use infer_worker::domain::{
    cache::ModelCacheView,
    component::{Hidden, LayerRange},
    draft::ConditionedDraft,
    exec::{HostScope, StepCtx},
    features::TargetFeatures,
    forward_scratch::ForwardScratch,
    kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool},
    model::{DecoderModel, SampleRows},
    plan::{BatchKind, BatchPlan},
    tensor::Tensor,
};
use infer_worker::infrastructure::{cpu::Cpu, io::safetensors::SafetensorsReader};
use infer_worker::models::{
    loader::{LoadConfig, WeightLoader},
    qwen3::{
        self,
        eagle3::{Eagle3Checkpoint, Eagle3Config, Eagle3Format, build_draft},
    },
};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eagle3")
        .join(name)
}
fn golden() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(fixture("golden.json")).unwrap()).unwrap()
}
fn config() -> Eagle3Config {
    serde_json::from_slice(&std::fs::read(fixture("config.json")).unwrap()).unwrap()
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
    let reader = SafetensorsReader::open(fixture("target.safetensors")).unwrap();
    qwen3::build(&WeightLoader::new(&reader), &target_config(), &Cpu).unwrap()
}
fn head() -> Eagle3DraftHead<f32, Cpu> {
    let reader = SafetensorsReader::open(fixture("draft.safetensors")).unwrap();
    build_draft(
        &WeightLoader::new(&reader),
        &config(),
        target().dims(),
        64,
        &Cpu,
    )
    .unwrap()
}

fn specforge() -> (Eagle3Checkpoint, Eagle3DraftHead<f32, Cpu>) {
    let reader = SafetensorsReader::open(fixture("specforge/draft.safetensors")).unwrap();
    let loader = WeightLoader::new(&reader);
    let checkpoint = Eagle3Checkpoint::parse(
        &std::fs::read(fixture("specforge/config.json")).unwrap(),
        &loader,
    )
    .unwrap();
    let target = target();
    let head = checkpoint
        .build(&loader, target.dims(), &target.embed.table, 64, &Cpu)
        .unwrap();
    (checkpoint, head)
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

#[test]
fn draft_projection_hidden_logits_and_incremental_kv_match_pytorch() {
    let g = golden();
    let scope = HostScope::new(Cpu);
    let mut head = head();
    head.prepare(32, 1).unwrap();
    let (plan, index) = index(0, 5);
    let ctx = StepCtx::new(&scope, &plan);
    let mut kv = pool(1);
    let mut projected = Tensor::zeros([5, 8], &Cpu).unwrap();
    head.project_features_into(
        &matrix(&g, "features").narrow(0, 0, 5).unwrap(),
        &mut projected,
        &ctx,
    )
    .unwrap();
    close(&projected, &matrix(&g, "projected"));
    let ids = Tensor::from_host_slice(&ints(&g, "paired"), [5], &Cpu).unwrap();
    let mut hidden = Tensor::zeros([5, 8], &Cpu).unwrap();
    head.forward_hidden_into(
        &ids,
        &projected,
        &mut ModelCacheView::full(&mut kv, &index),
        &mut hidden,
        &ctx,
    )
    .unwrap();
    close(&hidden, &matrix(&g, "hidden"));
    let mut logits = Tensor::zeros([5, 16], &Cpu).unwrap();
    head.project_logits_into(&hidden, &mut logits, &ctx)
        .unwrap();
    close(&logits, &matrix(&g, "logits"));
    for (actual, name) in [(&kv.layers[0].k, "k"), (&kv.layers[0].v, "v")] {
        close(
            &actual
                .narrow(0, 0, 5)
                .unwrap()
                .view_contiguous([5, 4].into())
                .unwrap(),
            &matrix(&g, name),
        );
    }
    let (plan, index) = self::index(5, 1);
    let ctx = StepCtx::new(&scope, &plan);
    let ids = Tensor::from_host_slice(&ints(&g, "next_token"), [1], &Cpu).unwrap();
    let mut next = Tensor::zeros([1, 8], &Cpu).unwrap();
    head.forward_hidden_into(
        &ids,
        &hidden.narrow(0, 4, 1).unwrap(),
        &mut ModelCacheView::full(&mut kv, &index),
        &mut next,
        &ctx,
    )
    .unwrap();
    close(&next, &matrix(&g, "next_hidden"));
    let mut logits = Tensor::zeros([1, 16], &Cpu).unwrap();
    head.project_logits_into(&next, &mut logits, &ctx).unwrap();
    close(&logits, &matrix(&g, "next_logits"));
    close(
        &kv.layers[0]
            .k
            .narrow(0, 0, 6)
            .unwrap()
            .view_contiguous([6, 4].into())
            .unwrap(),
        &matrix(&g, "next_k"),
    );
}

#[test]
fn taps_include_deferred_residual_and_do_not_change_target_logits() {
    let g = golden();
    let tokens = ints(&g, "trajectory");
    let n = tokens.len();
    let scope = HostScope::new(Cpu);
    let mut model = target();
    let dims = model.dims();
    model.install_scratch(ForwardScratch::new(&Cpu, dims, 32, 1).unwrap());
    let (plan, index) = index(0, n);
    let ctx = StepCtx::new(&scope, &plan);
    let mut kv = pool(6);
    let ids = Tensor::from_host_slice(&tokens, [n], &Cpu).unwrap();
    let mut hidden = Hidden {
        stream: Tensor::zeros([n, 8], &Cpu).unwrap(),
        pending: None,
    };
    let mut taps = TargetFeatures::new(config().feature_spec(), 8, 32, &Cpu).unwrap();
    let TargetFeatures::Layers(layers) = &mut taps else {
        unreachable!()
    };
    layers.validate(dims, n, &Cpu).unwrap();
    model.embed(&ids, &mut hidden, &ctx).unwrap();
    model
        .decode_layers_observed(
            LayerRange::all(6),
            &mut hidden,
            &mut ModelCacheView::full(&mut kv, &index),
            &ctx,
            layers,
        )
        .unwrap();
    layers.concatenate(n, &scope).unwrap();
    close(&taps.rows(n).unwrap(), &matrix(&g, "features"));
    let observed = model
        .finalize(&hidden, SampleRows::All, &ctx)
        .unwrap()
        .0
        .to_host_vec()
        .unwrap();
    model.embed(&ids, &mut hidden, &ctx).unwrap();
    model
        .decode_layers(
            LayerRange::all(6),
            &mut hidden,
            &mut ModelCacheView::full(&mut kv, &index),
            &ctx,
        )
        .unwrap();
    let ordinary = model.finalize(&hidden, SampleRows::All, &ctx).unwrap().0;
    assert_eq!(observed, ordinary.to_host_vec().unwrap());
    close(&ordinary, &matrix(&g, "target_logits"));
}

#[test]
fn every_retained_prefix_rebuilds_the_same_next_draft_as_fresh_history() {
    let g = golden();
    let tokens = ints(&g, "trajectory");
    let features = matrix(&g, "features");
    let scope = HostScope::new(Cpu);
    for kept in 1..=8 {
        for chunk in [1, 3, 5] {
            let mut proposer = ConditionedProposer::new(head(), 64, 32, &Cpu).unwrap();
            let mut start = 0;
            for ids in tokens[..5].chunks(chunk) {
                proposer
                    .observe(
                        ids,
                        start,
                        &features.narrow(0, start, ids.len()).unwrap(),
                        &scope,
                    )
                    .unwrap();
                start += ids.len();
            }
            proposer.draft(tokens[5], 7, &scope).unwrap();
            proposer
                .observe(
                    &tokens[5..5 + kept],
                    5,
                    &features.narrow(0, 5, kept).unwrap(),
                    &scope,
                )
                .unwrap();
            let mut fresh = ConditionedProposer::new(head(), 64, 32, &Cpu).unwrap();
            fresh
                .observe(
                    &tokens[..5 + kept],
                    0,
                    &features.narrow(0, 0, 5 + kept).unwrap(),
                    &scope,
                )
                .unwrap();
            assert_eq!(proposer.committed_len(), 4 + kept);
            assert_eq!(
                proposer.draft(tokens[5 + kept], 5, &scope).unwrap(),
                fresh.draft(tokens[5 + kept], 5, &scope).unwrap()
            );
        }
    }
}

#[test]
fn session_matches_greedy_trajectory_for_chunking_rejection_and_reuse() {
    let g = golden();
    let prompt = ints(&g, "prompt");
    let expected = ints(&g, "trajectory");
    let mut rejected = false;
    for k in [0, 1, 3, 7] {
        for chunk in [k + 1, 16] {
            let limits = SpeculativeLimits {
                max_context: 64,
                max_step_tokens: chunk,
                max_output_tokens: 20,
                draft_tokens: k,
                eos_ids: vec![],
            };
            let mut session = SpeculativeSession::with_features(
                target(),
                head(),
                HostScope::new(Cpu),
                limits,
                config().feature_spec(),
            )
            .unwrap();
            for _ in 0..2 {
                session.reset().unwrap();
                let mut step = session.prefill(&prompt).unwrap();
                let mut output = step.tokens.clone();
                while !step.finished {
                    step = session.decode().unwrap();
                    rejected |= step.accepted < step.proposed;
                    output.extend_from_slice(&step.tokens);
                    assert_eq!(session.cached_tokens(), session.draft_cached_tokens() + 1);
                }
                assert_eq!(output, expected[prompt.len()..], "K={k}, chunk={chunk}");
            }
        }
    }
    assert!(rejected);
}

#[test]
fn invalid_checkpoint_geometry_is_rejected_before_loading() {
    let dims = target().dims();
    let reader = SafetensorsReader::open(fixture("draft.safetensors")).unwrap();
    let loader = WeightLoader::new(&reader);
    let mut cfg = config();
    cfg.head_dim = 8;
    assert!(build_draft::<f32, Cpu>(&loader, &cfg, dims, 64, &Cpu).is_err());
    let mut cfg = config();
    cfg.target_layer_ids[4] = 5;
    assert!(cfg.validate(dims, 64).is_err());
    let mut cfg = config();
    cfg.num_hidden_layers = 2;
    assert!(cfg.validate(dims, 64).is_err());
    assert!(config().validate(dims, 65).is_err());
}

#[test]
fn eos_output_budget_and_context_limit_commit_only_the_retained_prefix() {
    let g = golden();
    let prompt = ints(&g, "prompt");
    let trajectory = ints(&g, "trajectory");
    let generated = &trajectory[prompt.len()..];
    for k in [1, 3, 7] {
        for (budget, context, eos) in [
            (1, 64, vec![]),
            (2, 64, vec![]),
            (9, 64, vec![]),
            (20, prompt.len() + 8, vec![]),
            (20, 64, vec![generated[3]]),
        ] {
            let limits = SpeculativeLimits {
                max_context: context,
                max_step_tokens: 8,
                max_output_tokens: budget,
                draft_tokens: k,
                eos_ids: eos.clone(),
            };
            let mut session = SpeculativeSession::with_features(
                target(),
                head(),
                HostScope::new(Cpu),
                limits,
                config().feature_spec(),
            )
            .unwrap();
            let mut step = session.prefill(&prompt).unwrap();
            let mut output = step.tokens.clone();
            while !step.finished {
                step = session.decode().unwrap();
                output.extend_from_slice(&step.tokens);
                assert_eq!(session.cached_tokens(), session.draft_cached_tokens() + 1);
            }
            let expected_len = (budget as usize).min(context - prompt.len() + 1).min(
                generated
                    .iter()
                    .position(|t| eos.contains(t))
                    .map_or(generated.len(), |i| i + 1),
            );
            assert_eq!(
                output,
                generated[..expected_len],
                "K={k}, budget={budget}, context={context}, EOS={eos:?}"
            );
            assert_eq!(session.cached_tokens(), prompt.len() + output.len() - 1);
        }
    }
}

#[test]
fn detects_both_formats_and_rejects_conflicting_or_unknown_layouts() {
    for (prefix, expected) in [
        ("", Eagle3Format::DeepSpec),
        ("specforge/", Eagle3Format::SpecForge),
    ] {
        let reader =
            SafetensorsReader::open(fixture(&format!("{prefix}draft.safetensors"))).unwrap();
        let loader = WeightLoader::new(&reader);
        let raw = std::fs::read(fixture(&format!("{prefix}config.json"))).unwrap();
        assert_eq!(
            Eagle3Checkpoint::parse(&raw, &loader).unwrap().format(),
            expected
        );
        let config: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        for architecture in [
            "UnknownEagle",
            if expected == Eagle3Format::DeepSpec {
                "LlamaForCausalLMEagle3"
            } else {
                "Qwen3Eagle3Model"
            },
        ] {
            let mut wrong = config.clone();
            wrong["architectures"] = serde_json::json!([architecture]);
            assert!(
                Eagle3Checkpoint::parse(&serde_json::to_vec(&wrong).unwrap(), &loader).is_err()
            );
        }
        let mut wrong = config.clone();
        wrong["hidden_size"] = serde_json::json!(16);
        assert!(Eagle3Checkpoint::parse(&serde_json::to_vec(&wrong).unwrap(), &loader).is_err());
        wrong["architectures"] = serde_json::json!([]);
        assert!(Eagle3Checkpoint::parse(&serde_json::to_vec(&wrong).unwrap(), &loader).is_err());
    }
}

#[test]
fn specforge_compact_vocabulary_and_shared_embedding_match_pytorch() {
    use infer_worker::domain::ports::FusedOps;
    let g: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("specforge/golden.json")).unwrap()).unwrap();
    let scope = HostScope::new(Cpu);
    let (_, mut head) = specforge();
    head.prepare(32, 1).unwrap();
    assert_eq!(head.feature_width(), 24);
    assert_eq!(head.logits_vocab_size(), 8);
    assert_eq!(head.dims().vocab_size, 16);
    assert_eq!(
        head.token_map().unwrap().to_host_vec().unwrap(),
        ints(&g, "mapping")
    );
    let (plan, idx) = index(0, 5);
    let ctx = StepCtx::new(&scope, &plan);
    let mut projected = Tensor::zeros([5, 8], &Cpu).unwrap();
    head.project_features_into(
        &matrix(&g, "features").narrow(0, 0, 5).unwrap(),
        &mut projected,
        &ctx,
    )
    .unwrap();
    close(&projected, &matrix(&g, "projected"));
    let mut kv = pool(1);
    let ids = Tensor::from_host_slice(&ints(&g, "paired"), [5], &Cpu).unwrap();
    let mut hidden = Tensor::zeros([5, 8], &Cpu).unwrap();
    head.forward_hidden_into(
        &ids,
        &projected,
        &mut ModelCacheView::full(&mut kv, &idx),
        &mut hidden,
        &ctx,
    )
    .unwrap();
    close(&hidden, &matrix(&g, "hidden"));
    let mut logits = Tensor::zeros([5, 8], &Cpu).unwrap();
    head.project_logits_into(&hidden, &mut logits, &ctx)
        .unwrap();
    close(&logits, &matrix(&g, "logits"));
    for (t, name) in [(&kv.layers[0].k, "k"), (&kv.layers[0].v, "v")] {
        close(
            &t.narrow(0, 0, 5)
                .unwrap()
                .view_contiguous([5, 4].into())
                .unwrap(),
            &matrix(&g, name),
        );
    }
    let mut selected = Tensor::zeros([1], &Cpu).unwrap();
    Cpu::argmax_into(
        &ctx,
        &logits.narrow(0, 4, 1).unwrap(),
        &mut selected,
        &Tensor::zeros([512], &Cpu).unwrap(),
        None,
    )
    .unwrap();
    Cpu::remap_ids_inplace(&scope, &mut selected, head.token_map().unwrap()).unwrap();
    assert_eq!(selected.to_host_vec().unwrap(), ints(&g, "next_token"));
    let (plan, idx) = index(5, 1);
    let ctx = StepCtx::new(&scope, &plan);
    let mut next = Tensor::zeros([1, 8], &Cpu).unwrap();
    head.forward_hidden_into(
        &selected,
        &hidden.narrow(0, 4, 1).unwrap(),
        &mut ModelCacheView::full(&mut kv, &idx),
        &mut next,
        &ctx,
    )
    .unwrap();
    close(&next, &matrix(&g, "next_hidden"));
    let mut logits = Tensor::zeros([1, 8], &Cpu).unwrap();
    head.project_logits_into(&next, &mut logits, &ctx).unwrap();
    close(&logits, &matrix(&g, "next_logits"));
    close(
        &kv.layers[0]
            .k
            .narrow(0, 0, 6)
            .unwrap()
            .view_contiguous([6, 4].into())
            .unwrap(),
        &matrix(&g, "next_k"),
    );
    let mut bad = Tensor::from_host_slice(&[-1, 8, 0, 7], [4], &Cpu).unwrap();
    Cpu::remap_ids_inplace(&scope, &mut bad, head.token_map().unwrap()).unwrap();
    assert_eq!(bad.to_host_vec().unwrap(), [-1, -1, 0, 15]);
}

#[test]
fn specforge_sessions_match_target_greedy_across_draft_widths_and_reuse() {
    let g = golden();
    let prompt = ints(&g, "prompt");
    let expected = ints(&g, "trajectory");
    for k in [0, 1, 3, 7] {
        for chunk in [k + 1, 16] {
            let (checkpoint, head) = specforge();
            let model = target();
            let features = checkpoint.feature_spec(model.dims()).unwrap();
            let limits = SpeculativeLimits {
                max_context: 64,
                max_step_tokens: chunk,
                max_output_tokens: 20,
                draft_tokens: k,
                eos_ids: vec![],
            };
            let mut session = SpeculativeSession::with_features(
                model,
                head,
                HostScope::new(Cpu),
                limits,
                features,
            )
            .unwrap();
            for _ in 0..2 {
                session.reset().unwrap();
                let mut step = session.prefill(&prompt).unwrap();
                let mut output = step.tokens.clone();
                while !step.finished {
                    step = session.decode().unwrap();
                    output.extend_from_slice(&step.tokens);
                    assert_eq!(session.cached_tokens(), session.draft_cached_tokens() + 1);
                }
                assert_eq!(output, expected[prompt.len()..], "K={k}, chunk={chunk}");
            }
        }
    }
}

#[test]
fn specforge_rejects_corrupt_maps_before_loading_a_head() {
    let bytes = std::fs::read(fixture("specforge/draft.safetensors")).unwrap();
    let original = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let raw = std::fs::read(fixture("specforge/config.json")).unwrap();
    let path = std::env::temp_dir().join(format!(
        "rustinfer-specforge-map-{}-{}.safetensors",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let model = target();
    for (name, replacement) in [
        ("d2t", (-1_i64).to_le_bytes().to_vec()), // negative target ID
        ("d2t", 2_i64.to_le_bytes().to_vec()),    // duplicate target ID 2
        ("d2t", i64::MAX.to_le_bytes().to_vec()),
        ("t2d", vec![0]), // target ID 0 is mapped, but omitted by the mask
    ] {
        let tensor = original.tensor(name).unwrap();
        let mut data = tensor.data().to_vec();
        data[..replacement.len()].copy_from_slice(&replacement);
        let replacement =
            safetensors::tensor::TensorView::new(tensor.dtype(), tensor.shape().to_vec(), &data)
                .unwrap();
        let tensors: Vec<_> = original
            .tensors()
            .into_iter()
            .map(|(key, value)| {
                if key == name {
                    (key, replacement.clone())
                } else {
                    (key, value)
                }
            })
            .collect();
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        let reader = SafetensorsReader::open(&path).unwrap();
        let loader = WeightLoader::new(&reader);
        let checkpoint = Eagle3Checkpoint::parse(&raw, &loader).unwrap();
        let result = checkpoint.build(&loader, model.dims(), &model.embed.table, 64, &Cpu);
        assert!(result.is_err(), "corrupt {name} accepted");
    }
    std::fs::remove_file(path).unwrap();
}
