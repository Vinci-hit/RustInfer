use super::speculative_checkpoint_support::{indices, pool};
use super::*;
use infer_worker::domain::{
    ExecScope,
    cache::ModelCacheView,
    component::{Hidden, LayerRange},
    draft::ConditionedDraft,
    exec::StepCtx,
    features::TargetFeatures,
    forward_scratch::ForwardScratch,
    tensor::Tensor,
};

#[test]
#[ignore = "requires EAGLE3_REFERENCE and CUDA memory for target and draft weights"]
fn qwen3_eagle3_checkpoint() {
    let reference = std::path::PathBuf::from(std::env::var("EAGLE3_REFERENCE").unwrap());
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("metadata.json")).unwrap()).unwrap();
    let target_path = meta["target"].as_str().unwrap();
    let draft_path = meta["draft"].as_str().unwrap();
    let ids: Vec<i32> = meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let n = ids.len();
    let cuda = Cuda::new(0).unwrap();
    let scope = cuda.scope();
    let cfg = build_load_config(
        &parse_hf_config(&std::fs::read(Path::new(target_path).join("config.json")).unwrap())
            .unwrap(),
        n + 16,
    )
    .unwrap();
    let target_reader = SafetensorsReader::open(target_path).unwrap();
    let mut model =
        qwen3::build::<bf16, Cuda>(&WeightLoader::new(&target_reader), &cfg, &cuda).unwrap();
    let dims = model.dims();
    model.install_scratch(ForwardScratch::new(&cuda, dims, n, 1).unwrap());
    let draft_reader = SafetensorsReader::open(draft_path).unwrap();
    let loader = WeightLoader::new(&draft_reader);
    let checkpoint = qwen3::eagle3::Eagle3Checkpoint::parse(
        &std::fs::read(Path::new(draft_path).join("config.json")).unwrap(),
        &loader,
    )
    .unwrap();
    println!("Detected format: {:?}", checkpoint.format());
    let mut head = checkpoint
        .build::<bf16, Cuda>(&loader, dims, &model.embed.table, n + 16, &cuda)
        .unwrap();
    head.prepare(n, 1).unwrap();
    let read = |name: &str| -> Vec<f32> {
        std::fs::read(reference.join(format!("{name}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let compare = |name: &str, got: &Tensor<bf16, Cuda>, limit: f64| {
        scope.synchronize().unwrap();
        let actual = got.to_host_vec().unwrap();
        let expected = read(name);
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a.to_f32() as f64 - *b as f64).powi(2))
            .sum::<f64>();
        let scale = expected.iter().map(|b| (*b as f64).powi(2)).sum::<f64>();
        let rel = (error / scale.max(1e-30)).sqrt();
        println!("{name}: relative_l2={rel}");
        assert!(rel < limit, "{name} relative_l2={rel}");
        if name == "logits" {
            let v = head.logits_vocab_size();
            let mut matched = 0;
            for (i, (a, b)) in actual.chunks(v).zip(expected.chunks(v)).enumerate() {
                let best = |x: &[f32]| {
                    x.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                let actual: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                let ai = best(&actual);
                let bi = best(b);
                matched += usize::from(ai == bi);
                let second = b
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != bi)
                    .map(|(_, v)| *v)
                    .fold(f32::NEG_INFINITY, f32::max);
                if b[bi] - second > 0.5 {
                    assert_eq!(ai, bi, "confident argmax row {i}");
                }
            }
            println!("draft argmax matches {matched}/{}", expected.len() / v);
        }
    };
    let mut features =
        TargetFeatures::new(checkpoint.feature_spec(dims).unwrap(), dims.dim, n, &cuda).unwrap();
    let (plan, index) = indices(0, n, n + 16, &cuda);
    let ctx = StepCtx::new(&scope, &plan);
    let mut target_kv = pool(dims.num_layers, n + 16, dims.kv_dim, &cuda);
    let mut hidden = Hidden {
        stream: Tensor::zeros([n, dims.dim], &cuda).unwrap(),
        pending: None,
    };
    let tokens = Tensor::from_host_slice(&ids, [n], &cuda).unwrap();
    let TargetFeatures::Layers(layers) = &mut features else {
        unreachable!()
    };
    model.embed(&tokens, &mut hidden, &ctx).unwrap();
    model
        .decode_layers_observed(
            LayerRange::all(dims.num_layers),
            &mut hidden,
            &mut ModelCacheView::full(&mut target_kv, &index),
            &ctx,
            layers,
        )
        .unwrap();
    layers.concatenate(n, &scope).unwrap();
    compare("features", &features.rows(n).unwrap(), 0.04);
    // Isolate the draft computation from target backend numerical variation.
    let input: Vec<bf16> = read("features").into_iter().map(bf16::from_f32).collect();
    let input = Tensor::from_host_slice(&input, [n, head.feature_width()], &cuda).unwrap();
    let (plan, index) = indices(0, n - 1, n + 16, &cuda);
    let ctx = StepCtx::new(&scope, &plan);
    let mut draft_kv = pool(1, n + 16, head.dims().kv_dim, &cuda);
    let mut projected = Tensor::zeros([n - 1, dims.dim], &cuda).unwrap();
    head.project_features_into(&input.narrow(0, 0, n - 1).unwrap(), &mut projected, &ctx)
        .unwrap();
    compare("projected", &projected, 0.02);
    let tokens = Tensor::from_host_slice(&ids[1..], [n - 1], &cuda).unwrap();
    let mut hidden = Tensor::zeros([n - 1, dims.dim], &cuda).unwrap();
    head.forward_hidden_into(
        &tokens,
        &projected,
        &mut ModelCacheView::full(&mut draft_kv, &index),
        &mut hidden,
        &ctx,
    )
    .unwrap();
    compare("hidden", &hidden, 0.04);
    let mut logits = Tensor::zeros([n - 1, head.logits_vocab_size()], &cuda).unwrap();
    head.project_logits_into(&hidden, &mut logits, &ctx)
        .unwrap();
    compare("logits", &logits, 0.04);
    if let Some(map) = head.token_map() {
        use infer_worker::domain::ports::FusedOps;
        let mut ids = Tensor::from_host_slice(
            &[0, (map.numel() - 1) as i32, -1, map.numel() as i32],
            [4],
            &cuda,
        )
        .unwrap();
        let expected = map.to_host_vec().unwrap();
        Cuda::remap_ids_inplace(&scope, &mut ids, map).unwrap();
        scope.synchronize().unwrap();
        assert_eq!(
            ids.to_host_vec().unwrap(),
            [expected[0], *expected.last().unwrap(), -1, -1]
        );
    }
}

/// Diagnoses batch-dependent target numerics without loading any draft weights.
/// Input JSON contains target, prompt, prefix (already generated), and suffix.
#[test]
#[ignore = "requires EAGLE3_BATCH_PROBE JSON and CUDA target weights"]
fn qwen3_target_batch_probe() {
    use infer_worker::domain::model::SampleRows;
    let path = std::env::var("EAGLE3_BATCH_PROBE").unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let ids = |name: &str| -> Vec<i32> {
        meta[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect()
    };
    let prompt = ids("prompt");
    let prefix = ids("prefix");
    let suffix = ids("suffix");
    assert!(!prefix.is_empty() && suffix.len() >= 7);
    let cuda = Cuda::new(0).unwrap();
    let scope = cuda.scope();
    let target_path = meta["target"].as_str().unwrap();
    let context = prompt.len() + prefix.len() + 16;
    let cfg = build_load_config(
        &parse_hf_config(&std::fs::read(Path::new(target_path).join("config.json")).unwrap())
            .unwrap(),
        context,
    )
    .unwrap();
    let reader = SafetensorsReader::open(target_path).unwrap();
    let mut model = qwen3::build::<bf16, Cuda>(&WeightLoader::new(&reader), &cfg, &cuda).unwrap();
    let dims = model.dims();
    model.install_scratch(ForwardScratch::new(&cuda, dims, 128, 1).unwrap());
    let mut kv = pool(dims.num_layers, context, dims.kv_dim, &cuda);
    let mut features = TargetFeatures::new(
        infer_worker::domain::features::FeatureSpec::DecoderLayers(vec![1, 9, 17, 25, 33]),
        dims.dim,
        128,
        &cuda,
    )
    .unwrap();
    let mut forward = |start: usize, tokens: &[i32], observe: bool| {
        let n = tokens.len();
        let (plan, index) = indices(start, n, context, &cuda);
        let ctx = StepCtx::new(&scope, &plan);
        let tokens = Tensor::from_host_slice(tokens, [n], &cuda).unwrap();
        let mut hidden = Hidden {
            stream: Tensor::zeros([n, dims.dim], &cuda).unwrap(),
            pending: None,
        };
        model.embed(&tokens, &mut hidden, &ctx).unwrap();
        let mut cache = ModelCacheView::full(&mut kv, &index);
        if observe {
            let TargetFeatures::Layers(layers) = &mut features else {
                unreachable!()
            };
            model
                .decode_layers_observed(
                    LayerRange::all(dims.num_layers),
                    &mut hidden,
                    &mut cache,
                    &ctx,
                    layers,
                )
                .unwrap();
        } else {
            model
                .decode_layers(
                    LayerRange::all(dims.num_layers),
                    &mut hidden,
                    &mut cache,
                    &ctx,
                )
                .unwrap();
        }
        let logits = model.finalize(&hidden, SampleRows::All, &ctx).unwrap().0;
        scope.synchronize().unwrap();
        logits.to_host_vec().unwrap()
    };
    let mut start = 0;
    for chunk in prompt.chunks(32) {
        forward(start, chunk, false);
        start += chunk.len();
    }
    for &token in &prefix[..prefix.len() - 1] {
        forward(start, &[token], false);
        start += 1;
    }
    let mut tape = vec![*prefix.last().unwrap()];
    tape.extend_from_slice(&suffix);
    let reference = forward(start, &tape[..1], false);
    for n in [1, 2, 4, 8] {
        let logits = forward(start, &tape[..n], false);
        let observed = forward(start, &tape[..n], true);
        assert_eq!(
            logits, observed,
            "observation changed target output at batch {n}"
        );
        let mut ranked: Vec<_> = logits[..dims.vocab_size]
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32()))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let max_delta = logits
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0f32, f32::max);
        println!(
            "query_rows={n} top5={:?} max_logit_delta={max_delta}",
            &ranked[..5]
        );
    }
    for stride in [1, 2, 4, 8] {
        let mut start = 0;
        for chunk in prompt.chunks(32) {
            forward(start, chunk, false);
            start += chunk.len();
        }
        let mut logits = Vec::new();
        for chunk in prefix.chunks(stride) {
            let rows = forward(start, chunk, false);
            logits = rows[rows.len() - dims.vocab_size..].to_vec();
            start += chunk.len();
        }
        let mut ranked: Vec<_> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32()))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("history_stride={stride} top5={:?}", &ranked[..5]);
    }
}
