use super::speculative_checkpoint_support::{indices, pool};
use super::*;
use infer_worker::domain::{
    ExecScope,
    cache::ModelCacheView,
    component::{Hidden, LayerRange},
    draft::BlockDraft,
    exec::StepCtx,
    features::TargetFeatures,
    forward_scratch::ForwardScratch,
    plan::{BatchKind, MaskMode},
    ports::FusedOps,
    tensor::Tensor,
};

#[test]
#[ignore = "requires DFLASH_REFERENCE and CUDA memory for target and draft"]
fn qwen3_dflash_checkpoint() {
    let dir = std::path::PathBuf::from(std::env::var("DFLASH_REFERENCE").unwrap());
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    let target_path = meta["target"].as_str().unwrap();
    let draft_path = meta["draft"].as_str().unwrap();
    let ids = meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let n = ids.len();
    let block = meta["block_size"].as_u64().unwrap() as usize;
    let context = n + block;
    let cuda = Cuda::new(0).unwrap();
    let scope = cuda.scope();
    let cfg = build_load_config(
        &parse_hf_config(&std::fs::read(Path::new(target_path).join("config.json")).unwrap())
            .unwrap(),
        context,
    )
    .unwrap();
    let reader = SafetensorsReader::open(target_path).unwrap();
    let mut model = qwen3::build::<bf16, Cuda>(&WeightLoader::new(&reader), &cfg, &cuda).unwrap();
    let dims = model.dims();
    model.install_scratch(ForwardScratch::new(&cuda, dims, context, 1).unwrap());
    let draft_reader = SafetensorsReader::open(draft_path).unwrap();
    let cfg = qwen3::dflash::DFlashConfig::parse(
        &std::fs::read(Path::new(draft_path).join("config.json")).unwrap(),
    )
    .unwrap();
    let mut head = qwen3::dflash::build(
        &WeightLoader::new(&draft_reader),
        &cfg,
        dims,
        &model.embed.table,
        model.lm_head.proj.weight.as_dense().unwrap(),
        context,
        &cuda,
    )
    .unwrap();
    head.prepare(context).unwrap();
    let read = |name: &str| -> Vec<f32> {
        std::fs::read(dir.join(format!("{name}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    };
    let compare = |name: &str, got: &Tensor<bf16, Cuda>, limit: f64| {
        scope.synchronize().unwrap();
        let got = got.to_host_vec().unwrap();
        let expected = read(name);
        assert_eq!(got.len(), expected.len());
        let err = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a.to_f32() as f64 - *b as f64).powi(2))
            .sum::<f64>();
        let mag = expected.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        let rel = (err / mag.max(1e-30)).sqrt();
        println!("{name}: relative_l2={rel:.6}");
        assert!(rel < limit, "{name}: {rel}");
        if name == "logits" {
            let mut matched = 0;
            for (row, (a, b)) in got
                .chunks(dims.vocab_size)
                .zip(expected.chunks(dims.vocab_size))
                .enumerate()
            {
                let mut ranked = b.iter().copied().enumerate().collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                let actual = a
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
                    .unwrap()
                    .0;
                matched += usize::from(actual == ranked[0].0);
                if ranked[0].1 - ranked[1].1 > 0.5 {
                    assert_eq!(actual, ranked[0].0, "confident row {row}");
                }
            }
            println!("draft argmax matches {matched}/{}", block - 1);
        }
    };
    let mut features =
        TargetFeatures::new(cfg.feature_spec(dims).unwrap(), dims.dim, context, &cuda).unwrap();
    let mut target_kv = pool(dims.num_layers, context, dims.kv_dim, &cuda);
    let (plan, index) = indices(0, n, context, &cuda);
    let ctx = StepCtx::new(&scope, &plan);
    let mut hidden = Hidden {
        stream: Tensor::zeros([n, dims.dim], &cuda).unwrap(),
        pending: None,
    };
    let TargetFeatures::Layers(layers) = &mut features else {
        unreachable!()
    };
    model
        .embed(
            &Tensor::from_host_slice(&ids, [n], &cuda).unwrap(),
            &mut hidden,
            &ctx,
        )
        .unwrap();
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
    // Isolate draft arithmetic from target numerical variation.
    let reference = read("features")
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let input = Tensor::from_host_slice(&reference, [n, head.feature_width()], &cuda).unwrap();
    let mut kv = pool(head.dims().num_layers, context, head.dims().kv_dim, &cuda);
    head.cache_features(&input, &mut ModelCacheView::full(&mut kv, &index), &ctx)
        .unwrap();
    for (i, layer) in kv.layers.iter().enumerate() {
        compare(
            &format!("context_k_{i}"),
            &layer.k.narrow(0, 0, n).unwrap(),
            0.04,
        );
        compare(
            &format!("context_v_{i}"),
            &layer.v.narrow(0, 0, n).unwrap(),
            0.04,
        );
    }
    let (mut plan, mut ix) = indices(n, block, context, &cuda);
    plan.kind = BatchKind::Spec {
        mask: MaskMode::Full,
        mask_handle: None,
    };
    ix.decode_rows = Cuda::allocate_paged_decode_rows(&cuda, block, context).unwrap();
    Cuda::prepare_paged_attention_index(&scope, &plan, &mut ix).unwrap();
    let mut ids = vec![cfg.dflash_config.mask_token_id; block];
    ids[0] = meta["anchor"].as_i64().unwrap() as i32;
    let ids = Tensor::from_host_slice(&ids, [block], &cuda).unwrap();
    let mut logits = Tensor::zeros([block - 1, dims.vocab_size], &cuda).unwrap();
    head.forward_block(
        &ids,
        &mut ModelCacheView::full(&mut kv, &ix),
        &mut logits,
        &StepCtx::new(&scope, &plan),
    )
    .unwrap();
    compare("logits", &logits, 0.04);
}
