use super::*;
use infer_worker::application::speculative::prefill::MtpPrefill;
use infer_worker::components::mtp::MtpInput;
use infer_worker::domain::cache::{LinearBatch, LinearLayerState, ModelCacheView};
use infer_worker::domain::component::{Hidden, LayerRange};
use infer_worker::domain::exec::StepCtx;
use infer_worker::domain::forward_scratch::ForwardScratch;
use infer_worker::domain::gdn_scratch::GdnScratch;
use infer_worker::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use infer_worker::domain::model::DecoderReadout;
use infer_worker::domain::plan::{BatchKind, BatchPlan};
use infer_worker::domain::tensor::Tensor;
fn indices<D: infer_worker::domain::ports::backend::LlmBackend>(
    start: usize,
    n: usize,
    blocks: usize,
    device: &D,
) -> (BatchPlan, KvIndexTensors<D>) {
    let positions: Vec<i32> = (start as i32..(start + n) as i32).collect();
    let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&[n as i32]);
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
        max_blocks_per_seq: blocks,
        block_size: 1,
        total_q_tiles: req.len() as i32,
    };
    let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], device).unwrap();
    let idx = KvIndexTensors {
        block_tables: Tensor::from_host_slice(
            &(0..blocks as i32).collect::<Vec<_>>(),
            [1, blocks],
            device,
        )
        .unwrap(),
        cu_q_lens: ints(&cu),
        kv_lens: ints(&plan.kv_lens),
        seq_positions: ints(&[start as i32]),
        seq_lens_step: ints(&[n as i32]),
        rope_positions: ints(&positions),
        block2req: ints(&req),
        block2tile: ints(&tile),
        valid_q_tiles: ints(&[req.len() as i32]),
        valid_suffix_q_tiles: ints(&[req.len() as i32]),
    };
    (plan, idx)
}
fn pool<
    T: infer_worker::domain::dtype::Dtype,
    D: infer_worker::domain::ports::backend::LlmBackend,
>(
    layers: usize,
    blocks: usize,
    kv_dim: usize,
    device: &D,
) -> PagedKvPool<T, D> {
    PagedKvPool {
        layers: (0..layers)
            .map(|_| PagedKvLayer {
                k: Tensor::zeros([blocks, 1, kv_dim], device).unwrap(),
                v: Tensor::zeros([blocks, 1, kv_dim], device).unwrap(),
            })
            .collect(),
        num_blocks: blocks,
        block_size: 1,
        kv_dim,
        quant: KvQuantTier::None,
        seq_kv_len: Default::default(),
    }
}

#[test]
#[ignore = "requires QWEN35_MTP_REFERENCE, QWEN35_MODEL_PATH and an idle CUDA device"]
fn qwen35_mtp_checkpoint() {
    let path = std::env::var("QWEN35_MODEL_PATH").unwrap();
    let reference = std::path::PathBuf::from(std::env::var("QWEN35_MTP_REFERENCE").unwrap());
    let dump_dir = std::path::PathBuf::from(std::env::var("QWEN35_MTP_DUMP_DIR").unwrap());
    std::fs::create_dir_all(&dump_dir).unwrap();
    let raw = std::fs::read(Path::new(&path).join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let ids: Vec<i32> =
        serde_json::from_slice(&std::fs::read(reference.join("tokens.json")).unwrap()).unwrap();
    let n = ids.len();
    assert!(n >= 12);
    let cfg = build_load_config(&parse_hf_config(&raw).unwrap(), n + 16).unwrap();
    let reader = SafetensorsReader::open(&path).unwrap();
    let loader = WeightLoader::new(&reader);
    let cuda = Cuda::new(
        std::env::var("QWEN35_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap(),
    )
    .unwrap();
    let scope = cuda.scope();
    let mut model = qwen3_5::build::<bf16, Cuda>(&loader, &cfg, &cuda).unwrap();
    let dims = model.dims();
    let mut head = model
        .load_mtp(
            &loader,
            &cfg,
            &serde_json::from_value(json["text_config"].clone()).unwrap(),
        )
        .unwrap();
    assert_eq!(head.cache_layout().num_full_layers(), 1);
    assert!(head.cache_layout().linear_dims().is_empty());
    head.prepare(n, 1).unwrap();
    model.install_scratch(ForwardScratch::new(&cuda, dims, n, 1).unwrap());
    model
        .install_gdn_scratch(
            GdnScratch::new(&cuda, dims.dim, model.cache_layout().linear_dims()[0], n).unwrap(),
        )
        .unwrap();
    let mut target_kv = pool(
        model.cache_layout().num_full_layers(),
        n,
        dims.kv_dim,
        &cuda,
    );
    let mut states: Vec<_> = model
        .cache_layout()
        .linear_dims()
        .iter()
        .map(|&d| LinearLayerState::new(d, 1, &cuda).unwrap())
        .collect();
    let (plan, index) = indices(0, n, n, &cuda);
    let batch = LinearBatch::new(&[0], &[n as i32], 1, &cuda).unwrap();
    let mut cache = ModelCacheView::hybrid(&mut target_kv, &index, &mut states, &batch);
    let ctx = StepCtx::new(&scope, &plan);
    let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], &cuda).unwrap();
    let dump = |name: &str, values: &[bf16]| {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|x| x.to_f32().to_le_bytes())
            .collect();
        std::fs::write(dump_dir.join(format!("{name}.f32")), bytes).unwrap();
    };
    let mut residual = Hidden {
        stream: Tensor::zeros([n, dims.dim], &cuda).unwrap(),
        pending: None,
    };
    model.embed(&ints(&ids), &mut residual, &ctx).unwrap();
    model
        .decode_layers(
            LayerRange::all(dims.num_layers),
            &mut residual,
            &mut cache,
            &ctx,
        )
        .unwrap();
    let mut target = Tensor::zeros([n, dims.dim], &cuda).unwrap();
    model
        .normalize_hidden_into(&residual, &mut target, &ctx)
        .unwrap();
    dump("target", &target.to_host_vec().unwrap());
    let bytes = std::fs::read(reference.join("target.f32")).unwrap();
    let values: Vec<bf16> = bytes
        .chunks_exact(4)
        .map(|b| bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    assert_eq!(values.len(), n * dims.dim);
    let reference_hidden = Tensor::from_host_slice(&values, [n, dims.dim], &cuda).unwrap();
    for (name, conditioning, cuts) in [
        ("isolated", &reference_hidden, vec![n]),
        ("integrated", &target, vec![n]),
        ("chunked", &reference_hidden, vec![1, 2, 5, 9, n - 1, n]),
    ] {
        let mut kv = pool(1, n, dims.kv_dim, &cuda);
        let mut alignment = MtpPrefill::default();
        let (mut outputs, mut logits) = (Vec::new(), Vec::new());
        let mut start = 0;
        for end in cuts {
            let positions: Vec<i32> = (start as i32..end as i32).collect();
            let chunk = alignment
                .prepare(
                    &ids[start..end],
                    &positions,
                    &conditioning.narrow(0, start, end - start).unwrap(),
                )
                .unwrap();
            let count = chunk.next_token_ids.len();
            if count > 0 {
                let (plan, index) = indices(chunk.positions[0] as usize, count, n, &cuda);
                assert_eq!(plan.rope_positions, chunk.positions);
                let ctx = StepCtx::new(&scope, &plan);
                let mut cache = ModelCacheView::full(&mut kv, &index);
                let mut hidden = Tensor::zeros([count, dims.dim], &cuda).unwrap();
                head.forward_hidden_into(
                    MtpInput {
                        next_token_ids: &ints(&chunk.next_token_ids),
                        target_hidden: &chunk.target_hidden,
                    },
                    &mut cache,
                    &mut hidden,
                    &ctx,
                )
                .unwrap();
                let mut projected = Tensor::zeros([count, dims.vocab_size], &cuda).unwrap();
                head.project_logits_into(&hidden, &mut projected, &ctx)
                    .unwrap();
                outputs.extend(hidden.to_host_vec().unwrap());
                logits.extend(projected.to_host_vec().unwrap());
            }
            chunk.commit();
            start = end;
        }
        assert_eq!(alignment.pending().unwrap().0, n as i32 - 1);
        dump(&format!("{name}_hidden"), &outputs);
        dump(&format!("{name}_logits"), &logits);
    }
}

#[test]
#[ignore = "requires QWEN35_MODEL_PATH and an idle CUDA device"]
fn qwen35_mtp_eager_generation() {
    use infer_worker::application::runtime::Runtime;
    use infer_worker::application::sampler_stack::GreedySampler;
    use infer_worker::application::speculative::{MtpLimits, MtpSession};
    use infer_worker::domain::plan::{SeqStep, StepRequest, StopCriteria};
    let path = std::env::var("QWEN35_MODEL_PATH").unwrap();
    let raw = std::fs::read(Path::new(&path).join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let max_context = 256;
    let max_step = 64;
    let output_tokens = 32;
    let cfg = build_load_config(&parse_hf_config(&raw).unwrap(), max_context).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(&path).join("tokenizer.json")).unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap() as i32;
    let prompts = [
        "What is 2 + 2? Answer with just the number.",
        "用一句话介绍 Rust 编程语言。",
        "Write a Python function that adds two numbers. Output only code.",
    ];
    let ids: Vec<Vec<i32>> = prompts.iter().map(|prompt| {
        tokenizer.encode(format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), false)
            .unwrap().get_ids().iter().map(|&id| id as i32).collect()
    }).collect();
    let reader = SafetensorsReader::open(&path).unwrap();
    let loader = WeightLoader::new(&reader);
    let cuda = Cuda::new(
        std::env::var("QWEN35_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap(),
    )
    .unwrap();
    let baseline = {
        let model = qwen3_5::build::<bf16, Cuda>(&loader, &cfg, &cuda).unwrap();
        let blocks = max_context.div_ceil(16);
        let mut runtime = Runtime::new(
            model,
            cuda.scope(),
            Box::new(GreedySampler),
            blocks,
            16,
            blocks,
            max_context,
            max_step,
            1,
            vec![],
        )
        .unwrap();
        ids.iter()
            .map(|prompt| {
                runtime.release_sequence(0);
                let mut len = 0;
                let mut output = Vec::new();
                let mut run = |tokens: &[i32]| {
                    let req = StepRequest {
                        seqs: vec![SeqStep {
                            sequence_id: 0,
                            input_ids: tokens.to_vec(),
                            positions: (len as i32..(len + tokens.len()) as i32).collect(),
                            kv_write_start: len as i32,
                            kv_len_after: (len + tokens.len()) as i32,
                            block_table: (0..blocks as u32).collect(),
                        }],
                        sampling: vec![],
                        stop: StopCriteria {
                            eos_ids: vec![eos],
                            generated_counts: vec![0],
                            max_tokens: vec![output_tokens],
                            ignore_eos: vec![false],
                        },
                        draft_tokens: vec![],
                    };
                    let result = runtime.step(&req).unwrap();
                    len += tokens.len();
                    result.tokens[0][0].token_id
                };
                let mut next = 0;
                for chunk in prompt.chunks(max_step) {
                    next = run(chunk);
                }
                output.push(next);
                while output.len() < output_tokens as usize && next != eos {
                    next = run(&[next]);
                    output.push(next);
                }
                output
            })
            .collect::<Vec<_>>()
    };
    let mut reports = Vec::new();
    for k in [0, 1, 3] {
        let model = qwen3_5::build::<bf16, Cuda>(&loader, &cfg, &cuda).unwrap();
        let head = model
            .load_mtp(
                &loader,
                &cfg,
                &serde_json::from_value(json["text_config"].clone()).unwrap(),
            )
            .unwrap();
        let mut session = MtpSession::new(
            model,
            head,
            cuda.scope(),
            MtpLimits {
                max_context,
                max_step_tokens: max_step,
                max_output_tokens: output_tokens,
                draft_tokens: k,
                eos_ids: vec![eos],
            },
        )
        .unwrap();
        for (i, prompt) in ids.iter().enumerate() {
            session.reset().unwrap();
            let mut step = session.prefill(prompt).unwrap();
            let mut output = step.tokens.clone();
            let mut proposed = 0;
            let mut accepted = 0;
            let mut rounds = 0;
            while !step.finished {
                step = session.decode().unwrap();
                proposed += step.proposed;
                accepted += step.accepted;
                rounds += 1;
                output.extend_from_slice(&step.tokens);
                assert_eq!(session.cached_tokens(), session.draft_cached_tokens() + 1);
            }
            let text = tokenizer
                .decode(
                    &output.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                    true,
                )
                .unwrap();
            eprintln!(
                "MTP K={k} prompt={i}: {accepted}/{proposed} accepted, {rounds} rounds, output={text:?}"
            );
            reports.push(serde_json::json!({ "k": k, "prompt": prompts[i], "tokens": output,
                "baseline_tokens": baseline[i], "match": output == baseline[i], "accepted": accepted,
                "proposed": proposed, "rounds": rounds, "text": text }));
        }
    }
    if let Ok(path) = std::env::var("QWEN35_MTP_GENERATION_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
    }
    assert!(
        reports.iter().all(|r| r["match"] == true),
        "MTP generation differs from ordinary greedy; see report"
    );
}
