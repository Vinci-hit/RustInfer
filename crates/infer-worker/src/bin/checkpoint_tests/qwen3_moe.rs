use super::*;
use infer_worker::domain::cache::ModelCacheView;
use infer_worker::domain::component::{Hidden, LayerRange};
use infer_worker::domain::exec::StepCtx;
use infer_worker::domain::forward_scratch::ForwardScratch;
use infer_worker::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use infer_worker::domain::model::SampleRows;
use infer_worker::domain::plan::{BatchKind, BatchPlan};
use infer_worker::domain::tensor::Tensor;
use std::collections::HashMap;

/// Opt-in fixed-token layer/logit diagnostic; duplicate requests exercise batch invariance.
#[test]
#[ignore = "requires QWEN3_MOE_MODEL_PATH and a CUDA device with enough free memory"]
fn qwen3_moe_checkpoint_diagnostic() {
    let path = std::env::var("QWEN3_MOE_MODEL_PATH").expect("QWEN3_MOE_MODEL_PATH");
    let ordinal = std::env::var("QWEN3_MOE_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    let cfg =
        parse_hf_config(&std::fs::read(Path::new(&path).join("config.json")).unwrap()).unwrap();
    let dump_dir = std::env::var("QWEN3_MOE_DUMP_DIR").ok();
    let steps: Vec<Vec<i32>> = if let Ok(path) = std::env::var("QWEN3_MOE_INPUT_JSON") {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    } else {
        vec![vec![1, 2], vec![3]]
    };
    let capacity = steps.iter().map(Vec::len).max().unwrap();
    let total: usize = steps.iter().map(Vec::len).sum();
    let blocks = total.max(4);
    let cfg = build_load_config(&cfg, blocks.max(16)).unwrap();
    let dump = |name: String, values: Vec<bf16>| {
        if let Some(dir) = &dump_dir {
            std::fs::create_dir_all(dir).unwrap();
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| v.to_f32().to_le_bytes())
                .collect();
            std::fs::write(Path::new(dir).join(name), bytes).unwrap();
        }
    };
    let reader = SafetensorsReader::open(&path).unwrap();
    let cuda = Cuda::new(ordinal).unwrap();
    let loader = WeightLoader::new(&reader);
    let mut model = qwen3_moe::build::<bf16, Cuda>(&loader, &cfg, &cuda).unwrap();
    let dims = model.dims();
    let batch: usize = std::env::var("QWEN3_MOE_BATCH")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!(batch > 0);
    model.install_scratch(ForwardScratch::new(&cuda, dims, capacity * batch, batch).unwrap());
    let mut kv = PagedKvPool {
        layers: (0..dims.num_layers)
            .map(|_| PagedKvLayer {
                k: Tensor::zeros([blocks * batch, 1, dims.kv_dim], &cuda).unwrap(),
                v: Tensor::zeros([blocks * batch, 1, dims.kv_dim], &cuda).unwrap(),
            })
            .collect(),
        num_blocks: blocks * batch,
        block_size: 1,
        kv_dim: dims.kv_dim,
        quant: KvQuantTier::None,
        seq_kv_len: HashMap::new(),
    };
    let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], &cuda).unwrap();
    let scope = cuda.scope();
    let mut direct_tokens = Vec::new();
    let mut start = 0i32;
    for (step, ids) in steps.iter().enumerate() {
        if ids.is_empty() {
            start = 0;
            continue;
        }
        let n = ids.len();
        let positions: Vec<i32> = (0..batch).flat_map(|_| start..start + n as i32).collect();
        let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&vec![n as i32; batch]);
        let plan = BatchPlan {
            kind: if n == 1 {
                BatchKind::DecodeOnly
            } else {
                BatchKind::Ragged
            },
            num_tokens: n * batch,
            batch,
            q_lens: vec![n as i32; batch],
            kv_lens: vec![start + n as i32; batch],
            seq_positions: vec![start; batch],
            rope_positions: positions.clone(),
            max_blocks_per_seq: blocks,
            block_size: 1,
            total_q_tiles: req.len() as i32,
        };
        let index = KvIndexTensors {
            block_tables: Tensor::from_host_slice(
                &(0..(blocks * batch) as i32).collect::<Vec<_>>(),
                [batch, blocks],
                &cuda,
            )
            .unwrap(),
            cu_q_lens: ints(&cu),
            kv_lens: ints(&plan.kv_lens),
            seq_positions: ints(&vec![start; batch]),
            seq_lens_step: ints(&vec![n as i32; batch]),
            rope_positions: ints(&positions),
            block2req: ints(&req),
            block2tile: ints(&tile),
            valid_q_tiles: ints(&[req.len() as i32]),
            valid_suffix_q_tiles: ints(&[req.len() as i32]),
        };
        let mut cache = ModelCacheView::full(&mut kv, &index);
        let ctx = StepCtx::new(&scope, &plan);
        let mut hidden = Hidden {
            stream: Tensor::zeros([n * batch, dims.dim], &cuda).unwrap(),
            pending: None,
        };
        model
            .embed(&ints(&ids.repeat(batch)), &mut hidden, &ctx)
            .unwrap();
        dump(
            format!("step{step}_embed.f32"),
            hidden.stream.to_host_vec().unwrap(),
        );
        if dump_dir.is_some() {
            for layer in 0..dims.num_layers {
                model
                    .decode_layers(
                        LayerRange {
                            start: layer,
                            end: layer + 1,
                        },
                        &mut hidden,
                        &mut cache,
                        &ctx,
                    )
                    .unwrap();
                dump(
                    format!("step{step}_layer{layer}.f32"),
                    hidden.stream.to_host_vec().unwrap(),
                );
            }
        } else {
            model
                .decode_layers(
                    LayerRange {
                        start: 0,
                        end: dims.num_layers,
                    },
                    &mut hidden,
                    &mut cache,
                    &ctx,
                )
                .unwrap();
        }
        let logits_tensor = model
            .finalize(&hidden, SampleRows::LastPerSeq, &ctx)
            .unwrap()
            .0;
        let sampled =
            <Cuda as infer_worker::domain::ports::FusedOps>::argmax(&ctx, &logits_tensor).unwrap();
        if let Some(dir) = &dump_dir {
            std::fs::write(
                Path::new(dir).join(format!("step{step}_sampled.json")),
                serde_json::to_vec(&sampled).unwrap(),
            )
            .unwrap();
        }
        let all_logits = logits_tensor.to_host_vec().unwrap();
        let logits = all_logits[..dims.vocab_size].to_vec();
        for row in all_logits.chunks_exact(dims.vocab_size) {
            assert_eq!(
                row,
                logits.as_slice(),
                "identical requests differ within batch at step {step}"
            );
        }
        dump(format!("step{step}_logits.f32"), logits.clone());
        direct_tokens.push(
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.to_f32()
                        .total_cmp(&b.1.to_f32())
                        .then_with(|| b.0.cmp(&a.0))
                })
                .unwrap()
                .0 as i32,
        );
        assert_eq!(
            sampled[0],
            *direct_tokens.last().unwrap(),
            "device/host argmax mismatch at step {step}"
        );
        assert_eq!(logits.len(), dims.vocab_size);
        assert!(logits.iter().all(|v| v.to_f32().is_finite()));
        assert!(logits.windows(2).any(|v| v[0] != v[1]));
        eprintln!(
            "Qwen3 MoE checkpoint: {n} tokens at position {start}, finite logits={}",
            logits.len()
        );
        start += n as i32;
    }
}
