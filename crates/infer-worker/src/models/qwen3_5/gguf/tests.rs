use super::*;
use crate::components::{embed::EmbeddingWeight, linear::LinearWeight};
use crate::domain::{
    cache::{LayerCacheId, LinearBatch, LinearLayerState, ModelCacheView},
    component::{Hidden, LayerRange},
    exec::{HostScope, StepCtx},
    kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool},
    model::{DecoderModel, SampleRows},
    plan::{BatchKind, BatchPlan},
};
use crate::infrastructure::io::{
    SafetensorsReader,
    gguf::{GgufArray, GgufValue},
};
use crate::models::loader::{LinearAttnConfig, LoadConfig, WeightLoader};
use infer_backend_cpu::{Cpu, block_quant::decode_row};
use std::collections::{BTreeMap, HashMap};

struct Entry {
    name: String,
    dims: Vec<u64>,
    kind: GgmlType,
    bytes: Vec<u8>,
}
struct Fixture {
    meta: BTreeMap<String, GgufValue>,
    tensors: Vec<Entry>,
}

impl Fixture {
    fn new() -> Self {
        let mut f = Self {
            meta: BTreeMap::new(),
            tensors: Vec::new(),
        };
        f.meta.insert(
            "general.architecture".into(),
            GgufValue::String("qwen35".into()),
        );
        for (key, n) in [
            ("block_count", 3),
            ("nextn_predict_layers", 1),
            ("context_length", 32),
            ("embedding_length", 256),
            ("feed_forward_length", 256),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 64),
            ("attention.value_length", 64),
            ("rope.dimension_count", 8),
            ("ssm.group_count", 2),
            ("ssm.time_step_rank", 6),
            ("ssm.inner_size", 96),
            ("ssm.state_size", 16),
            ("ssm.conv_kernel", 3),
            ("full_attention_interval", 2),
        ] {
            f.meta.insert(format!("qwen35.{key}"), GgufValue::U32(n));
        }
        f.meta.insert(
            "qwen35.rope.dimension_sections".into(),
            GgufValue::Array(GgufArray::I32(vec![1, 1, 2, 0])),
        );
        f.meta
            .insert("qwen35.rope.freq_base".into(), GgufValue::F32(10000.0));
        f.meta.insert(
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(1e-6),
        );
        f.matrix("token_embd.weight", 32, 256, GgmlType::Q8_0);
        f.matrix("output.weight", 32, 256, GgmlType::Q8_0);
        f.dense("output_norm.weight", &[256], true);
        for i in 0..2 {
            for s in ["attn_norm.weight", "post_attention_norm.weight"] {
                f.dense(&format!("blk.{i}.{s}"), &[256], true);
            }
            f.matrix(
                &format!("blk.{i}.ffn_gate.weight"),
                256,
                256,
                GgmlType::Q8_0,
            );
            f.matrix(
                &format!("blk.{i}.ffn_up.weight"),
                256,
                256,
                GgmlType::IQ4_NL,
            );
            f.matrix(
                &format!("blk.{i}.ffn_down.weight"),
                256,
                256,
                GgmlType::Q8_0,
            );
        }
        for (s, n, k) in [
            ("attn_qkv.weight", 160, 256),
            ("attn_gate.weight", 96, 256),
            ("ssm_alpha.weight", 6, 256),
            ("ssm_beta.weight", 6, 256),
            ("ssm_out.weight", 256, 96),
        ] {
            f.matrix(&format!("blk.0.{s}"), n, k, GgmlType::Q8_0);
        }
        for (s, dims) in [
            ("ssm_conv1d.weight", vec![3, 160]),
            ("ssm_dt.bias", vec![6]),
            ("ssm_norm.weight", vec![16]),
        ] {
            f.dense(&format!("blk.0.{s}"), &dims, false);
        }
        f.tensors.push(Entry {
            name: "blk.0.ssm_a".into(),
            dims: vec![6],
            kind: GgmlType::F32,
            bytes: (0..6)
                .flat_map(|i| (-((i as f32 + 1.0) * 0.2).exp()).to_le_bytes())
                .collect(),
        });
        for (s, n, k, t) in [
            ("attn_q.weight", 256, 256, GgmlType::IQ4_NL),
            ("attn_k.weight", 64, 256, GgmlType::Q8_0),
            ("attn_v.weight", 64, 256, GgmlType::Q8_0),
            ("attn_output.weight", 256, 128, GgmlType::Q8_0),
        ] {
            f.matrix(&format!("blk.1.{s}"), n, k, t);
        }
        for s in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            f.dense(&format!("blk.1.{s}"), &[64], true);
        }
        f.dense("blk.2.nextn.enorm.weight", &[256], true);
        f
    }
    fn dense(&mut self, name: &str, dims: &[u64], norm: bool) {
        self.tensors.push(Entry {
            name: name.into(),
            dims: dims.to_vec(),
            kind: GgmlType::F32,
            bytes: (0..dims.iter().product::<u64>())
                .flat_map(|i| {
                    let x = if norm {
                        1.0 + (i % 7) as f32 / 256.0
                    } else {
                        ((i * 7 % 23) as f32 - 11.0) / 32.0
                    };
                    x.to_le_bytes()
                })
                .collect(),
        });
    }
    fn matrix(&mut self, name: &str, n: usize, k: usize, kind: GgmlType) {
        let mut bytes = Vec::new();
        let seed = self.tensors.len();
        for b in 0..n * k / 32 {
            bytes.extend(
                half::f16::from_f32(if kind == GgmlType::Q8_0 {
                    1.0 / 128.0
                } else {
                    1.0 / 4096.0
                })
                .to_le_bytes(),
            );
            for i in 0..if kind == GgmlType::Q8_0 { 32 } else { 16 } {
                bytes.push(if kind == GgmlType::Q8_0 {
                    (((b * 3 + i * 7 + seed) % 11) as i8 - 5) as u8
                } else {
                    ((b * 17 + i * 13 + seed) % 256) as u8
                });
            }
        }
        self.tensors.push(Entry {
            name: name.into(),
            dims: vec![k as u64, n as u64],
            kind,
            bytes,
        });
    }
    fn reader(&self) -> GgufReader {
        fn string(b: &mut Vec<u8>, s: &str) {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        }
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((self.tensors.len() as u64).to_le_bytes());
        b.extend((self.meta.len() as u64).to_le_bytes());
        for (k, v) in &self.meta {
            string(&mut b, k);
            match v {
                GgufValue::String(v) => {
                    b.extend(8u32.to_le_bytes());
                    string(&mut b, v);
                }
                GgufValue::U32(v) => {
                    b.extend(4u32.to_le_bytes());
                    b.extend(v.to_le_bytes());
                }
                GgufValue::F32(v) => {
                    b.extend(6u32.to_le_bytes());
                    b.extend(v.to_le_bytes());
                }
                GgufValue::Array(GgufArray::I32(v)) => {
                    b.extend(9u32.to_le_bytes());
                    b.extend(5u32.to_le_bytes());
                    b.extend((v.len() as u64).to_le_bytes());
                    for x in v {
                        b.extend(x.to_le_bytes());
                    }
                }
                GgufValue::Array(GgufArray::Bool(v)) => {
                    b.extend(9u32.to_le_bytes());
                    b.extend(7u32.to_le_bytes());
                    b.extend((v.len() as u64).to_le_bytes());
                    b.extend(v.iter().map(|&v| u8::from(v)));
                }
                _ => panic!("unsupported fixture metadata"),
            }
        }
        let mut offset = 0u64;
        for t in &self.tensors {
            string(&mut b, &t.name);
            b.extend((t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                b.extend(d.to_le_bytes());
            }
            b.extend(t.kind.id().to_le_bytes());
            b.extend(offset.to_le_bytes());
            offset += (t.bytes.len() as u64).div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        for t in &self.tensors {
            b.extend(&t.bytes);
            b.resize(b.len().div_ceil(32) * 32, 0);
        }
        let p = std::env::temp_dir().join(format!(
            "rustinfer-loader-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&p, b).unwrap();
        let r = GgufReader::open(&p).unwrap();
        std::fs::remove_file(p).unwrap();
        r
    }
}
fn options() -> LoadOptions {
    LoadOptions { context_length: 16 }
}

#[test]
fn builds_owned_mixed_model_and_preserves_norm_precision() {
    let r = Fixture::new().reader();
    let l = Qwen35GgufLoader::new(&r, options()).unwrap();
    assert_eq!(l.report().loaded_tensors, 28);
    assert_eq!(l.report().skipped_mtp_tensors.len(), 1);
    assert_eq!(l.config().layer_is_full, [false, true]);
    assert_eq!(l.config().dims.num_layers, 2);
    assert_eq!(l.config().mtp_layers, 1);
    let model = l.load::<half::bf16, _>(&Cpu).unwrap();
    assert!(model.decoder.norm.zero_centered);
    assert_eq!(
        model.decoder.norm.weight.to_host_vec().unwrap()[1].to_f32(),
        1.0 / 256.0
    );
    assert_eq!(
        model.cache_layout().layers(),
        [LayerCacheId::Linear(0), LayerCacheId::Full(0)]
    );
    let EmbeddingWeight::BlockQuant(w) = model.decoder.embed.weight() else {
        panic!()
    };
    assert_eq!(
        w.bytes().to_host_vec().unwrap(),
        r.read_view("token_embd.weight").unwrap().bytes
    );
    assert!(
        l.device_weight_bytes::<half::bf16>().unwrap() < l.device_weight_bytes::<f32>().unwrap()
    );
    assert!(l.load::<i32, _>(&Cpu).is_err());
    drop(l);
    drop(r);
    assert_eq!(w.layout().shape(), [32, 256]);
    assert!(w.bytes().to_host_vec().is_ok());
}

#[test]
fn absent_output_ties_encoded_storage_and_mtp_is_not_inferred_as_main() {
    let mut f = Fixture::new();
    f.tensors.retain(|t| t.name != "output.weight");
    let r = f.reader();
    let l = Qwen35GgufLoader::new(&r, options()).unwrap();
    assert!(l.report().tied_output);
    let m = l.load::<f32, _>(&Cpu).unwrap();
    let EmbeddingWeight::BlockQuant(w) = m.decoder.embed.weight() else {
        panic!()
    };
    let LinearWeight::BlockQuant(p) = &m.decoder.lm_head.proj.weight else {
        panic!()
    };
    assert!(std::sync::Arc::ptr_eq(
        w.bytes().storage(),
        p.parts().next().unwrap().1.bytes().storage()
    ));
    f.meta.remove("qwen35.nextn_predict_layers");
    assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
}

#[test]
fn rejects_bad_metadata_shapes_missing_extra_and_transformed_values() {
    for (key, value) in [
        (
            "general.architecture",
            GgufValue::String("qwen35moe".into()),
        ),
        ("qwen35.nextn_predict_layers", GgufValue::U32(3)),
        ("qwen35.ssm.inner_size", GgufValue::U32(95)),
        ("qwen35.full_attention_interval", GgufValue::U32(0)),
        ("qwen35.rope.dimension_count", GgufValue::U32(9)),
        (
            "qwen35.attention.layer_norm_rms_epsilon",
            GgufValue::F32(f32::NAN),
        ),
        ("qwen35.rope.scaling.type", GgufValue::String("yarn".into())),
        ("qwen35.attention.head_count", GgufValue::String("2".into())),
    ] {
        let mut f = Fixture::new();
        f.meta.insert(key.into(), value);
        assert!(
            Qwen35GgufLoader::new(&f.reader(), options()).is_err(),
            "{key}"
        );
    }
    let r = Fixture::new().reader();
    for context_length in [0, 33] {
        assert!(Qwen35GgufLoader::new(&r, LoadOptions { context_length }).is_err());
    }
    let mut f = Fixture::new();
    f.tensors.retain(|t| t.name != "blk.1.attn_k.weight");
    assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
    let mut f = Fixture::new();
    f.dense("unexpected.weight", &[1], false);
    assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
    let mut f = Fixture::new();
    f.tensors
        .iter_mut()
        .find(|t| t.name == "blk.1.attn_q_norm.weight")
        .unwrap()
        .dims = vec![8, 4];
    assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
    for x in [0.0f32, 1.0, f32::NEG_INFINITY] {
        let mut f = Fixture::new();
        let t = f
            .tensors
            .iter_mut()
            .find(|t| t.name == "blk.0.ssm_a")
            .unwrap();
        t.bytes[..4].copy_from_slice(&x.to_le_bytes());
        assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
    }
}

#[test]
fn explicit_recurrent_metadata_overrides_interval_and_rejects_mtp_recurrence() {
    let mut f = Fixture::new();
    f.meta
        .insert("qwen35.full_attention_interval".into(), GgufValue::U32(1));
    f.meta.insert(
        "qwen35.attention.recurrent_layers".into(),
        GgufValue::Array(GgufArray::Bool(vec![true, false, false])),
    );
    assert_eq!(
        Qwen35GgufLoader::new(&f.reader(), options())
            .unwrap()
            .config()
            .layer_is_full,
        [false, true]
    );
    f.meta.insert(
        "qwen35.attention.recurrent_layers".into(),
        GgufValue::Array(GgufArray::Bool(vec![true, false, true])),
    );
    assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
}

/// Independent dense HF-layout checkpoint: undo the converter's tiled V order
/// in weights, unlike the GGUF execution path which tiles Q/K activations.
fn dense_hf_reference(
    r: &GgufReader,
    cfg: &ModelConfig,
    restore_tiled: bool,
) -> Qwen3_5Model<f32, Cpu> {
    let mut tensors = BTreeMap::new();
    for t in r.tensors() {
        let name = t.name();
        if name.starts_with("blk.2.") {
            continue;
        }
        let mut shape: Vec<usize> = t.dimensions().iter().rev().map(|&d| d as usize).collect();
        let v = r.read_view(name).unwrap();
        let mut x = if t.ggml_type().block_quant_format().is_some() {
            let v = crate::models::gguf_weights::block_quant_view(v).unwrap();
            let [n, k] = v.layout().shape();
            let mut x = vec![0.0; n * k];
            for row in 0..n {
                decode_row(v, row, &mut x[row * k..(row + 1) * k]).unwrap();
            }
            x
        } else {
            v.bytes
                .chunks_exact(if t.ggml_type() == GgmlType::F32 { 4 } else { 2 })
                .map(|b| match t.ggml_type() {
                    GgmlType::F32 => f32::from_le_bytes(b.try_into().unwrap()),
                    GgmlType::F16 => half::f16::from_le_bytes(b.try_into().unwrap()).to_f32(),
                    GgmlType::BF16 => half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32(),
                    _ => panic!(),
                })
                .collect()
        };
        let dest = match name {
            "token_embd.weight" => "model.language_model.embed_tokens.weight".to_string(),
            "output.weight" => "lm_head.weight".to_string(),
            "output_norm.weight" => "model.language_model.norm.weight".to_string(),
            _ => {
                let mut p = name.splitn(3, '.');
                p.next();
                let layer = p.next().unwrap();
                let suffix = p.next().unwrap();
                let hf = match suffix {
                    "attn_norm.weight" => "input_layernorm.weight",
                    "post_attention_norm.weight" => "post_attention_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
                    "attn_gate.weight" => "linear_attn.in_proj_z.weight",
                    "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
                    "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
                    "ssm_out.weight" => "linear_attn.out_proj.weight",
                    "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
                    "ssm_a" => "linear_attn.A_log",
                    "ssm_dt.bias" => "linear_attn.dt_bias",
                    "ssm_norm.weight" => "linear_attn.norm.weight",
                    _ => panic!("{suffix}"),
                };
                format!("model.language_model.layers.{layer}.{hf}")
            }
        };
        if dest.ends_with("norm.weight") && !dest.ends_with("linear_attn.norm.weight") {
            for v in &mut x {
                *v -= 1.0;
            }
        }
        if name.ends_with("ssm_a") {
            for v in &mut x {
                *v = (-*v).ln();
            }
        }
        let rows = if name.ends_with("attn_qkv.weight") || name.ends_with("ssm_conv1d.weight") {
            Some((64, 16))
        } else if name.ends_with("attn_gate.weight") {
            Some((0, 16))
        } else if name.ends_with("ssm_alpha.weight")
            || name.ends_with("ssm_beta.weight")
            || name.ends_with("ssm_a")
            || name.ends_with("ssm_dt.bias")
        {
            Some((0, 1))
        } else {
            None
        };
        if let Some((offset, width)) = rows.filter(|_| restore_tiled) {
            let cols = if shape.len() == 1 { 1 } else { shape[1] };
            let before = x.clone();
            for g in 0..2 {
                for sub in 0..3 {
                    for d in 0..width {
                        let to = offset + (g * 3 + sub) * width + d;
                        let from = offset + (sub * 2 + g) * width + d;
                        x[to * cols..(to + 1) * cols]
                            .copy_from_slice(&before[from * cols..(from + 1) * cols]);
                    }
                }
            }
        }
        if restore_tiled && name.ends_with("ssm_out.weight") {
            let before = x.clone();
            for row in 0..256 {
                for g in 0..2 {
                    for sub in 0..3 {
                        for d in 0..16 {
                            x[row * 96 + (g * 3 + sub) * 16 + d] =
                                before[row * 96 + (sub * 2 + g) * 16 + d];
                        }
                    }
                }
            }
        }
        if name.ends_with("ssm_conv1d.weight") {
            shape.insert(1, 1);
        }
        tensors.insert(
            dest,
            (
                shape,
                x.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
            ),
        );
    }
    let views: BTreeMap<_, _> = tensors
        .iter()
        .map(|(n, (s, b))| {
            (
                n.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                    .unwrap(),
            )
        })
        .collect();
    let path = std::env::temp_dir().join(format!(
        "rustinfer-gguf-dense-{}-{:?}.safetensors",
        std::process::id(),
        std::thread::current().id()
    ));
    safetensors::tensor::serialize_to_file(views, None, &path).unwrap();
    let reader = SafetensorsReader::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let d = cfg.dims;
    let l = cfg.linear;
    let c = LoadConfig {
        dim: d.dim,
        intermediate_size: d.intermediate_size,
        layer_num: d.num_layers,
        head_num: d.head_num,
        kv_head_num: d.kv_head_num,
        head_dim: d.head_dim,
        vocab_size: d.vocab_size,
        seq_len: cfg.context_length,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        rope_scaling: None,
        mlp_quant: None,
        fp8_block: None,
        rotary_dim: cfg.rotary_dim,
        attn_output_gate: true,
        linear_attn: Some(LinearAttnConfig {
            num_key_heads: l.num_key_heads,
            num_value_heads: l.num_value_heads,
            key_head_dim: l.key_head_dim,
            value_head_dim: l.value_head_dim,
            conv_kernel_dim: l.conv_kernel_dim,
            layer_is_full: cfg.layer_is_full.clone(),
        }),
        num_experts: 0,
        experts_per_tok: 0,
        moe_intermediate_size: 0,
        norm_topk_prob: false,
        decoder_sparse_step: 1,
    };
    super::super::build(&WeightLoader::new(&reader), &c, &Cpu).unwrap()
}

fn run_cpu(model: &Qwen3_5Model<f32, Cpu>) -> Vec<f32> {
    run_model(model, &Cpu, &HostScope::new(Cpu))
}

fn run_model<T: Dtype, D: LlmBackend + crate::domain::ports::OpBackend>(
    model: &Qwen3_5Model<T, D>,
    device: &D,
    scope: &D::Scope,
) -> Vec<f32> {
    let d = model.dims();
    let mut kv = PagedKvPool {
        layers: (0..model.cache_layout().num_full_layers())
            .map(|_| PagedKvLayer {
                k: Tensor::zeros([32, 1, d.kv_dim], device).unwrap(),
                v: Tensor::zeros([32, 1, d.kv_dim], device).unwrap(),
            })
            .collect(),
        num_blocks: 32,
        block_size: 1,
        kv_dim: d.kv_dim,
        quant: KvQuantTier::None,
        seq_kv_len: HashMap::new(),
    };
    let mut states: Vec<_> = model
        .cache_layout()
        .linear_dims()
        .iter()
        .map(|&l| LinearLayerState::new(l, 2, device).unwrap())
        .collect();
    let ints = |x: &[i32]| Tensor::from_host_slice(x, [x.len()], device).unwrap();
    let mut outputs = Vec::new();
    for (q_lens, positions, ids) in [
        (vec![2, 1], vec![0, 0], vec![3, 5, 7]),
        (vec![1, 1], vec![2, 1], vec![11, 13]),
    ] {
        let kv_lens: Vec<_> = q_lens.iter().zip(&positions).map(|(q, p)| q + p).collect();
        let rope_positions: Vec<_> = positions
            .iter()
            .zip(&q_lens)
            .flat_map(|(&p, &q)| p..p + q)
            .collect();
        let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&q_lens);
        let plan = BatchPlan {
            kind: if ids.len() == 2 {
                BatchKind::DecodeOnly
            } else {
                BatchKind::Ragged
            },
            num_tokens: ids.len(),
            batch: 2,
            q_lens: q_lens.clone(),
            kv_lens: kv_lens.clone(),
            seq_positions: positions.clone(),
            rope_positions: rope_positions.clone(),
            max_blocks_per_seq: 16,
            block_size: 1,
            total_q_tiles: req.len() as i32,
        };
        let table: Vec<_> = (16..32).chain(0..16).collect();
        let index = KvIndexTensors {
            decode_rows: None,
            block_tables: Tensor::from_host_slice(&table, [2, 16], device).unwrap(),
            cu_q_lens: ints(&cu),
            kv_lens: ints(&kv_lens),
            seq_positions: ints(&positions),
            seq_lens_step: ints(&q_lens),
            rope_positions: ints(&rope_positions),
            block2req: ints(&req),
            block2tile: ints(&tile),
            valid_q_tiles: ints(&[req.len() as i32]),
            valid_suffix_q_tiles: ints(&[req.len() as i32]),
        };
        let linear = LinearBatch::new(&[1, 0], &q_lens, 2, device).unwrap();
        let mut cache = ModelCacheView::hybrid(&mut kv, &index, &mut states, &linear);
        let ctx = StepCtx::new(scope, &plan);
        let mut hidden = Hidden {
            stream: Tensor::zeros([ids.len(), d.dim], device).unwrap(),
            pending: None,
        };
        model.embed(&ints(&ids), &mut hidden, &ctx).unwrap();
        model
            .decode_layers(
                LayerRange {
                    start: 0,
                    end: d.num_layers,
                },
                &mut hidden,
                &mut cache,
                &ctx,
            )
            .unwrap();
        outputs.extend(
            model
                .finalize(&hidden, SampleRows::All, &ctx)
                .unwrap()
                .0
                .to_host_vec()
                .unwrap()
                .iter()
                .map(|x| T::read_f64(x) as f32),
        );
    }
    outputs
}

#[test]
fn mixed_quantized_forward_matches_dense_hf_weights_for_prefill_and_stateful_decode() {
    let r = Fixture::new().reader();
    let l = Qwen35GgufLoader::new(&r, options()).unwrap();
    let dense = dense_hf_reference(&r, l.config(), true);
    let model = l.load::<f32, _>(&Cpu).unwrap();
    let actual = run_cpu(&model);
    let expected = run_cpu(&dense);
    assert_eq!(actual.len(), 160);
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() < 2e-4 + 2e-4 * b.abs(),
            "logit {i}: {a} vs {b}"
        );
    }
    // Head order must matter: a deliberately wrong grouped interpretation
    // should not accidentally pass because of constant/zero fixture weights.
    let grouped = dense_hf_reference(&r, l.config(), false);
    let wrong = run_cpu(&grouped);
    assert!(
        wrong
            .iter()
            .zip(&expected)
            .any(|(a, b)| (a - b).abs() > 1e-3)
    );
    // Explicitly test the reference has a nontrivial signal.
    assert!(
        expected.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - expected.iter().copied().fold(f32::INFINITY, f32::min)
            > 0.1
    );
    drop(grouped);
}

#[test]
fn dense_f32_f16_bf16_weights_and_mixed_storage_rejection() {
    for kind in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
        let mut f = Fixture::new();
        let original = f.reader();
        for t in &mut f.tensors {
            if t.kind.block_quant_format().is_none() {
                continue;
            }
            let v =
                crate::models::gguf_weights::block_quant_view(original.read_view(&t.name).unwrap())
                    .unwrap();
            let [n, k] = v.layout().shape();
            let mut x = vec![0.0; n * k];
            for row in 0..n {
                decode_row(v, row, &mut x[row * k..(row + 1) * k]).unwrap();
            }
            t.kind = kind;
            t.bytes = x
                .into_iter()
                .flat_map(|x| match kind {
                    GgmlType::F32 => x.to_le_bytes().to_vec(),
                    GgmlType::F16 => half::f16::from_f32(x).to_le_bytes().to_vec(),
                    GgmlType::BF16 => half::bf16::from_f32(x).to_le_bytes().to_vec(),
                    _ => unreachable!(),
                })
                .collect();
        }
        let r = f.reader();
        let loader = Qwen35GgufLoader::new(&r, options()).unwrap();
        let model = loader.load::<f32, _>(&Cpu).unwrap();
        let reference = dense_hf_reference(&r, loader.config(), true);
        for (a, b) in run_cpu(&model).iter().zip(run_cpu(&reference)) {
            assert!((a - b).abs() < 2e-4);
        }
        // One dense and one encoded MLP part cannot be silently misinterpreted.
        f.tensors.retain(|t| t.name != "blk.0.ffn_up.weight");
        f.matrix("blk.0.ffn_up.weight", 256, 256, GgmlType::Q8_0);
        assert!(Qwen35GgufLoader::new(&f.reader(), options()).is_err());
    }
}

#[cfg(feature = "cute-dsl")]
#[test]
#[ignore = "requires CUDA sm89 + CuTe DSL"]
fn cuda_tiny_loaded_model_matches_cpu_prefill_and_decode() {
    let r = Fixture::new().reader();
    let l = Qwen35GgufLoader::new(&r, options()).unwrap();
    let cpu = l.load::<half::bf16, _>(&Cpu).unwrap();
    let expected = run_model(&cpu, &Cpu, &HostScope::new(Cpu));
    let dev = infer_backend_cuda::Cuda::with_memory_plan(
        0,
        infer_backend_cuda::CudaMemoryPlan {
            kernel_workspace_bytes: 1 << 20,
            graph_arena_bytes: 1 << 20,
            pool_retain_bytes: 4 << 20,
        },
    )
    .unwrap();
    assert!(dev.config.cute_dsl_available());
    let gpu = l.load::<half::bf16, _>(&dev).unwrap();
    let actual = run_model(&gpu, &dev, &dev.scope());
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() < 0.015 + 0.02 * b.abs(),
            "CUDA logit {i}: {a} vs {b}"
        );
    }
}

#[cfg(feature = "cute-dsl")]
#[test]
#[ignore = "requires 16 GiB CUDA GPU and RUSTINFER_GGUF_MODEL"]
fn cuda_real_model_load_and_embedding() {
    use infer_core::exec::ExecScope;
    let r =
        GgufReader::open(std::env::var_os("RUSTINFER_GGUF_MODEL").expect("RUSTINFER_GGUF_MODEL"))
            .unwrap();
    let l = Qwen35GgufLoader::new(
        &r,
        LoadOptions {
            context_length: 128,
        },
    )
    .unwrap();
    assert_eq!(l.config().dims.num_layers, 64);
    assert_eq!(l.report().loaded_tensors, 851);
    assert_eq!(l.report().skipped_mtp_tensors.len(), 15);
    eprintln!(
        "GGUF main model: {:?}; device weight+RoPE={} bytes",
        l.report(),
        l.device_weight_bytes::<half::bf16>().unwrap()
    );
    let scope = infer_backend_cuda::Cuda::with_memory_plan(
        0,
        infer_backend_cuda::CudaMemoryPlan {
            kernel_workspace_bytes: 1 << 20,
            graph_arena_bytes: 1 << 20,
            pool_retain_bytes: 4 << 20,
        },
    )
    .unwrap()
    .scope();
    assert!(scope.device().config.cute_dsl_available());
    let now = std::time::Instant::now();
    let model = l.load::<half::bf16, _>(scope.device()).unwrap();
    assert_eq!(model.cache_layout().num_full_layers(), 16);
    assert_eq!(model.cache_layout().linear_dims().len(), 48);
    let ids = [0, 42, 248319];
    let mut expected = Vec::new();
    let view =
        crate::models::gguf_weights::block_quant_view(r.read_view("token_embd.weight").unwrap())
            .unwrap();
    for &id in &ids {
        let mut row = vec![0.0; 5120];
        decode_row(view, id as usize, &mut row).unwrap();
        expected.extend(row.into_iter().map(half::bf16::from_f32));
    }
    drop(l);
    drop(r);
    let plan = BatchPlan {
        kind: BatchKind::Ragged,
        num_tokens: 3,
        batch: 1,
        q_lens: vec![3],
        kv_lens: vec![3],
        seq_positions: vec![0],
        rope_positions: vec![0, 1, 2],
        max_blocks_per_seq: 3,
        block_size: 1,
        total_q_tiles: 1,
    };
    let ctx = StepCtx::new(&scope, &plan);
    let input = Tensor::from_host_slice(&ids, [3], scope.device()).unwrap();
    let mut hidden = Hidden {
        stream: Tensor::zeros([3, 5120], scope.device()).unwrap(),
        pending: None,
    };
    model.embed(&input, &mut hidden, &ctx).unwrap();
    assert_eq!(hidden.stream.to_host_vec().unwrap(), expected);
    eprintln!(
        "Loaded all 64 layers and checked embedding after dropping mmap; elapsed {:.2}s",
        now.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "cute-dsl")]
#[test]
#[ignore = "requires CUDA"]
fn cuda_tiny_f32_residual_norm_uses_existing_operators() {
    use infer_core::ports::FusedOps;
    let dev = infer_backend_cuda::Cuda::new(0).unwrap();
    let scope = dev.scope();
    let plan = BatchPlan {
        kind: BatchKind::DecodeOnly,
        num_tokens: 2,
        batch: 2,
        q_lens: vec![1, 1],
        kv_lens: vec![1, 1],
        seq_positions: vec![0, 0],
        rope_positions: vec![0, 0],
        max_blocks_per_seq: 1,
        block_size: 1,
        total_q_tiles: 0,
    };
    let ctx = StepCtx::new(&scope, &plan);
    let a: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
    let b: Vec<f32> = (0..128)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let mut residual = Tensor::from_host_slice(&a, [2, 64], &dev).unwrap();
    let input = Tensor::from_host_slice(&b, [2, 64], &dev).unwrap();
    let weights = Tensor::from_host_slice(&[1.25f32; 64], [64], &dev).unwrap();
    let mut output = Tensor::<f32, _>::zeros([2, 64], &dev).unwrap();
    infer_backend_cuda::Cuda::fused_add_rmsnorm(
        &ctx,
        &mut output,
        &mut residual,
        &input,
        &weights,
        1e-6,
    )
    .unwrap();
    let sum: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
    assert_eq!(residual.to_host_vec().unwrap(), sum);
    let output = output.to_host_vec().unwrap();
    for row in 0..2 {
        let rms = (sum[row * 64..(row + 1) * 64]
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            / 64.0
            + 1e-6)
            .sqrt();
        for col in 0..64 {
            assert!(
                (f64::from(output[row * 64 + col]) - f64::from(sum[row * 64 + col]) * 1.25 / rms)
                    .abs()
                    < 2e-6
            );
        }
    }
}

#[test]
fn probe_chunking_reset_trace_and_rejected_inputs_preserve_state() {
    use super::probe::GgufProbe;
    let r = Fixture::new().reader();
    let loader = Qwen35GgufLoader::new(&r, options()).unwrap();
    let mut p = GgufProbe::<f32, Cpu>::load(&loader, HostScope::new(Cpu), 4).unwrap();
    drop(loader);
    drop(r);
    let full = p.step(&[3, 5, 7], true).unwrap();
    assert_eq!(full.layers.len(), 2);
    assert_eq!(full.position, 3);
    assert_eq!(full.logits.len(), 32);
    let next = p.step(&[11], false).unwrap();
    p.reset().unwrap();
    assert_eq!(p.position(), 0);
    p.step(&[3], false).unwrap();
    for ids in [vec![], vec![-1], vec![32], vec![0; 5]] {
        assert!(p.step(&ids, false).is_err());
        assert_eq!(p.position(), 1);
    }
    p.step(&[5], false).unwrap();
    let chunk = p.step(&[7], false).unwrap();
    for (a, b) in full.logits.iter().zip(&chunk.logits) {
        assert!((a - b).abs() < 5e-4, "{a} {b}");
    }
    let chunk_next = p.step(&[11], false).unwrap();
    for (a, b) in next.logits.iter().zip(&chunk_next.logits) {
        assert!((a - b).abs() < 5e-4);
    }
    p.reset().unwrap();
    let again = p.step(&[3, 5, 7], false).unwrap();
    assert_eq!(full.logits, again.logits);
    for _ in 0..3 {
        p.step(&[1, 2, 3, 4], false).unwrap();
    }
    assert!(p.step(&[1, 2], false).is_err());
    assert_eq!(p.position(), 15);
    p.step(&[1], false).unwrap();
    assert_eq!(p.position(), 16);
    assert!(p.step(&[1], false).is_err());
}

#[test]
fn probe_execution_failure_requires_reset() {
    use super::probe::GgufProbe;
    let mut fixture = Fixture::new();
    let broken = fixture
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_down.weight")
        .unwrap();
    // Encoded scales are intentionally not exhaustively scanned at load time.
    for block in broken.bytes.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
    }
    let reader = fixture.reader();
    let loader = Qwen35GgufLoader::new(&reader, options()).unwrap();
    let mut p = GgufProbe::<f32, Cpu>::load(&loader, HostScope::new(Cpu), 2).unwrap();
    assert!(
        p.step(&[3, 5], true)
            .unwrap_err()
            .to_string()
            .contains("layer 0")
    );
    assert_eq!(p.position(), 0);
    assert!(
        p.step(&[3], false)
            .unwrap_err()
            .to_string()
            .contains("requires reset")
    );
    p.reset().unwrap();
    assert!(
        p.step(&[3], true)
            .unwrap_err()
            .to_string()
            .contains("layer 0")
    );
}

#[cfg(feature = "cute-dsl")]
#[test]
#[ignore = "requires CUDA and CuTe DSL"]
fn cuda_probe_shared_scratch_matches_cpu_and_reset() {
    use super::probe::GgufProbe;
    let reader = Fixture::new().reader();
    let loader = Qwen35GgufLoader::new(&reader, options()).unwrap();
    let device = infer_backend_cuda::Cuda::with_memory_plan(
        0,
        infer_backend_cuda::CudaMemoryPlan {
            kernel_workspace_bytes: 1 << 20,
            graph_arena_bytes: 1 << 20,
            pool_retain_bytes: 4 << 20,
        },
    )
    .unwrap();
    let mut gpu =
        GgufProbe::<half::bf16, infer_backend_cuda::Cuda>::load(&loader, device.scope(), 3)
            .unwrap();
    let mut cpu = GgufProbe::<half::bf16, Cpu>::load(&loader, HostScope::new(Cpu), 3).unwrap();
    let full = gpu.step(&[3, 5, 7], true).unwrap();
    let expected = cpu.step(&[3, 5, 7], true).unwrap();
    for (a, b) in full.logits.iter().zip(&expected.logits) {
        assert!((a - b).abs() < 0.015 + 0.02 * b.abs());
    }
    let next = gpu.step(&[11], false).unwrap();
    let expected_next = cpu.step(&[11], false).unwrap();
    for (a, b) in next.logits.iter().zip(&expected_next.logits) {
        assert!((a - b).abs() < 0.015 + 0.02 * b.abs());
    }
    gpu.reset().unwrap();
    let repeat = gpu.step(&[3, 5, 7], false).unwrap();
    assert_eq!(full.logits, repeat.logits);
    gpu.reset().unwrap();
    gpu.step(&[3], false).unwrap();
    gpu.step(&[5], false).unwrap();
    let chunk = gpu.step(&[7], false).unwrap();
    assert_eq!(full.logits, chunk.logits);
    let chunk_next = gpu.step(&[11], false).unwrap();
    assert_eq!(next.logits, chunk_next.logits);
    gpu.reset().unwrap();
    gpu.step(&[3], false).unwrap();
    let suffix = gpu.step(&[5, 7], false).unwrap();
    assert_eq!(full.logits, suffix.logits);
    assert_eq!(next.logits, gpu.step(&[11], false).unwrap().logits);
}
