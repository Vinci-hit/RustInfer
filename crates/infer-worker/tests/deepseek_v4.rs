//! Independent Transformers fixtures cover all three attention types and both
//! routing modes, including the first HCA block, CSA overlap and sliding wrap.
use std::path::PathBuf;

use infer_worker::{
    domain::exec::HostScope,
    infrastructure::{cpu::Cpu, io::SafetensorsReader},
    models::{
        deepseek_v4::{TinyConfig, TinyModel},
        loader::WeightLoader,
    },
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deepseek_v4")
}

fn config() -> TinyConfig {
    serde_json::from_slice(&std::fs::read(fixture().join("config.json")).unwrap()).unwrap()
}

fn model() -> TinyModel<f32, Cpu> {
    let reader = SafetensorsReader::open(fixture().join("model.safetensors")).unwrap();
    TinyModel::load(&WeightLoader::new(&reader), config(), &Cpu).unwrap()
}

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: shape");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5,
            "{name} element {i}: actual={a}, reference={b}"
        );
    }
}

#[test]
fn fp32_matches_transformers_across_prefill_chunking_and_decode() {
    let model = model();
    let scope = HostScope::new(Cpu);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture().join("manifest.json")).unwrap()).unwrap();
    let ids: Vec<usize> = manifest["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let oracle = SafetensorsReader::open(fixture().join("reference.safetensors")).unwrap();
    let mut full_logits = Vec::new();
    for case in ["full", "chunked", "decode"] {
        let mut cache = model.new_cache();
        let mut cursor = 0;
        let mut parts = vec![Vec::new(); model.config().num_hidden_layers + 1];
        for length in manifest["cases"][case].as_array().unwrap() {
            let n = length.as_u64().unwrap() as usize;
            let out = model
                .forward(&ids[cursor..cursor + n], &mut cache, &scope)
                .unwrap();
            parts[0].extend(out.logits);
            for (dst, src) in parts[1..].iter_mut().zip(out.layers) {
                dst.extend(src);
            }
            cursor += n;
            assert_eq!(cache.position(), cursor);
        }
        assert_eq!(cursor, ids.len());
        assert_eq!(
            cache.compressed_entries(),
            [(0, 0), (34, 34), (1, 0), (34, 34)]
        );
        for (i, actual) in parts.iter().enumerate() {
            let part = if i == 0 {
                "logits".to_owned()
            } else {
                format!("layer{}", i - 1)
            };
            let name = format!("{case}.{part}");
            let view = oracle.read_view(&name).unwrap();
            assert_eq!(view.dtype(), safetensors::Dtype::F32);
            let expected: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            close(actual, &expected, &name);
        }
        if case == "full" {
            full_logits = parts.remove(0);
        } else {
            close(&parts[0], &full_logits, "Rust chunk invariance");
        }
    }
}

#[test]
fn independent_caches_and_invalid_inputs_do_not_contaminate_requests() {
    let model = model();
    let scope = HostScope::new(Cpu);
    let mut a = model.new_cache();
    let mut b = model.new_cache();
    let mut fresh = model.new_cache();
    model.forward(&[3, 7, 11], &mut a, &scope).unwrap();
    let expected = model.forward(&[9, 2, 5, 6], &mut fresh, &scope).unwrap();
    let actual = model.forward(&[9, 2, 5, 6], &mut b, &scope).unwrap();
    close(&actual.logits, &expected.logits, "independent requests");
    assert!(model.forward(&[], &mut a, &scope).is_err());
    assert!(
        model
            .forward(&[model.config().vocab_size], &mut a, &scope)
            .is_err()
    );
    assert!(
        model
            .forward(
                &vec![0; model.config().max_position_embeddings],
                &mut a,
                &scope
            )
            .is_err()
    );
    assert_eq!(a.position(), 3);
    let tail = model.forward(&[6], &mut a, &scope).unwrap();
    let mut fresh = model.new_cache();
    let whole = model.forward(&[3, 7, 11, 6], &mut fresh, &scope).unwrap();
    close(
        &tail.logits,
        &whole.logits[3 * model.config().vocab_size..],
        "continued request",
    );
}

#[test]
fn unsupported_checkpoint_features_fail_before_loading() {
    let base = config();
    base.validate().unwrap();
    let mut variants = Vec::new();
    let mut c = base.clone();
    c.rustinfer_tiny = false;
    variants.push(c);
    let mut c = base.clone();
    c.quantization_config = Some(serde_json::json!({"quant_method":"fp8"}));
    variants.push(c);
    let mut c = base.clone();
    c.num_nextn_predict_layers = 1;
    variants.push(c);
    let mut c = base.clone();
    c.rope_parameters["compress"]["rope_type"] = "yarn".into();
    variants.push(c);
    let mut c = base.clone();
    c.num_hidden_layers = usize::MAX;
    variants.push(c);
    let mut c = base.clone();
    c.o_groups = 0;
    variants.push(c);
    let mut c = base.clone();
    c.partial_rotary_factor = 0.0;
    variants.push(c);
    let mut c = base.clone();
    c.index_head_dim = 1;
    variants.push(c);
    let mut c = base.clone();
    c.layer_types.pop();
    variants.push(c);
    let mut c = base.clone();
    c.num_experts_per_tok = c.n_routed_experts + 1;
    variants.push(c);
    let mut c = base.clone();
    c.mlp_bias = true;
    variants.push(c);
    let mut c = base.clone();
    c.attention_bias = true;
    variants.push(c);
    let mut c = base.clone();
    c.attention_dropout = 0.1;
    variants.push(c);
    let mut c = base.clone();
    c.norm_topk_prob = false;
    variants.push(c);
    for c in variants {
        assert!(c.validate().is_err(), "unexpectedly accepted {c:?}");
    }
}
