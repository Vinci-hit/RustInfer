//! Qwen3.5 hybrid decoder with an optional vision encoder.

pub mod mtp;
pub mod vision;

use crate::components::{
    Attention, DecoderBlock, DenseFfn, FullAttention, GatedDeltaNet, GdnWeights, Linear, LmHead,
    RmsNorm,
};
use crate::domain::cache::LinearDims;
use crate::domain::dtype::Dtype;
use crate::domain::model::ModelDims;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpBackend, OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::models::decoder::Decoder;
use crate::models::loader::{LoadConfig, WeightLoader, compute_rope_cache};

pub struct Qwen3_5Model<T: Dtype, D: LlmBackend> {
    pub decoder: Decoder<T, D>,
    pub vision: Option<vision::VisionEncoder<T, D>>,
    rotary_dim: usize,
    rope_theta: f64,
    mrope_section: [usize; 3],
}

impl<T: Dtype, D: OpBackend + LlmBackend> Qwen3_5Model<T, D> {
    pub fn load_vision(
        &mut self,
        loader: &WeightLoader<'_>,
        config: &serde_json::Value,
        device: &D,
    ) -> OpResult<()> {
        let vision: vision::VisionConfig = serde_json::from_value(config["vision_config"].clone())
            .map_err(|e| OpError::Shape(format!("vision_config: {e}")))?;
        let sections: [usize; 3] = serde_json::from_value(
            config["text_config"]["rope_parameters"]["mrope_section"].clone(),
        )
        .map_err(|e| OpError::Shape(format!("mrope_section: {e}")))?;
        if sections.iter().sum::<usize>() * 2 != self.rotary_dim
            || config["text_config"]["rope_parameters"]["mrope_interleaved"] != true
            || vision.out_hidden_size != crate::domain::model::DecoderModel::dims(&self.decoder).dim
        {
            return Err(OpError::Shape(
                "incompatible vision/MRoPE dimensions".into(),
            ));
        }
        self.vision = Some(vision::VisionEncoder::load(loader, vision, device)?);
        self.mrope_section = sections;
        Ok(())
    }
}

impl<T: Dtype, D: LlmBackend> crate::domain::model::DecoderModel<T, D> for Qwen3_5Model<T, D> {
    fn dims(&self) -> ModelDims {
        self.decoder.dims()
    }
    fn cache_layout(&self) -> &crate::domain::cache::CacheLayout {
        self.decoder.cache_layout()
    }
    fn stages(&self) -> &[crate::domain::component::StageKind] {
        self.decoder.stages()
    }
    fn install_scratch(
        &mut self,
        s: std::rc::Rc<crate::domain::forward_scratch::ForwardScratch<T, D>>,
    ) {
        self.decoder.install_scratch(s);
    }
    fn install_gdn_scratch(
        &mut self,
        s: std::rc::Rc<crate::domain::gdn_scratch::GdnScratch<T, D>>,
    ) -> OpResult<()> {
        self.decoder.install_gdn_scratch(s)
    }
    fn embed(
        &self,
        ids: &Tensor<i32, D>,
        hidden: &mut crate::domain::component::Hidden<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.decoder.embed(ids, hidden, ctx)
    }
    fn decode_layers(
        &self,
        range: crate::domain::component::LayerRange,
        hidden: &mut crate::domain::component::Hidden<T, D>,
        cache: &mut crate::domain::cache::ModelCacheView<'_, T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.decoder.decode_layers(range, hidden, cache, ctx)
    }
    fn finalize(
        &self,
        hidden: &crate::domain::component::Hidden<T, D>,
        rows: crate::domain::model::SampleRows<'_>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<crate::domain::model::Logits<T, D>> {
        self.decoder.finalize(hidden, rows, ctx)
    }
    fn multimodal_rope(&self) -> Option<(usize, f64, [usize; 3])> {
        self.vision
            .as_ref()
            .map(|_| (self.rotary_dim, self.rope_theta, self.mrope_section))
    }
    fn encode_image(
        &self,
        image: &infer_protocol::multimodal::ImageInput,
        scope: &D::Scope,
    ) -> OpResult<Tensor<T, D>> {
        self.vision
            .as_ref()
            .ok_or_else(|| OpError::unsupported("Qwen3.5", "vision encoder not loaded"))?
            .forward(image, scope)
    }
}

pub fn build<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    device: &D,
) -> OpResult<Qwen3_5Model<T, D>> {
    // GDN currently owns complete projection weights and recurrent heads.
    if loader.tensor_parallel().size != 1 {
        return Err(OpError::unsupported("qwen3_5::build", "tensor parallelism"));
    }
    if cfg.mlp_quant.is_some() || cfg.fp8_block.is_some() || cfg.num_experts != 0 {
        return Err(OpError::unsupported(
            "qwen3_5::build",
            "quantized or MoE weights",
        ));
    }
    let linear = cfg.linear_attn.as_ref().ok_or_else(|| {
        OpError::Shape("qwen3_5 requires a linear attention configuration".into())
    })?;
    if linear.layer_is_full.len() != cfg.layer_num {
        return Err(OpError::Shape(
            "qwen3_5 layer_types length does not match layer count".into(),
        ));
    }
    if cfg.rotary_dim == 0 || cfg.rotary_dim > cfg.head_dim || !cfg.rotary_dim.is_multiple_of(2) {
        return Err(OpError::Shape("qwen3_5 invalid rotary_dim".into()));
    }
    let linear_dims = LinearDims {
        num_key_heads: linear.num_key_heads,
        num_value_heads: linear.num_value_heads,
        key_head_dim: linear.key_head_dim,
        value_head_dim: linear.value_head_dim,
        conv_kernel_dim: linear.conv_kernel_dim,
    };
    linear_dims.validate()?;
    let q_dim = cfg.head_num * cfg.head_dim;
    let kv_dim = cfg.kv_head_num * cfg.head_dim;
    let dims = ModelDims {
        dim: cfg.dim,
        q_dim,
        kv_dim,
        qkv_dim: q_dim + 2 * kv_dim,
        intermediate_size: cfg.intermediate_size,
        vocab_size: cfg.vocab_size,
        head_num: cfg.head_num,
        head_dim: cfg.head_dim,
        kv_head_num: cfg.kv_head_num,
        num_layers: cfg.layer_num,
        ..ModelDims::default()
    };
    dims.validate()?;
    let prefix = "model.language_model";
    let embed = loader.load_vocab_parallel_embedding::<T, D>(
        &format!("{prefix}.embed_tokens.weight"),
        cfg.vocab_size,
        cfg.dim,
        device,
    )?;
    let lm_head = if loader.has_tensor("lm_head.weight") {
        loader.load_vocab_parallel_linear::<T, D>(
            "lm_head.weight",
            None,
            cfg.vocab_size,
            cfg.dim,
            device,
        )?
    } else {
        loader.vocab_parallel_linear_from_weight(
            embed.table.clone(),
            None,
            cfg.vocab_size,
            device,
        )?
    };
    let (sin, cos) = compute_rope_cache::<T, D>(
        cfg.seq_len,
        cfg.rotary_dim,
        cfg.rope_theta,
        cfg.rope_scaling.as_ref(),
        device,
    )?;
    let mut blocks = Vec::with_capacity(cfg.layer_num);
    let prefetch = loader.prefetch_layers(&format!("{prefix}.layers"), cfg.layer_num, device)?;
    for (i, &full) in linear.layer_is_full.iter().enumerate() {
        let prefetched = prefetch.next_layer()?;
        let layer_loader = loader.with_prefetched(&prefetched);
        let loader = &layer_loader;
        let layer = format!("{prefix}.layers.{i}");
        let input_layernorm = load_norm(
            loader,
            &format!("{layer}.input_layernorm.weight"),
            cfg.dim,
            cfg.rms_norm_eps,
            device,
        )?;
        let attention = if full {
            Attention::Full(FullAttention {
                input_layernorm,
                qkv_proj: loader.load_fused_qkv_with_fp8(
                    &layer,
                    q_dim * (1 + usize::from(cfg.attn_output_gate)),
                    kv_dim,
                    cfg.dim,
                    None,
                    device,
                )?,
                o_proj: load_linear(
                    loader,
                    &format!("{layer}.self_attn.o_proj.weight"),
                    cfg.dim,
                    q_dim,
                    device,
                )?,
                q_norm: Some(load_norm(
                    loader,
                    &format!("{layer}.self_attn.q_norm.weight"),
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    device,
                )?),
                k_norm: Some(load_norm(
                    loader,
                    &format!("{layer}.self_attn.k_norm.weight"),
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    device,
                )?),
                sin: sin.clone(),
                cos: cos.clone(),
                head_num: cfg.head_num,
                kv_head_num: cfg.kv_head_num,
                head_dim: cfg.head_dim,
                rotary_dim: cfg.rotary_dim,
                attn_output_gate: cfg.attn_output_gate,
                scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                scratch: None,
            })
        } else {
            let attn = format!("{layer}.linear_attn");
            Attention::Linear(GatedDeltaNet::new(
                GdnWeights {
                    input_layernorm,
                    in_proj_qkv: load_linear(
                        loader,
                        &format!("{attn}.in_proj_qkv.weight"),
                        linear.conv_dim(),
                        cfg.dim,
                        device,
                    )?,
                    in_proj_a: load_linear(
                        loader,
                        &format!("{attn}.in_proj_a.weight"),
                        linear.num_value_heads,
                        cfg.dim,
                        device,
                    )?,
                    in_proj_b: load_linear(
                        loader,
                        &format!("{attn}.in_proj_b.weight"),
                        linear.num_value_heads,
                        cfg.dim,
                        device,
                    )?,
                    in_proj_z: load_linear(
                        loader,
                        &format!("{attn}.in_proj_z.weight"),
                        linear.value_dim(),
                        cfg.dim,
                        device,
                    )?,
                    conv1d: loader.load_tensor(&format!("{attn}.conv1d.weight"), device)?,
                    a_log: loader.load_tensor::<f32, D>(&format!("{attn}.A_log"), device)?,
                    dt_bias: loader.load_tensor(&format!("{attn}.dt_bias"), device)?,
                    // GDN output norm is ordinary RMSNorm, not a zero-centered norm.
                    norm_weight: loader
                        .load_tensor::<f32, D>(&format!("{attn}.norm.weight"), device)?,
                    norm_eps: cfg.rms_norm_eps,
                    out_proj: load_linear(
                        loader,
                        &format!("{attn}.out_proj.weight"),
                        cfg.dim,
                        linear.value_dim(),
                        device,
                    )?,
                },
                linear_dims,
            )?)
        };
        blocks.push(DecoderBlock {
            attention,
            ffn: DenseFfn {
                post_attention_layernorm: load_norm(
                    loader,
                    &format!("{layer}.post_attention_layernorm.weight"),
                    cfg.dim,
                    cfg.rms_norm_eps,
                    device,
                )?,
                gate_up_proj: loader.load_fused_gate_up_with_fp8(
                    &layer,
                    cfg.intermediate_size,
                    cfg.dim,
                    None,
                    device,
                )?,
                down_proj: load_linear(
                    loader,
                    &format!("{layer}.mlp.down_proj.weight"),
                    cfg.dim,
                    cfg.intermediate_size,
                    device,
                )?,
                scratch: None,
            },
        });
        prefetch.recycle(prefetched);
    }
    let norm = load_norm(
        loader,
        &format!("{prefix}.norm.weight"),
        cfg.dim,
        cfg.rms_norm_eps,
        device,
    )?;
    Ok(Qwen3_5Model {
        decoder: Decoder::new(embed, blocks, norm, LmHead { proj: lm_head }, dims)?,
        vision: None,
        rotary_dim: cfg.rotary_dim,
        rope_theta: cfg.rope_theta,
        mrope_section: [11, 11, 10],
    })
}

fn load_linear<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    name: &str,
    rows: usize,
    cols: usize,
    device: &D,
) -> OpResult<Linear<T, D>> {
    let weight = load_shaped(loader, name, &[rows, cols], device)?;
    Ok(Linear::new(weight, None))
}

fn load_shaped<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    name: &str,
    shape: &[usize],
    device: &D,
) -> OpResult<Tensor<T, D>> {
    let view = loader.read_view(name).map_err(OpError::Kernel)?;
    if view.shape() != shape {
        return Err(OpError::Shape(format!(
            "{name}: expected {shape:?}, got {:?}",
            view.shape()
        )));
    }
    loader.load_tensor(name, device)
}

/// Keep zero-centered scales unmodified; normalization adds one in FP32.
fn load_norm<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    name: &str,
    dim: usize,
    eps: f32,
    device: &D,
) -> OpResult<RmsNorm<T, D>> {
    Ok(RmsNorm {
        weight: load_shaped(loader, name, &[dim], device)?,
        eps,
        zero_centered: true,
    })
}

impl<T: Dtype, D: LlmBackend> crate::domain::model::DecoderReadout<T, D> for Qwen3_5Model<T, D> {
    fn normalize_hidden_into(
        &self,
        hidden: &crate::domain::component::Hidden<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.decoder.normalize_hidden_into(hidden, output, ctx)
    }
    fn project_logits_into(
        &self,
        normalized: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.decoder.project_logits_into(normalized, output, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::cache::LayerCacheId;
    use crate::domain::model::DecoderModel;
    use crate::infrastructure::{cpu::Cpu, io::SafetensorsReader};
    use crate::models::loader::LinearAttnConfig;
    use safetensors::{Dtype as SafeDtype, tensor::TensorView};
    use std::collections::BTreeMap;

    pub(super) fn config() -> LoadConfig {
        LoadConfig {
            dim: 8,
            intermediate_size: 16,
            layer_num: 2,
            head_num: 2,
            kv_head_num: 1,
            head_dim: 4,
            vocab_size: 16,
            seq_len: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            mlp_quant: None,
            fp8_block: None,
            rotary_dim: 2,
            attn_output_gate: true,
            linear_attn: Some(LinearAttnConfig {
                num_key_heads: 1,
                num_value_heads: 2,
                key_head_dim: 2,
                value_head_dim: 2,
                conv_kernel_dim: 4,
                layer_is_full: vec![false, true],
            }),
            num_experts: 0,
            experts_per_tok: 0,
            moe_intermediate_size: 0,
            norm_topk_prob: false,
            decoder_sparse_step: 1,
        }
    }

    pub(super) fn checkpoint(mtp: bool) -> SafetensorsReader {
        let mut tensors = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>, value: f32| {
            let bytes: Vec<u8> = (0..shape.iter().product::<usize>())
                .flat_map(|_| value.to_le_bytes())
                .collect();
            tensors.insert(name, (shape, bytes));
        };
        let root = "model.language_model";
        add(format!("{root}.embed_tokens.weight"), vec![16, 8], 0.2);
        add(format!("{root}.norm.weight"), vec![8], 0.25);
        // Unrelated multimodal/MTP tensors must not be loaded.
        add("model.visual.unused.weight".into(), vec![1], 42.0);
        add("mtp.unused.weight".into(), vec![1], 42.0);
        for i in 0..2 {
            let p = format!("{root}.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{p}.{name}.weight"), vec![8], 0.0);
            }
            add(format!("{p}.mlp.gate_proj.weight"), vec![16, 8], 0.0);
            add(format!("{p}.mlp.up_proj.weight"), vec![16, 8], 0.0);
            add(format!("{p}.mlp.down_proj.weight"), vec![8, 16], 0.0);
            if i == 0 {
                for (name, shape, value) in [
                    ("in_proj_qkv.weight", vec![8, 8], 0.0),
                    ("in_proj_a.weight", vec![2, 8], 0.0),
                    ("in_proj_b.weight", vec![2, 8], 0.0),
                    ("in_proj_z.weight", vec![4, 8], 0.0),
                    ("out_proj.weight", vec![8, 4], 0.0),
                    ("conv1d.weight", vec![8, 1, 4], 0.0),
                    ("A_log", vec![2], 0.0),
                    ("dt_bias", vec![2], 0.0),
                    ("norm.weight", vec![2], 0.5),
                ] {
                    add(format!("{p}.linear_attn.{name}"), shape, value);
                }
            } else {
                for (name, shape) in [
                    ("q_proj", vec![16, 8]),
                    ("k_proj", vec![4, 8]),
                    ("v_proj", vec![4, 8]),
                    ("o_proj", vec![8, 8]),
                    ("q_norm", vec![4]),
                    ("k_norm", vec![4]),
                ] {
                    add(format!("{p}.self_attn.{name}.weight"), shape, 0.0);
                }
            }
        }
        if mtp {
            for (name, shape) in [
                ("fc", vec![8, 16]),
                ("pre_fc_norm_embedding", vec![8]),
                ("pre_fc_norm_hidden", vec![8]),
                ("norm", vec![8]),
                ("layers.0.input_layernorm", vec![8]),
                ("layers.0.post_attention_layernorm", vec![8]),
                ("layers.0.self_attn.q_proj", vec![16, 8]),
                ("layers.0.self_attn.k_proj", vec![4, 8]),
                ("layers.0.self_attn.v_proj", vec![4, 8]),
                ("layers.0.self_attn.o_proj", vec![8, 8]),
                ("layers.0.self_attn.q_norm", vec![4]),
                ("layers.0.self_attn.k_norm", vec![4]),
                ("layers.0.mlp.gate_proj", vec![16, 8]),
                ("layers.0.mlp.up_proj", vec![16, 8]),
                ("layers.0.mlp.down_proj", vec![8, 16]),
            ] {
                let bytes = (0..shape.iter().product::<usize>())
                    .flat_map(|i| (((i * 7 % 19) as f32 - 9.) * 0.035).to_le_bytes())
                    .collect();
                tensors.insert(format!("mtp.{name}.weight"), (shape, bytes));
            }
        }
        let views: BTreeMap<_, _> = tensors
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(SafeDtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let path = std::env::temp_dir().join(format!(
            "rustinfer-qwen35-{}-{}-{:?}.safetensors",
            std::process::id(),
            mtp,
            std::thread::current().id()
        ));
        safetensors::tensor::serialize_to_file(views, None, &path).unwrap();
        let reader = SafetensorsReader::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        reader
    }

    #[test]
    fn builds_checkpoint_layout_and_ties_embeddings() {
        let reader = checkpoint(false);
        let loader = WeightLoader::new(&reader);
        let cfg = config();
        let model = build::<f32, Cpu>(&loader, &cfg, &Cpu).unwrap();
        assert_eq!(
            model.cache_layout().layers(),
            &[LayerCacheId::Linear(0), LayerCacheId::Full(0)]
        );
        assert_eq!(
            model.decoder.norm.weight.to_host_vec().unwrap(),
            vec![0.25; 8]
        );
        assert_eq!(
            model.decoder.embed.table.data_ptr(),
            model
                .decoder
                .lm_head
                .proj
                .weight
                .as_dense()
                .unwrap()
                .data_ptr()
        );
        // No weight upload should precede unsupported/configuration checks.
        let tp = WeightLoader::with_tensor_parallel(&reader, 0, 2).unwrap();
        assert!(build::<f32, Cpu>(&tp, &cfg, &Cpu).is_err());
        let mut bad = cfg.clone();
        bad.linear_attn.as_mut().unwrap().layer_is_full.pop();
        assert!(build::<f32, Cpu>(&loader, &bad, &Cpu).is_err());
        let mut bad = cfg;
        bad.rotary_dim = 3;
        assert!(build::<f32, Cpu>(&loader, &bad, &Cpu).is_err());
    }
}
