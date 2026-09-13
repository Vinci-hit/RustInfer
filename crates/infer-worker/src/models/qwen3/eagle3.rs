//! Automatic EAGLE3 checkpoint dispatch. File conventions stay in the model
//! assembly layer; both formats build the same conditioned draft component.
mod deep_spec;
mod spec_forge;

pub use deep_spec::{DeepSpecConfig, DraftRope, build_draft};
pub use spec_forge::SpecForgeConfig;

/// Compatibility name for the original DeepSpec-only config API.
pub type Eagle3Config = DeepSpecConfig;

use crate::{
    components::eagle3::Eagle3DraftHead,
    domain::{
        dtype::Dtype,
        features::FeatureSpec,
        model::ModelDims,
        ports::{OpBackend, OpError, OpResult, backend::LlmBackend},
        tensor::Tensor,
    },
    models::loader::WeightLoader,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eagle3Format {
    DeepSpec,
    SpecForge,
}

impl Eagle3Format {
    /// The declared architecture and tensor convention must agree. Never infer
    /// a format from a directory name, or silently fall back on malformed input.
    pub fn detect(config: &serde_json::Value, loader: &WeightLoader<'_>) -> OpResult<Self> {
        let architectures = config["architectures"]
            .as_array()
            .ok_or_else(|| OpError::Shape("EAGLE3 config requires architectures".into()))?;
        if architectures.len() != 1 {
            return Err(OpError::Shape("ambiguous EAGLE3 architectures".into()));
        }
        let (format, count, prefix, other) = match architectures[0].as_str() {
            Some("Qwen3Eagle3Model") => (Self::DeepSpec, 5, "layers.0", "midlayer"),
            Some("LlamaForCausalLMEagle3") => (Self::SpecForge, 3, "midlayer", "layers.0"),
            _ => {
                return Err(OpError::Shape(format!(
                    "unsupported EAGLE3 architecture {}",
                    architectures[0]
                )));
            }
        };
        let h = config["hidden_size"]
            .as_u64()
            .and_then(|h| usize::try_from(h).ok())
            .filter(|&h| h > 0)
            .ok_or_else(|| OpError::Shape("invalid EAGLE3 hidden_size".into()))?;
        let width = h
            .checked_mul(count)
            .ok_or_else(|| OpError::Shape("EAGLE3 feature width overflow".into()))?;
        let fc = loader.read_view("fc.weight").map_err(OpError::Shape)?;
        if fc.shape() != [h, width]
            || !loader.has_tensor(&format!("{prefix}.self_attn.q_proj.weight"))
            || loader.has_tensor(&format!("{other}.self_attn.q_proj.weight"))
        {
            return Err(OpError::Shape(format!(
                "EAGLE3 {format:?}: architecture and weight layout disagree"
            )));
        }
        Ok(format)
    }
}

#[derive(Debug)]
pub enum Eagle3Checkpoint {
    DeepSpec(DeepSpecConfig),
    SpecForge(SpecForgeConfig),
}

impl Eagle3Checkpoint {
    pub fn parse(raw: &[u8], loader: &WeightLoader<'_>) -> OpResult<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(raw).map_err(|e| OpError::Shape(e.to_string()))?;
        Ok(match Eagle3Format::detect(&config, loader)? {
            Eagle3Format::DeepSpec => Self::DeepSpec(
                serde_json::from_value(config).map_err(|e| OpError::Shape(e.to_string()))?,
            ),
            Eagle3Format::SpecForge => Self::SpecForge(
                serde_json::from_value(config).map_err(|e| OpError::Shape(e.to_string()))?,
            ),
        })
    }

    pub fn format(&self) -> Eagle3Format {
        match self {
            Self::DeepSpec(_) => Eagle3Format::DeepSpec,
            Self::SpecForge(_) => Eagle3Format::SpecForge,
        }
    }

    pub fn trained_target(&self) -> Option<&str> {
        match self {
            Self::DeepSpec(c) => Some(&c.target_model_name_or_path),
            Self::SpecForge(c) => c.target_model_name_or_path.as_deref(),
        }
    }

    pub fn validate(&self, target: ModelDims, context: usize, draft_tokens: usize) -> OpResult<()> {
        match self {
            Self::DeepSpec(c) => {
                c.validate(target, context)?;
                if draft_tokens > c.ttt_length {
                    return Err(OpError::Shape(
                        "EAGLE3 draft width exceeds checkpoint ttt_length".into(),
                    ));
                }
            }
            Self::SpecForge(c) => c.validate(target, context)?,
        }
        Ok(())
    }

    pub fn feature_spec(&self, target: ModelDims) -> OpResult<FeatureSpec> {
        match self {
            Self::DeepSpec(c) => Ok(c.feature_spec()),
            Self::SpecForge(c) => c.feature_spec(target),
        }
    }

    pub fn build<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        loader: &WeightLoader<'_>,
        target: ModelDims,
        target_embedding: &Tensor<T, D>,
        context: usize,
        device: &D,
    ) -> OpResult<Eagle3DraftHead<T, D>> {
        match self {
            Self::DeepSpec(c) => deep_spec::build_draft(loader, c, target, context, device),
            Self::SpecForge(c) => {
                spec_forge::build_draft(loader, c, target, target_embedding, context, device)
            }
        }
    }
}
