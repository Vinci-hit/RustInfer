// src/model/config.rs

use std::io::Read;

use serde::Deserialize;
use serde_json::Value;

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};

const NESTED_CONFIG_KEYS: [&str; 2] = ["text_config", "language_config"];

/// 这个函数支持两类 config.json：
/// 1) 顶层直接是模型字段
/// 2) 字段嵌套在 text_config/language_config 下
pub fn load_config_from_json<R: Read>(mut json_reader: R) -> Result<ModelFileConfig> {
    let mut json_bytes = Vec::new();
    json_reader.read_to_end(&mut json_bytes)?;

    let root: Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| Error::InvalidArgument(format!("Invalid JSON in config.json: {}", e)))?;

    let mut parse_errors = Vec::new();

    for key in NESTED_CONFIG_KEYS {
        if let Some(nested) = root.get(key) {
            match serde_json::from_value::<ModelFileConfig>(nested.clone()) {
                Ok(config) => return Ok(config),
                Err(e) => parse_errors.push(format!("{}: {}", key, e)),
            }
        }
    }

    match serde_json::from_value::<ModelFileConfig>(root) {
        Ok(config) => Ok(config),
        Err(e) => {
            parse_errors.push(format!("root: {}", e));
            Err(Error::InvalidArgument(format!(
                "Failed to parse config.json; attempted nested keys [{}] and root; details: {}",
                NESTED_CONFIG_KEYS.join(", "),
                parse_errors.join(" | ")
            )).into())
        }
    }
}

/// compressed-tensors 量化方案在 config.json 中的 quantization_config 字段。
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub quant_method: String,      // "compressed-tensors"
    pub bits: usize,               // 4
    pub group_size: usize,         // 128
    pub zero_point: bool,          // true
}

/// 用于中间反序列化的 raw 结构体
#[derive(Debug, Clone, Deserialize)]
struct RawQuantizationConfig {
    pub quant_method: String,
    #[serde(default)]
    pub config_groups: Option<serde_json::Value>,
}

impl<'de> serde::Deserialize<'de> for QuantizationConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawQuantizationConfig::deserialize(deserializer)?;

        // Extract from config_groups.group_0.weights
        let mut bits = 4usize;
        let mut group_size = 128usize;
        let mut zero_point = false;
        if let Some(cg) = &raw.config_groups {
            if let Some(g0) = cg.get("group_0") {
                if let Some(w) = g0.get("weights") {
                    if let Some(nb) = w.get("num_bits").and_then(|v| v.as_u64()) {
                        bits = nb as usize;
                    }
                    if let Some(gs) = w.get("group_size").and_then(|v| v.as_u64()) {
                        group_size = gs as usize;
                    }
                    if let Some(sym) = w.get("symmetric").and_then(|v| v.as_bool()) {
                        zero_point = !sym;
                    }
                }
            }
        }
        Ok(QuantizationConfig {
            quant_method: raw.quant_method,
            bits,
            group_size,
            zero_point,
        })
    }
}

/// 直接从 config.json 文件反序列化的原始模型配置。
/// 不提供默认值，缺字段直接报错。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelFileConfig {
    // 基础参数
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: i32,

    // 模型元数据：兼容部分仓库使用 dtype 命名
    #[serde(alias = "dtype")]
    pub torch_dtype: String,

    // 可选参数
    pub immediate_dim: Option<usize>,
    pub head_dim: Option<usize>,

    pub rope_theta: f64,
    #[serde(alias = "layer_norm_eps")]
    pub rms_norm_eps: f32,

    // 量化配置（AWQ 等），非量化模型中不存在此字段
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,

    // RoPE scaling 配置（Llama 3.1/3.2 等），非 scaling 模型中不存在此字段
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
}

/// RoPE scaling 配置 (Llama 3.1/3.2 等)
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,          // "default", "llama3", "linear", etc.
    #[serde(default = "default_factor")]
    pub factor: f64,                // 32.0
    #[serde(default)]
    pub high_freq_factor: Option<f64>,  // 4.0 for llama3
    #[serde(default)]
    pub low_freq_factor: Option<f64>,   // 1.0 for llama3
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,  // 8192 for llama3
}

fn default_rope_type() -> String { "default".to_string() }
fn default_factor() -> f64 { 1.0 }

#[derive(Debug, Clone)]
pub struct RuntimeModelConfig {
    pub dim: usize,
    pub intermediate_size: usize,
    pub layer_num: usize,
    pub head_num: usize,
    pub kv_head_num: usize,
    pub seq_len: usize,
    pub vocab_size: usize,

    pub kv_dim: usize,
    pub kv_mul: usize,
    pub head_size: usize,
    pub q_dim: usize,

    pub is_shared_weight: bool,
    pub torch_dtype: String,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tokenizer_vocab_size: usize,

    pub immediate_dim: Option<usize>,

    /// AWQ 等量化配置，None 表示非量化模型
    pub quant_config: Option<QuantizationConfig>,

    /// RoPE scaling 配置
    pub rope_scaling: Option<RopeScalingConfig>,
}

impl RuntimeModelConfig {
    pub fn new(file_config: &ModelFileConfig) -> Result<Self> {
        if !file_config
            .hidden_size
            .is_multiple_of(file_config.num_attention_heads)
        {
            return Err(Error::InvalidArgument(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                file_config.hidden_size, file_config.num_attention_heads
            ))
            .into());
        }

        if !file_config
            .num_attention_heads
            .is_multiple_of(file_config.num_key_value_heads)
        {
            return Err(Error::InvalidArgument(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                file_config.num_attention_heads, file_config.num_key_value_heads
            ))
            .into());
        }

        let normalized_torch_dtype = normalize_torch_dtype(file_config.torch_dtype.as_str())
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "Unsupported dtype '{}' in config.json (expected one of float32/f32/fp32, bfloat16/bf16)",
                    file_config.torch_dtype
                ))
            })?;

        let dim = file_config.hidden_size;
        let intermediate_size = file_config.intermediate_size;
        let layer_num = file_config.num_hidden_layers;
        let head_num = file_config.num_attention_heads;
        let kv_head_num = file_config.num_key_value_heads;
        let seq_len = 2048;

        let head_size = file_config.head_dim.unwrap_or(dim / head_num);
        let q_dim = head_num * head_size;
        let kv_dim = kv_head_num * head_size;
        let kv_mul = head_num / kv_head_num;

        let is_shared_weight = file_config.vocab_size > 0;
        let vocab_size = file_config.vocab_size.unsigned_abs() as usize;

        Ok(Self {
            dim,
            intermediate_size,
            layer_num,
            head_num,
            kv_head_num,
            seq_len,
            vocab_size,
            kv_dim,
            kv_mul,
            head_size,
            q_dim,
            is_shared_weight,
            torch_dtype: normalized_torch_dtype.to_string(),
            rope_theta: file_config.rope_theta as f32,
            rms_norm_eps: file_config.rms_norm_eps,
            tokenizer_vocab_size: vocab_size,
            immediate_dim: file_config.immediate_dim,
            quant_config: file_config.quantization_config.clone(),
            rope_scaling: file_config.rope_scaling.clone(),
        })
    }

    pub fn runtime_float_dtype(&self, device: DeviceType) -> Result<DataType> {
        if device.is_cpu() {
            return Ok(DataType::F32);
        }

        match self.torch_dtype.as_str() {
            "float32" => Ok(DataType::F32),
            "float16" => Ok(DataType::F16),
            "bfloat16" => Ok(DataType::BF16),
            _ => Err(Error::InvalidArgument(format!(
                "Unsupported torch_dtype for runtime allocation: {}",
                self.torch_dtype
            ))
            .into()),
        }
    }
}

fn normalize_torch_dtype(dtype: &str) -> Option<&'static str> {
    match dtype {
        "float32" | "float" | "f32" | "fp32" => Some("float32"),
        "float16" | "f16" | "fp16" | "half" => Some("float16"),
        "bfloat16" | "bf16" => Some("bfloat16"),
        _ => None,
    }
}
