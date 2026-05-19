use serde::{Deserialize, Serialize};

/// Server -> Scheduler 的推理请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 唯一请求 ID。
    pub request_id: String,
    /// 请求模态。默认 LLM，保持旧请求兼容。
    #[serde(default)]
    pub modality: InferenceModality,

    // ─── LLM fields ───
    /// 已 tokenize 的输入 token ids。
    #[serde(default)]
    pub input_ids: Vec<i32>,
    /// 最大生成 token 数量。
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// 温度参数。
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-p sampling。
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k sampling。
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    /// 是否流式返回。
    #[serde(default)]
    pub stream: bool,
    /// 优先级。
    #[serde(default)]
    pub priority: i32,
    /// Stop sequences。
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    // ─── Diffusion fields ───
    #[serde(default)]
    pub diffusion: Option<DiffusionRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceModality {
    Llm,
    Diffusion,
}

impl Default for InferenceModality {
    fn default() -> Self {
        Self::Llm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionRequest {
    /// Original prompt text for logging / response metadata.
    pub prompt: String,
    /// Server-tokenized prompt ids after applying the diffusion text-encoder template.
    #[serde(default)]
    pub prompt_input_ids: Vec<i32>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Optional server-tokenized negative prompt ids.
    #[serde(default)]
    pub negative_prompt_input_ids: Option<Vec<i32>>,
    #[serde(default = "default_image_height")]
    pub height: u32,
    #[serde(default = "default_image_width")]
    pub width: u32,
    #[serde(default = "default_num_inference_steps")]
    pub num_inference_steps: usize,
    #[serde(default)]
    pub sigmas: Option<Vec<f32>>,
    #[serde(default)]
    pub guidance_scale: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_output_format")]
    pub output_format: String,
}

impl Default for DiffusionRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            prompt_input_ids: Vec::new(),
            negative_prompt: None,
            negative_prompt_input_ids: None,
            height: default_image_height(),
            width: default_image_width(),
            num_inference_steps: default_num_inference_steps(),
            sigmas: None,
            guidance_scale: 0.0,
            seed: None,
            output_format: default_output_format(),
        }
    }
}

fn default_max_tokens() -> usize { 2048 }
fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }
fn default_top_k() -> i32 { -1 }
fn default_image_height() -> u32 { 1024 }
fn default_image_width() -> u32 { 1024 }
fn default_num_inference_steps() -> usize { 8 }
fn default_output_format() -> String { "rgb8".to_string() }
