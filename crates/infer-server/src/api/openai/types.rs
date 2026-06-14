//! OpenAI 兼容 API 类型定义
//!
//! 包含所有 OpenAI API 请求/响应的 Rust 类型。

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// Chat Completion 请求/响应
// ═══════════════════════════════════════════════════════════════

/// POST /v1/chat/completions 请求体
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

    /// Force generation of exactly `max_tokens` by ignoring EOS tokens.
    /// Used for fixed-length benchmarking (mirrors vLLM's `ignore_eos`).
    #[serde(default)]
    pub ignore_eos: bool,

    #[serde(default)]
    pub stream: bool,

    /// 流式选项
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<StopSequence>,

    /// Frequency penalty [-2.0, 2.0]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty [-2.0, 2.0]
    pub presence_penalty: Option<f32>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Stop 序列：可以是单个字符串或字符串数组
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

/// 流式选项
#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    /// 是否在最后一个 chunk 中包含 usage 信息
    #[serde(default)]
    pub include_usage: bool,
}

/// 非流式 Chat Completion 响应
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

// ═══════════════════════════════════════════════════════════════
// Text Completion 请求/响应 (/v1/completions)
// ═══════════════════════════════════════════════════════════════

/// POST /v1/completions 请求体
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: CompletionPrompt,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

    /// Force generation of exactly `max_tokens` by ignoring EOS tokens.
    #[serde(default)]
    pub ignore_eos: bool,

    #[serde(default)]
    pub stream: bool,

    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,

    #[serde(default)]
    pub stop: Option<StopSequence>,

    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
}

/// Prompt 输入：可以是单个字符串或 token ids 数组
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Tokens(Vec<i32>),
}

/// 非流式 Completion 响应
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

// ═══════════════════════════════════════════════════════════════
// 流式 Chunk 类型 (SSE)
// ═══════════════════════════════════════════════════════════════

/// 流式 Chat Completion chunk
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// 流式 Completion chunk
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ═══════════════════════════════════════════════════════════════
// Image Generation 请求/响应 (/v1/images/generations)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,

    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Number of images. For binary response this must be 1.
    pub n: Option<usize>,
    /// OpenAI-style size string, e.g. "1024x1024".
    pub size: Option<String>,
    /// "b64_json" (default) or "binary".
    pub response_format: Option<ImageResponseFormat>,
    /// Encoded image format: "png" (default) or "jpeg".
    pub output_format: Option<ImageOutputFormat>,
    /// JPEG quality 1..100. Ignored for PNG.
    pub jpeg_quality: Option<u8>,

    pub num_inference_steps: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub sigmas: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseFormat {
    B64Json,
    Binary,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: i64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

// ═══════════════════════════════════════════════════════════════
// 共享类型
// ═══════════════════════════════════════════════════════════════

/// Token 使用统计
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ═══════════════════════════════════════════════════════════════
// Models 端点
// ═══════════════════════════════════════════════════════════════

/// GET /v1/models 响应
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

// ═══════════════════════════════════════════════════════════════
// 默认值函数
// ═══════════════════════════════════════════════════════════════

fn default_max_tokens() -> Option<usize> {
    Some(2048)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_accepts_missing_model() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hello"}]}"#).unwrap();
        assert!(req.model.is_none());
    }

    #[test]
    fn completion_request_accepts_missing_model() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt":"hello"}"#).unwrap();
        assert!(req.model.is_none());
    }

    #[test]
    fn image_request_accepts_missing_model() {
        let req: ImageGenerationRequest = serde_json::from_str(r#"{"prompt":"hello"}"#).unwrap();
        assert!(req.model.is_none());
    }
}
