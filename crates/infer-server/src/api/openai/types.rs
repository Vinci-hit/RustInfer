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
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

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
    pub model: String,
    pub prompt: CompletionPrompt,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

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
