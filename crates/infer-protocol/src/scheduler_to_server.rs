use serde::{Deserialize, Serialize};

/// Scheduler -> Server 的完整响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub status: ResponseStatus,
    pub output_token_ids: Vec<i32>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub metrics: InferenceMetrics,
}

/// Scheduler -> Server 的流式响应 chunk。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub request_id: String,
    pub chunk_type: ChunkType,
    pub token_id: Option<i32>,
    pub finish_reason: Option<String>,
    pub metrics: Option<InferenceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    Token,
    Done,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub total_ms: u64,
    pub num_tokens: u32,
    pub tokens_per_second: f64,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            total_ms: 0,
            num_tokens: 0,
            tokens_per_second: 0.0,
        }
    }
}
