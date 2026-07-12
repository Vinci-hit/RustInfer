use serde::{Deserialize, Serialize};

/// Protocol version of the server↔scheduler (frontend) plane. Bumped on any
/// wire-incompatible change. The scheduler reports it in [`SchedulerPong`];
/// the server refuses readiness (`/ready` 503) on mismatch.
pub const FRONTEND_PROTOCOL_VERSION: u32 = 2;

/// Scheduler -> Server 的统一回复信封（tagged union）。
///
/// 服务端此前对裸 `InferenceResponse` / `StreamChunk` 做试探性反序列化来区分
/// 两种回复；rmp 的 positional 编码下任何字段增删都可能让一种类型静默解析成
/// 另一种。所有回复统一走这个带 tag 的枚举，与 `ServerCommand` 对称。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerReply {
    Full(InferenceResponse),
    Chunk(StreamChunk),
    /// Liveness reply to [`crate::server_to_scheduler::ServerCommand::Ping`].
    Pong(SchedulerPong),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerPong {
    pub protocol_version: u32,
}

/// Scheduler -> Server 的完整响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub status: ResponseStatus,
    #[serde(default)]
    pub output_token_ids: Vec<i32>,
    #[serde(default)]
    pub images: Vec<ImageOutput>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub metrics: InferenceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOutput {
    pub width: u32,
    pub height: u32,
    /// Number of channels. Text-to-image currently returns RGB = 3.
    pub channels: u32,
    /// Raw image payload format, e.g. "rgb8" or "rgb_f32".
    pub format: String,
    /// Raw image payload. For "rgb8", this is interleaved HWC RGB bytes.
    pub data: Vec<u8>,
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
