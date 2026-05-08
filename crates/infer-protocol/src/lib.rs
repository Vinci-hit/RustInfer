use serde::{Deserialize, Serialize};

#[cfg(test)]
mod syntax_test;

/// Server -> Scheduler 的推理请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 唯一请求ID (UUID v4)
    pub request_id: String,

    /// 已 tokenize 的输入 token ids
    pub input_ids: Vec<i32>,

    /// 最大生成tokens数量
    pub max_tokens: usize,

    /// 温度参数 (0.0 = greedy, 1.0 = random)
    pub temperature: f32,

    /// Top-p (nucleus) sampling
    pub top_p: f32,

    /// Top-k sampling
    pub top_k: i32,

    /// 是否流式返回
    pub stream: bool,

    /// 优先级 (0=normal, 1=high, -1=low)
    pub priority: i32,

    /// Stop sequences（生成遇到这些字符串时停止）
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

/// Scheduler -> Server 的响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// 对应的请求ID
    pub request_id: String,

    /// 响应状态
    pub status: ResponseStatus,

    /// 生成的 token ids（Server 负责 decode 为文本）
    pub output_token_ids: Vec<i32>,

    /// 完成原因: "stop" (遇到 EOS/stop sequence) | "length" (达到 max_tokens)
    #[serde(default)]
    pub finish_reason: Option<String>,

    /// 错误信息
    pub error: Option<String>,

    /// 性能指标
    pub metrics: InferenceMetrics,
}

/// 流式响应的chunk
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
    Token,      // 正常token
    Done,       // 生成完成
    Error,      // 错误
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Error,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// 总生成耗时 (ms)
    pub total_ms: u64,

    /// 生成 token 数
    pub num_tokens: u32,

    /// 吞吐量 (tokens/s)
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
