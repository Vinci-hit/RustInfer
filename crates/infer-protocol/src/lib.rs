use serde::{Deserialize, Serialize};

#[cfg(test)]
mod syntax_test;

/// Server -> Engine 的推理请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 唯一请求ID (UUID v4)
    pub request_id: String,

    /// 输入文本 (已应用chat template)
    pub prompt: String,

    /// 最大生成tokens数量
    pub max_tokens: usize,

    /// 温度参数 (0.0 = greedy, 1.0 = random)
    pub temperature: f32,

    /// Top-p (nucleus) sampling
    pub top_p: f32,

    /// Top-k sampling
    pub top_k: i32,

    /// 停止序列
    pub stop_sequences: Vec<String>,

    /// 是否流式返回
    pub stream: bool,

    /// 优先级 (0=normal, 1=high, -1=low)
    pub priority: i32,
}

/// Engine -> Server 的响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// 对应的请求ID
    pub request_id: String,

    /// 响应状态
    pub status: ResponseStatus,

    /// 生成的文本 (非流式时返回)
    pub text: Option<String>,

    /// Token IDs (可选，调试用)
    pub tokens: Option<Vec<i32>>,

    /// 生成的token数量
    pub num_tokens: u32,

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
    pub token: Option<String>,
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
    /// Prefill阶段耗时 (ms)
    pub prefill_ms: u64,

    /// Decode阶段耗时 (ms)
    pub decode_ms: u64,

    /// 排队等待时间 (ms)
    pub queue_ms: u64,

    /// 实际batch大小
    pub batch_size: usize,

    /// 吞吐量 (tokens/s)
    pub tokens_per_second: f64,

    /// Decode迭代次数
    pub decode_iterations: usize,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            prefill_ms: 0,
            decode_ms: 0,
            queue_ms: 0,
            batch_size: 1,
            tokens_per_second: 0.0,
            decode_iterations: 0,
        }
    }
}
