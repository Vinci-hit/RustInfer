use serde::{Deserialize, Serialize};

/// Server -> Scheduler 的推理请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 唯一请求 ID。
    pub request_id: String,
    /// 已 tokenize 的输入 token ids。
    pub input_ids: Vec<i32>,
    /// 最大生成 token 数量。
    pub max_tokens: usize,
    /// 温度参数。
    pub temperature: f32,
    /// Top-p sampling。
    pub top_p: f32,
    /// Top-k sampling。
    pub top_k: i32,
    /// 是否流式返回。
    pub stream: bool,
    /// 优先级。
    pub priority: i32,
    /// Stop sequences。
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}
