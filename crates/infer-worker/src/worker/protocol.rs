use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════
// 调度器 → Worker (ZMQ_IN, MessagePack)
// ═══════════════════════════════════════════

/// 调度器发来的 prefill 请求 batch。
///
/// `input_ids` 是所有请求的 token 拼成的一维数组，
/// `q_start_loc[i]` 标记第 i 个请求在 `input_ids` 中的起始偏移。
/// 请求数量 = `q_start_loc.len()`（Vec 自带长度，不需要额外字段）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillBatchCmd {
    /// 所有请求的 token 拼成一维
    pub input_ids: Vec<i32>,
    /// 每个请求在 input_ids 中的起始位置
    pub q_start_loc: Vec<u32>,
    /// 每个请求已经处理过的 token 数 (chunked prefill 时 > 0)
    pub num_computed_tokens: Vec<u32>,
    /// 调度器分配的 KV cache slot
    pub kv_slots: Vec<u32>,
    /// 采样参数
    pub sampling_params: Vec<SamplingParams>,
    /// 元信息
    pub request_metas: Vec<RequestMeta>,
}

impl PrefillBatchCmd {
    /// 请求数量
    pub fn num_requests(&self) -> usize {
        self.q_start_loc.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMeta {
    pub request_id: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

// ═══════════════════════════════════════════
// Worker → 调度器 (ZMQ_OUT, MessagePack)
// ═══════════════════════════════════════════

/// 一步的全量输出，每个活跃序列都包含在内。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub tokens: Vec<SeqToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqToken {
    pub request_id: String,
    pub token_id: i32,
    /// Worker 侧判断: true = 该请求已结束 (EOS 或 max_tokens)
    pub finished: bool,
}
