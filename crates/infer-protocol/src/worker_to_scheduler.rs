use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAck {
    pub sequence_id: u64,
    pub removed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainAck {
    pub remaining_requests: usize,
}

/// Worker -> Scheduler 的 LLM step 输出。
///
/// 热路径只回传 Scheduler 无法自行推导的最小事实：哪些 prefill segment 已完成，
/// 以及本步采样出的 token 与 Worker 判断出的 finished 状态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub prefill_done: Vec<u64>,
    pub tokens: Vec<GeneratedToken>,
    #[serde(default)]
    pub need_blocks: Vec<NeedBlocksRequest>,
    #[serde(default)]
    pub error: Option<WorkerStepError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStepError {
    pub sequence_ids: Vec<u64>,
    pub message: String,
    pub fatal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeedBlocksRequest {
    pub sequence_id: u64,
    pub current_blocks: u32,
    pub required_blocks: u32,
    pub request_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub sequence_id: u64,
    pub token_id: i32,
    pub finished: bool,
}

/// Worker -> Scheduler 的 diffusion batch 输出。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionBatchOutput {
    pub results: Vec<DiffusionOutputItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionOutputItem {
    pub request_id: String,
    pub status: DiffusionOutputStatus,
    pub image: Option<DiffusionImage>,
    pub error: Option<String>,
    pub metrics: DiffusionOutputMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffusionOutputStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionImage {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// Raw image payload format, e.g. "rgb8".
    pub format: String,
    /// Raw payload. For "rgb8", data is interleaved HWC RGB bytes.
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffusionOutputMetrics {
    pub encode_prompt_ms: u64,
    pub denoise_ms: u64,
    pub decode_ms: u64,
    pub total_ms: u64,
}
