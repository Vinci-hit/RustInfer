//! Worker → Scheduler **data plane** outputs.
//!
//! Carries only the per-step facts the scheduler cannot derive on its own:
//! which prefill segments completed and which tokens were sampled. KV
//! extension requests, step errors, and lifecycle ACKs all travel on the
//! control plane — see [`crate::worker_to_scheduler_control::WorkerControlMessage`].

use serde::{Deserialize, Serialize};

/// Worker → Scheduler 的 LLM step 输出。
///
/// 热路径只回传 Scheduler 无法自行推导的最小事实：哪些 prefill segment
/// 已完成，以及本步采样出的 token 与 Worker 判断出的 finished 状态。
///
/// **`assigned_indices`(Phase 4 新增)**：本步 worker 端 `GlobalKvAllocator`
/// 给每个参与 seq 新分配的全局 KV 索引段。同 step 同 seq 的 new indices 必然
/// 是一段连续区间,所以压成 `(sequence_id, base, len)` 8B/seq。Scheduler 收
/// 到后逐 token 调 `RadixTree::append_token`,把 token_id 与索引绑定到
/// 该 seq 的链尾,以便 LRU 回收和未来的前缀复用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub prefill_done: Vec<u64>,
    pub tokens: Vec<GeneratedToken>,
    /// 本步给每个 seq 新分配的全局 KV 索引段。Phase 4 引入,默认 `vec![]` 时
    /// 等价于旧路径(scheduler 仍持物理 block 编号,无需逐 token 跟踪)。
    #[serde(default)]
    pub assigned_indices: Vec<AssignedIndices>,
}

/// 一个 seq 在某 step 中拿到的连续全局索引段 `[base, base+len)`。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AssignedIndices {
    pub sequence_id: u64,
    pub base: u32,
    pub len: u16,
}

impl AssignedIndices {
    pub fn end(&self) -> u32 {
        self.base + self.len as u32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub sequence_id: u64,
    pub token_id: i32,
    pub finished: bool,
}

/// Worker → Scheduler 的 diffusion batch 输出。
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
