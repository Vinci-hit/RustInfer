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

/// Worker -> Scheduler 的一步输出。
///
/// 热路径只回传 Scheduler 无法自行推导的最小事实：哪些 prefill segment 已完成，
/// 以及本步采样出的 token 与 Worker 判断出的 finished 状态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub prefill_done: Vec<u64>,
    pub tokens: Vec<GeneratedToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub sequence_id: u64,
    pub token_id: i32,
    pub finished: bool,
}
