use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAck {
    pub request_id: String,
    pub removed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainAck {
    pub remaining_requests: usize,
}

/// Worker -> Scheduler 的一步输出。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub tokens: Vec<SeqToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqToken {
    pub request_id: String,
    pub token_id: i32,
    pub finished: bool,
}
