use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerState {
    Spawned,
    Connecting,
    Registered,
    LoadingModel,
    ProfilingMemory,
    AllocatingRuntime,
    Warmup,
    Ready,
    Running,
    Draining,
    Error,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHello {
    pub worker_id: String,
    pub pid: u32,
    pub hostname: String,
    pub device: String,
    pub protocol_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerProgress {
    pub worker_id: String,
    pub state: WorkerState,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapacity {
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
    pub max_running_requests: usize,
    pub max_total_kv_tokens: Option<usize>,
    pub free_mem_before_load_gb: Option<f64>,
    pub free_mem_after_load_gb: Option<f64>,
    pub weight_mem_usage_gb: Option<f64>,
    pub workspace_mem_usage_gb: Option<f64>,
    pub graph_mem_usage_gb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerReady {
    pub worker_id: String,
    pub model_instance_id: String,
    pub model_path: String,
    pub model_type: String,
    pub device: String,
    pub capacity: WorkerCapacity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    pub worker_id: String,
    pub state: WorkerState,
    pub active_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerError {
    pub worker_id: String,
    pub state: WorkerState,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerControlMessage {
    Hello(WorkerHello),
    Progress(WorkerProgress),
    Ready(WorkerReady),
    Heartbeat(WorkerHeartbeat),
    Error(WorkerError),
}

pub const WORKER_CONTROL_PROTOCOL_VERSION: u32 = 1;
