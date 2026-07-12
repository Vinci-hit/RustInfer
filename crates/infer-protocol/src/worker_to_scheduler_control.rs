//! Worker → Scheduler **control plane** messages.
//!
//! Spans bootstrap (Hello/Progress/Ready/MemoryProfile/PagedKvReady) and
//! runtime (Heartbeat, StepError, ACKs). Wrapped in
//! [`crate::ControlEnvelope`] on the wire.

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
    /// Worker-side outstanding KV slots (`allocator.outstanding()`).
    /// Enables scheduler-side drift detection against its own `KvBudget`.
    #[serde(default)]
    pub kv_outstanding: Option<u32>,
    /// Worker-side KV slots that are allocated but not yet visible through a
    /// `StepOutput.assigned_indices` report: in-flight decode slots and
    /// next-step speculative reservations. Drift detection subtracts these
    /// before comparing with scheduler-confirmed outstanding.
    #[serde(default)]
    pub kv_transient_reserved: Option<u32>,
    /// Worker-side total free KV slots (`allocator.total_free()`).
    #[serde(default)]
    pub kv_total_free: Option<u32>,
    /// Number of released-but-not-recycled indices on the worker side.
    #[serde(default)]
    pub kv_released_pending: Option<u32>,
}

/// Worker-originated KV-pressure signal: `alloc_indices(n)` failed
/// against the worker-local `GlobalKvAllocator`.
///
/// Sent only on actual allocation failure (no periodic / heartbeat-driven
/// emission). The scheduler interprets `round` as the level in the
/// pressure-relief escalation:
/// - `round = 0` → Level 1 (RadixTree LRU eviction). Scheduler may reply
///   with `FreeKvIndices` after evicting up to ~5% of total slots.
/// - `round = 1` → Level 2 (victim preemption). Scheduler picks decoding
///   / chunked-prefilling victims, marks their chains finished, sends a
///   `Preempt(sequence_ids)` reply. Worker drops them from `active` and
///   frees their `block_table` locally.
///
/// `shortfall` is the original `n` that failed. It is informational —
/// the scheduler does not use it as a target. Useful for tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocFailed {
    pub worker_id: String,
    pub shortfall: u32,
    pub round: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerError {
    pub worker_id: String,
    pub state: WorkerState,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMemoryProfile {
    pub worker_id: String,
    pub model_instance_id: String,
    pub device: String,
    pub total_mem_bytes: u64,
    pub free_mem_before_load_bytes: u64,
    pub free_mem_after_dummy_bytes: u64,
    pub layer_num: u32,
    pub kv_head_num: u32,
    pub head_size: u32,
    pub dtype_size: u32,
    pub max_batch_tokens: u32,
    pub max_batch_seqs: u32,
    pub max_model_len: u32,
    /// Worker 建议留给 KV cache 的显存预算；通常是可用显存的可配置比例（如 95%）。
    pub suggested_kv_budget_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedKvReady {
    pub worker_id: String,
    pub model_instance_id: String,
    pub block_size: u32,
    pub initial_num_blocks: u32,
    pub max_num_blocks: u32,
    pub max_blocks_per_seq: u32,
    pub bytes_allocated: u64,
}

/// Per-step execution error. Migrated from `StepOutput.error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStepError {
    pub sequence_ids: Vec<u64>,
    pub message: String,
    pub fatal: bool,
}

/// Reply to `SchedulerControlMessage::Cancel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAck {
    pub sequence_id: u64,
    pub removed: bool,
}

/// Reply to `SchedulerControlMessage::Drain`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainAck {
    pub remaining_requests: usize,
}

/// Reply to `SchedulerControlMessage::UnloadModel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnloadAck {
    pub model_instance_id: String,
}

/// Top-level worker-originated control message.
///
/// RPC replies (`Pong`, `CancelAck`, `DrainAck`, `UnloadAck`) carry the same
/// `RequestId` as the originating scheduler message. Spontaneous events
/// (`Heartbeat`, `StepError`, bootstrap progress) carry `RequestId::NONE`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerControlMessage {
    // ── bootstrap ──
    Hello(WorkerHello),
    Progress(WorkerProgress),
    Ready(WorkerReady),
    MemoryProfile(WorkerMemoryProfile),
    PagedKvReady(PagedKvReady),

    // ── runtime: spontaneous ──
    Heartbeat(WorkerHeartbeat),
    StepError(WorkerStepError),
    Error(WorkerError),
    /// Worker hit `alloc_indices` failure and is asking the scheduler
    /// for KV pressure relief (LRU evict at round 0, preempt at round 1).
    AllocFailed(AllocFailed),

    // ── runtime: RPC replies ──
    Pong,
    CancelAck(CancelAck),
    DrainAck(DrainAck),
    UnloadAck(UnloadAck),
}

pub const WORKER_CONTROL_PROTOCOL_VERSION: u32 = 4;
