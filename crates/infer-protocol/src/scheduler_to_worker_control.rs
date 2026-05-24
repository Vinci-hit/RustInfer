//! Scheduler → Worker **control plane** messages.
//!
//! Spans bootstrap (Hello, LoadModel, InitPagedKv) and runtime (KV grants,
//! cancel, drain, unload, liveness). Wrapped in [`crate::ControlEnvelope`]
//! on the wire.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerHello {
    pub protocol_version: u32,
    pub heartbeat_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModel {
    pub model_instance_id: String,
    pub model_path: String,
    pub model_type: String,
    pub device: String,
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
    pub max_model_len: usize,
    pub mem_fraction_static: f32,
    pub tp_rank: usize,
    pub tp_size: usize,
    pub pp_rank: usize,
    pub pp_size: usize,
    /// "slot" or "paged:<block_size>". Optional for compatibility with older launchers.
    #[serde(default)]
    pub kv_cache_mode: Option<String>,
    /// Fraction of post-profile free memory used for KV cache, e.g. 0.95.
    #[serde(default)]
    pub kv_cache_memory_fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitPagedKv {
    pub model_instance_id: String,
    pub block_size: u32,
    pub initial_num_blocks: u32,
    pub max_num_blocks: u32,
    pub max_blocks_per_seq: u32,
    /// 每次 decode block table 不足时默认申请多少个 block。
    pub decode_block_request_blocks: u32,
    /// 剩余多少 token 容量时提前异步申请 block。
    pub decode_block_prefetch_margin: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrantBlocks {
    pub model_instance_id: String,
    pub sequence_id: u64,
    pub block_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrantBlocksDenied {
    pub model_instance_id: String,
    pub sequence_id: u64,
    pub reason: BlockGrantDeniedReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockGrantDeniedReason {
    CacheExhausted,
    SequenceNotFound,
    WorkerNotReady,
}

/// Cancel an in-flight sequence. Migrated from the data plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelSequence {
    pub sequence_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrainMode {
    Graceful,
    Immediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainWorker {
    pub mode: DrainMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnloadModel {
    pub model_instance_id: String,
}

/// Top-level scheduler-originated control message.
///
/// Bootstrap variants are emitted by `ControlPlane<Bootstrapping>::bootstrap`;
/// runtime variants by `ControlPlaneCmdTx`. RPC variants (`Ping`, `Drain`,
/// `UnloadModel`) carry a non-zero `RequestId` in their envelope and are
/// matched against pending calls by the scheduler when the worker replies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerControlMessage {
    // ── bootstrap ──
    Hello(SchedulerHello),
    LoadModel(LoadModel),
    InitPagedKv(InitPagedKv),

    // ── runtime: KV grants ──
    GrantBlocks(GrantBlocks),
    GrantBlocksDenied(GrantBlocksDenied),

    // ── runtime: lifecycle ──
    Cancel(CancelSequence),
    Drain(DrainWorker),
    UnloadModel(UnloadModel),

    // ── liveness / shutdown ──
    Ping,
    Shutdown,
}
