//! Scheduler → Worker **control plane** messages.
//!
//! Spans bootstrap (Hello, LoadModel, InitPagedKv) and runtime (KV index
//! release, cancel, drain, unload, liveness). Wrapped in
//! [`crate::ControlEnvelope`] on the wire.

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
    /// Whether RadixTree prefix caching is enabled. When false, the worker
    /// uses real-time KV recycling (released → free) instead of scheduler-led
    /// eviction via `FreeKvIndices` / `Preempt`.
    #[serde(default)]
    pub enable_prefix_caching: bool,
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

/// Scheduler asking the worker to release a batch of global KV indices back
/// to its `GlobalKvAllocator` free pool. Sent in response to RadixTree LRU
/// eviction; worker is purely passive — it never decides which indices to
/// free.
///
/// Indices need not be contiguous; the worker's `GlobalKvAllocator::free`
/// sorts and coalesces adjacent runs into the free-range list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeKvIndices {
    pub model_instance_id: String,
    pub indices: Vec<u32>,
}

/// Cancel an in-flight sequence. Migrated from the data plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelSequence {
    pub sequence_id: u64,
}

/// Scheduler asking the worker to drop a list of in-flight sequences,
/// freeing their `block_table` slots back to the worker's allocator.
///
/// Sent in response to a Level-2 `AllocFailed` round (`round=1`) when
/// LRU eviction was insufficient and the scheduler picked victims from
/// the active decoding / chunked-prefilling pool. The scheduler has
/// already called `radix.mark_finished_chain(sid)` for each victim and
/// marked them as preempted in its `RequestTable` before sending this
/// message — the worker is purely passive: drop them from `active`
/// and `kv_allocator.free(&block_table)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preempt {
    pub model_instance_id: String,
    pub sequence_ids: Vec<u64>,
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

    // ── runtime: KV release ──
    /// Scheduler asks the worker to free a batch of global KV indices
    /// back to its allocator (driven by `RadixTree` LRU eviction).
    FreeKvIndices(FreeKvIndices),
    /// Scheduler asks the worker to drop a list of in-flight sequences
    /// after Level-2 victim preemption.
    Preempt(Preempt),

    // ── runtime: lifecycle ──
    Cancel(CancelSequence),
    Drain(DrainWorker),
    UnloadModel(UnloadModel),

    // ── liveness / shutdown ──
    Ping,
    Shutdown,
}
