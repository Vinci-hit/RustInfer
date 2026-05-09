//! Scheduler configuration.

use std::ops::Range;

/// KV Cache management mode — determined at startup, immutable at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheMode {
    /// Contiguous slot mode: each sequence owns a dedicated slot id.
    /// Worker manages KV tensors internally. Compatible with current worker.
    Slot,
    /// Paged mode: scheduler maintains block tables, worker uses PagedAttention kernels.
    /// Currently a stub — returns NotImplemented.
    Paged { block_size: usize },
}

impl Default for KvCacheMode {
    fn default() -> Self {
        Self::Slot
    }
}

/// Preemption strategy when cache is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionMode {
    /// Drop KV cache, re-prefill later (simpler, less memory).
    Recompute,
    /// Swap KV blocks to CPU memory (stub).
    Swap,
    /// No preemption — reject new requests when full.
    Disabled,
}

impl Default for PreemptionMode {
    fn default() -> Self {
        Self::Disabled
    }
}

/// Scheduling policy selection.
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyConfig {
    /// FCFS continuous batching (default).
    ContinuousBatching,
    /// Priority-aware multi-tier QoS (stub).
    PriorityAware { tiers: Vec<PriorityTier> },
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self::ContinuousBatching
    }
}

/// A priority tier for QoS scheduling.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorityTier {
    pub name: String,
    pub priority_range: Range<i32>,
    pub max_concurrency: Option<usize>,
    pub timeout_ms: Option<u64>,
}

/// Top-level scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    // ─── Capacity limits ───
    /// Maximum number of concurrent sequences.
    pub max_num_seqs: usize,
    /// Maximum total tokens per scheduling iteration.
    pub max_batch_tokens: usize,
    /// Maximum sequence length (prompt + generation).
    pub max_model_len: usize,

    // ─── KV Cache ───
    /// KV cache management mode (slot or paged).
    pub kv_cache_mode: KvCacheMode,
    /// Total number of GPU blocks (only relevant for Paged mode).
    pub num_gpu_blocks: usize,

    // ─── Scheduling ───
    /// Which scheduling policy to use.
    pub policy: PolicyConfig,
    /// Max tokens per prefill chunk (None = no chunking).
    pub chunked_prefill_size: Option<usize>,
    /// Whether prefix caching is enabled (requires Paged mode).
    pub enable_prefix_caching: bool,
    /// Preemption strategy.
    pub preemption_mode: PreemptionMode,

    // ─── Transport ───
    /// ZMQ frontend endpoint (ROUTER socket, connects to HTTP server).
    pub frontend_endpoint: String,
    /// ZMQ worker push endpoint (sends BatchCommand).
    pub worker_push_endpoint: String,
    /// ZMQ worker pull endpoint (receives StepOutput).
    pub worker_pull_endpoint: String,

    // ─── Observability ───
    /// Enable metrics recording.
    pub metrics_enabled: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 32,
            max_batch_tokens: 1024,
            max_model_len: 4096,
            kv_cache_mode: KvCacheMode::default(),
            num_gpu_blocks: 256,
            policy: PolicyConfig::default(),
            chunked_prefill_size: None,
            enable_prefix_caching: false,
            preemption_mode: PreemptionMode::default(),
            frontend_endpoint: "ipc:///tmp/rustinfer.ipc".to_string(),
            worker_push_endpoint: "ipc:///tmp/rustinfer-worker-in.ipc".to_string(),
            worker_pull_endpoint: "ipc:///tmp/rustinfer-worker-out.ipc".to_string(),
            metrics_enabled: true,
        }
    }
}
