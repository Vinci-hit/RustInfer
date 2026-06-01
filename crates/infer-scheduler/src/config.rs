//! Scheduler configuration.

use crate::domain::ids::BlockSize;

/// Scheduler operating mode — determines scheduling behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum SchedulerMode {
    /// LLM: prefill → decode, continuous batching, chunked prefill.
    #[default]
    Llm,
    /// Diffusion: batch-in batch-out, no KV cache management,
    /// all requests in a batch finish together.
    Diffusion,
}


/// Top-level scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    // ─── Mode ───
    /// Scheduler operating mode (LLM or Diffusion).
    pub mode: SchedulerMode,
    // ─── Capacity limits ───
    /// Maximum number of concurrent sequences.
    pub max_num_seqs: usize,
    /// Maximum total tokens per scheduling iteration.
    pub max_batch_tokens: usize,
    /// Maximum sequence length (prompt + generation).
    pub max_model_len: usize,

    // ─── KV Cache ───
    /// Paged KV block size (tokens per block).
    pub paged_block_size: BlockSize,
    /// Total number of GPU blocks (only relevant for Paged mode).
    pub num_gpu_blocks: usize,

    // ─── Scheduling ───
    /// Max tokens per prefill chunk (None = no chunking).
    pub chunked_prefill_size: Option<usize>,
    /// Whether prefix caching is enabled (requires Paged mode).
    pub enable_prefix_caching: bool,

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
            mode: SchedulerMode::default(),
            max_num_seqs: 32,
            max_batch_tokens: 1024,
            max_model_len: 4096,
            paged_block_size: BlockSize::new(16),
            num_gpu_blocks: 256,
            chunked_prefill_size: None,
            enable_prefix_caching: false,
            frontend_endpoint: "ipc:///tmp/rustinfer.ipc".to_string(),
            worker_push_endpoint: "ipc:///tmp/rustinfer-worker-in.ipc".to_string(),
            worker_pull_endpoint: "ipc:///tmp/rustinfer-worker-out.ipc".to_string(),
            metrics_enabled: true,
        }
    }
}
