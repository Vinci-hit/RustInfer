//! Scheduler configuration.

use infer_protocol::RustInferConfig;

use crate::domain::ids::BlockSize;

/// Scheduler operating mode — determines scheduling behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    /// Max NEW prefill sequences admitted per iteration (B1). 0 = unlimited.
    pub max_prefill_seqs_per_iter: usize,
    /// Shortest-job-first prefill ordering within an iteration (B2).
    pub prefill_sjf: bool,
    /// Max time to wait accumulating freshly arrived requests into a larger
    /// prefill batch before dispatching. `None` => low-latency mode (dispatch
    /// each request immediately). Throughput knob — see
    /// [`infer_protocol::RustInferConfig::batch_wait_ms`].
    pub batch_wait: Option<std::time::Duration>,
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

impl SchedulerConfig {
    /// Build a scheduler config from the shared launch config.
    ///
    /// Centralizes the launch-config → scheduler-config mapping (previously a
    /// hand-copied field list in `main.rs`) and applies cross-field validation:
    /// prefix caching is only meaningful in LLM mode, so it is forced off for
    /// other modes. Worker-derived capacity (`num_gpu_blocks`) is filled in
    /// later via [`Self::apply_worker_capacity`] once the worker profile is
    /// known.
    pub fn from_launch(
        cfg: &RustInferConfig,
        mode: SchedulerMode,
        paged_block_size: BlockSize,
    ) -> Self {
        let enable_prefix_caching = cfg.enable_prefix_caching && matches!(mode, SchedulerMode::Llm);
        if cfg.enable_prefix_caching && !enable_prefix_caching {
            tracing::warn!(
                "enable_prefix_caching is only supported in LLM mode; disabling for {:?} mode",
                mode
            );
        }
        Self {
            mode,
            max_num_seqs: cfg.max_batch_seqs,
            max_batch_tokens: cfg.max_batch_tokens,
            max_model_len: cfg.max_model_len,
            paged_block_size,
            chunked_prefill_size: cfg.chunked_prefill(),
            max_prefill_seqs_per_iter: cfg.max_prefill_seqs_per_iter,
            prefill_sjf: cfg.prefill_sjf,
            batch_wait: cfg.batch_wait(),
            enable_prefix_caching,
            frontend_endpoint: cfg.frontend_endpoint(),
            worker_push_endpoint: cfg.worker_in_endpoint(),
            worker_pull_endpoint: cfg.worker_out_endpoint(),
            ..Default::default()
        }
    }

    /// Fold the worker-profiled paged-KV capacity into the config once the
    /// control-plane handshake reports it: derive `num_gpu_blocks` from the
    /// worker's `max_total_kv_tokens`. No-op if the worker did not report a
    /// total (e.g. non-paged backends).
    pub fn apply_worker_capacity(&mut self, max_total_kv_tokens: Option<usize>) {
        if let Some(tokens) = max_total_kv_tokens {
            let block_size = self.paged_block_size.as_usize();
            self.num_gpu_blocks = tokens / block_size;
            tracing::info!(
                num_gpu_blocks = self.num_gpu_blocks,
                block_size,
                max_total_kv_tokens = tokens,
                "Paged KV capacity from worker profile"
            );
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            mode: SchedulerMode::default(),
            max_num_seqs: 32,
            max_batch_tokens: 1024,
            max_model_len: 4096,
            paged_block_size: BlockSize::new(1),
            num_gpu_blocks: 256,
            chunked_prefill_size: None,
            max_prefill_seqs_per_iter: 0,
            prefill_sjf: false,
            batch_wait: None,
            enable_prefix_caching: false,
            frontend_endpoint: "ipc:///tmp/rustinfer.ipc".to_string(),
            worker_push_endpoint: "ipc:///tmp/rustinfer-worker-in.ipc".to_string(),
            worker_pull_endpoint: "ipc:///tmp/rustinfer-worker-out.ipc".to_string(),
            metrics_enabled: true,
        }
    }
}
