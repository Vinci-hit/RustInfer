//! Shared launch configuration for all RustInfer processes.
//!
//! One TOML file drives every binary (server launcher, scheduler, worker, api).
//! Each process takes only `--config <path>`; all knobs and the four IPC
//! endpoints are derived from this file. `model_type` is intentionally NOT a
//! config field — it is resolved from the model's `config.json` via
//! [`resolve_model_type`] so the worker dispatch / chat template always match
//! the loaded weights.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// CUDA scratch-memory sizing from the shared launch TOML.
///
/// MiB keeps deployment configuration readable; the worker converts these
/// values to backend byte capacities exactly once during startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaMemoryConfig {
    /// Shared cuBLASLt/cuDNN/CUTLASS/FP8 kernel workspace.
    #[serde(default = "default_cuda_kernel_workspace_mib")]
    pub kernel_workspace_mib: usize,

    /// Lazily allocated CUDA Graph transient arena. `0` disables graph capture.
    #[serde(default = "default_cuda_graph_arena_mib")]
    pub graph_arena_mib: usize,

    /// Maximum freed allocation bytes retained by the eager recycling pool.
    #[serde(default = "default_cuda_pool_retain_mib")]
    pub pool_retain_mib: usize,
}

impl Default for CudaMemoryConfig {
    fn default() -> Self {
        Self {
            kernel_workspace_mib: default_cuda_kernel_workspace_mib(),
            graph_arena_mib: default_cuda_graph_arena_mib(),
            pool_retain_mib: default_cuda_pool_retain_mib(),
        }
    }
}

/// Top-level launch config, deserialized from `rustinfer.toml`.
///
/// Every field has a default so a minimal file containing only `model = "..."`
/// loads cleanly. The four IPC endpoints are derived from [`Self::cluster_id`].
///
/// `deny_unknown_fields`: a typo'd key (`max_batch_seq` for `max_batch_seqs`)
/// must fail the load instead of silently falling back to the default — all
/// three processes share this file, and a silent default here surfaces as a
/// confusing capacity/limit mismatch at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustInferConfig {
    /// Model directory (weights + tokenizer.json + config.json). Required.
    #[serde(default)]
    pub model: String,

    /// Display name for `/v1/models`. Empty => derive from `model` basename.
    #[serde(default)]
    pub model_name: String,

    /// Cluster id. All four IPC endpoints are derived from this.
    #[serde(default = "default_cluster_id")]
    pub cluster_id: String,

    /// CUDA device, e.g. "cuda:0".
    #[serde(default = "default_device")]
    pub device: String,

    /// Number of ranks in the tensor-parallel group. `1` disables TP.
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,

    /// HTTP server host.
    #[serde(default = "default_host")]
    pub host: String,

    /// HTTP server port.
    #[serde(default = "default_port")]
    pub port: u16,

    /// Per-request timeout (seconds).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Hard deadline for one mirrored TP startup/operation. A timeout exits
    /// the worker because blocking CUDA/NCCL calls cannot be safely unwound.
    #[serde(default = "default_tp_operation_timeout_secs")]
    pub tp_operation_timeout_secs: u64,

    /// Deadline for NCCL group creation and follower model/runtime startup.
    /// Kept separate because loading large shards can legitimately take much
    /// longer than one inference operation.
    #[serde(default = "default_tp_startup_timeout_secs")]
    pub tp_startup_timeout_secs: u64,

    /// Worker-liveness timeout (seconds). The scheduler declares a worker lost
    /// if no heartbeat arrives within this window. Decoupled from
    /// `request_timeout_secs` so a crashed/hung worker is detected in seconds
    /// rather than after a full request timeout. Must be safely above the
    /// worst-case single decode/prefill step (heartbeats are emitted inline on
    /// the worker serve loop), so the default is conservative.
    #[serde(default = "default_worker_heartbeat_timeout_secs")]
    pub worker_heartbeat_timeout_secs: u64,

    /// Max concurrent in-flight HTTP requests admitted by the server. Excess
    /// requests are shed at ingress with HTTP 429 instead of being queued into
    /// unbounded internal channels (which would grow scheduler memory until
    /// OOM). Because the server is the sole frontend, bounding in-flight here
    /// also bounds the scheduler's waiting queue. `0` => unlimited (no
    /// admission gate).
    #[serde(default = "default_max_inflight_requests")]
    pub max_inflight_requests: usize,

    /// Max batch tokens per iteration.
    #[serde(default = "default_max_batch_tokens")]
    pub max_batch_tokens: usize,

    /// Max concurrent sequences.
    #[serde(default = "default_max_batch_seqs")]
    pub max_batch_seqs: usize,

    /// Max sequence length (prompt + generation).
    #[serde(default = "default_max_model_len")]
    pub max_model_len: usize,

    /// Paged KV block size (tokens per block).
    #[serde(default = "default_paged_block_size")]
    pub paged_block_size: usize,

    /// Max tokens per prefill chunk. 0 => disabled.
    #[serde(default)]
    pub chunked_prefill_size: usize,

    /// Max NEW prefill sequences admitted per scheduling iteration (B1
    /// admission control). 0 => unlimited (default; bounded only by the
    /// token/seq budget). A smaller value smooths the TTFT tail under bursty
    /// arrivals by capping how much fresh prefill work enters at once.
    #[serde(default)]
    pub max_prefill_seqs_per_iter: usize,

    /// Shortest-job-first prefill ordering (B2). When true the scheduler
    /// admits shorter prompts first within an iteration, reducing
    /// head-of-line blocking from long prompts. Default false (strict FCFS).
    #[serde(default)]
    pub prefill_sjf: bool,

    /// Max time (milliseconds) the scheduler waits accumulating freshly
    /// arrived requests into a single, larger prefill batch before
    /// dispatching. `0` => low-latency mode: every request is dispatched as
    /// soon as it arrives (no waiting). A non-zero value raises throughput
    /// under bursty / high-QPS load by amortizing per-step overhead across a
    /// bigger prefill, at the cost of up to this much added TTFT.
    #[serde(default)]
    pub batch_wait_ms: u64,

    /// Enable RadixTree prefix caching (paged mode only).
    #[serde(default)]
    pub enable_prefix_caching: bool,

    /// Static memory fraction reserved for KV cache.
    #[serde(default = "default_mem_fraction_static")]
    pub mem_fraction_static: f32,

    /// Override worker KV pool size (blocks). 0 => auto-size.
    #[serde(default)]
    pub num_blocks: usize,

    /// Ignore EOS tokens; decode to `max_tokens` (fixed-length benchmarking).
    #[serde(default)]
    pub ignore_eos: bool,

    /// Scheduler mode. Only `"llm"` is supported in this release.
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Worker identifier (appears in logs).
    #[serde(default = "default_worker_id")]
    pub worker_id: String,

    /// Log level (tracing filter).
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// CUDA graph capture sizes for decode batches.
    /// Determines which batch sizes have pre-captured CUDA graphs.
    #[serde(default = "default_capture_sizes")]
    pub capture_sizes: Vec<usize>,

    /// CUDA scratch-memory plan. Read only from this shared launch config.
    #[serde(default)]
    pub cuda_memory: CudaMemoryConfig,
}

fn default_cluster_id() -> String {
    "rustinfer".to_string()
}
fn default_device() -> String {
    "cuda:0".to_string()
}
fn default_tensor_parallel_size() -> usize {
    1
}
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8000
}
fn default_request_timeout_secs() -> u64 {
    120
}
fn default_tp_operation_timeout_secs() -> u64 {
    120
}
fn default_tp_startup_timeout_secs() -> u64 {
    900
}
fn default_worker_heartbeat_timeout_secs() -> u64 {
    15
}
fn default_max_inflight_requests() -> usize {
    1024
}
fn default_max_batch_tokens() -> usize {
    8192
}
fn default_max_batch_seqs() -> usize {
    32
}
fn default_max_model_len() -> usize {
    4096
}
fn default_paged_block_size() -> usize {
    1
}
fn default_mem_fraction_static() -> f32 {
    0.9
}
fn default_mode() -> String {
    "llm".to_string()
}
fn default_worker_id() -> String {
    "worker-0".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_capture_sizes() -> Vec<usize> {
    vec![1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
}
fn default_cuda_kernel_workspace_mib() -> usize {
    256
}
fn default_cuda_graph_arena_mib() -> usize {
    256
}
fn default_cuda_pool_retain_mib() -> usize {
    256
}

impl RustInferConfig {
    /// Load and parse a TOML config file. Errors are returned as `String`
    /// (infer-protocol has no error-handling dep); callers wrap with context.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes =
            std::fs::read_to_string(path).map_err(|e| format!("read config {}: {}", path, e))?;
        let cfg: RustInferConfig =
            toml::from_str(&bytes).map_err(|e| format!("parse config {}: {}", path, e))?;
        cfg.validate()
            .map_err(|e| format!("config {}: {}", path, e))?;
        Ok(cfg)
    }

    /// Range-check the fields every process depends on. All three binaries load
    /// the same file through [`Self::load`], so a bad value fails at startup in
    /// each of them with the same message instead of surfacing later as a
    /// worker abort or a nonsense capacity.
    pub fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("`model` is required".into());
        }
        if self.tensor_parallel_size == 0 {
            return Err("`tensor_parallel_size` must be > 0".into());
        }
        if !(self.mem_fraction_static > 0.0 && self.mem_fraction_static <= 1.0) {
            return Err(format!(
                "`mem_fraction_static` must be in (0, 1], got {}",
                self.mem_fraction_static
            ));
        }
        if self.max_model_len == 0 {
            return Err("`max_model_len` must be > 0".into());
        }
        if self.max_batch_tokens == 0 {
            return Err("`max_batch_tokens` must be > 0".into());
        }
        if self.max_batch_seqs == 0 {
            return Err("`max_batch_seqs` must be > 0".into());
        }
        if self.paged_block_size == 0 {
            return Err("`paged_block_size` must be > 0".into());
        }
        if self.request_timeout_secs == 0 {
            return Err("`request_timeout_secs` must be > 0".into());
        }
        if self.tp_operation_timeout_secs == 0 {
            return Err("`tp_operation_timeout_secs` must be > 0".into());
        }
        if self.tp_startup_timeout_secs == 0 {
            return Err("`tp_startup_timeout_secs` must be > 0".into());
        }
        if self.port == 0 {
            return Err("`port` must be > 0".into());
        }
        const MAX_MIB: usize = usize::MAX / (1024 * 1024);
        for (name, value) in [
            (
                "cuda_memory.kernel_workspace_mib",
                self.cuda_memory.kernel_workspace_mib,
            ),
            (
                "cuda_memory.graph_arena_mib",
                self.cuda_memory.graph_arena_mib,
            ),
            (
                "cuda_memory.pool_retain_mib",
                self.cuda_memory.pool_retain_mib,
            ),
        ] {
            if value > MAX_MIB {
                return Err(format!("`{name}` is too large to represent in bytes"));
            }
        }
        if self.mode != "llm" {
            return Err(format!(
                "`mode` must be \"llm\"; diffusion is disabled in this release, got {:?}",
                self.mode
            ));
        }
        Ok(())
    }

    /// ZMQ frontend endpoint (ROUTER; HTTP server connects here).
    pub fn frontend_endpoint(&self) -> String {
        format!("ipc:///tmp/rustinfer-{}-frontend.ipc", self.cluster_id)
    }

    /// Scheduler → worker data plane endpoint.
    pub fn worker_in_endpoint(&self) -> String {
        format!("ipc:///tmp/rustinfer-{}-worker-in.ipc", self.cluster_id)
    }

    /// Worker → scheduler data plane endpoint.
    pub fn worker_out_endpoint(&self) -> String {
        format!("ipc:///tmp/rustinfer-{}-worker-out.ipc", self.cluster_id)
    }

    /// Control plane endpoint (lifecycle handshake + runtime control).
    pub fn worker_control_endpoint(&self) -> String {
        format!(
            "ipc:///tmp/rustinfer-{}-worker-control.ipc",
            self.cluster_id
        )
    }

    /// All four endpoints (for cleanup).
    pub fn all_endpoints(&self) -> [String; 4] {
        [
            self.frontend_endpoint(),
            self.worker_in_endpoint(),
            self.worker_out_endpoint(),
            self.worker_control_endpoint(),
        ]
    }

    /// Effective model name: explicit `model_name` or `model` basename.
    pub fn effective_model_name(&self) -> String {
        if !self.model_name.trim().is_empty() {
            return self.model_name.clone();
        }
        Path::new(&self.model)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string()
    }

    /// Chunked prefill size as Option (0 => None).
    pub fn chunked_prefill(&self) -> Option<usize> {
        if self.chunked_prefill_size == 0 {
            None
        } else {
            Some(self.chunked_prefill_size)
        }
    }

    /// Batch-accumulation wait as `Option<Duration>` (0 => None /
    /// low-latency mode). See [`Self::batch_wait_ms`].
    pub fn batch_wait(&self) -> Option<std::time::Duration> {
        if self.batch_wait_ms == 0 {
            None
        } else {
            Some(std::time::Duration::from_millis(self.batch_wait_ms))
        }
    }

    /// KV pool size override as Option (0 => None / auto-size).
    pub fn num_blocks_opt(&self) -> Option<usize> {
        if self.num_blocks == 0 {
            None
        } else {
            Some(self.num_blocks)
        }
    }
}

pub const SUPPORTED_MODEL_TYPES: &[&str] = &["llama3", "qwen3", "qwen3_moe"];

pub fn supported_model_types() -> &'static [&'static str] {
    SUPPORTED_MODEL_TYPES
}

pub fn supported_model_types_csv() -> String {
    SUPPORTED_MODEL_TYPES.join(", ")
}

/// Resolve the internal model type (`"llama3"` / `"qwen3"` / `"qwen3_moe"`)
/// from a model directory's `config.json`.
///
/// Reads the HuggingFace `model_type` field, falling back to
/// `architectures[0]`. The mapping itself is pure ([`classify_model_type`]) so
/// it can be unit-tested without touching the filesystem; this wrapper adds the
/// I/O and the visible unsupported-model error.
pub fn resolve_model_type(model_path: &str) -> Result<String, String> {
    let cfg_path = Path::new(model_path).join("config.json");
    let bytes =
        std::fs::read(&cfg_path).map_err(|e| format!("read {}: {}", cfg_path.display(), e))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {}: {}", cfg_path.display(), e))?;

    let hint = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| {
            cfg.get("architectures")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
        .to_lowercase();

    let Some(resolved) = classify_model_type(&hint) else {
        let shown = if hint.is_empty() {
            "<missing model_type/architectures>".to_string()
        } else {
            hint
        };
        return Err(format!(
            "unsupported model hint {} in {}; supported models: {}",
            shown,
            cfg_path.display(),
            supported_model_types_csv()
        ));
    };
    Ok(resolved.to_string())
}

/// Pure hint → internal model-type mapping. `hint` is a lowercased
/// `model_type`/`architectures[0]` string. Returns `None` when nothing matches
/// so the caller can produce the visible unsupported-model error.
///
/// Order matters: more specific Qwen variants must be tested before plain
/// `qwen3`, and unsupported variants such as Qwen3.5 must not fall through.
fn classify_model_type(hint: &str) -> Option<&'static str> {
    if hint.contains("qwen3_5") {
        None
    } else if hint.contains("qwen3_moe") || hint.contains("qwen3moe") {
        Some("qwen3_moe")
    } else if hint.contains("qwen3") {
        Some("qwen3")
    } else if hint.contains("llama") {
        Some("llama3")
    } else {
        None
    }
}

#[cfg(test)]
mod model_type_tests {
    use super::classify_model_type;

    #[test]
    fn unsupported_qwen3_5_does_not_fall_through() {
        assert_eq!(classify_model_type("qwen3_5"), None);
        assert_eq!(classify_model_type("qwen3_5forconditionalgeneration"), None);
        assert_eq!(classify_model_type("qwen3_5_text"), None);
    }

    #[test]
    fn qwen3_moe_beats_plain_qwen3() {
        assert_eq!(classify_model_type("qwen3_moe"), Some("qwen3_moe"));
        assert_eq!(
            classify_model_type("qwen3moeforcausallm"),
            Some("qwen3_moe")
        );
    }

    #[test]
    fn plain_qwen_and_llama_unchanged() {
        assert_eq!(classify_model_type("qwen3"), Some("qwen3"));
        assert_eq!(classify_model_type("qwen2"), None);
        assert_eq!(classify_model_type("llamaforcausallm"), Some("llama3"));
        assert_eq!(classify_model_type("llama"), Some("llama3"));
    }

    #[test]
    fn unknown_hint_falls_through() {
        assert_eq!(classify_model_type(""), None);
        assert_eq!(classify_model_type("mistral"), None);
    }
}

#[cfg(test)]
mod launch_config_tests {
    use super::RustInferConfig;

    #[test]
    fn tensor_parallel_size_defaults_to_one_and_accepts_explicit_value() {
        let defaults: RustInferConfig = toml::from_str("model = '/tmp/model'").unwrap();
        assert_eq!(defaults.tensor_parallel_size, 1);

        let configured: RustInferConfig = toml::from_str(
            r#"
                model = "/tmp/model"
                tensor_parallel_size = 4
            "#,
        )
        .unwrap();
        assert_eq!(configured.tensor_parallel_size, 4);
        configured.validate().unwrap();
    }

    #[test]
    fn zero_tensor_parallel_size_is_rejected() {
        let config: RustInferConfig = toml::from_str(
            r#"
                model = "/tmp/model"
                tensor_parallel_size = 0
            "#,
        )
        .unwrap();

        let error = config.validate().unwrap_err();

        assert!(error.contains("`tensor_parallel_size` must be > 0"));
    }

    #[test]
    fn diffusion_mode_is_rejected() {
        let config: RustInferConfig = toml::from_str(
            r#"
                model = "/tmp/model"
                mode = "diffusion"
            "#,
        )
        .unwrap();

        let error = config.validate().unwrap_err();

        assert!(error.contains("diffusion is disabled"));
    }

    #[test]
    fn cuda_memory_defaults_and_explicit_values_parse() {
        let defaults: RustInferConfig = toml::from_str("model = '/tmp/model'").unwrap();
        assert_eq!(defaults.cuda_memory.kernel_workspace_mib, 256);
        assert_eq!(defaults.cuda_memory.graph_arena_mib, 256);
        assert_eq!(defaults.cuda_memory.pool_retain_mib, 256);

        let configured: RustInferConfig = toml::from_str(
            r#"
                model = "/tmp/model"
                [cuda_memory]
                kernel_workspace_mib = 128
                graph_arena_mib = 192
                pool_retain_mib = 64
            "#,
        )
        .unwrap();
        assert_eq!(configured.cuda_memory.kernel_workspace_mib, 128);
        assert_eq!(configured.cuda_memory.graph_arena_mib, 192);
        assert_eq!(configured.cuda_memory.pool_retain_mib, 64);
    }
}
