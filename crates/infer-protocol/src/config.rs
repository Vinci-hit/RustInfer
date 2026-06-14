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

/// Top-level launch config, deserialized from `rustinfer.toml`.
///
/// Every field has a default so a minimal file containing only `model = "..."`
/// loads cleanly. The four IPC endpoints are derived from [`Self::cluster_id`].
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// HTTP server host.
    #[serde(default = "default_host")]
    pub host: String,

    /// HTTP server port.
    #[serde(default = "default_port")]
    pub port: u16,

    /// Per-request timeout (seconds).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

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

    /// Scheduler mode: "llm" or "diffusion".
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Worker identifier (appears in logs).
    #[serde(default = "default_worker_id")]
    pub worker_id: String,

    /// Log level (tracing filter).
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

fn default_cluster_id() -> String {
    "rustinfer".to_string()
}
fn default_device() -> String {
    "cuda:0".to_string()
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

impl RustInferConfig {
    /// Load and parse a TOML config file. Errors are returned as `String`
    /// (infer-protocol has no error-handling dep); callers wrap with context.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes =
            std::fs::read_to_string(path).map_err(|e| format!("read config {}: {}", path, e))?;
        let cfg: RustInferConfig =
            toml::from_str(&bytes).map_err(|e| format!("parse config {}: {}", path, e))?;
        if cfg.model.trim().is_empty() {
            return Err(format!("config {}: `model` is required", path));
        }
        Ok(cfg)
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

    /// KV pool size override as Option (0 => None / auto-size).
    pub fn num_blocks_opt(&self) -> Option<usize> {
        if self.num_blocks == 0 {
            None
        } else {
            Some(self.num_blocks)
        }
    }
}

/// Resolve the internal model type (`"llama3"` / `"qwen3"`) from a model
/// directory's `config.json`.
///
/// Reads the HuggingFace `model_type` field, falling back to
/// `architectures[0]`. Returns `"qwen3"` when the hint contains `"qwen"`,
/// otherwise `"llama3"`.
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

    let resolved = if hint.contains("qwen") {
        "qwen3"
    } else {
        "llama3"
    };
    Ok(resolved.to_string())
}
