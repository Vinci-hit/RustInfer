//! 服务器配置

use clap::Parser;

/// CLI 参数 & 服务器配置
#[derive(Parser, Debug, Clone)]
#[command(name = "rustinfer-server")]
#[command(about = "RustInfer API Server (All-in-one Launcher)")]
pub struct ServerConfig {
    /// Server host
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    pub host: String,

    /// Server port
    #[arg(short, long, default_value = "8000", env = "PORT")]
    pub port: u16,

    /// Model path (directory containing model weights + tokenizer.json).
    #[arg(short, long, env = "MODEL_PATH")]
    pub model: String,

    /// Model type: llama3, qwen3.
    #[arg(long, default_value = "llama3", env = "MODEL_TYPE")]
    pub model_type: String,

    /// Device(s): comma-separated. e.g. "cuda:0" or "cuda:0,cuda:1" for multi-GPU.
    #[arg(short, long, default_value = "cuda:0", env = "DEVICE")]
    pub device: String,

    /// Maximum batch tokens per iteration.
    #[arg(long, default_value = "4096", env = "MAX_BATCH_TOKENS")]
    pub max_batch_tokens: usize,

    /// Maximum concurrent sequences.
    #[arg(long, default_value = "32", env = "MAX_BATCH_SEQS")]
    pub max_batch_seqs: usize,

    /// Maximum model sequence length.
    #[arg(long, default_value = "8192", env = "MAX_MODEL_LEN")]
    pub max_model_len: usize,

    /// Chunked prefill size (None = disabled).
    #[arg(long, env = "CHUNKED_PREFILL_SIZE")]
    pub chunked_prefill_size: Option<usize>,

    /// KV cache mode forwarded to the scheduler. "slot" or "paged:<block_size>".
    #[arg(long, default_value = "slot", env = "KV_CACHE_MODE")]
    pub kv_cache_mode: String,

    /// Enable RadixTree prefix caching (paged mode only). Forwarded to scheduler.
    #[arg(long, default_value_t = false, env = "ENABLE_PREFIX_CACHING")]
    pub enable_prefix_caching: bool,

    /// Static memory fraction reserved for model runtime planning. Forwarded to scheduler.
    #[arg(long, default_value = "1.0", env = "MEM_FRACTION_STATIC")]
    pub mem_fraction_static: f32,

    /// 模型名称（用于 /v1/models 返回）
    #[arg(long, env = "MODEL_NAME")]
    pub model_name: Option<String>,

    /// 请求超时时间 (秒)
    #[arg(long, default_value = "120", env = "REQUEST_TIMEOUT_SECS")]
    pub request_timeout_secs: u64,

    /// Log level
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    pub log_level: String,
}

impl ServerConfig {
    /// Parse device list from comma-separated string.
    pub fn devices(&self) -> Vec<&str> {
        self.device.split(',').map(|s| s.trim()).collect()
    }

    /// Derive model name from path if not explicitly set.
    pub fn effective_model_name(&self) -> String {
        self.model_name.clone().unwrap_or_else(|| {
            std::path::Path::new(&self.model)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("default")
                .to_string()
        })
    }
}
