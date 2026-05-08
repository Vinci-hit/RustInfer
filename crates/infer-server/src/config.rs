//! 服务器配置

use clap::Parser;

/// CLI 参数 & 服务器配置
#[derive(Parser, Debug, Clone)]
#[command(name = "rustinfer-server")]
#[command(about = "RustInfer API Server")]
pub struct ServerConfig {
    /// Server host
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    pub host: String,

    /// Server port
    #[arg(short, long, default_value = "8000", env = "PORT")]
    pub port: u16,

    /// Scheduler endpoint (ZMQ 地址)
    #[arg(short, long, default_value = "ipc:///tmp/rustinfer.ipc", env = "SCHEDULER_ENDPOINT")]
    pub engine_endpoint: String,

    /// Tokenizer 路径（tokenizer.json 所在目录）
    #[arg(short, long, env = "TOKENIZER_PATH")]
    pub tokenizer: String,

    /// 模型名称（用于 /v1/models 返回）
    #[arg(short, long, default_value = "default", env = "MODEL_NAME")]
    pub model_name: String,

    /// 请求超时时间 (秒)
    #[arg(long, default_value = "120", env = "REQUEST_TIMEOUT_SECS")]
    pub request_timeout_secs: u64,

    /// Log level
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    pub log_level: String,
}
