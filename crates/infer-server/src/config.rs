//! Server Configuration
//!
//! 管理 Server 的配置项，包括端口、Scheduler 地址、模型路径等。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Server 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP 服务器配置
    pub http: HttpConfig,
    
    /// Scheduler 连接配置
    pub scheduler: SchedulerConfig,
    
    /// 模型配置
    pub model: ModelConfig,
    
    /// 日志配置
    pub log: LogConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            http: HttpConfig::default(),
            scheduler: SchedulerConfig::default(),
            model: ModelConfig::default(),
            log: LogConfig::default(),
        }
    }
}

/// HTTP 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// 监听地址
    pub host: String,
    
    /// 监听端口
    pub port: u16,
    
    /// 是否启用 CORS
    pub enable_cors: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            enable_cors: true,
        }
    }
}

/// Scheduler 连接配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler 地址 (ZeroMQ)
    /// 例如: "tcp://localhost:5555" 或 "ipc:///tmp/rustinfer.ipc"
    pub address: String,
    
    /// 请求超时时间 (秒)
    pub request_timeout_sec: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            address: "ipc:///tmp/rustinfer.ipc".to_string(),
            request_timeout_sec: 300, // 5 minutes
        }
    }
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 模型名称 (用于 OpenAI API 响应)
    pub model_name: String,
    
    /// Tokenizer 路径
    pub tokenizer_path: PathBuf,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: "llama3-1b".to_string(),
            tokenizer_path: PathBuf::from("/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct/tokenizer.json"),
        }
    }
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// 日志级别: trace, debug, info, warn, error
    pub level: String,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
        }
    }
}

impl ServerConfig {
    /// 从环境变量加载配置
    pub fn from_env() -> Self {
        Self {
            http: HttpConfig {
                host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: std::env::var("PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(8000),
                enable_cors: true,
            },
            scheduler: SchedulerConfig {
                address: std::env::var("SCHEDULER_ADDRESS")
                    .unwrap_or_else(|_| "ipc:///tmp/rustinfer.ipc".to_string()),
                request_timeout_sec: 300,
            },
            model: ModelConfig {
                model_name: std::env::var("MODEL_NAME")
                    .unwrap_or_else(|_| "llama3-1b".to_string()),
                tokenizer_path: PathBuf::from(
                    std::env::var("TOKENIZER_PATH")
                        .unwrap_or_else(|_| "/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct/tokenizer.json".to_string())
                ),
            },
            log: LogConfig {
                level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            },
        }
    }
}
