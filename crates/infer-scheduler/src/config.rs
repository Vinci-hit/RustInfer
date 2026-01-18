//! Configuration Management - 配置管理
//!
//! 提供统一的配置管理系统，支持命令行参数和配置文件。
//!
//! # 架构说明
//!
//! **重要**: Scheduler 层不持有 Tokenizer
//! - Tokenizer 属于 Server/Frontend 层，负责 String ↔ Token IDs 转换
//! - Scheduler 只处理 Token IDs，与字符串解耦
//! - `model_path` 仅用于：
//!   1. 读取模型元数据（config.json 中的 eos_token_id, max_position_embeddings 等）
//!   2. 传递给 Worker 用于加载模型权重
//!
//! # 配置层次
//!
//! ```text
//! SchedulerConfig (顶层配置)
//!   ├─ NetworkConfig (网络配置)
//!   ├─ ModelConfig (模型配置)
//!   ├─ MemoryConfig (显存配置)
//!   ├─ SchedulingConfig (调度策略配置)
//!   ├─ ParallelismConfig (并行配置)
//!   └─ LoggingConfig (日志配置)
//! ```

use crate::coordinator::CoordinatorConfig;
use crate::policy::SchedulePolicyConfig;
use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Scheduler 主配置
///
/// 可以从命令行参数或配置文件加载
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
#[command(name = "rustinfer-scheduler")]
#[command(about = "RustInfer Scheduler - LLM Inference Scheduler", long_about = None)]
pub struct SchedulerConfig {
    /// 网络配置
    #[command(flatten)]
    pub network: NetworkConfig,

    /// 模型配置
    #[command(flatten)]
    pub model: ModelConfig,

    /// 显存配置
    #[command(flatten)]
    pub memory: MemoryConfig,

    /// 调度策略配置
    #[command(flatten)]
    pub scheduling: SchedulingConfig,

    /// 并行配置
    #[command(flatten)]
    pub parallelism: ParallelismConfig,

    /// 日志配置
    #[command(flatten)]
    pub logging: LoggingConfig,

    /// 可选：从配置文件加载
    #[arg(long, value_name = "FILE")]
    #[serde(skip)]
    pub config_file: Option<PathBuf>,
}

/// 网络配置
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct NetworkConfig {
    /// ZeroMQ endpoint for Server/Frontend communication
    ///
    /// Server connects via DEALER, Scheduler binds via ROUTER.
    ///
    /// Examples:
    /// - IPC: "ipc:///tmp/rustinfer.ipc"
    /// - TCP: "tcp://*:5556"
    #[arg(long, default_value = "ipc:///tmp/rustinfer.ipc")]
    pub frontend_endpoint: String,

    /// ZeroMQ endpoint for Worker communication
    ///
    /// Examples:
    /// - IPC: "ipc:///tmp/rustinfer-scheduler.ipc"
    /// - TCP: "tcp://*:5555"
    #[arg(long, default_value = "ipc:///tmp/rustinfer-scheduler.ipc")]
    pub worker_endpoint: String,

    /// Worker RPC timeout (milliseconds)
    #[arg(long, default_value_t = 30000)]
    pub worker_timeout_ms: u64,

    /// Number of workers to wait for
    #[arg(long, default_value_t = 1)]
    pub num_workers: usize,
}

/// 模型配置
///
/// **注意**: 不包含 Tokenizer（Tokenizer 属于 Server 层）
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct ModelConfig {
    /// Model path (safetensors format)
    ///
    /// 用途：
    /// 1. 读取 config.json 元数据（eos_token_id, max_position_embeddings）
    /// 2. 传递给 Worker 加载权重
    #[arg(long)]
    pub model_path: String,

    /// Model dtype (bf16, fp16, fp32)
    #[arg(long, default_value = "bf16")]
    pub dtype: String,

    /// Enable Flash Attention
    #[arg(long, default_value_t = true)]
    pub enable_flash_attn: bool,

    /// Optional custom config overrides (JSON string)
    #[arg(long)]
    pub custom_config: Option<String>,
}

/// 显存配置
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct MemoryConfig {
    /// Block size (number of tokens per block)
    ///
    /// 典型值：16（vLLM 默认）
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Total number of GPU blocks
    ///
    /// 由Scheduler确定KVCache的参数，为了防止不同显卡之间的木桶效应。
    #[arg(long, default_value_t = 0)]
    pub total_blocks: usize,

    /// GPU memory utilization ratio (0.0 - 1.0)
    ///
    /// 用于自动计算 total_blocks
    /// 典型值：0.9 表示使用 90% GPU 显存用于 KV Cache
    #[arg(long, default_value_t = 0.9)]
    pub gpu_memory_utilization: f32,

    /// Enable prefix caching (RadixTree)
    #[arg(long, default_value_t = true)]
    pub enable_prefix_cache: bool,

    /// Prefix cache capacity (max tokens)
    #[arg(long, default_value_t = 100000)]
    pub prefix_cache_capacity: usize,

    /// Enable Copy-on-Write (for Beam Search)
    #[arg(long, default_value_t = false)]
    pub enable_cow: bool,
}

/// 调度策略配置
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct SchedulingConfig {
    /// Scheduling policy name
    ///
    /// Options: "continuous" (default), "priority", "fair"
    #[arg(long, default_value = "continuous")]
    pub policy: String,

    /// Max batch size (number of concurrent requests)
    #[arg(long, default_value_t = 256)]
    pub max_batch_size: usize,

    /// Max tokens per step (防止单次计算过大)
    #[arg(long, default_value_t = 4096)]
    pub max_tokens_per_step: usize,

    /// Enable preemption (抢占低优先级请求)
    #[arg(long, default_value_t = true)]
    pub enable_preemption: bool,

    /// Preemption threshold (free blocks)
    ///
    /// 当空闲块低于此值时触发抢占
    #[arg(long, default_value_t = 0)]
    pub preemption_threshold: usize,

    /// Enable swap to CPU memory
    #[arg(long, default_value_t = false)]
    pub enable_swap: bool,

    /// Default request priority
    #[arg(long, default_value_t = 0)]
    pub default_priority: i32,

    /// Idle sleep duration (milliseconds)
    ///
    /// 当没有任务时，主循环休眠时间
    #[arg(long, default_value_t = 1)]
    pub idle_sleep_ms: u64,
}

/// 并行配置
///
/// 支持 Tensor Parallel 和 Pipeline Parallel
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct ParallelismConfig {
    /// Tensor Parallel size
    ///
    /// 将模型权重切分到多个 GPU
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Pipeline Parallel size
    ///
    /// 将模型层切分到多个 GPU
    #[arg(long, default_value_t = 1)]
    pub pp_size: usize,

    /// Tensor Parallel rank (auto-assigned by Worker)
    #[arg(skip)]
    pub tp_rank: Option<usize>,

    /// Pipeline Parallel rank (auto-assigned by Worker)
    #[arg(skip)]
    pub pp_rank: Option<usize>,
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Log format (text, json)
    #[arg(long, default_value = "text")]
    pub log_format: String,

    /// Enable file logging
    #[arg(long, default_value_t = false)]
    pub log_to_file: bool,

    /// Log file path
    #[arg(long, default_value = "/tmp/rustinfer-scheduler.log")]
    pub log_file: String,
}

impl SchedulerConfig {
    /// 从命令行参数加载配置
    pub fn from_args() -> Self {
        Self::parse()
    }

    /// 从配置文件加载（YAML 或 JSON）
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context(format!("Failed to read config file: {}", path))?;

        // 根据文件后缀判断格式
        if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::from_str(&content)
                .context("Failed to parse YAML config")
        } else if path.ends_with(".json") {
            serde_json::from_str(&content)
                .context("Failed to parse JSON config")
        } else {
            anyhow::bail!("Unsupported config file format (use .yaml, .yml, or .json)")
        }
    }

    /// 加载配置（优先使用 --config-file，否则使用命令行参数）
    pub fn load() -> Result<Self> {
        let mut config = Self::from_args();

        // 如果指定了配置文件，从文件加载并合并
        if let Some(config_file) = &config.config_file {
            let file_config = Self::from_file(config_file.to_str().unwrap())?;
            config = config.merge_with(file_config);
        }

        // 验证配置
        config.validate()?;

        Ok(config)
    }

    /// 合并两个配置（命令行参数优先）
    fn merge_with(self, _file_config: Self) -> Self {
        // 对于未显式指定的参数，使用文件中的值
        // 这里简化处理，实际可以用 derive_builder 或其他工具
        // TODO: 实现更精细的合并逻辑
        // 当前版本：命令行参数始终优先
        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 验证 block_size
        if self.memory.block_size == 0 {
            anyhow::bail!("block_size must be greater than 0");
        }

        // 验证 gpu_memory_utilization
        if self.memory.gpu_memory_utilization <= 0.0
            || self.memory.gpu_memory_utilization > 1.0
        {
            anyhow::bail!("gpu_memory_utilization must be in range (0.0, 1.0]");
        }

        // 验证 max_batch_size
        if self.scheduling.max_batch_size == 0 {
            anyhow::bail!("max_batch_size must be greater than 0");
        }

        // 验证并行配置
        if self.parallelism.tp_size == 0 {
            anyhow::bail!("tp_size must be greater than 0");
        }
        if self.parallelism.pp_size == 0 {
            anyhow::bail!("pp_size must be greater than 0");
        }

        let total_workers = self.parallelism.tp_size * self.parallelism.pp_size;
        if self.network.num_workers < total_workers {
            anyhow::bail!(
                "num_workers ({}) must be >= tp_size * pp_size ({})",
                self.network.num_workers,
                total_workers
            );
        }

        // 验证 model_path
        if self.model.model_path.is_empty() {
            anyhow::bail!("model_path is required");
        }

        // 验证 log_level
        match self.logging.log_level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => anyhow::bail!(
                "Invalid log_level: {} (must be trace/debug/info/warn/error)",
                self.logging.log_level
            ),
        }

        Ok(())
    }

    /// 转换为 CoordinatorConfig
    pub fn to_coordinator_config(&self) -> CoordinatorConfig {
        CoordinatorConfig {
            block_size: self.memory.block_size,
            total_blocks: self.memory.total_blocks,
            enable_prefix_cache: self.memory.enable_prefix_cache,
            enable_cow: self.memory.enable_cow,
            default_priority: self.scheduling.default_priority,
            idle_sleep_ms: self.scheduling.idle_sleep_ms,
        }
    }

    /// 转换为 SchedulePolicyConfig
    pub fn to_policy_config(&self) -> SchedulePolicyConfig {
        SchedulePolicyConfig {
            max_batch_size: self.scheduling.max_batch_size,
            max_tokens_per_step: self.scheduling.max_tokens_per_step,
            enable_preemption: self.scheduling.enable_preemption,
            preemption_threshold: self.scheduling.preemption_threshold,
            enable_swap: self.scheduling.enable_swap,
            enable_prefix_cache: self.memory.enable_prefix_cache,
        }
    }

    /// 转换为 MemoryConfig (for MemoryManager)
    pub fn to_memory_config(&self) -> crate::memory::MemoryConfig {
        crate::memory::MemoryConfig {
            total_blocks: self.memory.total_blocks,
            block_size: self.memory.block_size,
            enable_cow: self.memory.enable_cow,
            enable_prefix_cache: self.memory.enable_prefix_cache,
            prefix_cache_capacity: Some(self.memory.prefix_cache_capacity),
        }
    }

    /// 读取模型元数据（从 config.json）
    ///
    /// 返回 (eos_token_id, max_position_embeddings, num_layers, num_heads, head_dim)
    pub fn read_model_metadata(&self) -> Result<ModelMetadata> {
        let config_path = format!("{}/config.json", self.model.model_path);
        let content = std::fs::read_to_string(&config_path)
            .context(format!("Failed to read model config.json: {}", config_path))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse config.json")?;

        let eos_token_id = json["eos_token_id"]
            .as_i64()
            .context("Missing eos_token_id in config.json")? as i32;

        let max_position_embeddings = json["max_position_embeddings"]
            .as_u64()
            .context("Missing max_position_embeddings in config.json")? as usize;

        let num_layers = json["num_hidden_layers"]
            .as_u64()
            .context("Missing num_hidden_layers in config.json")? as usize;

        let num_attention_heads = json["num_attention_heads"]
            .as_u64()
            .context("Missing num_attention_heads in config.json")? as usize;

        let hidden_size = json["hidden_size"]
            .as_u64()
            .context("Missing hidden_size in config.json")? as usize;

        let head_dim = hidden_size / num_attention_heads;

        // 对于 GQA 模型，尝试读取 num_key_value_heads
        let num_kv_heads = json["num_key_value_heads"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or(num_attention_heads);

        Ok(ModelMetadata {
            eos_token_id,
            max_position_embeddings,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// 打印配置摘要
    pub fn print_summary(&self) {
        println!("📋 Scheduler Configuration:");
        println!("  Network:");
        println!("    Frontend endpoint: {}", self.network.frontend_endpoint);
        println!("    Worker endpoint: {}", self.network.worker_endpoint);
        println!("    Num workers: {}", self.network.num_workers);
        println!("  Model:");
        println!("    Path: {}", self.model.model_path);
        println!("    Dtype: {}", self.model.dtype);
        println!("  Memory:");
        println!("    Block size: {}", self.memory.block_size);
        // 对 total_blocks=0 做特殊显示，因为它会在 profile 后自动计算
        if self.memory.total_blocks == 0 {
            println!("    Total blocks: auto (will be determined after profiling)");
        } else {
            println!("    Total blocks: {}", self.memory.total_blocks);
        }
        println!("    Prefix cache: {}", self.memory.enable_prefix_cache);
        println!("  Scheduling:");
        println!("    Policy: {}", self.scheduling.policy);
        println!("    Max batch size: {}", self.scheduling.max_batch_size);
        println!("    Preemption: {}", self.scheduling.enable_preemption);
        println!("  Parallelism:");
        println!("    TP size: {}", self.parallelism.tp_size);
        println!("    PP size: {}", self.parallelism.pp_size);
        println!("  Logging:");
        println!("    Level: {}", self.logging.log_level);
    }
}

/// 模型元数据
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub eos_token_id: i32,
    pub max_position_embeddings: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            model: ModelConfig::default(),
            memory: MemoryConfig::default(),
            scheduling: SchedulingConfig::default(),
            parallelism: ParallelismConfig::default(),
            logging: LoggingConfig::default(),
            config_file: None,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            frontend_endpoint: "ipc:///tmp/rustinfer.ipc".to_string(),
            worker_endpoint: "ipc:///tmp/rustinfer-scheduler.ipc".to_string(),
            worker_timeout_ms: 30000,
            num_workers: 1,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dtype: "bf16".to_string(),
            enable_flash_attn: true,
            custom_config: None,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            total_blocks: 1000,
            gpu_memory_utilization: 0.9,
            enable_prefix_cache: true,
            prefix_cache_capacity: 100000,
            enable_cow: false,
        }
    }
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            policy: "continuous".to_string(),
            max_batch_size: 256,
            max_tokens_per_step: 4096,
            enable_preemption: true,
            preemption_threshold: 0,
            enable_swap: false,
            default_priority: 0,
            idle_sleep_ms: 1,
        }
    }
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            tp_rank: None,
            pp_rank: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            log_format: "text".to_string(),
            log_to_file: false,
            log_file: "/tmp/rustinfer-scheduler.log".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SchedulerConfig::default();
        assert_eq!(config.memory.block_size, 16);
        assert_eq!(config.scheduling.max_batch_size, 256);
        assert_eq!(config.parallelism.tp_size, 1);
    }

    #[test]
    fn test_config_validation() {
        let mut config = SchedulerConfig::default();
        config.model.model_path = "/tmp/test-model".to_string();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid block_size
        config.memory.block_size = 0;
        assert!(config.validate().is_err());
        config.memory.block_size = 16;

        // Invalid gpu_memory_utilization
        config.memory.gpu_memory_utilization = 1.5;
        assert!(config.validate().is_err());
        config.memory.gpu_memory_utilization = 0.9;

        // Invalid max_batch_size
        config.scheduling.max_batch_size = 0;
        assert!(config.validate().is_err());
        config.scheduling.max_batch_size = 256;

        // Invalid log_level
        config.logging.log_level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_to_coordinator_config() {
        let config = SchedulerConfig::default();
        let coord_config = config.to_coordinator_config();

        assert_eq!(coord_config.block_size, config.memory.block_size);
        assert_eq!(coord_config.total_blocks, config.memory.total_blocks);
        assert_eq!(
            coord_config.enable_prefix_cache,
            config.memory.enable_prefix_cache
        );
    }

    #[test]
    fn test_to_policy_config() {
        let config = SchedulerConfig::default();
        let policy_config = config.to_policy_config();

        assert_eq!(
            policy_config.max_batch_size,
            config.scheduling.max_batch_size
        );
        assert_eq!(
            policy_config.enable_preemption,
            config.scheduling.enable_preemption
        );
    }
}
