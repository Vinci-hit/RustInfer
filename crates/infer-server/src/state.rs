//! 共享应用状态

use std::sync::Arc;
use tokenizers::Tokenizer;

use infer_protocol::RustInferConfig;

use crate::client::ZmqClient;

/// 模型信息
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// 模型 ID（用于 /v1/models 和 OpenAI-compatible responses）
    pub model_id: String,
    /// 模型所有者
    pub owned_by: String,
}

/// 共享应用状态，通过 Axum State 注入到所有 handler
pub struct AppState {
    /// ZMQ 客户端（与 Scheduler 通信）
    pub client: ZmqClient,
    /// Tokenizer（Server 端负责 encode/decode）
    pub tokenizer: Arc<Tokenizer>,
    /// 服务器配置（来自共享 TOML）
    pub config: RustInferConfig,
    /// 服务端实际加载的模型类型（从 config.json 解析，用于 chat template）
    pub model_type: String,
    /// 加载的模型信息
    pub model_info: ModelInfo,
}

/// 方便类型别名
pub type SharedState = Arc<AppState>;
