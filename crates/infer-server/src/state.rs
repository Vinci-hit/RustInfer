//! Global State Management
//!
//! 管理 Server 的全局状态，包括：
//! - Tokenizer: 文本处理器
//! - Scheduler Sender: 发送命令给 Scheduler
//! - Request Map: 请求 ID -> 响应 Channel 的映射

use crate::config::ServerConfig;
use crate::processor::TokenizerWrapper;
use anyhow::Result;
use infer_protocol::{SchedulerCommand, SchedulerOutput};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// 全局应用状态
///
/// 在所有 HTTP Handler 间共享。
#[derive(Clone)]
pub struct AppState {
    /// 配置
    pub config: Arc<ServerConfig>,
    
    /// Tokenizer (线程安全)
    pub tokenizer: Arc<TokenizerWrapper>,
    
    /// 发送给 Scheduler 的通道
    /// HTTP Handler 通过这个发送 SchedulerCommand
    pub scheduler_tx: mpsc::UnboundedSender<SchedulerCommand>,
    
    /// 请求映射表: request_id -> response_channel
    /// ZMQ Receiver 通过这个找到对应的 HTTP Handler
    pub request_map: Arc<RwLock<RequestMap>>,
}

/// 请求映射表
///
/// Key: request_id (UUID)
/// Value: 响应通道，用于将 SchedulerOutput 发送回 HTTP Handler
pub type RequestMap = HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>;

impl AppState {
    /// 创建新的应用状态
    pub fn new(
        config: ServerConfig,
        tokenizer: TokenizerWrapper,
        scheduler_tx: mpsc::UnboundedSender<SchedulerCommand>,
    ) -> Self {
        Self {
            config: Arc::new(config),
            tokenizer: Arc::new(tokenizer),
            scheduler_tx,
            request_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 注册新请求
    ///
    /// 返回一个 Receiver，HTTP Handler 通过它接收 SchedulerOutput
    pub async fn register_request(
        &self,
        request_id: String,
    ) -> mpsc::UnboundedReceiver<SchedulerOutput> {
        let (tx, rx) = mpsc::unbounded_channel();
        
        self.request_map.write().await.insert(request_id.clone(), tx);
        
        tracing::debug!("Registered request: {}", request_id);
        
        rx
    }
    
    /// 取消注册请求
    ///
    /// 在 HTTP 连接关闭或请求完成时调用
    pub async fn unregister_request(&self, request_id: &str) {
        self.request_map.write().await.remove(request_id);
        tracing::debug!("Unregistered request: {}", request_id);
    }
    
    /// 获取活跃请求数
    pub async fn active_requests_count(&self) -> usize {
        self.request_map.read().await.len()
    }
}

/// 构建器，用于初始化 AppState
pub struct AppStateBuilder {
    config: ServerConfig,
}

impl AppStateBuilder {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }
    
    /// 构建 AppState
    ///
    /// 需要提供 scheduler_tx，因为它是在启动 ZMQ 循环时创建的
    pub fn build(
        self,
        tokenizer: TokenizerWrapper,
        scheduler_tx: mpsc::UnboundedSender<SchedulerCommand>,
    ) -> Result<AppState> {
        Ok(AppState::new(self.config, tokenizer, scheduler_tx))
    }
}
