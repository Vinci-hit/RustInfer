//! Worker Proxy - ZeroMQ-based Worker Communication
//!
//! 负责与 GPU Worker 进行 ZeroMQ 通信
//!
//! # Architecture
//!
//! ```text
//! Scheduler (RouterSocket)
//!      ↓
//! WorkerProxy
//!      ↓
//! Worker (DealerSocket)
//! ```
//!
//! # Protocol
//!
//! 1. Handshake: Worker 主动发起注册
//! 2. LoadModel: 加载模型到 GPU
//! 3. Profile: 探测显存使用
//! 4. InitKVCache: 初始化 KV Cache
//! 5. Forward: 执行推理

use infer_protocol::{
    WorkerCommand, WorkerResponse,
    ForwardParams, ForwardResult,
    ModelLoadParams, ModelLoadedInfo,
    ProfileParams, ProfileResult,
    InitKVCacheParams, KVCacheInfo,
    WorkerRegistrationAck,
    WorkerStatus,
};
use anyhow::{Result, Context};
use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use tracing::{debug, info, warn, error};

/// Worker 连接状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerConnectionState {
    /// 未连接
    Disconnected,
    /// 已连接，等待握手
    Connected,
    /// 握手成功，已注册
    Registered,
    /// 模型已加载
    ModelLoaded,
    /// KV Cache 已初始化，可以执行推理
    Ready,
    /// 错误状态
    Error,
}

/// Worker 信息
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// Worker ID
    pub worker_id: String,
    /// Rank
    pub rank: u32,
    /// Device ID
    pub device_id: u32,
    /// Device 类型 (cuda/cpu)
    pub device_type: String,
    /// 连接状态
    pub state: WorkerConnectionState,
    /// ZeroMQ 地址（用于路由）
    pub zmq_address: Vec<u8>,
}

/// Worker Proxy
///
/// 封装与单个或多个 GPU Worker 的 ZeroMQ 通信
pub struct WorkerProxy {
    /// ZeroMQ Router socket
    socket: zeromq::RouterSocket,

    /// 已连接的 Workers (worker_id -> WorkerInfo)
    workers: HashMap<String, WorkerInfo>,

    /// 绑定的 endpoint
    endpoint: String,

    /// 默认超时时间（毫秒）
    timeout_ms: u64,
}

impl WorkerProxy {
    /// 创建新的 WorkerProxy
    ///
    /// # Arguments
    /// * `endpoint` - ZeroMQ endpoint (e.g., "ipc:///tmp/scheduler.ipc" 或 "tcp://*:5555")
    /// * `timeout_ms` - RPC 超时时间
    pub async fn new(endpoint: String, timeout_ms: u64) -> Result<Self> {
        let mut socket = zeromq::RouterSocket::new();

        info!("WorkerProxy binding to {}", endpoint);
        socket.bind(&endpoint).await
            .context("Failed to bind RouterSocket")?;
        info!("WorkerProxy bound successfully");

        Ok(Self {
            socket,
            workers: HashMap::new(),
            endpoint,
            timeout_ms,
        })
    }

    /// 等待 Worker 注册（握手）
    ///
    /// 阻塞等待 Worker 发送 Register 请求，返回注册的 Worker 信息
    pub async fn wait_for_registration(&mut self) -> Result<WorkerInfo> {
        info!("Waiting for Worker registration...");

        loop {
            let msg = self.socket.recv().await
                .context("Failed to receive registration message")?;

            // RouterSocket 收到的消息格式：[address, empty_frame, payload]
            let frames = msg.into_vec();
            if frames.len() < 3 {
                warn!("Invalid message format: expected at least 3 frames, got {}", frames.len());
                continue;
            }

            let address = frames[0].clone();
            let payload = &frames[2]; // 跳过 empty delimiter frame

            // 反序列化命令
            let command: WorkerCommand = match bincode::deserialize(payload) {
                Ok(cmd) => cmd,
                Err(e) => {
                    error!("Failed to deserialize command: {}", e);
                    continue;
                }
            };

            // 处理注册请求
            match command {
                WorkerCommand::Register(registration) => {
                    info!(
                        "Received registration from Worker: id={}, rank={}, device={}:{}",
                        registration.worker_id,
                        registration.rank,
                        registration.device_type,
                        registration.device_id
                    );

                    // 创建 Worker 信息
                    let worker_info = WorkerInfo {
                        worker_id: registration.worker_id.clone(),
                        rank: registration.rank,
                        device_id: registration.device_id,
                        device_type: registration.device_type.clone(),
                        state: WorkerConnectionState::Registered,
                        zmq_address: address.to_vec(), // Bytes -> Vec<u8>
                    };

                    // 发送 ACK
                    let ack = WorkerRegistrationAck {
                        status: "ok".to_string(),
                        message: format!("Worker {} registered successfully", registration.worker_id),
                        scheduler_protocol_version: Some("0.1.0".to_string()),
                        assigned_worker_id: None,
                    };

                    let response = WorkerResponse::RegisterAck(ack);
                    self.send_to_worker(&address, response).await?;

                    // 保存 Worker 信息
                    self.workers.insert(registration.worker_id.clone(), worker_info.clone());

                    info!("Worker {} registered successfully", registration.worker_id);
                    return Ok(worker_info);
                }
                _ => {
                    warn!("Expected Register command, got {:?}", command);
                    continue;
                }
            }
        }
    }

    /// 加载模型
    ///
    /// 向指定 Worker 发送 LoadModel 命令
    pub async fn load_model(
        &mut self,
        worker_id: &str,
        params: ModelLoadParams,
    ) -> Result<ModelLoadedInfo> {
        debug!("Loading model on Worker {}", worker_id);

        let zmq_address = self.get_worker(worker_id)?.zmq_address.clone();
        let command = WorkerCommand::LoadModel(params);

        let response = self.send_command(&zmq_address, command).await?;

        match response {
            WorkerResponse::ModelLoaded(info) => {
                // 更新状态
                if let Some(w) = self.workers.get_mut(worker_id) {
                    w.state = WorkerConnectionState::ModelLoaded;
                }
                info!("Model loaded on Worker {}: {} MB", worker_id, info.memory_used / 1024 / 1024);
                Ok(info)
            }
            WorkerResponse::Error(err) => {
                error!("Failed to load model on Worker {}: {:?}", worker_id, err);
                Err(anyhow::anyhow!("Model load failed: {}", err.message))
            }
            _ => Err(anyhow::anyhow!("Unexpected response: {:?}", response)),
        }
    }

    /// 探测显存使用
    ///
    /// 向指定 Worker 发送 Profile 命令
    pub async fn profile(
        &mut self,
        worker_id: &str,
        params: ProfileParams,
    ) -> Result<ProfileResult> {
        debug!("Profiling Worker {}", worker_id);

        let zmq_address = self.get_worker(worker_id)?.zmq_address.clone();
        let command = WorkerCommand::Profile(params);

        let response = self.send_command(&zmq_address, command).await?;

        match response {
            WorkerResponse::ProfileCompleted(result) => {
                info!(
                    "Profile completed on Worker {}: {} MB available for KV Cache",
                    worker_id,
                    result.available_kv_cache_memory / 1024 / 1024
                );
                Ok(result)
            }
            WorkerResponse::Error(err) => {
                error!("Failed to profile Worker {}: {:?}", worker_id, err);
                Err(anyhow::anyhow!("Profile failed: {}", err.message))
            }
            _ => Err(anyhow::anyhow!("Unexpected response: {:?}", response)),
        }
    }

    /// 初始化 KV Cache
    ///
    /// 向指定 Worker 发送 InitKVCache 命令
    pub async fn init_kv_cache(
        &mut self,
        worker_id: &str,
        params: InitKVCacheParams,
    ) -> Result<KVCacheInfo> {
        debug!("Initializing KV Cache on Worker {}", worker_id);

        let zmq_address = self.get_worker(worker_id)?.zmq_address.clone();
        let command = WorkerCommand::InitKVCache(params);

        let response = self.send_command(&zmq_address, command).await?;

        match response {
            WorkerResponse::KVCacheInitialized(info) => {
                // 更新状态
                if let Some(w) = self.workers.get_mut(worker_id) {
                    w.state = WorkerConnectionState::Ready;
                }
                info!(
                    "KV Cache initialized on Worker {}: {} blocks, {} MB",
                    worker_id,
                    info.allocated_blocks,
                    info.memory_used / 1024 / 1024
                );
                Ok(info)
            }
            WorkerResponse::Error(err) => {
                error!("Failed to init KV Cache on Worker {}: {:?}", worker_id, err);
                Err(anyhow::anyhow!("KV Cache init failed: {}", err.message))
            }
            _ => Err(anyhow::anyhow!("Unexpected response: {:?}", response)),
        }
    }

    /// 执行 Forward 推理
    ///
    /// 向指定 Worker 发送 Forward 命令
    pub async fn forward(
        &mut self,
        worker_id: &str,
        params: ForwardParams,
    ) -> Result<ForwardResult> {
        let zmq_address = self.get_worker(worker_id)?.zmq_address.clone();
        let command = WorkerCommand::Forward(params);

        let response = self.send_command(&zmq_address, command).await?;

        match response {
            WorkerResponse::ForwardCompleted(result) => Ok(result),
            WorkerResponse::Error(err) => {
                error!("Forward failed on Worker {}: {:?}", worker_id, err);
                Err(anyhow::anyhow!("Forward failed: {}", err.message))
            }
            _ => Err(anyhow::anyhow!("Unexpected response: {:?}", response)),
        }
    }

    /// 获取 Worker 状态
    pub async fn get_status(&mut self, worker_id: &str) -> Result<WorkerStatus> {
        let zmq_address = self.get_worker(worker_id)?.zmq_address.clone();
        let command = WorkerCommand::GetStatus;

        let response = self.send_command(&zmq_address, command).await?;

        match response {
            WorkerResponse::Status(status) => Ok(status),
            WorkerResponse::Error(err) => {
                Err(anyhow::anyhow!("Get status failed: {}", err.message))
            }
            _ => Err(anyhow::anyhow!("Unexpected response: {:?}", response)),
        }
    }

    /// 健康检查
    pub async fn health_check(&mut self, worker_id: &str) -> Result<bool> {
        let zmq_address = self.get_worker(worker_id)?.zmq_address.clone();
        let command = WorkerCommand::HealthCheck;

        let response = self.send_command(&zmq_address, command).await?;

        match response {
            WorkerResponse::Healthy => Ok(true),
            _ => Ok(false),
        }
    }

    /// 发送命令并等待响应（带超时）
    async fn send_command(
        &mut self,
        worker_address: &[u8],
        command: WorkerCommand,
    ) -> Result<WorkerResponse> {
        // 序列化命令
        let payload = bincode::serialize(&command)
            .context("Failed to serialize command")?;

        // 发送到 Worker (RouterSocket格式：[address, empty_frame, payload])
        // 构建多帧消息
        let mut msg = ZmqMessage::try_from(worker_address.to_vec())?;
        msg.push_back(vec![].into()); // empty delimiter frame
        msg.push_back(payload.into());

        self.socket.send(msg).await
            .context("Failed to send command")?;

        // 等待响应（带超时）
        let timeout_duration = Duration::from_millis(self.timeout_ms);
        let recv_future = self.socket.recv();

        let msg = timeout(timeout_duration, recv_future)
            .await
            .context("Worker response timeout")?
            .context("Failed to receive response")?;

        // 解析响应
        let frames = msg.into_vec();
        if frames.len() < 3 {
            return Err(anyhow::anyhow!("Invalid response format"));
        }

        let payload = &frames[2];
        let response: WorkerResponse = bincode::deserialize(payload)
            .context("Failed to deserialize response")?;

        Ok(response)
    }

    /// 发送响应到指定 Worker（用于 Register ACK）
    async fn send_to_worker(
        &mut self,
        worker_address: &[u8],
        response: WorkerResponse,
    ) -> Result<()> {
        let payload = bincode::serialize(&response)
            .context("Failed to serialize response")?;

        // RouterSocket 发送格式：[address, empty_frame, payload]
        let mut msg = ZmqMessage::try_from(worker_address.to_vec())?;
        msg.push_back(vec![].into()); // empty delimiter frame
        msg.push_back(payload.into());

        self.socket.send(msg).await
            .context("Failed to send response")?;

        Ok(())
    }

    /// 获取 Worker 信息
    fn get_worker(&self, worker_id: &str) -> Result<&WorkerInfo> {
        self.workers.get(worker_id)
            .ok_or_else(|| anyhow::anyhow!("Worker {} not found", worker_id))
    }

    /// 获取所有 Workers
    pub fn workers(&self) -> &HashMap<String, WorkerInfo> {
        &self.workers
    }

    /// 获取 Worker 数量
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// 检查 Worker 是否 Ready
    pub fn is_worker_ready(&self, worker_id: &str) -> bool {
        self.workers.get(worker_id)
            .map(|w| w.state == WorkerConnectionState::Ready)
            .unwrap_or(false)
    }

    /// 获取绑定的 endpoint 地址
    ///
    /// 用于日志记录、状态查询等场景
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_proxy_creation() {
        // 使用临时 IPC endpoint
        let endpoint = format!("ipc:///tmp/test-scheduler-{}.ipc", std::process::id());
        let proxy = WorkerProxy::new(endpoint, 5000).await;
        assert!(proxy.is_ok());
    }

    #[test]
    fn test_worker_info() {
        let info = WorkerInfo {
            worker_id: "worker-0".to_string(),
            rank: 0,
            device_id: 0,
            device_type: "cuda".to_string(),
            state: WorkerConnectionState::Registered,
            zmq_address: vec![1, 2, 3],
        };

        assert_eq!(info.worker_id, "worker-0");
        assert_eq!(info.state, WorkerConnectionState::Registered);
    }
}
