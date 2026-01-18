//! ZMQ Frontend Server - 接收来自 HTTP Server 的推理请求
//!
//! 实现 ZeroMQ ROUTER 模式，与 Server 的 DEALER socket 通信。
//!
//! ## 架构
//! ```text
//! ┌──────────────┐        SchedulerCommand
//! │   Server     │ ──────────────────────┐
//! │ (DEALER)     │                       │
//! └──────────────┘                       │
//!      ▲                                ▼
//!      │                        ┌──────────────┐
//!      │                        │  ZMQ Socket │
//!      │                        │   (ROUTER)  │
//!      └────────────────────────┴──────────────┘
//!                                       │
//!                                       ▼
//!                              ┌──────────────┐
//!                              │ Coordinator  │
//!                              └──────────────┘
//! ```

use anyhow::{Context, Result};
use infer_protocol::{SchedulerCommand, SchedulerOutput};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

/// ZMQ Frontend Server
///
/// 监听来自 Server 的请求，并转发给 Coordinator
pub struct ZmqFrontendServer {
    endpoint: String,
    socket: zmq::Socket,
    output_tx: mpsc::UnboundedSender<SchedulerOutput>,
    command_tx: mpsc::UnboundedSender<SchedulerCommand>,
    output_router: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
}

impl ZmqFrontendServer {
    /// 创建并绑定 ZMQ ROUTER socket
    pub async fn bind(
        endpoint: String,
        output_tx: mpsc::UnboundedSender<SchedulerOutput>,
        command_tx: mpsc::UnboundedSender<SchedulerCommand>,
        output_router: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
    ) -> Result<Self> {
        // 清理旧的 IPC 文件
        if endpoint.starts_with("ipc://") {
            let ipc_file = endpoint.trim_start_matches("ipc://");
            if std::path::Path::new(ipc_file).exists() {
                tracing::info!("Cleaning up old IPC file: {}", ipc_file);
                let _ = std::fs::remove_file(ipc_file);
            }
        }

        // 创建 ZMQ Context 和 ROUTER Socket
        let context = zmq::Context::new();
        let socket = context
            .socket(zmq::ROUTER)
            .context("Failed to create ZMQ ROUTER socket")?;

        // 绑定到 endpoint
        socket
            .bind(&endpoint)
            .context(format!("Failed to bind to endpoint: {}", endpoint))?;

        tracing::info!("✅ ZMQ Frontend Server bound to: {}", endpoint);

        // 设置接收超时
        socket
            .set_rcvtimeo(10)
            .context("Failed to set recv timeout")?;

        Ok(Self {
            endpoint,
            socket,
            output_tx,
            command_tx,
            output_router,
        })
    }

    /// 启动接收循环
    ///
    /// 这个方法会阻塞，应该在后台任务中运行
    pub fn start_loop(&self) {
        let endpoint = self.endpoint.clone();

        // 由于 ZMQ 是同步库，需要在 spawn_blocking 中运行
        tokio::task::spawn_blocking({
            let output_tx = self.output_tx.clone();
            let command_tx = self.command_tx.clone();
            let output_router = self.output_router.clone();

            move || {
                // 在这个任务中重新创建 socket
                let context = zmq::Context::new();
                let socket = context.socket(zmq::ROUTER)
                    .expect("Failed to create ZMQ ROUTER socket");

                // 绑定到 endpoint
                socket.bind(&endpoint)
                    .expect(&format!("Failed to bind to endpoint: {}", endpoint));

                // 设置接收超时
                socket.set_rcvtimeo(10)
                    .expect("Failed to set recv timeout");

                zmq_receive_loop(socket, endpoint, output_tx, command_tx, output_router)
            }
        });
    }

    /// 注册请求的输出通道
    ///
    /// 当收到新请求时，调用此方法注册输出通道
    pub fn register_request(
        &self,
        request_id: String,
        output_tx: mpsc::UnboundedSender<SchedulerOutput>,
    ) {
        let mut map = self.output_router.write().unwrap();
        map.insert(request_id, output_tx);
    }

    /// 注销请求
    pub fn unregister_request(&self, request_id: &str) {
        let mut map = self.output_router.write().unwrap();
        map.remove(request_id);
    }
}

/// ZMQ 接收循环 (在 spawn_blocking 中运行)
fn zmq_receive_loop(
    socket: zmq::Socket,
    endpoint: String,
    output_tx: mpsc::UnboundedSender<SchedulerOutput>,
    command_tx: mpsc::UnboundedSender<SchedulerCommand>,
    output_router: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
) -> Result<()> {
    tracing::info!("ZMQ Frontend receiving loop started on: {}", endpoint);

    loop {
        // 接收 ROUTER 消息：identity + empty + data
        match recv_router_message(&socket) {
            Ok(Some((identity, cmd))) => {
                // 处理命令
                match handle_command(&cmd, &command_tx, &output_tx, &output_router) {
                    Ok(output_tx_opt) => {
                        if let Some(tx) = output_tx_opt {
                            // 注册输出通道
                            let mut map = output_router.write().unwrap();
                            if let Some(request_id) = extract_request_id(&cmd) {
                                map.insert(request_id, tx);
                            }
                        }

                        // 发送响应（ACK）
                        if let Err(e) = send_router_response(&socket, &identity, &cmd) {
                            tracing::error!("Failed to send ACK: {:?}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to handle command: {:?}", e);
                    }
                }
            }
            Ok(None) => {
                // 超时，继续循环
            }
            Err(e) => {
                tracing::error!("Failed to receive message: {:?}", e);
            }
        }
    }
}

/// 接收 ROUTER 消息（identity + empty + data）
fn recv_router_message(socket: &zmq::Socket) -> Result<Option<(Vec<u8>, SchedulerCommand)>> {
    // 接收 identity
    let identity = match socket.recv_bytes(0) {
        Ok(id) => id,
        Err(zmq::Error::EAGAIN) => return Ok(None),
        Err(e) => return Err(anyhow::anyhow!("Failed to receive identity: {}", e)),
    };

    // 接收空帧
    if let Err(e) = socket.recv_bytes(0) {
        if e == zmq::Error::EAGAIN {
            return Ok(None);
        }
        return Err(anyhow::anyhow!("Failed to receive empty frame: {}", e));
    }

    // 接收数据帧
    let data = socket.recv_bytes(0)
        .map_err(|e| anyhow::anyhow!("Failed to receive data frame: {}", e))?;

    // 反序列化为 SchedulerCommand
    let cmd: SchedulerCommand = rmp_serde::from_slice(&data)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize SchedulerCommand: {}", e))?;

    Ok(Some((identity, cmd)))
}

/// 处理命令，返回需要注册的输出 channel（如果有）
fn handle_command(
    cmd: &SchedulerCommand,
    command_tx: &mpsc::UnboundedSender<SchedulerCommand>,
    _output_tx: &mpsc::UnboundedSender<SchedulerOutput>,
    _output_router: &Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
) -> Result<Option<mpsc::UnboundedSender<SchedulerOutput>>> {
    match cmd {
        SchedulerCommand::AddRequest(req) => {
            tracing::info!("Received AddRequest: {}", req.request_id);

            // 转发命令到 Coordinator
            command_tx.send(cmd.clone())
                .map_err(|e| anyhow::anyhow!("Failed to send command: {}", e))?;

            // 创建输出 channel
            let (tx, _rx) = mpsc::unbounded_channel();
            Ok(Some(tx))
        }
        SchedulerCommand::AbortRequest(request_id) => {
            tracing::warn!("Received AbortRequest: {} (not implemented yet)", request_id);

            // 转发命令到 Coordinator
            command_tx.send(cmd.clone())
                .map_err(|e| anyhow::anyhow!("Failed to send command: {}", e))?;

            Ok(None)
        }
    }
}

/// 发送 ROUTER 响应（ACK）
fn send_router_response(
    socket: &zmq::Socket,
    identity: &[u8],
    _cmd: &SchedulerCommand,
) -> Result<()> {
    // 发送 identity
    socket.send(identity, zmq::SNDMORE)
        .context("Failed to send identity")?;

    // 发送空帧
    socket.send(vec![], zmq::SNDMORE)
        .context("Failed to send empty frame")?;

    // 发送数据（ACK）
    let ack = b"ACK";
    socket.send(&ack[..], 0)
        .context("Failed to send ACK")?;

    Ok(())
}

/// 从命令中提取 request_id
fn extract_request_id(cmd: &SchedulerCommand) -> Option<String> {
    match cmd {
        SchedulerCommand::AddRequest(req) => Some(req.request_id.clone()),
        SchedulerCommand::AbortRequest(id) => Some(id.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_request_id() {
        use infer_protocol::{InferRequest, SamplingParams, StoppingCriteria};

        let req = InferRequest {
            request_id: "test-123".to_string(),
            input_token_ids: vec![1, 2, 3],
            sampling_params: SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                top_k: 50,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                seed: None,
            },
            stopping_criteria: StoppingCriteria {
                stop_token_ids: vec![2],
                max_new_tokens: 100,
                ignore_eos: false,
            },
            adapter_id: None,
        };

        let cmd = SchedulerCommand::AddRequest(req.clone());
        assert_eq!(extract_request_id(&cmd), Some(req.request_id));

        let abort_cmd = SchedulerCommand::AbortRequest("abc".to_string());
        assert_eq!(extract_request_id(&abort_cmd), Some("abc".to_string()));
    }
}
