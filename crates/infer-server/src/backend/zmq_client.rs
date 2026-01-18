//! ZeroMQ Client - Async Streaming Mode
//!
//! 实现与 Scheduler 的双向通信：
//! - 发送方向: HTTP Handler -> Channel -> ZMQ -> Scheduler
//! - 接收方向: Scheduler -> ZMQ -> Request Map -> HTTP Handler

use anyhow::{Context, Result};
use infer_protocol::{SchedulerCommand, SchedulerOutput};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use std::collections::HashMap;

/// 启动 ZMQ 通信循环
///
/// 这是一个后台任务，处理所有与 Scheduler 的通信。
///
/// ## 参数
/// - `scheduler_address`: Scheduler 的 ZeroMQ 地址
/// - `cmd_rx`: 接收来自 HTTP Handler 的命令
/// - `request_map`: 全局请求映射表，用于路由响应
///
/// ## 架构
/// ```text
/// HTTP Handler --cmd_rx--> ZMQ Loop --socket--> Scheduler
///                            |
///                            v
///                        Request Map --tx--> HTTP Handler
/// ```
pub async fn start_zmq_loop(
    scheduler_address: String,
    cmd_rx: mpsc::UnboundedReceiver<SchedulerCommand>,
    request_map: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
) {
    tracing::info!("Starting ZMQ loop, connecting to: {}", scheduler_address);

    // ZeroMQ 是同步库，需要在 spawn_blocking 中运行
    let result = tokio::task::spawn_blocking(move || {
        zmq_blocking_loop(scheduler_address, cmd_rx, request_map)
    }).await;
    
    match result {
        Ok(Ok(_)) => tracing::info!("ZMQ loop exited normally"),
        Ok(Err(e)) => tracing::error!("ZMQ loop error: {:?}", e),
        Err(e) => tracing::error!("ZMQ task panicked: {:?}", e),
    }
}

/// ZMQ 阻塞循环 (在 spawn_blocking 中运行)
fn zmq_blocking_loop(
    scheduler_address: String,
    mut cmd_rx: mpsc::UnboundedReceiver<SchedulerCommand>,
    request_map: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
) -> Result<()> {
    // 创建 ZMQ Context 和 DEALER Socket
    let context = zmq::Context::new();
    let socket = context.socket(zmq::DEALER)
        .context("Failed to create ZMQ socket")?;
    
    // 连接到 Scheduler
    socket.connect(&scheduler_address)
        .context("Failed to connect to Scheduler")?;
    
    // 设置接收超时 (10ms)，避免阻塞
    socket.set_rcvtimeo(10)
        .context("Failed to set recv timeout")?;
    
    tracing::info!("✅ ZMQ connected to Scheduler: {}", scheduler_address);
    
    // 主循环
    loop {
        // 1. 处理发送方向: cmd_rx -> ZMQ
        // 使用 try_recv 避免阻塞
        while let Ok(cmd) = cmd_rx.try_recv() {
            if let Err(e) = send_command(&socket, &cmd) {
                tracing::error!("Failed to send command: {:?}", e);
            }
        }
        
        // 2. 处理接收方向: ZMQ -> Request Map
        match recv_output(&socket) {
            Ok(Some(output)) => {
                // 路由到对应的 HTTP Handler
                route_output(output, &request_map);
            }
            Ok(None) => {
                // 超时，继续循环
            }
            Err(e) => {
                tracing::error!("Failed to receive output: {:?}", e);
            }
        }
        
        // 3. 检查 cmd_rx 是否已关闭
        if cmd_rx.is_closed() && cmd_rx.is_empty() {
            tracing::info!("Command channel closed, exiting ZMQ loop");
            break;
        }
    }
    
    Ok(())
}

/// 发送命令给 Scheduler
fn send_command(socket: &zmq::Socket, cmd: &SchedulerCommand) -> Result<()> {
    // 序列化为 MessagePack
    let data = rmp_serde::to_vec(cmd)
        .context("Failed to serialize command")?;

    // ZeroMQ DEALER 需要发送空帧 + 数据帧
    socket.send(vec![], zmq::SNDMORE)
        .context("Failed to send empty frame")?;
    socket.send(data.as_slice(), 0)
        .context("Failed to send data frame")?;
    
    let req_id = match cmd {
        SchedulerCommand::AddRequest(req) => &req.request_id,
        SchedulerCommand::AbortRequest(id) => id,
    };
    tracing::debug!("Sent command for request: {}", req_id);
    
    Ok(())
}

/// 接收 Scheduler 的输出
fn recv_output(socket: &zmq::Socket) -> Result<Option<SchedulerOutput>> {
    // 接收空帧
    match socket.recv_bytes(0) {
        Ok(_) => {
            // 接收数据帧
            let data = socket.recv_bytes(0)
                .context("Failed to receive data frame")?;
            
            // 反序列化
            let output: SchedulerOutput = rmp_serde::from_slice(&data)
                .context("Failed to deserialize output")?;
            
            Ok(Some(output))
        }
        Err(zmq::Error::EAGAIN) => {
            // 超时，正常情况
            Ok(None)
        }
        Err(e) => {
            Err(e).context("Failed to receive empty frame")
        }
    }
}

/// 路由输出到对应的 HTTP Handler
fn route_output(
    output: SchedulerOutput,
    request_map: &Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
) {
    // 提取 request_id（在移动 output 之前）
    let request_id = get_request_id(&output).to_string();

    // 查找对应的 Sender
    // 注意：这里使用 blocking_read，因为我们在 spawn_blocking 中
    let map = request_map.blocking_read();

    if let Some(tx) = map.get(&request_id) {
        // 发送给 HTTP Handler
        if let Err(e) = tx.send(output) {
            tracing::warn!("Failed to send output to handler for {}: {:?}", request_id, e);
        } else {
            tracing::debug!("Routed output to handler: {}", request_id);
        }
    } else {
        tracing::warn!("Received output for unknown request: {}", request_id);
    }
}

/// 从 SchedulerOutput 提取 request_id
fn get_request_id(output: &SchedulerOutput) -> &str {
    match output {
        SchedulerOutput::Step(step) => &step.request_id,
        SchedulerOutput::Finish(finish) => &finish.request_id,
        SchedulerOutput::Error(error) => &error.request_id,
    }
}
