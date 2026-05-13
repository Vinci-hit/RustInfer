//! ZMQ 客户端 — 通过 ZMQ DEALER socket 与 Scheduler 通信
//!
//! 架构：
//! - Tokio handler 通过 `std::sync::mpsc` 向 ZMQ 专用线程发送请求
//! - ZMQ 线程维护 pending 请求表，按 request_id 匹配响应
//! - 非流式请求用 `oneshot` 回传完整响应
//! - 流式请求用 `tokio::sync::mpsc` 逐 chunk 回传

use anyhow::Result;
use infer_protocol::scheduler_to_server::{ChunkType, InferenceResponse, StreamChunk};
use infer_protocol::server_to_scheduler::InferenceRequest;
use std::collections::HashMap;
use std::thread;
use tokio::sync::{mpsc, oneshot};

use super::InferClient;

/// 发送到 ZMQ 线程的请求信封
enum RequestEnvelope {
    /// 非流式：完整响应通过 oneshot 回传
    Oneshot {
        request: InferenceRequest,
        reply_tx: oneshot::Sender<InferenceResponse>,
    },
    /// 流式：逐 chunk 通过 mpsc 回传
    Stream {
        request: InferenceRequest,
        chunk_tx: mpsc::Sender<StreamChunk>,
    },
}

/// ZMQ pending 请求记录
enum PendingRequest {
    Oneshot(oneshot::Sender<InferenceResponse>),
    Stream(mpsc::Sender<StreamChunk>),
}

/// ZMQ 客户端
pub struct ZmqClient {
    /// 向 ZMQ 线程发送请求的通道
    request_tx: std::sync::mpsc::Sender<RequestEnvelope>,
    /// 请求超时时间
    timeout: std::time::Duration,
}

impl ZmqClient {
    /// 创建新的 ZMQ 客户端并启动后台线程
    pub async fn new(endpoint: &str, timeout_secs: u64) -> Result<Self> {
        let endpoint = endpoint.to_string();
        let (request_tx, request_rx) = std::sync::mpsc::channel::<RequestEnvelope>();

        // 在专用线程中运行 ZMQ 操作（ZMQ socket 不是 Send，不能跨线程）
        thread::Builder::new()
            .name("zmq-client".to_string())
            .spawn(move || {
                if let Err(e) = Self::zmq_thread(endpoint, request_rx) {
                    tracing::error!("ZMQ thread exited with error: {:?}", e);
                }
            })?;

        Ok(Self {
            request_tx,
            timeout: std::time::Duration::from_secs(timeout_secs),
        })
    }

    /// ZMQ 专用线程 — 处理所有 ZMQ socket 操作
    fn zmq_thread(
        endpoint: String,
        request_rx: std::sync::mpsc::Receiver<RequestEnvelope>,
    ) -> Result<()> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::DEALER)?;
        socket.connect(&endpoint)?;
        // 非阻塞接收超时 (10ms)
        socket.set_rcvtimeo(10)?;

        tracing::info!("ZMQ client connected to {}", endpoint);

        let mut pending: HashMap<String, PendingRequest> = HashMap::new();

        loop {
            // 1. 从 Tokio 侧接收新请求（非阻塞）
            while let Ok(envelope) = request_rx.try_recv() {
                match envelope {
                    RequestEnvelope::Oneshot { request, reply_tx } => {
                        let request_id = request.request_id.clone();
                        pending.insert(request_id.clone(), PendingRequest::Oneshot(reply_tx));
                        Self::send_request(&socket, &request);
                    }
                    RequestEnvelope::Stream { request, chunk_tx } => {
                        let request_id = request.request_id.clone();
                        pending.insert(request_id.clone(), PendingRequest::Stream(chunk_tx));
                        Self::send_request(&socket, &request);
                    }
                }
            }

            // 2. 从 Scheduler 接收响应（非阻塞，10ms 超时）
            match socket.recv_bytes(0) {
                Ok(_empty_frame) => {
                    // DEALER: [empty delimiter, data]
                    match socket.recv_bytes(0) {
                        Ok(data) => {
                            Self::handle_response(&mut pending, &data);
                        }
                        Err(zmq::Error::EAGAIN) => {}
                        Err(e) => {
                            tracing::error!("ZMQ recv data error: {:?}", e);
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // 超时，继续循环
                }
                Err(e) => {
                    tracing::error!("ZMQ recv error: {:?}", e);
                }
            }
        }
    }

    /// 序列化并发送请求到 ZMQ socket
    fn send_request(socket: &zmq::Socket, request: &InferenceRequest) {
        let data = match rmp_serde::to_vec(request) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("Failed to serialize request {}: {:?}", request.request_id, e);
                return;
            }
        };

        // DEALER 发送: [empty frame, data]
        if let Err(e) = socket.send(&b""[..], zmq::SNDMORE) {
            tracing::error!("ZMQ send empty frame error: {:?}", e);
            return;
        }
        if let Err(e) = socket.send(&data, 0) {
            tracing::error!("ZMQ send data error: {:?}", e);
            return;
        }

        tracing::debug!("Sent request: {}", request.request_id);
    }

    /// 处理从 Scheduler 收到的响应数据
    fn handle_response(pending: &mut HashMap<String, PendingRequest>, data: &[u8]) {
        // 尝试解析为 InferenceResponse (非流式完整响应)
        if let Ok(response) = rmp_serde::from_slice::<InferenceResponse>(data) {
            let request_id = response.request_id.clone();
            tracing::debug!("Received response: {}", request_id);

            if let Some(pending_req) = pending.remove(&request_id) {
                match pending_req {
                    PendingRequest::Oneshot(tx) => {
                        let _ = tx.send(response);
                    }
                    PendingRequest::Stream(tx) => {
                        // Scheduler 发了完整响应给一个流式请求
                        // 转换为 Done chunk 后关闭
                        let chunk = StreamChunk {
                            request_id: request_id.clone(),
                            chunk_type: ChunkType::Done,
                            token_id: None,
                            finish_reason: response.finish_reason.clone(),
                            metrics: Some(response.metrics),
                        };
                        let _ = tx.blocking_send(chunk);
                    }
                }
            } else {
                tracing::warn!("Received response for unknown request: {}", request_id);
            }
            return;
        }

        // 尝试解析为 StreamChunk (流式逐 token 响应)
        if let Ok(chunk) = rmp_serde::from_slice::<StreamChunk>(data) {
            let request_id = chunk.request_id.clone();
            let is_done = matches!(chunk.chunk_type, ChunkType::Done | ChunkType::Error);

            tracing::debug!("Received stream chunk: {} (done={})", request_id, is_done);

            if let Some(pending_req) = pending.get(&request_id) {
                match pending_req {
                    PendingRequest::Stream(tx) => {
                        if tx.blocking_send(chunk).is_err() {
                            // 客户端已断开，清理 pending
                            tracing::debug!("Stream receiver dropped for {}", request_id);
                            pending.remove(&request_id);
                            return;
                        }
                    }
                    PendingRequest::Oneshot(_) => {
                        tracing::warn!("Received stream chunk for non-stream request: {}", request_id);
                    }
                }

                // Done/Error 后清理
                if is_done {
                    pending.remove(&request_id);
                }
            } else {
                tracing::warn!("Received chunk for unknown request: {}", request_id);
            }
            return;
        }

        tracing::error!("Failed to deserialize response (neither InferenceResponse nor StreamChunk)");
    }
}

impl InferClient for ZmqClient {
    /// 非流式推理
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let (tx, rx) = oneshot::channel();

        self.request_tx
            .send(RequestEnvelope::Oneshot {
                request: req,
                reply_tx: tx,
            })
            .map_err(|_| anyhow::anyhow!("ZMQ thread disconnected"))?;

        // 等待响应（带超时）
        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow::anyhow!("Response channel closed")),
            Err(_) => Err(anyhow::anyhow!(
                "Request timeout after {}s",
                self.timeout.as_secs()
            )),
        }
    }

    /// 流式推理
    async fn infer_stream(&self, req: InferenceRequest) -> Result<mpsc::Receiver<StreamChunk>> {
        let (tx, rx) = mpsc::channel(64);

        self.request_tx
            .send(RequestEnvelope::Stream {
                request: req,
                chunk_tx: tx,
            })
            .map_err(|_| anyhow::anyhow!("ZMQ thread disconnected"))?;

        Ok(rx)
    }
}
