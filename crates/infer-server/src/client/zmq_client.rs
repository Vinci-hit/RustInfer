//! ZMQ 客户端 — 通过 ZMQ DEALER socket 与 Scheduler 通信

use anyhow::Result;
use infer_protocol::scheduler_to_server::{
    ChunkType, InferenceMetrics, InferenceResponse, StreamChunk,
};
use infer_protocol::server_to_scheduler::{
    CancelReason, CancelRequest as ServerCancelRequest, InferenceRequest, ServerCommand,
};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TrySendError as TokioTrySendError;
use tokio::sync::{mpsc, oneshot};

use super::InferClient;

const CLIENT_COMMAND_BUFFER: usize = 1024;
const STREAM_CHUNK_BUFFER: usize = 64;
/// 上限保护：即使没有 stream 也最多阻塞这么久（防御性，正常情况下应靠 wakeup/POLLIN 唤醒）。
const POLL_MAX_TIMEOUT: Duration = Duration::from_millis(1);

enum RequestEnvelope {
    Oneshot {
        request: InferenceRequest,
        reply_tx: oneshot::Sender<InferenceResponse>,
    },
    Stream {
        request: InferenceRequest,
        chunk_tx: mpsc::Sender<StreamChunk>,
    },
    Cancel {
        request_id: String,
        reason: CancelReason,
    },
}

enum PendingRequest {
    Oneshot {
        tx: oneshot::Sender<InferenceResponse>,
        deadline: Instant,
    },
    Stream {
        tx: mpsc::Sender<StreamChunk>,
        deadline: Instant,
    },
}

pub struct StreamHandle {
    request_id: String,
    rx: mpsc::Receiver<StreamChunk>,
    command_tx: SyncSender<RequestEnvelope>,
    waker: std::sync::Arc<Waker>,
    finished: bool,
}

impl StreamHandle {
    pub async fn recv(&mut self) -> Option<StreamChunk> {
        self.rx.recv().await
    }

    pub fn mark_finished(&mut self) {
        self.finished = true;
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if self
            .command_tx
            .try_send(RequestEnvelope::Cancel {
                request_id: self.request_id.clone(),
                reason: CancelReason::ClientDisconnected,
            })
            .is_ok()
        {
            self.waker.wake();
        }
    }
}

/// No-op wake handle. The ZMQ thread uses a short finite poll timeout to avoid
/// cross-thread ZMQ wake sockets, which can trip libzmq's signaler assertions.
struct Waker;

impl Waker {
    fn wake(&self) {}
}

pub struct ZmqClient {
    command_tx: SyncSender<RequestEnvelope>,
    waker: std::sync::Arc<Waker>,
    timeout: Duration,
}

impl ZmqClient {
    pub async fn new(endpoint: &str, timeout_secs: u64) -> Result<Self> {
        let endpoint = endpoint.to_string();
        let timeout = Duration::from_secs(timeout_secs);
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<RequestEnvelope>(CLIENT_COMMAND_BUFFER);

        let waker = std::sync::Arc::new(Waker);

        thread::Builder::new()
            .name("zmq-client".to_string())
            .spawn(move || {
                if let Err(e) = Self::zmq_thread(endpoint, command_rx, timeout) {
                    tracing::error!("ZMQ thread exited with error: {:?}", e);
                }
            })?;

        Ok(Self {
            command_tx,
            waker,
            timeout,
        })
    }

    fn zmq_thread(
        endpoint: String,
        command_rx: Receiver<RequestEnvelope>,
        timeout: Duration,
    ) -> Result<()> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::DEALER)?;
        socket.connect(&endpoint)?;

        tracing::info!("ZMQ client connected to {}", endpoint);
        let mut pending: HashMap<String, PendingRequest> = HashMap::new();

        loop {
            // 1) 处理所有已到达的命令（无论是被 wakeup 唤醒还是被 POLLIN 唤醒，都先排空 channel）
            match Self::drain_commands(&socket, &command_rx, &mut pending, timeout) {
                Ok(true) => {}              // 仍在运行
                Ok(false) => return Ok(()), // command_tx 全部 drop，退出
                Err(e) => tracing::error!("drain_commands error: {:?}", e),
            }

            // 2) 计算下一次 poll 的超时：取 stream deadline 的最近值，无则 POLL_MAX_TIMEOUT。
            let poll_timeout_ms = Self::next_poll_timeout_ms(&pending);

            // 3) 监听 DEALER；命令通道靠短 timeout 轮询，避免跨线程 ZMQ wake socket。
            let mut items = [socket.as_poll_item(zmq::POLLIN)];
            match zmq::poll(&mut items, poll_timeout_ms) {
                Ok(_) => {}
                Err(zmq::Error::EINTR) => continue,
                Err(e) => {
                    tracing::error!("zmq::poll error: {:?}", e);
                    continue;
                }
            }

            // 4) DEALER 有数据：尽量多收（一次 poll 唤醒可能对应多个消息到达）
            if items[0].is_readable() {
                Self::drain_dealer(&socket, &mut pending, timeout);
            }

            // 5) 兜底：处理 stream 超时
            Self::cancel_timed_out_requests(&socket, &mut pending);
        }
    }

    /// 排空 command_rx。返回 `Ok(false)` 表示 channel 已断开，应退出线程。
    fn drain_commands(
        socket: &zmq::Socket,
        command_rx: &Receiver<RequestEnvelope>,
        pending: &mut HashMap<String, PendingRequest>,
        timeout: Duration,
    ) -> Result<bool> {
        loop {
            match command_rx.try_recv() {
                Ok(RequestEnvelope::Oneshot { request, reply_tx }) => {
                    let request_id = request.request_id.clone();
                    pending.insert(
                        request_id.clone(),
                        PendingRequest::Oneshot {
                            tx: reply_tx,
                            deadline: Instant::now() + timeout,
                        },
                    );
                    if let Err(e) = Self::send_command(socket, &ServerCommand::Infer(request)) {
                        tracing::error!("Failed to send request {}: {:?}", request_id, e);
                        pending.remove(&request_id);
                    }
                }
                Ok(RequestEnvelope::Stream { request, chunk_tx }) => {
                    let request_id = request.request_id.clone();
                    pending.insert(
                        request_id.clone(),
                        PendingRequest::Stream {
                            tx: chunk_tx,
                            deadline: Instant::now() + timeout,
                        },
                    );
                    if let Err(e) = Self::send_command(socket, &ServerCommand::Infer(request)) {
                        tracing::error!("Failed to send stream request {}: {:?}", request_id, e);
                        pending.remove(&request_id);
                    }
                }
                Ok(RequestEnvelope::Cancel { request_id, reason }) => {
                    pending.remove(&request_id);
                    if let Err(e) = Self::send_cancel(socket, &request_id, reason) {
                        tracing::error!("Failed to send cancel {}: {:?}", request_id, e);
                    }
                }
                Err(TryRecvError::Empty) => return Ok(true),
                Err(TryRecvError::Disconnected) => return Ok(false),
            }
        }
    }

    /// DEALER 可读时尽量多收，直到 EAGAIN。
    fn drain_dealer(
        socket: &zmq::Socket,
        pending: &mut HashMap<String, PendingRequest>,
        timeout: Duration,
    ) {
        loop {
            // DEALER 收到的第一帧是空 delimiter（来自 ROUTER 的回程）
            match socket.recv_bytes(zmq::DONTWAIT) {
                Ok(_delim) => match socket.recv_bytes(zmq::DONTWAIT) {
                    Ok(data) => Self::handle_response(socket, pending, &data, timeout),
                    Err(zmq::Error::EAGAIN) => {
                        tracing::warn!("ZMQ response delimiter without payload");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("ZMQ recv payload error: {:?}", e);
                        break;
                    }
                },
                Err(zmq::Error::EAGAIN) => break,
                Err(e) => {
                    tracing::error!("ZMQ recv error: {:?}", e);
                    break;
                }
            }
        }
    }

    /// 计算下一次 zmq::poll 的超时（毫秒）。
    /// - 无 pending request：返回 `POLL_MAX_TIMEOUT`（兜底，正常会被 wakeup/POLLIN 提前唤醒）。
    /// - 有 pending request：取最近的 deadline，限制在 `[1ms, POLL_MAX_TIMEOUT]`。
    fn next_poll_timeout_ms(pending: &HashMap<String, PendingRequest>) -> i64 {
        let now = Instant::now();
        let nearest = pending
            .values()
            .filter_map(|r| match r {
                PendingRequest::Oneshot { deadline, .. } => Some(*deadline),
                PendingRequest::Stream { deadline, .. } => Some(*deadline),
            })
            .min();

        let target = match nearest {
            Some(deadline) => deadline
                .saturating_duration_since(now)
                .min(POLL_MAX_TIMEOUT),
            None => POLL_MAX_TIMEOUT,
        };
        // 至少 1ms，避免 0 退化为非阻塞 spin。
        target.as_millis().max(1) as i64
    }

    fn send_command(socket: &zmq::Socket, command: &ServerCommand) -> Result<()> {
        let data = rmp_serde::to_vec(command)?;
        socket.send(&b""[..], zmq::SNDMORE)?;
        socket.send(&data, 0)?;
        Ok(())
    }

    fn send_cancel(socket: &zmq::Socket, request_id: &str, reason: CancelReason) -> Result<()> {
        Self::send_command(
            socket,
            &ServerCommand::Cancel(ServerCancelRequest {
                request_id: request_id.to_string(),
                reason,
            }),
        )
    }

    fn handle_response(
        socket: &zmq::Socket,
        pending: &mut HashMap<String, PendingRequest>,
        data: &[u8],
        timeout: Duration,
    ) {
        if let Ok(response) = rmp_serde::from_slice::<InferenceResponse>(data) {
            let request_id = response.request_id.clone();
            if let Some(pending_req) = pending.remove(&request_id) {
                match pending_req {
                    PendingRequest::Oneshot { tx, .. } => {
                        if tx.send(response).is_err() {
                            tracing::debug!("Response receiver dropped for request {}", request_id);
                            if let Err(e) = Self::send_cancel(
                                socket,
                                &request_id,
                                CancelReason::ClientDisconnected,
                            ) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = ?e,
                                    "failed to send cancel after oneshot receiver dropped"
                                );
                            }
                        }
                    }
                    PendingRequest::Stream { tx, .. } => {
                        let chunk = StreamChunk {
                            request_id: request_id.clone(),
                            chunk_type: ChunkType::Done,
                            token_id: None,
                            finish_reason: response.finish_reason.clone(),
                            metrics: Some(response.metrics),
                        };
                        if tx.try_send(chunk).is_err() {
                            if let Err(e) = Self::send_cancel(
                                socket,
                                &request_id,
                                CancelReason::ClientDisconnected,
                            ) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = ?e,
                                    "failed to send cancel after stream receiver dropped"
                                );
                            }
                        }
                    }
                }
            } else {
                tracing::debug!("Received response for inactive request: {}", request_id);
            }
            return;
        }

        if let Ok(chunk) = rmp_serde::from_slice::<StreamChunk>(data) {
            let request_id = chunk.request_id.clone();
            let is_done = matches!(chunk.chunk_type, ChunkType::Done | ChunkType::Error);

            let mut cancel_reason = None;
            if let Some(PendingRequest::Stream { tx, deadline }) = pending.get_mut(&request_id) {
                *deadline = Instant::now() + timeout;
                match tx.try_send(chunk) {
                    Ok(()) => {}
                    Err(TokioTrySendError::Full(_)) => {
                        cancel_reason = Some(CancelReason::StreamTimeout)
                    }
                    Err(TokioTrySendError::Closed(_)) => {
                        cancel_reason = Some(CancelReason::ClientDisconnected)
                    }
                }
            } else if pending.contains_key(&request_id) {
                tracing::warn!(
                    "Received stream chunk for non-stream request: {}",
                    request_id
                );
            } else {
                tracing::debug!("Received chunk for inactive request: {}", request_id);
            }

            if is_done || cancel_reason.is_some() {
                pending.remove(&request_id);
            }
            if let Some(reason) = cancel_reason {
                if let Err(e) = Self::send_cancel(socket, &request_id, reason) {
                    tracing::warn!(
                        request_id = %request_id,
                        error = ?e,
                        "failed to send cancel after stream send failure"
                    );
                }
            }
            return;
        }

        tracing::error!(
            "Failed to deserialize response (neither InferenceResponse nor StreamChunk)"
        );
    }

    fn cancel_timed_out_requests(
        socket: &zmq::Socket,
        pending: &mut HashMap<String, PendingRequest>,
    ) {
        let now = Instant::now();
        let timed_out: Vec<(String, bool)> = pending
            .iter()
            .filter_map(|(request_id, pending_req)| match pending_req {
                PendingRequest::Stream { deadline, .. } if *deadline <= now => {
                    Some((request_id.clone(), true))
                }
                PendingRequest::Oneshot { deadline, .. } if *deadline <= now => {
                    Some((request_id.clone(), false))
                }
                _ => None,
            })
            .collect();

        for (request_id, is_stream) in timed_out {
            if let Some(pending_req) = pending.remove(&request_id) {
                if let PendingRequest::Stream { tx, .. } = pending_req {
                    let _ = tx.try_send(StreamChunk {
                        request_id: request_id.clone(),
                        chunk_type: ChunkType::Error,
                        token_id: None,
                        finish_reason: Some("stream timeout".to_string()),
                        metrics: Some(InferenceMetrics {
                            total_ms: 0,
                            num_tokens: 0,
                            tokens_per_second: 0.0,
                        }),
                    });
                } else if is_stream {
                    tracing::warn!(
                        request_id = %request_id,
                        "timed-out request changed kind before cleanup"
                    );
                }
                let reason = if is_stream {
                    CancelReason::StreamTimeout
                } else {
                    CancelReason::RequestTimeout
                };
                if let Err(e) = Self::send_cancel(socket, &request_id, reason) {
                    tracing::warn!(
                        request_id = %request_id,
                        error = ?e,
                        "failed to send cancel after request timeout"
                    );
                }
            }
        }
    }
}

impl InferClient for ZmqClient {
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let request_id = req.request_id.clone();
        let (tx, rx) = oneshot::channel();

        self.command_tx
            .try_send(RequestEnvelope::Oneshot {
                request: req,
                reply_tx: tx,
            })
            .map_err(|_| anyhow::anyhow!("ZMQ command channel full or disconnected"))?;
        self.waker.wake();

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow::anyhow!("Response channel closed")),
            Err(_) => {
                if self
                    .command_tx
                    .try_send(RequestEnvelope::Cancel {
                        request_id: request_id.clone(),
                        reason: CancelReason::RequestTimeout,
                    })
                    .is_ok()
                {
                    self.waker.wake();
                }
                Err(anyhow::anyhow!(
                    "Request {} timeout after {}s",
                    request_id,
                    self.timeout.as_secs()
                ))
            }
        }
    }

    async fn infer_stream(&self, req: InferenceRequest) -> Result<StreamHandle> {
        let request_id = req.request_id.clone();
        let (tx, rx) = mpsc::channel(STREAM_CHUNK_BUFFER);

        self.command_tx
            .try_send(RequestEnvelope::Stream {
                request: req,
                chunk_tx: tx,
            })
            .map_err(|_| anyhow::anyhow!("ZMQ command channel full or disconnected"))?;
        self.waker.wake();

        Ok(StreamHandle {
            request_id,
            rx,
            command_tx: self.command_tx.clone(),
            waker: self.waker.clone(),
            finished: false,
        })
    }
}
