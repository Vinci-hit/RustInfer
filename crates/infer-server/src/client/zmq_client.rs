//! ZMQ 客户端 — 通过 ZMQ DEALER socket 与 Scheduler 通信

use anyhow::Result;
use infer_protocol::scheduler_to_server::{
    ChunkType, InferenceMetrics, InferenceResponse, StreamChunk,
};
use infer_protocol::server_to_scheduler::{
    CancelReason, CancelRequest as ServerCancelRequest, InferenceRequest, ServerCommand,
};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TrySendError as TokioTrySendError;
use tokio::sync::{mpsc, oneshot};

use super::InferClient;

const CLIENT_COMMAND_BUFFER: usize = 1024;
const STREAM_CHUNK_BUFFER: usize = 64;
/// Idle ceiling on a single `zmq::poll`: with the wake pipe restored, new
/// commands interrupt the poll instantly and responses arrive via POLLIN, so
/// this only bounds how often `cancel_timed_out_requests` runs while fully
/// idle. Kept generous (was briefly dropped to 1ms as a band-aid after the
/// wake socket was removed in `cac326a`, which cost up to ~1ms per request
/// submit on the TTFT path).
const POLL_MAX_TIMEOUT: Duration = Duration::from_secs(1);

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

/// Wakes the ZMQ thread the instant a new command is queued, so request
/// submission does not wait for the poll timeout to elapse.
///
/// Uses a plain OS pipe (not a ZMQ inproc PAIR socket) so it cannot trip
/// libzmq's cross-thread signaler assertions — which is why the previous
/// inproc-PAIR waker was removed in `cac326a`. The write end is shared across
/// the axum worker threads; the read end is polled by the ZMQ thread. The
/// `Mutex` is held only for a single best-effort 1-byte write, never across an
/// `.await`.
struct Waker {
    writer: Mutex<std::io::PipeWriter>,
}

impl Waker {
    fn wake(&self) {
        if let Ok(mut w) = self.writer.lock() {
            // One byte is enough to mark the pipe readable; the reader coalesces
            // by draining. A full pipe (reader stalled) just means a wake is
            // already in flight, so dropping this one is harmless.
            let _ = w.write(&[1u8]);
        }
    }
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

        // Self-pipe wake: producers (axum threads) write 1 byte to `wake_tx`;
        // the ZMQ thread polls `wake_rx`'s fd alongside the DEALER socket.
        let (wake_rx, wake_tx) = std::io::pipe()?;
        let waker = std::sync::Arc::new(Waker {
            writer: Mutex::new(wake_tx),
        });

        thread::Builder::new()
            .name("zmq-client".to_string())
            .spawn(move || {
                if let Err(e) = Self::zmq_thread(endpoint, command_rx, wake_rx, timeout) {
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
        mut wake_rx: std::io::PipeReader,
        timeout: Duration,
    ) -> Result<()> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::DEALER)?;
        socket.connect(&endpoint)?;
        let wake_fd = wake_rx.as_raw_fd();

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

            // 3) 同时监听 DEALER 与 wake 管道：新命令写 wake 管道即可立刻打断 poll，
            //    无需等到 timeout（wake 管道是普通 OS pipe，不会触发 libzmq signaler 断言）。
            let mut items = [
                socket.as_poll_item(zmq::POLLIN),
                zmq::PollItem::from_fd(wake_fd, zmq::POLLIN),
            ];
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

            // 5) wake 管道有信号：排空一次（POLLIN 保证至少 1 字节，不会阻塞；
            //    残留字节会让下一轮 poll 立刻返回，无害）。下一轮顶部 drain_commands
            //    会消费新命令。
            if items[1].is_readable() {
                let mut sink = [0u8; 64];
                let _ = wake_rx.read(&mut sink);
            }

            // 6) 兜底：处理 stream 超时
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
