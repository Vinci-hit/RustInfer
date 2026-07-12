//! ZMQ 客户端 — 通过 ZMQ DEALER socket 与 Scheduler 通信

use anyhow::Result;
use infer_protocol::scheduler_to_server::{
    ChunkType, FRONTEND_PROTOCOL_VERSION, InferenceMetrics, InferenceResponse, SchedulerReply,
    StreamChunk,
};
use infer_protocol::server_to_scheduler::{
    CancelReason, CancelRequest as ServerCancelRequest, InferenceRequest, ServerCommand,
};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TrySendError as TokioTrySendError;
use tokio::sync::{mpsc, oneshot};

use super::InferClient;

const CLIENT_COMMAND_BUFFER: usize = 1024;
const STREAM_CHUNK_BUFFER: usize = 64;
/// Send a liveness `Ping` when nothing has been heard from the scheduler for
/// this long. The scheduler's frontend thread answers `Pong` immediately, so
/// under a healthy link `last_contact` stays fresh even with zero traffic.
const PING_INTERVAL: Duration = Duration::from_secs(3);
/// `/ready` reports ready only if the scheduler was heard from within this
/// window. Must comfortably exceed `PING_INTERVAL` so one lost pong does not
/// flap readiness.
const READY_STALE_MS: u64 = 10_000;
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
    /// Unix millis of the last decoded scheduler reply (any kind). `0` until
    /// first contact. Written by the ZMQ thread, read by `/ready`.
    last_contact: std::sync::Arc<AtomicU64>,
}

/// Current unix time in milliseconds (0 if the clock is before the epoch).
fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Cancels a oneshot (non-stream) request if the HTTP handler future is
/// dropped mid-await — the non-stream twin of [`StreamHandle`]'s drop-cancel.
/// Without this, an abandoned non-stream request keeps decoding (and holding
/// KV) until completion or timeout.
struct OneshotCancelGuard {
    /// `Some` while armed; `take()`n on disarm or drop.
    request_id: Option<String>,
    command_tx: SyncSender<RequestEnvelope>,
    waker: std::sync::Arc<Waker>,
}

impl OneshotCancelGuard {
    fn disarm(&mut self) {
        self.request_id = None;
    }
}

impl Drop for OneshotCancelGuard {
    fn drop(&mut self) {
        if let Some(request_id) = self.request_id.take()
            && self
                .command_tx
                .try_send(RequestEnvelope::Cancel {
                    request_id,
                    reason: CancelReason::ClientDisconnected,
                })
                .is_ok()
        {
            self.waker.wake();
        }
    }
}

/// Classify a failed submit into the bounded command channel: a full channel
/// means the server is overloaded (→ HTTP 429 via [`crate::error::AppError::from_submit`]);
/// a disconnected channel means the scheduler link is gone (→ 500).
fn map_submit_err(err: std::sync::mpsc::TrySendError<RequestEnvelope>) -> anyhow::Error {
    match err {
        std::sync::mpsc::TrySendError::Full(_) => {
            anyhow::Error::new(crate::error::ServerOverloaded)
        }
        std::sync::mpsc::TrySendError::Disconnected(_) => {
            anyhow::anyhow!("scheduler command channel disconnected")
        }
    }
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

        let last_contact = std::sync::Arc::new(AtomicU64::new(0));
        let last_contact_thread = last_contact.clone();

        thread::Builder::new()
            .name("zmq-client".to_string())
            .spawn(move || {
                if let Err(e) =
                    Self::zmq_thread(endpoint, command_rx, wake_rx, timeout, last_contact_thread)
                {
                    tracing::error!("ZMQ thread exited with error: {:?}", e);
                }
            })?;

        Ok(Self {
            command_tx,
            waker,
            timeout,
            last_contact,
        })
    }

    /// Whether the scheduler has been heard from recently enough to accept
    /// traffic. Drives `/ready`: false until the first reply/pong after boot,
    /// and false again within `READY_STALE_MS` of the scheduler dying (DEALER
    /// `connect` is lazy and never fails, so socket state says nothing).
    pub fn scheduler_alive(&self) -> bool {
        let last = self.last_contact.load(Ordering::Relaxed);
        last != 0 && unix_ms().saturating_sub(last) < READY_STALE_MS
    }

    fn zmq_thread(
        endpoint: String,
        command_rx: Receiver<RequestEnvelope>,
        mut wake_rx: std::io::PipeReader,
        timeout: Duration,
        last_contact: std::sync::Arc<AtomicU64>,
    ) -> Result<()> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::DEALER)?;
        socket.connect(&endpoint)?;
        let wake_fd = wake_rx.as_raw_fd();

        tracing::info!("ZMQ client connected to {}", endpoint);
        let mut pending: HashMap<String, PendingRequest> = HashMap::new();
        let mut last_ping = Instant::now() - PING_INTERVAL;

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
                Self::drain_dealer(&socket, &mut pending, &last_contact, timeout);
            }

            // 4b) Liveness probe: nothing heard within PING_INTERVAL → send a
            //     Ping (rate-limited by last_ping). The poll above wakes at
            //     least every POLL_MAX_TIMEOUT, so this runs on time even when
            //     fully idle.
            let contact_age = unix_ms().saturating_sub(last_contact.load(Ordering::Relaxed));
            if contact_age >= PING_INTERVAL.as_millis() as u64
                && last_ping.elapsed() >= PING_INTERVAL
            {
                last_ping = Instant::now();
                if let Err(e) = Self::send_command(&socket, &ServerCommand::Ping) {
                    tracing::warn!("failed to send liveness ping: {:?}", e);
                }
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
                    // Unary timeout ownership lives solely in `infer()`'s
                    // Tokio timeout. Giving this ZMQ-thread entry the same
                    // deadline creates a race where this sender is dropped
                    // first and the handler observes a closed channel (500)
                    // instead of the typed timeout marker (504).
                    pending.insert(request_id.clone(), PendingRequest::Oneshot { tx: reply_tx });
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
        last_contact: &AtomicU64,
        timeout: Duration,
    ) {
        loop {
            // DEALER 收到的第一帧是空 delimiter（来自 ROUTER 的回程）
            match socket.recv_bytes(zmq::DONTWAIT) {
                Ok(_delim) => match socket.recv_bytes(zmq::DONTWAIT) {
                    Ok(data) => {
                        Self::handle_response(socket, pending, last_contact, &data, timeout)
                    }
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
    /// - 有 pending stream：取最近的 deadline，限制在 `[1ms, POLL_MAX_TIMEOUT]`。
    ///
    /// Unary requests are timed out by their async waiter and do not own a
    /// second deadline here.
    fn next_poll_timeout_ms(pending: &HashMap<String, PendingRequest>) -> i64 {
        let now = Instant::now();
        let nearest = pending
            .values()
            .filter_map(|r| match r {
                PendingRequest::Oneshot { .. } => None,
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
        last_contact: &AtomicU64,
        data: &[u8],
        timeout: Duration,
    ) {
        // Single tagged decode — no trial deserialization. The scheduler wraps
        // every reply in `SchedulerReply`.
        let reply = match rmp_serde::from_slice::<SchedulerReply>(data) {
            Ok(reply) => reply,
            Err(e) => {
                tracing::error!("Failed to deserialize SchedulerReply: {}", e);
                return;
            }
        };

        match reply {
            SchedulerReply::Pong(pong) => {
                if pong.protocol_version != FRONTEND_PROTOCOL_VERSION {
                    // Refuse readiness on mismatch: don't refresh last_contact,
                    // so /ready keeps reporting 503.
                    tracing::error!(
                        scheduler_version = pong.protocol_version,
                        server_version = FRONTEND_PROTOCOL_VERSION,
                        "scheduler frontend protocol version mismatch; holding /ready at 503"
                    );
                    return;
                }
                last_contact.store(unix_ms(), Ordering::Relaxed);
            }
            SchedulerReply::Full(response) => {
                last_contact.store(unix_ms(), Ordering::Relaxed);
                Self::handle_full_response(socket, pending, response);
            }
            SchedulerReply::Chunk(chunk) => {
                last_contact.store(unix_ms(), Ordering::Relaxed);
                Self::handle_stream_chunk(socket, pending, chunk, timeout);
            }
        }
    }

    fn handle_full_response(
        socket: &zmq::Socket,
        pending: &mut HashMap<String, PendingRequest>,
        response: InferenceResponse,
    ) {
        let request_id = response.request_id.clone();
        if let Some(pending_req) = pending.remove(&request_id) {
            match pending_req {
                PendingRequest::Oneshot { tx, .. } => {
                    if tx.send(response).is_err() {
                        tracing::debug!("Response receiver dropped for request {}", request_id);
                        if let Err(e) =
                            Self::send_cancel(socket, &request_id, CancelReason::ClientDisconnected)
                        {
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
                    if tx.try_send(chunk).is_err()
                        && let Err(e) =
                            Self::send_cancel(socket, &request_id, CancelReason::ClientDisconnected)
                    {
                        tracing::warn!(
                            request_id = %request_id,
                            error = ?e,
                            "failed to send cancel after stream receiver dropped"
                        );
                    }
                }
            }
        } else {
            tracing::debug!("Received response for inactive request: {}", request_id);
        }
    }

    fn handle_stream_chunk(
        socket: &zmq::Socket,
        pending: &mut HashMap<String, PendingRequest>,
        chunk: StreamChunk,
        timeout: Duration,
    ) {
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
        if let Some(reason) = cancel_reason
            && let Err(e) = Self::send_cancel(socket, &request_id, reason)
        {
            tracing::warn!(
                request_id = %request_id,
                error = ?e,
                "failed to send cancel after stream send failure"
            );
        }
    }

    fn cancel_timed_out_requests(
        socket: &zmq::Socket,
        pending: &mut HashMap<String, PendingRequest>,
    ) {
        let now = Instant::now();
        let timed_out: Vec<String> = pending
            .iter()
            .filter_map(|(request_id, pending_req)| match pending_req {
                PendingRequest::Stream { deadline, .. } if *deadline <= now => {
                    Some(request_id.clone())
                }
                _ => None,
            })
            .collect();

        for request_id in timed_out {
            if let Some(PendingRequest::Stream { tx, .. }) = pending.remove(&request_id) {
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
                if let Err(e) = Self::send_cancel(socket, &request_id, CancelReason::StreamTimeout)
                {
                    tracing::warn!(
                        request_id = %request_id,
                        error = ?e,
                        "failed to send cancel after stream timeout"
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
            .map_err(map_submit_err)?;
        self.waker.wake();

        // If the HTTP handler future is dropped while parked on `rx` (client
        // disconnected), the guard's Drop cancels the request scheduler-side —
        // without it the engine decodes to completion for a dead connection.
        let mut cancel_guard = OneshotCancelGuard {
            request_id: Some(request_id.clone()),
            command_tx: self.command_tx.clone(),
            waker: self.waker.clone(),
        };

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(response)) => {
                cancel_guard.disarm();
                Ok(response)
            }
            Ok(Err(_)) => {
                // ZMQ thread dropped the sender — pending entry is already
                // gone; a follow-up cancel would be noise.
                cancel_guard.disarm();
                Err(anyhow::anyhow!("Response channel closed"))
            }
            Err(_) => {
                cancel_guard.disarm();
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
                // Typed marker so the handler maps this to HTTP 504, not 500.
                Err(anyhow::Error::new(crate::error::RequestTimedOut {
                    request_id,
                    secs: self.timeout.as_secs(),
                }))
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
            .map_err(map_submit_err)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_pending_requests_have_no_zmq_thread_deadline() {
        let (tx, _rx) = oneshot::channel();
        let mut pending = HashMap::new();
        pending.insert("req".to_string(), PendingRequest::Oneshot { tx });

        assert_eq!(
            ZmqClient::next_poll_timeout_ms(&pending),
            POLL_MAX_TIMEOUT.as_millis() as i64
        );
    }
}
