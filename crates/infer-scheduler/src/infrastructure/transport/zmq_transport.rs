//! Async ZMQ transport — uses a dedicated blocking thread for ZMQ I/O
//! with tokio channels bridging to the async world.

use async_trait::async_trait;
use infer_protocol::scheduler_to_server::{
    ChunkType, FRONTEND_PROTOCOL_VERSION, InferenceMetrics, InferenceResponse, ResponseStatus,
    SchedulerPong, SchedulerReply, StreamChunk,
};
use infer_protocol::server_to_scheduler::ServerCommand;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use crate::domain::inference_session::handle::ClientId;
use crate::error::{Result, SchedulerError, TransportError};
use crate::infrastructure::transport::codec::{Codec, MsgPackCodec};
use crate::infrastructure::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};

/// Bound on the scheduler → ZMQ-thread outbound queues (responses / chunks /
/// worker commands). The ZMQ thread drains these with a tight `blocking_recv`
/// loop and forwards to non-blocking socket sends, so the queue is normally
/// near-empty. The bound only matters as an OOM backstop: if a peer stalls, the
/// async producer applies backpressure (`send().await`) instead of letting an
/// unbounded queue grow without limit. Large enough to never throttle healthy
/// traffic.
const OUTBOUND_QUEUE_BOUND: usize = 16_384;
/// Hard bound on decoded frontend commands waiting for the scheduler event
/// loop. When full, new inference requests receive an immediate overload
/// response instead of accumulating without limit.
const FRONTEND_INFERENCE_BOUND: usize = 1_024;
/// Slots inference submissions cannot consume, so cancellation remains
/// deliverable while the request queue is saturated.
const FRONTEND_CANCEL_RESERVE: usize = 64;
const FRONTEND_INGRESS_BOUND: usize = FRONTEND_INFERENCE_BOUND + FRONTEND_CANCEL_RESERVE;

fn frontend_ingress_channel<T>() -> (mpsc::Sender<T>, mpsc::Receiver<T>) {
    mpsc::channel(FRONTEND_INGRESS_BOUND)
}

fn outbound_bridge_channel<T>() -> (std::sync::mpsc::SyncSender<T>, std::sync::mpsc::Receiver<T>) {
    std::sync::mpsc::sync_channel(OUTBOUND_QUEUE_BOUND)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ZMQ Frontend Transport
// ═══════════════════════════════════════════════════════════════════════════════

/// Message from scheduler → ZMQ thread.
pub enum OutgoingResponse {
    Full {
        client_id: ClientId,
        response: InferenceResponse,
    },
    Chunk {
        client_id: ClientId,
        chunk: StreamChunk,
    },
}

/// Frontend transport backed by ZMQ ROUTER socket.
///
/// ZMQ operations run in a dedicated std thread (ZMQ sockets are !Send).
/// Communication with the async scheduler is via tokio mpsc channels.
pub struct ZmqFrontendTransport {
    /// Receive channel: ZMQ thread sends incoming requests here.
    incoming_rx: mpsc::Receiver<FrontendEvent>,
    /// Send channel: scheduler sends responses here, ZMQ thread drains.
    /// Bounded so a stalled client cannot make the scheduler accumulate an
    /// unbounded backlog of undelivered responses (OOM backstop).
    outgoing_tx: mpsc::Sender<OutgoingResponse>,
}

impl ZmqFrontendTransport {
    /// Spawn the ZMQ I/O thread and return the transport handle.
    pub fn new(endpoint: &str) -> Result<Self> {
        let (incoming_tx, incoming_rx) = frontend_ingress_channel();
        let (outgoing_tx, outgoing_rx) = mpsc::channel(OUTBOUND_QUEUE_BOUND);
        let endpoint = endpoint.to_string();

        std::thread::Builder::new()
            .name("zmq-frontend".to_string())
            .spawn(move || {
                if let Err(e) = Self::zmq_thread(endpoint, incoming_tx, outgoing_rx) {
                    tracing::error!("ZMQ frontend thread exited: {:?}", e);
                }
            })
            .map_err(|e| {
                SchedulerError::Transport(TransportError::ConnectionFailed(format!(
                    "Failed to spawn ZMQ frontend thread: {}",
                    e
                )))
            })?;

        Ok(Self {
            incoming_rx,
            outgoing_tx,
        })
    }

    /// ZMQ I/O thread — epoll-driven, zero-latency wakeup.
    ///
    /// 用 zmq_poll 同时监听 ROUTER + inproc PAIR，任一有数据立即唤醒。
    fn zmq_thread(
        endpoint: String,
        incoming_tx: mpsc::Sender<FrontendEvent>,
        mut outgoing_rx: mpsc::Receiver<OutgoingResponse>,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::ROUTER)?;
        socket.bind(&endpoint)?;
        socket.set_rcvtimeo(0)?;

        // inproc wakeup: main thread binds, helper thread connects (after barrier).
        let wakeup_rx = ctx.socket(zmq::PAIR)?;
        wakeup_rx.bind("inproc://fe-wakeup")?;
        wakeup_rx.set_rcvtimeo(0)?;

        tracing::info!("ZMQ frontend ROUTER bound to {}", endpoint);
        let codec = MsgPackCodec;

        // Helper: bridge outgoing_rx → bounded std channel + inproc wakeup.
        // Keeping both sides bounded is essential: an unbounded second hop
        // would silently erase the Tokio queue's backpressure.
        let (msg_tx, msg_rx) = outbound_bridge_channel::<OutgoingResponse>();
        let ctx_clone = ctx.clone();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier2 = barrier.clone();

        std::thread::Builder::new()
            .name("fe-outgoing-bridge".into())
            .spawn(move || {
                // Create wakeup_tx in THIS thread (ZMQ socket thread-affinity).
                let wakeup_tx = ctx_clone.socket(zmq::PAIR).unwrap();
                wakeup_tx.connect("inproc://fe-wakeup").unwrap();
                barrier2.wait(); // signal main thread that connect is done

                loop {
                    match outgoing_rx.blocking_recv() {
                        Some(msg) => {
                            let _ = msg_tx.send(msg);
                            let _ = wakeup_tx.send(&[1u8][..], zmq::DONTWAIT);
                        }
                        None => return,
                    }
                }
            })?;

        barrier.wait(); // wait for helper to connect inproc

        loop {
            // Block until ROUTER has data OR wakeup fires.
            let mut poll_items = [
                socket.as_poll_item(zmq::POLLIN),
                wakeup_rx.as_poll_item(zmq::POLLIN),
            ];
            let _ = zmq::poll(&mut poll_items, -1);

            // 1. Drain incoming.
            loop {
                match socket.recv_bytes(0) {
                    Ok(identity) => {
                        let _ = socket.recv_bytes(0);
                        match socket.recv_bytes(0) {
                            Ok(data) => match codec.decode::<ServerCommand>(&data) {
                                Ok(ServerCommand::Infer(request)) => {
                                    let request_id = request.request_id.clone();
                                    let stream = request.stream;
                                    if incoming_tx.capacity() <= FRONTEND_CANCEL_RESERVE {
                                        Self::reject_overload(
                                            &socket, &codec, &identity, request_id, stream,
                                        );
                                    } else {
                                        match incoming_tx.try_send(FrontendEvent::Infer {
                                            client_id: ClientId::new(identity.clone()),
                                            request,
                                        }) {
                                            Ok(()) => {}
                                            Err(TrySendError::Full(_)) => Self::reject_overload(
                                                &socket, &codec, &identity, request_id, stream,
                                            ),
                                            Err(TrySendError::Closed(_)) => return Ok(()),
                                        }
                                    }
                                }
                                Ok(ServerCommand::Cancel(cancel)) => {
                                    match incoming_tx.try_send(FrontendEvent::Cancel {
                                        external_id: cancel.request_id,
                                        reason: cancel.reason,
                                    }) {
                                        Ok(()) => {}
                                        Err(TrySendError::Full(_)) => tracing::error!(
                                            "frontend cancellation reserve exhausted; dropping cancel"
                                        ),
                                        Err(TrySendError::Closed(_)) => return Ok(()),
                                    }
                                }
                                Ok(ServerCommand::Ping) => {
                                    // Liveness probe: answer directly from this
                                    // thread (no engine round-trip) so the
                                    // server's `/ready` reflects this process
                                    // being alive and its frontend thread
                                    // responsive.
                                    let pong = SchedulerReply::Pong(SchedulerPong {
                                        protocol_version: FRONTEND_PROTOCOL_VERSION,
                                    });
                                    if let Err(e) =
                                        Self::send_reply(&socket, &codec, &identity, &pong)
                                    {
                                        tracing::error!(error = ?e, "failed to send scheduler pong");
                                    }
                                }
                                Err(e) => tracing::error!("Decode: {}", e),
                            },
                            Err(e) => tracing::error!("ZMQ recv data: {:?}", e),
                        }
                    }
                    Err(zmq::Error::EAGAIN) => break,
                    Err(e) => {
                        tracing::error!("ZMQ recv: {:?}", e);
                        break;
                    }
                }
            }

            // 2. Drain wakeup + send outgoing.
            while wakeup_rx.recv_bytes(zmq::DONTWAIT).is_ok() {}
            while let Ok(msg) = msg_rx.try_recv() {
                // All replies go out wrapped in the tagged `SchedulerReply`
                // envelope — the server decodes exactly one type, never by
                // trial deserialization.
                let (client_id, reply) = match msg {
                    OutgoingResponse::Full {
                        client_id,
                        response,
                    } => (client_id, SchedulerReply::Full(response)),
                    OutgoingResponse::Chunk { client_id, chunk } => {
                        (client_id, SchedulerReply::Chunk(chunk))
                    }
                };
                if let Err(e) = Self::send_reply(&socket, &codec, client_id.as_bytes(), &reply) {
                    tracing::error!(error = ?e, "failed to send scheduler frontend reply");
                }
            }
        }
    }

    fn overload_reply(request_id: String, stream: bool) -> SchedulerReply {
        const MESSAGE: &str = "scheduler overloaded, please retry later";
        if stream {
            SchedulerReply::Chunk(StreamChunk {
                request_id,
                chunk_type: ChunkType::Error,
                token_id: None,
                finish_reason: Some(MESSAGE.to_string()),
                metrics: Some(InferenceMetrics::default()),
            })
        } else {
            SchedulerReply::Full(InferenceResponse {
                request_id,
                status: ResponseStatus::Error,
                output_token_ids: vec![],
                images: vec![],
                finish_reason: Some("error".to_string()),
                error: Some(MESSAGE.to_string()),
                metrics: InferenceMetrics::default(),
            })
        }
    }

    fn reject_overload(
        socket: &zmq::Socket,
        codec: &MsgPackCodec,
        identity: &[u8],
        request_id: String,
        stream: bool,
    ) {
        tracing::warn!(%request_id, "frontend ingress queue full; rejecting request");
        let reply = Self::overload_reply(request_id, stream);
        if let Err(e) = Self::send_reply(socket, codec, identity, &reply) {
            tracing::error!(error = ?e, "failed to send frontend overload response");
        }
    }

    fn send_reply(
        socket: &zmq::Socket,
        codec: &MsgPackCodec,
        identity: &[u8],
        reply: &SchedulerReply,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let data = codec.encode(reply)?;
        socket.send(identity, zmq::SNDMORE)?;
        socket.send(&b""[..], zmq::SNDMORE)?;
        socket.send(&data, 0)?;
        Ok(())
    }
}

#[async_trait]
impl FrontendTransport for ZmqFrontendTransport {
    async fn recv_event(&mut self) -> Result<FrontendEvent> {
        self.incoming_rx
            .recv()
            .await
            .ok_or(SchedulerError::Shutdown)
    }

    async fn send_response(
        &mut self,
        client: &ClientId,
        response: InferenceResponse,
    ) -> Result<()> {
        self.outgoing_tx
            .send(OutgoingResponse::Full {
                client_id: client.clone(),
                response,
            })
            .await
            .map_err(|_| SchedulerError::Shutdown)
    }

    async fn send_stream_chunk(&mut self, client: &ClientId, chunk: StreamChunk) -> Result<()> {
        self.outgoing_tx
            .send(OutgoingResponse::Chunk {
                client_id: client.clone(),
                chunk,
            })
            .await
            .map_err(|_| SchedulerError::Shutdown)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ZMQ Worker Transport
// ═══════════════════════════════════════════════════════════════════════════════

/// Worker transport backed by ZMQ PUSH/PULL sockets.
pub struct ZmqWorkerTransport {
    /// Receive channel: ZMQ thread sends worker outputs here.
    /// Wrapped in `Option` so `take_output_rx()` can extract it
    /// for the background decode task.
    output_rx: Option<mpsc::UnboundedReceiver<Vec<u8>>>,
    /// Send channel: scheduler sends batch commands here.
    /// Bounded as an OOM backstop if the worker PUSH socket stalls.
    command_tx: mpsc::Sender<Vec<u8>>,
}

impl ZmqWorkerTransport {
    /// Spawn the ZMQ I/O thread and return the transport handle.
    pub fn new(push_endpoint: &str, pull_endpoint: &str) -> Result<Self> {
        // ZMQ I/O thread must never block on the Tokio bridge: blocking here can
        // deadlock command sending against worker output receiving under load.
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (command_tx, command_rx) = mpsc::channel(OUTBOUND_QUEUE_BOUND);
        let push_ep = push_endpoint.to_string();
        let pull_ep = pull_endpoint.to_string();

        std::thread::Builder::new()
            .name("zmq-worker".to_string())
            .spawn(move || {
                if let Err(e) = Self::zmq_thread(push_ep, pull_ep, output_tx, command_rx) {
                    tracing::error!("ZMQ worker thread exited: {:?}", e);
                }
            })
            .map_err(|e| {
                SchedulerError::Transport(TransportError::ConnectionFailed(format!(
                    "Failed to spawn ZMQ worker thread: {}",
                    e
                )))
            })?;

        Ok(Self {
            output_rx: Some(output_rx),
            command_tx,
        })
    }

    /// ZMQ I/O thread — epoll-driven with inproc wakeup for commands.
    fn zmq_thread(
        push_endpoint: String,
        pull_endpoint: String,
        output_tx: mpsc::UnboundedSender<Vec<u8>>,
        mut command_rx: mpsc::Receiver<Vec<u8>>,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ctx = zmq::Context::new();

        let push_socket = ctx.socket(zmq::PUSH)?;
        push_socket.bind(&push_endpoint)?;
        tracing::info!("ZMQ worker PUSH bound to {}", push_endpoint);

        let pull_socket = ctx.socket(zmq::PULL)?;
        pull_socket.bind(&pull_endpoint)?;
        pull_socket.set_rcvtimeo(0)?;
        tracing::info!("ZMQ worker PULL bound to {}", pull_endpoint);

        // inproc wakeup for commands.
        let wakeup_rx = ctx.socket(zmq::PAIR)?;
        wakeup_rx.bind("inproc://wk-cmd-wakeup")?;
        wakeup_rx.set_rcvtimeo(0)?;

        // Preserve the bounded Tokio command queue through the bridge. If the
        // ZMQ PUSH socket stalls, this bounded hop fills and backpressure
        // reaches `send_batch()` instead of accumulating in std mpsc memory.
        let (cmd_tx, cmd_rx) = outbound_bridge_channel::<Vec<u8>>();
        let ctx_clone = ctx.clone();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier2 = barrier.clone();

        std::thread::Builder::new()
            .name("wk-cmd-bridge".into())
            .spawn(move || {
                let wakeup_tx = ctx_clone.socket(zmq::PAIR).unwrap();
                wakeup_tx.connect("inproc://wk-cmd-wakeup").unwrap();
                barrier2.wait();

                loop {
                    match command_rx.blocking_recv() {
                        Some(cmd) => {
                            let _ = cmd_tx.send(cmd);
                            let _ = wakeup_tx.send(&[1u8][..], zmq::DONTWAIT);
                        }
                        None => return,
                    }
                }
            })?;

        barrier.wait();

        loop {
            let mut poll_items = [
                pull_socket.as_poll_item(zmq::POLLIN),
                wakeup_rx.as_poll_item(zmq::POLLIN),
            ];
            let _ = zmq::poll(&mut poll_items, -1);

            // 1. Send queued commands.
            while wakeup_rx.recv_bytes(zmq::DONTWAIT).is_ok() {}
            while let Ok(cmd) = cmd_rx.try_recv() {
                if let Err(e) = push_socket.send(&cmd, 0) {
                    tracing::error!("ZMQ PUSH send: {:?}", e);
                }
            }

            // 2. Drain worker outputs.
            loop {
                match pull_socket.recv_bytes(0) {
                    Ok(data) => {
                        if output_tx.send(data).is_err() {
                            return Ok(());
                        }
                    }
                    Err(zmq::Error::EAGAIN) => break,
                    Err(e) => {
                        tracing::error!("ZMQ PULL: {:?}", e);
                        break;
                    }
                }
            }
        }
    }
}

#[async_trait]
impl WorkerTransport for ZmqWorkerTransport {
    async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
        self.command_tx
            .send(cmd)
            .await
            .map_err(|_| SchedulerError::Shutdown)
    }

    async fn recv_step_output(&mut self) -> Result<Vec<u8>> {
        match self.output_rx.as_mut() {
            Some(rx) => match rx.recv().await {
                Some(data) => Ok(data),
                None => Err(SchedulerError::Shutdown),
            },
            None => Err(SchedulerError::Shutdown),
        }
    }

    fn take_output_rx(&mut self) -> Option<mpsc::UnboundedReceiver<Vec<u8>>> {
        self.output_rx.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outbound_bridge_remains_bounded() {
        let (tx, _rx) = outbound_bridge_channel();
        for value in 0..OUTBOUND_QUEUE_BOUND {
            tx.try_send(value).unwrap();
        }
        assert!(matches!(
            tx.try_send(OUTBOUND_QUEUE_BOUND),
            Err(std::sync::mpsc::TrySendError::Full(_))
        ));
    }

    #[tokio::test]
    async fn frontend_ingress_queue_remains_bounded() {
        let (tx, _rx) = frontend_ingress_channel();
        for value in 0..FRONTEND_INGRESS_BOUND {
            tx.try_send(value).unwrap();
        }
        assert!(matches!(
            tx.try_send(FRONTEND_INGRESS_BOUND),
            Err(TrySendError::Full(_))
        ));
    }

    #[test]
    fn overload_reply_preserves_stream_shape() {
        assert!(matches!(
            ZmqFrontendTransport::overload_reply("stream".to_string(), true),
            SchedulerReply::Chunk(StreamChunk {
                chunk_type: ChunkType::Error,
                ..
            })
        ));
        assert!(matches!(
            ZmqFrontendTransport::overload_reply("full".to_string(), false),
            SchedulerReply::Full(InferenceResponse {
                status: ResponseStatus::Error,
                ..
            })
        ));
    }
}
