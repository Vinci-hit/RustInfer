//! Async ZMQ transport — uses a dedicated blocking thread for ZMQ I/O
//! with tokio channels bridging to the async world.

use async_trait::async_trait;
use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
use infer_protocol::server_to_scheduler::ServerCommand;
use tokio::sync::mpsc;

use crate::error::{Result, SchedulerError, TransportError};
use crate::domain::inference_session::handle::ClientId;
use crate::infrastructure::transport::codec::{Codec, MsgPackCodec};
use crate::infrastructure::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};

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
    incoming_rx: mpsc::UnboundedReceiver<FrontendEvent>,
    /// Send channel: scheduler sends responses here, ZMQ thread drains.
    outgoing_tx: mpsc::UnboundedSender<OutgoingResponse>,
}

impl ZmqFrontendTransport {
    /// Spawn the ZMQ I/O thread and return the transport handle.
    pub fn new(endpoint: &str) -> Result<Self> {
        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();
        let (outgoing_tx, outgoing_rx) = mpsc::unbounded_channel();
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
        incoming_tx: mpsc::UnboundedSender<FrontendEvent>,
        mut outgoing_rx: mpsc::UnboundedReceiver<OutgoingResponse>,
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

        // Helper: bridge outgoing_rx → std channel + inproc wakeup.
        let (msg_tx, msg_rx) = std::sync::mpsc::channel::<OutgoingResponse>();
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
                                    let _ = incoming_tx.send(FrontendEvent::Infer {
                                        client_id: ClientId::new(identity),
                                        request,
                                    });
                                }
                                Ok(ServerCommand::Cancel(cancel)) => {
                                    let _ = incoming_tx.send(FrontendEvent::Cancel {
                                        external_id: cancel.request_id,
                                        reason: cancel.reason,
                                    });
                                }
                                Err(e) => tracing::error!("Decode: {}", e),
                            },
                            Err(e) => tracing::error!("ZMQ recv data: {:?}", e),
                        }
                    }
                    Err(zmq::Error::EAGAIN) => break,
                    Err(e) => { tracing::error!("ZMQ recv: {:?}", e); break; }
                }
            }

            // 2. Drain wakeup + send outgoing.
            while wakeup_rx.recv_bytes(zmq::DONTWAIT).is_ok() {}
            while let Ok(msg) = msg_rx.try_recv() {
                match msg {
                    OutgoingResponse::Full { client_id, response } => {
                        if let Ok(data) = codec.encode(&response) {
                            let _ = socket.send(client_id.as_bytes(), zmq::SNDMORE);
                            let _ = socket.send(&b""[..], zmq::SNDMORE);
                            let _ = socket.send(&data, 0);
                        }
                    }
                    OutgoingResponse::Chunk { client_id, chunk } => {
                        if let Ok(data) = codec.encode(&chunk) {
                            let _ = socket.send(client_id.as_bytes(), zmq::SNDMORE);
                            let _ = socket.send(&b""[..], zmq::SNDMORE);
                            let _ = socket.send(&data, 0);
                        }
                    }
                }
            }
        }
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
                client_id: ClientId::new(client.as_bytes().to_vec()),
                response,
            })
            .map_err(|_| SchedulerError::Shutdown)
    }

    async fn send_stream_chunk(&mut self, client: &ClientId, chunk: StreamChunk) -> Result<()> {
        self.outgoing_tx
            .send(OutgoingResponse::Chunk {
                client_id: ClientId::new(client.as_bytes().to_vec()),
                chunk,
            })
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
    command_tx: mpsc::UnboundedSender<Vec<u8>>,
}

impl ZmqWorkerTransport {
    /// Spawn the ZMQ I/O thread and return the transport handle.
    pub fn new(push_endpoint: &str, pull_endpoint: &str) -> Result<Self> {
        // ZMQ I/O thread must never block on the Tokio bridge: blocking here can
        // deadlock command sending against worker output receiving under load.
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
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
        mut command_rx: mpsc::UnboundedReceiver<Vec<u8>>,
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

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Vec<u8>>();
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
                    Err(e) => { tracing::error!("ZMQ PULL: {:?}", e); break; }
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
