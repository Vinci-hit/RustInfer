//! Async ZMQ transport — uses a dedicated blocking thread for ZMQ I/O
//! with tokio channels bridging to the async world.

use async_trait::async_trait;
use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
use infer_protocol::server_to_scheduler::ServerCommand;
use tokio::sync::mpsc;

use crate::error::{Result, SchedulerError, TransportError};
use crate::request::handle::ClientId;
use crate::request::lifecycle::RequestId;
use crate::transport::codec::{Codec, MsgPackCodec};
use crate::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};

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

    /// ZMQ I/O thread — blocking, owns the ROUTER socket.
    fn zmq_thread(
        endpoint: String,
        incoming_tx: mpsc::UnboundedSender<FrontendEvent>,
        mut outgoing_rx: mpsc::UnboundedReceiver<OutgoingResponse>,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::ROUTER)?;
        socket.bind(&endpoint)?;
        socket.set_rcvtimeo(10)?; // 10ms non-blocking poll

        tracing::info!("ZMQ frontend ROUTER bound to {}", endpoint);
        let codec = MsgPackCodec;

        loop {
            // 1. Try to receive requests from HTTP server.
            match socket.recv_bytes(0) {
                Ok(identity) => {
                    // ROUTER frame: [identity, empty, data]
                    let _ = socket.recv_bytes(0); // empty delimiter
                    match socket.recv_bytes(0) {
                        Ok(data) => match codec.decode::<ServerCommand>(&data) {
                            Ok(ServerCommand::Infer(request)) => {
                                let _ = incoming_tx.send(FrontendEvent::Infer {
                                    client_id: ClientId(identity),
                                    request,
                                });
                            }
                            Ok(ServerCommand::Cancel(cancel)) => {
                                let _ = incoming_tx.send(FrontendEvent::Cancel {
                                    request_id: RequestId(cancel.request_id),
                                    reason: cancel.reason,
                                });
                            }
                            Err(e) => {
                                tracing::error!("Failed to decode ServerCommand: {}", e);
                            }
                        },
                        Err(e) => tracing::error!("ZMQ recv data frame error: {:?}", e),
                    }
                }
                Err(zmq::Error::EAGAIN) => {} // No message ready.
                Err(e) => tracing::error!("ZMQ recv error: {:?}", e),
            }

            // 2. Try to send responses (non-blocking drain).
            while let Ok(msg) = outgoing_rx.try_recv() {
                match msg {
                    OutgoingResponse::Full {
                        client_id,
                        response,
                    } => {
                        if let Ok(data) = codec.encode(&response) {
                            if let Err(e) = socket.send(&client_id.0, zmq::SNDMORE) {
                                tracing::error!(
                                    "ZMQ frontend send identity failed for request {}: {:?}",
                                    response.request_id,
                                    e
                                );
                                continue;
                            }
                            if let Err(e) = socket.send(&b""[..], zmq::SNDMORE) {
                                tracing::error!(
                                    "ZMQ frontend send delimiter failed for request {}: {:?}",
                                    response.request_id,
                                    e
                                );
                                continue;
                            }
                            if let Err(e) = socket.send(&data, 0) {
                                tracing::error!(
                                    "ZMQ frontend send response failed for request {}: {:?}",
                                    response.request_id,
                                    e
                                );
                            }
                        }
                    }
                    OutgoingResponse::Chunk { client_id, chunk } => {
                        if let Ok(data) = codec.encode(&chunk) {
                            if let Err(e) = socket.send(&client_id.0, zmq::SNDMORE) {
                                tracing::error!(
                                    "ZMQ frontend send identity failed for stream {}: {:?}",
                                    chunk.request_id,
                                    e
                                );
                                continue;
                            }
                            if let Err(e) = socket.send(&b""[..], zmq::SNDMORE) {
                                tracing::error!(
                                    "ZMQ frontend send delimiter failed for stream {}: {:?}",
                                    chunk.request_id,
                                    e
                                );
                                continue;
                            }
                            if let Err(e) = socket.send(&data, 0) {
                                tracing::error!(
                                    "ZMQ frontend send stream chunk failed for {}: {:?}",
                                    chunk.request_id,
                                    e
                                );
                            }
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
                client_id: ClientId(client.0.clone()),
                response,
            })
            .map_err(|_| SchedulerError::Shutdown)
    }

    async fn send_stream_chunk(&mut self, client: &ClientId, chunk: StreamChunk) -> Result<()> {
        self.outgoing_tx
            .send(OutgoingResponse::Chunk {
                client_id: ClientId(client.0.clone()),
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
    output_rx: mpsc::UnboundedReceiver<Vec<u8>>,
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
            output_rx,
            command_tx,
        })
    }

    /// ZMQ I/O thread — blocking, owns PUSH+PULL sockets.
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
        pull_socket.set_rcvtimeo(10)?; // 10ms poll
        tracing::info!("ZMQ worker PULL bound to {}", pull_endpoint);

        loop {
            // 1. Send commands to worker (non-blocking drain).
            while let Ok(cmd) = command_rx.try_recv() {
                if let Err(e) = push_socket.send(&cmd, 0) {
                    tracing::error!("ZMQ PUSH send error: {:?}", e);
                }
            }

            // 2. Receive outputs from worker.
            match pull_socket.recv_bytes(0) {
                Ok(data) => {
                    if output_tx.send(data).is_err() {
                        tracing::info!("Worker output channel closed, shutting down");
                        return Ok(());
                    }
                }
                Err(zmq::Error::EAGAIN) => {} // No message ready.
                Err(e) => tracing::error!("ZMQ PULL recv error: {:?}", e),
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
        match self.output_rx.recv().await {
            Some(data) => Ok(data),
            None => Err(SchedulerError::Shutdown),
        }
    }
}
