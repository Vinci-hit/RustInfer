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
    Oneshot(oneshot::Sender<InferenceResponse>),
    Stream {
        tx: mpsc::Sender<StreamChunk>,
        deadline: Instant,
    },
}

pub struct StreamHandle {
    request_id: String,
    rx: mpsc::Receiver<StreamChunk>,
    command_tx: SyncSender<RequestEnvelope>,
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
        let _ = self.command_tx.try_send(RequestEnvelope::Cancel {
            request_id: self.request_id.clone(),
            reason: CancelReason::ClientDisconnected,
        });
    }
}

pub struct ZmqClient {
    command_tx: SyncSender<RequestEnvelope>,
    timeout: Duration,
}

impl ZmqClient {
    pub async fn new(endpoint: &str, timeout_secs: u64) -> Result<Self> {
        let endpoint = endpoint.to_string();
        let timeout = Duration::from_secs(timeout_secs);
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<RequestEnvelope>(CLIENT_COMMAND_BUFFER);

        thread::Builder::new()
            .name("zmq-client".to_string())
            .spawn(move || {
                if let Err(e) = Self::zmq_thread(endpoint, command_rx, timeout) {
                    tracing::error!("ZMQ thread exited with error: {:?}", e);
                }
            })?;

        Ok(Self {
            command_tx,
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
        socket.set_rcvtimeo(10)?;

        tracing::info!("ZMQ client connected to {}", endpoint);
        let mut pending: HashMap<String, PendingRequest> = HashMap::new();

        loop {
            loop {
                match command_rx.try_recv() {
                    Ok(RequestEnvelope::Oneshot { request, reply_tx }) => {
                        let request_id = request.request_id.clone();
                        pending.insert(request_id.clone(), PendingRequest::Oneshot(reply_tx));
                        if let Err(e) = Self::send_command(&socket, &ServerCommand::Infer(request))
                        {
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
                        if let Err(e) = Self::send_command(&socket, &ServerCommand::Infer(request))
                        {
                            tracing::error!(
                                "Failed to send stream request {}: {:?}",
                                request_id,
                                e
                            );
                            pending.remove(&request_id);
                        }
                    }
                    Ok(RequestEnvelope::Cancel { request_id, reason }) => {
                        pending.remove(&request_id);
                        if let Err(e) = Self::send_cancel(&socket, &request_id, reason) {
                            tracing::error!("Failed to send cancel {}: {:?}", request_id, e);
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            }

            match socket.recv_bytes(0) {
                Ok(_) => match socket.recv_bytes(0) {
                    Ok(data) => Self::handle_response(&socket, &mut pending, &data),
                    Err(zmq::Error::EAGAIN) => {}
                    Err(e) => tracing::error!("ZMQ recv data error: {:?}", e),
                },
                Err(zmq::Error::EAGAIN) => {}
                Err(e) => tracing::error!("ZMQ recv error: {:?}", e),
            }

            Self::cancel_timed_out_streams(&socket, &mut pending);
        }
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
    ) {
        if let Ok(response) = rmp_serde::from_slice::<InferenceResponse>(data) {
            let request_id = response.request_id.clone();
            if let Some(pending_req) = pending.remove(&request_id) {
                match pending_req {
                    PendingRequest::Oneshot(tx) => {
                        if tx.send(response).is_err() {
                            tracing::debug!("Response receiver dropped for request {}", request_id);
                            let _ = Self::send_cancel(
                                socket,
                                &request_id,
                                CancelReason::ClientDisconnected,
                            );
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
                            let _ = Self::send_cancel(
                                socket,
                                &request_id,
                                CancelReason::ClientDisconnected,
                            );
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
                *deadline = Instant::now() + Duration::from_secs(180);
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
                let _ = Self::send_cancel(socket, &request_id, reason);
            }
            return;
        }

        tracing::error!(
            "Failed to deserialize response (neither InferenceResponse nor StreamChunk)"
        );
    }

    fn cancel_timed_out_streams(
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
                let _ = Self::send_cancel(socket, &request_id, CancelReason::StreamTimeout);
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

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow::anyhow!("Response channel closed")),
            Err(_) => {
                let _ = self.command_tx.try_send(RequestEnvelope::Cancel {
                    request_id: request_id.clone(),
                    reason: CancelReason::StreamTimeout,
                });
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

        Ok(StreamHandle {
            request_id,
            rx,
            command_tx: self.command_tx.clone(),
            finished: false,
        })
    }
}
