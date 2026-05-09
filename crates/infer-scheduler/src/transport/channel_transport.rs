//! In-process channel transport (for testing).
//!
//! Uses tokio mpsc channels instead of ZMQ. No serialization overhead.

use async_trait::async_trait;
use infer_protocol::{InferenceRequest, InferenceResponse, StreamChunk};
use tokio::sync::mpsc;

use crate::error::{Result, SchedulerError};
use crate::request::handle::ClientId;
use crate::transport::traits::{FrontendTransport, WorkerTransport};

// ═══════════════════════════════════════════════════════════════════════════════
//  Channel Frontend Transport
// ═══════════════════════════════════════════════════════════════════════════════

/// Frontend transport using in-process channels (for testing).
pub struct ChannelFrontendTransport {
    request_rx: mpsc::UnboundedReceiver<(ClientId, InferenceRequest)>,
    response_tx: mpsc::UnboundedSender<(ClientId, InferenceResponse)>,
    chunk_tx: mpsc::UnboundedSender<(ClientId, StreamChunk)>,
}

/// Handle for the test harness to send requests and receive responses.
pub struct ChannelFrontendHandle {
    pub request_tx: mpsc::UnboundedSender<(ClientId, InferenceRequest)>,
    pub response_rx: mpsc::UnboundedReceiver<(ClientId, InferenceResponse)>,
    pub chunk_rx: mpsc::UnboundedReceiver<(ClientId, StreamChunk)>,
}

/// Create a paired frontend transport and test handle.
pub fn channel_frontend() -> (ChannelFrontendTransport, ChannelFrontendHandle) {
    let (req_tx, req_rx) = mpsc::unbounded_channel();
    let (resp_tx, resp_rx) = mpsc::unbounded_channel();
    let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();

    let transport = ChannelFrontendTransport {
        request_rx: req_rx,
        response_tx: resp_tx,
        chunk_tx,
    };
    let handle = ChannelFrontendHandle {
        request_tx: req_tx,
        response_rx: resp_rx,
        chunk_rx,
    };
    (transport, handle)
}

#[async_trait]
impl FrontendTransport for ChannelFrontendTransport {
    async fn recv_request(&mut self) -> Result<(ClientId, InferenceRequest)> {
        self.request_rx
            .recv()
            .await
            .ok_or(SchedulerError::Shutdown)
    }

    async fn send_response(&mut self, client: &ClientId, response: InferenceResponse) -> Result<()> {
        self.response_tx
            .send((ClientId(client.0.clone()), response))
            .map_err(|_| SchedulerError::Shutdown)
    }

    async fn send_stream_chunk(&mut self, client: &ClientId, chunk: StreamChunk) -> Result<()> {
        self.chunk_tx
            .send((ClientId(client.0.clone()), chunk))
            .map_err(|_| SchedulerError::Shutdown)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Channel Worker Transport
// ═══════════════════════════════════════════════════════════════════════════════

/// Worker transport using in-process channels (for testing).
pub struct ChannelWorkerTransport {
    command_tx: mpsc::Sender<Vec<u8>>,
    output_rx: mpsc::Receiver<Vec<u8>>,
}

/// Handle for a mock worker to receive commands and send outputs.
pub struct ChannelWorkerHandle {
    pub command_rx: mpsc::Receiver<Vec<u8>>,
    pub output_tx: mpsc::Sender<Vec<u8>>,
}

/// Create a paired worker transport and mock worker handle.
pub fn channel_worker() -> (ChannelWorkerTransport, ChannelWorkerHandle) {
    let (cmd_tx, cmd_rx) = mpsc::channel(1);
    let (out_tx, out_rx) = mpsc::channel(1);

    let transport = ChannelWorkerTransport {
        command_tx: cmd_tx,
        output_rx: out_rx,
    };
    let handle = ChannelWorkerHandle {
        command_rx: cmd_rx,
        output_tx: out_tx,
    };
    (transport, handle)
}

#[async_trait]
impl WorkerTransport for ChannelWorkerTransport {
    async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
        self.command_tx
            .send(cmd)
            .await
            .map_err(|_| SchedulerError::Shutdown)
    }

    async fn recv_step_output(&mut self) -> Result<Vec<u8>> {
        self.output_rx
            .recv()
            .await
            .ok_or(SchedulerError::Shutdown)
    }
}
