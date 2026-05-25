//! Transport traits — abstract async communication.

use async_trait::async_trait;
use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
use infer_protocol::server_to_scheduler::{CancelReason, InferenceRequest};

use crate::error::Result;
use crate::domain::inference_session::handle::ClientId;

pub enum FrontendEvent {
    Infer {
        client_id: ClientId,
        request: InferenceRequest,
    },
    /// Client-issued cancel. The id is the **client-supplied external id**
    /// (string from the HTTP request body); engine resolves it to an
    /// internal `RequestId(Uuid)` via the `by_external_id` index.
    Cancel {
        external_id: String,
        reason: CancelReason,
    },
}

/// Abstract frontend transport (HTTP Server ↔ Scheduler).
#[async_trait]
pub trait FrontendTransport: Send + Sync + 'static {
    /// Receive the next frontend event from the HTTP server.
    async fn recv_event(&mut self) -> Result<FrontendEvent>;

    /// Send a complete response back to the HTTP server.
    async fn send_response(&mut self, client: &ClientId, response: InferenceResponse)
    -> Result<()>;

    /// Send a streaming chunk back to the HTTP server.
    async fn send_stream_chunk(&mut self, client: &ClientId, chunk: StreamChunk) -> Result<()>;
}

/// Abstract worker transport (Scheduler ↔ Worker).
#[async_trait]
pub trait WorkerTransport: Send + Sync + 'static {
    /// Send a batch command to the worker.
    async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()>;

    /// Receive step output from the worker.
    async fn recv_step_output(&mut self) -> Result<Vec<u8>>;
}
