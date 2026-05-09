//! Transport traits — abstract async communication.

use async_trait::async_trait;
use infer_protocol::{InferenceRequest, InferenceResponse, StreamChunk};

use crate::error::Result;
use crate::request::handle::ClientId;

/// Abstract frontend transport (HTTP Server ↔ Scheduler).
#[async_trait]
pub trait FrontendTransport: Send + Sync + 'static {
    /// Receive the next inference request from the HTTP server.
    async fn recv_request(&mut self) -> Result<(ClientId, InferenceRequest)>;

    /// Send a complete response back to the HTTP server.
    async fn send_response(&mut self, client: &ClientId, response: InferenceResponse) -> Result<()>;

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
