//! Request handle — response channel back to the HTTP server.

/// Handle for sending responses back to the client.
///
/// Holds the opaque client identity (ZMQ routing frame) and optionally
/// a channel for streaming chunks.
pub struct RequestHandle {
    /// ZMQ ROUTER identity frame (opaque bytes for routing response).
    pub client_id: ClientId,
    /// Whether this is a streaming request.
    pub stream: bool,
}

/// Opaque client identity for routing responses.
#[derive(Debug, Clone)]
pub struct ClientId(pub Vec<u8>);

impl ClientId {
    /// Create a dummy client ID (for testing).
    pub fn dummy() -> Self {
        Self(vec![0u8; 4])
    }
}

impl RequestHandle {
    /// Create a new request handle.
    pub fn new(client_id: ClientId, stream: bool) -> Self {
        Self { client_id, stream }
    }

    /// Create a no-op handle (for testing).
    pub fn noop() -> Self {
        Self {
            client_id: ClientId::dummy(),
            stream: false,
        }
    }
}
