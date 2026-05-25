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
///
/// The inner `Vec<u8>` is the ZMQ ROUTER identity frame; treat it as opaque.
/// Construct via [`ClientId::new`] and consume via [`ClientId::as_bytes`] /
/// [`ClientId::into_bytes`] / [`ClientId::clone`]. The field is private so
/// external crates cannot reach in and break the routing invariant.
#[derive(Debug, Clone)]
pub struct ClientId(Vec<u8>);

impl ClientId {
    /// Wrap raw routing bytes (typically the ZMQ socket identity).
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Borrow the routing bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Consume self into the raw routing bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

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
