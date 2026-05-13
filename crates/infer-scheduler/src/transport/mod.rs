//! Transport abstraction layer.
//!
//! Decouples the scheduler from the specific IPC mechanism.

pub mod traits;
pub mod zmq_transport;
pub mod worker_control;
pub mod channel_transport;
pub mod codec;

pub use traits::{FrontendTransport, WorkerTransport};
pub use codec::{Codec, MsgPackCodec};
