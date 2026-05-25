//! Transport abstraction layer.
//!
//! Decouples the scheduler from the specific IPC mechanism.

pub mod traits;
pub mod zmq_transport;
pub mod control_plane;
pub mod codec;

pub use traits::{FrontendTransport, WorkerTransport};
pub use codec::{Codec, MsgPackCodec};
pub use control_plane::{
    ControlError, ControlEvent, ControlPlane, ControlPlaneCmdTx, ControlPlaneConfig,
    ControlPlaneEventRx, ControlResult, WorkerId,
};
