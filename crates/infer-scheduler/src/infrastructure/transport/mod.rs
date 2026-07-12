//! Transport abstraction layer.
//!
//! Decouples the scheduler from the specific IPC mechanism.

pub mod codec;
pub mod control_plane;
pub mod traits;
pub mod zmq_transport;

pub use codec::{Codec, MsgPackCodec};
pub use control_plane::{
    ControlError, ControlEvent, ControlPlane, ControlPlaneCmdTx, ControlPlaneConfig,
    ControlPlaneEventRx, ControlResult, WorkerId,
};
pub use traits::{FrontendTransport, WorkerTransport};
