//! `SchedulerEvent` — strongly-typed event consumed by the engine.
//!
//! All MsgPack deserialization happens in background decode tasks
//! *before* events reach the engine. The main event loop only
//! processes these typed variants — no `Vec<u8>` decoding on the
//! hot path.

use infer_protocol::server_to_scheduler::{CancelReason, InferenceRequest};
use infer_protocol::worker_to_scheduler_data::{DiffusionBatchOutput, StepOutput};

use crate::domain::inference_session::handle::ClientId;
use crate::infrastructure::transport::control_plane::ControlEvent;

/// Strongly-typed event that the engine processes.
///
/// Every variant carries a fully-decoded payload. Decode errors are
/// represented as `WorkerDecodeError`; the engine logs them and
/// continues.
#[derive(Debug)]
pub enum SchedulerEvent {
    /// New inference request from the frontend.
    NewRequest {
        client_id: ClientId,
        request: InferenceRequest,
    },
    /// Cancel request from the frontend.
    Cancel {
        external_id: String,
        reason: CancelReason,
    },
    /// Worker step output (LLM mode) — already decoded.
    WorkerLlmStep(StepOutput),
    /// Worker step output (Diffusion mode) — already decoded.
    WorkerDiffusionStep(DiffusionBatchOutput),
    /// Control-plane event (heartbeat, AllocFailed, etc.).
    ControlSignal(ControlEvent),
    /// Batch-accumulation deadline elapsed — flush deferred prefills.
    /// Only emitted when `batch_wait` is enabled (throughput mode).
    BatchTimer,
    /// Frontend transport closed.
    FrontendShutdown,
    /// Worker transport closed.
    WorkerShutdown,
    /// Deserialization error on worker data (non-fatal; log and skip).
    WorkerDecodeError(String),
    /// Non-fatal frontend recv error (log and continue).
    FrontendError(String),
}
