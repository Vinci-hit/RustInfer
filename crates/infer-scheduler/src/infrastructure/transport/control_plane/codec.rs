//! MsgPack encode/decode helpers for the control-plane envelope.
//!
//! Centralized so error mapping is identical at every send/receive site.

use infer_protocol::worker_to_scheduler_control::WorkerControlMessage;
use infer_protocol::{ControlEnvelope, scheduler_to_worker_control::SchedulerControlMessage};

use super::handle::{ControlError, ControlResult};

#[inline]
pub(crate) fn encode_scheduler(
    env: &ControlEnvelope<SchedulerControlMessage>,
) -> ControlResult<Vec<u8>> {
    rmp_serde::to_vec(env)
        .map_err(|e| ControlError::Encode(format!("scheduler control envelope: {}", e)))
}

#[inline]
pub(crate) fn decode_worker(bytes: &[u8]) -> ControlResult<ControlEnvelope<WorkerControlMessage>> {
    rmp_serde::from_slice(bytes)
        .map_err(|e| ControlError::Decode(format!("worker control envelope: {}", e)))
}
