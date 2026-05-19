pub mod common;
pub mod scheduler_to_server;
pub mod scheduler_to_worker;
pub mod scheduler_to_worker_control;
pub mod server_to_scheduler;
pub mod worker_to_scheduler;
pub mod worker_to_scheduler_control;

#[cfg(test)]
mod syntax_test;

// Compatibility re-exports for existing call sites. New code should prefer the
// direction-specific modules above.
pub use common::{ProtocolError, ProtocolResult};
pub use scheduler_to_server::{
    ChunkType, InferenceMetrics, InferenceResponse, ResponseStatus, StreamChunk,
};
pub use scheduler_to_worker::{
    CancelRequest, DrainMode, DrainWorker, PrefillBatchCmd, PrefillSegmentCompletion,
    PrefillSegmentMeta, SamplingParams, UnloadModel, WorkerCommand,
};
pub use scheduler_to_worker_control::{LoadModel, SchedulerControlMessage, SchedulerHello};
pub use server_to_scheduler::InferenceRequest;
pub use worker_to_scheduler::{CancelAck, DrainAck, GeneratedToken, StepOutput};
pub use worker_to_scheduler_control::{
    WorkerCapacity, WorkerControlMessage, WorkerError, WorkerHeartbeat, WorkerHello,
    WorkerProgress, WorkerReady, WorkerState, WORKER_CONTROL_PROTOCOL_VERSION,
};
