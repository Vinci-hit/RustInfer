//! Unload Handler
//!
//! Handles WorkerCommand::UnloadModel.
//! State transitions: Running → Unloading → WaitingModel

use infer_protocol::WorkerResponse;

use crate::worker::state_machine::WorkerState;

use super::super::Worker;
use super::make_error;

pub fn handle_unload_model(worker: &mut Worker) -> WorkerResponse {
    println!("[Worker-{}] Unloading model...", worker.worker_id());

    // Transition: Running → Unloading
    if let Err(e) = worker.transition_state(WorkerState::Unloading) {
        return make_error(worker, infer_protocol::ErrorCode::InvalidState, e.to_string());
    }

    // Release resources
    worker.unload_model();

    // Transition: Unloading → WaitingModel
    if let Err(e) = worker.transition_state(WorkerState::WaitingModel) {
        return make_error(worker, infer_protocol::ErrorCode::InvalidState, e.to_string());
    }

    WorkerResponse::ModelUnloaded
}
