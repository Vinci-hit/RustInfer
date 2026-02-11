//! Register Handler
//!
//! Handles WorkerCommand::Register — validates rank and returns RegisterAck.

use infer_protocol::{WorkerResponse, WorkerRegistration, WorkerRegistrationAck};

use super::super::Worker;

pub fn handle_register(worker: &mut Worker, registration: WorkerRegistration) -> WorkerResponse {
    println!("[Worker-{}] Received registration request", worker.worker_id());

    let expected_rank = worker.config().tp_rank;

    if registration.rank != expected_rank {
        return WorkerResponse::RegisterAck(WorkerRegistrationAck {
            status: "rejected".to_string(),
            message: format!(
                "Rank mismatch: expected {}, got {}",
                expected_rank, registration.rank,
            ),
            scheduler_protocol_version: Some("0.1.0".to_string()),
            assigned_worker_id: None,
        });
    }

    WorkerResponse::RegisterAck(WorkerRegistrationAck {
        status: "ok".to_string(),
        message: format!("Worker {} registered successfully", worker.worker_id()),
        scheduler_protocol_version: Some("0.1.0".to_string()),
        assigned_worker_id: None,
    })
}
