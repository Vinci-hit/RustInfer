//! Load Model Handler
//!
//! Handles WorkerCommand::LoadModel.
//! State transitions: WaitingModel → LoadingModel → Profiling

use std::time::Instant;

use infer_protocol::{WorkerResponse, ModelLoadParams, ModelLoadedInfo, ErrorCode};

use crate::base::DeviceType;
use crate::model::architectures::llama3::Llama3;
use crate::worker::state_machine::WorkerState;

use super::super::Worker;
use super::make_error;

pub fn handle_load_model(worker: &mut Worker, params: ModelLoadParams) -> WorkerResponse {
    println!("[Worker-{}] Loading model from {}", worker.worker_id(), params.model_path);

    // Transition: WaitingModel → LoadingModel
    if let Err(e) = worker.transition_state(WorkerState::LoadingModel) {
        return make_error(worker, ErrorCode::InvalidState, e.to_string());
    }

    let start_time = Instant::now();

    // Determine device type
    #[cfg(feature = "cuda")]
    let device_type = DeviceType::Cuda(params.device_id as i32);
    #[cfg(not(feature = "cuda"))]
    let device_type = DeviceType::Cpu;

    // Create model
    let model: Box<dyn crate::model::Model> = match Llama3::new(
        std::path::Path::new(&params.model_path),
        device_type,
        128,  // max_batch_size — TODO: from config
        16,   // block_size — TODO: from config
    ) {
        Ok(m) => Box::new(m),
        Err(e) => {
            let _ = worker.transition_state(WorkerState::Error(format!("Model load failed: {}", e)));
            return make_error(
                worker,
                ErrorCode::ModelLoadFailed,
                format!("Failed to create model: {}", e),
            );
        }
    };

    // Load into worker
    if let Err(e) = worker.load_model(model) {
        let _ = worker.transition_state(WorkerState::Error(format!("Model load failed: {}", e)));
        return make_error(
            worker,
            ErrorCode::ModelLoadFailed,
            format!("Failed to load model into worker: {}", e),
        );
    }

    let load_time_ms = start_time.elapsed().as_millis() as u64;

    // Transition: LoadingModel → Profiling
    if let Err(e) = worker.transition_state(WorkerState::Profiling) {
        return make_error(worker, ErrorCode::InvalidState, e.to_string());
    }

    // Gather stats
    let _ = worker.refresh_memory_stats();
    let memory_stats = worker.memory_stats();

    let num_parameters = worker
        .model()
        .map(|m| m.config().estimate_num_parameters())
        .unwrap_or(0);

    WorkerResponse::ModelLoaded(ModelLoadedInfo {
        worker_id: worker.worker_id().to_string(),
        device_id: params.device_id,
        model_name: "llama3".to_string(),
        num_parameters,
        memory_used: memory_stats.used,
        tp_rank: params.tp_rank,
        tp_world_size: params.tp_world_size,
        load_time_ms,
    })
}
