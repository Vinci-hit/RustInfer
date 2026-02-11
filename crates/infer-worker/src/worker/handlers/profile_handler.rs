//! Profile Handler
//!
//! Handles WorkerCommand::Profile.
//! State transition: Profiling → WaitingKVConfig

use infer_protocol::{WorkerResponse, ProfileParams, ProfileResult, ErrorCode};

use crate::worker::state_machine::WorkerState;

use super::super::Worker;
use super::make_error;

pub fn handle_profile(worker: &mut Worker, params: ProfileParams) -> WorkerResponse {
    println!("[Worker-{}] Profiling memory...", worker.worker_id());

    if !worker.is_model_loaded() {
        return make_error(worker, ErrorCode::InvalidState, "Model not loaded, cannot profile".into());
    }

    if let Err(e) = worker.refresh_memory_stats() {
        return make_error(worker, ErrorCode::DeviceError, format!("Failed to get memory stats: {}", e));
    }

    let memory_stats = worker.memory_stats().clone();

    // Reserve 20% headroom for activations
    let headroom = (memory_stats.total as f64 * 0.2) as u64;
    let available_kv_cache = memory_stats.free.saturating_sub(headroom);

    // Transition: Profiling → WaitingKVConfig
    if let Err(e) = worker.transition_state(WorkerState::WaitingKVConfig) {
        return make_error(worker, ErrorCode::InvalidState, e.to_string());
    }

    WorkerResponse::ProfileCompleted(ProfileResult {
        peak_memory_forward: memory_stats.workspace_memory,
        memory_model: memory_stats.model_memory,
        total_memory: memory_stats.total,
        available_kv_cache_memory: available_kv_cache,
        avg_prefill_time_ms: 0.0, // TODO: run actual profile forward
        avg_decode_time_ms: 0.0,
        profiled_batch_size: params.batch_size,
        profiled_seq_len: params.seq_len,
    })
}
