//! Command Handlers
//!
//! Each handler is a free function: `fn handle_xxx(worker, params) -> WorkerResponse`.
//! Handlers validate state, execute logic, then transition state.
//! Transport-independent — no ZMQ, no server dependency.

mod register_handler;
mod load_model_handler;
mod profile_handler;
mod init_kv_handler;
mod forward_handler;
mod unload_handler;

pub use register_handler::handle_register;
pub use load_model_handler::handle_load_model;
pub use profile_handler::handle_profile;
pub use init_kv_handler::handle_init_kv_cache;
pub use forward_handler::handle_forward;
pub use unload_handler::handle_unload_model;

use infer_protocol::{
    WorkerCommand, WorkerResponse, WorkerError, ErrorCode,
};

use super::Worker;
use super::state_machine::WorkerState;

/// Dispatch a WorkerCommand to the appropriate handler.
///
/// State-aware: validates that the current state allows the command,
/// returns InvalidState error if not.
pub fn dispatch(worker: &mut Worker, cmd: WorkerCommand) -> WorkerResponse {
    // Commands allowed in any state
    match &cmd {
        WorkerCommand::GetStatus => return handle_get_status(worker),
        WorkerCommand::HealthCheck => return WorkerResponse::Healthy,
        _ => {}
    }

    // State-gated commands
    let state = worker.state().clone();
    match cmd {
        WorkerCommand::Register(reg) => handle_register(worker, reg),

        WorkerCommand::LoadModel(params) => {
            if !matches!(state, WorkerState::WaitingModel) {
                return make_error(worker, ErrorCode::InvalidState,
                    format!("LoadModel requires WaitingModel state, current: {}", state));
            }
            handle_load_model(worker, params)
        }

        WorkerCommand::Profile(params) => {
            if !matches!(state, WorkerState::Profiling) {
                return make_error(worker, ErrorCode::InvalidState,
                    format!("Profile requires Profiling state, current: {}", state));
            }
            handle_profile(worker, params)
        }

        WorkerCommand::InitKVCache(params) => {
            if !matches!(state, WorkerState::WaitingKVConfig) {
                return make_error(worker, ErrorCode::InvalidState,
                    format!("InitKVCache requires WaitingKVConfig state, current: {}", state));
            }
            handle_init_kv_cache(worker, params)
        }

        WorkerCommand::Forward(params) => {
            if !state.can_forward() {
                return make_error(worker, ErrorCode::InvalidState,
                    format!("Forward requires Running state, current: {}", state));
            }
            handle_forward(worker, params)
        }

        WorkerCommand::UnloadModel => {
            if !matches!(state, WorkerState::Running) {
                return make_error(worker, ErrorCode::InvalidState,
                    format!("UnloadModel requires Running state, current: {}", state));
            }
            handle_unload_model(worker)
        }

        // Already handled above
        WorkerCommand::GetStatus | WorkerCommand::HealthCheck => unreachable!(),
    }
}

/// Build a WorkerStatus response (allowed in any state).
pub fn handle_get_status(worker: &Worker) -> WorkerResponse {
    use infer_protocol::{
        WorkerStatus,
        MemoryStats as ProtocolMemoryStats,
        PerformanceStats as ProtocolPerformanceStats,
    };

    let memory_stats = worker.memory_stats();
    let perf_stats = worker.perf_stats();
    let config = worker.config();

    WorkerResponse::Status(WorkerStatus {
        worker_id: worker.worker_id().to_string(),
        device_id: config.device_id,
        state: worker.state().to_protocol(),
        model_loaded: worker.is_model_loaded(),
        kv_cache_initialized: worker.is_kv_cache_initialized(),
        memory_stats: ProtocolMemoryStats {
            total: memory_stats.total,
            used: memory_stats.used,
            free: memory_stats.free,
            model_memory: memory_stats.model_memory,
            kv_cache_memory: memory_stats.kv_cache_memory,
            activation_memory: memory_stats.workspace_memory,
        },
        performance_stats: ProtocolPerformanceStats {
            total_requests: perf_stats.total_requests,
            total_tokens: perf_stats.total_tokens,
            avg_prefill_time_ms: perf_stats.avg_prefill_ms(),
            avg_decode_time_ms: perf_stats.avg_decode_per_token_ms(),
            throughput_tokens_per_sec: perf_stats.throughput_tps(),
            gpu_utilization: 0.0,
        },
        tp_rank: Some(config.tp_rank),
        tp_world_size: Some(config.tp_world_size),
    })
}

/// Helper: build a WorkerResponse::Error.
pub fn make_error(worker: &Worker, code: ErrorCode, message: String) -> WorkerResponse {
    eprintln!("[Worker-{}] Error: {:?} - {}", worker.worker_id(), code, message);
    WorkerResponse::Error(WorkerError {
        code,
        message,
        details: None,
        worker_id: worker.worker_id().to_string(),
        device_id: worker.config().device_id,
    })
}
