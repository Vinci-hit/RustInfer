//! Init KV Cache Handler
//!
//! Handles WorkerCommand::InitKVCache.
//! State transitions: WaitingKVConfig → InitializingKV → Running

use std::time::Instant;

use infer_protocol::{WorkerResponse, InitKVCacheParams, KVCacheInfo, ErrorCode};

use crate::worker::state_machine::WorkerState;

use super::super::Worker;
use super::make_error;

pub fn handle_init_kv_cache(worker: &mut Worker, params: InitKVCacheParams) -> WorkerResponse {
    println!(
        "[Worker-{}] Initializing KV Cache: {} blocks x {} size (dtype: {})",
        worker.worker_id(), params.num_blocks, params.block_size, params.dtype,
    );

    // Transition: WaitingKVConfig → InitializingKV
    if let Err(e) = worker.transition_state(WorkerState::InitializingKV) {
        return make_error(worker, ErrorCode::InvalidState, e.to_string());
    }

    let start_time = Instant::now();

    if let Err(e) = worker.init_kv_cache(&params) {
        let _ = worker.transition_state(WorkerState::Error(format!("KV init failed: {}", e)));
        return make_error(worker, ErrorCode::KVCacheInitFailed, format!("Failed to init KV cache: {}", e));
    }

    let init_time_ms = start_time.elapsed().as_millis() as u64;

    // Transition: InitializingKV → Running
    if let Err(e) = worker.transition_state(WorkerState::Running) {
        return make_error(worker, ErrorCode::InvalidState, e.to_string());
    }

    let kv_cache = worker.kv_cache().expect("KV cache just initialized");
    let memory_used = kv_cache.memory_bytes() as u64;
    let bytes_per_block = (params.block_size
        * (params.num_heads as usize)
        * (params.head_dim as usize)
        * 2   // K + V
        * 2)  // BF16 = 2 bytes
        as u64;

    WorkerResponse::KVCacheInitialized(KVCacheInfo {
        allocated_blocks: params.num_blocks,
        memory_used,
        bytes_per_block,
        total_capacity_tokens: params.num_blocks * params.block_size,
        init_time_ms,
    })
}
