//! Forward Handler
//!
//! Handles WorkerCommand::Forward — builds tensors from ForwardParams,
//! runs model forward, and returns sampling output.

use infer_protocol::{WorkerResponse, ForwardParams, ErrorCode};

use crate::tensor::Tensor;

use super::super::Worker;
use super::make_error;

pub fn handle_forward(worker: &mut Worker, params: ForwardParams) -> WorkerResponse {
    // 1. Basic checks
    if !worker.is_model_loaded() {
        return make_error(worker, ErrorCode::InvalidState, "Model not loaded".into());
    }

    let device = worker.device_type();

    // 2. Build tensors on device
    let input_tokens = match Tensor::from_slice(&params.input_ids, device) {
        Ok(t) => t,
        Err(e) => return make_error(worker, ErrorCode::ForwardFailed, format!("Input tensor error: {}", e)),
    };

    let positions = match Tensor::from_slice(&params.position_ids, device) {
        Ok(t) => t,
        Err(e) => return make_error(worker, ErrorCode::ForwardFailed, format!("Pos tensor error: {}", e)),
    };

    let kv_indices_i32: Vec<i32> = params.block_tables.iter().map(|&x| x as i32).collect();
    let kv_indices = match Tensor::from_slice(&kv_indices_i32, device) {
        Ok(t) => t,
        Err(e) => return make_error(worker, ErrorCode::ForwardFailed, format!("KV indices error: {}", e)),
    };

    let kv_indptr_i32: Vec<i32> = params.context_lens.iter().map(|&x| x as i32).collect();
    let kv_indptr = match Tensor::from_slice(&kv_indptr_i32, device) {
        Ok(t) => t,
        Err(e) => return make_error(worker, ErrorCode::ForwardFailed, format!("KV indptr error: {}", e)),
    };

    let new_slots = match Tensor::from_slice(&params.slot_mapping, device) {
        Ok(t) => t,
        Err(e) => return make_error(worker, ErrorCode::ForwardFailed, format!("SlotMap error: {}", e)),
    };

    // 3. Execute forward
    let sampling_output = match worker.forward(
        &input_tokens,
        &positions,
        &kv_indices,
        &kv_indptr,
        &new_slots,
        params.num_decode_tokens,
    ) {
        Ok(output) => output,
        Err(e) => return make_error(worker, ErrorCode::ForwardFailed, format!("Forward failed: {}", e)),
    };

    // 4. Return result
    WorkerResponse::ForwardCompleted(sampling_output)
}
