use serde::{Deserialize, Serialize};

use crate::common::{ProtocolError, ProtocolResult};

/// Scheduler -> Worker 的数据面命令。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerCommand {
    Prefill(PrefillBatchCmd),
    Cancel(CancelRequest),
    Drain(DrainWorker),
    UnloadModel(UnloadModel),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainWorker {
    pub mode: DrainMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DrainMode {
    Graceful,
    Immediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnloadModel {
    pub model_instance_id: String,
}

/// Scheduler -> Worker 的 prefill 请求 batch。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillBatchCmd {
    pub input_ids: Vec<i32>,
    pub q_start_loc: Vec<u32>,
    pub num_computed_tokens: Vec<u32>,
    pub kv_slots: Vec<u32>,
    pub sampling_params: Vec<SamplingParams>,
    pub request_metas: Vec<RequestMeta>,
}

impl PrefillBatchCmd {
    pub fn num_requests(&self) -> usize {
        self.q_start_loc.len()
    }

    pub fn validate(
        &self,
        max_batch_tokens: usize,
        max_seqs: usize,
        max_kv_slots: usize,
    ) -> ProtocolResult<()> {
        let n = self.num_requests();
        if n == 0 {
            return Err(ProtocolError::invalid_argument(
                "PrefillBatchCmd must contain at least one request",
            ));
        }
        if n > max_seqs {
            return Err(ProtocolError::invalid_argument(format!(
                "PrefillBatchCmd has {} requests, exceeds max_seqs {}",
                n, max_seqs
            )));
        }
        if self.input_ids.len() > max_batch_tokens {
            return Err(ProtocolError::invalid_argument(format!(
                "PrefillBatchCmd has {} tokens, exceeds max_batch_tokens {}",
                self.input_ids.len(), max_batch_tokens
            )));
        }

        let lens = [
            ("num_computed_tokens", self.num_computed_tokens.len()),
            ("kv_slots", self.kv_slots.len()),
            ("sampling_params", self.sampling_params.len()),
            ("request_metas", self.request_metas.len()),
        ];
        for (name, len) in lens {
            if len != n {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd field {} length {} != q_start_loc length {}",
                    name, len, n
                )));
            }
        }

        for i in 0..n {
            let start = self.q_start_loc[i] as usize;
            let end = if i + 1 < n {
                self.q_start_loc[i + 1] as usize
            } else {
                self.input_ids.len()
            };
            if start > end || end > self.input_ids.len() {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd request {} has invalid token range [{}..{}) for input len {}",
                    i,
                    start,
                    end,
                    self.input_ids.len()
                )));
            }
            if start == end {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd request {} has empty token range",
                    i
                )));
            }
            let kv_slot = self.kv_slots[i] as usize;
            if kv_slot >= max_kv_slots {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd request {} kv_slot {} out of range {}",
                    i, kv_slot, max_kv_slots
                )));
            }
            if self.request_metas[i].request_id.is_empty() {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd request {} has empty request_id",
                    i
                )));
            }
            if self.request_metas[i].max_tokens == 0 {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd request {} has max_tokens=0",
                    i
                )));
            }
            self.sampling_params[i].validate().map_err(|e| {
                ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd request {} invalid sampling params: {}",
                    i, e
                ))
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMeta {
    pub request_id: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl SamplingParams {
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be finite and >= 0");
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err("top_p must be finite and in [0, 1]");
        }
        if self.top_k < -1 {
            return Err("top_k must be -1 or non-negative");
        }
        Ok(())
    }
}
