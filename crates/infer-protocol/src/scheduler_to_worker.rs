use serde::{Deserialize, Serialize};

use crate::common::{ProtocolError, ProtocolResult};

/// Scheduler -> Worker 的数据面命令。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerCommand {
    Prefill(PrefillBatchCmd),
    DiffusionBatch(DiffusionBatchCmd),
    Cancel(CancelRequest),
    Drain(DrainWorker),
    UnloadModel(UnloadModel),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    pub sequence_id: u64,
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

/// Scheduler -> Worker 的 prefill segment batch。
///
/// 每个 segment 明确描述：写入哪个 KV slot、写入 prompt/KV 的哪个绝对区间，
/// 以及该 segment 完成后是否进入 decode。`q_start_loc` 只表示该 segment 在
/// `input_ids` 扁平数组中的起点，不承载 KV 语义。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillBatchCmd {
    pub input_ids: Vec<i32>,
    pub q_start_loc: Vec<u32>,
    pub segments: Vec<PrefillSegmentMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillSegmentMeta {
    pub sequence_id: u64,
    pub kv_slot: u32,
    pub prompt_len: u32,
    pub segment_start: u32,
    pub segment_end: u32,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub completion: PrefillSegmentCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefillSegmentCompletion {
    /// 只写 KV，不进入 decode。
    ContinuePrefill,
    /// prompt 已完整写入 KV；本次 prefill 输出 token 是第一个生成 token。
    FinishPrefillAndStartDecode,
}

impl PrefillBatchCmd {
    pub fn num_requests(&self) -> usize {
        self.segments.len()
    }

    pub fn segment_token_range(&self, i: usize) -> std::ops::Range<usize> {
        let start = self.q_start_loc[i] as usize;
        let end = if i + 1 < self.q_start_loc.len() {
            self.q_start_loc[i + 1] as usize
        } else {
            self.input_ids.len()
        };
        start..end
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
                "PrefillBatchCmd must contain at least one segment",
            ));
        }
        if n > max_seqs {
            return Err(ProtocolError::invalid_argument(format!(
                "PrefillBatchCmd has {} segments, exceeds max_seqs {}",
                n, max_seqs
            )));
        }
        if self.input_ids.len() > max_batch_tokens {
            return Err(ProtocolError::invalid_argument(format!(
                "PrefillBatchCmd has {} tokens, exceeds max_batch_tokens {}",
                self.input_ids.len(), max_batch_tokens
            )));
        }
        if self.q_start_loc.len() != n {
            return Err(ProtocolError::invalid_argument(format!(
                "q_start_loc length {} != segments length {}",
                self.q_start_loc.len(), n
            )));
        }

        for i in 0..n {
            let range = self.segment_token_range(i);
            if range.start > range.end || range.end > self.input_ids.len() {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} has invalid token range [{}..{}) for input len {}",
                    i, range.start, range.end, self.input_ids.len()
                )));
            }
            if range.is_empty() {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} has empty token range",
                    i
                )));
            }

            let segment = &self.segments[i];
            if segment.sequence_id == 0 {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} has sequence_id=0",
                    i
                )));
            }
            let kv_slot = segment.kv_slot as usize;
            if kv_slot >= max_kv_slots {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} kv_slot {} out of range {}",
                    i, kv_slot, max_kv_slots
                )));
            }
            if segment.segment_end <= segment.segment_start {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} has invalid segment range [{}..{})",
                    i, segment.segment_start, segment.segment_end
                )));
            }
            if segment.segment_end > segment.prompt_len {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} end {} exceeds prompt_len {}",
                    i, segment.segment_end, segment.prompt_len
                )));
            }
            let segment_len = (segment.segment_end - segment.segment_start) as usize;
            if segment_len != range.len() {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} token len {} != segment len {}",
                    i, range.len(), segment_len
                )));
            }
            match segment.completion {
                PrefillSegmentCompletion::ContinuePrefill => {
                    if segment.segment_end >= segment.prompt_len {
                        return Err(ProtocolError::invalid_argument(format!(
                            "ContinuePrefill segment {} must end before prompt_len",
                            i
                        )));
                    }
                }
                PrefillSegmentCompletion::FinishPrefillAndStartDecode => {
                    if segment.segment_end != segment.prompt_len {
                        return Err(ProtocolError::invalid_argument(format!(
                            "FinishPrefill segment {} must end at prompt_len",
                            i
                        )));
                    }
                }
            }
            if segment.max_tokens == 0 {
                return Err(ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} has max_tokens=0",
                    i
                )));
            }
            segment.sampling_params.validate().map_err(|e| {
                ProtocolError::invalid_argument(format!(
                    "PrefillBatchCmd segment {} invalid sampling params: {}",
                    i, e
                ))
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionBatchCmd {
    pub requests: Vec<DiffusionBatchItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionBatchItem {
    pub request_id: String,
    /// Original prompt text for logging / response metadata.
    pub prompt: String,
    /// Server-tokenized prompt ids consumed by the Worker text encoder.
    pub prompt_input_ids: Vec<i32>,
    pub negative_prompt: Option<String>,
    pub negative_prompt_input_ids: Option<Vec<i32>>,
    pub height: u32,
    pub width: u32,
    pub num_inference_steps: usize,
    pub sigmas: Option<Vec<f32>>,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub output_format: String,
}

impl DiffusionBatchCmd {
    pub fn validate(&self, max_batch_size: usize) -> ProtocolResult<()> {
        if self.requests.is_empty() {
            return Err(ProtocolError::invalid_argument(
                "DiffusionBatchCmd must contain at least one request",
            ));
        }
        if self.requests.len() > max_batch_size {
            return Err(ProtocolError::invalid_argument(format!(
                "DiffusionBatchCmd has {} requests, exceeds max_batch_size {}",
                self.requests.len(), max_batch_size
            )));
        }
        for (i, req) in self.requests.iter().enumerate() {
            if req.request_id.is_empty() {
                return Err(ProtocolError::invalid_argument(format!(
                    "DiffusionBatchCmd request {} has empty request_id",
                    i
                )));
            }
            if req.prompt.is_empty() {
                return Err(ProtocolError::invalid_argument(format!(
                    "DiffusionBatchCmd request {} has empty prompt",
                    i
                )));
            }
            if req.prompt_input_ids.is_empty() {
                return Err(ProtocolError::invalid_argument(format!(
                    "DiffusionBatchCmd request {} has empty server-tokenized prompt_input_ids",
                    i
                )));
            }
            if req.height == 0 || req.width == 0 || req.height % 16 != 0 || req.width % 16 != 0 {
                return Err(ProtocolError::invalid_argument(format!(
                    "DiffusionBatchCmd request {} invalid shape {}x{}; dimensions must be positive multiples of 16",
                    i, req.height, req.width
                )));
            }
            if req.num_inference_steps == 0 && req.sigmas.as_ref().map(|s| s.is_empty()).unwrap_or(true) {
                return Err(ProtocolError::invalid_argument(format!(
                    "DiffusionBatchCmd request {} needs num_inference_steps > 0 or non-empty sigmas",
                    i
                )));
            }
        }
        Ok(())
    }
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
