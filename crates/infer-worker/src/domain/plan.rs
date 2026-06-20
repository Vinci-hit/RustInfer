pub use infer_core::plan::{BatchKind, BatchPlan, MaskMode, RAGGED_Q_TILE};
// SampledToken now lives with the Sampler interface in infer-core::ports; the
// runtime result types (StepOutput etc.) below still reference it via this path.
pub use infer_core::ports::SampledToken;

#[derive(Debug, Clone)]
pub struct SeqStep {
    pub sequence_id: u64,
    pub input_ids: Vec<i32>,
    pub positions: Vec<i32>,
    pub kv_write_start: i32,
    pub kv_len_after: i32,
    pub block_table: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct StepRequest {
    pub seqs: Vec<SeqStep>,
    pub sampling: Vec<crate::domain::ports::sampler::SamplingParams>,
    pub stop: StopCriteria,
    pub draft_tokens: Vec<Vec<i32>>,
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub tokens: Vec<Vec<SampledToken>>,
    pub accepted: Vec<u32>,
    pub finished: Vec<bool>,
    pub hidden_tap: Option<HiddenTap>,
}

#[derive(Debug, Clone)]
pub struct StopCriteria {
    pub eos_ids: Vec<i32>,
    pub generated_counts: Vec<u32>,
    pub max_tokens: Vec<u32>,
    pub ignore_eos: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct HiddenTap {
    pub at_layer: usize,
}
