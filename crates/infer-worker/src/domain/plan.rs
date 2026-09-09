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
    /// Empty for ordinary execution. Otherwise one draft row per sequence,
    /// with `seq.input_ids == [pending_token] + draft_tokens[row]`.
    /// A row with K drafts requires K+1 target prediction rows, including the
    /// independent correction/bonus prediction. K=0 is a valid verify row.
    pub draft_tokens: Vec<Vec<i32>>,
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub tokens: Vec<Vec<SampledToken>>,
    /// Leading input tokens eligible for retention by the caller. Ordinary
    /// steps retain every input; verification retains the pending token plus
    /// accepted drafts, shortened if output stops early. The last emitted
    /// token remains pending rather than becoming part of this input prefix.
    /// The caller owns KV leases and must return the unretained suffix.
    pub materialized_tokens: Vec<u32>,
    /// Consecutive matching drafts before EOS/output-budget truncation.
    /// `None` for ordinary execution; these counts are NOT KV increments.
    pub accepted_drafts: Option<Vec<u32>>,
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
