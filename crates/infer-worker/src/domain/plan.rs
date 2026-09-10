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

impl StepRequest {
    /// CPU staging paired with the device workspace. Reuse nested vectors too.
    pub(crate) fn workspace(batch: usize, tokens: usize, blocks: usize) -> Self {
        Self {
            seqs: (0..batch)
                .map(|_| SeqStep {
                    sequence_id: 0,
                    input_ids: Vec::with_capacity(tokens),
                    positions: Vec::with_capacity(tokens),
                    kv_write_start: 0,
                    kv_len_after: 0,
                    block_table: Vec::with_capacity(blocks),
                })
                .collect(),
            sampling: Vec::with_capacity(batch),
            stop: StopCriteria {
                eos_ids: Vec::with_capacity(16),
                generated_counts: Vec::with_capacity(batch),
                max_tokens: Vec::with_capacity(batch),
                ignore_eos: Vec::with_capacity(batch),
            },
            draft_tokens: Vec::new(),
        }
    }

    pub(crate) fn retain_from(&mut self, req: &Self, lengths: &[u32]) {
        self.seqs.resize_with(req.seqs.len(), || SeqStep {
            sequence_id: 0,
            input_ids: Vec::new(),
            positions: Vec::new(),
            kv_write_start: 0,
            kv_len_after: 0,
            block_table: Vec::new(),
        });
        for ((dst, src), &n) in self.seqs.iter_mut().zip(&req.seqs).zip(lengths) {
            dst.sequence_id = src.sequence_id;
            dst.input_ids.clear();
            dst.input_ids
                .extend_from_slice(&src.input_ids[..n as usize]);
            dst.positions.clear();
            dst.positions
                .extend_from_slice(&src.positions[..n as usize]);
            dst.kv_write_start = src.kv_write_start;
            dst.kv_len_after = src.kv_write_start + n as i32;
            dst.block_table.clone_from(&src.block_table);
        }
        self.sampling.clone_from(&req.sampling);
        self.stop.eos_ids.clone_from(&req.stop.eos_ids);
        self.stop
            .generated_counts
            .clone_from(&req.stop.generated_counts);
        self.stop.max_tokens.clone_from(&req.stop.max_tokens);
        self.stop.ignore_eos.clone_from(&req.stop.ignore_eos);
        self.draft_tokens.clear();
    }
}

#[cfg(test)]
mod request_workspace_tests {
    use super::*;

    #[test]
    fn retaining_prefixes_reuses_nested_storage_without_mutating_request() {
        let mut request = StepRequest::workspace(1, 4, 16);
        request.seqs[0].input_ids.extend([10, 11, 12, 13]);
        request.seqs[0].positions.extend([7, 8, 9, 10]);
        request.seqs[0].kv_write_start = 7;
        request.seqs[0].kv_len_after = 11;
        request.seqs[0].block_table.extend(0..10);
        request.draft_tokens.push(vec![11, 12, 13]);
        let mut retained = StepRequest::workspace(1, 4, 16);
        let ids = retained.seqs[0].input_ids.as_ptr();
        let blocks = retained.seqs[0].block_table.as_ptr();
        for n in [1, 4, 2] {
            retained.retain_from(&request, &[n]);
            assert_eq!(
                retained.seqs[0].input_ids,
                request.seqs[0].input_ids[..n as usize]
            );
            assert_eq!(retained.seqs[0].kv_len_after, 7 + n as i32);
            assert!(retained.draft_tokens.is_empty());
            assert_eq!(retained.seqs[0].input_ids.as_ptr(), ids);
            assert_eq!(retained.seqs[0].block_table.as_ptr(), blocks);
        }
        assert_eq!(request.seqs[0].input_ids.len(), 4);
        assert_eq!(request.draft_tokens[0], [11, 12, 13]);
    }
}
