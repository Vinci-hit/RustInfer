use crate::domain::exec::MaskHandle;

pub const RAGGED_Q_TILE: i32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    Full,
    Causal,
    SlidingWindow { window: u32 },
    Tree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    DecodeOnly,
    Ragged,
    Spec {
        mask: MaskMode,
        mask_handle: Option<MaskHandle>,
    },
}

#[derive(Debug, Clone)]
pub struct BatchPlan {
    pub kind: BatchKind,
    pub num_tokens: usize,
    pub batch: usize,
    pub q_lens: Vec<i32>,
    pub kv_lens: Vec<i32>,
    pub seq_positions: Vec<i32>,
    pub rope_positions: Vec<i32>,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub total_q_tiles: i32,
}

impl BatchPlan {
    pub fn is_decode_only(&self) -> bool {
        matches!(self.kind, BatchKind::DecodeOnly)
    }

    pub fn plan_ragged_tiles(q_lens: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut cu_q_lens = Vec::with_capacity(q_lens.len() + 1);
        cu_q_lens.push(0);

        let mut running = 0;
        let mut block2req = Vec::new();
        let mut block2tile = Vec::new();

        for (req, &q_len) in q_lens.iter().enumerate() {
            running += q_len;
            cu_q_lens.push(running);

            let tiles = (q_len + RAGGED_Q_TILE - 1) / RAGGED_Q_TILE;
            for tile in 0..tiles {
                block2req.push(req as i32);
                block2tile.push(tile);
            }
        }

        (cu_q_lens, block2req, block2tile)
    }
}

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
pub struct SampledToken {
    pub token_id: i32,
    pub logprob: f32,
    pub top_logprobs: Vec<(i32, f32)>,
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
