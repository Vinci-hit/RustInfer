use std::collections::HashMap;

use crate::application::hosting::ModelHost;
use crate::domain::dtype::Dtype;
use crate::domain::kv::SeqId;
use crate::domain::plan::{SeqStep, StepOutput, StepRequest, StopCriteria};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::sampler::SamplingParams;
use crate::domain::ports::{OpError, OpResult};

#[derive(Debug, Clone)]
pub struct SpecActiveSeq {
    pub input_ids: Vec<i32>,
    pub positions: Vec<i32>,
    pub kv_write_start: i32,
    pub kv_len_after: i32,
    pub block_table: Vec<u32>,
    pub sampling: SamplingParams,
    pub draft_tokens: Vec<i32>,
    pub eos_ids: Vec<i32>,
    pub generated_count: u32,
    pub max_tokens: u32,
    pub ignore_eos: bool,
}

pub type ActiveSeqMap = HashMap<SeqId, SpecActiveSeq>;

pub struct SpecRuntime {
    pub num_draft: usize,
}

impl SpecRuntime {
    pub fn step_request<T: Dtype, D: LlmBackend>(
        &mut self,
        host: &mut ModelHost<T, D>,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        if self.num_draft == 0 {
            return Err(OpError::Shape(
                "SpecRuntime::step_request: num_draft=0".into(),
            ));
        }
        if req.draft_tokens.is_empty() {
            return Err(OpError::Shape(
                "SpecRuntime::step_request requires draft_tokens".into(),
            ));
        }
        for (i, draft) in req.draft_tokens.iter().enumerate() {
            if draft.len() > self.num_draft {
                return Err(OpError::Shape(format!(
                    "SpecRuntime::step_request: seq[{}] draft len {} > num_draft {}",
                    i,
                    draft.len(),
                    self.num_draft
                )));
            }
        }
        host.primary.step(req)
    }

    pub fn step<T: Dtype, D: LlmBackend>(
        &mut self,
        host: &mut ModelHost<T, D>,
        active: &mut ActiveSeqMap,
    ) -> OpResult<StepOutput> {
        if active.is_empty() {
            return Ok(StepOutput {
                tokens: Vec::new(),
                accepted: Vec::new(),
                finished: Vec::new(),
                hidden_tap: None,
            });
        }
        let req = request_from_active(active)?;
        self.step_request(host, &req)
    }
}

fn request_from_active(active: &ActiveSeqMap) -> OpResult<StepRequest> {
    let mut entries = active.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(sid, _)| **sid);

    let mut eos_ids = Vec::new();
    let mut seqs = Vec::with_capacity(entries.len());
    let mut sampling = Vec::with_capacity(entries.len());
    let mut draft_tokens = Vec::with_capacity(entries.len());
    let mut generated_counts = Vec::with_capacity(entries.len());
    let mut max_tokens = Vec::with_capacity(entries.len());
    let mut ignore_eos = Vec::with_capacity(entries.len());

    for (&sid, seq) in entries {
        if seq.input_ids.is_empty() {
            return Err(OpError::Shape(format!(
                "SpecRuntime::step: seq {} has no input ids",
                sid
            )));
        }
        if seq.input_ids.len() != seq.positions.len() {
            return Err(OpError::Shape(format!(
                "SpecRuntime::step: seq {} input_ids {} != positions {}",
                sid,
                seq.input_ids.len(),
                seq.positions.len()
            )));
        }
        if seq.input_ids.len() != seq.draft_tokens.len() {
            return Err(OpError::Shape(format!(
                "SpecRuntime::step: seq {} input_ids {} != draft_tokens {}",
                sid,
                seq.input_ids.len(),
                seq.draft_tokens.len()
            )));
        }
        seqs.push(SeqStep {
            sequence_id: sid,
            input_ids: seq.input_ids.clone(),
            positions: seq.positions.clone(),
            kv_write_start: seq.kv_write_start,
            kv_len_after: seq.kv_len_after,
            block_table: seq.block_table.clone(),
        });
        sampling.push(seq.sampling);
        draft_tokens.push(seq.draft_tokens.clone());
        generated_counts.push(seq.generated_count);
        max_tokens.push(seq.max_tokens);
        ignore_eos.push(seq.ignore_eos);
        eos_ids.extend_from_slice(&seq.eos_ids);
    }
    eos_ids.sort_unstable();
    eos_ids.dedup();

    Ok(StepRequest {
        seqs,
        sampling,
        stop: StopCriteria {
            eos_ids,
            generated_counts,
            max_tokens,
            ignore_eos,
        },
        draft_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::hosting::{ErasedRuntime, ModelHost};
    use crate::domain::component::LayerRange;
    use crate::domain::exec::TopologyShape;
    use crate::domain::model::ModelDims;
    use crate::domain::plan::{HiddenTap, SampledToken};
    use crate::infrastructure::cpu::Cpu;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeRuntime {
        calls: Arc<AtomicUsize>,
    }

    impl ErasedRuntime<f32, Cpu> for FakeRuntime {
        fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(StepOutput {
                tokens: vec![vec![SampledToken {
                    token_id: req.draft_tokens[0][0],
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                }]],
                accepted: vec![1],
                finished: vec![false],
                hidden_tap: None,
            })
        }

        fn run_layers(&mut self, range: LayerRange, _req: &StepRequest) -> OpResult<HiddenTap> {
            Ok(HiddenTap {
                at_layer: range.end,
            })
        }

        fn prime_graphs(&mut self) -> OpResult<()> {
            Ok(())
        }

        fn dims(&self) -> &ModelDims {
            static DIMS: ModelDims = ModelDims {
                dim: 0,
                q_dim: 0,
                kv_dim: 0,
                qkv_dim: 0,
                intermediate_size: 0,
                vocab_size: 0,
                head_num: 0,
                head_dim: 0,
                kv_head_num: 0,
                num_layers: 0,
                num_experts: 0,
                experts_per_tok: 0,
                moe_intermediate_size: 0,
                num_shared_experts: 0,
            };
            &DIMS
        }
    }

    fn request_with_draft(draft: Vec<i32>) -> StepRequest {
        StepRequest {
            seqs: vec![SeqStep {
                sequence_id: 7,
                input_ids: draft.clone(),
                positions: (0..draft.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: draft.len() as i32,
                block_table: vec![0],
            }],
            sampling: Vec::new(),
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: vec![0],
                max_tokens: vec![16],
                ignore_eos: vec![false],
            },
            draft_tokens: vec![draft],
        }
    }

    fn active_with_draft(draft: Vec<i32>) -> ActiveSeqMap {
        let mut active = ActiveSeqMap::new();
        active.insert(
            7,
            SpecActiveSeq {
                input_ids: draft.clone(),
                positions: (0..draft.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: draft.len() as i32,
                block_table: vec![0],
                sampling: Default::default(),
                draft_tokens: draft,
                eos_ids: Vec::new(),
                generated_count: 0,
                max_tokens: 16,
                ignore_eos: false,
            },
        );
        active
    }

    #[test]
    fn step_request_delegates_to_primary_runtime() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut host = ModelHost {
            primary: Box::new(FakeRuntime {
                calls: Arc::clone(&calls),
            }),
            aux: Vec::new(),
            topology: TopologyShape::SINGLE,
        };
        let mut spec = SpecRuntime { num_draft: 2 };

        let out = spec
            .step_request::<f32, Cpu>(&mut host, &request_with_draft(vec![42]))
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(out.accepted, vec![1]);
        assert_eq!(out.tokens[0][0].token_id, 42);
    }

    #[test]
    fn step_request_rejects_draft_longer_than_configured() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut host = ModelHost {
            primary: Box::new(FakeRuntime { calls }),
            aux: Vec::new(),
            topology: TopologyShape::SINGLE,
        };
        let mut spec = SpecRuntime { num_draft: 1 };

        let err = spec
            .step_request::<f32, Cpu>(&mut host, &request_with_draft(vec![1, 2]))
            .unwrap_err();

        assert!(format!("{err:?}").contains("draft len 2 > num_draft 1"));
    }

    #[test]
    fn step_builds_request_from_active_sequences() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut host = ModelHost {
            primary: Box::new(FakeRuntime {
                calls: Arc::clone(&calls),
            }),
            aux: Vec::new(),
            topology: TopologyShape::SINGLE,
        };
        let mut spec = SpecRuntime { num_draft: 2 };
        let mut active = active_with_draft(vec![9]);

        let out = spec.step::<f32, Cpu>(&mut host, &mut active).unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(out.tokens[0][0].token_id, 9);
    }
}
