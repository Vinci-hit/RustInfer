use half::bf16;

use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::decode_common::fail_decode_seqs;
use crate::application::kv_relief::{AllocWithReliefOutcome, alloc_with_relief};
use crate::application::runtime::Runtime;
use crate::application::worker_state::{ActiveSeqMap, DecodeRows, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::plan::{SeqStep, StepRequest, StopCriteria};
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;

/// DecodeEngine owns the worker-side decode row order.
///
/// `ActiveSeqMap` owns per-sequence facts; `DecodeRows` owns the stable
/// admission order so a burst of identical requests decodes as one cohort and
/// greedy output stays reproducible regardless of HashMap iteration order.
pub struct DecodeEngine {
    rows: DecodeRows,
}

impl DecodeEngine {
    pub fn new() -> Self {
        Self {
            rows: DecodeRows::new(),
        }
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn retain_active(&mut self, active: &ActiveSeqMap) {
        self.rows.retain_active(active);
    }

    /// Drive every active sequence one token forward through `Runtime::step`.
    ///
    /// One eager forward over a `DecodeOnly` batch: each row contributes a
    /// single token and one freshly allocated KV slot. The pool commits each
    /// seq's KV length internally; the worker mirrors it in `ActiveSeq` and
    /// owns the physical block release on finish.
    pub fn run_step<M>(
        &mut self,
        runner: &mut Runtime<bf16, Cuda, M>,
        active: &mut ActiveSeqMap,
        prefilling: &mut PrefillSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        data: &DataPump,
        eos_ids: &[i32],
        enable_prefix_caching: bool,
    ) -> OpResult<()>
    where
        M: DecoderModel<bf16, Cuda>,
    {
        if active.is_empty() {
            self.rows.clear();
            return Ok(());
        }

        let (order, new_indices) =
            match self.prepare_step(active, prefilling, kv_allocator, control, enable_prefix_caching)? {
                Some(ready) => ready,
                None => return Ok(()),
            };

        // Build the decode StepRequest: one new token + one new KV slot per row.
        let mut seqs = Vec::with_capacity(order.len());
        let mut assigned = Vec::with_capacity(order.len());
        let mut generated_counts = Vec::with_capacity(order.len());
        let mut max_tokens = Vec::with_capacity(order.len());
        let mut ignore_eos = Vec::with_capacity(order.len());
        for (i, &sid) in order.iter().enumerate() {
            let new_idx = new_indices[i];
            let seq = active
                .get(&sid)
                .expect("decode order row must be active after prepare_step");
            let mut block_table = Vec::with_capacity(seq.block_table.len() + 1);
            block_table.extend_from_slice(&seq.block_table);
            block_table.push(new_idx);
            seqs.push(SeqStep {
                sequence_id: sid,
                input_ids: vec![seq.last_token],
                positions: vec![seq.kv_len as i32],
                kv_write_start: seq.kv_len as i32,
                kv_len_after: (seq.kv_len + 1) as i32,
                block_table,
            });
            assigned.push(AssignedIndices {
                sequence_id: sid,
                base: new_idx,
                len: 1,
                token_ids: if enable_prefix_caching {
                    vec![seq.last_token]
                } else {
                    Vec::new()
                },
            });
            generated_counts.push(seq.generated_count as u32);
            max_tokens.push(seq.max_tokens as u32);
            ignore_eos.push(seq.ignore_eos);
        }

        let req = StepRequest {
            sampling: Vec::new(),
            stop: StopCriteria {
                eos_ids: eos_ids.to_vec(),
                generated_counts,
                max_tokens,
                ignore_eos,
            },
            draft_tokens: Vec::new(),
            seqs,
        };

        let out = match runner.step(&req) {
            Ok(out) => out,
            Err(e) => {
                if !new_indices.is_empty() {
                    kv_allocator.free(&new_indices);
                }
                fail_decode_seqs(
                    control,
                    active,
                    kv_allocator,
                    &order,
                    format!("decode step failed: {:?}", e),
                    enable_prefix_caching,
                );
                self.rows.clear();
                return Ok(());
            }
        };

        let output = self.commit_results(
            active,
            kv_allocator,
            &order,
            &new_indices,
            assigned,
            &out,
            enable_prefix_caching,
        )?;

        data.send_step_output(&output)
            .map_err(|e| OpError::Kernel(format!("data plane send_step_output failed: {}", e)))?;
        Ok(())
    }

    /// Materialize the row order, admit pending sequences, and allocate exactly
    /// one new KV slot per row (with relief). Returns `None` when there is
    /// nothing to run or the step already failed and reported its sequences.
    fn prepare_step(
        &mut self,
        active: &mut ActiveSeqMap,
        prefilling: &mut PrefillSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        enable_prefix_caching: bool,
    ) -> OpResult<Option<(Vec<u64>, Vec<u32>)>> {
        self.rows.retain_active(active);
        let pending = self.rows.pending_admissions(active);
        if !pending.is_empty() {
            self.rows.append_admissions(&pending);
        }

        let mut order: Vec<u64> = self.rows.as_slice().to_vec();
        if order.is_empty() {
            return Ok(None);
        }

        let initial_n = order.len() as u32;
        let new_indices = match alloc_with_relief(
            kv_allocator,
            control,
            active,
            prefilling,
            initial_n,
            enable_prefix_caching,
            true,
        ) {
            AllocWithReliefOutcome::Allocated(v) => v,
            AllocWithReliefOutcome::Unavailable => {
                let order: Vec<u64> = self.rows.as_slice().to_vec();
                tracing::warn!(
                    seqs = order.len(),
                    "decode alloc still failing after relief -- failing seqs"
                );
                fail_decode_seqs(
                    control,
                    active,
                    kv_allocator,
                    &order,
                    "worker KV pool exhausted at decode".to_string(),
                    enable_prefix_caching,
                );
                self.rows.clear();
                return Ok(None);
            }
            AllocWithReliefOutcome::Shutdown => return Err(OpError::Shutdown),
        };

        // Relief may have preempted active rows; resync against the survivors.
        self.rows.retain_active(active);
        order = self.rows.as_slice().to_vec();
        if order.is_empty() {
            if !new_indices.is_empty() {
                kv_allocator.free(&new_indices);
            }
            return Ok(None);
        }

        let new_indices: Vec<u32> = if new_indices.len() > order.len() {
            let (take, give_back) = new_indices.split_at(order.len());
            kv_allocator.free(give_back);
            take.to_vec()
        } else {
            new_indices
        };
        if new_indices.len() < order.len() {
            let failed: Vec<u64> = order[new_indices.len()..].to_vec();
            fail_decode_seqs(
                control,
                active,
                kv_allocator,
                &failed,
                format!(
                    "decode KV allocation returned {} slots for {} rows",
                    new_indices.len(),
                    order.len()
                ),
                enable_prefix_caching,
            );
            self.rows.retain_active(active);
            order.truncate(new_indices.len());
            if order.is_empty() {
                if !new_indices.is_empty() {
                    kv_allocator.free(&new_indices);
                }
                return Ok(None);
            }
        }

        Ok(Some((order, new_indices)))
    }

    fn commit_results(
        &mut self,
        active: &mut ActiveSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        order: &[u64],
        new_indices: &[u32],
        assigned: Vec<AssignedIndices>,
        out: &crate::domain::plan::StepOutput,
        enable_prefix_caching: bool,
    ) -> OpResult<StepOutput> {
        let mut output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
            assigned_indices: assigned,
        };

        let mut next_rows: Vec<u64> = Vec::with_capacity(order.len());
        let mut to_remove: Vec<u64> = Vec::new();
        for (i, &sid) in order.iter().enumerate() {
            let token = out
                .tokens
                .get(i)
                .and_then(|row| row.first())
                .map(|t| t.token_id)
                .unwrap_or(0);
            let finished = out.finished.get(i).copied().unwrap_or(true);
            let Some(&new_index) = new_indices.get(i) else {
                return Err(OpError::Shape(format!(
                    "decode commit missing allocated KV index for row {} seq {}",
                    i, sid
                )));
            };
            if let Some(seq) = active.get_mut(&sid) {
                seq.commit_accepted(token, 1, &[new_index]).map_err(|e| {
                    OpError::Shape(format!("decode commit failed for seq {}: {}", sid, e))
                })?;
            }
            output.tokens.push(GeneratedToken {
                sequence_id: sid,
                token_id: token,
                finished,
            });
            if finished {
                to_remove.push(sid);
            } else {
                next_rows.push(sid);
            }
        }
        for sid in &to_remove {
            if let Some(removed) = active.remove(sid) {
                kv_allocator.release_owned(&removed.block_table, enable_prefix_caching);
            }
        }
        self.rows.replace_rows(next_rows);
        Ok(output)
    }
}

impl Default for DecodeEngine {
    fn default() -> Self {
        Self::new()
    }
}
