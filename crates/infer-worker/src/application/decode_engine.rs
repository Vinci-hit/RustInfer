use half::bf16;

use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::decode_common::fail_decode_seqs;
use crate::application::kv_relief::{AllocWithReliefOutcome, alloc_with_relief};
use crate::application::runtime::{DecodeCompactOutput, Runtime};
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
/// One decode step issued on the GPU but not yet finalized. Held across one
/// `run_step` call so the next step's compute overlaps this step's host commit.
struct PendingDecode {
    order: Vec<u64>,
    new_indices: Vec<u32>,
    assigned: Vec<AssignedIndices>,
    batch: usize,
}

pub struct DecodeEngine {
    rows: DecodeRows,
    /// Sequence ids whose decode token currently sits in buffer A, in row
    /// order, as left by the last successful ABC step's compact merge. The
    /// longest common prefix of this and the next step's row order is the
    /// portion of A that can be reused without a host token upload.
    prev_a_rows: Vec<u64>,
    /// The in-flight ABC step (issued, awaiting `finalize_decode_abc`). The
    /// 1-deep pipeline: at most one step is in flight at a time.
    pending: Option<PendingDecode>,
}

impl DecodeEngine {
    pub fn new() -> Self {
        Self {
            rows: DecodeRows::new(),
            prev_a_rows: Vec::new(),
            pending: None,
        }
    }

    /// Hard reset. Drops any in-flight step WITHOUT finalizing it (its tokens
    /// are lost) — only call on drain/shutdown, never mid-stream.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.prev_a_rows.clear();
        self.pending = None;
    }

    /// True while a step is issued but not yet finalized. The serve loop must
    /// keep calling `run_step` until this is false so the last step's tokens
    /// are collected and sent even after `active` drains to empty.
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Free the in-flight step's freshly-allocated KV slots and drop it. Those
    /// slots are NOT yet in any seq's block table (`commit_results` appends
    /// them), so an `Immediate` drain that evicts every seq would otherwise
    /// leak them. Call this BEFORE `clear()` on drain.
    pub fn reclaim_pending(&mut self, kv_allocator: &mut GlobalKvAllocator) {
        if let Some(p) = self.pending.take() {
            if !p.new_indices.is_empty() {
                kv_allocator.free(&p.new_indices);
            }
        }
    }

    pub fn retain_active(&mut self, active: &ActiveSeqMap) {
        self.rows.retain_active(active);
    }

    /// Drive the GPU-resident decode loop one step (synchronous).
    ///
    /// Issue + finalize a single step in one call. All GPU compute work
    /// (forward + finalize + argmax + compact merge) is enqueued BEFORE
    /// `synchronize_copy_out` blocks the CPU, so the GPU stays busy the
    /// entire time — no idle bubble. This mirrors the baseline's
    /// `step_decode_abc_compact` which was also fully synchronous.
    ///
    /// The 1-deep pipeline approach (issue step N, finalize step N-1) is
    /// fundamentally broken here because `prepare_step` reads `active` seq
    /// state (last_token, kv_len, block_table) that is only updated by
    /// `commit_results` from the PRIOR step's finalize. Issuing before
    /// finalizing would read stale seq state and the pending overwrite would
    /// silently drop every other step's tokens.
    #[allow(clippy::too_many_arguments)]
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
        if active.is_empty() && !self.has_pending() {
            self.rows.clear();
            return Ok(());
        }

        // 1. Issue a new step (enqueues ALL compute: forward + argmax + merge
        //    + async copy-out on So).
        self.issue_new(runner, active, prefilling, kv_allocator, control, eos_ids, enable_prefix_caching)?;

        // 2. Finalize immediately — sync So, read host mirrors, commit results.
        //    GPU compute is already enqueued so the sync overlaps it.
        let to_send = self.finalize_pending(runner, active, kv_allocator, control, enable_prefix_caching)?;

        // 3. Send the step's output.
        if let Some(output) = to_send {
            data.send_step_output(&output)
                .map_err(|e| OpError::Kernel(format!("data plane send_step_output failed: {}", e)))?;
        }
        Ok(())
    }

    /// Collect + commit the in-flight step. Returns its `StepOutput` to send.
    fn finalize_pending<M>(
        &mut self,
        runner: &mut Runtime<bf16, Cuda, M>,
        active: &mut ActiveSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        enable_prefix_caching: bool,
    ) -> OpResult<Option<StepOutput>>
    where
        M: DecoderModel<bf16, Cuda>,
    {
        let Some(p) = self.pending.take() else {
            return Ok(None);
        };
        match runner.finalize_decode_abc(p.batch) {
            Ok(compact) => {
                let output = self.commit_results(
                    active,
                    kv_allocator,
                    &p.order,
                    &p.new_indices,
                    p.assigned,
                    &compact,
                    enable_prefix_caching,
                )?;
                // A now holds the surviving tokens compacted to the front in
                // `rows` order (commit_results just set `rows` to the survivors).
                self.prev_a_rows = self.rows.as_slice().to_vec();
                Ok(Some(output))
            }
            Err(e) => {
                if !p.new_indices.is_empty() {
                    kv_allocator.free(&p.new_indices);
                }
                fail_decode_seqs(
                    control,
                    active,
                    kv_allocator,
                    &p.order,
                    format!("decode finalize failed: {:?}", e),
                    enable_prefix_caching,
                );
                self.rows.clear();
                self.prev_a_rows.clear();
                Ok(None)
            }
        }
    }

    /// Prepare and asynchronously issue a new decode step, stashing it as the
    /// pending in-flight step.
    #[allow(clippy::too_many_arguments)]
    fn issue_new<M>(
        &mut self,
        runner: &mut Runtime<bf16, Cuda, M>,
        active: &mut ActiveSeqMap,
        prefilling: &mut PrefillSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        eos_ids: &[i32],
        enable_prefix_caching: bool,
    ) -> OpResult<()>
    where
        M: DecoderModel<bf16, Cuda>,
    {
        let (order, new_indices) =
            match self.prepare_step(active, prefilling, kv_allocator, control, enable_prefix_caching)? {
                Some(ready) => ready,
                None => return Ok(()),
            };

        let req = build_decode_request(
            &order,
            &new_indices,
            active,
            eos_ids,
            enable_prefix_caching,
        );

        // ABC A-reuse: the leading rows of `order` that match the prior step's
        // device-row order already hold the right token in buffer A (written by
        // last step's compact merge), so only the divergent suffix re-uploads.
        let a_valid_prefix = common_prefix_len(&order, &self.prev_a_rows);

        match runner.issue_decode_abc(
            &req.req,
            a_valid_prefix,
            &req.generated_counts,
            &req.max_tokens,
            &req.ignore_eos,
            eos_ids,
        ) {
            Ok(()) => {
                let batch = order.len();
                self.pending = Some(PendingDecode {
                    order,
                    new_indices,
                    assigned: req.assigned,
                    batch,
                });
                Ok(())
            }
            Err(e) => {
                if !new_indices.is_empty() {
                    kv_allocator.free(&new_indices);
                }
                fail_decode_seqs(
                    control,
                    active,
                    kv_allocator,
                    &order,
                    format!("decode issue failed: {:?}", e),
                    enable_prefix_caching,
                );
                self.rows.clear();
                // A's contents are unknown after a failed issue; force a full
                // re-upload on the next step.
                self.prev_a_rows.clear();
                Ok(())
            }
        }
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

    /// Commit one ABC compact step. The merge already split surviving vs
    /// finished rows; here we (1) advance every row that produced a token —
    /// appending this step's `new_index` to its block table, INCLUDING finished
    /// rows, so that (2) removing a finished row reclaims its full KV (all prior
    /// blocks plus the slot allocated this step). Surviving rows become the next
    /// device-row order, in compaction (active_src_rows) order, matching A.
    fn commit_results(
        &mut self,
        active: &mut ActiveSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        order: &[u64],
        new_indices: &[u32],
        assigned: Vec<AssignedIndices>,
        compact: &DecodeCompactOutput,
        enable_prefix_caching: bool,
    ) -> OpResult<StepOutput> {
        let mut output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
            assigned_indices: assigned,
        };

        // (token, finished) per original row, plus the next-step row order.
        let mut row_results: Vec<Option<(i32, bool)>> = vec![None; order.len()];
        let mut next_rows: Vec<u64> = Vec::with_capacity(compact.active.len());
        for row in &compact.active {
            if row.src_row >= order.len() {
                continue;
            }
            row_results[row.src_row] = Some((row.token_id, false));
            next_rows.push(order[row.src_row]);
        }
        let mut to_remove: Vec<u64> = Vec::with_capacity(compact.finished.len());
        for row in &compact.finished {
            if row.src_row >= order.len() {
                continue;
            }
            row_results[row.src_row] = Some((row.token_id, true));
            to_remove.push(order[row.src_row]);
        }

        for (i, &sid) in order.iter().enumerate() {
            let Some((token, finished)) = row_results[i] else {
                continue;
            };
            let Some(&new_index) = new_indices.get(i) else {
                return Err(OpError::Shape(format!(
                    "decode commit missing allocated KV index for row {} seq {}",
                    i, sid
                )));
            };
            // Append the slot allocated this step to EVERY row that ran —
            // finished rows too — so the release below reclaims it. If the seq
            // was cancelled/preempted out-of-band while this step was in flight
            // (pipelined: control is drained between issue and finalize), it is
            // gone from `active` and its other blocks were already released —
            // so free this step's orphaned slot directly to avoid a leak.
            if let Some(seq) = active.get_mut(&sid) {
                seq.commit_accepted(token, 1, &[new_index]).map_err(|e| {
                    OpError::Shape(format!("decode commit failed for seq {}: {}", sid, e))
                })?;
            } else {
                kv_allocator.free(&[new_index]);
            }
            output.tokens.push(GeneratedToken {
                sequence_id: sid,
                token_id: token,
                finished,
            });
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

/// Length of the longest common prefix of two row orders.
fn common_prefix_len(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// The per-step decode request plus the stop metadata vectors the merge needs.
struct DecodeRequestBuild {
    req: StepRequest,
    generated_counts: Vec<u32>,
    max_tokens: Vec<u32>,
    ignore_eos: Vec<bool>,
    assigned: Vec<AssignedIndices>,
}

/// Build the decode `StepRequest`: one new token + one new KV slot per row,
/// in `order` row order. Each row's block table is its committed table plus
/// the slot freshly allocated for this step.
fn build_decode_request(
    order: &[u64],
    new_indices: &[u32],
    active: &ActiveSeqMap,
    eos_ids: &[i32],
    enable_prefix_caching: bool,
) -> DecodeRequestBuild {
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
            generated_counts: generated_counts.clone(),
            max_tokens: max_tokens.clone(),
            ignore_eos: ignore_eos.clone(),
        },
        draft_tokens: Vec::new(),
        seqs,
    };

    DecodeRequestBuild {
        req,
        generated_counts,
        max_tokens,
        ignore_eos,
        assigned,
    }
}

impl Default for DecodeEngine {
    fn default() -> Self {
        Self::new()
    }
}
