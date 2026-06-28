use half::bf16;

use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::decode_common::fail_decode_seqs;
use crate::application::kv_relief::{AllocWithReliefOutcome, alloc_with_relief};
use crate::application::runtime::{DecodeCompactOutput, Runtime};
use crate::application::worker_state::{ActiveSeqMap, DecodeRows, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::plan::{SampledToken, SeqStep, StepRequest, StopCriteria};
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
    /// True when this step ran the async `compact_extend` (device control for
    /// the next step is ready). Drives whether commit records `device_rows`.
    device_prepared: bool,
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
    /// KV slots speculatively reserved during the PREVIOUS step's in-flight
    /// forward, for the NEXT step's rows. Reserving here moves the bulk KV
    /// allocation off the decode critical path (it overlaps GPU compute instead
    /// of running after the copy-out sync). Sized to the issuing step's batch on
    /// the optimistic assumption every row survives; `prepare_step` consumes
    /// these, frees any surplus when rows finished, and tops up for fresh
    /// admissions. Best-effort: under a tight pool nothing is reserved and
    /// `prepare_step` falls back to the on-path relief alloc. These slots are
    /// outstanding-but-unowned (in no block table), so they must be freed on
    /// drain/idle to avoid a pool leak.
    prealloc: Vec<u32>,
    /// Device-resident decode control plane: each step maintains the per-row
    /// control (block tables, kv_lens, positions) ON DEVICE via
    /// `compact_extend_control` instead of rebuilding + uploading it from the
    /// host, cutting the per-step host cost from O(batch * seq_len) to O(batch).
    /// Used whenever the next-step KV slots were speculatively reserved (the
    /// common case); the host-upload path is the fallback on the first step,
    /// under pool pressure, or any step whose row set changed via an admission
    /// or out-of-band eviction.
    ///
    /// `device_rows` is the row order whose control plane the previous step's
    /// `compact_extend` left ready on device (= last step's survivors). When it
    /// matches the current step's order exactly the device control is valid and
    /// the host rebuild+upload is skipped; otherwise the step re-seeds it.
    device_rows: Vec<u64>,
}

impl DecodeEngine {
    pub fn new() -> Self {
        Self {
            rows: DecodeRows::new(),
            prev_a_rows: Vec::new(),
            pending: None,
            prealloc: Vec::new(),
            device_rows: Vec::new(),
        }
    }

    /// Hard reset. Drops any in-flight step WITHOUT finalizing it (its tokens
    /// are lost) — only call on drain/shutdown, never mid-stream. `prealloc` is
    /// reclaimed separately by `reclaim_pending` (it needs the allocator).
    pub fn clear(&mut self) {
        self.rows.clear();
        self.prev_a_rows.clear();
        self.pending = None;
        self.device_rows.clear();
    }

    /// True while a step is issued but not yet finalized. The serve loop must
    /// keep calling `run_step` until this is false so the last step's tokens
    /// are collected and sent even after `active` drains to empty.
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Free the in-flight step's freshly-allocated KV slots and drop it, plus
    /// any speculatively-reserved next-step slots. Neither set is yet in a seq's
    /// block table (`commit_results` appends the in-flight ones; `prepare_step`
    /// consumes the speculative ones), so an `Immediate` drain that evicts every
    /// seq would otherwise leak them. Call this BEFORE `clear()` on drain.
    pub fn reclaim_pending(&mut self, kv_allocator: &mut GlobalKvAllocator) {
        if let Some(p) = self.pending.take() {
            if !p.new_indices.is_empty() {
                kv_allocator.free(&p.new_indices);
            }
        }
        self.release_prealloc(kv_allocator);
    }

    /// Return any speculatively-reserved next-step slots to the pool. Called
    /// when no next step will consume them (active drained to empty, or drain).
    fn release_prealloc(&mut self, kv_allocator: &mut GlobalKvAllocator) {
        if !self.prealloc.is_empty() {
            let slots = std::mem::take(&mut self.prealloc);
            kv_allocator.free(&slots);
        }
    }

    /// Best-effort reservation of `n` KV slots for the NEXT decode step, issued
    /// while THIS step's forward runs on the GPU. Plain alloc (no relief): on a
    /// tight pool it reserves nothing and the next `prepare_step` allocs on-path
    /// as before. Speculation must never preempt live rows.
    fn reserve_next_step_slots(&mut self, kv_allocator: &mut GlobalKvAllocator, n: usize) {
        debug_assert!(self.prealloc.is_empty(), "prealloc must be drained before reserving");
        if n == 0 {
            return;
        }
        if let Ok(slots) = kv_allocator.alloc_indices(n as u32) {
            self.prealloc = slots;
        }
    }

    pub fn retain_active(&mut self, active: &ActiveSeqMap) {
        self.rows.retain_active(active);
    }

    /// Finalize the in-flight ABC decode step (if any) and send its output.
    /// Used before a fused (prefill+decode) step so the prior step's tokens are
    /// committed and the copy-out is synced — the fused eager forward then can't
    /// race buffer A. Does NOT issue a new step.
    pub(crate) fn finalize_and_send<M>(
        &mut self,
        runner: &mut Runtime<bf16, Cuda, M>,
        active: &mut ActiveSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        data: &DataPump,
        enable_prefix_caching: bool,
    ) -> OpResult<()>
    where
        M: DecoderModel<bf16, Cuda>,
    {
        if let Some(output) =
            self.finalize_pending(runner, active, kv_allocator, control, enable_prefix_caching)?
        {
            data.send_step_output(&output).map_err(|e| {
                OpError::Kernel(format!("data plane send_step_output (fused finalize) failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Prepare the current decode rows for a fused step: materialize the row
    /// order and allocate exactly one new KV slot per row (prealloc fast path /
    /// relief fallback), WITHOUT reserving next-step slots — a fused step does
    /// not pipeline. Returns `None` (releasing any stranded speculative
    /// reservation) when there are no active decode rows this step.
    pub(crate) fn prepare_fused_decode(
        &mut self,
        active: &mut ActiveSeqMap,
        prefilling: &mut PrefillSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        enable_prefix_caching: bool,
    ) -> OpResult<Option<(Vec<u64>, Vec<u32>)>> {
        let ready = self.prepare_step(active, prefilling, kv_allocator, control, enable_prefix_caching)?;
        if ready.is_none() {
            // No decode rows: a speculative reservation from a prior ABC step
            // would otherwise be stranded (the fused path never consumes it).
            self.release_prealloc(kv_allocator);
            self.rows.clear();
        }
        Ok(ready)
    }

    /// Commit a fused step's decode-row outputs (rows `[0, order.len())` of the
    /// eager `StepOutput`). Mirrors `commit_results` but is driven by the eager
    /// per-row `tokens`/`finished` instead of the ABC compact output: each
    /// surviving row appends its new slot, finished rows free their KV in one
    /// batched call, and every row emits a `GeneratedToken`. Resets the ABC
    /// ephemeral state (`prev_a_rows`, `device_rows`) so the next pure-decode
    /// step re-seeds buffer A and the device control plane from the host.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_fused_decode(
        &mut self,
        active: &mut ActiveSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        order: &[u64],
        new_indices: &[u32],
        tokens: &[Vec<SampledToken>],
        finished: &[bool],
        enable_prefix_caching: bool,
        output: &mut StepOutput,
    ) -> OpResult<()> {
        let mut to_free: Vec<u32> = Vec::new();
        let mut to_remove: Vec<u64> = Vec::new();
        for (i, &sid) in order.iter().enumerate() {
            let token = tokens.get(i).and_then(|r| r.first()).map(|t| t.token_id).unwrap_or(0);
            let done = finished.get(i).copied().unwrap_or(false);
            let Some(&new_index) = new_indices.get(i) else {
                return Err(OpError::Shape(format!(
                    "fused decode commit missing KV index for row {} seq {}",
                    i, sid
                )));
            };
            if let Some(seq) = active.get_mut(&sid) {
                seq.commit_accepted(token, 1, &[new_index]).map_err(|e| {
                    OpError::Shape(format!("fused decode commit failed for seq {}: {}", sid, e))
                })?;
            } else {
                // Seq evicted out-of-band; its other blocks are already gone —
                // free this step's orphaned slot to avoid a leak.
                to_free.push(new_index);
            }
            output.tokens.push(GeneratedToken {
                sequence_id: sid,
                token_id: token,
                finished: done,
            });
            if done {
                to_remove.push(sid);
            }
        }
        for sid in &to_remove {
            if let Some(removed) = active.remove(sid) {
                if !enable_prefix_caching {
                    to_free.extend_from_slice(&removed.block_table);
                }
            }
        }
        if !to_free.is_empty() {
            kv_allocator.free(&to_free);
        }
        self.rows.retain_active(active);
        // The fused step bypassed the ABC pipeline: buffer A and the device
        // control plane are stale. Force the next decode step to re-seed both.
        self.prev_a_rows.clear();
        self.device_rows.clear();
        Ok(())
    }

    /// Reset ABC row/buffer state after a failed fused step. The fused forward's
    /// decode rows have been failed and removed from `active`; drop their row
    /// order and force a host re-seed next step.
    pub(crate) fn reset_after_fused(&mut self) {
        self.rows.clear();
        self.prev_a_rows.clear();
        self.device_rows.clear();
    }

    /// Drive the GPU-resident decode loop one step, pipelined 1-deep.
    ///
    /// Order matters: (1) finalize the step issued on the *previous* call —
    /// drain its copy-out, commit its tokens, reclaim finished KV; (2) issue a
    /// new step (append B, forward, merge, async copy-out) if there is work;
    /// (3) send the finalized output *after* issuing, so the new step's GPU
    /// compute overlaps the previous step's host commit + the ZMQ send + the
    /// serve loop's inter-step work. The serve loop must keep calling this while
    /// `has_pending()` so the last step is collected after `active` empties.
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
        // Track whether this call starts with nothing in flight. If so, the
        // 1-deep pipeline would otherwise emit *no* token this iteration
        // (finalize sees `pending=None` → `to_send=None`) and the first decode
        // token of a fresh stream would not reach the scheduler until the NEXT
        // serve-loop turn — doubling the inter-token latency between the
        // prefill's first token and the first pure-decode token.
        //
        // Detection of "cold start": no in-flight step AND we're about to
        // issue work this round. In that case, after issuing, immediately
        // finalize + send the freshly-issued step (degenerating to 0-deep for
        // this one iteration). Steady-state decode keeps the 1-deep pipeline
        // because `pending` will be non-empty on entry.
        let cold_start = self.pending.is_none() && !active.is_empty();

        // 1. Finalize the in-flight step (issued last call) and commit it.
        let to_send = self.finalize_pending(runner, active, kv_allocator, control, enable_prefix_caching)?;

        // 2. Issue a new step if there is work; otherwise idle (and return any
        //    speculatively-reserved slots — no next step will consume them).
        if active.is_empty() {
            self.rows.clear();
            self.release_prealloc(kv_allocator);
        } else {
            self.issue_new(runner, active, prefilling, kv_allocator, control, eos_ids, enable_prefix_caching)?;
        }

        // 3. Send the previous step's output AFTER issuing the new step so the
        //    send (and the serve loop's following work) overlaps GPU compute.
        if let Some(output) = to_send {
            data.send_step_output(&output)
                .map_err(|e| OpError::Kernel(format!("data plane send_step_output failed: {}", e)))?;
        }

        // 4. Cold-start drain: if this call started the pipeline from empty
        //    AND `issue_new` actually queued a step, immediately finalize +
        //    send it so the FIRST decode token after prefill is not delayed
        //    by a full extra round-trip. We then re-issue so the next call
        //    re-enters steady-state with a non-empty `pending` — otherwise
        //    every subsequent call would also see `pending=None` and the
        //    pipeline would never start.
        if cold_start && self.pending.is_some() {
            let drained = self.finalize_pending(runner, active, kv_allocator, control, enable_prefix_caching)?;
            if let Some(output) = drained {
                data.send_step_output(&output)
                    .map_err(|e| OpError::Kernel(format!("data plane send_step_output (cold) failed: {}", e)))?;
            }
            // Re-issue so steady-state pipelining resumes on the next call.
            // `commit_results` (inside `finalize_pending`) may have emptied
            // `active` (every row finished its single token); guard for that.
            if !active.is_empty() {
                self.issue_new(runner, active, prefilling, kv_allocator, control, eos_ids, enable_prefix_caching)?;
            } else {
                self.rows.clear();
                self.release_prealloc(kv_allocator);
            }
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
                    p.device_prepared,
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
                // Device control state is unknown after a finalize failure;
                // force a host re-seed next step and drop stale reservations.
                self.device_rows.clear();
                self.release_prealloc(kv_allocator);
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

        // Reserve the NEXT step's KV slots before issuing. In the async path the
        // device control builder (`compact_extend`) appends these to each
        // survivor on-device; in the proven path they become the next step's
        // `prepare_step` reservation (drawn off the critical path either way).
        self.reserve_next_step_slots(kv_allocator, order.len());

        // The device control plane is usable this step only if the pool gave a
        // full next-step reservation (compact_extend appends one slot per
        // survivor; a short reservation would leave survivors without a slot).
        let device_ctrl_ok = self.prealloc.len() == order.len();
        // Reuse the device-resident control plane when the row set is exactly
        // what the previous step's compact_extend prepared — no admission, no
        // out-of-band eviction. Otherwise re-seed it from the host request.
        let reuse_device_control = device_ctrl_ok && self.device_rows == order;
        let next_slots: Option<Vec<u32>> = if device_ctrl_ok {
            Some(self.prealloc.clone())
        } else {
            None
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
            next_slots.as_deref(),
            reuse_device_control,
        ) {
            Ok(()) => {
                let batch = order.len();
                self.pending = Some(PendingDecode {
                    order,
                    new_indices,
                    assigned: req.assigned,
                    batch,
                    device_prepared: device_ctrl_ok,
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
                // The device control plane was not advanced (compact_extend did
                // not run / is unreliable) and the reserved next-step slots are
                // orphaned; force a host re-seed next step and reclaim them.
                self.device_rows.clear();
                self.release_prealloc(kv_allocator);
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

        let initial_n = order.len();
        let new_indices = if self.prealloc.len() >= initial_n {
            // Fast path: slots reserved during the previous step's forward cover
            // this step. Take exactly `initial_n`; return any surplus (rows that
            // finished since the speculative reservation) to the pool.
            let mut slots = std::mem::take(&mut self.prealloc);
            if slots.len() > initial_n {
                let surplus = slots.split_off(initial_n);
                kv_allocator.free(&surplus);
            }
            slots
        } else {
            // Speculation fell short (batch grew via admissions, or the pool was
            // too tight to reserve last step). Return the partial reservation
            // and allocate the full set on the critical path, with relief —
            // identical to the pre-speculation behaviour.
            self.release_prealloc(kv_allocator);
            match alloc_with_relief(
                kv_allocator,
                control,
                active,
                prefilling,
                initial_n as u32,
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
            }
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
        device_prepared: bool,
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

        // Accumulate every slot this step returns to the pool — orphaned
        // per-step slots and finished sequences' block tables — and free them
        // in ONE call below. `GlobalKvAllocator::free` is O(pool) per call
        // (compact + merge), so a single merged free per step replaces one pass
        // per sequence (previously K full-pool passes when K rows finished, and
        // a full-pool pass to return a single orphaned slot).
        let mut to_free: Vec<u32> = Vec::new();
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
                to_free.push(new_index);
            }
            output.tokens.push(GeneratedToken {
                sequence_id: sid,
                token_id: token,
                finished,
            });
        }
        for sid in &to_remove {
            if let Some(removed) = active.remove(sid) {
                // `release_owned` with prefix caching ON is a no-op (slots stay
                // pinned by the scheduler RadixTree); with it OFF the blocks
                // return to the pool. Accumulate the OFF case into the shared
                // batch instead of one `free()` per finished sequence.
                if !enable_prefix_caching {
                    to_free.extend_from_slice(&removed.block_table);
                }
            }
        }
        // One merged free for the whole step. `free()` sorts + dedups its input
        // internally, so combining orphaned slots with finished block tables is
        // correct and costs a single compact+merge pass.
        if !to_free.is_empty() {
            kv_allocator.free(&to_free);
        }
        // If the async control builder ran this step, the device control plane
        // now holds exactly these survivors — record them so the next step can
        // detect an unchanged row set and skip the host rebuild+upload. If it
        // did not run (proven path / pool too tight), the device is not prepared.
        if device_prepared {
            self.device_rows = next_rows.clone();
        } else {
            self.device_rows.clear();
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
pub(crate) struct DecodeRequestBuild {
    pub(crate) req: StepRequest,
    pub(crate) generated_counts: Vec<u32>,
    pub(crate) max_tokens: Vec<u32>,
    pub(crate) ignore_eos: Vec<bool>,
    pub(crate) assigned: Vec<AssignedIndices>,
}

/// Build the decode `StepRequest`: one new token + one new KV slot per row,
/// in `order` row order. Each row's block table is its committed table plus
/// the slot freshly allocated for this step.
pub(crate) fn build_decode_request(
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

#[cfg(test)]
mod prealloc_tests {
    use super::*;

    // Slot conservation for the speculative next-step reservation. These guard
    // the two leak-critical seams (`reserve_next_step_slots`, `release_prealloc`
    // + the surplus split in `prepare_step`) without needing a GPU `Runtime`.

    #[test]
    fn reserve_then_release_conserves_pool() {
        let mut kv = GlobalKvAllocator::new(16);
        let mut eng = DecodeEngine::new();
        eng.reserve_next_step_slots(&mut kv, 4);
        assert_eq!(eng.prealloc.len(), 4);
        assert_eq!(kv.outstanding(), 4, "reserved slots are outstanding");
        eng.release_prealloc(&mut kv);
        assert!(eng.prealloc.is_empty());
        assert_eq!(kv.outstanding(), 0, "release returns every reserved slot");
    }

    #[test]
    fn reserve_is_best_effort_on_tight_pool() {
        let mut kv = GlobalKvAllocator::new(4);
        let _held = kv.alloc_indices(3).unwrap(); // total_free = 1
        let mut eng = DecodeEngine::new();
        // Cannot fit 2 → reserve nothing rather than partially; the on-path
        // relief alloc handles the step instead.
        eng.reserve_next_step_slots(&mut kv, 2);
        assert!(eng.prealloc.is_empty());
        assert_eq!(kv.outstanding(), 3, "no speculative slots taken under pressure");
    }

    #[test]
    fn surplus_split_returns_only_unused_slots() {
        // Mirrors the prepare_step fast path: reserved 4, this step needs 2 →
        // take 2, free the 2-slot surplus; the kept slots stay outstanding until
        // committed into a block table.
        let mut kv = GlobalKvAllocator::new(16);
        let mut eng = DecodeEngine::new();
        eng.reserve_next_step_slots(&mut kv, 4);
        let needed = 2;
        let mut slots = std::mem::take(&mut eng.prealloc);
        let surplus = slots.split_off(needed);
        kv.free(&surplus);
        assert_eq!(slots.len(), 2, "kept exactly the needed slots");
        assert_eq!(kv.outstanding(), 2, "only the 2 kept slots remain outstanding");
        kv.free(&slots); // emulate later commit/free of the kept slots
        assert_eq!(kv.outstanding(), 0);
    }

    #[test]
    fn reclaim_pending_frees_prealloc() {
        let mut kv = GlobalKvAllocator::new(16);
        let mut eng = DecodeEngine::new();
        eng.reserve_next_step_slots(&mut kv, 5);
        assert_eq!(kv.outstanding(), 5);
        eng.reclaim_pending(&mut kv); // drain path: no pending step, just prealloc
        assert!(eng.prealloc.is_empty());
        assert_eq!(kv.outstanding(), 0, "drain reclaims speculative slots");
    }
}
