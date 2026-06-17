use half::bf16;

use infer_protocol::scheduler_to_worker_data::{PrefillBatchCmd, PrefillSegmentCompletion};
use infer_protocol::worker_to_scheduler_control::{WorkerControlMessage, WorkerStepError};
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::forward_workspace::ForwardWorkspace;
use crate::application::kv_relief::{AllocWithReliefOutcome, alloc_with_relief};
use crate::application::model_runner::{DecodeCompactOutput, ModelRunner, SeqStep};
use crate::application::worker_state::{
    ActiveSeq, ActiveSeqMap, DecodeRows, PrefillSeq, PrefillSeqMap,
};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::LlmModel;
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;

/// P2: Context struct bundling the mutable state that `handle_prefill` passes
/// around, replacing 10 separate parameters with a single reference.
pub struct PrefillCtx<'a, M>
where
    M: LlmModel<bf16, Cuda, ForwardWorkspace<bf16, Cuda>>,
{
    pub runner: &'a mut ModelRunner<bf16, Cuda, M>,
    pub active: &'a mut ActiveSeqMap,
    pub prefilling: &'a mut PrefillSeqMap,
    pub kv_allocator: &'a mut GlobalKvAllocator,
    pub control: &'a ControlPump,
    pub data: &'a DataPump,
    pub eos_ids: &'a [i32],
    pub enable_prefix_caching: bool,
    pub cap_batch: usize,
}

/// Per-segment prefill plan. Replaces the parallel `per_seg_*` arrays that
/// were index-correlated across three loops: each segment now carries its own
/// data in one place, indexed 1:1 with `cmd.segments`. Skipped segments keep
/// `indices`/`token_ids` empty so downstream loops produce no work for them.
struct SegmentPlan {
    /// New (non-prefix) tokens this segment writes.
    new_tokens: u32,
    /// Prefix-hint slots prepended verbatim (already hold valid KV).
    prefix: Vec<u32>,
    /// Block table covering everything before this segment's new tokens.
    base_table: Vec<u32>,
    /// True when the segment is stale (base table length mismatch) and skipped.
    skipped: bool,
    /// KV slots allocated for the new tokens (filled during step building).
    indices: Vec<u32>,
    /// Input token ids fed for the new tokens (filled during step building).
    token_ids: Vec<i32>,
    /// P0: Full block table (base ++ indices), built once during step building
    /// and reused in the post-forward commit phase, eliminating the duplicate
    /// `concat_block_table` call.
    full_block_table: Vec<u32>,
}

/// Concatenate a base block table with freshly allocated slots. Centralizes
/// the `base ++ appended` construction shared by step building and the
/// post-forward `prefilling`/`active` insert.
fn concat_block_table(base: &[u32], appended: &[u32]) -> Vec<u32> {
    let mut bt = Vec::with_capacity(base.len() + appended.len());
    bt.extend_from_slice(base);
    bt.extend_from_slice(appended);
    bt
}

/// Run one prefill batch (one PrefillBatchCmd = potentially many segments).
///
/// Each segment's KV slots come from the worker-owned
/// `GlobalKvAllocator`, not from `seg.block_table`. Optional
/// `seg.prefix_hint` is prepended verbatim; those slots already hold valid
/// KV from a previous request and are pinned by scheduler-side RadixTree
/// policy while this step runs.
pub fn handle_prefill<M>(
    ctx: &mut PrefillCtx<'_, M>,
    cmd: &PrefillBatchCmd,
) -> OpResult<()>
where
    M: LlmModel<bf16, Cuda, ForwardWorkspace<bf16, Cuda>>,
{
    let mut plans: Vec<SegmentPlan> = Vec::with_capacity(cmd.segments.len());
    let mut total_new: u32 = 0;
    for seg in &cmd.segments {
        let seg_len = seg.segment_end - seg.segment_start;
        let prefix = seg.prefix_hint.clone().unwrap_or_default();
        let prefix_hit = prefix.len() as u32;
        let new_tokens = if seg.segment_start == 0 {
            seg_len.saturating_sub(prefix_hit)
        } else {
            seg_len
        };
        let base_table = ctx.prefilling
            .get(&seg.sequence_id)
            .map(|state| state.block_table.clone())
            .unwrap_or_else(|| prefix.clone());
        let expected_base_len = if seg.segment_start == 0 && !prefix.is_empty() {
            prefix.len()
        } else {
            seg.segment_start as usize
        };
        let skipped = base_table.len() != expected_base_len;
        if skipped {
            tracing::warn!(
                seq = seg.sequence_id,
                base_len = base_table.len(),
                expected = expected_base_len,
                "skipping stale prefill segment"
            );
        } else {
            total_new = total_new.saturating_add(new_tokens);
        }
        plans.push(SegmentPlan {
            new_tokens,
            prefix,
            base_table,
            skipped,
            indices: Vec::new(),
            token_ids: Vec::new(),
            full_block_table: Vec::new(),
        });
    }

    let new_decode_segments = cmd
        .segments
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            !plans[*i].skipped
                && matches!(
                    s.completion,
                    PrefillSegmentCompletion::FinishPrefillAndStartDecode
                )
        })
        .count();
    if ctx.active.len() + new_decode_segments > ctx.cap_batch {
        let overflow_ids: Vec<u64> = cmd
            .segments
            .iter()
            .enumerate()
            .filter(|(i, _)| !plans[*i].skipped)
            .map(|(_, s)| s.sequence_id)
            .collect();
        tracing::warn!(
            active = ctx.active.len(),
            new = new_decode_segments,
            cap = ctx.cap_batch,
            failing = overflow_ids.len(),
            "prefill rejected: active + new > cap"
        );
        send_step_error(
            ctx.control,
            overflow_ids,
            format!(
                "worker batch slot exhausted: active={} + new={} > cap={}",
                ctx.active.len(),
                new_decode_segments,
                ctx.cap_batch,
            ),
        );
        return Ok(());
    }

    let base_indices = match alloc_with_relief(
        ctx.kv_allocator,
        ctx.control,
        ctx.active,
        ctx.prefilling,
        total_new,
        ctx.enable_prefix_caching,
        false,
    ) {
        AllocWithReliefOutcome::Allocated(v) => v,
        AllocWithReliefOutcome::Unavailable => {
            tracing::warn!(
                n = total_new,
                "prefill alloc still failing after relief -- failing batch"
            );
            let failed_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
            send_step_error(
                ctx.control,
                failed_ids,
                format!("worker KV pool exhausted (n={})", total_new),
            );
            return Ok(());
        }
        AllocWithReliefOutcome::Shutdown => return Err(OpError::Shutdown),
    };

    let mut steps: Vec<SeqStep> = Vec::with_capacity(cmd.segments.len());
    let mut idx_cursor = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        if plans[i].skipped {
            continue;
        }
        let new_tokens = plans[i].new_tokens as usize;
        let range = cmd.segment_token_range(i);
        // P0: Avoid intermediate Vec allocations for trimmed/positions.
        // Use direct slice operations instead of iterator collect.
        let (input_ids, positions): (Vec<i32>, Vec<i32>) = if seg.segment_start == 0
            && !plans[i].prefix.is_empty()
        {
            let prefix_len = plans[i].prefix.len();
            let prompt = &cmd.input_ids[range.clone()];
            let trimmed = prompt[prefix_len..].to_vec();
            let start = prefix_len as i32;
            let positions: Vec<i32> = (start..start + trimmed.len() as i32).collect();
            (trimmed, positions)
        } else {
            let prompt = cmd.input_ids[range.clone()].to_vec();
            let positions: Vec<i32> = (seg.segment_start as i32..seg.segment_end as i32).collect();
            (prompt, positions)
        };

        let new_indices: Vec<u32> = base_indices[idx_cursor..idx_cursor + new_tokens].to_vec();
        idx_cursor += new_tokens;

        let block_table = concat_block_table(&plans[i].base_table, &new_indices);
        let kv_write_start = block_table.len() as i32 - new_tokens as i32;
        let kv_len_after = block_table.len() as i32;
        steps.push(SeqStep {
            input_ids: input_ids.clone(),
            positions,
            kv_write_start,
            kv_len_after,
            block_table: block_table.clone(),
        });
        plans[i].indices = new_indices;
        plans[i].token_ids = input_ids;
        // P0: Store full block table once, reuse in commit phase.
        plans[i].full_block_table = block_table;
    }
    debug_assert_eq!(idx_cursor, total_new as usize);

    if steps.is_empty() {
        if !base_indices.is_empty() {
            ctx.kv_allocator.free(&base_indices);
        }
        return Ok(());
    }

    let first_tokens = match ctx.runner.step_batch(&steps) {
        Ok(tokens) => tokens,
        Err(e) => {
            if !base_indices.is_empty() {
                ctx.kv_allocator.free(&base_indices);
            }
            for seg in &cmd.segments {
                if let Some(removed) = ctx.prefilling.remove(&seg.sequence_id) {
                    ctx.kv_allocator.release_owned(&removed.block_table, ctx.enable_prefix_caching);
                }
            }
            let failed_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
            send_step_error(ctx.control, failed_ids, format!("prefill step failed: {:?}", e));
            return Ok(());
        }
    };
    let assigned = assigned_runs(cmd, &plans, ctx.enable_prefix_caching);

    let mut output = StepOutput {
        prefill_done: Vec::new(),
        tokens: Vec::new(),
        assigned_indices: assigned,
    };
    let mut token_cursor = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        if plans[i].skipped {
            continue;
        }
        let token = first_tokens[token_cursor];
        token_cursor += 1;
        // P0: Reuse the full block table built during step construction
        // instead of rebuilding it with a second concat_block_table call.
        let full_block_table = std::mem::take(&mut plans[i].full_block_table);
        match seg.completion {
            PrefillSegmentCompletion::ContinuePrefill => {
                ctx.prefilling.insert(
                    seg.sequence_id,
                    PrefillSeq {
                        kv_len: full_block_table.len(),
                        block_table: full_block_table,
                    },
                );
                output.prefill_done.push(seg.sequence_id);
            }
            PrefillSegmentCompletion::FinishPrefillAndStartDecode => {
                ctx.prefilling.remove(&seg.sequence_id);
                output.prefill_done.push(seg.sequence_id);
                let finished = (!seg.ignore_eos && ctx.eos_ids.contains(&token)) || seg.max_tokens <= 1;
                output.tokens.push(GeneratedToken {
                    sequence_id: seg.sequence_id,
                    token_id: token,
                    finished,
                });
                if !finished {
                    ctx.active.insert(
                        seg.sequence_id,
                        ActiveSeq {
                            last_token: token,
                            kv_len: full_block_table.len(),
                            block_table: full_block_table,
                            max_tokens: seg.max_tokens,
                            generated_count: 1,
                            ignore_eos: seg.ignore_eos,
                        },
                    );
                } else {
                    ctx.kv_allocator.release_owned(&full_block_table, ctx.enable_prefix_caching);
                }
            }
        }
    }
    ctx.data.send_step_output(&output)
        .map_err(|e| OpError::Kernel(format!("data plane send_step_output failed: {}", e)))?;
    Ok(())
}
pub fn run_decode_step<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    decode_rows: &mut DecodeRows,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    data: &DataPump,
    eos_ids: &[i32],
    enable_prefix_caching: bool,
) -> OpResult<()>
where
    M: LlmModel<bf16, Cuda, ForwardWorkspace<bf16, Cuda>>,
{
    if active.is_empty() {
        decode_rows.clear();
        return Ok(());
    }

    let (order, new_indices) = match prepare_decode_step(
        runner,
        active,
        prefilling,
        decode_rows,
        kv_allocator,
        control,
        enable_prefix_caching,
    )? {
        DecodePrep::Ready { order, new_indices } => (order, new_indices),
        DecodePrep::Done => return Ok(()),
    };

    let inputs = build_decode_inputs(&order, &new_indices, active, enable_prefix_caching);

    let compact = match runner.step_decode_abc_compact(
        &inputs.steps,
        &inputs.generated_counts,
        &inputs.max_tokens,
        &inputs.ignore_eos,
        eos_ids,
    ) {
        Ok(output) => output,
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
            decode_rows.clear();
            return Ok(());
        }
    };

    let output = commit_decode_results(
        active,
        decode_rows,
        kv_allocator,
        &order,
        &new_indices,
        inputs.assigned,
        &compact,
        enable_prefix_caching,
    );

    data.send_step_output(&output)
        .map_err(|e| OpError::Kernel(format!("data plane send_step_output failed: {}", e)))?;
    Ok(())
}

/// Outcome of [`prepare_decode_step`]: either a ready-to-forward batch or a
/// signal that the caller should return `Ok(())` (nothing to run, or the step
/// already failed and reported its sequences).
enum DecodePrep {
    Ready { order: Vec<u64>, new_indices: Vec<u32> },
    Done,
}

/// Per-row inputs for one decode forward. The three trailing vectors mirror
/// `steps` row-for-row and are passed as separate slices because
/// `step_decode_abc_compact` consumes them that way.
struct DecodeInputs {
    steps: Vec<SeqStep>,
    assigned: Vec<AssignedIndices>,
    generated_counts: Vec<usize>,
    max_tokens: Vec<usize>,
    ignore_eos: Vec<bool>,
}

/// Report a non-fatal decode failure for `sids`, evict them from `active`, and
/// release their KV. The shared rollback for every decode alloc/forward
/// failure path (previously copy-pasted four times inline).
fn fail_decode_seqs(
    control: &ControlPump,
    active: &mut ActiveSeqMap,
    kv_allocator: &mut GlobalKvAllocator,
    sids: &[u64],
    message: String,
    enable_prefix_caching: bool,
) {
    send_step_error(control, sids.to_vec(), message);
    for sid in sids {
        if let Some(removed) = active.remove(sid) {
            kv_allocator.release_owned(&removed.block_table, enable_prefix_caching);
        }
    }
}

/// Admit pending rows into buffer A, allocate one KV slot per active row, and
/// reconcile the row order against late cancellations. Returns
/// [`DecodePrep::Ready`] with the surviving row order and its freshly
/// allocated slots, or [`DecodePrep::Done`] when there is nothing left to run.
/// Returns `Err(OpError::Shutdown)` only on a shutdown during relief.
fn prepare_decode_step<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    decode_rows: &mut DecodeRows,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    enable_prefix_caching: bool,
) -> OpResult<DecodePrep>
where
    M: LlmModel<bf16, Cuda, ForwardWorkspace<bf16, Cuda>>,
{
    decode_rows.retain_active(active);
    let pending = decode_rows.pending_admissions(active);
    if !pending.is_empty() {
        let mut admit_tokens = Vec::with_capacity(pending.len());
        for sid in &pending {
            if let Some(seq) = active.get(sid) {
                admit_tokens.push(seq.last_token);
            }
        }
        if let Err(e) = runner.append_decode_admissions_to_a(decode_rows.len(), &admit_tokens) {
            let failed: Vec<u64> = active.keys().copied().collect();
            fail_decode_seqs(
                control,
                active,
                kv_allocator,
                &failed,
                format!("decode admission append failed: {:?}", e),
                enable_prefix_caching,
            );
            decode_rows.clear();
            return Ok(DecodePrep::Done);
        }
        decode_rows.append_admissions(&pending);
    }

    let mut order: Vec<u64> = decode_rows.as_slice().to_vec();
    if order.is_empty() {
        return Ok(DecodePrep::Done);
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
            let order: Vec<u64> = decode_rows.as_slice().to_vec();
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
            decode_rows.clear();
            return Ok(DecodePrep::Done);
        }
        AllocWithReliefOutcome::Shutdown => return Err(OpError::Shutdown),
    };

    // Relief may have evicted rows (preempt/cancel) while we waited; re-sync
    // the order against the surviving active set before binding slots to rows.
    decode_rows.retain_active(active);
    order = decode_rows.as_slice().to_vec();
    if order.is_empty() {
        if !new_indices.is_empty() {
            kv_allocator.free(&new_indices);
        }
        return Ok(DecodePrep::Done);
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
        decode_rows.retain_active(active);
        order.truncate(new_indices.len());
        if order.is_empty() {
            if !new_indices.is_empty() {
                kv_allocator.free(&new_indices);
            }
            return Ok(DecodePrep::Done);
        }
    }

    Ok(DecodePrep::Ready { order, new_indices })
}

/// Build the per-row decode forward inputs: one appended KV slot per row, the
/// single-token step, and the metadata slices the sampler consumes.
fn build_decode_inputs(
    order: &[u64],
    new_indices: &[u32],
    active: &ActiveSeqMap,
    enable_prefix_caching: bool,
) -> DecodeInputs {
    let mut inputs = DecodeInputs {
        steps: Vec::with_capacity(order.len()),
        assigned: Vec::with_capacity(order.len()),
        generated_counts: Vec::with_capacity(order.len()),
        max_tokens: Vec::with_capacity(order.len()),
        ignore_eos: Vec::with_capacity(order.len()),
    };
    for (i, &sid) in order.iter().enumerate() {
        let new_idx = new_indices[i];
        let seq = active.get(&sid).unwrap();
        // C1: build the step's block table in one allocation. The old
        // `clone()` then `push()` allocated twice (clone sized to len, push
        // reallocated to len+1); reserving len+1 up front does it once.
        let mut bt = Vec::with_capacity(seq.block_table.len() + 1);
        bt.extend_from_slice(&seq.block_table);
        bt.push(new_idx);
        inputs.steps.push(SeqStep {
            input_ids: vec![seq.last_token],
            positions: vec![seq.kv_len as i32],
            kv_write_start: seq.kv_len as i32,
            kv_len_after: (seq.kv_len + 1) as i32,
            block_table: bt,
        });
        inputs.assigned.push(AssignedIndices {
            sequence_id: sid,
            base: new_idx,
            len: 1,
            token_ids: if enable_prefix_caching {
                vec![seq.last_token]
            } else {
                Vec::new()
            },
        });
        inputs.generated_counts.push(seq.generated_count);
        inputs.max_tokens.push(seq.max_tokens);
        inputs.ignore_eos.push(seq.ignore_eos);
    }
    inputs
}

/// Apply the compacted forward output: advance surviving sequences one token,
/// emit their tokens, evict finished ones (releasing KV), and rewrite the row
/// order to the compacted survivors. Returns the `StepOutput` to ship.
fn commit_decode_results(
    active: &mut ActiveSeqMap,
    decode_rows: &mut DecodeRows,
    kv_allocator: &mut GlobalKvAllocator,
    order: &[u64],
    new_indices: &[u32],
    assigned: Vec<AssignedIndices>,
    compact: &DecodeCompactOutput,
    enable_prefix_caching: bool,
) -> StepOutput {
    let mut output = StepOutput {
        prefill_done: Vec::new(),
        tokens: Vec::new(),
        assigned_indices: assigned,
    };

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
        if let Some(seq) = active.get_mut(&sid) {
            seq.last_token = token;
            seq.kv_len += 1;
            seq.generated_count += 1;
            seq.block_table.push(new_indices[i]);
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
    decode_rows.replace_rows(next_rows);
    output
}

/// Send a non-fatal StepError to the scheduler, logging (not silently
/// dropping) if the control channel is broken. Centralizes the boilerplate
/// previously copy-pasted across every prefill/decode failure path, and
/// makes a torn control plane observable instead of a silent hang (H8/M6).
fn send_step_error(control: &ControlPump, sequence_ids: Vec<u64>, message: String) {
    if let Err(e) = control.send(
        WorkerControlMessage::StepError(WorkerStepError {
            sequence_ids,
            message,
            fatal: false,
        }),
        infer_protocol::control_envelope::RequestId::NONE,
    ) {
        tracing::error!(error = %e, "failed to send StepError to scheduler (control plane may be down)");
    }
}

fn assigned_runs(
    cmd: &PrefillBatchCmd,
    plans: &[SegmentPlan],
    include_token_ids: bool,
) -> Vec<AssignedIndices> {
    let mut assigned = Vec::new();
    for (i, seg) in cmd.segments.iter().enumerate() {
        let indices = &plans[i].indices;
        let token_ids = &plans[i].token_ids;
        debug_assert_eq!(indices.len(), token_ids.len());
        let mut run_start = 0usize;
        while run_start < indices.len() {
            let mut run_end = run_start + 1;
            while run_end < indices.len() && indices[run_end] == indices[run_end - 1] + 1 {
                run_end += 1;
            }
            let mut chunk_start = run_start;
            while chunk_start < run_end {
                let chunk_len = (run_end - chunk_start).min(u16::MAX as usize);
                assigned.push(AssignedIndices {
                    sequence_id: seg.sequence_id,
                    base: indices[chunk_start],
                    len: chunk_len as u16,
                    token_ids: if include_token_ids {
                        token_ids[chunk_start..chunk_start + chunk_len].to_vec()
                    } else {
                        Vec::new()
                    },
                });
                chunk_start += chunk_len;
            }
            run_start = run_end;
        }
    }
    assigned
}
