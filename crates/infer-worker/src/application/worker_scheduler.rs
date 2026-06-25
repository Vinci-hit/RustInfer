use half::bf16;

use infer_protocol::scheduler_to_worker_data::{PrefillBatchCmd, PrefillSegmentCompletion};
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::decode_common::send_step_error;
use crate::application::kv_relief::{AllocWithReliefOutcome, alloc_with_relief};
use crate::application::runtime::Runtime;
use crate::application::worker_state::{ActiveSeq, ActiveSeqMap, PrefillSeq, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::plan::{SeqStep, StepRequest, StopCriteria};
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;

/// P2: Context struct bundling the mutable state that `handle_prefill` passes
/// around, replacing 10 separate parameters with a single reference.
pub struct PrefillCtx<'a, M>
where
    M: DecoderModel<bf16, Cuda>,
{
    pub runner: &'a mut Runtime<bf16, Cuda, M>,
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
    /// Input token ids fed for the new tokens. Moved out of the forwarded
    /// `SeqStep` after the forward (#6 zero-copy) — see `step_idx`.
    token_ids: Vec<i32>,
    /// P0: Full block table (base ++ indices), built once during step building
    /// and reused in the post-forward commit phase, eliminating the duplicate
    /// `concat_block_table` call. Moved out of the forwarded `SeqStep` after the
    /// forward (#6 zero-copy) rather than cloned up front.
    full_block_table: Vec<u32>,
    /// Index of this segment's entry in `steps`, or `None` if skipped. The
    /// forward only borrows `steps`, so afterwards `token_ids`/`full_block_table`
    /// are reclaimed by move from `steps[step_idx]` instead of being cloned.
    step_idx: Option<usize>,
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
pub fn handle_prefill<M>(ctx: &mut PrefillCtx<'_, M>, cmd: &PrefillBatchCmd) -> OpResult<()>
where
    M: DecoderModel<bf16, Cuda>,
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
        let base_table = ctx
            .prefilling
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
            step_idx: None,
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
    let mut step_max_tokens: Vec<u32> = Vec::with_capacity(cmd.segments.len());
    let mut step_ignore_eos: Vec<bool> = Vec::with_capacity(cmd.segments.len());
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
        // #6 zero-copy: hand `input_ids`/`block_table` to the step by move. The
        // forward only borrows them; they are reclaimed below (post-forward)
        // into `token_ids`/`full_block_table` rather than cloned here.
        plans[i].step_idx = Some(steps.len());
        steps.push(SeqStep {
            sequence_id: seg.sequence_id,
            input_ids,
            positions,
            kv_write_start,
            kv_len_after,
            block_table,
        });
        step_max_tokens.push(seg.max_tokens as u32);
        step_ignore_eos.push(seg.ignore_eos);
        plans[i].indices = new_indices;
    }
    debug_assert_eq!(idx_cursor, total_new as usize);

    if steps.is_empty() {
        if !base_indices.is_empty() {
            ctx.kv_allocator.free(&base_indices);
        }
        return Ok(());
    }

    // One ragged forward over every non-skipped segment. Greedy sampling
    // (empty `sampling` → GreedySampler) keeps the first-token result
    // byte-identical to the old argmax path. The pool commits each seq's KV
    // length from `kv_len_after` internally.
    let req = StepRequest {
        sampling: Vec::new(),
        stop: StopCriteria {
            eos_ids: ctx.eos_ids.to_vec(),
            generated_counts: vec![0u32; steps.len()],
            max_tokens: step_max_tokens,
            ignore_eos: step_ignore_eos,
        },
        draft_tokens: Vec::new(),
        seqs: steps,
    };
    let _step_t0 = if std::env::var_os("RUSTINFER_TTFT_TRACE").is_some() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let out = match ctx.runner.step(&req) {
        Ok(out) => out,
        Err(e) => {
            if !base_indices.is_empty() {
                ctx.kv_allocator.free(&base_indices);
            }
            for seg in &cmd.segments {
                if let Some(removed) = ctx.prefilling.remove(&seg.sequence_id) {
                    ctx.kv_allocator
                        .release_owned(&removed.block_table, ctx.enable_prefix_caching);
                }
            }
            let failed_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
            send_step_error(
                ctx.control,
                failed_ids,
                format!("prefill step failed: {:?}", e),
            );
            return Ok(());
        }
    };
    if let Some(t0) = _step_t0 {
        tracing::info!(
            "[ttft-trace] runner.step (forward+sample) = {:.2}ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    // #6 zero-copy: the forward only borrowed the request, so move the input
    // tokens and block tables back out of the owned `seqs` into their plans
    // (reclaimed by `assigned_runs` and the commit loop below) rather than
    // having cloned them during step building.
    let StepRequest { seqs: mut steps, .. } = req;
    for plan in plans.iter_mut() {
        if let Some(k) = plan.step_idx {
            plan.token_ids = std::mem::take(&mut steps[k].input_ids);
            plan.full_block_table = std::mem::take(&mut steps[k].block_table);
        }
    }

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
        let row = out.tokens.get(token_cursor);
        let token = row.and_then(|r| r.first()).map(|t| t.token_id).unwrap_or(0);
        let finished = out.finished.get(token_cursor).copied().unwrap_or(false);
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
                output.tokens.push(GeneratedToken {
                    sequence_id: seg.sequence_id,
                    token_id: token,
                    finished,
                });
                if !finished {
                    ctx.active.insert(
                        seg.sequence_id,
                        ActiveSeq::new(token, full_block_table, seg.max_tokens, seg.ignore_eos),
                    );
                } else {
                    ctx.kv_allocator
                        .release_owned(&full_block_table, ctx.enable_prefix_caching);
                }
            }
        }
    }
    ctx.data
        .send_step_output(&output)
        .map_err(|e| OpError::Kernel(format!("data plane send_step_output failed: {}", e)))?;
    Ok(())
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
