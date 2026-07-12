use half::bf16;

use infer_protocol::scheduler_to_worker_data::{
    PrefillBatchCmd, PrefillSegmentCompletion, SamplingParams as WireSamplingParams,
};
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::decode_common::send_step_error;
use crate::application::decode_engine::{
    DecodeEngine, DecodeRequestBuild, build_decode_request, build_overlap_decode_request,
};
use crate::application::kv_relief::{AllocWithReliefOutcome, alloc_with_relief};
use crate::application::runtime::{RaggedRowKind, Runtime};
use crate::application::worker_state::{ActiveSeq, ActiveSeqMap, PrefillSeq, PrefillSeqMap};
use crate::domain::global_kv_alloc::{GlobalKvAllocator, KvLease};
use crate::domain::model::DecoderModel;
use crate::domain::plan::{SampledToken, SeqStep, StepRequest, StopCriteria};
use crate::domain::ports::sampler::SamplingParams;
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;

fn sampling_params(params: &WireSamplingParams) -> SamplingParams {
    SamplingParams {
        temperature: params.temperature,
        top_k: u32::try_from(params.top_k).unwrap_or(0),
        top_p: params.top_p,
        ..SamplingParams::default()
    }
}

fn cmd_requires_stochastic_sampling(cmd: &PrefillBatchCmd) -> bool {
    cmd.segments
        .iter()
        .filter(|segment| {
            matches!(
                segment.completion,
                PrefillSegmentCompletion::FinishPrefillAndStartDecode
            )
        })
        .any(|segment| !sampling_params(&segment.sampling_params).is_greedy())
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

/// Plan every segment of one prefill cmd: compute the new (non-prefix) token
/// count, resolve each segment's base block table, and mark stale segments
/// skipped. Returns the per-segment plans (1:1 with `cmd.segments`) and the
/// total new tokens to allocate KV for.
fn plan_prefill_segments(
    cmd: &PrefillBatchCmd,
    prefilling: &PrefillSeqMap,
) -> (Vec<SegmentPlan>, u32) {
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
        let base_table = prefilling
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
    (plans, total_new)
}

/// Count a cmd's non-skipped segments that finish prefill this step (each
/// becomes a new decode row).
fn count_new_decode(cmd: &PrefillBatchCmd, plans: &[SegmentPlan]) -> usize {
    cmd.segments
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            !plans[*i].skipped
                && matches!(
                    s.completion,
                    PrefillSegmentCompletion::FinishPrefillAndStartDecode
                )
        })
        .count()
}

/// Number of forward rows a cmd contributes (its non-skipped segments).
fn prefill_row_count(plans: &[SegmentPlan]) -> usize {
    plans.iter().filter(|p| !p.skipped).count()
}

/// Build `SeqStep`s for one cmd's non-skipped segments, appending them (and
/// their per-row stop vectors) onto the shared accumulators that form the fused
/// forward. `base_indices` are this cmd's freshly-allocated KV slots
/// (`len == total_new`). Each plan's `step_idx` records its row index in the
/// merged `seqs` so `reclaim_prefill_step_data` can move its buffers back.
// The output vectors are parallel row accumulators whose shared indices are a
// deliberate invariant; grouping them would not reduce the launch state.
#[allow(clippy::too_many_arguments)]
fn build_prefill_steps_into(
    cmd: &PrefillBatchCmd,
    plans: &mut [SegmentPlan],
    base_indices: &[u32],
    seqs: &mut Vec<SeqStep>,
    sampling: &mut Vec<SamplingParams>,
    max_tokens: &mut Vec<u32>,
    ignore_eos: &mut Vec<bool>,
    row_kinds: &mut Vec<RaggedRowKind>,
) {
    let mut idx_cursor = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        if plans[i].skipped {
            continue;
        }
        let new_tokens = plans[i].new_tokens as usize;
        let range = cmd.segment_token_range(i);
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
        plans[i].step_idx = Some(seqs.len());
        seqs.push(SeqStep {
            sequence_id: seg.sequence_id,
            input_ids,
            positions,
            kv_write_start,
            kv_len_after,
            block_table,
        });
        max_tokens.push(seg.max_tokens as u32);
        sampling.push(
            if matches!(
                seg.completion,
                PrefillSegmentCompletion::FinishPrefillAndStartDecode
            ) {
                sampling_params(&seg.sampling_params)
            } else {
                SamplingParams::default()
            },
        );
        ignore_eos.push(seg.ignore_eos);
        row_kinds.push(match seg.completion {
            PrefillSegmentCompletion::ContinuePrefill => RaggedRowKind::PrefillCont,
            PrefillSegmentCompletion::FinishPrefillAndStartDecode => RaggedRowKind::PrefillFinal,
        });
        plans[i].indices = new_indices;
    }
    debug_assert_eq!(idx_cursor, base_indices.len());
}

fn append_prefill_abc_next_rows(
    cmd: &PrefillBatchCmd,
    plans: &[SegmentPlan],
    out_finished: &[bool],
    token_offset: usize,
    next_rows: &mut Vec<u64>,
) {
    let mut local = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        if plans[i].skipped {
            continue;
        }
        let row = token_offset + local;
        local += 1;
        if matches!(
            seg.completion,
            PrefillSegmentCompletion::FinishPrefillAndStartDecode
        ) && !out_finished.get(row).copied().unwrap_or(false)
        {
            next_rows.push(seg.sequence_id);
        }
    }
}

/// #6 zero-copy: the forward only borrowed the request, so move the input
/// tokens and block tables back out of the (now-owned) merged `seqs` into each
/// plan, where `assigned_runs` + `commit_prefill_outputs` reclaim them.
fn reclaim_prefill_step_data(seqs: &mut [SeqStep], plans: &mut [SegmentPlan]) {
    for plan in plans.iter_mut() {
        if let Some(k) = plan.step_idx {
            plan.token_ids = std::mem::take(&mut seqs[k].input_ids);
            plan.full_block_table = std::mem::take(&mut seqs[k].block_table);
        }
    }
}

/// Route one cmd's prefill outputs into the shared `output`. This cmd's rows
/// occupy `out_tokens[token_offset .. token_offset + non_skipped]`.
/// Continuation chunks store their KV in `prefilling` and emit no token; final
/// chunks emit the first decode token and admit the sequence into `active`.
#[allow(clippy::too_many_arguments)]
fn commit_prefill_outputs(
    cmd: &PrefillBatchCmd,
    plans: &mut [SegmentPlan],
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    kv_allocator: &mut GlobalKvAllocator,
    enable_prefix_caching: bool,
    out_tokens: &[Vec<SampledToken>],
    out_finished: &[bool],
    token_offset: usize,
    output: &mut StepOutput,
) {
    let mut local = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        if plans[i].skipped {
            continue;
        }
        let row = token_offset + local;
        local += 1;
        let token = out_tokens
            .get(row)
            .and_then(|r| r.first())
            .map(|t| t.token_id)
            .unwrap_or(0);
        let finished = out_finished.get(row).copied().unwrap_or(false);
        let full_block_table = std::mem::take(&mut plans[i].full_block_table);
        match seg.completion {
            PrefillSegmentCompletion::ContinuePrefill => {
                prefilling.insert(
                    seg.sequence_id,
                    PrefillSeq {
                        kv_len: full_block_table.len(),
                        block_table: full_block_table,
                    },
                );
                output.prefill_done.push(seg.sequence_id);
            }
            PrefillSegmentCompletion::FinishPrefillAndStartDecode => {
                prefilling.remove(&seg.sequence_id);
                output.prefill_done.push(seg.sequence_id);
                output.tokens.push(GeneratedToken {
                    sequence_id: seg.sequence_id,
                    token_id: token,
                    finished,
                });
                if !finished {
                    active.insert(
                        seg.sequence_id,
                        ActiveSeq::new(
                            token,
                            full_block_table,
                            seg.max_tokens,
                            seg.ignore_eos,
                            sampling_params(&seg.sampling_params),
                        ),
                    );
                } else {
                    kv_allocator.release_owned(&full_block_table, enable_prefix_caching);
                }
            }
        }
    }
}

/// Planned + KV-allocated prefill cmd, awaiting its forward.
struct CmdPrep<'a> {
    cmd: &'a PrefillBatchCmd,
    plans: Vec<SegmentPlan>,
    /// Slots for this cmd's new tokens. Committed after the group's outputs
    /// are routed (ownership moves into `prefilling`/`active` block tables),
    /// or released on the group's failure path / the post-loop sweep.
    base_indices: KvLease,
    /// Non-skipped segment count = rows this cmd adds to the forward.
    rows: usize,
    /// New tokens this cmd adds to the forward.
    tokens: usize,
}

/// A single fused forward: the decode rows (group 0 only) plus a set of prefill
/// cmds, packed so `rows <= cap_batch` and `tokens <= cap_num_tokens`.
struct ForwardGroup {
    /// True for the first group, which carries the decode rows.
    decode: bool,
    /// Indices into the `preps` vec.
    preps: Vec<usize>,
    rows: usize,
    tokens: usize,
}

/// Run ONE fused step: finalize the in-flight decode, then drive all active
/// decode rows + admitted pending prefill chunks through a single ragged
/// forward.
///
/// Replaces the old "prefill-first, then decode" serialization. Prefill chunks
/// and decode rows share one ragged forward, so the fixed per-forward host
/// overhead is paid ONCE per step instead of once per prefill, and in-flight
/// decode rows are never stalled behind prefills — they advance a token in the
/// same forward.
///
/// Pipelining: the prior decode step's tokens are sent AFTER the fused forward
/// is issued (the send overlaps GPU compute), and on exit the next pure-decode
/// step is issued and left pending, so the fused step's host tail overlaps the
/// next step's compute instead of idling the GPU.
///
/// Admission is bounded to the largest prewarmed mixed-graph token bucket;
/// surplus cmds are pushed to `deferred_out` for the next serve-loop iteration
/// (decode advances every iteration, so a burst is spread across consecutive
/// graphed steps instead of one huge eager step). Cmds that still overflow the
/// batch/token cap spill into extra pure-prefill forwards; the decode rows
/// always ride the first forward.
#[allow(clippy::too_many_arguments)]
pub fn handle_fused_step<M>(
    runner: &mut Runtime<bf16, Cuda, M>,
    decode_engine: &mut DecodeEngine,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    data: &DataPump,
    eos_ids: &[i32],
    enable_prefix_caching: bool,
    cap_batch: usize,
    cap_num_tokens: usize,
    pending_prefills: Vec<PrefillBatchCmd>,
    deferred_out: &mut Vec<PrefillBatchCmd>,
) -> OpResult<()>
where
    M: DecoderModel<bf16, Cuda>,
{
    // 1. Drain OR overlap the in-flight ABC decode step. Overlap (eager-mixed
    //    mode): leave it in flight, build this step's decode rows
    //    OPTIMISTICALLY from its row set (every row assumed to survive; input
    //    tokens gathered from buffer C on device), and finalize it only AFTER
    //    the fused forward is issued — its GPU tail then runs concurrently
    //    with this step's host planning + issue instead of stalling it (the
    //    measured ~4.6ms fused-step drain). Fall back to the drain whenever
    //    the optimistic build is blocked: no in-flight step, a row evicted
    //    mid-flight, no KV slots without relief (relief could preempt the
    //    in-flight rows), or RUSTINFER_FUSED_OVERLAP=0.
    let stochastic_requested = active.values().any(|seq| !seq.sampling.is_greedy())
        || pending_prefills
            .iter()
            .any(cmd_requires_stochastic_sampling);
    let overlap_enabled = !stochastic_requested
        && runner.mixed_eager_mode()
        && std::env::var_os("RUSTINFER_FUSED_OVERLAP").is_none_or(|v| v != "0");
    let mut overlap_prior: Option<(Vec<u64>, Vec<u32>, KvLease)> = None;
    if overlap_enabled
        && let Some((order, step_slots)) = decode_engine.overlap_fused_snapshot(active)
    {
        let lease = decode_engine
            .take_prealloc_for_overlap(order.len(), kv_allocator)
            .or_else(|| kv_allocator.lease(order.len() as u32).ok());
        if let Some(lease) = lease {
            overlap_prior = Some((order, step_slots, lease));
        }
    }
    let optimistic_decode = overlap_prior.is_some();
    let mut overlap_active = optimistic_decode;

    let mut prior_output = if overlap_active {
        None
    } else {
        decode_engine.finalize_pending(
            runner,
            active,
            kv_allocator,
            control,
            enable_prefix_caching,
        )?
    };
    // Rows of the overlapped prior step that it turns out to have FINISHED:
    // filled once the prior step is finalized (post-issue), consumed by the
    // fused commit to discard their optimistic extra forward.
    let mut skip_rows: Vec<bool> = Vec::new();

    // 2. Decode rows for this step: one new KV slot each. Overlap path: the
    //    in-flight step's rows advanced by one position; drain path: the
    //    committed active rows as before.
    let (decode_order, mut decode_lease, mut decode_build) =
        if let Some((order, step_slots, lease)) = overlap_prior {
            let build = build_overlap_decode_request(
                &order,
                &step_slots,
                lease.as_slice(),
                active,
                eos_ids,
            );
            (order, Some(lease), Some(build))
        } else {
            match decode_engine.prepare_fused_decode(
                active,
                prefilling,
                kv_allocator,
                control,
                enable_prefix_caching,
            ) {
                Ok(Some((order, lease))) => {
                    let build = build_decode_request(
                        &order,
                        lease.as_slice(),
                        active,
                        eos_ids,
                        enable_prefix_caching,
                    );
                    (order, Some(lease), Some(build))
                }
                Ok(None) => (Vec::new(), None, None),
                Err(e) => {
                    let _ = send_prior_output(data, &mut prior_output);
                    return Err(e);
                }
            }
        };
    let decode_count = decode_order.len();

    // 2.5 Bound this step's prefill admission to the largest prewarmed
    //     mixed-graph token bucket. Surplus cmds are deferred to the next
    //     serve-loop iteration: decode rows advance every iteration
    //     regardless, so a burst of arrivals is spread across consecutive
    //     graphed steps instead of one oversized eager step that stalls every
    //     decode row for the burst's whole prefill time. FCFS: once one cmd
    //     defers, every later cmd defers behind it. The first cmd is always
    //     admitted (progress guarantee even past the budget).
    let decode_slot = runner
        .next_capture_slot(decode_count)
        .unwrap_or(decode_count);
    let step_token_budget = runner
        .mixed_step_token_budget()
        .unwrap_or(cap_num_tokens)
        .min(cap_num_tokens);
    let mut admitted_cmds: Vec<PrefillBatchCmd> = Vec::with_capacity(pending_prefills.len());
    let mut budget_tokens = 0usize;
    let mut deferring = false;
    for cmd in pending_prefills {
        // Upper-bound token estimate (ignores prefix-cache hits) — fine for a
        // packing budget.
        let est = cmd.input_ids.len();
        if deferring
            || (!admitted_cmds.is_empty() && decode_slot + budget_tokens + est > step_token_budget)
        {
            deferring = true;
            deferred_out.push(cmd);
            continue;
        }
        budget_tokens += est;
        admitted_cmds.push(cmd);
    }

    // 3. Plan + allocate KV for every admitted prefill cmd. Reject cmds that
    //    would push the concurrent decode-row count past the batch cap.
    let mut admitted_decode = active.len();
    let mut preps: Vec<CmdPrep> = Vec::new();
    for cmd in &admitted_cmds {
        let (plans, total_new) = plan_prefill_segments(cmd, prefilling);
        let new_decode = count_new_decode(cmd, &plans);
        if admitted_decode + new_decode > cap_batch {
            let overflow_ids: Vec<u64> = cmd
                .segments
                .iter()
                .enumerate()
                .filter(|(i, _)| !plans[*i].skipped)
                .map(|(_, s)| s.sequence_id)
                .collect();
            tracing::warn!(
                active = admitted_decode,
                new = new_decode,
                cap = cap_batch,
                "fused prefill rejected: active + new > cap"
            );
            send_step_error(
                control,
                overflow_ids,
                format!(
                    "worker batch slot exhausted: active={} + new={} > cap={}",
                    admitted_decode, new_decode, cap_batch
                ),
            );
            continue;
        }
        let rows = prefill_row_count(&plans);
        if rows == 0 {
            continue;
        }
        let base_indices = match alloc_with_relief(
            kv_allocator,
            control,
            active,
            prefilling,
            total_new,
            enable_prefix_caching,
            false,
        ) {
            AllocWithReliefOutcome::Allocated(v) => v,
            AllocWithReliefOutcome::Unavailable => {
                tracing::warn!(n = total_new, "fused prefill alloc failing after relief");
                let failed_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
                send_step_error(
                    control,
                    failed_ids,
                    format!("worker KV pool exhausted (n={})", total_new),
                );
                continue;
            }
            AllocWithReliefOutcome::Shutdown => {
                let _ = send_prior_output(data, &mut prior_output);
                return Err(OpError::Shutdown);
            }
        };
        admitted_decode += new_decode;
        preps.push(CmdPrep {
            cmd,
            plans,
            base_indices,
            rows,
            tokens: total_new as usize,
        });
    }

    // 4. Pack cmds into forward groups. Group 0 carries the decode rows; cmds
    //    that exceed the batch/token cap spill into extra pure-prefill forwards
    //    (rare — the scheduler chunks within cap_num_tokens).
    let mut groups: Vec<ForwardGroup> = vec![ForwardGroup {
        decode: true,
        preps: Vec::new(),
        rows: decode_count,
        tokens: decode_count,
    }];
    for (i, prep) in preps.iter().enumerate() {
        let last = groups.last_mut().unwrap();
        let fits =
            last.rows + prep.rows <= cap_batch && last.tokens + prep.tokens <= cap_num_tokens;
        if fits {
            last.rows += prep.rows;
            last.tokens += prep.tokens;
            last.preps.push(i);
        } else {
            groups.push(ForwardGroup {
                decode: false,
                preps: vec![i],
                rows: prep.rows,
                tokens: prep.tokens,
            });
        }
    }

    // Overlap is single-group only: the fused issue must precede the prior
    // step's finalize, and the deferred copy-out interleave assumes exactly
    // one fused forward. On the rare spill, drain NOW — the optimistic
    // request stays valid (it matches the post-commit row state exactly);
    // only the C-gathered input prefix still requires the overlapped issue.
    if overlap_active && groups.len() > 1 {
        prior_output = decode_engine.finalize_pending(
            runner,
            active,
            kv_allocator,
            control,
            enable_prefix_caching,
        )?;
        skip_rows = overlap_skip_rows(&prior_output, &decode_order, active);
        if enable_prefix_caching && let Some(b) = decode_build.as_mut() {
            patch_overlap_assigned_tokens(&prior_output, b);
        }
        overlap_active = false;
    }

    // 5. Issue each group as one ragged forward and route its outputs. A hard
    //    error (commit/send failure) breaks out so the post-loop sweep can
    //    release the leases of any group that never ran — the pre-lease code
    //    silently leaked those slots on this path.
    let can_record_abc_rows = groups.len() == 1;
    let mut recorded_abc_rows = false;
    let mut hard_err: Option<OpError> = None;
    'groups: for group in &groups {
        let has_decode = group.decode && decode_count > 0;
        if !has_decode && group.preps.is_empty() {
            continue;
        }

        let mut seqs: Vec<SeqStep> = Vec::with_capacity(group.rows);
        let mut sampling: Vec<SamplingParams> = Vec::with_capacity(group.rows);
        let mut max_tokens: Vec<u32> = Vec::with_capacity(group.rows);
        let mut ignore_eos: Vec<bool> = Vec::with_capacity(group.rows);
        let mut generated_counts: Vec<u32> = Vec::with_capacity(group.rows);
        let mut row_kinds: Vec<RaggedRowKind> = Vec::with_capacity(group.rows);
        if has_decode {
            let b = decode_build.as_ref().unwrap();
            seqs.extend(b.req.seqs.iter().cloned());
            sampling.extend_from_slice(&b.req.sampling);
            max_tokens.extend_from_slice(&b.max_tokens);
            ignore_eos.extend_from_slice(&b.ignore_eos);
            generated_counts.extend_from_slice(&b.generated_counts);
            row_kinds.extend(std::iter::repeat_n(RaggedRowKind::Decode, decode_count));
        }
        // Pad the decode prefix up to a decode capture slot so the split
        // attention's cuDNN decode-prefix call (the leading run of q=1 rows)
        // reuses a PREWARMED cuDNN SDPA plan instead of paying a ~370ms
        // HEURISTICS_CHOICE build for every novel decode-row count (the measured
        // fused tail spike — see [[eager-fused-mixed-tail-regression]]). The pad
        // rows are inert q=1 decode steps over throwaway KV (freed after the
        // step); their forward output lands in rows [decode_count, slot) which
        // the commit below never reads. Only padded when the group carries
        // prefill (so the Ragged split path actually runs). Best-effort: if the
        // tiny KV alloc fails, skip and accept the cold build for this one step.
        let mut decode_prefix_len = if has_decode { decode_count } else { 0 };
        let mut pad_lease = KvLease::empty();
        if has_decode
            && !group.preps.is_empty()
            && let Some(slot) = runner.next_capture_slot(decode_count)
        {
            let pad = slot - decode_count;
            let prefill_rows = group.rows.saturating_sub(decode_count);
            let padded_rows_fit = slot.saturating_add(prefill_rows) <= cap_batch;
            let padded_tokens_fit = group.tokens.saturating_add(pad) <= cap_num_tokens;
            if pad > 0
                && padded_rows_fit
                && padded_tokens_fit
                && let Ok(lease) = kv_allocator.lease(pad as u32)
            {
                for &blk in lease.as_slice() {
                    seqs.push(SeqStep {
                        sequence_id: u64::MAX, // sentinel: inert pad row
                        input_ids: vec![1],
                        positions: vec![0],
                        kv_write_start: 0,
                        kv_len_after: 1,
                        block_table: vec![blk],
                    });
                    sampling.push(SamplingParams::default());
                    max_tokens.push(u32::MAX);
                    ignore_eos.push(true);
                    generated_counts.push(0);
                    row_kinds.push(RaggedRowKind::Pad);
                }
                decode_prefix_len = slot;
                pad_lease = lease;
            }
        }
        for &pi in &group.preps {
            let prep = &mut preps[pi];
            build_prefill_steps_into(
                prep.cmd,
                &mut prep.plans,
                prep.base_indices.as_slice(),
                &mut seqs,
                &mut sampling,
                &mut max_tokens,
                &mut ignore_eos,
                &mut row_kinds,
            );
        }
        // Prefill rows have generated 0 tokens so far.
        generated_counts.resize(seqs.len(), 0);
        if seqs.is_empty() {
            continue;
        }

        let req = StepRequest {
            sampling,
            stop: StopCriteria {
                eos_ids: eos_ids.to_vec(),
                generated_counts,
                max_tokens,
                ignore_eos,
            },
            draft_tokens: Vec::new(),
            seqs,
        };
        let stochastic_group = req.sampling.iter().any(|params| !params.is_greedy());

        let mut mixed_lease = KvLease::empty();
        let mut mixed_device_prepared = false;
        if can_record_abc_rows && !stochastic_group {
            let potential_next = row_kinds
                .iter()
                .filter(|&&k| matches!(k, RaggedRowKind::Decode | RaggedRowKind::PrefillFinal))
                .count();
            if potential_next > 0 {
                match kv_allocator.lease(potential_next as u32) {
                    Ok(lease) => {
                        mixed_lease = lease;
                        mixed_device_prepared = true;
                    }
                    Err(e) => {
                        tracing::debug!(
                            "mixed ABC next-slot reservation failed; falling back to host alloc: {:?}",
                            e
                        );
                    }
                }
            }
        }
        let group_next_slots = if mixed_lease.is_empty() {
            None
        } else {
            Some(mixed_lease.as_slice())
        };
        let (out_res, prior_send) = if stochastic_group {
            // Mixed ABC bakes argmax into its graph/eager region. The regular
            // eager runtime leaves logits available to the filtered sampler.
            // Stochastic admission disabled overlap above, so no placeholder
            // decode ids can reach this branch.
            let out = runner.step(&req);
            let sent = send_prior_output(data, &mut prior_output);
            (out, sent)
        } else {
            // The optimistic decode rows carry placeholder input ids: their
            // real tokens (the prior step's argmax) are gathered from buffer C
            // on device, so any group holding them must use the overlapped
            // issue even after a fallback drain (C is untouched by finalize).
            let issue_res = if optimistic_decode && has_decode {
                runner.issue_fused_abc_overlapped(&req, &row_kinds, group_next_slots, decode_count)
            } else {
                runner.issue_fused_abc(&req, &row_kinds, group_next_slots)
            };
            // Drain the prior step only after the fused issue so its GPU tail
            // overlaps planning and issue. The failure path also drains it.
            if overlap_active {
                match decode_engine.finalize_pending(
                    runner,
                    active,
                    kv_allocator,
                    control,
                    enable_prefix_caching,
                ) {
                    Ok(out) => {
                        prior_output = out;
                        skip_rows = overlap_skip_rows(&prior_output, &decode_order, active);
                        if enable_prefix_caching && let Some(b) = decode_build.as_mut() {
                            patch_overlap_assigned_tokens(&prior_output, b);
                        }
                    }
                    Err(e) => {
                        mixed_lease.release(kv_allocator);
                        hard_err = Some(e);
                        break 'groups;
                    }
                }
                overlap_active = false;
            }
            // The fused forward is on the GPU now: send the prior decode tokens
            // while it runs, but propagate a send failure only after commit.
            let sent = send_prior_output(data, &mut prior_output);
            let out =
                issue_res.and_then(|ticket| runner.finalize_fused_abc(ticket, &req, &row_kinds));
            (out, sent)
        };
        let out = match out_res {
            Ok(out) => out,
            Err(e) => {
                mixed_lease.release(kv_allocator);
                if has_decode && let Some(lease) = decode_lease.take() {
                    lease.release(kv_allocator);
                }
                fail_fused_group(
                    e,
                    has_decode,
                    &decode_order,
                    decode_engine,
                    group,
                    &mut preps,
                    active,
                    prefilling,
                    kv_allocator,
                    control,
                    enable_prefix_caching,
                );
                pad_lease.release(kv_allocator);
                decode_engine.invalidate_abc_reuse();
                if let Err(send_err) = prior_send {
                    hard_err = Some(send_err);
                    break 'groups;
                }
                continue;
            }
        };

        let StepRequest {
            seqs: mut merged_seqs,
            ..
        } = req;
        let mut output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
            assigned_indices: Vec::new(),
        };
        let mut abc_next_rows: Vec<u64> = Vec::new();
        let mut cursor = 0usize;
        if has_decode {
            let b = decode_build.as_ref().unwrap();
            // Skipped rows' slots are freed by the commit below — don't report
            // them assigned (the prior step already finished those sequences).
            output.assigned_indices.extend(
                b.assigned
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !skip_rows.get(*i).copied().unwrap_or(false))
                    .map(|(_, a)| a.clone()),
            );
            for (i, &sid) in decode_order.iter().enumerate() {
                if !out.finished.get(i).copied().unwrap_or(false)
                    && !skip_rows.get(i).copied().unwrap_or(false)
                {
                    abc_next_rows.push(sid);
                }
            }
            let lease = decode_lease
                .take()
                .expect("decode lease present when has_decode");
            if let Err(e) = decode_engine.commit_fused_decode(
                active,
                kv_allocator,
                &decode_order,
                lease,
                &out.tokens,
                &out.finished,
                enable_prefix_caching,
                &mut output,
                &skip_rows,
            ) {
                pad_lease.release(kv_allocator);
                hard_err = Some(e);
                break 'groups;
            }
            cursor = decode_prefix_len;
        }
        for &pi in &group.preps {
            let prep = &mut preps[pi];
            reclaim_prefill_step_data(&mut merged_seqs, &mut prep.plans);
            let assigned = assigned_runs(prep.cmd, &prep.plans, enable_prefix_caching);
            output.assigned_indices.extend(assigned);
            append_prefill_abc_next_rows(
                prep.cmd,
                &prep.plans,
                &out.finished,
                cursor,
                &mut abc_next_rows,
            );
            commit_prefill_outputs(
                prep.cmd,
                &mut prep.plans,
                active,
                prefilling,
                kv_allocator,
                enable_prefix_caching,
                &out.tokens,
                &out.finished,
                cursor,
                &mut output,
            );
            // The cmd's slots now live in `prefilling`/`active` block tables
            // (or were released for finished seqs by commit_prefill_outputs).
            let _ = prep.base_indices.take().commit();
            cursor += prep.rows;
        }
        // Reclaim the inert decode-prefix pad KV (transient; never owned by any
        // sequence). The forward has completed (finalize above), so nothing is
        // reading these slots.
        pad_lease.release(kv_allocator);
        // A skipped row the device merge kept ACTIVE (an EOS finish the prior
        // step saw but the fused stop-check cannot re-detect) sits in A / the
        // device control plane while the host dropped it — the on-device state
        // no longer matches `abc_next_rows`, so it must not be recorded for
        // reuse. (Max-token finishes re-finish on device and stay consistent.)
        let device_rows_tainted = skip_rows
            .iter()
            .enumerate()
            .any(|(i, &s)| s && !out.finished.get(i).copied().unwrap_or(true));
        if can_record_abc_rows && !stochastic_group && !device_rows_tainted {
            decode_engine.record_mixed_abc_rows(
                abc_next_rows,
                mixed_lease,
                mixed_device_prepared,
                kv_allocator,
            );
            recorded_abc_rows = true;
        } else {
            mixed_lease.release(kv_allocator);
        }
        if let Err(e) = data.send_step_output(&output) {
            hard_err = Some(OpError::Kernel(format!(
                "data plane send_step_output failed: {}",
                e
            )));
            break 'groups;
        }
        if let Err(e) = prior_send {
            hard_err = Some(e);
            break 'groups;
        }
    }
    // Post-loop sweep: on a hard error, groups after the break never ran —
    // return their leases (and an uncommitted decode lease) to the pool. The
    // pre-lease code leaked these slots. On the normal path every lease was
    // consumed and this sweep is a no-op.
    if let Some(lease) = decode_lease.take() {
        lease.release(kv_allocator);
    }
    for prep in &mut preps {
        let lease = prep.base_indices.take();
        if !lease.is_empty() {
            tracing::warn!(
                slots = lease.len(),
                "returning unconsumed prefill KV lease (group never ran)"
            );
            lease.release(kv_allocator);
        }
    }
    if let Some(e) = hard_err {
        return Err(e);
    }
    // Rare path: no group issued a forward (e.g. every decode row finished in
    // the drained step and all cmds were rejected) — the prior step's tokens
    // still must go out.
    send_prior_output(data, &mut prior_output)?;
    if !recorded_abc_rows {
        decode_engine.invalidate_abc_reuse();
    }
    // Overlap the fused step's host tail (scheduler send + serve-loop
    // turnaround) with GPU compute: issue the next pure-decode step now and
    // leave it pending for the next `run_step` call to finalize. Also removes
    // the cold-start 0-deep restart the pipeline used to pay after every
    // fused step.
    decode_engine.issue_if_idle(
        runner,
        active,
        prefilling,
        kv_allocator,
        control,
        eos_ids,
        enable_prefix_caching,
    )?;
    Ok(())
}

/// Per-row skip mask for an overlapped fused step's decode rows: rows the
/// just-drained prior step FINISHED (the client already saw `finished`, so
/// the optimistic extra token must not be emitted). When the prior finalize
/// failed internally (its rows were failed and evicted), skip everything no
/// longer active.
fn overlap_skip_rows(
    prior_output: &Option<StepOutput>,
    decode_order: &[u64],
    active: &ActiveSeqMap,
) -> Vec<bool> {
    match prior_output {
        Some(out) => {
            let finished: std::collections::HashSet<u64> = out
                .tokens
                .iter()
                .filter(|t| t.finished)
                .map(|t| t.sequence_id)
                .collect();
            decode_order
                .iter()
                .map(|sid| finished.contains(sid))
                .collect()
        }
        None => decode_order
            .iter()
            .map(|sid| active.get(sid).is_none())
            .collect(),
    }
}

/// Prefix caching reports the token WRITTEN into each assigned slot. For the
/// overlapped decode rows that token is the prior step's output — unknown at
/// build time — so patch it in once the prior step is finalized (always
/// before this step's output is sent).
fn patch_overlap_assigned_tokens(
    prior_output: &Option<StepOutput>,
    build: &mut DecodeRequestBuild,
) {
    let Some(out) = prior_output else {
        return;
    };
    let tokens: std::collections::HashMap<u64, i32> = out
        .tokens
        .iter()
        .map(|t| (t.sequence_id, t.token_id))
        .collect();
    for a in &mut build.assigned {
        if let Some(&t) = tokens.get(&a.sequence_id) {
            a.token_ids = vec![t];
        }
    }
}

/// Send the prior decode step's finalized output, if still unsent. Called
/// right after the fused forward is issued so the ZMQ send overlaps GPU
/// compute; the early-return paths call it too so the tokens are never lost.
fn send_prior_output(data: &DataPump, prior: &mut Option<StepOutput>) -> OpResult<()> {
    if let Some(out) = prior.take() {
        data.send_step_output(&out).map_err(|e| {
            OpError::Kernel(format!(
                "data plane send_step_output (fused prior) failed: {}",
                e
            ))
        })?;
    }
    Ok(())
}

/// Clean up a fused forward that failed: free the group's freshly-allocated KV
/// (the decode rows' step lease is released by the caller), drop its decode
/// rows + prefilling state, and report the error for every affected sequence.
#[allow(clippy::too_many_arguments)]
fn fail_fused_group(
    e: OpError,
    has_decode: bool,
    decode_order: &[u64],
    decode_engine: &mut DecodeEngine,
    group: &ForwardGroup,
    preps: &mut [CmdPrep],
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    enable_prefix_caching: bool,
) {
    let mut failed: Vec<u64> = Vec::new();
    if has_decode {
        for &sid in decode_order {
            if let Some(removed) = active.remove(&sid)
                && !enable_prefix_caching
            {
                kv_allocator.free(&removed.block_table);
            }
            failed.push(sid);
        }
        decode_engine.reset_after_fused();
    }
    for &pi in &group.preps {
        let prep = &mut preps[pi];
        prep.base_indices.take().release(kv_allocator);
        for seg in &prep.cmd.segments {
            if let Some(removed) = prefilling.remove(&seg.sequence_id) {
                kv_allocator.release_owned(&removed.block_table, enable_prefix_caching);
            }
            failed.push(seg.sequence_id);
        }
    }
    tracing::warn!(
        failed = failed.len(),
        has_decode,
        prefill_groups = group.preps.len(),
        error = ?e,
        "fused step failed; failing affected sequences"
    );
    send_step_error(control, failed, format!("fused step failed: {:?}", e));
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
