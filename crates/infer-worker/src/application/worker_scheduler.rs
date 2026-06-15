use half::bf16;

use infer_protocol::scheduler_to_worker_data::{PrefillBatchCmd, PrefillSegmentCompletion};
use infer_protocol::worker_to_scheduler_control::{WorkerControlMessage, WorkerStepError};
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::forward_workspace::ForwardWorkspace;
use crate::application::kv_relief::alloc_with_relief;
use crate::application::model_runner::{ModelRunner, SeqStep};
use crate::application::worker_state::{
    ActiveSeq, ActiveSeqMap, DecodeRows, PrefillSeq, PrefillSeqMap,
};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::LlmModel;
use crate::domain::ports::OpResult;
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;

/// Run one prefill batch (one PrefillBatchCmd = potentially many segments).
///
/// Each segment's KV slots come from the worker-owned
/// `GlobalKvAllocator`, not from `seg.block_table`. Optional
/// `seg.prefix_hint` is prepended verbatim; those slots already hold valid
/// KV from a previous request and are pinned by scheduler-side RadixTree
/// policy while this step runs.
pub fn handle_prefill<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    cmd: &PrefillBatchCmd,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    data: &DataPump,
    eos_ids: &[i32],
    enable_prefix_caching: bool,
    cap_batch: usize,
) -> OpResult<()>
where
    M: LlmModel<bf16, Cuda, ForwardWorkspace<bf16, Cuda>>,
{
    let new_decode_segments = cmd
        .segments
        .iter()
        .filter(|s| {
            matches!(
                s.completion,
                PrefillSegmentCompletion::FinishPrefillAndStartDecode
            )
        })
        .count();
    if active.len() + new_decode_segments > cap_batch {
        let overflow_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
        eprintln!(
            "[serve] prefill rejected: active ({}) + new ({}) > cap ({}) -- failing {} seqs",
            active.len(),
            new_decode_segments,
            cap_batch,
            overflow_ids.len(),
        );
        let _ = control.send(
            WorkerControlMessage::StepError(WorkerStepError {
                sequence_ids: overflow_ids,
                message: format!(
                    "worker batch slot exhausted: active={} + new={} > cap={}",
                    active.len(),
                    new_decode_segments,
                    cap_batch,
                ),
                fatal: false,
            }),
            infer_protocol::control_envelope::RequestId(0),
        );
        return Ok(());
    }

    let mut per_seg_new_tokens: Vec<u32> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_prefix_hint: Vec<Vec<u32>> = Vec::with_capacity(cmd.segments.len());
    let mut total_new: u32 = 0;
    for seg in &cmd.segments {
        let seg_len = seg.segment_end - seg.segment_start;
        let prefix_hit = seg
            .prefix_hint
            .as_ref()
            .map(|h| h.len() as u32)
            .unwrap_or(0);
        let new_tokens = if seg.segment_start == 0 {
            seg_len.saturating_sub(prefix_hit)
        } else {
            seg_len
        };
        per_seg_new_tokens.push(new_tokens);
        per_seg_prefix_hint.push(seg.prefix_hint.clone().unwrap_or_default());
        total_new = total_new.saturating_add(new_tokens);
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
        Some(v) => v,
        None => {
            eprintln!(
                "[serve] prefill alloc still failing after relief -- failing batch (n={})",
                total_new
            );
            let failed_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: failed_ids,
                    message: format!("worker KV pool exhausted (n={})", total_new),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            return Ok(());
        }
    };

    let mut steps: Vec<SeqStep> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_indices: Vec<Vec<u32>> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_token_ids: Vec<Vec<i32>> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_base_tables: Vec<Vec<u32>> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_skipped: Vec<bool> = Vec::with_capacity(cmd.segments.len());
    let mut unused_indices: Vec<u32> = Vec::new();
    let mut idx_cursor = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        let new_tokens = per_seg_new_tokens[i] as usize;
        let prefix = &per_seg_prefix_hint[i];
        let base_table = prefilling
            .get(&seg.sequence_id)
            .map(|state| state.block_table.clone())
            .unwrap_or_else(|| prefix.clone());
        let expected_base_len = seg.segment_start as usize;
        let skip_segment = if base_table.len() != expected_base_len {
            eprintln!(
                "[serve] skipping stale prefill segment seq={} base_len={} expected={}",
                seg.sequence_id,
                base_table.len(),
                expected_base_len,
            );
            true
        } else {
            false
        };
        let range = cmd.segment_token_range(i);
        let (input_ids, positions): (Vec<i32>, Vec<i32>) = if seg.segment_start == 0
            && !prefix.is_empty()
        {
            let prefix_len = prefix.len();
            let prompt = &cmd.input_ids[range.clone()];
            let trimmed: Vec<i32> = prompt.iter().skip(prefix_len).copied().collect();
            let positions: Vec<i32> =
                ((prefix_len as i32)..(prefix_len as i32 + trimmed.len() as i32)).collect();
            (trimmed, positions)
        } else {
            let prompt = cmd.input_ids[range.clone()].to_vec();
            let positions: Vec<i32> = (seg.segment_start as i32..seg.segment_end as i32).collect();
            (prompt, positions)
        };

        let new_indices: Vec<u32> = base_indices[idx_cursor..idx_cursor + new_tokens].to_vec();
        idx_cursor += new_tokens;
        if skip_segment {
            unused_indices.extend_from_slice(&new_indices);
            per_seg_indices.push(Vec::new());
            per_seg_token_ids.push(Vec::new());
            per_seg_base_tables.push(base_table);
            per_seg_skipped.push(true);
            continue;
        }

        let mut block_table: Vec<u32> = Vec::with_capacity(base_table.len() + new_tokens);
        block_table.extend_from_slice(&base_table);
        block_table.extend_from_slice(&new_indices);

        let kv_write_start = block_table.len() as i32 - new_tokens as i32;
        let kv_len_after = block_table.len() as i32;
        steps.push(SeqStep {
            input_ids: input_ids.clone(),
            positions,
            kv_write_start,
            kv_len_after,
            block_table,
        });
        per_seg_indices.push(new_indices);
        per_seg_token_ids.push(input_ids);
        per_seg_base_tables.push(base_table);
        per_seg_skipped.push(false);
    }
    debug_assert_eq!(idx_cursor, total_new as usize);

    if steps.is_empty() {
        if !base_indices.is_empty() {
            kv_allocator.free(&base_indices);
        }
        return Ok(());
    }

    let first_tokens = match runner.step_batch(&steps) {
        Ok(tokens) => tokens,
        Err(e) => {
            if !base_indices.is_empty() {
                kv_allocator.free(&base_indices);
            }
            for seg in &cmd.segments {
                if let Some(removed) = prefilling.remove(&seg.sequence_id) {
                    release_prefill_state(removed, kv_allocator, enable_prefix_caching);
                }
            }
            let failed_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: failed_ids,
                    message: format!("prefill step failed: {:?}", e),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            return Ok(());
        }
    };
    if !unused_indices.is_empty() {
        kv_allocator.free(&unused_indices);
    }
    let assigned = assigned_runs(
        cmd,
        &per_seg_indices,
        &per_seg_token_ids,
        enable_prefix_caching,
    );

    let mut output = StepOutput {
        prefill_done: Vec::new(),
        tokens: Vec::new(),
        assigned_indices: assigned,
    };
    let mut token_cursor = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        if per_seg_skipped[i] {
            continue;
        }
        let token = first_tokens[token_cursor];
        token_cursor += 1;
        let new_indices = &per_seg_indices[i];
        let mut full_block_table =
            Vec::with_capacity(per_seg_base_tables[i].len() + new_indices.len());
        full_block_table.extend_from_slice(&per_seg_base_tables[i]);
        full_block_table.extend_from_slice(new_indices);
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
                let finished = (!seg.ignore_eos && eos_ids.contains(&token)) || seg.max_tokens <= 1;
                output.tokens.push(GeneratedToken {
                    sequence_id: seg.sequence_id,
                    token_id: token,
                    finished,
                });
                if !finished {
                    active.insert(
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
                } else if !enable_prefix_caching && !full_block_table.is_empty() {
                    kv_allocator.release(&full_block_table);
                }
            }
        }
    }
    let _ = data.send_step_output(&output);
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
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: failed.clone(),
                    message: format!("decode admission append failed: {:?}", e),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            for sid in failed {
                if let Some(removed) = active.remove(&sid) {
                    if !enable_prefix_caching && !removed.block_table.is_empty() {
                        kv_allocator.release(&removed.block_table);
                    }
                }
            }
            decode_rows.clear();
            return Ok(());
        }
        decode_rows.append_admissions(&pending);
    }

    let mut order: Vec<u64> = decode_rows.as_slice().to_vec();
    if order.is_empty() {
        return Ok(());
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
        Some(v) => v,
        None => {
            let order: Vec<u64> = decode_rows.as_slice().to_vec();
            eprintln!(
                "[serve] decode alloc still failing after relief -- failing {} seqs",
                order.len()
            );
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: order.clone(),
                    message: "worker KV pool exhausted at decode".to_string(),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            for sid in &order {
                if let Some(removed) = active.remove(sid) {
                    if !enable_prefix_caching && !removed.block_table.is_empty() {
                        kv_allocator.release(&removed.block_table);
                    }
                }
            }
            decode_rows.clear();
            return Ok(());
        }
    };

    decode_rows.retain_active(active);
    order = decode_rows.as_slice().to_vec();
    if order.is_empty() {
        if !new_indices.is_empty() {
            kv_allocator.free(&new_indices);
        }
        return Ok(());
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
        let _ = control.send(
            WorkerControlMessage::StepError(WorkerStepError {
                sequence_ids: failed.clone(),
                message: format!(
                    "decode KV allocation returned {} slots for {} rows",
                    new_indices.len(),
                    order.len()
                ),
                fatal: false,
            }),
            infer_protocol::control_envelope::RequestId(0),
        );
        for sid in &failed {
            if let Some(removed) = active.remove(sid) {
                if !enable_prefix_caching && !removed.block_table.is_empty() {
                    kv_allocator.release(&removed.block_table);
                }
            }
        }
        decode_rows.retain_active(active);
        order.truncate(new_indices.len());
        if order.is_empty() {
            if !new_indices.is_empty() {
                kv_allocator.free(&new_indices);
            }
            return Ok(());
        }
    }

    let mut steps: Vec<SeqStep> = Vec::with_capacity(order.len());
    let mut assigned: Vec<AssignedIndices> = Vec::with_capacity(order.len());
    let mut generated_counts: Vec<usize> = Vec::with_capacity(order.len());
    let mut max_tokens_vec: Vec<usize> = Vec::with_capacity(order.len());
    let mut ignore_eos_vec: Vec<bool> = Vec::with_capacity(order.len());
    for (i, &sid) in order.iter().enumerate() {
        let new_idx = new_indices[i];
        let seq = active.get(&sid).unwrap();
        let mut bt = seq.block_table.clone();
        bt.push(new_idx);
        steps.push(SeqStep {
            input_ids: vec![seq.last_token],
            positions: vec![seq.kv_len as i32],
            kv_write_start: seq.kv_len as i32,
            kv_len_after: (seq.kv_len + 1) as i32,
            block_table: bt,
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
        generated_counts.push(seq.generated_count);
        max_tokens_vec.push(seq.max_tokens);
        ignore_eos_vec.push(seq.ignore_eos);
    }
    let compact = match runner.step_decode_abc_compact(
        &steps,
        &generated_counts,
        &max_tokens_vec,
        &ignore_eos_vec,
        eos_ids,
    ) {
        Ok(output) => output,
        Err(e) => {
            if !new_indices.is_empty() {
                kv_allocator.free(&new_indices);
            }
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: order.clone(),
                    message: format!("decode step failed: {:?}", e),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            for sid in &order {
                if let Some(removed) = active.remove(sid) {
                    if !enable_prefix_caching && !removed.block_table.is_empty() {
                        kv_allocator.release(&removed.block_table);
                    }
                }
            }
            decode_rows.clear();
            return Ok(());
        }
    };

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
        let removed = active.remove(sid);
        if let Some(removed) = removed {
            if !enable_prefix_caching && !removed.block_table.is_empty() {
                kv_allocator.release(&removed.block_table);
            }
        }
    }
    decode_rows.replace_rows(next_rows);
    let _ = data.send_step_output(&output);
    Ok(())
}

fn release_prefill_state(
    removed: PrefillSeq,
    kv_allocator: &mut GlobalKvAllocator,
    enable_prefix_caching: bool,
) {
    if removed.block_table.is_empty() {
        return;
    }
    if !enable_prefix_caching {
        kv_allocator.release(&removed.block_table);
    }
}

fn assigned_runs(
    cmd: &PrefillBatchCmd,
    per_seg_indices: &[Vec<u32>],
    per_seg_token_ids: &[Vec<i32>],
    include_token_ids: bool,
) -> Vec<AssignedIndices> {
    let mut assigned = Vec::new();
    for (i, seg) in cmd.segments.iter().enumerate() {
        let indices = &per_seg_indices[i];
        let token_ids = &per_seg_token_ids[i];
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
