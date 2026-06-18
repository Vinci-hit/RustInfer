use half::bf16;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::application::forward_workspace::ForwardWorkspace;
use infer_protocol::worker_to_scheduler_control::{WorkerControlMessage, WorkerStepError};
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, GeneratedToken, StepOutput};

use crate::application::kv_relief::{alloc_with_relief, AllocWithReliefOutcome};
use crate::application::model_runner::{DecodeCompactOutput, ModelRunner, SeqStep};
use crate::application::worker_state::{ActiveSeqMap, DecodeRows, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::LlmModel;
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;

/// DecodeEngine owns the worker-side decode row state.
///
/// This is the boundary for the next GPU rewrite: `DecodeRows` can be replaced
/// with persistent device-side row state without changing serve_loop again.
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

    pub fn run_step<M>(
        &mut self,
        runner: &mut ModelRunner<bf16, Cuda, M>,
        active: &mut ActiveSeqMap,
        prefilling: &mut PrefillSeqMap,
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
            self.rows.clear();
            return Ok(());
        }

        let (order, new_indices, append_start_row, append_tokens) = match self.prepare_step(
            active,
            prefilling,
            kv_allocator,
            control,
            enable_prefix_caching,
        )? {
            DecodePrep::Ready {
                order,
                new_indices,
                append_start_row,
                append_tokens,
            } => (order, new_indices, append_start_row, append_tokens),
            DecodePrep::Done => return Ok(()),
        };

        let inputs = build_decode_inputs(&order, &new_indices, active, enable_prefix_caching);

        let compact = match runner.step_decode_abc_compact(
            &inputs.steps,
            append_start_row,
            &append_tokens,
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
                self.rows.clear();
                return Ok(());
            }
        };

        let output = self.commit_results(
            active,
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

    fn prepare_step(
        &mut self,
        active: &mut ActiveSeqMap,
        prefilling: &mut PrefillSeqMap,
        kv_allocator: &mut GlobalKvAllocator,
        control: &ControlPump,
        enable_prefix_caching: bool,
    ) -> OpResult<DecodePrep> {
        self.rows.retain_active(active);
        let pending = self.rows.pending_admissions(active);
        if !pending.is_empty() {
            self.rows.append_admissions(&pending);
        }

        let mut order: Vec<u64> = self.rows.as_slice().to_vec();
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
                return Ok(DecodePrep::Done);
            }
            AllocWithReliefOutcome::Shutdown => return Err(OpError::Shutdown),
        };

        self.rows.retain_active(active);
        order = self.rows.as_slice().to_vec();
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
            self.rows.retain_active(active);
            order.truncate(new_indices.len());
            if order.is_empty() {
                if !new_indices.is_empty() {
                    kv_allocator.free(&new_indices);
                }
                return Ok(DecodePrep::Done);
            }
        }

        let (append_start_row, append_tokens) = build_a_append(&order, &pending, active);

        Ok(DecodePrep::Ready {
            order,
            new_indices,
            append_start_row,
            append_tokens,
        })
    }

    fn commit_results(
        &mut self,
        active: &mut ActiveSeqMap,
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

        trace_decode_commit(order, new_indices, active, &row_results, compact);

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
        self.rows.replace_rows(next_rows);
        output
    }
}

impl Default for DecodeEngine {
    fn default() -> Self {
        Self::new()
    }
}

enum DecodePrep {
    Ready {
        order: Vec<u64>,
        new_indices: Vec<u32>,
        append_start_row: usize,
        append_tokens: Vec<i32>,
    },
    Done,
}

struct DecodeInputs {
    steps: Vec<SeqStep>,
    assigned: Vec<AssignedIndices>,
    generated_counts: Vec<usize>,
    max_tokens: Vec<usize>,
    ignore_eos: Vec<bool>,
}

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

fn build_a_append(
    order: &[u64],
    pending_admissions: &[u64],
    active: &ActiveSeqMap,
) -> (usize, Vec<i32>) {
    if pending_admissions.is_empty() {
        return (order.len(), Vec::new());
    }

    let pending: HashSet<u64> = pending_admissions.iter().copied().collect();
    let start = order
        .iter()
        .position(|sid| pending.contains(sid))
        .unwrap_or(order.len());
    let tokens = order[start..]
        .iter()
        .filter_map(|sid| active.get(sid).map(|seq| seq.last_token))
        .collect();
    (start, tokens)
}

fn trace_decode_commit(
    order: &[u64],
    new_indices: &[u32],
    active: &ActiveSeqMap,
    row_results: &[Option<(i32, bool)>],
    compact: &DecodeCompactOutput,
) {
    if !crate::env_flags::trace_decode_compact() {
        return;
    }
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let step = TRACE_COUNT.fetch_add(1, Ordering::Relaxed);
    if step >= 32 {
        return;
    }

    tracing::warn!(
        step,
        rows = order.len(),
        active_rows = compact.active.len(),
        finished_rows = compact.finished.len(),
        "decode commit trace"
    );
    for (row, &sid) in order.iter().enumerate().take(8) {
        let (token, finished) = row_results.get(row).and_then(|v| *v).unwrap_or((-1, false));
        let seq = active.get(&sid);
        tracing::warn!(
            step,
            row,
            sequence_id = sid,
            input_token = seq.map(|s| s.last_token).unwrap_or(-1),
            kv_len = seq.map(|s| s.kv_len).unwrap_or(0),
            block_table_len = seq.map(|s| s.block_table.len()).unwrap_or(0),
            new_idx = new_indices.get(row).copied().unwrap_or(u32::MAX),
            token,
            finished,
            "decode commit row"
        );
    }
}

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
