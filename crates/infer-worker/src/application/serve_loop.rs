use std::time::{Duration, Instant};

use half::bf16;

use infer_protocol::scheduler_to_worker_control::{DrainMode, LoadModel, SchedulerControlMessage};
use infer_protocol::scheduler_to_worker_data::{BatchCommand, PrefillBatchCmd};
use infer_protocol::worker_to_scheduler_control::{
    CancelAck, DrainAck, UnloadAck, WorkerCapacity, WorkerControlMessage,
};
use infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput;

use crate::application::forward_workspace::ForwardWorkspace;
use crate::application::model_runner::ModelRunner;
use crate::application::worker_scheduler::{handle_prefill, run_decode_step};
use crate::application::worker_state::{ActiveSeqMap, DecodeRows, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::LlmModel;
use crate::infrastructure::cuda::{Cuda, device_utils, kernels::attention_paged};
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;
use crate::models::loader::LoadConfig;

// cudaProfiler API for precise nsys capture.
unsafe extern "C" {
    fn cudaProfilerStart() -> u32;
    fn cudaProfilerStop() -> u32;
}

/// Bundle of bootstrap parameters that do not depend on the concrete model type.
pub struct Bootstrap<'a> {
    pub load: &'a LoadModel,
    pub cuda: &'a Cuda,
    pub load_cfg: &'a LoadConfig,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub num_blocks_override: usize,
    pub server_heartbeat_ms: Option<u64>,
    /// Resolved from config.json, not `load.model_type`.
    pub model_type: String,
}

pub fn run_with_model<M>(
    control: &ControlPump,
    data: &DataPump,
    model: M,
    bs: Bootstrap<'_>,
    eos_ids: &[i32],
    profile_cuda_steps: Option<u32>,
) -> Result<(), String>
where
    M: LlmModel<bf16, Cuda, ForwardWorkspace<bf16, Cuda>>,
{
    let _ = bs.load_cfg;
    let max_blocks_per_seq = (bs.max_seq_len + bs.block_size - 1) / bs.block_size;

    let num_blocks = if bs.num_blocks_override != 0 {
        eprintln!(
            "[bootstrap] num_blocks override from CLI: {} (skipping GPU mem probe)",
            bs.num_blocks_override,
        );
        bs.num_blocks_override
    } else {
        let bytes_per_block =
            model.num_layers() * 2 * bs.block_size * model.kv_dim() * std::mem::size_of::<bf16>();
        let fraction = bs
            .load
            .kv_cache_memory_fraction
            .unwrap_or(0.9)
            .clamp(0.05, 0.98);
        let (free, total) =
            device_utils::mem_get_info().map_err(|e| format!("cudaMemGetInfo: {:?}", e))?;
        let budget = (free as f64 * fraction as f64) as usize;
        let derived = (budget / bytes_per_block.max(1)).saturating_sub(1).max(1);
        eprintln!(
            "[bootstrap] KV mem probe: free={:.2}GiB total={:.2}GiB fraction={} bytes/block={} -> num_blocks={} (~{:.2}GiB KV pool)",
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64,
            fraction,
            bytes_per_block,
            derived,
            (derived * bytes_per_block) as f64 / (1u64 << 30) as f64,
        );
        derived
    };

    let pool_blocks = num_blocks + 1;
    eprintln!(
        "[bootstrap] paged KV pool: block_size={} num_blocks={} pool_blocks={} (last block reserved as graph scratch) max_blocks_per_seq={}",
        bs.block_size, num_blocks, pool_blocks, max_blocks_per_seq,
    );

    let cap_num_tokens = bs.load.max_batch_tokens;
    let cap_batch = bs.load.max_batch_seqs;
    let flash_decode_capacity_f32 =
        attention_paged::flash_decode_workspace_capacity_f32(cap_batch.max(1), 128, 256);
    let mut runner: ModelRunner<bf16, Cuda, M> = ModelRunner::new(
        model,
        bs.cuda.clone(),
        pool_blocks,
        bs.block_size,
        max_blocks_per_seq,
        bs.max_seq_len,
        cap_num_tokens,
        cap_batch,
        flash_decode_capacity_f32,
        (1..=32).collect(),
    )
    .map_err(|e| format!("ModelRunner::new: {:?}", e))?;

    if let Err(e) = runner.prime_graphs_cuda() {
        eprintln!(
            "[bootstrap] CUDA Graph priming FAILED, continuing in eager mode: {:?}",
            e
        );
    } else {
        eprintln!(
            "[bootstrap] CUDA Graphs primed for decode-only batches in {:?}",
            runner.capture_sizes,
        );
    }

    let max_total_kv_tokens = num_blocks * bs.block_size;
    control.send_ready(
        bs.load.model_instance_id.clone(),
        bs.load.model_path.clone(),
        bs.model_type.clone(),
        WorkerCapacity {
            max_batch_tokens: bs.load.max_batch_tokens,
            max_batch_seqs: bs.load.max_batch_seqs,
            max_running_requests: bs.load.max_batch_seqs,
            max_total_kv_tokens: Some(max_total_kv_tokens),
            free_mem_before_load_gb: None,
            free_mem_after_load_gb: None,
            weight_mem_usage_gb: None,
            workspace_mem_usage_gb: None,
            graph_mem_usage_gb: None,
        },
    )?;
    eprintln!(
        "[bootstrap] Ready sent. max_total_kv_tokens={}. Entering serve loop...",
        max_total_kv_tokens,
    );

    let hb_ms = bs.server_heartbeat_ms.unwrap_or(1000).max(200);
    let heartbeat_interval = Duration::from_millis(hb_ms);
    eprintln!("[serve] heartbeat interval = {:?}", heartbeat_interval);
    let mut last_heartbeat = Instant::now();
    let mut active = ActiveSeqMap::new();
    let mut prefilling = PrefillSeqMap::new();
    let mut kv_allocator = GlobalKvAllocator::new(num_blocks as u32);
    eprintln!(
        "[serve] worker-owned KV allocator: total={} (block_size={})",
        num_blocks, bs.block_size,
    );
    let enable_prefix_caching = bs.load.enable_prefix_caching;
    eprintln!(
        "[serve] prefix caching: {}",
        if enable_prefix_caching {
            "enabled (RadixTree)"
        } else {
            "disabled (real-time recycling)"
        },
    );
    let mut profile_step_count: u32 = 0;
    let mut profile_started = false;
    let mut decode_rows = DecodeRows::new();

    loop {
        if drain_control(
            control,
            &mut active,
            &mut prefilling,
            &mut decode_rows,
            &mut kv_allocator,
            enable_prefix_caching,
        ) {
            return Ok(());
        }

        let mut pending_prefills = drain_data(data);
        if pending_prefills.is_empty() && active.is_empty() {
            maybe_heartbeat(
                control,
                active.len() + prefilling.len(),
                &mut last_heartbeat,
                heartbeat_interval,
            );
            let idle_wait_ms = (heartbeat_interval.as_millis() as i64 / 2).max(50);
            match data.try_recv_batch(idle_wait_ms) {
                Ok(Some(BatchCommand::Prefill(p))) => pending_prefills.push(p),
                Ok(Some(BatchCommand::DiffusionBatch(_))) => {
                    let _ = data.send_diffusion_output(&DiffusionBatchOutput { results: vec![] });
                }
                _ => {}
            }
            if pending_prefills.is_empty() && active.is_empty() {
                continue;
            }
        }

        if !active.is_empty() {
            if let Some(max_steps) = profile_cuda_steps {
                if !profile_started {
                    unsafe {
                        cudaProfilerStart();
                    }
                    profile_started = true;
                    eprintln!(
                        "[profile] cudaProfilerStart (will stop after {} steps)",
                        max_steps
                    );
                }
            }

            if let Err(e) = run_decode_step(
                &mut runner,
                &mut active,
                &mut prefilling,
                &mut decode_rows,
                &mut kv_allocator,
                control,
                data,
                eos_ids,
                enable_prefix_caching,
            ) {
                eprintln!("[serve] decode error: {}", e);
            }

            if let Some(max_steps) = profile_cuda_steps {
                if profile_started {
                    profile_step_count += 1;
                    if profile_step_count >= max_steps {
                        unsafe {
                            cudaProfilerStop();
                        }
                        eprintln!(
                            "[profile] cudaProfilerStop after {} steps. Exiting.",
                            profile_step_count
                        );
                        return Ok(());
                    }
                }
            }
        }

        for cmd in pending_prefills {
            if drain_control(
                control,
                &mut active,
                &mut prefilling,
                &mut decode_rows,
                &mut kv_allocator,
                enable_prefix_caching,
            ) {
                return Ok(());
            }
            if let Err(e) = handle_prefill(
                &mut runner,
                &cmd,
                &mut active,
                &mut prefilling,
                &mut kv_allocator,
                control,
                data,
                eos_ids,
                enable_prefix_caching,
                cap_batch,
            ) {
                eprintln!("[serve] prefill error: {}", e);
            }
        }

        maybe_heartbeat(
            control,
            active.len() + prefilling.len(),
            &mut last_heartbeat,
            heartbeat_interval,
        );
    }
}

fn drain_control(
    control: &ControlPump,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    decode_rows: &mut DecodeRows,
    kv_allocator: &mut GlobalKvAllocator,
    enable_prefix_caching: bool,
) -> bool {
    loop {
        match control.try_recv(0) {
            Ok(Some((msg, req_id))) => match msg {
                SchedulerControlMessage::Shutdown => {
                    eprintln!("[serve] Shutdown received, exiting.");
                    return true;
                }
                SchedulerControlMessage::Cancel(c) => {
                    let removed = active.remove(&c.sequence_id);
                    let removed_prefill = prefilling.remove(&c.sequence_id);
                    let removed_flag = removed.is_some() || removed_prefill.is_some();
                    if let Some(removed) = removed {
                        release_removed(removed.block_table, kv_allocator, enable_prefix_caching);
                        decode_rows.retain_active(active);
                        eprintln!("[serve] cancelled seq {}", c.sequence_id);
                    }
                    if let Some(removed) = removed_prefill {
                        release_removed(removed.block_table, kv_allocator, enable_prefix_caching);
                        eprintln!("[serve] cancelled prefilling seq {}", c.sequence_id);
                    }
                    if req_id.is_correlated() {
                        let _ = control.send(
                            WorkerControlMessage::CancelAck(CancelAck {
                                sequence_id: c.sequence_id,
                                removed: removed_flag,
                            }),
                            req_id,
                        );
                    }
                }
                SchedulerControlMessage::FreeKvIndices(free) => {
                    if !free.indices.is_empty() {
                        kv_allocator.free(&free.indices);
                    }
                }
                SchedulerControlMessage::Preempt(p) => {
                    for sid in &p.sequence_ids {
                        if let Some(removed) = active.remove(sid) {
                            release_removed(
                                removed.block_table,
                                kv_allocator,
                                enable_prefix_caching,
                            );
                            eprintln!("[serve] preempted seq {}", sid);
                        }
                        if let Some(removed) = prefilling.remove(sid) {
                            release_removed(
                                removed.block_table,
                                kv_allocator,
                                enable_prefix_caching,
                            );
                            eprintln!("[serve] preempted prefilling seq {}", sid);
                        }
                    }
                    decode_rows.retain_active(active);
                    if !p.free_indices.is_empty() {
                        kv_allocator.free(&p.free_indices);
                    }
                }
                SchedulerControlMessage::Ping => {
                    let _ = control.send(WorkerControlMessage::Pong, req_id);
                }
                SchedulerControlMessage::Drain(d) => {
                    if matches!(d.mode, DrainMode::Immediate) {
                        let removed: Vec<_> = active.drain().map(|(_, seq)| seq).collect();
                        for seq in removed {
                            release_removed(seq.block_table, kv_allocator, enable_prefix_caching);
                        }
                        let removed_prefills: Vec<_> =
                            prefilling.drain().map(|(_, seq)| seq).collect();
                        for seq in removed_prefills {
                            release_removed(seq.block_table, kv_allocator, enable_prefix_caching);
                        }
                        decode_rows.clear();
                    }
                    if req_id.is_correlated() {
                        let _ = control.send(
                            WorkerControlMessage::DrainAck(DrainAck {
                                remaining_requests: active.len() + prefilling.len(),
                            }),
                            req_id,
                        );
                    }
                }
                SchedulerControlMessage::UnloadModel(u) => {
                    if req_id.is_correlated() {
                        let _ = control.send(
                            WorkerControlMessage::UnloadAck(UnloadAck {
                                model_instance_id: u.model_instance_id,
                            }),
                            req_id,
                        );
                    }
                    eprintln!("[serve] UnloadModel received, exiting.");
                    return true;
                }
                _ => {}
            },
            _ => break,
        }
    }
    false
}

fn drain_data(data: &DataPump) -> Vec<PrefillBatchCmd> {
    let mut pending_prefills = Vec::new();
    loop {
        match data.try_recv_batch(0) {
            Ok(Some(BatchCommand::Prefill(p))) => pending_prefills.push(p),
            Ok(Some(BatchCommand::DiffusionBatch(_))) => {
                let _ = data.send_diffusion_output(&DiffusionBatchOutput { results: vec![] });
            }
            _ => break,
        }
    }
    pending_prefills
}

fn release_removed(
    block_table: Vec<u32>,
    kv_allocator: &mut GlobalKvAllocator,
    enable_prefix_caching: bool,
) {
    if block_table.is_empty() {
        return;
    }
    if !enable_prefix_caching {
        kv_allocator.release(&block_table);
    }
}

fn maybe_heartbeat(control: &ControlPump, active_n: usize, last: &mut Instant, interval: Duration) {
    if last.elapsed() >= interval {
        let _ = control.send_heartbeat(active_n);
        *last = Instant::now();
    }
}
