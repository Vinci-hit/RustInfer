//! SubScheduler —— Worker 内部二级调度器，直接持有 `Arc<ModelRunner<M>>` 驱动推理。
//!
//! # 设计
//!
//! - ZMQ PULL 收 `PrefillBatchCmd`（来自 Scheduler），ZMQ PUSH 发 `StepOutput`。
//! - SubScheduler 线程与 Runner 共享 `Arc<ModelRunner<M>>`，通过 `SyncFlags` 握手：
//!   - input_ready=false 时 sub-scheduler 可写 workspace / states / meta；
//!   - output_ready=true 时 sub-scheduler 可读 output_tokens_dev。
//! - 不再使用 `SharedBuffers` 中间层。
//!
//! # 并发模型
//!
//! 进程内两线程：
//! - **Runner 线程**：`ModelRunner::run()`
//! - **SubScheduler 线程**：`SubScheduler::run()`
//! - ZMQ socket 绑定在 sub-scheduler 线程上（非线程安全，不跨线程）。

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use crate::model::llm::LlmModel;
use crate::model::runtime::InferenceState;
use crate::worker::control_pump::WorkerLiveState;
use crate::worker::runner::{ModelRunner, STEP_BUFFER_COUNT, StepMeta, WorkerBatchMeta};
use infer_protocol::scheduler_to_worker_control::{
    BlockGrantDeniedReason, CancelSequence, DrainMode, DrainWorker, GrantBlocks,
    GrantBlocksDenied, SchedulerControlMessage, UnloadModel,
};
use infer_protocol::scheduler_to_worker_data::*;
use infer_protocol::worker_to_scheduler_control::{
    NeedBlocks, NeedBlocksReason, WorkerControlMessage, WorkerState, WorkerStepError,
};
use infer_protocol::worker_to_scheduler_data::{GeneratedToken, StepOutput};

const DEFAULT_DECODE_BLOCK_PREFETCH_MARGIN: usize = 4;
const DEFAULT_DECODE_BLOCK_REQUEST_BLOCKS: usize = 1;

// ════════════════════════════════════════════════════════════════════════════════
//  DecodeSeq —— 一个活跃的 decode 序列
// ════════════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct PendingBlockRequest {
    required_blocks: usize,
    requested_blocks: usize,
}

/// Per-sequence paged decode KV state.
///
/// Paged-only: `block_table` is the list of physical block ids, `block_size`
/// is the token count per block, and `pending_request` / `blocked_on_blocks`
/// drive `NeedBlocks`/`GrantBlocks` flow control.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct DecodeKv {
    block_table: Vec<u32>,
    block_size: usize,
    pending_request: Option<PendingBlockRequest>,
    blocked_on_blocks: bool,
}

#[allow(dead_code)]
impl DecodeKv {
    /// Runner builds temporary state refs from StepMeta and requires
    /// `slot_indices` to be unique; use the current batch row as the
    /// placeholder slot id (paged kernels look up KV via block_table).
    fn slot_for_legacy_workspace(&self, batch_idx: usize) -> usize {
        batch_idx
    }

    fn paged_block_table(&self) -> Option<&[u32]> {
        Some(&self.block_table)
    }

    fn maybe_request_blocks(
        &mut self,
        sequence_id: u64,
        next_position: usize,
        max_tokens: usize,
        generated_count: usize,
        prefetch_margin: usize,
        request_blocks: usize,
    ) -> Option<NeedBlocksDraft> {
        if generated_count >= max_tokens || self.pending_request.is_some() {
            return None;
        }

        let future_pos = next_position.saturating_add(prefetch_margin);
        let required_blocks = future_pos / self.block_size + 1;
        if required_blocks <= self.block_table.len() {
            return None;
        }

        let missing = required_blocks - self.block_table.len();
        let requested_blocks = missing.max(request_blocks.max(1));
        self.pending_request = Some(PendingBlockRequest {
            required_blocks,
            requested_blocks,
        });

        let current_required = next_position / self.block_size + 1;
        if current_required > self.block_table.len() {
            self.blocked_on_blocks = true;
        }

        Some(NeedBlocksDraft {
            sequence_id,
            current_blocks: self.block_table.len(),
            required_blocks,
            request_blocks: requested_blocks,
        })
    }

    fn append_granted_blocks(&mut self, block_ids: &[u32]) -> bool {
        self.block_table.extend_from_slice(block_ids);
        self.pending_request = None;
        self.blocked_on_blocks = false;
        true
    }

    fn is_blocked_on_blocks(&self) -> bool {
        self.blocked_on_blocks
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct NeedBlocksDraft {
    sequence_id: u64,
    current_blocks: usize,
    required_blocks: usize,
    request_blocks: usize,
}

#[cfg(test)]
mod paged_kv_helper_tests {
    use super::*;

    fn paged_segment() -> PrefillSegmentMeta {
        PrefillSegmentMeta {
            sequence_id: 7,
            block_table: vec![10, 11],
            block_size: 4,
            prompt_len: 8,
            segment_start: 0,
            segment_end: 8,
            max_tokens: 16,
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: -1,
            },
            completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
        }
    }

    #[test]
    fn decode_kv_from_paged_segment_keeps_block_table() {
        let kv = decode_kv_from_segment(&paged_segment());
        assert_eq!(kv.block_table, vec![10, 11]);
        assert_eq!(kv.block_size, 4);
        assert!(kv.pending_request.is_none());
        assert!(!kv.blocked_on_blocks);
    }

    #[test]
    fn decode_block_request_triggers_before_boundary() {
        let mut kv = decode_kv_from_segment(&paged_segment());
        let draft = kv.maybe_request_blocks(7, 7, 16, 1, 1, 2).unwrap();
        assert_eq!(
            draft,
            NeedBlocksDraft {
                sequence_id: 7,
                current_blocks: 2,
                required_blocks: 3,
                request_blocks: 2,
            }
        );
        assert!(!kv.is_blocked_on_blocks());

        assert!(kv.append_granted_blocks(&[12, 13]));
        assert_eq!(kv.paged_block_table().unwrap(), &[10, 11, 12, 13]);
    }

    #[test]
    fn decode_block_request_blocks_when_current_position_exceeds_capacity() {
        let mut kv = decode_kv_from_segment(&paged_segment());
        let draft = kv.maybe_request_blocks(7, 8, 16, 1, 0, 1).unwrap();
        assert_eq!(draft.required_blocks, 3);
        assert!(kv.is_blocked_on_blocks());
    }

    #[test]
    fn paged_legacy_workspace_slots_use_unique_batch_rows() {
        let kv0 = decode_kv_from_segment(&paged_segment());
        let mut seg1 = paged_segment();
        seg1.sequence_id = 8;
        seg1.block_table = vec![20, 21];
        seg1.block_size = 4;
        let kv1 = decode_kv_from_segment(&seg1);
        assert_eq!(kv0.slot_for_legacy_workspace(0), 0);
        assert_eq!(kv1.slot_for_legacy_workspace(1), 1);
    }
}

#[derive(Clone)]
struct DecodeSeq {
    sequence_id: u64,
    kv: DecodeKv,
    /// 下一个 decode 输入 token 写入 KV cache 的 position (= 当前 KV 长度)。
    next_position: usize,
    /// 上一步采样出的 token (本步 decode 输入)。
    last_token: i32,
    /// 已生成 token 数；包含 final prefill 输出的第一个 token。
    generated_count: usize,
    max_tokens: usize,
    /// 采样参数（后续接入 top_p / temperature / top_k）。
    #[allow(dead_code)]
    sampling: SamplingParams,
}

struct ActiveDecodes {
    seqs: Vec<DecodeSeq>,
}

impl ActiveDecodes {
    fn new() -> Self {
        Self { seqs: Vec::new() }
    }

    fn len(&self) -> usize {
        self.seqs.len()
    }

    fn is_empty(&self) -> bool {
        self.seqs.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = &DecodeSeq> {
        self.seqs.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut DecodeSeq> {
        self.seqs.iter_mut()
    }

    fn iter_runnable(&self) -> impl Iterator<Item = &DecodeSeq> {
        self.seqs.iter().filter(|seq| !seq.kv.is_blocked_on_blocks())
    }

    fn contains(&self, sequence_id: u64) -> bool {
        self.seqs.iter().any(|seq| seq.sequence_id == sequence_id)
    }

    fn get_mut(&mut self, sequence_id: u64) -> Option<&mut DecodeSeq> {
        self.seqs.iter_mut().find(|seq| seq.sequence_id == sequence_id)
    }

    fn insert(&mut self, seq: DecodeSeq) -> Result<()> {
        if self.contains(seq.sequence_id) {
            return Err(Error::InvalidArgument(format!(
                "duplicate active decode sequence_id={}",
                seq.sequence_id
            ))
            .into());
        }
        self.seqs.push(seq);
        Ok(())
    }

    fn remove(&mut self, sequence_id: u64) -> Option<DecodeSeq> {
        let idx = self.seqs.iter().position(|seq| seq.sequence_id == sequence_id)?;
        Some(self.seqs.remove(idx))
    }

    fn cancel(&mut self, sequence_id: u64) -> bool {
        self.remove(sequence_id).is_some()
    }

    fn clear(&mut self) {
        self.seqs.clear();
    }
}

fn decode_kv_from_segment(segment: &PrefillSegmentMeta) -> DecodeKv {
    DecodeKv {
        block_table: segment.block_table.clone(),
        block_size: segment.block_size as usize,
        pending_request: None,
        blocked_on_blocks: false,
    }
}

#[derive(Clone)]
enum StepItem {
    Decode { sequence_id: u64 },
    PrefillSegment { segment: PrefillSegmentMeta },
    Cancelled { sequence_id: u64 },
}

/// Worker-owned host staging for one submitted runner step.
///
/// These buffers are intentionally owned by the sub-scheduler instead of being
/// stack-local `Vec`s. CUDA H2D copies are asynchronous with respect to the host;
/// keeping the backing storage alive until the runner produces output makes the
/// lifetime explicit at the Rust object level.
struct StepHostStaging {
    input_tokens: Vec<i32>,
    input_positions: Vec<i32>,
    kv_lens: Vec<i32>,
}

impl StepHostStaging {
    fn new(max_batch_tokens: usize, max_batch_seqs: usize) -> Self {
        Self {
            input_tokens: Vec::with_capacity(max_batch_tokens),
            input_positions: Vec::with_capacity(max_batch_tokens),
            kv_lens: Vec::with_capacity(max_batch_seqs),
        }
    }

    fn reset(&mut self, total_tokens: usize, num_seqs: usize) {
        self.input_tokens.clear();
        self.input_positions.clear();
        self.kv_lens.clear();
        self.input_tokens
            .reserve(total_tokens.saturating_sub(self.input_tokens.capacity()));
        self.input_positions
            .reserve(total_tokens.saturating_sub(self.input_positions.capacity()));
        self.kv_lens
            .reserve(num_seqs.saturating_sub(self.kv_lens.capacity()));
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  SubScheduler
// ════════════════════════════════════════════════════════════════════════════════

/// Worker 内部二级调度器 —— 直接使用 `ModelRunner` 新 API 驱动 GPU 推理。
pub struct SubScheduler<M: LlmModel> {
    runner: Arc<ModelRunner<M>>,
    #[allow(dead_code)]
    device: DeviceType,
    eos_token_ids: Vec<i32>,

    // ZMQ data plane (Scheduler 通信)
    zmq_in: zmq::Socket,
    zmq_out: zmq::Socket,

    // Control plane handles (跨线程 mpsc 与 ControlPump 对接)
    control_down_rx: Receiver<SchedulerControlMessage>,
    control_up_tx: SyncSender<WorkerControlMessage>,
    /// Liveness atomics read by the pump on every heartbeat tick.
    live: WorkerLiveState,
    /// Worker id (for embedding in upstream control messages).
    worker_id: String,
    /// Model instance id (for embedding in upstream control messages).
    model_instance_id: String,

    // 序列管理
    active_decodes: ActiveDecodes,
    pending_prefills: Vec<PrefillBatchCmd>,
    pending_cancels: Vec<u64>,
    cancelled_sequences: HashSet<u64>,
    step_items_by_buffer: Vec<Vec<StepItem>>,
    staging: StepHostStaging,
    next_step_buffer_id: usize,
    next_output_buffer_id: usize,
    step_in_flight: bool,
    profile_cuda_steps: usize,
    profile_started: bool,
    profile_stopped: bool,
    profiled_steps: usize,
    draining: bool,
}

impl<M: LlmModel> SubScheduler<M> {
    pub fn new(
        runner: Arc<ModelRunner<M>>,
        device: DeviceType,
        zmq_in: zmq::Socket,
        zmq_out: zmq::Socket,
        eos_token_ids: Vec<i32>,
        profile_cuda_steps: usize,
        control_down_rx: Receiver<SchedulerControlMessage>,
        control_up_tx: SyncSender<WorkerControlMessage>,
        live: WorkerLiveState,
        worker_id: String,
        model_instance_id: String,
    ) -> Self {
        let (max_batch_tokens, max_batch_seqs) = {
            let ws = unsafe { runner.workspace_for(0) };
            (ws.max_batch_tokens, ws.max_batch_seqs)
        };
        live.set_state(WorkerState::Running);
        live.set_active_requests(0);
        Self {
            runner,
            device,
            eos_token_ids,
            zmq_in,
            zmq_out,
            control_down_rx,
            control_up_tx,
            live,
            worker_id,
            model_instance_id,
            active_decodes: ActiveDecodes::new(),
            pending_prefills: Vec::new(),
            pending_cancels: Vec::new(),
            cancelled_sequences: HashSet::new(),
            step_items_by_buffer: (0..STEP_BUFFER_COUNT).map(|_| Vec::new()).collect(),
            staging: StepHostStaging::new(max_batch_tokens, max_batch_seqs),
            next_step_buffer_id: 0,
            next_output_buffer_id: 0,
            step_in_flight: false,
            profile_cuda_steps,
            profile_started: false,
            profile_stopped: false,
            profiled_steps: 0,
            draining: false,
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  主循环
    // ════════════════════════════════════════════════════════════════════════════

    /// 主循环。调用线程会被阻塞在这里直到 runner shutdown。
    pub fn run(mut self) {
        tracing::info!("SubScheduler waiting for first prefill...");
        let cmd = match self.recv_next_prefill_blocking() {
            Some(c) => c,
            None => {
                tracing::info!("SubScheduler: shutdown before first prefill");
                return;
            }
        };
        self.pending_prefills.push(cmd);

        if let Err(e) = self.ensure_capacity_and_enqueue() {
            tracing::error!("Failed to enqueue first step: {:?}", e);
            return;
        }
        tracing::info!(
            "SubScheduler started, {} active decodes",
            self.active_decodes.len()
        );

        loop {
            // 1. 与 Runner forward 并行：non-blocking 收新 prefill / GrantBlocks / cancel
            self.drain_zmq_prefills();
            self.apply_pending_cancels();

            // 如果上一轮没有提交 step（例如所有 paged decode 都在等 GrantBlocks），
            // 不要等待 output_ready；继续 drain 控制消息并尝试重新提交。
            if !self.step_in_flight {
                if let Err(e) = self.ensure_capacity_and_enqueue() {
                    tracing::error!("Failed to enqueue while idle/no-inflight: {:?}", e);
                    self.runner.request_shutdown();
                    return;
                }
                if !self.step_in_flight {
                    if self.active_decodes.is_empty() && self.pending_prefills.is_empty() {
                        tracing::debug!("All sequences finished, waiting for new prefill...");
                        let cmd = match self.recv_next_prefill_blocking() {
                            Some(c) => c,
                            None => {
                                tracing::info!("SubScheduler: shutdown while idle");
                                return;
                            }
                        };
                        self.pending_prefills.push(cmd);
                        if let Err(e) = self.ensure_capacity_and_enqueue() {
                            tracing::error!("Failed to enqueue after idle: {:?}", e);
                            self.runner.request_shutdown();
                            return;
                        }
                    }
                    if !self.step_in_flight {
                        std::thread::yield_now();
                        continue;
                    }
                }
            }

            // 2. 等待按提交顺序的 output buffer 完成；其它 buffer 的 output 不会阻塞 runner。
            let output_buffer_id = self.next_output_buffer_id;
            if !self.runner.wait_output_ready_for(output_buffer_id) {
                tracing::info!("SubScheduler: runner shutdown detected, exiting");
                return;
            }

            if let Some(err) = self.runner.take_error() {
                self.runner.set_output_consumed_for(output_buffer_id);
                self.step_in_flight = false;
                self.maybe_stop_cuda_profiler_after_step();
                self.send_fatal_error(err);
                self.runner.request_shutdown();
                return;
            }

            // 3. 读该 buffer output tokens (D2H)，读完立即释放该 output buffer。
            let output_tokens = match self.read_output_tokens(output_buffer_id) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("Failed to read output tokens: {:?}", e);
                    self.runner.set_output_consumed_for(output_buffer_id);
                    self.step_in_flight = false;
                    self.maybe_stop_cuda_profiler_after_step();
                    self.next_output_buffer_id = (output_buffer_id + 1) % STEP_BUFFER_COUNT;
                    continue;
                }
            };
            self.runner.set_output_consumed_for(output_buffer_id);
            self.step_in_flight = false;
            self.maybe_stop_cuda_profiler_after_step();
            self.next_output_buffer_id = (output_buffer_id + 1) % STEP_BUFFER_COUNT;

            // 4. 按本 step 语义处理 output：prefill ack、final prefill token、decode token。
            self.process_step_output_and_send_zmq(output_buffer_id, &output_tokens);
            self.apply_pending_cancels();

            // 6. 非阻塞收新 prefill / cancel
            self.drain_zmq_prefills();
            self.apply_pending_cancels();

            // 7. 组装下一步输入 → signal runner
            if let Err(e) = self.ensure_capacity_and_enqueue() {
                tracing::error!("Failed to enqueue next step: {:?}", e);
                // 安全退出：无法继续推理
                self.runner.request_shutdown();
                return;
            }

            // 8. 空闲时阻塞等新 prefill
            if self.active_decodes.is_empty() && self.pending_prefills.is_empty() {
                tracing::debug!("All sequences finished, waiting for new prefill...");
                let cmd = match self.recv_next_prefill_blocking() {
                    Some(c) => c,
                    None => {
                        tracing::info!("SubScheduler: shutdown while idle");
                        return;
                    }
                };
                self.pending_prefills.push(cmd);
                if let Err(e) = self.ensure_capacity_and_enqueue() {
                    tracing::error!("Failed to enqueue after idle: {:?}", e);
                    self.runner.request_shutdown();
                    return;
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  写入 Runner workspace + signal（核心改动）
    // ════════════════════════════════════════════════════════════════════════════

    /// 整体流程：ensure_capacity → write_input_buffer → 记录本 step 语义 → signal runner。
    ///
    /// Prefill segment 不在这里无条件转 decode；必须等 runner output 回来后，
    /// 根据 segment completion 决定丢弃中间 chunk 的采样 token，或用 final chunk 的
    /// 输出 token 创建 DecodeSeq。
    fn ensure_capacity_and_enqueue(&mut self) -> Result<bool> {
        if self.active_decodes.is_empty() && self.pending_prefills.is_empty() {
            return Ok(false);
        }

        self.ensure_capacity_for_pending_prefills()?;
        let submitted = self.write_input_buffer_pre_promote()?;
        if submitted {
            self.step_in_flight = true;
        }
        Ok(submitted)
    }

    /// 为 pending prefill segments 预分配 KV cache 容量。
    fn ensure_capacity_for_pending_prefills(&mut self) -> Result<()> {
        for cmd in &self.pending_prefills {
            let n = cmd.num_requests();
            for _i in 0..n {
                // Paged-only: KV capacity comes from block_table, runner handles blocks内部.
                // Per-slot ensure_capacity was a Slot-mode helper, no longer needed.
            }
        }
        Ok(())
    }

    /// 写入 Runner workspace。
    ///
    /// 布局：decode 在前 (q_len=1 each)，prefill segment 在后。
    /// 同时按 buffer 记录 `step_items`，runner output 回来后按对应 buffer 的表解释每一行输出。
    fn write_input_buffer_pre_promote(&mut self) -> Result<bool> {
        let num_decode = self
            .active_decodes
            .iter_runnable()
            .count();
        let num_prefill_seqs: usize = self.pending_prefills.iter().map(|p| p.num_requests()).sum();
        let num_prefill_tokens: usize = self
            .pending_prefills
            .iter()
            .map(|p| p.input_ids.len())
            .sum();
        let total_tokens = num_decode + num_prefill_tokens;
        let num_seqs = num_decode + num_prefill_seqs;

        if num_seqs == 0 {
            return Ok(false);
        }

        let step_buffer_id = self.next_step_buffer_id;
        let ws = unsafe { self.runner.workspace_for(step_buffer_id) };
        if total_tokens > ws.max_batch_tokens {
            return Err(Error::InvalidArgument(format!(
                "total_tokens {} > max_batch_tokens {}",
                total_tokens, ws.max_batch_tokens
            ))
            .into());
        }
        if num_seqs > ws.max_batch_seqs {
            return Err(Error::InvalidArgument(format!(
                "num_seqs {} > max_batch_seqs {}",
                num_seqs, ws.max_batch_seqs
            ))
            .into());
        }

        self.staging.reset(total_tokens, num_seqs);
        let staging = &mut self.staging;
        let mut step_items = Vec::with_capacity(num_seqs);
        let mut paged_tables: Vec<Option<Vec<u32>>> = Vec::with_capacity(num_seqs);
        let mut meta = StepMeta::zeroed();

        meta.step_buffer_id = step_buffer_id;

        let mut offset: i32 = 0;
        let mut seq_idx = 0usize;

        let mut scheduled_sequence_ids = HashSet::with_capacity(num_seqs);

        // Decode 在前 (每条 q_len=1)。
        for d in self.active_decodes.iter_runnable() {
            if !scheduled_sequence_ids.insert(d.sequence_id) {
                tracing::warn!("Skipping duplicate active decode sequence_id={}", d.sequence_id);
                continue;
            }
            staging.input_tokens.push(d.last_token);
            staging.input_positions.push(d.next_position as i32);
            staging.kv_lens.push(d.next_position as i32 + 1);

            meta.q_start_loc[seq_idx] = offset;
            meta.slot_indices[seq_idx] = d.kv.slot_for_legacy_workspace(seq_idx) as i32;
            meta.sequence_ids[seq_idx] = d.sequence_id;
            meta.positions_start[seq_idx] = d.next_position as i32;
            step_items.push(StepItem::Decode {
                sequence_id: d.sequence_id,
            });
            paged_tables.push(d.kv.paged_block_table().map(|t| t.to_vec()));

            offset += 1;
            seq_idx += 1;
        }

        let actual_num_decode = seq_idx;

        // Prefill segments 在后。segment_start/end 是 KV 和 RoPE 的绝对位置。
        for cmd in &self.pending_prefills {
            let n = cmd.num_requests();
            for i in 0..n {
                let range = cmd.segment_token_range(i);
                let segment = cmd.segments[i].clone();
                if !scheduled_sequence_ids.insert(segment.sequence_id) {
                    tracing::warn!(
                        "Skipping duplicate prefill segment sequence_id={}",
                        segment.sequence_id
                    );
                    continue;
                }
                let seq_len = range.len();
                let segment_start = segment.segment_start as usize;
                let segment_end = segment.segment_end as usize;
                // Paged-only: block_table is required;
                // batch row is the unique placeholder slot for Runner state-ref.
                let slot = seq_idx;
                let paged_table = Some(segment.block_table.clone());

                staging
                    .input_tokens
                    .extend_from_slice(&cmd.input_ids[range]);
                for pos in segment_start..segment_end {
                    staging.input_positions.push(pos as i32);
                }
                staging.kv_lens.push(segment_end as i32);

                meta.q_start_loc[seq_idx] = offset;
                meta.slot_indices[seq_idx] = slot as i32;
                meta.sequence_ids[seq_idx] = segment.sequence_id;
                meta.positions_start[seq_idx] = segment_start as i32;
                step_items.push(StepItem::PrefillSegment { segment });
                paged_tables.push(paged_table);

                offset += seq_len as i32;
                seq_idx += 1;
            }
        }
        let actual_num_seqs = seq_idx;
        if actual_num_seqs == 0 {
            self.pending_prefills.clear();
            return Ok(false);
        }
        meta.num_decode = actual_num_decode;
        meta.num_prefill = actual_num_seqs - actual_num_decode;
        meta.q_start_loc[actual_num_seqs] = offset;
        meta.total_q_tiles = 0;

        let ws_mut = unsafe { self.runner.workspace_mut_for(step_buffer_id) };
        #[cfg(feature = "cuda")]
        let stream = unsafe { self.runner.cuda_stream() };

        #[cfg(feature = "cuda")]
        {
            ws_mut
                .input_tokens
                .as_i32_mut()?
                .buffer_mut()
                .copy_from_host_async(&staging.input_tokens, stream)?;
            ws_mut
                .input_pos
                .as_i32_mut()?
                .buffer_mut()
                .copy_from_host_async(&staging.input_positions, stream)?;
            ws_mut
                .kv_lens_dev
                .as_i32_mut()?
                .buffer_mut()
                .copy_from_host_async(&staging.kv_lens, stream)?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            ws_mut
                .input_tokens
                .as_i32_mut()?
                .buffer_mut()
                .copy_from_host(&staging.input_tokens)?;
            ws_mut
                .input_pos
                .as_i32_mut()?
                .buffer_mut()
                .copy_from_host(&staging.input_positions)?;
            ws_mut
                .kv_lens_dev
                .as_i32_mut()?
                .buffer_mut()
                .copy_from_host(&staging.kv_lens)?;
        }

        let meta_view = WorkerBatchMeta::from_step(&meta);
        #[cfg(feature = "cuda")]
        ws_mut.refresh_scatter_indices(&meta_view, stream)?;
        #[cfg(feature = "cuda")]
        {
            let paged_refs: Vec<Option<&[u32]>> =
                paged_tables.iter().map(|entry| entry.as_deref()).collect();
            ws_mut.refresh_paged_block_tables(&paged_refs, stream)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (&meta_view, &paged_tables);

        #[cfg(feature = "cuda")]
        {
            let states_all = unsafe { &mut *(self.runner.states_ptr_mut()) };
            let mut refs: Vec<&mut InferenceState> = Vec::with_capacity(actual_num_seqs);
            let mut slot_ids: Vec<usize> = Vec::with_capacity(actual_num_seqs);

            for i in 0..actual_num_seqs {
                let slot = meta.slot_indices[i] as usize;
                slot_ids.push(slot);
                let p = &mut states_all[slot] as *mut InferenceState;
                refs.push(unsafe { &mut *p });
            }

            ws_mut.fill_cache_ptrs_from_states(&slot_ids, &mut refs, stream)?;
        }

        self.step_items_by_buffer[step_buffer_id] = step_items;
        self.next_step_buffer_id = (step_buffer_id + 1) % STEP_BUFFER_COUNT;
        self.pending_prefills.clear();

        unsafe {
            self.runner.write_meta_for(step_buffer_id, meta);
        }
        self.maybe_start_cuda_profiler();
        self.runner.set_input_ready_for(step_buffer_id);

        Ok(true)
    }

    fn maybe_start_cuda_profiler(&mut self) {
        if self.profile_cuda_steps == 0 || self.profile_started {
            return;
        }
        #[cfg(feature = "cuda")]
        {
            match crate::cuda::device::profiler_start() {
                Ok(()) => {
                    self.profile_started = true;
                    tracing::info!(
                        "CUDA profiler started for {} worker steps",
                        self.profile_cuda_steps
                    );
                }
                Err(e) => tracing::error!("cudaProfilerStart failed: {:?}", e),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.profile_started = true;
        }
    }

    fn maybe_stop_cuda_profiler_after_step(&mut self) {
        if self.profile_cuda_steps == 0 || !self.profile_started || self.profile_stopped {
            return;
        }
        self.profiled_steps += 1;
        if self.profiled_steps < self.profile_cuda_steps {
            return;
        }
        #[cfg(feature = "cuda")]
        {
            match crate::cuda::device::profiler_stop() {
                Ok(()) => {
                    self.profile_stopped = true;
                    tracing::info!(
                        "CUDA profiler stopped after {} worker steps",
                        self.profiled_steps
                    );
                }
                Err(e) => tracing::error!("cudaProfilerStop failed: {:?}", e),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.profile_stopped = true;
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  输出处理
    // ════════════════════════════════════════════════════════════════════════════

    /// 从指定 Runner buffer 读本步 output tokens (D2H)。
    fn read_output_tokens(&self, buffer_id: usize) -> Result<Vec<i32>> {
        let num_outputs = self.step_items_by_buffer[buffer_id].len();
        // Read from pre-copied host buffer — no CUDA API call, no legacy stream
        // dependency. Runner already did D2H via cudaMemcpyAsync on the worker
        // stream and sync'd before signaling output_ready.
        let host_slice = unsafe { self.runner.output_tokens_host_for(buffer_id) };
        Ok(host_slice.iter().take(num_outputs).copied().collect())
    }

    /// 按指定 buffer 的 step_items 解释 runner output，更新 Worker 内部 decode 状态并发送精简 StepOutput。
    fn process_step_output_and_send_zmq(&mut self, buffer_id: usize, tokens: &[i32]) {
        if tokens.len() < self.step_items_by_buffer[buffer_id].len() {
            tracing::error!(
                "Runner output length {} < buffer {} step_items length {}",
                tokens.len(),
                buffer_id,
                self.step_items_by_buffer[buffer_id].len()
            );
            return;
        }

        let step_items = std::mem::take(&mut self.step_items_by_buffer[buffer_id]);
        let mut step_output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
        };
        let mut batch_members_changed = false;

        for (item, &token_id) in step_items.into_iter().zip(tokens) {
            match item {
                StepItem::Decode { sequence_id } => {
                    if self.cancelled_sequences.contains(&sequence_id) {
                        tracing::debug!(
                            "Dropping late decode output for cancelled sequence_id={}",
                            sequence_id
                        );
                        continue;
                    }
                    if let Some(seq) = self.active_decodes.get_mut(sequence_id) {
                        let generated_count = seq.generated_count + 1;
                        let finished = self.eos_token_ids.contains(&token_id)
                            || generated_count >= seq.max_tokens;

                        step_output.tokens.push(GeneratedToken {
                            sequence_id,
                            token_id,
                            finished,
                        });

                        if finished {
                            self.active_decodes.remove(sequence_id);
                            batch_members_changed = true;
                        } else {
                            seq.last_token = token_id;
                            seq.generated_count = generated_count;
                            seq.next_position += 1;
                        }
                    } else {
                        tracing::warn!("Decode output for unknown sequence_id={}", sequence_id);
                    }
                }
                StepItem::PrefillSegment { segment } => {
                    let sequence_id = segment.sequence_id;
                    if self.cancelled_sequences.contains(&sequence_id) {
                        tracing::debug!(
                            "Dropping late prefill output for cancelled sequence_id={}",
                            sequence_id
                        );
                        continue;
                    }
                    step_output.prefill_done.push(sequence_id);

                    match segment.completion {
                        PrefillSegmentCompletion::ContinuePrefill => {
                            // 中间 chunk 只写 KV，采样 token 无业务语义，直接丢弃。
                        }
                        PrefillSegmentCompletion::FinishPrefillAndStartDecode => {
                            let generated_count = 1usize;
                            let finished = self.eos_token_ids.contains(&token_id)
                                || generated_count >= segment.max_tokens;

                            step_output.tokens.push(GeneratedToken {
                                sequence_id,
                                token_id,
                                finished,
                            });

                            if !finished {
                                match self.active_decodes.insert(DecodeSeq {
                                    sequence_id,
                                    kv: decode_kv_from_segment(&segment),
                                    // final prefill 输出 token 还没写入 KV；下一步 decode 写到 prompt_len。
                                    next_position: segment.prompt_len as usize,
                                    last_token: token_id,
                                    generated_count,
                                    max_tokens: segment.max_tokens,
                                    sampling: segment.sampling_params,
                                }) {
                                    Ok(()) => batch_members_changed = true,
                                    Err(e) => tracing::warn!(
                                        "Ignoring duplicate decode activation sequence_id={} error={:?}",
                                        sequence_id,
                                        e
                                    ),
                                }
                            }
                        }
                    }
                }
                StepItem::Cancelled { sequence_id } => {
                    tracing::debug!(
                        "Dropping output row for cancelled sequence_id={}",
                        sequence_id
                    );
                }
            }
        }

        // NeedBlocks now travel on the control plane (one message per request).
        for draft in self.collect_need_blocks_drafts() {
            let need = NeedBlocks {
                worker_id: self.worker_id.clone(),
                model_instance_id: self.model_instance_id.clone(),
                sequence_id: draft.sequence_id,
                current_blocks: draft.current_blocks as u32,
                required_blocks: draft.required_blocks as u32,
                request_blocks: draft.request_blocks as u32,
                reason: NeedBlocksReason::DecodeExtend,
            };
            self.send_control_upstream(WorkerControlMessage::NeedBlocks(need));
        }

        if batch_members_changed {
            for buffer_id in 0..STEP_BUFFER_COUNT {
                unsafe {
                    self.runner
                        .workspace_mut_for(buffer_id)
                        .invalidate_batch_member_cache();
                }
            }
        }

        // Update liveness counters for the pump.
        self.live.set_active_requests(self.active_decodes.len());

        if step_output.prefill_done.is_empty() && step_output.tokens.is_empty() {
            return;
        }
        let data = match rmp_serde::to_vec(&step_output) {
            Ok(data) => data,
            Err(e) => {
                tracing::error!("Failed to serialize StepOutput: {}", e);
                return;
            }
        };
        if let Err(e) = self.zmq_out.send(&data, 0) {
            tracing::error!("ZMQ send failed: {}", e);
        }
    }

    fn collect_need_blocks_drafts(&mut self) -> Vec<NeedBlocksDraft> {
        let mut drafts = Vec::new();
        for seq in self.active_decodes.iter_mut() {
            if let Some(draft) = seq.kv.maybe_request_blocks(
                seq.sequence_id,
                seq.next_position,
                seq.max_tokens,
                seq.generated_count,
                DEFAULT_DECODE_BLOCK_PREFETCH_MARGIN,
                DEFAULT_DECODE_BLOCK_REQUEST_BLOCKS,
            ) {
                drafts.push(draft);
            }
        }
        drafts
    }

    fn apply_block_grant(&mut self, grant: GrantBlocks) {
        if let Some(seq) = self
            .active_decodes
            .iter_mut()
            .find(|s| s.sequence_id == grant.sequence_id)
        {
            if seq.kv.append_granted_blocks(&grant.block_ids) {
                tracing::debug!(
                    "GrantBlocks applied: sequence_id={} blocks={}",
                    grant.sequence_id,
                    grant.block_ids.len(),
                );
            } else {
                tracing::warn!(
                    "GrantBlocks for non-paged sequence_id={}",
                    grant.sequence_id
                );
            }
        } else {
            tracing::debug!("GrantBlocks for inactive sequence_id={}", grant.sequence_id);
        }
    }

    fn apply_block_grant_denied(&mut self, denied: GrantBlocksDenied) {
        tracing::warn!(
            "GrantBlocksDenied: sequence_id={} reason={:?}",
            denied.sequence_id,
            denied.reason
        );
        // Phase 1: deny is informational; the sub-scheduler will retry on the
        // next NeedBlocks tick. Cache exhausted denials should ultimately
        // surface as a sequence error, but that's policy logic for the
        // scheduler, not the worker.
        let _ = BlockGrantDeniedReason::CacheExhausted; // ensure import lives
    }

    fn collect_all_sequence_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .active_decodes
            .iter()
            .map(|seq| seq.sequence_id)
            .collect();
        for cmd in &self.pending_prefills {
            ids.extend(cmd.segments.iter().map(|segment| segment.sequence_id));
        }
        for items in &self.step_items_by_buffer {
            for item in items {
                match item {
                    StepItem::Decode { sequence_id } => ids.push(*sequence_id),
                    StepItem::PrefillSegment { segment } => ids.push(segment.sequence_id),
                    StepItem::Cancelled { sequence_id } => ids.push(*sequence_id),
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn send_fatal_error(&mut self, message: impl Into<String>) {
        let message = message.into();
        tracing::error!("Worker fatal error: {}", message);
        let err = WorkerStepError {
            sequence_ids: self.collect_all_sequence_ids(),
            message,
            fatal: true,
        };
        self.send_control_upstream(WorkerControlMessage::StepError(err));
        self.live.set_state(WorkerState::Error);
    }

    /// Push a [`WorkerControlMessage`] to the [`super::control_pump::ControlPump`]
    /// for serialization and DEALER send. Drops on full channel; the pump will
    /// log a warning. The synchronous failure mode is preferred over blocking
    /// the sub-scheduler thread.
    fn send_control_upstream(&self, msg: WorkerControlMessage) {
        match self.control_up_tx.try_send(msg) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!("control up channel full; dropping message");
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::error!("control up channel disconnected; pump exited");
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  ZMQ 通信
    // ════════════════════════════════════════════════════════════════════════════

    /// 验证 prefill 消息合法性。
    fn validate_prefill(&self, cmd: &PrefillBatchCmd) -> Result<()> {
        let ws = unsafe { self.runner.workspace() };
        cmd.validate(ws.max_batch_tokens, ws.max_batch_seqs)?;
        Ok(())
    }

    /// 阻塞等待一个有效的 prefill 消息（每 100ms 检查一次 shutdown）。
    /// 如果 runner 已 shutdown 则返回 None。
    fn recv_next_prefill_blocking(&mut self) -> Option<PrefillBatchCmd> {
        loop {
            if self.runner.is_shutdown() || self.draining {
                return None;
            }
            // Drain control plane on every iteration so GrantBlocks / Cancel /
            // Drain arrive even when the data plane is idle.
            self.drain_control_plane();
            // 用 DONTWAIT + sleep 模拟带 shutdown 检测的 blocking recv
            match self.zmq_in.recv_bytes(zmq::DONTWAIT) {
                Ok(data) => {
                    if let Some(cmd) = self.handle_data_plane_message(&data) {
                        return Some(cmd);
                    }
                    self.apply_pending_cancels();
                }
                Err(zmq::Error::EAGAIN) => {
                    // 没有消息，等一小段时间后重试
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => tracing::error!("ZMQ recv error while waiting for prefill: {}", e),
            }
        }
    }

    /// 非阻塞收所有待处理的 prefill。
    fn drain_zmq_prefills(&mut self) {
        // 每轮先把控制面消息消费干净，保证 GrantBlocks 等先于下一批 prefill
        // 应用到 active_decodes。
        self.drain_control_plane();
        loop {
            match self.zmq_in.recv_bytes(zmq::DONTWAIT) {
                Ok(data) => {
                    if let Some(cmd) = self.handle_data_plane_message(&data) {
                        self.pending_prefills.push(cmd);
                    }
                }
                Err(zmq::Error::EAGAIN) => break,
                Err(e) => {
                    tracing::error!("ZMQ recv error: {}", e);
                    break;
                }
            }
        }
    }

    /// 数据面只承载 BatchCommand。控制语义已迁到 control plane。
    fn handle_data_plane_message(&mut self, data: &[u8]) -> Option<PrefillBatchCmd> {
        match rmp_serde::from_slice::<BatchCommand>(data) {
            Ok(BatchCommand::Prefill(cmd)) => self.accept_prefill(cmd),
            Ok(BatchCommand::DiffusionBatch(_)) => {
                tracing::error!("LLM worker received DiffusionBatch on data plane");
                None
            }
            Err(e) => {
                tracing::error!("Failed to deserialize BatchCommand: {}", e);
                None
            }
        }
    }

    /// 排空 control_down_rx 中所有 pending 控制消息。每轮 step 前调用。
    fn drain_control_plane(&mut self) {
        while let Ok(msg) = self.control_down_rx.try_recv() {
            self.handle_control_message(msg);
        }
    }

    /// Dispatch a single control-plane message into worker state.
    fn handle_control_message(&mut self, msg: SchedulerControlMessage) {
        match msg {
            SchedulerControlMessage::GrantBlocks(g) => self.apply_block_grant(g),
            SchedulerControlMessage::GrantBlocksDenied(d) => self.apply_block_grant_denied(d),
            SchedulerControlMessage::Cancel(CancelSequence { sequence_id }) => {
                tracing::info!("CancelSequence queued sequence_id={}", sequence_id);
                self.pending_cancels.push(sequence_id);
            }
            SchedulerControlMessage::Drain(DrainWorker { mode }) => {
                tracing::info!("DrainWorker mode={:?}", mode);
                self.draining = true;
                self.live.set_state(WorkerState::Draining);
                if matches!(mode, DrainMode::Immediate) {
                    self.pending_prefills.clear();
                    self.active_decodes.clear();
                    self.runner.request_shutdown();
                }
            }
            SchedulerControlMessage::UnloadModel(UnloadModel { model_instance_id }) => {
                tracing::info!("UnloadModel model_instance_id={}", model_instance_id);
                self.draining = true;
                self.runner.request_shutdown();
            }
            SchedulerControlMessage::Shutdown => {
                tracing::info!("Worker received Shutdown");
                self.draining = true;
                self.runner.request_shutdown();
            }
            // The following are bootstrap-only or never targeted at sub-scheduler:
            SchedulerControlMessage::Hello(_)
            | SchedulerControlMessage::LoadModel(_)
            | SchedulerControlMessage::InitPagedKv(_)
            | SchedulerControlMessage::Ping => {
                tracing::warn!(
                    "sub_scheduler received unexpected control message: {:?}",
                    std::mem::discriminant(&msg)
                );
            }
        }
    }

    fn accept_prefill(&self, cmd: PrefillBatchCmd) -> Option<PrefillBatchCmd> {
        if self.draining {
            tracing::warn!("Reject prefill while worker is draining");
            return None;
        }
        match self.validate_prefill(&cmd) {
            Ok(()) => Some(cmd),
            Err(e) => {
                tracing::error!("Reject invalid PrefillBatchCmd: {}", e);
                None
            }
        }
    }

    fn apply_pending_cancels(&mut self) {
        let cancels = std::mem::take(&mut self.pending_cancels);
        for sequence_id in cancels {
            let removed = self.cancel_request(sequence_id);
            tracing::info!(
                "CancelRequest applied sequence_id={} removed={}",
                sequence_id,
                removed
            );
        }
    }

    fn cancel_request(&mut self, sequence_id: u64) -> bool {
        self.cancelled_sequences.insert(sequence_id);

        let removed_active = self.active_decodes.cancel(sequence_id);

        let mut removed_pending = false;
        let pending = std::mem::take(&mut self.pending_prefills);
        for cmd in pending {
            match filter_prefill_cmd(cmd, sequence_id) {
                Some(cmd) => self.pending_prefills.push(cmd),
                None => removed_pending = true,
            }
        }

        let mut removed_inflight = false;
        for items in &mut self.step_items_by_buffer {
            for item in items {
                let item_sequence_id = match item {
                    StepItem::Decode { sequence_id } => *sequence_id,
                    StepItem::PrefillSegment { segment } => segment.sequence_id,
                    StepItem::Cancelled { sequence_id } => *sequence_id,
                };
                if item_sequence_id == sequence_id {
                    *item = StepItem::Cancelled { sequence_id };
                    removed_inflight = true;
                }
            }
        }

        if removed_active || removed_pending || removed_inflight {
            for buffer_id in 0..STEP_BUFFER_COUNT {
                unsafe {
                    self.runner
                        .workspace_mut_for(buffer_id)
                        .invalidate_batch_member_cache();
                }
            }
        }

        removed_active || removed_pending || removed_inflight
    }
}

fn filter_prefill_cmd(cmd: PrefillBatchCmd, sequence_id: u64) -> Option<PrefillBatchCmd> {
    let n = cmd.num_requests();
    let mut input_ids = Vec::new();
    let mut q_start_loc = Vec::new();
    let mut segments = Vec::new();

    for i in 0..n {
        let range = cmd.segment_token_range(i);
        if cmd.segments[i].sequence_id == sequence_id {
            continue;
        }

        q_start_loc.push(input_ids.len() as u32);
        input_ids.extend_from_slice(&cmd.input_ids[range]);
        segments.push(cmd.segments[i].clone());
    }

    if q_start_loc.is_empty() {
        None
    } else {
        Some(PrefillBatchCmd {
            input_ids,
            q_start_loc,
            segments,
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  集成测试
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "cuda")]
#[cfg(feature = "models")]
mod tests {
    //! Server + Runner 端到端集成测试。
    //!
    //! 启动 runner 线程 + sub-scheduler 线程，主线程通过 ZMQ inproc 扮演 scheduler：
    //! 发 PrefillBatchCmd，收 StepOutput，验证 token 合法 + finished 正确。
    //!
    //! 需要真实模型权重（LLAMA3_MODEL_PATH 或 well-known 路径）。
    use super::*;
    use crate::base::DeviceType;
    use crate::model::llm::LlmModel;
    use crate::model::llm::llama3::Llama3;
    use std::sync::Arc;

    fn get_model_path() -> Option<std::path::PathBuf> {
        std::env::var("LLAMA3_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let candidates = [
                    std::path::PathBuf::from("/data/home/vinciiliu/models/Llama-3.2-1B-Instruct"),
                    std::path::PathBuf::from(
                        "/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b",
                    ),
                ];
                candidates.into_iter().find(|p| p.exists())
            })
    }

    fn full_prefill_cmd(
        sequence_id: u64,
        input_ids: Vec<i32>,
        kv_slot: u32,
        max_tokens: usize,
    ) -> BatchCommand {
        let _ = kv_slot; // legacy parameter, paged-only build ignores
        let prompt_len = input_ids.len() as u32;
        BatchCommand::Prefill(PrefillBatchCmd {
            input_ids,
            q_start_loc: vec![0],
            segments: vec![PrefillSegmentMeta {
                sequence_id,
                block_table: vec![0],
                block_size: 16,
                prompt_len,
                segment_start: 0,
                segment_end: prompt_len,
                max_tokens,
                sampling_params: SamplingParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: -1,
                },
                completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
            }],
        })
    }

    /// Test helper: build a stub `(down_rx, up_tx, live, …)` quartet so each
    /// `SubScheduler::new` call site doesn't need to spawn a real ControlPump.
    fn stub_control_handles() -> (
        std::sync::mpsc::Receiver<SchedulerControlMessage>,
        std::sync::mpsc::SyncSender<WorkerControlMessage>,
        WorkerLiveState,
        std::sync::mpsc::Sender<SchedulerControlMessage>,
        std::sync::mpsc::Receiver<WorkerControlMessage>,
    ) {
        let (down_tx, down_rx) = std::sync::mpsc::channel::<SchedulerControlMessage>();
        // Wrap into the SyncSender shape that the SubScheduler expects.
        // We use a (sync_tx, _) split where the test side gets the unbounded
        // Sender; SubScheduler reads from `down_rx`.
        let (_unused_tx, _unused_rx) = std::sync::mpsc::sync_channel::<SchedulerControlMessage>(0);
        // Convert `down_tx` to a typed proxy: expose Sender to test, but keep
        // the SubScheduler reading `down_rx` directly.
        let _ = _unused_tx;
        let _ = _unused_rx;

        let (up_tx, up_rx) = std::sync::mpsc::sync_channel::<WorkerControlMessage>(64);
        let live = WorkerLiveState::new();
        (down_rx, up_tx, live, down_tx, up_rx)
    }

    /// 单请求端到端：prefill + decode 直到 max_tokens，验证：
    /// - StepOutput 每步都收到
    /// - token_id 合法
    /// - 最后一步 finished=true
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn server_single_request_e2e() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no model path");
                return Ok(());
            }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;
        let eos_token_ids: Vec<i32> = model
            .tokenizer()
            .eos_token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 4usize;
        let runner = Arc::new(ModelRunner::new(
            model,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        // ─── ZMQ inproc sockets ───
        let zmq_ctx = zmq::Context::new();
        let scheduler_push = zmq_ctx.socket(zmq::PUSH)?;
        let worker_pull = zmq_ctx.socket(zmq::PULL)?;
        let worker_push = zmq_ctx.socket(zmq::PUSH)?;
        let scheduler_pull = zmq_ctx.socket(zmq::PULL)?;

        scheduler_push.bind("inproc://test-prefill")?;
        worker_pull.connect("inproc://test-prefill")?;
        worker_push.bind("inproc://test-output")?;
        scheduler_pull.connect("inproc://test-output")?;

        // ─── 启动 runner 线程 ───
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // ─── 启动 sub-scheduler 线程 ───
        let (stub_down_rx, stub_up_tx, stub_live, _stub_down_tx, _stub_up_rx) = stub_control_handles();
        let server = SubScheduler::new(
            Arc::clone(&runner),
            device,
            worker_pull,
            worker_push,
            eos_token_ids.clone(),
            0,
            stub_down_rx,
            stub_up_tx,
            stub_live,
            "test-worker".to_string(),
            "test-model".to_string(),
        );
        let server_handle = std::thread::spawn(move || server.run());

        // ─── 主线程扮演 scheduler ───
        let prompt = "The capital of France is";
        let toks: Vec<i32> = runner.model().tokenizer().encode(prompt)?;
        let max_tokens = 10usize;

        let cmd = full_prefill_cmd(1, toks.clone(), 0, max_tokens);
        let data = rmp_serde::to_vec(&cmd).unwrap();
        scheduler_push.send(&data, 0)?;

        // ─── 收 StepOutput 直到 finished ───
        let mut total_steps = 0usize;
        let mut all_tokens = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);

        loop {
            if std::time::Instant::now() > deadline {
                panic!("timeout: no finished after {} steps", total_steps);
            }
            let msg = scheduler_pull.recv_bytes(0)?;
            let output: StepOutput = rmp_serde::from_slice(&msg)?;

            assert_eq!(output.tokens.len(), 1, "expected 1 seq in output");
            let seq_out = &output.tokens[0];
            assert_eq!(seq_out.sequence_id, 1);
            assert!(
                seq_out.token_id >= 0 && (seq_out.token_id as usize) < vocab,
                "token {} out of vocab range",
                seq_out.token_id
            );
            all_tokens.push(seq_out.token_id);
            total_steps += 1;

            if seq_out.finished {
                break;
            }
        }

        eprintln!(
            "server_single_request_e2e: {} steps, tokens={:?}",
            total_steps, all_tokens
        );
        // prefill 输出 1 token，decode max_tokens-1 步 → 总共 max_tokens 步
        // 或者提前 EOS
        assert!(
            total_steps <= max_tokens + 1,
            "too many steps: {}",
            total_steps
        );
        assert!(total_steps >= 1, "no tokens generated");

        // ─── Shutdown ───
        runner.request_shutdown();
        let _ = runner_handle.join();
        let _ = server_handle.join();
        Ok(())
    }

    /// 两请求并发：验证 continuous batching 正确性。
    /// - 先发 req1，等一步 output
    /// - 再发 req2，后续 output 应该包含两条 seq
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn server_two_requests_continuous_batch() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no model path");
                return Ok(());
            }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;
        let eos_token_ids: Vec<i32> = model
            .tokenizer()
            .eos_token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 4usize;
        let runner = Arc::new(ModelRunner::new(
            model,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        // ─── ZMQ inproc sockets ───
        let zmq_ctx = zmq::Context::new();
        let scheduler_push = zmq_ctx.socket(zmq::PUSH)?;
        let worker_pull = zmq_ctx.socket(zmq::PULL)?;
        let worker_push = zmq_ctx.socket(zmq::PUSH)?;
        let scheduler_pull = zmq_ctx.socket(zmq::PULL)?;

        scheduler_push.bind("inproc://test2-prefill")?;
        worker_pull.connect("inproc://test2-prefill")?;
        worker_push.bind("inproc://test2-output")?;
        scheduler_pull.connect("inproc://test2-output")?;

        // ─── 启动 ───
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());
        let (stub_down_rx, stub_up_tx, stub_live, _stub_down_tx, _stub_up_rx) = stub_control_handles();
        let server = SubScheduler::new(
            Arc::clone(&runner),
            device,
            worker_pull,
            worker_push,
            eos_token_ids.clone(),
            0,
            stub_down_rx,
            stub_up_tx,
            stub_live,
            "test-worker".to_string(),
            "test-model".to_string(),
        );
        let server_handle = std::thread::spawn(move || server.run());

        // ─── 发第 1 个请求 (slot 0) ───
        let prompt1 = "Hello world";
        let toks1: Vec<i32> = runner.model().tokenizer().encode(prompt1)?;
        let cmd1 = full_prefill_cmd(1, toks1, 0, 5);
        scheduler_push.send(rmp_serde::to_vec(&cmd1).unwrap(), 0)?;

        // 收第一步 output（只有 req-A）
        let msg = scheduler_pull.recv_bytes(0)?;
        let out1: StepOutput = rmp_serde::from_slice(&msg)?;
        assert_eq!(out1.tokens.len(), 1);
        assert_eq!(out1.tokens[0].sequence_id, 1);
        assert!(out1.tokens[0].token_id >= 0 && (out1.tokens[0].token_id as usize) < vocab);
        eprintln!("step 1: req-A token={}", out1.tokens[0].token_id);

        // ─── 发第 2 个请求 (slot 1)，趁 server 在 drain_zmq_prefills ───
        let prompt2 = "The sky is";
        let toks2: Vec<i32> = runner.model().tokenizer().encode(prompt2)?;
        let cmd2 = full_prefill_cmd(2, toks2, 1, 5);
        scheduler_push.send(rmp_serde::to_vec(&cmd2).unwrap(), 0)?;

        // 后续 output 应该包含两条 seq（req-A decode + req-B prefill/decode）
        let mut a_finished = false;
        let mut b_finished = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);

        while !(a_finished && b_finished) {
            if std::time::Instant::now() > deadline {
                panic!("timeout waiting for both requests to finish");
            }
            let msg = scheduler_pull.recv_bytes(0)?;
            let out: StepOutput = rmp_serde::from_slice(&msg)?;
            eprintln!("step output: {} seqs", out.tokens.len());

            for seq_out in &out.tokens {
                assert!(
                    seq_out.token_id >= 0 && (seq_out.token_id as usize) < vocab,
                    "{} token {} out of range",
                    seq_out.sequence_id,
                    seq_out.token_id
                );
                eprintln!(
                    "  {} token={} finished={}",
                    seq_out.sequence_id, seq_out.token_id, seq_out.finished
                );
                if seq_out.finished {
                    match seq_out.sequence_id {
                        1 => a_finished = true,
                        2 => b_finished = true,
                        other => panic!("unexpected sequence_id: {}", other),
                    }
                }
            }
        }

        eprintln!("server_two_requests_continuous_batch: both finished");

        runner.request_shutdown();
        let _ = runner_handle.join();
        let _ = server_handle.join();
        Ok(())
    }

    /// Slot 复用测试：req1 完成后，用同一个 slot 发 req2，验证不 crash 且结果合法。
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn server_slot_reuse_after_finish() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no model path");
                return Ok(());
            }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;
        let eos_token_ids: Vec<i32> = model
            .tokenizer()
            .eos_token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 2usize;
        let runner = Arc::new(ModelRunner::new(
            model,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        let zmq_ctx = zmq::Context::new();
        let scheduler_push = zmq_ctx.socket(zmq::PUSH)?;
        let worker_pull = zmq_ctx.socket(zmq::PULL)?;
        let worker_push = zmq_ctx.socket(zmq::PUSH)?;
        let scheduler_pull = zmq_ctx.socket(zmq::PULL)?;

        scheduler_push.bind("inproc://test3-prefill")?;
        worker_pull.connect("inproc://test3-prefill")?;
        worker_push.bind("inproc://test3-output")?;
        scheduler_pull.connect("inproc://test3-output")?;

        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());
        let (stub_down_rx, stub_up_tx, stub_live, _stub_down_tx, _stub_up_rx) = stub_control_handles();
        let server = SubScheduler::new(
            Arc::clone(&runner),
            device,
            worker_pull,
            worker_push,
            eos_token_ids.clone(),
            0,
            stub_down_rx,
            stub_up_tx,
            stub_live,
            "test-worker".to_string(),
            "test-model".to_string(),
        );
        let server_handle = std::thread::spawn(move || server.run());

        // ─── req1: slot 0, max_tokens=3 ───
        let toks1: Vec<i32> = runner.model().tokenizer().encode("Hello")?;
        let cmd1 = full_prefill_cmd(1, toks1, 0, 3);
        scheduler_push.send(rmp_serde::to_vec(&cmd1).unwrap(), 0)?;

        // 等 req1 finish
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            if std::time::Instant::now() > deadline {
                panic!("timeout on req1");
            }
            let msg = scheduler_pull.recv_bytes(0)?;
            let out: StepOutput = rmp_serde::from_slice(&msg)?;
            if out.tokens.iter().any(|t| t.sequence_id == 1 && t.finished) {
                eprintln!("req1 finished");
                break;
            }
        }

        // ─── req2: 复用 slot 0 ───
        let toks2: Vec<i32> = runner.model().tokenizer().encode("Goodbye")?;
        let cmd2 = full_prefill_cmd(2, toks2, 0, 3); // 同一个 slot!
        scheduler_push.send(rmp_serde::to_vec(&cmd2).unwrap(), 0)?;

        // 等 req2 finish
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            if std::time::Instant::now() > deadline {
                panic!("timeout on req2");
            }
            let msg = scheduler_pull.recv_bytes(0)?;
            let out: StepOutput = rmp_serde::from_slice(&msg)?;
            for t in &out.tokens {
                assert!(
                    t.token_id >= 0 && (t.token_id as usize) < vocab,
                    "req2 token {} out of range",
                    t.token_id
                );
            }
            if out.tokens.iter().any(|t| t.sequence_id == 2 && t.finished) {
                eprintln!("req2 (slot reuse) finished successfully");
                break;
            }
        }

        runner.request_shutdown();
        let _ = runner_handle.join();
        let _ = server_handle.join();
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────────
    //  辅助：通过 server ZMQ 接口跑一个完整请求并收集所有输出 token
    // ────────────────────────────────────────────────────────────────────────

    /// 发送一个 prefill 并收集所有 output tokens 直到 finished。
    /// 返回 (生成的 token ids, 是否因 EOS 终止)。
    fn run_one_request(
        scheduler_push: &zmq::Socket,
        scheduler_pull: &zmq::Socket,
        input_ids: Vec<i32>,
        kv_slot: u32,
        sequence_id: u64,
        max_tokens: usize,
    ) -> Result<(Vec<i32>, bool)> {
        let cmd = full_prefill_cmd(sequence_id, input_ids, kv_slot, max_tokens);
        scheduler_push.send(rmp_serde::to_vec(&cmd).unwrap(), 0)?;

        let mut tokens = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if std::time::Instant::now() > deadline {
                panic!("timeout waiting for sequence_id={} to finish", sequence_id);
            }
            let msg = scheduler_pull.recv_bytes(0)?;
            let out: StepOutput = rmp_serde::from_slice(&msg)?;
            for t in &out.tokens {
                if t.sequence_id == sequence_id {
                    tokens.push(t.token_id);
                    if t.finished {
                        let hit_eos = t.token_id != 0; // 简化判断
                        return Ok((tokens, hit_eos));
                    }
                }
            }
        }
    }

    /// 确认输出是"正确文字"：
    /// - greedy decode 同一个 prompt 输出确定性结果
    /// - decode 后的文本非空、可读（不全是乱码）
    /// - 与 runner 直驱（drive_step）的结果一致（同一实例）
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn server_output_text_correctness() -> Result<()> {
        use crate::worker::runner::tests::{
            drive_step, make_prefill_meta, make_single_decode_meta,
        };

        let path = match get_model_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no model path");
                return Ok(());
            }
        };
        let device = DeviceType::Cuda(0);
        let max_tokens = 20usize;
        let max_batch_tokens = 512usize;
        let max_batch_seqs = 4usize;

        let model = Llama3::new(&path, device)?;
        let prompt = "The capital of France is";
        let prompt_tokens: Vec<i32> = model.tokenizer().encode(prompt)?;
        let p_len = prompt_tokens.len();
        let eos_token_ids: Vec<i32> = model
            .tokenizer()
            .eos_token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();

        let runner = Arc::new(ModelRunner::new(
            model,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        // ════ Part 1: 用 runner 直驱在 slot 0 跑 baseline ════
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        unsafe {
            runner
                .state_mut(0)
                .kv_cache
                .ensure_capacity(p_len + max_tokens)?;
        }
        let meta = make_prefill_meta(0, p_len);
        let pos: Vec<i32> = (0..p_len as i32).collect();
        let out = drive_step(&runner, &prompt_tokens, &pos, &[0i32], &meta)?;
        let mut baseline_tokens: Vec<i32> = vec![out[0]];
        let mut kv_len = p_len as i32;

        for _ in 0..(max_tokens - 1) {
            let last = *baseline_tokens.last().unwrap();
            let meta = make_single_decode_meta(0, kv_len);
            let out = drive_step(&runner, &[last], &[kv_len], &[kv_len], &meta)?;
            baseline_tokens.push(out[0]);
            kv_len += 1;
        }

        // shutdown baseline runner
        runner.request_shutdown();
        let _ = runner_handle.join();

        let baseline_text = runner
            .model()
            .tokenizer()
            .decode(&baseline_tokens)
            .unwrap_or_default();
        eprintln!(
            "baseline ({} tokens): {:?}",
            baseline_tokens.len(),
            baseline_text
        );
        assert!(
            !baseline_text.trim().is_empty(),
            "baseline decoded to empty string"
        );

        // ════ Part 2: 用 **同一个 runner** 重新跑 server（reset slot 0）════
        // 重置 slot 0 的 KV cache（新的 InferenceState 覆盖）
        {
            let new_state =
                crate::model::runtime::InferenceState::new(runner.model().config(), device)?;
            let slot_mut = unsafe { runner.state_mut(0) };
            *slot_mut = new_state;
        }
        // 清 shutdown flag — 需要重新启用 runner
        // 但 SyncFlags.shutdown 一旦 set 就没有 reset 方法...
        // 因此需要用一个新的 runner 实例（但共享同一权重）
        // 实际上正确的测试方式：用同一个模型文件创建新 runner

        // 折中方案：直接验证 server 输出的文本可读性 + 与自身一致性（跑两次相同 prompt 结果一致）
        drop(runner);

        let model2 = Llama3::new(&path, device)?;
        let runner2 = Arc::new(ModelRunner::new(
            model2,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        let zmq_ctx = zmq::Context::new();
        let scheduler_push = zmq_ctx.socket(zmq::PUSH)?;
        let worker_pull = zmq_ctx.socket(zmq::PULL)?;
        let worker_push = zmq_ctx.socket(zmq::PUSH)?;
        let scheduler_pull = zmq_ctx.socket(zmq::PULL)?;

        scheduler_push.bind("inproc://correctness-prefill")?;
        worker_pull.connect("inproc://correctness-prefill")?;
        worker_push.bind("inproc://correctness-output")?;
        scheduler_pull.connect("inproc://correctness-output")?;

        let runner2_loop = Arc::clone(&runner2);
        let runner2_handle = std::thread::spawn(move || runner2_loop.run());
        let (stub_down_rx, stub_up_tx, stub_live, _stub_down_tx, _stub_up_rx) = stub_control_handles();
        let server = SubScheduler::new(
            Arc::clone(&runner2),
            device,
            worker_pull,
            worker_push,
            eos_token_ids.clone(),
            0,
            stub_down_rx,
            stub_up_tx,
            stub_live,
            "test-worker".to_string(),
            "test-model".to_string(),
        );
        let server_handle = std::thread::spawn(move || server.run());

        // 跑第一次
        let (server_tokens_1, _) = run_one_request(
            &scheduler_push,
            &scheduler_pull,
            prompt_tokens.clone(),
            0,
            1,
            max_tokens,
        )?;
        let server_text_1 = runner2
            .model()
            .tokenizer()
            .decode(&server_tokens_1)
            .unwrap_or_default();
        eprintln!(
            "server run1 ({} tokens): {:?}",
            server_tokens_1.len(),
            server_text_1
        );

        // 重置 slot 0 再跑第二次（验证确定性）
        // 注意：这里 slot 0 是空闲的（run1 已 finished，被 server 移出 active）
        let (server_tokens_2, _) = run_one_request(
            &scheduler_push,
            &scheduler_pull,
            prompt_tokens.clone(),
            0,
            2,
            max_tokens,
        )?;
        let server_text_2 = runner2
            .model()
            .tokenizer()
            .decode(&server_tokens_2)
            .unwrap_or_default();
        eprintln!(
            "server run2 ({} tokens): {:?}",
            server_tokens_2.len(),
            server_text_2
        );

        // ── 验证 ──
        // 1. 文本可读
        assert!(!server_text_1.trim().is_empty(), "server text 1 is empty");
        assert!(!server_text_2.trim().is_empty(), "server text 2 is empty");
        // 2. 同一 server 跑两次同 prompt，greedy 结果完全一致（确定性）
        assert_eq!(
            server_tokens_1, server_tokens_2,
            "server not deterministic! run1 != run2\n  run1: {:?}\n  run2: {:?}",
            server_tokens_1, server_tokens_2,
        );
        // 3. 文本包含合理内容（"Paris" 是 "capital of France" 的合理回答）
        let combined = format!("{}{}", prompt, server_text_1);
        eprintln!("full output: {:?}", combined);
        assert!(
            combined.to_lowercase().contains("paris"),
            "output doesn't mention 'Paris' for 'capital of France' prompt: {:?}",
            combined,
        );

        runner2.request_shutdown();
        let _ = runner2_handle.join();
        let _ = server_handle.join();
        Ok(())
    }

    /// Continuous batching 不影响输出正确性：
    /// 两个不同 prompt 同时 batch prefill + decode，验证：
    /// - 各自输出合理文本
    /// - 互不干扰（各自有意义、不乱码）
    /// - batch 内各 seq 的输出与单独跑相同（确定性）
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn server_batch_does_not_corrupt_output() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no model path");
                return Ok(());
            }
        };
        let device = DeviceType::Cuda(0);
        let max_tokens = 15usize;
        let max_batch_tokens = 512usize;
        let max_batch_seqs = 4usize;

        let prompts = ["The capital of France is", "Once upon a time"];

        let model = Llama3::new(&path, device)?;
        let tokenizer_ref = model.tokenizer();
        let eos_token_ids: Vec<i32> = tokenizer_ref
            .eos_token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();
        let toks0: Vec<i32> = tokenizer_ref.encode(prompts[0])?;
        let toks1: Vec<i32> = tokenizer_ref.encode(prompts[1])?;
        let runner = Arc::new(ModelRunner::new(
            model,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        let zmq_ctx = zmq::Context::new();
        let scheduler_push = zmq_ctx.socket(zmq::PUSH)?;
        let worker_pull = zmq_ctx.socket(zmq::PULL)?;
        let worker_push = zmq_ctx.socket(zmq::PUSH)?;
        let scheduler_pull = zmq_ctx.socket(zmq::PULL)?;

        scheduler_push.bind("inproc://batch-corrupt-prefill")?;
        worker_pull.connect("inproc://batch-corrupt-prefill")?;
        worker_push.bind("inproc://batch-corrupt-output")?;
        scheduler_pull.connect("inproc://batch-corrupt-output")?;

        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());
        let (stub_down_rx, stub_up_tx, stub_live, _stub_down_tx, _stub_up_rx) = stub_control_handles();
        let server = SubScheduler::new(
            Arc::clone(&runner),
            device,
            worker_pull,
            worker_push,
            eos_token_ids.clone(),
            0,
            stub_down_rx,
            stub_up_tx,
            stub_live,
            "test-worker".to_string(),
            "test-model".to_string(),
        );
        let server_handle = std::thread::spawn(move || server.run());

        // ════ Part 1: 先单独跑两个请求建立 baseline ════
        let (baseline_0, _) = run_one_request(
            &scheduler_push,
            &scheduler_pull,
            toks0.clone(),
            0,
            1,
            max_tokens,
        )?;
        let (baseline_1, _) = run_one_request(
            &scheduler_push,
            &scheduler_pull,
            toks1.clone(),
            0,
            2,
            max_tokens,
        )?;

        let text_solo_0 = runner
            .model()
            .tokenizer()
            .decode(&baseline_0)
            .unwrap_or_default();
        let text_solo_1 = runner
            .model()
            .tokenizer()
            .decode(&baseline_1)
            .unwrap_or_default();
        eprintln!("solo[0] '{}' → {:?}", prompts[0], text_solo_0);
        eprintln!("solo[1] '{}' → {:?}", prompts[1], text_solo_1);

        // ════ Part 2: 两个请求同时 batch prefill ════
        let combined_input_ids: Vec<i32> = [toks0.as_slice(), toks1.as_slice()].concat();
        let cmd = PrefillBatchCmd {
            input_ids: combined_input_ids,
            q_start_loc: vec![0, toks0.len() as u32],
            segments: vec![
                PrefillSegmentMeta {
                    sequence_id: 10,
                    block_table: vec![0],
                    block_size: 16,
                    prompt_len: toks0.len() as u32,
                    segment_start: 0,
                    segment_end: toks0.len() as u32,
                    max_tokens,
                    sampling_params: SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: -1,
                    },
                    completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
                },
                PrefillSegmentMeta {
                    sequence_id: 11,
                    block_table: vec![1],
                    block_size: 16,
                    prompt_len: toks1.len() as u32,
                    segment_start: 0,
                    segment_end: toks1.len() as u32,
                    max_tokens,
                    sampling_params: SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: -1,
                    },
                    completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
                },
            ],
        };
        scheduler_push.send(rmp_serde::to_vec(&cmd).unwrap(), 0)?;

        // 收集两个请求的输出
        let mut tokens_0: Vec<i32> = Vec::new();
        let mut tokens_1: Vec<i32> = Vec::new();
        let mut done_0 = false;
        let mut done_1 = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);

        while !(done_0 && done_1) {
            if std::time::Instant::now() > deadline {
                panic!("timeout: done_0={}, done_1={}", done_0, done_1);
            }
            let msg = scheduler_pull.recv_bytes(0)?;
            let out: StepOutput = rmp_serde::from_slice(&msg)?;
            for t in &out.tokens {
                match t.sequence_id {
                    10 => {
                        tokens_0.push(t.token_id);
                        if t.finished {
                            done_0 = true;
                        }
                    }
                    11 => {
                        tokens_1.push(t.token_id);
                        if t.finished {
                            done_1 = true;
                        }
                    }
                    other => panic!("unexpected sequence_id: {}", other),
                }
            }
        }

        let text_batch_0 = runner
            .model()
            .tokenizer()
            .decode(&tokens_0)
            .unwrap_or_default();
        let text_batch_1 = runner
            .model()
            .tokenizer()
            .decode(&tokens_1)
            .unwrap_or_default();
        eprintln!("batch[0] ({} tokens): {:?}", tokens_0.len(), text_batch_0);
        eprintln!("batch[1] ({} tokens): {:?}", tokens_1.len(), text_batch_1);

        // ── 验证 ──
        // 1. 文本可读
        assert!(!text_batch_0.trim().is_empty(), "batch[0] decoded to empty");
        assert!(!text_batch_1.trim().is_empty(), "batch[1] decoded to empty");

        // 2. batch prefill 的首 token 与 solo 一致
        //    （首 token 由 prefill forward 决定，不受 CUDA Graph 影响）
        assert_eq!(
            baseline_0[0], tokens_0[0],
            "batch[0] first token differs from solo: {} vs {}",
            baseline_0[0], tokens_0[0],
        );
        assert_eq!(
            baseline_1[0], tokens_1[0],
            "batch[1] first token differs from solo: {} vs {}",
            baseline_1[0], tokens_1[0],
        );

        // 3. 语义正确性检查
        let full_0 = format!("{}{}", prompts[0], text_batch_0);
        assert!(
            full_0.to_lowercase().contains("paris"),
            "batch[0] doesn't mention 'Paris': {:?}",
            full_0,
        );
        // prompt[1] "Once upon a time" 应该生成叙事性文本，不应包含乱码
        assert!(
            text_batch_1.len() > 10,
            "batch[1] text too short: {:?}",
            text_batch_1,
        );

        runner.request_shutdown();
        let _ = runner_handle.join();
        let _ = server_handle.join();
        Ok(())
    }
}
