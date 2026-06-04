//! SubScheduler — Worker 内部二级调度器。
//!
//! 职责：
//! - 接收 PrefillBatchCmd（from Scheduler via ZMQ）
//! - 管理 active decode sequences（自回归循环）
//! - 组装 mixed batch（decode在前 + prefill在后）
//! - 通过 SyncFlags 与 Runner 线程握手
//! - 采样输出 token → 发 StepOutput 回 Scheduler
//!
//! ## KV 分配模型
//!
//! 从 Phase 3 起，SubScheduler 持有 [`GlobalKvAllocator`]，每次 step 入口
//! 一次性 `alloc_segment(total_new_tokens)` 拿一段连续的全局 KV 索引
//! `[base, base+N)`，再按 batch 顺序切片分给每个 seq。每个 seq 的最终
//! `block_table = stored_history ++ prefix_hint? ++ new_indices`，正好对应
//! kernel 在 `block_size=1` 下 `block_table[seq][i] == seq 第 i 个 token 的
//! 全局 KV 索引` 的语义。
//!
//! Scheduler 通过 control plane 的 `FreeKvIndices` 通知 worker 把若干索引
//! 还回 free pool；SubScheduler 转发到 [`GlobalKvAllocator::free`]。

use std::collections::VecDeque;

use infer_protocol::scheduler_to_worker_data::{PrefillBatchCmd, SamplingParams};
use infer_protocol::worker_to_scheduler_data::{StepOutput, GeneratedToken};

use crate::domain::global_kv_alloc::{AllocFull, GlobalKvAllocator};

/// 单步最大 batch 容量。
pub const MAX_BATCH_SEQS: usize = 256;
pub const MAX_BATCH_TOKENS: usize = 8192;

/// 一个活跃的 decode sequence。
///
/// `block_table` 在 `block_size=1` 下就是该序列**所有已写入 KV 的 token**
/// 在全局 KV pool 里的索引列表。每次 decode step 之后，新分配的 `new_index`
/// 追加到末尾。
#[derive(Debug, Clone)]
pub struct DecodeSeq {
    pub sequence_id: u64,
    pub kv_len: usize,
    pub last_token: i32,
    pub max_tokens: usize,
    pub generated_count: usize,
    pub sampling_params: SamplingParams,
    pub block_table: Vec<u32>,
    pub block_size: usize,
    pub finished: bool,
}

/// 一个 prefill segment 的本地视图，含 prefix 命中和本步要分配的 token 数。
#[derive(Debug, Clone)]
pub struct PrefillView {
    pub sequence_id: u64,
    /// 完整 prompt 的 input token ids（仅本 segment 范围内的部分）。
    pub input_ids: Vec<i32>,
    /// 本 segment 在原 prompt 中的起始位置（用作 RoPE pos）。
    pub segment_start: u32,
    /// `Some(indices)` 表示该 segment 起点 N 个 token 命中了 scheduler 的
    /// RadixTree 前缀缓存，N == indices.len()，对应索引已在 worker 全局
    /// KV pool 上有效，**无需再写一次**；该段直接从 input_ids 中跳过。
    /// `None` 表示无命中或本 segment 不在 prompt 起点。
    pub prefix_hint: Option<Vec<u32>>,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub is_final_chunk: bool,
}

/// 一步 batch 的描述（SubScheduler 构建，Runner 执行）。
#[derive(Debug)]
pub struct StepBatch {
    /// 扁平的 input token IDs.
    pub input_tokens: Vec<i32>,
    /// Ragged batch 偏移: q_start_loc[i+1] - q_start_loc[i] = seq i 的 q_len.
    pub q_start_loc: Vec<i32>,
    /// 每个 seq 的 position 起始.
    pub positions: Vec<i32>,
    /// Batch 中 decode 部分的数量（排在前面）.
    pub num_decode: usize,
    /// Batch 中 prefill segment 的数量（排在后面）.
    pub num_prefill: usize,
    /// 每个 item 的 sequence_id（for output 映射）.
    pub sequence_ids: Vec<u64>,
    /// 每个 item 的 KV 长度（attention 需要）.
    pub kv_lens: Vec<usize>,
    /// 每个 item 的完整 `block_table`（含 prefix + 历史 + 本步新分配的索引）。
    /// 顺序与 `sequence_ids` 一致：先 num_decode 个 decode 的 block_table，
    /// 再 num_prefill 个 prefill 的 block_table。
    pub block_tables: Vec<Vec<u32>>,
    /// 本 step 给每个 seq 新分配的全局 KV 索引段 `(base, len)`。
    /// scheduler 通过 `StepOutput.assigned_indices` 拿到后接到 RadixTree。
    pub new_segments: Vec<NewSegment>,
}

/// 本 step 给一个 seq 新分配的全局 KV 索引段。`indices = [base..base+len)`。
#[derive(Debug, Clone, Copy)]
pub struct NewSegment {
    pub sequence_id: u64,
    pub base: u32,
    pub len: u32,
}

impl NewSegment {
    pub fn end(&self) -> u32 {
        self.base + self.len
    }
}

/// SubScheduler 内部状态。
pub struct SubScheduler {
    /// 正在 decode 的活跃 sequences。
    pub active_decodes: Vec<DecodeSeq>,
    /// 待处理的 prefill 命令队列。
    pub pending_prefills: VecDeque<PrefillBatchCmd>,
    /// 上一步提交的 batch 元信息（用于解释 output）。
    pub last_batch: Option<StepBatch>,
    /// Max capacity config.
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
    /// Worker-owned global KV allocator. `None` 直到通过 `attach_kv_allocator`
    /// 注入容量，此时所有 alloc 走 [`Self::build_step`] 新流程；`None` 时
    /// 仍然兼容旧的 `build_mixed_batch` 路径（block_table 由 PrefillSegmentMeta
    /// 携带）。Phase 7 删除旧路径后该字段变为必填。
    pub kv_alloc: Option<GlobalKvAllocator>,
}

/// step 装配失败的可能原因。
#[derive(Debug)]
pub enum BuildStepError {
    /// 没有任何要跑的 seq 或 prefill。
    Empty,
    /// `GlobalKvAllocator` 未注入（`attach_kv_allocator` 未调用）。
    NoAllocator,
    /// KV 容量不足。Worker 不应自己处理；fail-fast 上报 scheduler 重新校对。
    AllocFull(AllocFull),
}

impl std::fmt::Display for BuildStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildStepError::Empty => write!(f, "no work to schedule"),
            BuildStepError::NoAllocator => {
                write!(f, "GlobalKvAllocator not attached")
            }
            BuildStepError::AllocFull(e) => write!(f, "KV pool full: {}", e),
        }
    }
}

impl SubScheduler {
    pub fn new(max_batch_tokens: usize, max_batch_seqs: usize) -> Self {
        Self {
            active_decodes: Vec::new(),
            pending_prefills: VecDeque::new(),
            last_batch: None,
            max_batch_tokens,
            max_batch_seqs,
            kv_alloc: None,
        }
    }

    /// 注入全局 KV 索引分配器。在 worker 完成 paged KV 池构造之后、进入
    /// serve loop 之前调用。`total_indices` == `num_blocks`（block_size=1）。
    pub fn attach_kv_allocator(&mut self, total_indices: u32) {
        self.kv_alloc = Some(GlobalKvAllocator::new(total_indices));
    }

    /// `FreeKvIndices` 入站：把 indices 还给 allocator。
    pub fn free_indices(&mut self, indices: &[u32]) {
        if let Some(alloc) = self.kv_alloc.as_mut() {
            alloc.free(indices);
        }
    }

    /// 当前 outstanding（已分配未释放）KV token 数。
    pub fn outstanding_kv_tokens(&self) -> u32 {
        self.kv_alloc.as_ref().map_or(0, |a| a.outstanding())
    }

    /// 全局 KV pool 容量总量。
    pub fn kv_capacity(&self) -> u32 {
        self.kv_alloc.as_ref().map_or(0, |a| a.total())
    }

    // ─── Phase 3: 段式 KV 分配的 step 装配 ─────────────────────────────

    /// 构建下一步的 mixed batch（**段式 KV 分配**版本）。
    ///
    /// 流程：
    /// 1. 按 batch 顺序统计每个参与 seq 本步要写多少新 KV token：
    ///    - decode seq → 1
    ///    - prefill segment → `input_ids.len() - prefix_hit`
    /// 2. 一次 `alloc_segment(total)` 拿 `[base, base+total)`
    /// 3. 按 1) 顺序切片分给每个 seq，得到 `new_indices_i`
    /// 4. 拼 `block_table_i = stored_history ++ prefix_hint? ++ new_indices_i`
    /// 5. 收集 `new_segments`（base/len 二元组），由调用者写入 `StepOutput`
    ///
    /// **不消耗 `pending_prefills`**；该队列里的命令需要先转成
    /// [`PrefillView`] 列表（Phase 4 协议改造完成后由 worker 主路径完成
    /// 翻译）。本函数把 prefill 视图作为参数传入，保持单元可测。
    pub fn build_step(
        &mut self,
        prefill_views: &[PrefillView],
    ) -> Result<StepBatch, BuildStepError> {
        let alloc = self.kv_alloc.as_mut().ok_or(BuildStepError::NoAllocator)?;

        // ── 1. 统计 token 数 + 编排顺序 ──
        // 顺序：decode 在前，prefill 在后（保持现有 attention 内核约定）。
        let mut order_decode: Vec<usize> = Vec::new(); // 索引到 active_decodes
        let mut total_new = 0u32;
        let mut total_seqs = 0usize;
        let mut total_q_tokens = 0usize;

        for (i, seq) in self.active_decodes.iter().enumerate() {
            if seq.finished {
                continue;
            }
            if total_q_tokens + 1 > self.max_batch_tokens {
                break;
            }
            if total_seqs >= self.max_batch_seqs {
                break;
            }
            order_decode.push(i);
            total_new += 1;
            total_q_tokens += 1;
            total_seqs += 1;
        }

        // 每个 prefill 本步要写的 token 数 = input_ids.len() - prefix_hit_len
        struct PFork<'a> {
            v: &'a PrefillView,
            new_tokens: usize,
            prefix_hit: usize,
        }
        let mut order_prefill: Vec<PFork<'_>> = Vec::new();
        for v in prefill_views {
            let prefix_hit = v.prefix_hint.as_ref().map_or(0, |h| h.len());
            // 写到 KV 的部分 = input_ids 中跳过 prefix_hit 后剩下的
            let new_tokens = v.input_ids.len().saturating_sub(prefix_hit);
            if new_tokens == 0 {
                // 理论上不该发生（segment 至少有一个 token 要写）；保险跳过。
                continue;
            }
            if total_q_tokens + new_tokens > self.max_batch_tokens {
                break;
            }
            if total_seqs >= self.max_batch_seqs {
                break;
            }
            total_new += new_tokens as u32;
            total_q_tokens += new_tokens;
            total_seqs += 1;
            order_prefill.push(PFork {
                v,
                new_tokens,
                prefix_hit,
            });
        }

        if total_seqs == 0 {
            return Err(BuildStepError::Empty);
        }

        // ── 2. Allocate `total_new` indices in one call (may span ranges). ──
        let new_indices = alloc
            .alloc_indices(total_new)
            .map_err(BuildStepError::AllocFull)?;

        // ── 3+4+5. Slice + assemble block_table + collect new_segments ──
        let mut batch = StepBatch {
            input_tokens: Vec::with_capacity(total_q_tokens),
            q_start_loc: Vec::with_capacity(total_seqs + 1),
            positions: Vec::with_capacity(total_q_tokens),
            num_decode: order_decode.len(),
            num_prefill: order_prefill.len(),
            sequence_ids: Vec::with_capacity(total_seqs),
            kv_lens: Vec::with_capacity(total_seqs),
            block_tables: Vec::with_capacity(total_seqs),
            new_segments: Vec::with_capacity(total_seqs),
        };
        batch.q_start_loc.push(0);

        let mut acc_tokens: usize = 0;
        let mut idx_cursor: usize = 0;

        // Helper: emit one or more NewSegments for a seq's slice of
        // `new_indices`. Splits at every discontinuity so each NewSegment
        // is a contiguous run.
        fn push_runs(
            out: &mut Vec<NewSegment>,
            seq_id: u64,
            slots: &[u32],
        ) {
            if slots.is_empty() {
                return;
            }
            let mut run_start = 0usize;
            while run_start < slots.len() {
                let mut run_end = run_start + 1;
                while run_end < slots.len() && slots[run_end] == slots[run_end - 1] + 1 {
                    run_end += 1;
                }
                out.push(NewSegment {
                    sequence_id: seq_id,
                    base: slots[run_start],
                    len: (run_end - run_start) as u32,
                });
                run_start = run_end;
            }
        }

        // -- decode part --
        for &di in &order_decode {
            let seq = &self.active_decodes[di];
            batch.input_tokens.push(seq.last_token);
            batch.positions.push(seq.kv_len as i32);
            acc_tokens += 1;
            batch.q_start_loc.push(acc_tokens as i32);
            batch.sequence_ids.push(seq.sequence_id);
            batch.kv_lens.push(seq.kv_len);

            let new_idx = new_indices[idx_cursor];
            idx_cursor += 1;

            // block_table = stored history ++ this step's 1 slot.
            let mut bt = Vec::with_capacity(seq.block_table.len() + 1);
            bt.extend_from_slice(&seq.block_table);
            bt.push(new_idx);
            batch.block_tables.push(bt);

            push_runs(&mut batch.new_segments, seq.sequence_id, &[new_idx]);
        }

        // -- prefill part --
        for pf in &order_prefill {
            let v = pf.v;
            let pos_start = v.segment_start as usize + pf.prefix_hit;
            for tok in v.input_ids.iter().skip(pf.prefix_hit) {
                batch.input_tokens.push(*tok);
            }
            for i in 0..pf.new_tokens {
                batch.positions.push((pos_start + i) as i32);
            }
            acc_tokens += pf.new_tokens;
            batch.q_start_loc.push(acc_tokens as i32);
            batch.sequence_ids.push(v.sequence_id);
            batch.kv_lens.push(pos_start);

            // Take this seq's slice from the bulk-allocated index list.
            let seq_slots = &new_indices[idx_cursor..idx_cursor + pf.new_tokens];
            idx_cursor += pf.new_tokens;

            // block_table = prefix_hint (if any) ++ this seq's new slots.
            let mut bt = Vec::with_capacity(pf.prefix_hit + pf.new_tokens);
            if let Some(hint) = v.prefix_hint.as_ref() {
                bt.extend_from_slice(hint);
            }
            bt.extend_from_slice(seq_slots);
            batch.block_tables.push(bt);

            push_runs(&mut batch.new_segments, v.sequence_id, seq_slots);
        }

        debug_assert_eq!(idx_cursor, total_new as usize);
        Ok(batch)
    }

    /// 把 `build_step` 拿到的 `new_segments` 在 step 完成、对应 token 已写入
    /// KV 之后追加到各自 `DecodeSeq.block_table` 末尾，使下一 step 的 history
    /// 视图正确。
    ///
    /// 同时为新创建的 decode seq 提供初始 `block_table`：从 prefill 视图的
    /// 拼接结果（prefix_hint ++ new_indices）开始。`first_token` 是 prefill
    /// 步骤产出的第一个采样结果。
    ///
    /// 调用方负责生成 `StepOutput.assigned_indices` —— 直接把 `new_segments`
    /// 序列化即可（Phase 4 加上协议字段）。
    pub fn commit_step(
        &mut self,
        batch: &StepBatch,
        output_token_ids: &[i32],
        prefill_views: &[PrefillView],
    ) -> StepOutput {
        let mut step_output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
            assigned_indices: batch
                .new_segments
                .iter()
                .map(|s| infer_protocol::worker_to_scheduler_data::AssignedIndices {
                    sequence_id: s.sequence_id,
                    base: s.base,
                    len: s.len as u16,
                })
                .collect(),
        };

        // ── decode 部分 ──
        for di in 0..batch.num_decode {
            let token = output_token_ids[di];
            let seq_id = batch.sequence_ids[di];
            let new_idx = batch.new_segments[di];
            if let Some(seq) =
                self.active_decodes.iter_mut().find(|s| s.sequence_id == seq_id)
            {
                seq.last_token = token;
                seq.kv_len += 1;
                seq.generated_count += 1;
                seq.block_table.push(new_idx.base);
                let finished = seq.generated_count >= seq.max_tokens;
                seq.finished = finished;
                step_output.tokens.push(GeneratedToken {
                    sequence_id: seq_id,
                    token_id: token,
                    finished,
                });
            }
        }

        // ── prefill 部分 ──
        for pi in 0..batch.num_prefill {
            let token = output_token_ids[batch.num_decode + pi];
            let seq_id = batch.sequence_ids[batch.num_decode + pi];
            let v = &prefill_views[pi];

            step_output.prefill_done.push(seq_id);
            step_output.tokens.push(GeneratedToken {
                sequence_id: seq_id,
                token_id: token,
                finished: false,
            });

            if v.is_final_chunk {
                // 用本 segment 的 block_table（prefix + new_indices）作为
                // 新创建 DecodeSeq 的初始历史。
                let bt = batch.block_tables[batch.num_decode + pi].clone();
                let kv_len = bt.len();
                self.active_decodes.push(DecodeSeq {
                    sequence_id: seq_id,
                    kv_len,
                    last_token: token,
                    max_tokens: v.max_tokens,
                    generated_count: 1,
                    sampling_params: v.sampling_params.clone(),
                    block_table: bt,
                    block_size: 1,
                    finished: false,
                });
            }
            // ContinuePrefill chunked 续片场景：尚未支持自维护 history。
            // worker_main 旧路径仍然走原 SubScheduler::build_mixed_batch，
            // 不会触达这里。Phase 4 协议联通时补完。
        }

        // ── 移除已完成 ──
        self.active_decodes.retain(|s| !s.finished);

        step_output
    }

    // ─── 旧路径（保留至 Phase 7 删除） ─────────────────────────────────

    /// 构建下一步的 mixed batch（**旧版**：block_table 来自协议）。
    pub fn build_mixed_batch(&mut self) -> Option<StepBatch> {
        let mut batch = StepBatch {
            input_tokens: Vec::new(),
            q_start_loc: vec![0],
            positions: Vec::new(),
            num_decode: 0,
            num_prefill: 0,
            sequence_ids: Vec::new(),
            kv_lens: Vec::new(),
            block_tables: Vec::new(),
            new_segments: Vec::new(),
        };
        let mut total_tokens = 0usize;
        let mut total_seqs = 0usize;

        for seq in &self.active_decodes {
            if seq.finished { continue; }
            if total_tokens + 1 > self.max_batch_tokens { break; }
            if total_seqs >= self.max_batch_seqs { break; }

            batch.input_tokens.push(seq.last_token);
            batch.positions.push(seq.kv_len as i32);
            total_tokens += 1;
            batch.q_start_loc.push(total_tokens as i32);
            batch.sequence_ids.push(seq.sequence_id);
            batch.kv_lens.push(seq.kv_len);
            batch.block_tables.push(seq.block_table.clone());
            batch.num_decode += 1;
            total_seqs += 1;
        }

        while let Some(cmd) = self.pending_prefills.front() {
            for (i, segment) in cmd.segments.iter().enumerate() {
                let seg_len = (segment.segment_end - segment.segment_start) as usize;
                if total_tokens + seg_len > self.max_batch_tokens { break; }
                if total_seqs >= self.max_batch_seqs { break; }

                let range = cmd.segment_token_range(i);
                batch.input_tokens.extend_from_slice(&cmd.input_ids[range]);
                for pos in segment.segment_start..segment.segment_end {
                    batch.positions.push(pos as i32);
                }
                total_tokens += seg_len;
                batch.q_start_loc.push(total_tokens as i32);
                batch.sequence_ids.push(segment.sequence_id);
                batch.kv_lens.push(segment.segment_start as usize);
                batch.block_tables.push(segment.block_table.clone());
                batch.num_prefill += 1;
                total_seqs += 1;
            }
            self.pending_prefills.pop_front();
        }

        if total_seqs == 0 { return None; }
        Some(batch)
    }

    /// 处理 Runner 的输出 token，更新 decode 状态（**旧版**）。
    pub fn process_output(
        &mut self,
        output_token_ids: &[i32],
        original_cmd: Option<&PrefillBatchCmd>,
    ) -> StepOutput {
        let batch = self.last_batch.as_ref().expect("no batch to process");
        let mut step_output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
            // Legacy path doesn't drive scheduler-side RadixTree append; leave
            // empty. Phase 7 removes this path entirely.
            assigned_indices: Vec::new(),
        };

        let mut output_idx = 0;

        for i in 0..batch.num_decode {
            let token = output_token_ids[output_idx];
            output_idx += 1;
            let seq_id = batch.sequence_ids[i];

            if let Some(seq) = self.active_decodes.iter_mut().find(|s| s.sequence_id == seq_id) {
                seq.last_token = token;
                seq.kv_len += 1;
                seq.generated_count += 1;
                let finished = seq.generated_count >= seq.max_tokens;
                seq.finished = finished;
                step_output.tokens.push(GeneratedToken {
                    sequence_id: seq_id,
                    token_id: token,
                    finished,
                });
            }
        }

        for i in 0..batch.num_prefill {
            let token = output_token_ids[output_idx];
            output_idx += 1;
            let seq_id = batch.sequence_ids[batch.num_decode + i];

            step_output.prefill_done.push(seq_id);
            step_output.tokens.push(GeneratedToken {
                sequence_id: seq_id,
                token_id: token,
                finished: false,
            });

            let (max_tokens, sampling) = if let Some(cmd) = original_cmd {
                let seg = &cmd.segments[i];
                (seg.max_tokens, seg.sampling_params.clone())
            } else {
                (2048, SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 })
            };

            self.active_decodes.push(DecodeSeq {
                sequence_id: seq_id,
                kv_len: batch.kv_lens[batch.num_decode + i] + 1,
                last_token: token,
                max_tokens,
                generated_count: 1,
                sampling_params: sampling,
                block_table: Vec::new(),
                block_size: 1,
                finished: false,
            });
        }

        self.active_decodes.retain(|s| !s.finished);

        step_output
    }

    /// Cancel a sequence by ID.
    pub fn cancel_sequence(&mut self, sequence_id: u64) {
        self.active_decodes.retain(|s| s.sequence_id != sequence_id);
    }

    /// Is the scheduler idle (no active work)?
    pub fn is_idle(&self) -> bool {
        self.active_decodes.iter().all(|s| s.finished) && self.pending_prefills.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_protocol::scheduler_to_worker_data::*;

    fn make_prefill_cmd(seq_id: u64, tokens: &[i32]) -> PrefillBatchCmd {
        PrefillBatchCmd {
            input_ids: tokens.to_vec(),
            q_start_loc: vec![0],
            segments: vec![PrefillSegmentMeta {
                sequence_id: seq_id,
                block_table: vec![0, 1],
                block_size: 16,
                prompt_len: tokens.len() as u32,
                segment_start: 0,
                segment_end: tokens.len() as u32,
                max_tokens: 100,
                sampling_params: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 },
                completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
                ignore_eos: false,
                prefix_hint: None,
            }],
        }
    }

    fn pf_view(seq_id: u64, prompt: &[i32], prefix_hint: Option<Vec<u32>>) -> PrefillView {
        PrefillView {
            sequence_id: seq_id,
            input_ids: prompt.to_vec(),
            segment_start: 0,
            prefix_hint,
            max_tokens: 32,
            sampling_params: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 },
            is_final_chunk: true,
        }
    }

    fn dummy_decode(seq_id: u64, kv_len: usize, last_token: i32, history: &[u32]) -> DecodeSeq {
        DecodeSeq {
            sequence_id: seq_id,
            kv_len,
            last_token,
            max_tokens: 100,
            generated_count: kv_len,
            sampling_params: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 },
            block_table: history.to_vec(),
            block_size: 1,
            finished: false,
        }
    }

    #[test]
    fn build_batch_decode_only() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.active_decodes.push(dummy_decode(1, 5, 42, &[]));
        sched.active_decodes.push(dummy_decode(2, 3, 99, &[]));

        let batch = sched.build_mixed_batch().unwrap();
        assert_eq!(batch.num_decode, 2);
        assert_eq!(batch.num_prefill, 0);
        assert_eq!(batch.input_tokens, vec![42, 99]);
        assert_eq!(batch.positions, vec![5, 3]);
    }

    #[test]
    fn build_batch_mixed() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.active_decodes.push(dummy_decode(1, 5, 42, &[]));
        sched.pending_prefills.push_back(make_prefill_cmd(2, &[10, 20, 30]));

        let batch = sched.build_mixed_batch().unwrap();
        assert_eq!(batch.num_decode, 1);
        assert_eq!(batch.num_prefill, 1);
        assert_eq!(batch.input_tokens, vec![42, 10, 20, 30]);
        assert_eq!(batch.q_start_loc, vec![0, 1, 4]);
    }

    #[test]
    fn process_output_creates_decode_seq() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.pending_prefills.push_back(make_prefill_cmd(1, &[10, 20, 30]));

        let batch = sched.build_mixed_batch().unwrap();
        sched.last_batch = Some(batch);

        let output = sched.process_output(&[77], None);
        assert_eq!(output.prefill_done, vec![1]);
        assert_eq!(output.tokens[0].token_id, 77);
        assert_eq!(sched.active_decodes.len(), 1);
        assert_eq!(sched.active_decodes[0].last_token, 77);
    }

    // ─── Phase 3: 段式分配的新路径 ─────────────────────────────────

    #[test]
    fn build_step_decode_only_uses_one_segment() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.attach_kv_allocator(64);
        sched.active_decodes.push(dummy_decode(1, 5, 42, &[100, 101, 102, 103, 104]));
        sched.active_decodes.push(dummy_decode(2, 3, 99, &[200, 201, 202]));

        let batch = sched.build_step(&[]).unwrap();
        assert_eq!(batch.num_decode, 2);
        assert_eq!(batch.num_prefill, 0);
        assert_eq!(batch.input_tokens, vec![42, 99]);
        assert_eq!(batch.positions, vec![5, 3]);

        // 段从 0 开始：seq 1 拿 0,seq 2 拿 1。
        assert_eq!(batch.new_segments.len(), 2);
        assert_eq!(batch.new_segments[0].sequence_id, 1);
        assert_eq!(batch.new_segments[0].base, 0);
        assert_eq!(batch.new_segments[0].len, 1);
        assert_eq!(batch.new_segments[1].sequence_id, 2);
        assert_eq!(batch.new_segments[1].base, 1);

        // block_table = 历史 ++ new_index
        assert_eq!(batch.block_tables[0], vec![100, 101, 102, 103, 104, 0]);
        assert_eq!(batch.block_tables[1], vec![200, 201, 202, 1]);

        assert_eq!(sched.outstanding_kv_tokens(), 2);
    }

    #[test]
    fn build_step_prefill_no_prefix_hit_writes_full_prompt() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.attach_kv_allocator(64);

        let v = pf_view(7, &[10, 20, 30, 40], None);
        let batch = sched.build_step(&[v]).unwrap();

        assert_eq!(batch.num_decode, 0);
        assert_eq!(batch.num_prefill, 1);
        assert_eq!(batch.input_tokens, vec![10, 20, 30, 40]);
        assert_eq!(batch.positions, vec![0, 1, 2, 3]);
        // 4 个 token 拿 [0..4)。
        assert_eq!(batch.new_segments[0].base, 0);
        assert_eq!(batch.new_segments[0].len, 4);
        assert_eq!(batch.block_tables[0], vec![0, 1, 2, 3]);
        assert_eq!(sched.outstanding_kv_tokens(), 4);
    }

    #[test]
    fn build_step_prefix_hint_skips_already_cached_tokens() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.attach_kv_allocator(64);

        // prompt 是 6 个 token,前 3 个命中前缀(对应全局索引 100,101,102)。
        let v = pf_view(7, &[10, 20, 30, 40, 50, 60], Some(vec![100, 101, 102]));
        let batch = sched.build_step(&[v]).unwrap();

        // 只重新写后 3 个 token。
        assert_eq!(batch.input_tokens, vec![40, 50, 60]);
        assert_eq!(batch.positions, vec![3, 4, 5]);
        assert_eq!(batch.new_segments[0].base, 0);
        assert_eq!(batch.new_segments[0].len, 3);
        // block_table = 命中前缀 ++ 新分配的 3 个。
        assert_eq!(batch.block_tables[0], vec![100, 101, 102, 0, 1, 2]);
        // kv_lens 表示 KV 已写入到 prefix_hit_len。
        assert_eq!(batch.kv_lens[0], 3);
    }

    #[test]
    fn build_step_mixed_decode_and_prefill_share_one_segment() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.attach_kv_allocator(64);
        sched.active_decodes.push(dummy_decode(1, 5, 42, &[200, 201, 202, 203, 204]));

        let v = pf_view(2, &[10, 20, 30], None);
        let batch = sched.build_step(&[v]).unwrap();

        // total_new = 1 (decode) + 3 (prefill) = 4。
        assert_eq!(batch.input_tokens, vec![42, 10, 20, 30]);
        assert_eq!(batch.q_start_loc, vec![0, 1, 4]);
        // decode 拿 0,prefill 拿 [1..4)。
        assert_eq!(batch.new_segments[0].base, 0);
        assert_eq!(batch.new_segments[0].len, 1);
        assert_eq!(batch.new_segments[1].base, 1);
        assert_eq!(batch.new_segments[1].len, 3);
        assert_eq!(batch.block_tables[0], vec![200, 201, 202, 203, 204, 0]);
        assert_eq!(batch.block_tables[1], vec![1, 2, 3]);
        assert_eq!(sched.outstanding_kv_tokens(), 4);
    }

    #[test]
    fn build_step_returns_alloc_full_when_capacity_exhausted() {
        let mut sched = SubScheduler::new(8192, 256);
        // 容量 = 5,但要求 6 个 token。
        sched.attach_kv_allocator(5);

        let v = pf_view(1, &[1, 2, 3, 4, 5, 6], None);
        let err = sched.build_step(&[v]).unwrap_err();
        match err {
            BuildStepError::AllocFull(_) => {}
            other => panic!("expected AllocFull, got {:?}", other),
        }
    }

    #[test]
    fn build_step_without_allocator_returns_no_allocator() {
        let mut sched = SubScheduler::new(8192, 256);
        // 没调 attach_kv_allocator
        let v = pf_view(1, &[1, 2], None);
        let err = sched.build_step(&[v]).unwrap_err();
        assert!(matches!(err, BuildStepError::NoAllocator));
    }

    #[test]
    fn commit_step_appends_decode_indices_and_creates_new_decode() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.attach_kv_allocator(64);
        sched.active_decodes.push(dummy_decode(1, 5, 42, &[200, 201, 202, 203, 204]));

        let v = pf_view(2, &[10, 20, 30], None);
        let batch = sched.build_step(&[v.clone()]).unwrap();

        // Runner 假装产出: decode → 555,prefill 第一个 token → 777
        let out = sched.commit_step(&batch, &[555, 777], &[v]);
        assert_eq!(out.prefill_done, vec![2]);
        assert_eq!(out.tokens.len(), 2);
        // 旧 decode seq 推进:kv_len 5→6,history 末尾添加 0。
        let s1 = sched.active_decodes.iter().find(|s| s.sequence_id == 1).unwrap();
        assert_eq!(s1.kv_len, 6);
        assert_eq!(s1.last_token, 555);
        assert_eq!(s1.block_table, vec![200, 201, 202, 203, 204, 0]);
        // 新 decode seq 由 prefill 产生:history = [1, 2, 3](无 prefix hint)。
        let s2 = sched.active_decodes.iter().find(|s| s.sequence_id == 2).unwrap();
        assert_eq!(s2.kv_len, 3);
        assert_eq!(s2.last_token, 777);
        assert_eq!(s2.block_table, vec![1, 2, 3]);
    }

    #[test]
    fn free_indices_returns_capacity_to_pool() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.attach_kv_allocator(8);
        let v = pf_view(1, &[1, 2, 3, 4], None);
        let _batch = sched.build_step(&[v]).unwrap();
        assert_eq!(sched.outstanding_kv_tokens(), 4);
        sched.free_indices(&[0, 1, 2, 3]);
        assert_eq!(sched.outstanding_kv_tokens(), 0);
    }
}
