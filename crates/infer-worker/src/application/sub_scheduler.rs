//! SubScheduler — Worker 内部二级调度器。
//!
//! 职责：
//! - 接收 PrefillBatchCmd（from Scheduler via ZMQ）
//! - 管理 active decode sequences（自回归循环）
//! - 组装 mixed batch（decode在前 + prefill在后）
//! - 通过 SyncFlags 与 Runner 线程握手
//! - 采样输出 token → 发 StepOutput 回 Scheduler

use std::collections::VecDeque;

use infer_protocol::scheduler_to_worker_data::{PrefillBatchCmd, SamplingParams};
use infer_protocol::worker_to_scheduler_data::{StepOutput, GeneratedToken};

use crate::infrastructure::sync_flags::SyncFlags;

/// 单步最大 batch 容量。
pub const MAX_BATCH_SEQS: usize = 256;
pub const MAX_BATCH_TOKENS: usize = 8192;

/// 一个活跃的 decode sequence。
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
}

/// SubScheduler 内部状态。
pub struct SubScheduler {
    /// 正在 decode 的活跃 sequences。
    pub active_decodes: Vec<DecodeSeq>,
    /// 待处理的 prefill 命令队列。
    pub pending_prefills: VecDeque<PrefillBatchCmd>,
    /// 上一步提交的 batch 元信息（用于解释 output）。
    pub last_batch: Option<StepBatch>,
    /// Step buffer 握手.
    pub sync: SyncFlags,
    /// Max capacity config.
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
}

impl SubScheduler {
    pub fn new(max_batch_tokens: usize, max_batch_seqs: usize) -> Self {
        Self {
            active_decodes: Vec::new(),
            pending_prefills: VecDeque::new(),
            last_batch: None,
            sync: SyncFlags::new(),
            max_batch_tokens,
            max_batch_seqs,
        }
    }

    /// 构建下一步的 mixed batch。
    ///
    /// 布局：decode sequences 在前（各 q_len=1），prefill segments 在后。
    /// 总 token 数 ≤ max_batch_tokens.
    pub fn build_mixed_batch(&mut self) -> Option<StepBatch> {
        let mut batch = StepBatch {
            input_tokens: Vec::new(),
            q_start_loc: vec![0],
            positions: Vec::new(),
            num_decode: 0,
            num_prefill: 0,
            sequence_ids: Vec::new(),
            kv_lens: Vec::new(),
        };
        let mut total_tokens = 0usize;
        let mut total_seqs = 0usize;

        // 1. Decode sequences (q_len=1 each)
        for seq in &self.active_decodes {
            if seq.finished { continue; }
            if total_tokens + 1 > self.max_batch_tokens { break; }
            if total_seqs >= self.max_batch_seqs { break; }

            batch.input_tokens.push(seq.last_token);
            batch.positions.push(seq.kv_len as i32); // next position
            total_tokens += 1;
            batch.q_start_loc.push(total_tokens as i32);
            batch.sequence_ids.push(seq.sequence_id);
            batch.kv_lens.push(seq.kv_len);
            batch.num_decode += 1;
            total_seqs += 1;
        }

        // 2. Prefill segments (variable q_len)
        while let Some(cmd) = self.pending_prefills.front() {
            for (i, segment) in cmd.segments.iter().enumerate() {
                let seg_len = (segment.segment_end - segment.segment_start) as usize;
                if total_tokens + seg_len > self.max_batch_tokens { break; }
                if total_seqs >= self.max_batch_seqs { break; }

                let range = cmd.segment_token_range(i);
                batch.input_tokens.extend_from_slice(&cmd.input_ids[range]);
                // Positions: segment_start..segment_end
                for pos in segment.segment_start..segment.segment_end {
                    batch.positions.push(pos as i32);
                }
                total_tokens += seg_len;
                batch.q_start_loc.push(total_tokens as i32);
                batch.sequence_ids.push(segment.sequence_id);
                batch.kv_lens.push(segment.segment_start as usize); // KV already written up to here
                batch.num_prefill += 1;
                total_seqs += 1;
            }
            self.pending_prefills.pop_front();
        }

        if total_seqs == 0 { return None; }
        Some(batch)
    }

    /// 处理 Runner 的输出 token，更新 decode 状态。
    ///
    /// - Decode items: 读 output token → 更新 seq state → 检查 EOS
    /// - Prefill items (FinishPrefillAndStartDecode): 读 first token → 创建 DecodeSeq
    pub fn process_output(
        &mut self,
        output_token_ids: &[i32],
        original_cmd: Option<&PrefillBatchCmd>,
    ) -> StepOutput {
        let batch = self.last_batch.as_ref().expect("no batch to process");
        let mut step_output = StepOutput {
            prefill_done: Vec::new(),
            tokens: Vec::new(),
        };

        let mut output_idx = 0;

        // Process decode outputs
        for i in 0..batch.num_decode {
            let token = output_token_ids[output_idx];
            output_idx += 1;
            let seq_id = batch.sequence_ids[i];

            // Update decode seq
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

        // Process prefill outputs
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

            // 从 original_cmd 取 max_tokens 和 sampling_params
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
                block_size: 16,
                finished: false,
            });
        }

        // Remove finished sequences
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
            }],
        }
    }

    #[test]
    fn build_batch_decode_only() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.active_decodes.push(DecodeSeq {
            sequence_id: 1, kv_len: 5, last_token: 42, max_tokens: 100,
            generated_count: 5, sampling_params: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 },
            block_table: vec![], block_size: 16, finished: false,
        });
        sched.active_decodes.push(DecodeSeq {
            sequence_id: 2, kv_len: 3, last_token: 99, max_tokens: 100,
            generated_count: 3, sampling_params: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 },
            block_table: vec![], block_size: 16, finished: false,
        });

        let batch = sched.build_mixed_batch().unwrap();
        assert_eq!(batch.num_decode, 2);
        assert_eq!(batch.num_prefill, 0);
        assert_eq!(batch.input_tokens, vec![42, 99]);
        assert_eq!(batch.positions, vec![5, 3]);
    }

    #[test]
    fn build_batch_mixed() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.active_decodes.push(DecodeSeq {
            sequence_id: 1, kv_len: 5, last_token: 42, max_tokens: 100,
            generated_count: 5, sampling_params: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: -1 },
            block_table: vec![], block_size: 16, finished: false,
        });
        sched.pending_prefills.push_back(make_prefill_cmd(2, &[10, 20, 30]));

        let batch = sched.build_mixed_batch().unwrap();
        assert_eq!(batch.num_decode, 1);
        assert_eq!(batch.num_prefill, 1);
        assert_eq!(batch.input_tokens, vec![42, 10, 20, 30]); // decode first, then prefill
        assert_eq!(batch.q_start_loc, vec![0, 1, 4]); // [0..1), [1..4)
    }

    #[test]
    fn process_output_creates_decode_seq() {
        let mut sched = SubScheduler::new(8192, 256);
        sched.pending_prefills.push_back(make_prefill_cmd(1, &[10, 20, 30]));

        let batch = sched.build_mixed_batch().unwrap();
        sched.last_batch = Some(batch);

        // Simulate Runner output: one token for the prefill
        let output = sched.process_output(&[77], None);
        assert_eq!(output.prefill_done, vec![1]);
        assert_eq!(output.tokens[0].token_id, 77);
        assert_eq!(sched.active_decodes.len(), 1);
        assert_eq!(sched.active_decodes[0].last_token, 77);
    }
}
