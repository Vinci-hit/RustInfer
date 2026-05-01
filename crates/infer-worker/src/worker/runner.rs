use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;
use std::time::Instant;

use crate::base::{DataType, DeviceType};
use crate::base::error::Result;
use crate::model::llm::llama3::Llama3;
use crate::runtime::InferenceState;
use crate::tensor::Tensor;
use crate::worker::shared_buffers::SharedBuffers;

/// ModelRunner — GPU 线程，执行 forward + sample
///
/// Phase 1: 对 batch 中每个 seq 串行调用现有 forward 接口。
///           直接从共享 GPU buffer slice 出输入，不做多余的 H2D/D2H。
/// Phase 2 (TODO): 真正的 batched forward。
pub struct ModelRunner {
    model: Llama3,
    /// 每个 KV slot 一个 InferenceState
    states: Vec<InferenceState>,
    shared: Arc<SharedBuffers>,
    device_id: i32,
}

impl ModelRunner {
    pub fn new(
        model: Llama3,
        states: Vec<InferenceState>,
        shared: Arc<SharedBuffers>,
        device_id: i32,
    ) -> Self {
        Self { model, states, shared, device_id }
    }

    /// GPU 线程主循环
    pub fn run(mut self) {
        tracing::info!("ModelRunner started on device {}", self.device_id);

        loop {
            // spin wait input ready
            let total_tokens = loop {
                let v = self.shared.input_meta.ready.load(Acquire);
                if v > 0 { break v as usize; }
                std::hint::spin_loop();
            };

            let num_decode = self.shared.input_meta.num_decode_seqs.load(Acquire) as usize;
            let num_prefill = self.shared.input_meta.num_prefill_seqs.load(Acquire) as usize;
            let num_seqs = num_decode + num_prefill;

            tracing::debug!(
                "Runner step: total_tokens={}, decode={}, prefill={}",
                total_tokens, num_decode, num_prefill,
            );

            let start = Instant::now();

            // D2H 只拷 metadata (小数组, 几十个 i32)
            // Runner 需要知道每个 seq 的边界和 slot 来做循环
            let q_start_loc = self.shared.read_output_i32(
                &self.shared.input_q_start_loc, num_seqs + 1).expect("read q_start_loc");
            let slot_indices = self.shared.read_output_i32(
                &self.shared.input_slot_indices, num_seqs).expect("read slot_indices");
            let context_lens = self.shared.read_output_i32(
                &self.shared.input_context_lens, num_seqs).expect("read context_lens");

            // input token_ids 和 positions 留在 GPU, 通过 view_prefix + slice 直接用
            let all_token_ids = self.shared.input_token_ids.view_prefix(total_tokens)
                .expect("slice token_ids");
            let all_positions = self.shared.input_positions.view_prefix(total_tokens)
                .expect("slice positions");

            // Phase 1: 串行执行每个 seq
            let mut output_tokens = Vec::with_capacity(num_seqs);

            for seq_idx in 0..num_seqs {
                let seq_start = q_start_loc[seq_idx] as usize;
                let seq_end = q_start_loc[seq_idx + 1] as usize;
                let seq_len = seq_end - seq_start;
                let slot = slot_indices[seq_idx] as usize;
                let ctx_len = context_lens[seq_idx] as usize;

                // 从共享 GPU buffer slice 出该 seq 的 token_ids
                let seq_token_ids = all_token_ids.slice(&[seq_start], &[seq_len])
                    .expect("slice seq token_ids");
                // position: 只取该 seq 的第一个 pos 做 pos_cpu
                // (forward_prefill 用 pos_cpu[0] 作为 start_pos,
                //  forward_decoding 用 pos_cpu[0] 作为当前 pos)
                let seq_pos = all_positions.slice(&[seq_start], &[1])
                    .expect("slice seq pos");

                let state = &mut self.states[slot];

                let sampled_token = if seq_len > 1 {
                    // Prefill
                    self.model.forward_prefill(state, &seq_token_ids, &seq_pos, seq_len)
                        .expect("forward_prefill failed")
                } else {
                    // Decode
                    self.model.forward_decoding(state, &seq_token_ids, &seq_pos)
                        .expect("forward_decoding failed")
                };

                output_tokens.push(sampled_token);
            }

            let elapsed_us = start.elapsed().as_micros() as u64;
            tracing::debug!("Runner step done in {}us, {} tokens out", elapsed_us, output_tokens.len());

            // 写 output tokens 到共享 buffer (H2D)
            self.shared.write_input_i32(
                &self.shared.output_token_ids, &output_tokens, num_seqs,
            ).expect("write output_token_ids");

            // 信号: 先释放 input, 再标记 output ready
            self.shared.input_meta.ready.store(0, Release);
            self.shared.output_meta.ready.store(num_seqs as u32, Release);
        }
    }
}
