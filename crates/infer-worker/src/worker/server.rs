use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;

use crate::worker::protocol::*;
use crate::worker::shared_buffers::SharedBuffers;

/// 一个活跃的 decode 序列
#[derive(Clone)]
struct DecodeSeq {
    request_id: String,
    kv_slot: usize,
    /// 下一个 token 写入 KV cache 的 position
    next_position: usize,
    /// 上一步采样出的 token (本步 decode 输入)
    last_token: i32,
    /// 已生成 token 数
    generated_count: usize,
    max_tokens: usize,
    sampling: SamplingParams,
}

/// Worker Server 线程 — 负责所有 CPU 操作
pub struct WorkerServer {
    /// ZMQ PULL (收调度器 prefill)
    zmq_in: zmq::Socket,
    /// ZMQ PUSH (发结果给调度器)
    zmq_out: zmq::Socket,
    /// 共享显存
    shared: Arc<SharedBuffers>,
    device_id: i32,
    eos_token_id: i32,
    /// 活跃 decode 序列
    active_decodes: Vec<DecodeSeq>,
    /// 待处理的 prefill
    pending_prefills: Vec<PrefillBatchCmd>,
}

impl WorkerServer {
    pub fn new(
        zmq_in: zmq::Socket,
        zmq_out: zmq::Socket,
        shared: Arc<SharedBuffers>,
        device_id: i32,
        eos_token_id: i32,
    ) -> Self {
        Self {
            zmq_in,
            zmq_out,
            shared,
            device_id,
            eos_token_id,
            active_decodes: Vec::new(),
            pending_prefills: Vec::new(),
        }
    }

    /// 主循环
    pub fn run(mut self) {
        // ══ 启动: 阻塞等第一个 prefill ══
        tracing::info!("WorkerServer waiting for first prefill...");
        let data = self.zmq_in.recv_bytes(0).expect("ZMQ recv failed");
        let cmd: PrefillBatchCmd =
            rmp_serde::from_slice(&data).expect("Failed to deserialize PrefillBatchCmd");
        self.pending_prefills.push(cmd);

        self.write_input_buffer();
        self.promote_prefills_to_decodes();
        tracing::info!("WorkerServer started, {} active decodes", self.active_decodes.len());

        // ══ 稳态循环 ══
        loop {
            // ── 慢路径 (与 Runner 并行): 收新 prefill ──
            self.drain_zmq_prefills();

            // ── 等 Runner 完成 ──
            let num_output_seqs = self.wait_output_ready();

            // ── 快路径: 最小化气泡 ──
            // 1. D2H copy output tokens
            let output_tokens = self.read_output_tokens(num_output_seqs);
            self.shared.output_meta.ready.store(0, Release);

            // 2. 更新 active_decodes 的 last_token / position
            self.update_decode_tokens(&output_tokens);

            // 3. 收新 prefill (非阻塞)
            self.drain_zmq_prefills();

            // 4. 写 input buffer → signal Runner
            if !self.active_decodes.is_empty() || !self.pending_prefills.is_empty() {
                self.write_input_buffer();
                self.promote_prefills_to_decodes();
            }
            // ── 气泡结束, Runner 开始跑 ──

            // ── 慢路径 (与 Runner 并行): 判 EOS + ZMQ 发送 ──
            self.process_eos_and_send_zmq(&output_tokens);

            // 所有请求结束且无新 prefill → 阻塞等
            if self.active_decodes.is_empty() && self.pending_prefills.is_empty() {
                tracing::debug!("All sequences finished, waiting for new prefill...");
                let data = self.zmq_in.recv_bytes(0).expect("ZMQ recv failed");
                let cmd: PrefillBatchCmd =
                    rmp_serde::from_slice(&data).expect("Failed to deserialize PrefillBatchCmd");
                self.pending_prefills.push(cmd);
            }
        }
    }

    /// 快路径: 只更新 last_token (不动 position, position 在 write 之后更新)
    fn update_decode_tokens(&mut self, tokens: &[i32]) {
        for (i, seq) in self.active_decodes.iter_mut().enumerate() {
            seq.last_token = tokens[i];
            seq.generated_count += 1;
        }
    }

    /// 慢路径: 判 EOS / max_tokens, 移除已结束, 发 ZMQ_OUT
    fn process_eos_and_send_zmq(&mut self, tokens: &[i32]) {
        let mut step_output = StepOutput {
            tokens: Vec::with_capacity(tokens.len()),
        };
        let mut write_idx = 0;

        for i in 0..self.active_decodes.len() {
            let seq = &self.active_decodes[i];
            let token_id = tokens[i];
            let finished =
                token_id == self.eos_token_id || seq.generated_count >= seq.max_tokens;

            step_output.tokens.push(SeqToken {
                request_id: seq.request_id.clone(),
                token_id,
                finished,
            });

            if !finished {
                if write_idx != i {
                    self.active_decodes.swap(write_idx, i);
                }
                write_idx += 1;
            }
        }
        self.active_decodes.truncate(write_idx);

        let data = rmp_serde::to_vec(&step_output).expect("Failed to serialize StepOutput");
        self.zmq_out.send(&data, 0).expect("ZMQ send failed");
    }

    /// 写入共享 input buffer: Decode 在前, Prefill 在后
    /// 写入后 decode seq 的 next_position 自动 +1
    fn write_input_buffer(&mut self) {
        let num_decode = self.active_decodes.len();
        let num_prefill_seqs: usize = self
            .pending_prefills
            .iter()
            .map(|p| p.num_requests())
            .sum();
        let num_prefill_tokens: usize =
            self.pending_prefills.iter().map(|p| p.input_ids.len()).sum();
        let total_tokens = num_decode + num_prefill_tokens;
        let num_seqs = num_decode + num_prefill_seqs;

        let mut token_ids = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        let mut q_start_loc = Vec::with_capacity(num_seqs + 1);
        let mut context_lens = Vec::with_capacity(num_seqs);
        let mut slot_indices = Vec::with_capacity(num_seqs);
        let mut offset = 0u32;

        // Decode 在前
        for d in &self.active_decodes {
            token_ids.push(d.last_token);
            positions.push(d.next_position as i32);
            q_start_loc.push(offset);
            offset += 1;
            context_lens.push(d.next_position as i32);
            slot_indices.push(d.kv_slot as i32);
        }

        // 写入后 decode seq 的 position +1 (下一步用)
        for d in &mut self.active_decodes {
            d.next_position += 1;
        }

        // Prefill 在后
        for p in &self.pending_prefills {
            let n = p.num_requests();
            for i in 0..n {
                let start = p.q_start_loc[i] as usize;
                let end = if i + 1 < n {
                    p.q_start_loc[i + 1] as usize
                } else {
                    p.input_ids.len()
                };
                let seq_len = end - start;
                let computed = p.num_computed_tokens[i] as usize;

                token_ids.extend_from_slice(&p.input_ids[start..end]);
                positions.extend((0..seq_len).map(|j| (computed + j) as i32));
                q_start_loc.push(offset);
                offset += seq_len as u32;
                context_lens.push(computed as i32);
                slot_indices.push(p.kv_slots[i] as i32);
            }
        }
        q_start_loc.push(offset); // 末尾哨兵

        // H2D copy 到共享 buffer
        // q_start_loc 需要从 u32 转为 i32 (GPU buffer 统一 i32)
        let q_start_loc_i32: Vec<i32> = q_start_loc.iter().map(|&v| v as i32).collect();

        self.shared.write_input_i32(&self.shared.input_token_ids, &token_ids, total_tokens)
            .expect("Failed to write input_token_ids");
        self.shared.write_input_i32(&self.shared.input_positions, &positions, total_tokens)
            .expect("Failed to write input_positions");
        self.shared.write_input_i32(&self.shared.input_q_start_loc, &q_start_loc_i32, num_seqs + 1)
            .expect("Failed to write input_q_start_loc");
        self.shared.write_input_i32(&self.shared.input_context_lens, &context_lens, num_seqs)
            .expect("Failed to write input_context_lens");
        self.shared.write_input_i32(&self.shared.input_slot_indices, &slot_indices, num_seqs)
            .expect("Failed to write input_slot_indices");

        // 写元信息, ready 最后 store
        let batch_type: u8 = if num_prefill_seqs == 0 { 0 } else { 1 };
        self.shared
            .input_meta
            .batch_type
            .store(batch_type, Release);
        self.shared
            .input_meta
            .num_decode_seqs
            .store(num_decode as u32, Release);
        self.shared
            .input_meta
            .num_prefill_seqs
            .store(num_prefill_seqs as u32, Release);
        self.shared
            .input_meta
            .num_prefill_tokens
            .store(num_prefill_tokens as u32, Release);
        // 最后发信号
        self.shared
            .input_meta
            .ready
            .store(total_tokens as u32, Release);
    }

    /// Prefill 写入 input 后,对应序列进入 decode 状态
    fn promote_prefills_to_decodes(&mut self) {
        for cmd in self.pending_prefills.drain(..) {
            for i in 0..cmd.num_requests() {
                let start = cmd.q_start_loc[i] as usize;
                let end = if i + 1 < cmd.num_requests() {
                    cmd.q_start_loc[i + 1] as usize
                } else {
                    cmd.input_ids.len()
                };
                let seq_len = end - start;
                let computed = cmd.num_computed_tokens[i] as usize;

                self.active_decodes.push(DecodeSeq {
                    request_id: cmd.request_metas[i].request_id.clone(),
                    kv_slot: cmd.kv_slots[i] as usize,
                    next_position: computed + seq_len,
                    last_token: 0, // 首个 decode token 来自 Runner 输出
                    generated_count: 0,
                    max_tokens: cmd.request_metas[i].max_tokens,
                    sampling: cmd.sampling_params[i].clone(),
                });
            }
        }
    }

    /// 非阻塞收所有待处理的 prefill
    fn drain_zmq_prefills(&mut self) {
        loop {
            match self.zmq_in.recv_bytes(zmq::DONTWAIT) {
                Ok(data) => {
                    match rmp_serde::from_slice::<PrefillBatchCmd>(&data) {
                        Ok(cmd) => self.pending_prefills.push(cmd),
                        Err(e) => tracing::error!("Failed to deserialize PrefillBatchCmd: {}", e),
                    }
                }
                Err(zmq::Error::EAGAIN) => break, // 没有更多消息
                Err(e) => {
                    tracing::error!("ZMQ recv error: {}", e);
                    break;
                }
            }
        }
    }

    /// spin wait output ready
    fn wait_output_ready(&self) -> usize {
        loop {
            let v = self.shared.output_meta.ready.load(Acquire);
            if v > 0 {
                return v as usize;
            }
            std::hint::spin_loop();
        }
    }

    /// D2H copy output tokens
    fn read_output_tokens(&self, num_seqs: usize) -> Vec<i32> {
        self.shared.read_output_i32(&self.shared.output_token_ids, num_seqs)
            .expect("Failed to read output_token_ids")
    }
}
