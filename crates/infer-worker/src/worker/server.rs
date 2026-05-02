use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;

use crate::base::error::{Error, Result};
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
    /// 采样参数。Phase-1 sampler 走 greedy，没有读这个字段；
    /// 保留给后续接入 top_p / temperature。
    #[allow(dead_code)]
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
    /// 当前 Worker 绑定的 GPU 设备号。目前 single-GPU 模式下只用来打日志；
    /// 多卡 scheduler 引入后会用于选择 CUDA context。
    #[allow(dead_code)]
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

    fn validate_prefill(&self, cmd: &PrefillBatchCmd) -> Result<()> {
        cmd.validate(self.shared.max_batch_tokens, self.shared.max_seqs, self.shared.max_seqs)
    }

    fn recv_next_prefill_blocking(&mut self) -> PrefillBatchCmd {
        loop {
            match self.zmq_in.recv_bytes(0) {
                Ok(data) => match rmp_serde::from_slice::<PrefillBatchCmd>(&data) {
                    Ok(cmd) => match self.validate_prefill(&cmd) {
                        Ok(()) => return cmd,
                        Err(e) => tracing::error!("Reject invalid PrefillBatchCmd: {}", e),
                    },
                    Err(e) => tracing::error!("Failed to deserialize PrefillBatchCmd: {}", e),
                },
                Err(e) => tracing::error!("ZMQ recv error while waiting for prefill: {}", e),
            }
        }
    }

    fn enqueue_next_step(&mut self) {
        if self.active_decodes.is_empty() && self.pending_prefills.is_empty() {
            return;
        }

        match self.write_input_buffer() {
            Ok(()) => self.promote_prefills_to_decodes(),
            Err(e) => {
                tracing::error!("Failed to write worker input buffer; dropping pending prefills: {}", e);
                self.pending_prefills.clear();
                if !self.active_decodes.is_empty() {
                    self.write_input_buffer()
                        .expect("active decodes must fit in worker input buffer");
                }
            }
        }
    }

    /// 主循环
    pub fn run(mut self) {
        // ══ 启动: 阻塞等第一个有效 prefill ══
        tracing::info!("WorkerServer waiting for first prefill...");
        let cmd = self.recv_next_prefill_blocking();
        self.pending_prefills.push(cmd);

        self.enqueue_next_step();
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

            // 3. 先判 EOS / max_tokens 并移除已结束序列，避免下一步继续 decode finished 请求。
            self.process_eos_and_send_zmq(&output_tokens);

            // 4. 收新 prefill (非阻塞)
            self.drain_zmq_prefills();

            // 5. 基于过滤后的 active_decodes 写 input buffer → signal Runner
            self.enqueue_next_step();

            // 所有请求结束且无新 prefill → 阻塞等
            if self.active_decodes.is_empty() && self.pending_prefills.is_empty() {
                tracing::debug!("All sequences finished, waiting for new prefill...");
                let cmd = self.recv_next_prefill_blocking();
                self.pending_prefills.push(cmd);
                self.enqueue_next_step();
            }
        }
    }

    /// 快路径: 只更新 last_token (不动 position, position 在 write 之后更新)
    fn update_decode_tokens(&mut self, tokens: &[i32]) {
        if tokens.len() != self.active_decodes.len() {
            tracing::error!(
                "Runner output length {} != active decode length {}",
                tokens.len(),
                self.active_decodes.len()
            );
            return;
        }
        for (seq, &token) in self.active_decodes.iter_mut().zip(tokens) {
            seq.last_token = token;
            seq.generated_count += 1;
        }
    }

    /// 慢路径: 判 EOS / max_tokens, 移除已结束, 发 ZMQ_OUT
    fn process_eos_and_send_zmq(&mut self, tokens: &[i32]) {
        if tokens.len() != self.active_decodes.len() {
            tracing::error!(
                "Skip EOS processing: Runner output length {} != active decode length {}",
                tokens.len(),
                self.active_decodes.len()
            );
            return;
        }
        let mut step_output = StepOutput {
            tokens: Vec::with_capacity(tokens.len()),
        };
        let old_decodes = std::mem::take(&mut self.active_decodes);

        for (seq, &token_id) in old_decodes.into_iter().zip(tokens) {
            let finished =
                token_id == self.eos_token_id || seq.generated_count >= seq.max_tokens;

            step_output.tokens.push(SeqToken {
                request_id: seq.request_id.clone(),
                token_id,
                finished,
            });

            if !finished {
                self.active_decodes.push(seq);
            }
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

    /// 写入共享 input buffer: Decode 在前, Prefill 在后
    /// 写入后 decode seq 的 next_position 自动 +1
    fn write_input_buffer(&mut self) -> Result<()> {
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

        if total_tokens > self.shared.max_batch_tokens {
            return Err(Error::InvalidArgument(format!(
                "worker input has {} tokens, exceeds max_batch_tokens {}",
                total_tokens, self.shared.max_batch_tokens
            )).into());
        }
        if num_seqs > self.shared.max_seqs {
            return Err(Error::InvalidArgument(format!(
                "worker input has {} seqs, exceeds max_seqs {}",
                num_seqs, self.shared.max_seqs
            )).into());
        }

        let mut token_ids = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        let mut q_start_loc = Vec::with_capacity(num_seqs + 1);
        let mut context_lens = Vec::with_capacity(num_seqs);
        let mut slot_indices = Vec::with_capacity(num_seqs);
        let mut offset = 0usize;

        // Decode 在前
        for d in &self.active_decodes {
            token_ids.push(d.last_token);
            positions.push(i32::try_from(d.next_position).map_err(|_| {
                Error::InvalidArgument(format!("decode position {} exceeds i32", d.next_position))
            })?);
            q_start_loc.push(offset);
            offset += 1;
            context_lens.push(i32::try_from(d.next_position).map_err(|_| {
                Error::InvalidArgument(format!("context len {} exceeds i32", d.next_position))
            })?);
            slot_indices.push(i32::try_from(d.kv_slot).map_err(|_| {
                Error::InvalidArgument(format!("kv_slot {} exceeds i32", d.kv_slot))
            })?);
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
                let pos_start = computed;
                let pos_end = computed.checked_add(seq_len).ok_or_else(|| {
                    Error::InvalidArgument("prefill position overflow".into())
                })?;
                for pos in pos_start..pos_end {
                    positions.push(i32::try_from(pos).map_err(|_| {
                        Error::InvalidArgument(format!("prefill position {} exceeds i32", pos))
                    })?);
                }
                q_start_loc.push(offset);
                offset = offset.checked_add(seq_len).ok_or_else(|| {
                    Error::InvalidArgument("q_start_loc offset overflow".into())
                })?;
                context_lens.push(i32::try_from(computed).map_err(|_| {
                    Error::InvalidArgument(format!("computed tokens {} exceeds i32", computed))
                })?);
                slot_indices.push(i32::try_from(p.kv_slots[i]).map_err(|_| {
                    Error::InvalidArgument(format!("kv_slot {} exceeds i32", p.kv_slots[i]))
                })?);
            }
        }
        q_start_loc.push(offset); // 末尾哨兵

        // 写共享 CPU 交换区。安全性来自 SharedBuffers 的信号协议：
        // 调用 write_input_buffer 时 input_meta.ready == 0，Runner 不会读 input。
        let q_start_loc_i32: Vec<i32> = q_start_loc
            .iter()
            .map(|&v| i32::try_from(v).map_err(|_| {
                Error::InvalidArgument(format!("q_start_loc {} exceeds i32", v))
            }))
            .collect::<std::result::Result<_, _>>()?;
        unsafe {
            self.shared.input_token_ids.as_mut_slice(total_tokens)
                .copy_from_slice(&token_ids);
            self.shared.input_positions.as_mut_slice(total_tokens)
                .copy_from_slice(&positions);
            self.shared.input_q_start_loc.as_mut_slice(num_seqs + 1)
                .copy_from_slice(&q_start_loc_i32);
            self.shared.input_context_lens.as_mut_slice(num_seqs)
                .copy_from_slice(&context_lens);
            self.shared.input_slot_indices.as_mut_slice(num_seqs)
                .copy_from_slice(&slot_indices);
        }

        // 写元信息, ready 最后 store
        let batch_type: u8 = if num_prefill_seqs == 0 { 0 } else { 1 };
        self.shared
            .input_meta
            .batch_type
            .store(batch_type, Release);
        self.shared
            .input_meta
            .num_decode_seqs
            .store(u32::try_from(num_decode).map_err(|_| {
                Error::InvalidArgument(format!("num_decode {} exceeds u32", num_decode))
            })?, Release);
        self.shared
            .input_meta
            .num_prefill_seqs
            .store(u32::try_from(num_prefill_seqs).map_err(|_| {
                Error::InvalidArgument(format!("num_prefill_seqs {} exceeds u32", num_prefill_seqs))
            })?, Release);
        self.shared
            .input_meta
            .num_prefill_tokens
            .store(u32::try_from(num_prefill_tokens).map_err(|_| {
                Error::InvalidArgument(format!("num_prefill_tokens {} exceeds u32", num_prefill_tokens))
            })?, Release);
        // 最后发信号
        self.shared
            .input_meta
            .ready
            .store(u32::try_from(total_tokens).map_err(|_| {
                Error::InvalidArgument(format!("total_tokens {} exceeds u32", total_tokens))
            })?, Release);
        Ok(())
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
                        Ok(cmd) => match self.validate_prefill(&cmd) {
                            Ok(()) => self.pending_prefills.push(cmd),
                            Err(e) => tracing::error!("Reject invalid PrefillBatchCmd: {}", e),
                        },
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

    /// 读 Runner 写好的 output tokens。
    fn read_output_tokens(&self, num_seqs: usize) -> Vec<i32> {
        // SAFETY: 调用点保证 output_meta.ready > 0，Runner 不再写 output_token_ids。
        unsafe {
            self.shared.output_token_ids.as_slice(num_seqs).to_vec()
        }
    }
}
