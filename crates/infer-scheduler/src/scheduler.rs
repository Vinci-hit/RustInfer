//! Scheduler —— continuous batching 调度器。
//!
//! 职责：
//! - ZMQ ROUTER 收 `InferenceRequest` from HTTP Server
//! - Tokenize prompt → 入 waiting queue
//! - 管理 slot 池，为新请求分配 slot
//! - 每步选择哪些 waiting 做 prefill，满足 token budget
//! - 组装 `PrefillBatchCmd` 发给 Worker (ZMQ PUSH)
//! - 从 Worker 收 `StepOutput` (ZMQ PULL)
//! - 序列完成后 decode tokens → 回复 HTTP Server

use std::collections::VecDeque;
use std::time::Instant;

use infer_protocol::{InferenceRequest, InferenceResponse, InferenceMetrics, ResponseStatus};
use infer_worker::worker::protocol::*;

// ════════════════════════════════════════════════════════════════════════════════
//  SlotPool
// ════════════════════════════════════════════════════════════════════════════════

/// KV cache slot 位图管理。
struct SlotPool {
    /// true = 已占用
    slots: Vec<bool>,
}

impl SlotPool {
    fn new(max_slots: usize) -> Self {
        Self { slots: vec![false; max_slots] }
    }

    /// 分配一个空闲 slot，返回 slot id。无空闲时返回 None。
    fn alloc(&mut self) -> Option<usize> {
        for (i, occupied) in self.slots.iter_mut().enumerate() {
            if !*occupied {
                *occupied = true;
                return Some(i);
            }
        }
        None
    }

    /// 释放 slot。
    fn free(&mut self, slot: usize) {
        if slot < self.slots.len() {
            self.slots[slot] = false;
        }
    }

    /// 当前空闲数。
    #[allow(dead_code)]
    fn available(&self) -> usize {
        self.slots.iter().filter(|&&s| !s).count()
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Request / Seq 状态
// ════════════════════════════════════════════════════════════════════════════════

/// 等待 prefill 的请求。
struct WaitingRequest {
    request_id: String,
    /// ZMQ ROUTER identity frame（用于回复）
    identity: Vec<u8>,
    /// 已 tokenize 的输入
    input_ids: Vec<i32>,
    max_tokens: usize,
    sampling: SamplingParams,
    #[allow(dead_code)]
    enqueue_time: Instant,
}

/// 正在 decode 的序列。
struct RunningSeq {
    request_id: String,
    identity: Vec<u8>,
    kv_slot: usize,
    /// 累积所有输出 token
    generated_tokens: Vec<i32>,
    #[allow(dead_code)]
    max_tokens: usize,
    #[allow(dead_code)]
    sampling: SamplingParams,
    start_time: Instant,
}

// ════════════════════════════════════════════════════════════════════════════════
//  Scheduler
// ════════════════════════════════════════════════════════════════════════════════

pub struct Scheduler {
    // 状态
    waiting: VecDeque<WaitingRequest>,
    running: Vec<RunningSeq>,
    slot_pool: SlotPool,

    // Worker 通信 (PUSH/PULL)
    zmq_to_worker: zmq::Socket,
    zmq_from_worker: zmq::Socket,

    // HTTP Server 通信 (ROUTER)
    zmq_frontend: zmq::Socket,

    // 约束
    max_batch_tokens: usize,
    max_batch_seqs: usize,
}

impl Scheduler {
    pub fn new(
        zmq_frontend: zmq::Socket,
        zmq_to_worker: zmq::Socket,
        zmq_from_worker: zmq::Socket,
        max_batch_tokens: usize,
        max_batch_seqs: usize,
    ) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            slot_pool: SlotPool::new(max_batch_seqs),
            zmq_to_worker,
            zmq_from_worker,
            zmq_frontend,
            max_batch_tokens,
            max_batch_seqs,
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  主循环
    // ════════════════════════════════════════════════════════════════════════════

    pub fn run(mut self) {
        tracing::info!("Scheduler started, max_batch_seqs={}, max_batch_tokens={}",
            self.max_batch_seqs, self.max_batch_tokens);

        // 第一次启动：等到有请求来
        self.wait_for_first_request();

        loop {
            // 1. 非阻塞收 HTTP Server 请求
            self.drain_frontend_requests();

            // 2. 调度：从 waiting 选请求，组装 PrefillBatchCmd
            let maybe_cmd = self.schedule_step();

            // 3. 发给 Worker（有新 prefill 时）
            if let Some(cmd) = maybe_cmd {
                self.send_prefill_to_worker(&cmd);
            }
            // Worker 有 active decode 时会自动继续，不需要 Scheduler 每步驱动

            // 4. 等 Worker StepOutput
            if !self.running.is_empty() {
                match self.recv_step_output() {
                    Some(output) => self.process_step_output(&output),
                    None => {
                        tracing::warn!("Failed to receive StepOutput from Worker");
                    }
                }
            }

            // 5. 空闲：无 running 且无 waiting → 等待新请求
            if self.running.is_empty() && self.waiting.is_empty() {
                tracing::debug!("Scheduler idle, waiting for requests...");
                self.wait_for_first_request();
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  调度策略
    // ════════════════════════════════════════════════════════════════════════════

    /// 从 waiting queue 选请求做 prefill，满足 token/seq budget。
    /// 返回 PrefillBatchCmd（如果有新请求要 prefill），否则 None。
    fn schedule_step(&mut self) -> Option<PrefillBatchCmd> {
        if self.waiting.is_empty() {
            return None;
        }

        let seq_budget = self.max_batch_seqs.saturating_sub(self.running.len());
        // token budget: running 中每条占 1 token (decode)
        let token_budget = self.max_batch_tokens.saturating_sub(self.running.len());

        if seq_budget == 0 || token_budget == 0 {
            return None; // 满了，等下一步
        }

        let mut selected: Vec<WaitingRequest> = Vec::new();
        let mut tokens_used = 0usize;

        while let Some(req) = self.waiting.front() {
            let prompt_len = req.input_ids.len();
            if selected.len() >= seq_budget {
                break;
            }
            if tokens_used + prompt_len > token_budget {
                break;
            }
            // pop and take
            let req = self.waiting.pop_front().unwrap();
            tokens_used += prompt_len;
            selected.push(req);
        }

        if selected.is_empty() {
            return None;
        }

        // 分配 slot + 组装 PrefillBatchCmd
        let mut input_ids_all: Vec<i32> = Vec::new();
        let mut q_start_loc: Vec<u32> = Vec::new();
        let mut num_computed_tokens: Vec<u32> = Vec::new();
        let mut kv_slots: Vec<u32> = Vec::new();
        let mut sampling_params: Vec<SamplingParams> = Vec::new();
        let mut request_metas: Vec<RequestMeta> = Vec::new();

        for req in selected {
            let slot = match self.slot_pool.alloc() {
                Some(s) => s,
                None => {
                    // 不应该走到这里（seq_budget 限制了），但保险起见放回 waiting
                    tracing::error!("SlotPool exhausted unexpectedly");
                    self.waiting.push_front(req);
                    break;
                }
            };

            q_start_loc.push(input_ids_all.len() as u32);
            input_ids_all.extend_from_slice(&req.input_ids);
            num_computed_tokens.push(0); // Phase 1: 不做 chunked prefill
            kv_slots.push(slot as u32);
            sampling_params.push(req.sampling.clone());
            request_metas.push(RequestMeta {
                request_id: req.request_id.clone(),
                max_tokens: req.max_tokens,
            });

            // 移入 running
            self.running.push(RunningSeq {
                request_id: req.request_id,
                identity: req.identity,
                kv_slot: slot,
                generated_tokens: Vec::new(),
                max_tokens: req.max_tokens,
                sampling: req.sampling,
                start_time: Instant::now(),
            });
        }

        Some(PrefillBatchCmd {
            input_ids: input_ids_all,
            q_start_loc,
            num_computed_tokens,
            kv_slots,
            sampling_params,
            request_metas,
        })
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  处理 Worker 输出
    // ════════════════════════════════════════════════════════════════════════════

    fn process_step_output(&mut self, output: &StepOutput) {
        let mut finished_ids: Vec<String> = Vec::new();

        for seq_token in &output.tokens {
            // 找到对应的 running seq
            if let Some(seq) = self.running.iter_mut()
                .find(|s| s.request_id == seq_token.request_id)
            {
                seq.generated_tokens.push(seq_token.token_id);

                if seq_token.finished {
                    finished_ids.push(seq_token.request_id.clone());
                }
            } else {
                tracing::warn!("Received token for unknown seq: {}", seq_token.request_id);
            }
        }

        // 完成的 seq：decode text → 回复 → 释放 slot
        for req_id in finished_ids {
            let idx = self.running.iter().position(|s| s.request_id == req_id);
            if let Some(idx) = idx {
                let seq = self.running.remove(idx);
                self.slot_pool.free(seq.kv_slot);

                let elapsed_ms = seq.start_time.elapsed().as_millis() as u64;
                let num_tokens = seq.generated_tokens.len() as u32;
                let tokens_per_second = if elapsed_ms > 0 {
                    (num_tokens as f64 / elapsed_ms as f64) * 1000.0
                } else {
                    0.0
                };

                let response = InferenceResponse {
                    request_id: seq.request_id.clone(),
                    status: ResponseStatus::Success,
                    output_token_ids: seq.generated_tokens,
                    error: None,
                    metrics: InferenceMetrics {
                        total_ms: elapsed_ms,
                        num_tokens,
                        tokens_per_second,
                    },
                };

                self.send_response_to_frontend(&seq.identity, &response);
                tracing::info!(
                    "✅ {} completed: {} tokens in {}ms ({:.1} tok/s)",
                    seq.request_id, num_tokens, elapsed_ms, tokens_per_second,
                );
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  ZMQ Frontend (HTTP Server 通信)
    // ════════════════════════════════════════════════════════════════════════════

    /// 阻塞等待第一个请求到来。
    fn wait_for_first_request(&mut self) {
        loop {
            match self.recv_one_frontend_request(true) {
                Some(_) => return,
                None => continue,
            }
        }
    }

    /// 非阻塞收所有待处理的 HTTP 请求。
    fn drain_frontend_requests(&mut self) {
        loop {
            match self.recv_one_frontend_request(false) {
                Some(_) => {} // 继续收
                None => break,
            }
        }
    }

    /// 收一条 frontend 请求，tokenize，入 waiting queue。
    /// blocking=true 时阻塞等待。返回 Some(request_id) 表示成功入队。
    fn recv_one_frontend_request(&mut self, blocking: bool) -> Option<String> {
        let flags = if blocking { 0 } else { zmq::DONTWAIT };

        // ROUTER frame: [identity, empty, data]
        let identity = match self.zmq_frontend.recv_bytes(flags) {
            Ok(id) => id,
            Err(zmq::Error::EAGAIN) => return None,
            Err(e) => { tracing::error!("Frontend recv identity error: {:?}", e); return None; }
        };
        // empty delimiter
        if let Err(e) = self.zmq_frontend.recv_bytes(0) {
            tracing::error!("Frontend recv empty frame error: {:?}", e);
            return None;
        }
        // data
        let data = match self.zmq_frontend.recv_bytes(0) {
            Ok(d) => d,
            Err(e) => { tracing::error!("Frontend recv data error: {:?}", e); return None; }
        };

        let request: InferenceRequest = match rmp_serde::from_slice(&data) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to deserialize InferenceRequest: {:?}", e);
                return None;
            }
        };

        tracing::info!("Received request {}: {} input tokens", request.request_id, request.input_ids.len());

        let request_id = request.request_id.clone();
        self.waiting.push_back(WaitingRequest {
            request_id: request.request_id,
            identity,
            input_ids: request.input_ids,
            max_tokens: request.max_tokens,
            sampling: SamplingParams {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            },
            enqueue_time: Instant::now(),
        });

        Some(request_id)
    }

    /// 通过 ZMQ ROUTER 回复 HTTP Server。
    fn send_response_to_frontend(&self, identity: &[u8], response: &InferenceResponse) {
        let data = match rmp_serde::to_vec(response) {
            Ok(d) => d,
            Err(e) => { tracing::error!("Failed to serialize response: {:?}", e); return; }
        };

        if let Err(e) = self.zmq_frontend.send(identity, zmq::SNDMORE) {
            tracing::error!("Frontend send identity error: {:?}", e); return;
        }
        if let Err(e) = self.zmq_frontend.send(&b""[..], zmq::SNDMORE) {
            tracing::error!("Frontend send empty frame error: {:?}", e); return;
        }
        if let Err(e) = self.zmq_frontend.send(&data, 0) {
            tracing::error!("Frontend send data error: {:?}", e);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    //  ZMQ Worker 通信
    // ════════════════════════════════════════════════════════════════════════════

    /// 发 PrefillBatchCmd 给 Worker。
    fn send_prefill_to_worker(&self, cmd: &PrefillBatchCmd) {
        let data = match rmp_serde::to_vec(cmd) {
            Ok(d) => d,
            Err(e) => { tracing::error!("Failed to serialize PrefillBatchCmd: {:?}", e); return; }
        };
        if let Err(e) = self.zmq_to_worker.send(&data, 0) {
            tracing::error!("Failed to send PrefillBatchCmd to Worker: {:?}", e);
        }
    }

    /// 从 Worker 收 StepOutput（阻塞等待，带超时）。
    fn recv_step_output(&self) -> Option<StepOutput> {
        match self.zmq_from_worker.recv_bytes(0) {
            Ok(data) => {
                match rmp_serde::from_slice::<StepOutput>(&data) {
                    Ok(output) => Some(output),
                    Err(e) => {
                        tracing::error!("Failed to deserialize StepOutput: {:?}", e);
                        None
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to recv StepOutput: {:?}", e);
                None
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── SlotPool 单元测试 ───

    #[test]
    fn slot_pool_alloc_and_free() {
        let mut pool = SlotPool::new(4);
        assert_eq!(pool.available(), 4);

        let s0 = pool.alloc().unwrap();
        assert_eq!(s0, 0);
        assert_eq!(pool.available(), 3);

        let s1 = pool.alloc().unwrap();
        assert_eq!(s1, 1);

        let s2 = pool.alloc().unwrap();
        let s3 = pool.alloc().unwrap();
        assert_eq!(pool.available(), 0);

        // 满了
        assert_eq!(pool.alloc(), None);

        // 释放 s1
        pool.free(s1);
        assert_eq!(pool.available(), 1);

        // 再分配应该拿到 s1 (最小空闲)
        let s1_again = pool.alloc().unwrap();
        assert_eq!(s1_again, 1);
        assert_eq!(pool.available(), 0);

        // 释放全部
        pool.free(s0);
        pool.free(s1_again);
        pool.free(s2);
        pool.free(s3);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn slot_pool_free_out_of_range() {
        let mut pool = SlotPool::new(2);
        pool.free(999); // 不 panic
        assert_eq!(pool.available(), 2);
    }

    // ─── Scheduler schedule_step 测试 ───

    fn make_test_scheduler(max_batch_tokens: usize, max_batch_seqs: usize) -> Scheduler {
        let zmq_ctx = zmq::Context::new();
        let id = uuid::Uuid::new_v4().to_string();
        let frontend = zmq_ctx.socket(zmq::ROUTER).unwrap();
        frontend.bind(&format!("inproc://test-fe-{}", id)).unwrap();
        let to_worker = zmq_ctx.socket(zmq::PUSH).unwrap();
        to_worker.bind(&format!("inproc://test-push-{}", id)).unwrap();
        let from_worker = zmq_ctx.socket(zmq::PULL).unwrap();
        from_worker.bind(&format!("inproc://test-pull-{}", id)).unwrap();

        Scheduler::new(frontend, to_worker, from_worker, max_batch_tokens, max_batch_seqs)
    }

    fn push_waiting(sched: &mut Scheduler, request_id: &str, prompt_len: usize, max_tokens: usize) {
        sched.waiting.push_back(WaitingRequest {
            request_id: request_id.to_string(),
            identity: vec![0u8; 4],
            input_ids: vec![1i32; prompt_len], // dummy tokens
            max_tokens,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 },
            enqueue_time: Instant::now(),
        });
    }

    #[test]
    fn schedule_step_empty_waiting() {
        let mut sched = make_test_scheduler(512, 4);
        assert!(sched.schedule_step().is_none());
    }

    #[test]
    fn schedule_step_single_request() {
        let mut sched = make_test_scheduler(512, 4);
        push_waiting(&mut sched, "req-1", 10, 20);

        let cmd = sched.schedule_step().unwrap();
        assert_eq!(cmd.input_ids.len(), 10);
        assert_eq!(cmd.num_requests(), 1);
        assert_eq!(cmd.kv_slots[0], 0);
        assert_eq!(sched.running.len(), 1);
        assert_eq!(sched.running[0].request_id, "req-1");
        assert_eq!(sched.running[0].kv_slot, 0);
        assert!(sched.waiting.is_empty());
    }

    #[test]
    fn schedule_step_respects_seq_budget() {
        let mut sched = make_test_scheduler(512, 2);
        push_waiting(&mut sched, "req-1", 10, 20);
        push_waiting(&mut sched, "req-2", 10, 20);
        push_waiting(&mut sched, "req-3", 10, 20);

        let cmd = sched.schedule_step().unwrap();
        // max_batch_seqs=2, 只能调度 2 个
        assert_eq!(cmd.num_requests(), 2);
        assert_eq!(sched.running.len(), 2);
        assert_eq!(sched.waiting.len(), 1); // req-3 留在 waiting
    }

    #[test]
    fn schedule_step_respects_token_budget() {
        let mut sched = make_test_scheduler(25, 4);
        push_waiting(&mut sched, "req-1", 10, 20);
        push_waiting(&mut sched, "req-2", 10, 20);
        push_waiting(&mut sched, "req-3", 10, 20);

        let cmd = sched.schedule_step().unwrap();
        // token budget = 25, 每个 10 tokens, 最多 2 个
        assert_eq!(cmd.num_requests(), 2);
        assert_eq!(cmd.input_ids.len(), 20);
        assert_eq!(sched.waiting.len(), 1);
    }

    #[test]
    fn schedule_step_running_reduces_budget() {
        let mut sched = make_test_scheduler(512, 2);
        // 模拟已有 1 个 running seq
        sched.running.push(RunningSeq {
            request_id: "existing".to_string(),
            identity: vec![],
            kv_slot: 0,
            generated_tokens: vec![],
            max_tokens: 10,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 },
            start_time: Instant::now(),
        });
        sched.slot_pool.alloc(); // 占一个 slot

        push_waiting(&mut sched, "req-1", 10, 20);
        push_waiting(&mut sched, "req-2", 10, 20);

        let cmd = sched.schedule_step().unwrap();
        // seq_budget = 2 - 1 = 1, 只能调度 1 个
        assert_eq!(cmd.num_requests(), 1);
        assert_eq!(sched.running.len(), 2); // existing + new
        assert_eq!(sched.waiting.len(), 1);
    }

    #[test]
    fn schedule_step_slots_full_returns_none() {
        let mut sched = make_test_scheduler(512, 2);
        // 占满 slot
        sched.slot_pool.alloc();
        sched.slot_pool.alloc();
        sched.running.push(RunningSeq {
            request_id: "a".to_string(), identity: vec![], kv_slot: 0,
            generated_tokens: vec![], max_tokens: 10,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 },
            start_time: Instant::now(),
        });
        sched.running.push(RunningSeq {
            request_id: "b".to_string(), identity: vec![], kv_slot: 1,
            generated_tokens: vec![], max_tokens: 10,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 },
            start_time: Instant::now(),
        });

        push_waiting(&mut sched, "req-1", 10, 20);
        assert!(sched.schedule_step().is_none());
    }

    // ─── process_step_output 测试 ───

    #[test]
    fn process_output_accumulates_tokens() {
        let mut sched = make_test_scheduler(512, 4);
        sched.running.push(RunningSeq {
            request_id: "req-1".to_string(),
            identity: vec![],
            kv_slot: 0,
            generated_tokens: vec![],
            max_tokens: 10,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 },
            start_time: Instant::now(),
        });
        sched.slot_pool.alloc();

        let output = StepOutput {
            tokens: vec![SeqToken {
                request_id: "req-1".to_string(),
                token_id: 42,
                finished: false,
            }],
        };
        sched.process_step_output(&output);

        assert_eq!(sched.running.len(), 1);
        assert_eq!(sched.running[0].generated_tokens, vec![42]);
        assert_eq!(sched.slot_pool.available(), 3); // slot 还占着
    }

    #[test]
    fn process_output_finished_releases_slot() {
        let mut sched = make_test_scheduler(512, 4);
        sched.running.push(RunningSeq {
            request_id: "req-1".to_string(),
            identity: vec![],
            kv_slot: 2,
            generated_tokens: vec![10, 20],
            max_tokens: 10,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 },
            start_time: Instant::now(),
        });
        sched.slot_pool.slots[2] = true; // 手动标记占用

        let output = StepOutput {
            tokens: vec![SeqToken {
                request_id: "req-1".to_string(),
                token_id: 99,
                finished: true,
            }],
        };
        sched.process_step_output(&output);

        // seq 完成 → 从 running 移除 + slot 释放
        assert_eq!(sched.running.len(), 0);
        assert_eq!(sched.slot_pool.available(), 4); // slot 2 已释放
    }
}

