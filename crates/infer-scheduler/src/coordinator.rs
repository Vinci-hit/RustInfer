//! Coordinator - 核心调度器
//!
//! 将所有组件（State, Policy, Memory, Transport）串联起来的核心模块
//!
//! # 主循环流程
//!
//! ```text
//! Phase 1: Ingest (摄入) - 从前端接收新请求
//!   └─ 将请求加入 Waiting 队列
//!
//! Phase 2: Schedule (决策) - 策略模块决定下一步
//!   └─ 返回 BatchPlan (prefill/decode/preempt)
//!
//! Phase 3: Apply Plan & Prepare (副作用与转换)
//!   ├─ 处理抢占（释放显存）
//!   ├─ 处理 Prefill（分配显存）
//!   ├─ 处理 Decode（追加显存）
//!   └─ 构建 WorkerInput
//!
//! Phase 4: RPC Call (执行) - 调用 GPU Worker
//!   └─ worker.forward(params)
//!
//! Phase 5: Post-Process (后处理) - 更新状态
//!   ├─ 更新 generated_tokens
//!   ├─ 检查是否完成
//!   ├─ 流式返回结果
//!   └─ 插入 RadixTree 缓存
//! ```

use crate::batch_builder::BatchBuilder;
use crate::memory::{MemoryManager, MemoryConfig};
use crate::policy::{ScheduleContext, SchedulingPolicy};
use crate::state::GlobalState;
use crate::transport::{FrontendReceiver, WorkerProxy};
use infer_protocol::{StepOutput, FinishOutput, SchedulerOutput};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, info, warn, error};

/// Coordinator 配置
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Block size
    pub block_size: usize,

    /// 总 block 数
    pub total_blocks: usize,

    /// 是否启用前缀缓存
    pub enable_prefix_cache: bool,

    /// 是否启用 CoW
    pub enable_cow: bool,

    /// 默认请求优先级
    pub default_priority: i32,

    /// 主循环休眠时间（当没有任务时，毫秒）
    pub idle_sleep_ms: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            total_blocks: 1000,
            enable_prefix_cache: true,
            enable_cow: false,
            default_priority: 0,
            idle_sleep_ms: 1,
        }
    }
}

/// Coordinator - 核心调度器
pub struct Coordinator {
    /// 全局状态
    state: GlobalState,

    /// 内存管理器（整合 BlockAllocator + BlockTableManager + RadixTree）
    memory: MemoryManager,

    /// 调度策略
    policy: Box<dyn SchedulingPolicy>,

    /// Worker 代理
    worker: WorkerProxy,

    /// 前端接收器
    frontend: FrontendReceiver,

    /// 配置
    config: CoordinatorConfig,

    /// 默认 Worker ID (支持单Worker模式)
    default_worker_id: Option<String>,

    /// 输出路由表 (request_id -> output_channel)
    /// 用于将输出发送回 Server
    output_router: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
}

impl Coordinator {
    /// 创建新的 Coordinator
    pub fn new(
        policy: Box<dyn SchedulingPolicy>,
        worker: WorkerProxy,
        frontend: FrontendReceiver,
        config: CoordinatorConfig,
        output_router: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>>,
    ) -> Self {
        // 创建 MemoryManager 配置
        let memory_config = MemoryConfig {
            total_blocks: config.total_blocks,
            block_size: config.block_size,
            enable_cow: config.enable_cow,
            enable_prefix_cache: config.enable_prefix_cache,
            prefix_cache_capacity: Some(10000), // 默认 10k tokens
        };

        let memory = MemoryManager::new(memory_config);
        let state = GlobalState::new(config.block_size, config.enable_prefix_cache);

        Self {
            state,
            memory,
            policy,
            worker,
            frontend,
            config,
            default_worker_id: None, // 将在首次注册时设置
            output_router,
        }
    }

    /// 启动主循环 (Entry Point)
    pub async fn run(&mut self) {
        info!("Coordinator loop started...");
        info!("Policy: {}", self.policy.name());
        info!("Block size: {}, Total blocks: {}", self.config.block_size, self.config.total_blocks);

        loop {
            // ==========================================================
            // Phase 1: Ingest (摄入)
            // 从前端接收新请求
            // ==========================================================
            self.ingest_new_requests();

            // ==========================================================
            // Phase 2: Schedule (决策)
            // 询问策略模块，下一步该跑谁
            // ==========================================================
            let plan = {
                let waiting = self.state.get_waiting_requests();
                let running = self.state.get_running_requests();
                let swapped = self.state.get_swapped_requests();

                let radix_tree = self.state.radix_tree.lock().unwrap();

                let context = ScheduleContext {
                    waiting_requests: &waiting,
                    running_requests: &running,
                    swapped_requests: &swapped,
                    memory: &self.memory,
                    radix_tree: Some(&radix_tree),
                };

                self.policy.schedule(&context)
            };

            // 如果没有任务要跑，休眠一小会儿避免 CPU 空转
            if plan.is_empty() {
                sleep(Duration::from_millis(self.config.idle_sleep_ms)).await;
                continue;
            }

            debug!(
                "Scheduled batch: {} prefills, {} decodes, {} preempts",
                plan.num_prefills(),
                plan.num_decodes(),
                plan.num_preempts()
            );

            // ==========================================================
            // Phase 3: Apply Plan & Prepare Metadata
            // 在发送给 Worker 之前，完成显存分配
            // ==========================================================

            // 3.1 处理抢占（释放显存）
            for req_id in &plan.preempts {
                self.preempt_sequence(req_id);
            }

            // 3.2 处理新任务（分配显存 + 移动到 Running）
            for req_id in &plan.prefills {
                self.allocate_for_prefill(req_id);
            }

            // 3.3 处理解码任务（追加显存）
            for req_id in &plan.decodes {
                self.allocate_for_decode(req_id);
            }

            // 3.4 构建 WorkerInput（打包物理数据）
            let builder = BatchBuilder::new(&plan, &self.state, self.config.block_size);
            let forward_params = match builder.build() {
                Ok(params) => params,
                Err(e) => {
                    error!("Failed to build forward params: {}", e);
                    continue;
                }
            };

            // ==========================================================
            // Phase 4: RPC Call (执行)
            // 远程调用 GPU Worker
            // ==========================================================
            let worker_id = match &self.default_worker_id {
                Some(id) => id,
                None => {
                    warn!("No Worker registered, skipping forward");
                    continue;
                }
            };

            let forward_result = match self.worker.forward(worker_id, forward_params).await {
                Ok(result) => result,
                Err(e) => {
                    error!("Worker error: {}, attempting recovery...", e);
                    // TODO: 错误处理逻辑（如踢出导致错误的请求）
                    continue;
                }
            };

            // ==========================================================
            // Phase 5: Post-Process (后处理)
            // 更新状态，处理结果流式返回
            // ==========================================================
            self.process_output(forward_result, plan).await;
        }
    }

    /// Phase 1: 摄入新请求
    ///
    /// 从前端 channel 非阻塞接收所有待处理的请求
    fn ingest_new_requests(&mut self) {
        let requests = self.frontend.try_recv_all();

        for req in requests {
            info!("New request: {} ({} tokens)", req.request_id, req.input_token_ids.len());

            // 创建输出 channels
            let (output_tx, _output_rx) = mpsc::unbounded_channel::<StepOutput>();
            let (finish_tx, _finish_rx) = mpsc::unbounded_channel::<FinishOutput>();

            // 将输出 channel 包装为 SchedulerOutput channel并注册到 output_router
            // 这样 Coordinator 可以通过 output_router 发送输出
            // 注意：这里我们暂时不使用 output_tx 和 finish_tx，
            // 而是依赖 Sequence 内部的发送机制
            // TODO: 重构 Sequence 使用 SchedulerOutput 统一输出

            self.state.add_request(
                req,
                self.config.default_priority,
                output_tx,
                finish_tx,
            );
        }
    }

    /// Phase 3.1: 抢占序列（释放显存）
    fn preempt_sequence(&mut self, request_id: &str) {
        debug!("Preempting request: {}", request_id);

        // 通过 MemoryManager 释放序列
        let _ = self.memory.free_sequence(&request_id.to_string());

        // 将状态移动到 Swapped 队列
        self.state.move_to_swapped(request_id);
    }

    /// Phase 3.2: 为 Prefill 分配显存
    fn allocate_for_prefill(&mut self, request_id: &str) {
        debug!("Allocating for prefill: {}", request_id);

        // 初始化序列显存（包括 RadixTree 缓存命中检查）
        self.state.init_sequence_memory(request_id, &mut self.memory);

        // 移动到 Running 队列
        self.state.move_to_running(request_id);
    }

    /// Phase 3.3: 为 Decode 追加显存
    fn allocate_for_decode(&mut self, request_id: &str) {
        // 检查最后一个 Block 是否满了，满了就申请一个新的追加上去
        if self.state.needs_new_block(request_id) {
            // 通过 MemoryManager 追加一个 slot（会自动分配新 block）
            if let Ok(()) = self.memory.append_slots(&request_id.to_string(), 1) {
                // 获取新分配的块并同步到 Sequence
                if let Some(table) = self.memory.get_block_table(&request_id.to_string()) {
                    let blocks = table.get_physical_blocks();
                    if let Some(&new_block) = blocks.last() {
                        debug!("Allocated new block {} for decode: {}", new_block, request_id);
                        self.state.append_block(request_id, new_block);
                    }
                }
            } else {
                // 理论上 Schedule 阶段已经检查过显存足够
                // 如果这里发生说明 Policy 的计算逻辑有 Bug
                error!("Scheduler Logic Bug: OOM during decode allocation for {}", request_id);
                // Gracefully reject the request instead of panicking
                self.reject_request(
                    request_id,
                    anyhow::anyhow!("OOM during decode allocation - scheduler logic bug")
                );
            }
        }
    }

    /// Phase 5: 处理 Worker 输出
    async fn process_output(
        &mut self,
        output: infer_protocol::ForwardResult,
        plan: crate::policy::BatchPlan,
    ) {
        // Worker 返回的 token 顺序与 BatchPlan 中的顺序一致
        // prefills + decodes
        let all_request_ids: Vec<String> = plan
            .prefills
            .iter()
            .chain(plan.decodes.iter())
            .cloned()
            .collect();

        for (idx, request_id) in all_request_ids.iter().enumerate() {
            if idx >= output.next_token_ids.len() {
                warn!("Output mismatch: missing token for request {}", request_id);
                continue;
            }

            let new_token = output.next_token_ids[idx];

            // 先检查和更新状态，获取需要的信息
            let (is_finished, finish_reason) = {
                let seq = match self.state.get_mut_sequence(request_id) {
                    Some(s) => s,
                    None => {
                        warn!("Sequence not found: {}", request_id);
                        continue;
                    }
                };

                // 1. 追加 Token
                seq.append_token(new_token);

                // 2. 检查是否结束
                let (finished, reason) = seq.check_finished();

                // 3. 发送流式输出 (通过 output_router)
                if !finished {
                    self.send_step_output(request_id, new_token as u32);
                }

                (finished, reason)
            }; // seq 的借用在这里结束

            if is_finished {
                // === 结束处理 ===
                info!("Request finished: {} (reason: {:?})", request_id, finish_reason);

                // A. 更新 RadixTree（缓存这条完整的路径供未来复用）
                self.state.free_sequence_memory(request_id, &mut self.memory);

                // B. 通知前端结束 (通过 output_router)
                self.send_finish_output(request_id, finish_reason);

                // C. 从队列移除
                self.state.remove_request(request_id);
            } else {
                // === 继续运行 ===
                // 状态保持在 Running，等待下一轮 Schedule
                debug!("Request continuing: {} (token: {})", request_id, new_token);
            }
        }
    }

    /// 通过 output_router 发送 StepOutput
    fn send_step_output(&self, request_id: &str, token_id: u32) {
        let router = self.output_router.read().unwrap();
        if let Some(tx) = router.get(request_id) {
            let step_output = StepOutput {
                request_id: request_id.to_string(),
                new_token_id: token_id,
                logprob: None,
            };

            let output = SchedulerOutput::Step(step_output);

            if let Err(e) = tx.send(output) {
                warn!("Failed to send step output for {}: {:?}", request_id, e);
            } else {
                debug!("Sent step output for {}: token {}", request_id, token_id);
            }
        } else {
            warn!("No output channel found for request: {}", request_id);
        }
    }

    /// 通过 output_router 发送 FinishOutput
    fn send_finish_output(&self, request_id: &str, finish_reason: infer_protocol::FinishReason) {
        let router = self.output_router.read().unwrap();
        if let Some(tx) = router.get(request_id) {
            let finish_output = FinishOutput {
                request_id: request_id.to_string(),
                reason: finish_reason.clone(),
            };

            let output = SchedulerOutput::Finish(finish_output);

            if let Err(e) = tx.send(output) {
                warn!("Failed to send finish output for {}: {:?}", request_id, e);
            } else {
                debug!("Sent finish output for {}: {:?}", request_id, finish_reason);
            }
        } else {
            warn!("No output channel found for request: {}", request_id);
        }
    }

    /// 获取队列统计
    pub fn queue_stats(&self) -> (usize, usize, usize) {
        self.state.queue_stats()
    }

    /// 获取分配器统计
    pub fn allocator_stats(&self) -> &crate::memory::AllocatorStats {
        self.memory.allocator_stats()
    }

    /// 设置默认 Worker ID
    pub fn set_default_worker(&mut self, worker_id: String) {
        self.default_worker_id = Some(worker_id);
    }

    /// 获取 Worker Proxy
    pub fn worker_proxy(&mut self) -> &mut WorkerProxy {
        &mut self.worker
    }

    /// 拒绝请求（错误处理）
    ///
    /// 在发生错误时，通知前端并清理资源
    fn reject_request(&mut self, request_id: &str, error: anyhow::Error) {
        error!("Rejecting request {}: {}", request_id, error);

        // 1. 通知前端结束（发送错误）
        if let Some(seq) = self.state.get_sequence(request_id) {
            seq.send_finish_stream(infer_protocol::FinishReason::Abort);
        }

        // 2. 释放显存
        self.state.free_sequence_memory(request_id, &mut self.memory);

        // 3. 从队列移除
        self.state.remove_request(request_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::ContinuousBatchingPolicy;
    use crate::transport::create_frontend_channel;
    use infer_protocol::{InferRequest, SamplingParams, StoppingCriteria, SchedulerCommand};

    fn create_test_request(id: &str, tokens: Vec<u32>) -> InferRequest {
        InferRequest {
            request_id: id.to_string(),
            input_token_ids: tokens,
            sampling_params: SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                top_k: 50,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                seed: None,
            },
            stopping_criteria: StoppingCriteria {
                stop_token_ids: vec![2],
                max_new_tokens: 100,
                ignore_eos: false,
            },
            adapter_id: None,
        }
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let policy = Box::new(ContinuousBatchingPolicy::default_config());
        let endpoint = format!("ipc:///tmp/test-coord-{}.ipc", std::process::id());
        let worker = WorkerProxy::new(endpoint, 5000).await.unwrap();
        let (_tx, frontend) = create_frontend_channel();
        let config = CoordinatorConfig::default();
        let output_router = Arc::new(RwLock::new(HashMap::new()));

        let coordinator = Coordinator::new(policy, worker, frontend, config, output_router);

        let (waiting, running, swapped) = coordinator.queue_stats();
        assert_eq!(waiting, 0);
        assert_eq!(running, 0);
        assert_eq!(swapped, 0);
    }

    #[tokio::test]
    async fn test_coordinator_ingest() {
        let policy = Box::new(ContinuousBatchingPolicy::default_config());
        let endpoint = format!("ipc:///tmp/test-coord-ingest-{}.ipc", std::process::id());
        let worker = WorkerProxy::new(endpoint, 5000).await.unwrap();
        let (tx, frontend) = create_frontend_channel();
        let config = CoordinatorConfig::default();
        let output_router = Arc::new(RwLock::new(HashMap::new()));

        let mut coordinator = Coordinator::new(policy, worker, frontend, config, output_router);

        // 发送请求
        let req = create_test_request("req1", vec![1, 2, 3, 4, 5]);
        tx.send(SchedulerCommand::AddRequest(req)).unwrap();

        // 摄入请求
        coordinator.ingest_new_requests();

        let (waiting, running, swapped) = coordinator.queue_stats();
        assert_eq!(waiting, 1);
        assert_eq!(running, 0);
        assert_eq!(swapped, 0);
    }
}
