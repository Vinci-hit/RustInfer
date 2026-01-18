//! Global State Management - 全局状态管理
//!
//! 管理所有请求的生命周期和状态转换：
//! - Waiting -> Running -> Finished
//! - Running -> Swapped -> Running
//!
//! 同时维护每个请求的显存分配信息（Block Table）

use crate::memory::{MemoryManager, PhysicalBlockId, RadixTree};
use crate::policy::{ScheduleRequest, RequestState};
use infer_protocol::{InferRequest, FinishReason, StepOutput, FinishOutput};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

/// 序列 ID
pub type SequenceId = String;

/// 单个请求的完整状态
#[derive(Debug, Clone)]
pub struct Sequence {
    /// 请求 ID
    pub request_id: SequenceId,

    /// 输入 tokens
    pub input_tokens: Vec<i32>,

    /// 已生成的 tokens
    pub generated_tokens: Vec<i32>,

    /// 分配的物理块 IDs
    pub allocated_blocks: Vec<PhysicalBlockId>,

    /// 当前状态
    pub state: RequestState,

    /// 优先级
    pub priority: i32,

    /// 到达时间
    pub arrival_time: std::time::Instant,

    /// 最大生成长度
    pub max_tokens: usize,

    /// 停止 token IDs
    pub stop_token_ids: Vec<u32>,

    /// 输出流 channel (用于流式返回结果)
    pub output_tx: Option<mpsc::UnboundedSender<StepOutput>>,

    /// 完成通知 channel
    pub finish_tx: Option<mpsc::UnboundedSender<FinishOutput>>,

    /// Block size (from memory config)
    block_size: usize,
}

impl Sequence {
    /// 创建新的 Sequence
    pub fn new(
        request: InferRequest,
        block_size: usize,
        priority: i32,
        output_tx: mpsc::UnboundedSender<StepOutput>,
        finish_tx: mpsc::UnboundedSender<FinishOutput>,
    ) -> Self {
        Self {
            request_id: request.request_id.clone(),
            input_tokens: request.input_token_ids.iter().map(|&x| x as i32).collect(),
            generated_tokens: Vec::new(),
            allocated_blocks: Vec::new(),
            state: RequestState::Waiting,
            priority,
            arrival_time: std::time::Instant::now(),
            max_tokens: request.stopping_criteria.max_new_tokens,
            stop_token_ids: request.stopping_criteria.stop_token_ids,
            output_tx: Some(output_tx),
            finish_tx: Some(finish_tx),
            block_size,
        }
    }

    /// 追加新生成的 token
    pub fn append_token(&mut self, token_id: i32) {
        self.generated_tokens.push(token_id);
    }

    /// 获取总 token 数（输入 + 已生成）
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    /// 获取所有 tokens（输入 + 已生成）
    pub fn get_all_tokens(&self) -> Vec<i32> {
        let mut tokens = self.input_tokens.clone();
        tokens.extend_from_slice(&self.generated_tokens);
        tokens
    }

    /// 检查是否已完成
    pub fn check_finished(&self) -> (bool, FinishReason) {
        // 检查是否达到最大长度
        if self.generated_tokens.len() >= self.max_tokens {
            return (true, FinishReason::Length);
        }

        // 检查是否遇到停止 token
        if let Some(&last_token) = self.generated_tokens.last() {
            if self.stop_token_ids.contains(&(last_token as u32)) {
                return (true, FinishReason::Stop);
            }
        }

        (false, FinishReason::Length) // 未完成，默认值无意义
    }

    /// 检查最后一个 Block 是否已满
    pub fn is_last_block_full(&self) -> bool {
        if self.allocated_blocks.is_empty() {
            return false;
        }
        let total = self.total_tokens();
        total % self.block_size == 0
    }

    /// 计算需要的 Block 数量
    pub fn blocks_needed(&self) -> usize {
        let total = self.total_tokens();
        if total == 0 {
            0
        } else {
            (total + self.block_size - 1) / self.block_size
        }
    }

    /// 发送流式 token 输出
    pub fn send_token_stream(&self, token_id: i32) {
        if let Some(tx) = &self.output_tx {
            let result = StepOutput {
                request_id: self.request_id.clone(),
                new_token_id: token_id as u32,
                logprob: None,
            };
            let _ = tx.send(result); // 忽略发送失败（客户端可能已断开）
        }
    }

    /// 发送完成通知
    pub fn send_finish_stream(&self, finish_reason: FinishReason) {
        if let Some(tx) = &self.finish_tx {
            let result = FinishOutput {
                request_id: self.request_id.clone(),
                reason: finish_reason,
            };
            let _ = tx.send(result);
        }
    }

    /// 转换为 ScheduleRequest（用于调度策略）
    pub fn to_schedule_request(&self) -> ScheduleRequest {
        ScheduleRequest::new(
            self.request_id.clone(),
            self.input_tokens.clone(),
            self.max_tokens,
            self.priority,
        )
        .with_state(self.state)
        .with_generated(self.generated_tokens.len())
        .with_blocks(self.allocated_blocks.clone())
    }
}

// ScheduleRequest 的构建器方法扩展
impl ScheduleRequest {
    /// 设置状态
    pub fn with_state(mut self, state: RequestState) -> Self {
        self.state = state;
        self
    }

    /// 设置已生成的 token 数
    pub fn with_generated(mut self, num_generated: usize) -> Self {
        self.num_generated_tokens = num_generated;
        self
    }

    /// 设置已分配的块
    pub fn with_blocks(mut self, blocks: Vec<PhysicalBlockId>) -> Self {
        self.allocated_blocks = blocks;
        self
    }
}

/// 全局状态管理器
pub struct GlobalState {
    /// 所有 Sequences
    sequences: HashMap<SequenceId, Sequence>,

    /// Waiting 队列 (按到达顺序)
    waiting_queue: Vec<SequenceId>,

    /// Running 队列
    running_queue: Vec<SequenceId>,

    /// Swapped 队列（被抢占的请求）
    swapped_queue: Vec<SequenceId>,

    /// RadixTree 前缀缓存
    pub radix_tree: Arc<std::sync::Mutex<RadixTree>>,

    /// Block size
    block_size: usize,
}

impl GlobalState {
    /// 创建新的 GlobalState
    pub fn new(block_size: usize, enable_prefix_cache: bool) -> Self {
        let radix_tree = if enable_prefix_cache {
            RadixTree::with_eviction(100000) // 默认最大 100k tokens 缓存
        } else {
            RadixTree::new()
        };

        Self {
            sequences: HashMap::new(),
            waiting_queue: Vec::new(),
            running_queue: Vec::new(),
            swapped_queue: Vec::new(),
            radix_tree: Arc::new(std::sync::Mutex::new(radix_tree)),
            block_size,
        }
    }

    /// 添加新请求
    pub fn add_request(
        &mut self,
        request: InferRequest,
        priority: i32,
        output_tx: mpsc::UnboundedSender<StepOutput>,
        finish_tx: mpsc::UnboundedSender<FinishOutput>,
    ) {
        let request_id = request.request_id.clone();
        let seq = Sequence::new(request, self.block_size, priority, output_tx, finish_tx);

        self.sequences.insert(request_id.clone(), seq);
        self.waiting_queue.push(request_id);
    }

    /// 获取 Sequence (不可变)
    pub fn get_sequence(&self, request_id: &str) -> Option<&Sequence> {
        self.sequences.get(request_id)
    }

    /// 获取 Sequence (可变)
    pub fn get_mut_sequence(&mut self, request_id: &str) -> Option<&mut Sequence> {
        self.sequences.get_mut(request_id)
    }

    /// 移动到 Running 队列
    pub fn move_to_running(&mut self, request_id: &str) {
        // 从 waiting 或 swapped 移除
        self.waiting_queue.retain(|id| id != request_id);
        self.swapped_queue.retain(|id| id != request_id);

        // 加入 running
        if !self.running_queue.contains(&request_id.to_string()) {
            self.running_queue.push(request_id.to_string());
        }

        // 更新状态
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.state = RequestState::Running;
        }
    }

    /// 移动到 Swapped 队列
    pub fn move_to_swapped(&mut self, request_id: &str) {
        // 从 running 移除
        self.running_queue.retain(|id| id != request_id);

        // 加入 swapped
        if !self.swapped_queue.contains(&request_id.to_string()) {
            self.swapped_queue.push(request_id.to_string());
        }

        // 更新状态
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.state = RequestState::Swapped;
        }
    }

    /// 移除请求
    pub fn remove_request(&mut self, request_id: &str) {
        self.waiting_queue.retain(|id| id != request_id);
        self.running_queue.retain(|id| id != request_id);
        self.swapped_queue.retain(|id| id != request_id);
        self.sequences.remove(request_id);
    }

    /// 获取所有 Waiting 的 ScheduleRequests
    pub fn get_waiting_requests(&self) -> Vec<ScheduleRequest> {
        self.waiting_queue
            .iter()
            .filter_map(|id| self.sequences.get(id))
            .map(|seq| seq.to_schedule_request())
            .collect()
    }

    /// 获取所有 Running 的 ScheduleRequests
    pub fn get_running_requests(&self) -> Vec<ScheduleRequest> {
        self.running_queue
            .iter()
            .filter_map(|id| self.sequences.get(id))
            .map(|seq| seq.to_schedule_request())
            .collect()
    }

    /// 获取所有 Swapped 的 ScheduleRequests
    pub fn get_swapped_requests(&self) -> Vec<ScheduleRequest> {
        self.swapped_queue
            .iter()
            .filter_map(|id| self.sequences.get(id))
            .map(|seq| seq.to_schedule_request())
            .collect()
    }

    /// 获取请求的物理块
    pub fn get_blocks(&self, request_id: &str) -> Vec<PhysicalBlockId> {
        self.sequences
            .get(request_id)
            .map(|seq| seq.allocated_blocks.clone())
            .unwrap_or_default()
    }

    /// 初始化序列显存（Prefill 阶段）
    pub fn init_sequence_memory(&mut self, request_id: &str, memory: &mut MemoryManager) {
        if let Some(seq) = self.sequences.get_mut(request_id) {
            // 检查 RadixTree 是否命中
            let prefix_match = memory.match_prefix(&seq.input_tokens);

            if let Some(prefix_match) = prefix_match {
                // 命中的块可以直接复用（增加引用计数）
                for &block_id in &prefix_match.matched_blocks {
                    // MemoryManager 内部会增加引用计数
                    seq.allocated_blocks.push(block_id);
                }
            }

            // 分配剩余需要的新块
            let blocks_needed = seq.blocks_needed();
            let blocks_have = seq.allocated_blocks.len();
            let blocks_to_allocate = blocks_needed.saturating_sub(blocks_have);

            // 通过 MemoryManager 分配
            if blocks_to_allocate > 0 {
                if let Ok(()) = memory.allocate_for_sequence(request_id.to_string(), blocks_to_allocate) {
                    // 获取分配的块并同步到 Sequence
                    if let Some(table) = memory.get_block_table(&request_id.to_string()) {
                        let new_blocks: Vec<PhysicalBlockId> = table.get_physical_blocks()
                            .iter()
                            .skip(seq.allocated_blocks.len())
                            .copied()
                            .collect();
                        seq.allocated_blocks.extend(new_blocks);
                    }
                }
            }
        }
    }

    /// 为 Decode 追加块（如果需要）
    pub fn append_block(&mut self, request_id: &str, block_id: PhysicalBlockId) {
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.allocated_blocks.push(block_id);
        }
    }

    /// 检查是否需要新块
    pub fn needs_new_block(&self, request_id: &str) -> bool {
        self.sequences
            .get(request_id)
            .map(|seq| seq.is_last_block_full())
            .unwrap_or(false)
    }

    /// 释放序列显存
    pub fn free_sequence_memory(&mut self, request_id: &str, memory: &mut MemoryManager) {
        if let Some(seq) = self.sequences.get(request_id) {
            // 将完整的 token 序列插入 RadixTree
            let all_tokens = seq.get_all_tokens();
            memory.insert_prefix(&all_tokens, &seq.allocated_blocks);

            // 通过 MemoryManager 释放序列
            let _ = memory.free_sequence(&request_id.to_string());
        }
    }

    /// 获取队列统计
    pub fn queue_stats(&self) -> (usize, usize, usize) {
        (
            self.waiting_queue.len(),
            self.running_queue.len(),
            self.swapped_queue.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_protocol::{SamplingParams, StoppingCriteria};

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
    #[test]
    fn test_sequence_basic() {
        let req = create_test_request("req1", vec![1, 2, 3, 4, 5]);
        let (tx, _rx) = mpsc::unbounded_channel();
        let (finish_tx, _finish_rx) = mpsc::unbounded_channel();

        let mut seq = Sequence::new(req, 16, 0, tx, finish_tx);

        assert_eq!(seq.total_tokens(), 5);
        assert_eq!(seq.blocks_needed(), 1);

        seq.append_token(10);
        assert_eq!(seq.total_tokens(), 6);
        assert_eq!(seq.generated_tokens.len(), 1);
    }

    #[test]
    fn test_global_state_add_request() {
        let mut state = GlobalState::new(16, false);
        let req = create_test_request("req1", vec![1, 2, 3]);
        let (tx, _rx) = mpsc::unbounded_channel();
        let (finish_tx, _finish_rx) = mpsc::unbounded_channel();

        state.add_request(req, 0, tx, finish_tx);

        let (waiting, running, swapped) = state.queue_stats();
        assert_eq!(waiting, 1);
        assert_eq!(running, 0);
        assert_eq!(swapped, 0);
    }

    #[test]
    fn test_move_to_running() {
        let mut state = GlobalState::new(16, false);
        let req = create_test_request("req1", vec![1, 2, 3]);
        let (tx, _rx) = mpsc::unbounded_channel();
        let (finish_tx, _finish_rx) = mpsc::unbounded_channel();

        state.add_request(req, 0, tx, finish_tx);
        state.move_to_running("req1");

        let (waiting, running, swapped) = state.queue_stats();
        assert_eq!(waiting, 0);
        assert_eq!(running, 1);
        assert_eq!(swapped, 0);

        let seq = state.get_sequence("req1").unwrap();
        assert_eq!(seq.state, RequestState::Running);
    }
}
