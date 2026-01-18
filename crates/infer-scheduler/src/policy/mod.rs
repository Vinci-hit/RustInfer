//! Scheduling Policy - 调度策略
//!
//! 定义了调度器的接口和基础数据结构。
//! 调度策略决定了在每个 step 中：
//! - 哪些请求可以运行
//! - 哪些请求需要被抢占
//! - 如何分配有限的 GPU 显存资源
//!
//! # 设计原则
//!
//! 1. **Decode 优先**: 保证正在运行的任务能继续生成
//! 2. **公平性**: 防止任务饿死
//! 3. **吞吐量**: 最大化系统吞吐
//! 4. **延迟**: 最小化首 token 延迟（TTFT）

pub mod batch_plan;
pub mod continuous;

pub use batch_plan::{BatchPlan, RequestId, ScheduleStats};
pub use continuous::ContinuousBatchingPolicy;
use crate::memory::{MemoryManager, PhysicalBlockId, RadixTree};

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// 请求状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    /// 等待中（在 Waiting 队列）
    Waiting,

    /// 运行中（在 Running 队列）
    Running,

    /// 被抢占/交换出（在 Swapped 队列）
    Swapped,

    /// 已完成
    Finished,
}

/// 调度请求信息
///
/// 包含调度器需要的请求元数据
#[derive(Debug, Clone)]
pub struct ScheduleRequest {
    /// 请求 ID
    pub request_id: RequestId,

    /// 输入 token 列表
    pub input_tokens: Vec<i32>,

    /// 已生成的 token 数量
    pub num_generated_tokens: usize,

    /// 分配的物理块 ID 列表
    pub allocated_blocks: Vec<PhysicalBlockId>,

    /// 当前状态
    pub state: RequestState,

    /// 优先级（数值越大优先级越高）
    pub priority: i32,

    /// 到达时间
    pub arrival_time: Instant,

    /// 最大生成长度
    pub max_tokens: usize,
}

impl ScheduleRequest {
    /// 创建新的调度请求
    pub fn new(
        request_id: RequestId,
        input_tokens: Vec<i32>,
        max_tokens: usize,
        priority: i32,
    ) -> Self {
        Self {
            request_id,
            input_tokens,
            num_generated_tokens: 0,
            allocated_blocks: Vec::new(),
            state: RequestState::Waiting,
            priority,
            arrival_time: Instant::now(),
            max_tokens,
        }
    }

    /// 获取总 token 数（输入 + 已生成）
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.num_generated_tokens
    }

    /// 获取输入长度
    pub fn prompt_len(&self) -> usize {
        self.input_tokens.len()
    }

    /// 检查是否已完成
    pub fn is_finished(&self) -> bool {
        self.state == RequestState::Finished || self.num_generated_tokens >= self.max_tokens
    }

    /// 检查最后一个 Block 是否已满
    pub fn is_last_block_full(&self, block_size: usize) -> bool {
        if self.allocated_blocks.is_empty() {
            return false;
        }
        let total = self.total_tokens();
        total % block_size == 0
    }

    /// 计算需要的 Block 数量
    ///
    /// # Arguments
    /// * `block_size` - 每个 Block 的 Slot 数量
    pub fn blocks_needed(&self, block_size: usize) -> usize {
        let total = self.total_tokens();
        if total == 0 {
            0
        } else {
            (total + block_size - 1) / block_size
        }
    }

    /// 计算与 RadixTree 缓存命中后需要的额外 Block 数量
    ///
    /// # Arguments
    /// * `block_size` - 每个 Block 的 Slot 数量
    /// * `radix_tree` - 前缀缓存树
    ///
    /// # Returns
    /// 需要新分配的 Block 数量
    pub fn calculate_needed_blocks_with_cache(
        &self,
        block_size: usize,
        radix_tree: Option<&RadixTree>,
    ) -> usize {
        let total_blocks_needed = self.blocks_needed(block_size);

        // 如果有 RadixTree，检查缓存命中
        if let Some(_tree) = radix_tree {
            // TODO: 实际调用 match_prefix 计算命中的 token 数
            // let prefix_match = tree.match_prefix(&self.input_tokens);
            // let cached_blocks = prefix_match.matched_blocks.len();
            // total_blocks_needed.saturating_sub(cached_blocks)
            total_blocks_needed
        } else {
            total_blocks_needed
        }
    }

    /// 获取等待时间
    pub fn waiting_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }
}

/// 调度器上下文
///
/// 包含调度器决策所需的全局信息
pub struct ScheduleContext<'a> {
    /// Waiting 队列中的请求
    pub waiting_requests: &'a [ScheduleRequest],

    /// Running 队列中的请求
    pub running_requests: &'a [ScheduleRequest],

    /// Swapped 队列中的请求
    pub swapped_requests: &'a [ScheduleRequest],

    /// 内存管理器（只读，用于查询状态）
    pub memory: &'a MemoryManager,

    /// RadixTree 前缀缓存（可选）
    pub radix_tree: Option<&'a RadixTree>,
}

/// 调度策略配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulePolicyConfig {
    /// 最大 batch size（同时运行的请求数）
    pub max_batch_size: usize,

    /// 最大 tokens per step（防止单次计算过大）
    pub max_tokens_per_step: usize,

    /// 是否启用抢占
    pub enable_preemption: bool,

    /// 抢占阈值（剩余显存低于此值时触发）
    pub preemption_threshold: usize,

    /// 是否启用 Swap（将请求换出到 CPU 内存）
    pub enable_swap: bool,

    /// 是否启用前缀缓存
    pub enable_prefix_cache: bool,
}

impl Default for SchedulePolicyConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_tokens_per_step: 4096,
            enable_preemption: true,
            preemption_threshold: 0,
            enable_swap: false,
            enable_prefix_cache: true,
        }
    }
}

/// 调度策略 Trait
///
/// 实现此 trait 以定义自定义调度策略
pub trait SchedulingPolicy: Send + Sync {
    /// 核心调度方法
    ///
    /// 基于当前状态和可用资源，决定下一个 step 执行什么
    ///
    /// # Arguments
    /// * `context` - 调度上下文，包含所有请求队列和资源信息
    ///
    /// # Returns
    /// 批次计划，指示需要 prefill/decode/preempt 的请求
    fn schedule(&self, context: &ScheduleContext) -> BatchPlan;

    /// 获取策略名称
    fn name(&self) -> &'static str;

    /// 获取策略配置
    fn config(&self) -> &SchedulePolicyConfig;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_request_basic() {
        let req = ScheduleRequest::new(
            "req1".to_string(),
            vec![1, 2, 3, 4, 5],
            100,
            0,
        );

        assert_eq!(req.total_tokens(), 5);
        assert_eq!(req.prompt_len(), 5);
        assert_eq!(req.num_generated_tokens, 0);
        assert!(!req.is_finished());
        assert_eq!(req.state, RequestState::Waiting);
    }

    #[test]
    fn test_blocks_needed() {
        let mut req = ScheduleRequest::new(
            "req1".to_string(),
            vec![1; 30], // 30 tokens
            100,
            0,
        );

        let block_size = 16;

        // 30 tokens 需要 2 blocks (30 / 16 = 1.875, 向上取整 = 2)
        assert_eq!(req.blocks_needed(block_size), 2);

        // 生成 10 个 token 后，总共 40 tokens，需要 3 blocks
        req.num_generated_tokens = 10;
        assert_eq!(req.blocks_needed(block_size), 3);
    }

    #[test]
    fn test_is_last_block_full() {
        let mut req = ScheduleRequest::new(
            "req1".to_string(),
            vec![1; 16], // 正好 16 tokens
            100,
            0,
        );
        req.allocated_blocks = vec![1]; // 分配了一个 block

        let block_size = 16;

        // 16 tokens 正好填满 1 个 block
        assert!(req.is_last_block_full(block_size));

        // 添加 1 个 token，变成 17，不满
        req.num_generated_tokens = 1;
        assert!(!req.is_last_block_full(block_size));

        // 再添加到 32，又满了
        req.num_generated_tokens = 16;
        req.allocated_blocks = vec![1, 2]; // 分配了两个 blocks
        assert!(req.is_last_block_full(block_size));
    }
}
