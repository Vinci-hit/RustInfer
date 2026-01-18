//! Continuous Batching Scheduling Policy
//!
//! 实现了 Continuous Batching 调度策略，这是现代 LLM 服务的核心算法。
//!
//! # 核心思想
//!
//! 1. **Decode 优先**: 优先保证正在运行的请求能继续生成
//! 2. **动态批处理**: 在一个 batch 中可以同时处理 prefill 和 decode
//! 3. **抢占机制**: 显存不足时，抢占低优先级请求
//! 4. **前缀缓存**: 利用 RadixTree 减少重复计算
//!
//! # 调度流程
//!
//! ```text
//! Step 1: 处理 Running 队列 (Decode)
//!   ├─ 预估每个请求是否需要新 Block
//!   ├─ 如果显存不足，触发抢占
//!   └─ 保留能继续运行的请求
//!
//! Step 2: 处理 Swapped 队列 (可选)
//!   └─ 如果有剩余显存，恢复被换出的请求
//!
//! Step 3: 处理 Waiting 队列 (Prefill)
//!   ├─ 检查 RadixTree 缓存命中
//!   ├─ 计算所需 Block 数
//!   └─ 接纳新请求直到达到 batch size 或显存上限
//! ```

use super::{
    BatchPlan, ScheduleContext, SchedulePolicyConfig, ScheduleRequest,
    SchedulingPolicy,
};
use crate::memory::MemoryManager;

/// Continuous Batching 调度策略
///
/// 实现了 vLLM 风格的持续批处理调度
pub struct ContinuousBatchingPolicy {
    /// 策略配置
    config: SchedulePolicyConfig,
}

impl ContinuousBatchingPolicy {
    /// 创建新的 Continuous Batching 策略
    pub fn new(config: SchedulePolicyConfig) -> Self {
        Self { config }
    }

    /// 创建默认配置的策略
    pub fn default_config() -> Self {
        Self {
            config: SchedulePolicyConfig::default(),
        }
    }

    /// 调度 Running 队列（Decode 阶段）
    ///
    /// # Arguments
    /// * `running_requests` - 正在运行的请求列表
    /// * `memory` - 内存管理器
    /// * `free_blocks` - 可用的空闲 Block 数量（会被修改）
    ///
    /// # Returns
    /// (decode_ids, preempt_ids) - 可以 decode 的请求 ID 和需要抢占的请求 ID
    fn schedule_running(
        &self,
        running_requests: &[ScheduleRequest],
        memory: &MemoryManager,
        free_blocks: &mut usize,
    ) -> (Vec<String>, Vec<String>) {
        let mut decode_ids = Vec::new();
        let mut preempt_ids = Vec::new();

        let block_size = memory.config().block_size;

        // Step 1: 计算所有 running 请求需要的新 Block 数
        let mut needed_blocks = 0;
        for req in running_requests {
            if req.is_last_block_full(block_size) {
                needed_blocks += 1;
            }
        }

        // Step 2: 检查是否需要抢占
        if *free_blocks < needed_blocks && self.config.enable_preemption {
            // 需要抢占！按优先级排序（优先级低的先被踢）
            let mut sorted_requests: Vec<_> = running_requests.iter().collect();
            sorted_requests.sort_by_key(|req| (req.priority, req.arrival_time));

            // 尝试踢出低优先级请求
            for &req in &sorted_requests {
                // 计算踢掉这个请求能释放多少 Block
                let freed = req.allocated_blocks.len();

                preempt_ids.push(req.request_id.clone());
                *free_blocks += freed;

                // 减少需求（如果这个请求本身需要新 Block）
                if req.is_last_block_full(block_size) {
                    needed_blocks = needed_blocks.saturating_sub(1);
                }

                // 检查是否已经足够
                if *free_blocks >= needed_blocks {
                    break;
                }
            }
        }

        // Step 3: 确定哪些请求可以继续 decode
        for req in running_requests {
            if !preempt_ids.contains(&req.request_id) {
                decode_ids.push(req.request_id.clone());
            }
        }

        // Step 4: 扣除 decode 使用的 Block
        *free_blocks = free_blocks.saturating_sub(needed_blocks);

        (decode_ids, preempt_ids)
    }

    /// 调度 Waiting 队列（Prefill 阶段）
    ///
    /// # Arguments
    /// * `waiting_requests` - 等待中的请求列表
    /// * `memory` - 内存管理器
    /// * `radix_tree` - 前缀缓存树（可选）
    /// * `free_blocks` - 可用的空闲 Block 数量（会被修改）
    /// * `current_batch_size` - 当前 batch 中已有的请求数
    ///
    /// # Returns
    /// prefill_ids - 可以进行 prefill 的请求 ID 列表
    fn schedule_waiting(
        &self,
        waiting_requests: &[ScheduleRequest],
        memory: &MemoryManager,
        radix_tree: Option<&crate::memory::RadixTree>,
        free_blocks: &mut usize,
        current_batch_size: usize,
    ) -> Vec<String> {
        let mut prefill_ids = Vec::new();

        let block_size = memory.config().block_size;

        for req in waiting_requests {
            // 检查 batch size 限制
            if prefill_ids.len() + current_batch_size >= self.config.max_batch_size {
                break;
            }

            // 计算需要的 Block 数（考虑缓存命中）
            let required_blocks = if self.config.enable_prefix_cache {
                req.calculate_needed_blocks_with_cache(block_size, radix_tree)
            } else {
                req.blocks_needed(block_size)
            };

            // 检查显存是否足够
            if *free_blocks >= required_blocks {
                prefill_ids.push(req.request_id.clone());
                *free_blocks = free_blocks.saturating_sub(required_blocks);
            } else {
                // 策略选择：FCFS（First-Come-First-Serve）
                // 如果前面的进不去，后面的也不考虑，防止饿死
                break;
            }
        }

        prefill_ids
    }

    /// 调度 Swapped 队列（可选，恢复被换出的请求）
    ///
    /// # Arguments
    /// * `swapped_requests` - 被换出的请求列表
    /// * `memory` - 内存管理器
    /// * `free_blocks` - 可用的空闲 Block 数量（会被修改）
    /// * `current_batch_size` - 当前 batch 中已有的请求数
    ///
    /// # Returns
    /// swap_in_ids - 可以换回来的请求 ID 列表
    fn schedule_swapped(
        &self,
        swapped_requests: &[ScheduleRequest],
        memory: &MemoryManager,
        free_blocks: &mut usize,
        current_batch_size: usize,
    ) -> Vec<String> {
        if !self.config.enable_swap {
            return Vec::new();
        }

        let mut swap_in_ids = Vec::new();

        let block_size = memory.config().block_size;

        // 按优先级排序（高优先级先恢复）
        let mut sorted_requests: Vec<_> = swapped_requests.iter().collect();
        sorted_requests.sort_by_key(|req| std::cmp::Reverse((req.priority, req.arrival_time)));

        for &req in &sorted_requests {
            // 检查 batch size 限制
            if swap_in_ids.len() + current_batch_size >= self.config.max_batch_size {
                break;
            }

            // 计算恢复需要的 Block 数
            let required_blocks = req.blocks_needed(block_size);

            if *free_blocks >= required_blocks {
                swap_in_ids.push(req.request_id.clone());
                *free_blocks = free_blocks.saturating_sub(required_blocks);
            } else {
                break;
            }
        }

        swap_in_ids
    }
}

impl SchedulingPolicy for ContinuousBatchingPolicy {
    fn schedule(&self, context: &ScheduleContext) -> BatchPlan {
        let mut plan = BatchPlan::new();

        // 获取当前可用的空闲 Block 数
        let mut free_blocks = context.memory.num_free_blocks();

        // ====================================================
        // Step 1: 调度 Running 队列 (Decode)
        // ====================================================
        let (decode_ids, preempt_ids) = self.schedule_running(
            context.running_requests,
            context.memory,
            &mut free_blocks,
        );

        for id in decode_ids {
            plan.add_decode(id);
        }

        for id in preempt_ids {
            plan.add_preempt(id);
        }

        let current_batch_size = plan.num_decodes();

        // ====================================================
        // Step 2: 调度 Swapped 队列 (可选)
        // ====================================================
        if self.config.enable_swap && !context.swapped_requests.is_empty() {
            let swap_in_ids = self.schedule_swapped(
                context.swapped_requests,
                context.memory,
                &mut free_blocks,
                current_batch_size,
            );

            // 被换回的请求视为 prefill（需要重新计算）
            for id in swap_in_ids {
                plan.add_prefill(id);
            }
        }

        let current_batch_size = plan.total_active_requests();

        // ====================================================
        // Step 3: 调度 Waiting 队列 (Prefill)
        // ====================================================
        let prefill_ids = self.schedule_waiting(
            context.waiting_requests,
            context.memory,
            context.radix_tree,
            &mut free_blocks,
            current_batch_size,
        );

        for id in prefill_ids {
            plan.add_prefill(id);
        }

        plan
    }

    fn name(&self) -> &'static str {
        "ContinuousBatching"
    }

    fn config(&self) -> &SchedulePolicyConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryConfig, MemoryManager};
    use crate::policy::RequestState;

    fn create_test_memory() -> MemoryManager {
        let config = MemoryConfig {
            total_blocks: 100,
            block_size: 16,
            enable_cow: false,
            enable_prefix_cache: false,
            prefix_cache_capacity: None,
        };
        MemoryManager::new(config)
    }

    fn create_test_request(
        id: &str,
        tokens: usize,
        max_tokens: usize,
        priority: i32,
    ) -> ScheduleRequest {
        ScheduleRequest::new(id.to_string(), vec![1; tokens], max_tokens, priority)
    }

    #[test]
    fn test_schedule_running_no_preemption() {
        let policy = ContinuousBatchingPolicy::default_config();
        let memory = create_test_memory();

        let mut req1 = create_test_request("req1", 10, 100, 0);
        req1.state = RequestState::Running;
        req1.num_generated_tokens = 5; // 总共 15 tokens，不满一个 block

        let running = vec![req1];

        let mut free_blocks = 50;
        let (decode_ids, preempt_ids) =
            policy.schedule_running(&running, &memory, &mut free_blocks);

        // 应该没有抢占
        assert_eq!(decode_ids.len(), 1);
        assert_eq!(preempt_ids.len(), 0);
        assert_eq!(decode_ids[0], "req1");
    }

    #[test]
    fn test_schedule_running_with_preemption() {
        let policy = ContinuousBatchingPolicy::default_config();
        let memory = create_test_memory();

        // 创建两个请求，req1 优先级高，req2 优先级低
        let mut req1 = create_test_request("req1", 16, 100, 10); // 高优先级
        req1.state = RequestState::Running;
        req1.num_generated_tokens = 0;
        req1.allocated_blocks = vec![1];

        let mut req2 = create_test_request("req2", 16, 100, 0); // 低优先级
        req2.state = RequestState::Running;
        req2.num_generated_tokens = 0;
        req2.allocated_blocks = vec![2];

        let running = vec![req1, req2];

        // 设置很少的空闲块，触发抢占
        let mut free_blocks = 0;
        let (decode_ids, preempt_ids) =
            policy.schedule_running(&running, &memory, &mut free_blocks);

        // 应该抢占低优先级的 req2
        assert_eq!(preempt_ids.len(), 1);
        assert_eq!(preempt_ids[0], "req2");
        assert_eq!(decode_ids.len(), 1);
        assert_eq!(decode_ids[0], "req1");
    }

    #[test]
    fn test_schedule_waiting() {
        let policy = ContinuousBatchingPolicy::default_config();
        let memory = create_test_memory();

        let req1 = create_test_request("req1", 10, 100, 0);
        let req2 = create_test_request("req2", 20, 100, 0);

        let waiting = vec![req1, req2];

        let mut free_blocks = 10; // 足够分配
        let prefill_ids = policy.schedule_waiting(&waiting, &memory, None, &mut free_blocks, 0);

        // 两个请求都应该被接纳
        assert_eq!(prefill_ids.len(), 2);
        assert_eq!(prefill_ids[0], "req1");
        assert_eq!(prefill_ids[1], "req2");
    }

    #[test]
    fn test_schedule_waiting_limited_blocks() {
        let policy = ContinuousBatchingPolicy::default_config();
        let memory = create_test_memory();

        let req1 = create_test_request("req1", 10, 100, 0);
        let req2 = create_test_request("req2", 20, 100, 0);

        let waiting = vec![req1, req2];

        let mut free_blocks = 1; // 只够一个请求
        let prefill_ids = policy.schedule_waiting(&waiting, &memory, None, &mut free_blocks, 0);

        // 只有第一个请求能被接纳（FCFS）
        assert_eq!(prefill_ids.len(), 1);
        assert_eq!(prefill_ids[0], "req1");
    }

    #[test]
    fn test_full_schedule() {
        let policy = ContinuousBatchingPolicy::default_config();
        let memory = create_test_memory();

        let mut req1 = create_test_request("req1", 10, 100, 0);
        req1.state = RequestState::Running;
        req1.allocated_blocks = vec![1];

        let req2 = create_test_request("req2", 20, 100, 0);

        let context = ScheduleContext {
            running_requests: &[req1],
            waiting_requests: &[req2],
            swapped_requests: &[],
            memory: &memory,
            radix_tree: None,
        };

        let plan = policy.schedule(&context);

        assert!(!plan.is_empty());
        assert_eq!(plan.num_decodes(), 1);
        assert_eq!(plan.decodes[0], "req1");
    }
}
