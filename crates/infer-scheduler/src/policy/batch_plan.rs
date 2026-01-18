//! Batch Plan - 调度器输出的决策计划
//!
//! BatchPlan 是调度器每次调度循环的输出，告诉执行器：
//! - 哪些请求需要 Prefill（首次推理）
//! - 哪些请求需要 Decode（生成下一个 token）
//! - 哪些请求需要被抢占（显存不足时踢出）
//!
//! # 调度优先级
//!
//! 1. **Decode 优先**: 保证正在运行的任务能继续
//! 2. **Prefill 次之**: 接纳新任务或恢复被抢占的任务
//! 3. **Preempt 最后**: 显存不足时踢出低优先级任务

use serde::{Deserialize, Serialize};

/// 请求 ID
pub type RequestId = String;

/// 调度器输出的批次计划
///
/// 描述了下一个 step 需要执行的操作
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchPlan {
    /// 需要进行 Prefill (初次推理) 的请求列表
    ///
    /// 包括：
    /// - 新来的请求（首次进入 Running 队列）
    /// - 从 Swap 恢复的请求
    pub prefills: Vec<RequestId>,

    /// 需要进行 Decode (生成下一个 Token) 的请求列表
    ///
    /// 这些是正在运行的请求，需要继续生成
    pub decodes: Vec<RequestId>,

    /// 需要被抢占 (Preempt/Swap Out) 的请求列表
    ///
    /// 显存不足时，这些请求会被踢出 GPU，可能：
    /// - Swap to CPU memory (如果支持)
    /// - 丢弃 KV Cache，稍后重新计算
    pub preempts: Vec<RequestId>,

    /// 已完成的请求列表
    ///
    /// 这些请求已生成完毕或触发停止条件
    pub finished: Vec<RequestId>,
}

impl BatchPlan {
    /// 创建空的批次计划
    pub fn new() -> Self {
        Self::default()
    }

    /// 检查批次计划是否为空（没有任何操作）
    pub fn is_empty(&self) -> bool {
        self.prefills.is_empty()
            && self.decodes.is_empty()
            && self.preempts.is_empty()
            && self.finished.is_empty()
    }

    /// 获取总的活跃请求数（Prefill + Decode）
    pub fn total_active_requests(&self) -> usize {
        self.prefills.len() + self.decodes.len()
    }

    /// 获取 Prefill 请求数
    pub fn num_prefills(&self) -> usize {
        self.prefills.len()
    }

    /// 获取 Decode 请求数
    pub fn num_decodes(&self) -> usize {
        self.decodes.len()
    }

    /// 获取 Preempt 请求数
    pub fn num_preempts(&self) -> usize {
        self.preempts.len()
    }

    /// 获取 Finished 请求数
    pub fn num_finished(&self) -> usize {
        self.finished.len()
    }

    /// 添加 Prefill 请求
    pub fn add_prefill(&mut self, request_id: RequestId) {
        self.prefills.push(request_id);
    }

    /// 添加 Decode 请求
    pub fn add_decode(&mut self, request_id: RequestId) {
        self.decodes.push(request_id);
    }

    /// 添加 Preempt 请求
    pub fn add_preempt(&mut self, request_id: RequestId) {
        self.preempts.push(request_id);
    }

    /// 添加 Finished 请求
    pub fn add_finished(&mut self, request_id: RequestId) {
        self.finished.push(request_id);
    }

    /// 清空批次计划
    pub fn clear(&mut self) {
        self.prefills.clear();
        self.decodes.clear();
        self.preempts.clear();
        self.finished.clear();
    }
}

/// 调度统计信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleStats {
    /// 总调度次数
    pub total_schedules: u64,

    /// Prefill 总数
    pub total_prefills: u64,

    /// Decode 总数
    pub total_decodes: u64,

    /// Preempt 总数
    pub total_preempts: u64,

    /// 平均 batch size（活跃请求数）
    pub avg_batch_size: f64,

    /// 最大 batch size
    pub max_batch_size: usize,

    /// 显存耗尽次数（触发抢占的次数）
    pub oom_count: u64,
}

impl ScheduleStats {
    /// 创建新的统计信息
    pub fn new() -> Self {
        Self::default()
    }

    /// 更新统计信息
    pub fn update(&mut self, plan: &BatchPlan) {
        self.total_schedules += 1;
        self.total_prefills += plan.num_prefills() as u64;
        self.total_decodes += plan.num_decodes() as u64;
        self.total_preempts += plan.num_preempts() as u64;

        let batch_size = plan.total_active_requests();
        self.max_batch_size = self.max_batch_size.max(batch_size);

        // 更新平均 batch size（增量计算）
        let n = self.total_schedules as f64;
        self.avg_batch_size = (self.avg_batch_size * (n - 1.0) + batch_size as f64) / n;

        if plan.num_preempts() > 0 {
            self.oom_count += 1;
        }
    }

    /// 重置统计信息
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_plan_basic() {
        let mut plan = BatchPlan::new();

        assert!(plan.is_empty());
        assert_eq!(plan.total_active_requests(), 0);

        plan.add_prefill("req1".to_string());
        plan.add_decode("req2".to_string());
        plan.add_preempt("req3".to_string());

        assert!(!plan.is_empty());
        assert_eq!(plan.total_active_requests(), 2);
        assert_eq!(plan.num_prefills(), 1);
        assert_eq!(plan.num_decodes(), 1);
        assert_eq!(plan.num_preempts(), 1);

        plan.clear();
        assert!(plan.is_empty());
    }

    #[test]
    fn test_schedule_stats() {
        let mut stats = ScheduleStats::new();

        let mut plan = BatchPlan::new();
        plan.add_prefill("req1".to_string());
        plan.add_decode("req2".to_string());

        stats.update(&plan);

        assert_eq!(stats.total_schedules, 1);
        assert_eq!(stats.total_prefills, 1);
        assert_eq!(stats.total_decodes, 1);
        assert_eq!(stats.avg_batch_size, 2.0);
        assert_eq!(stats.max_batch_size, 2);

        stats.reset();
        assert_eq!(stats.total_schedules, 0);
    }
}
