//! Block Allocator - 管理物理块的分配和释放
//!
//! BlockAllocator 使用空闲链表（Free List）管理物理块，并维护引用计数（Reference Count）
//! 来支持 Copy-on-Write (CoW) 和多个 Sequence 共享同一物理块。
//!
//! # 核心功能
//!
//! 1. **分配 (Allocate)**: 从空闲链表分配物理块
//! 2. **释放 (Free)**: 将物理块归还到空闲链表
//! 3. **引用计数 (RefCount)**: 支持多个 Sequence 共享同一物理块
//! 4. **CoW (Copy-on-Write)**: 在写入共享块前先复制
//!
//! # 数据结构
//!
//! ```text
//! BlockAllocator
//!   ├─ Free List: [0, 1, 5, 7, 9, ...]  (空闲块 ID)
//!   ├─ RefCount:  {3: 2, 4: 1, ...}     (块 ID -> 引用计数)
//!   └─ Stats:     {total, free, used}    (统计信息)
//! ```

use std::collections::{HashMap, VecDeque};

use super::block_table::PhysicalBlockId;

/// Block 分配器配置
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// 总块数（从 Worker 获取）
    pub total_blocks: usize,

    /// Block 大小（每个 Block 包含的 Slot 数量）
    pub block_size: usize,

    /// 是否启用 CoW
    pub enable_cow: bool,
}

/// Block 分配器统计信息
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    /// 总块数
    pub total_blocks: usize,

    /// 空闲块数
    pub free_blocks: usize,

    /// 已分配块数
    pub allocated_blocks: usize,

    /// 共享块数（引用计数 > 1）
    pub shared_blocks: usize,

    /// 总分配次数
    pub total_allocations: u64,

    /// 总释放次数
    pub total_frees: u64,

    /// CoW 复制次数
    pub cow_copies: u64,
}

/// Block 分配器
///
/// 使用空闲链表管理物理块，支持引用计数和 CoW
pub struct BlockAllocator {
    /// 配置
    config: AllocatorConfig,

    /// 空闲链表（使用 VecDeque 实现队列，FIFO）
    free_list: VecDeque<PhysicalBlockId>,

    /// 引用计数（块 ID -> 引用数）
    /// 只记录引用数 > 0 的块
    ref_counts: HashMap<PhysicalBlockId, usize>,

    /// 统计信息
    stats: AllocatorStats,
}

impl BlockAllocator {
    /// 创建新的 BlockAllocator
    ///
    /// # Arguments
    /// * `config` - 分配器配置
    pub fn new(config: AllocatorConfig) -> Self {
        // 初始化空闲链表，包含所有块 ID
        let mut free_list = VecDeque::with_capacity(config.total_blocks);
        for block_id in 0..config.total_blocks as PhysicalBlockId {
            free_list.push_back(block_id);
        }

        let stats = AllocatorStats {
            total_blocks: config.total_blocks,
            free_blocks: config.total_blocks,
            allocated_blocks: 0,
            shared_blocks: 0,
            total_allocations: 0,
            total_frees: 0,
            cow_copies: 0,
        };

        Self {
            config,
            free_list,
            ref_counts: HashMap::new(),
            stats,
        }
    }

    /// 分配一个物理块
    ///
    /// # Returns
    /// - `Some(block_id)` 如果分配成功
    /// - `None` 如果没有空闲块
    pub fn allocate(&mut self) -> Option<PhysicalBlockId> {
        let block_id = self.free_list.pop_front()?;

        // 初始化引用计数为 1
        self.ref_counts.insert(block_id, 1);

        // 更新统计
        self.stats.free_blocks -= 1;
        self.stats.allocated_blocks += 1;
        self.stats.total_allocations += 1;

        Some(block_id)
    }

    /// 批量分配物理块
    ///
    /// # Arguments
    /// * `count` - 需要分配的块数
    ///
    /// # Returns
    /// - `Some(Vec<block_id>)` 如果全部分配成功
    /// - `None` 如果空闲块不足（不会部分分配）
    pub fn allocate_batch(&mut self, count: usize) -> Option<Vec<PhysicalBlockId>> {
        if self.free_list.len() < count {
            return None;
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(block_id) = self.allocate() {
                blocks.push(block_id);
            } else {
                // 不应该发生，因为已经检查过数量
                // 如果发生，需要回滚已分配的块
                for &b in &blocks {
                    self.free(b);
                }
                return None;
            }
        }

        Some(blocks)
    }

    /// 释放一个物理块
    ///
    /// 减少引用计数，如果引用计数降为 0，则归还到空闲链表
    ///
    /// # Arguments
    /// * `block_id` - 物理块 ID
    pub fn free(&mut self, block_id: PhysicalBlockId) {
        let ref_count = self.ref_counts.get_mut(&block_id);

        match ref_count {
            Some(count) => {
                *count -= 1;

                if *count == 0 {
                    // 引用计数归零，归还到空闲链表
                    self.ref_counts.remove(&block_id);
                    self.free_list.push_back(block_id);

                    // 更新统计
                    self.stats.free_blocks += 1;
                    self.stats.allocated_blocks -= 1;
                    self.stats.total_frees += 1;
                } else if *count == 1 {
                    // 从共享块降为非共享块
                    self.stats.shared_blocks = self.stats.shared_blocks.saturating_sub(1);
                }
            }
            None => {
                // 警告：尝试释放未分配的块
                eprintln!("Warning: Attempted to free unallocated block {}", block_id);
            }
        }
    }

    /// 批量释放物理块
    pub fn free_batch(&mut self, block_ids: &[PhysicalBlockId]) {
        for &block_id in block_ids {
            self.free(block_id);
        }
    }

    /// 增加物理块的引用计数
    ///
    /// 用于当多个 Sequence 共享同一物理块时（例如 Beam Search fork）
    pub fn incref(&mut self, block_id: PhysicalBlockId) -> Result<(), String> {
        let ref_count = self
            .ref_counts
            .get_mut(&block_id)
            .ok_or_else(|| format!("Block {} not allocated", block_id))?;

        if *ref_count == 1 {
            // 从非共享块变为共享块
            self.stats.shared_blocks += 1;
        }

        *ref_count += 1;
        Ok(())
    }

    /// 批量增加引用计数
    pub fn incref_batch(&mut self, block_ids: &[PhysicalBlockId]) -> Result<(), String> {
        for &block_id in block_ids {
            self.incref(block_id)?;
        }
        Ok(())
    }

    /// 获取物理块的引用计数
    pub fn get_refcount(&self, block_id: PhysicalBlockId) -> usize {
        self.ref_counts.get(&block_id).copied().unwrap_or(0)
    }

    /// 减少物理块的引用计数
    ///
    /// 当引用计数降至 0 时，自动释放该块
    pub fn decref(&mut self, block_id: PhysicalBlockId) -> Result<(), String> {
        let ref_count = self
            .ref_counts
            .get_mut(&block_id)
            .ok_or_else(|| format!("Block {} not allocated", block_id))?;

        if *ref_count == 0 {
            return Err(format!("Block {} already has refcount 0", block_id));
        }

        *ref_count -= 1;

        // 如果引用计数降至 0，释放该块
        if *ref_count == 0 {
            self.free(block_id);
        } else if *ref_count == 1 {
            // 从共享块变为非共享块
            self.stats.shared_blocks = self.stats.shared_blocks.saturating_sub(1);
        }

        Ok(())
    }

    /// 批量减少引用计数
    pub fn decref_batch(&mut self, block_ids: &[PhysicalBlockId]) -> Result<(), String> {
        for &block_id in block_ids {
            self.decref(block_id)?;
        }
        Ok(())
    }

    /// Copy-on-Write: 如果块被共享，复制一个新块
    ///
    /// # Arguments
    /// * `block_id` - 原始块 ID
    ///
    /// # Returns
    /// - `Ok(new_block_id)` 如果需要复制，返回新块 ID
    /// - `Ok(block_id)` 如果不需要复制，返回原块 ID
    /// - `Err` 如果分配失败
    pub fn copy_on_write(&mut self, block_id: PhysicalBlockId) -> Result<PhysicalBlockId, String> {
        let ref_count = self.get_refcount(block_id);

        if ref_count == 0 {
            return Err(format!("Block {} not allocated", block_id));
        }

        if ref_count == 1 {
            // 没有共享，不需要复制
            return Ok(block_id);
        }

        // 块被共享，需要复制
        let new_block_id = self
            .allocate()
            .ok_or_else(|| "Out of memory: cannot allocate block for CoW".to_string())?;

        // 减少原块的引用计数
        self.free(block_id);

        // 更新统计
        self.stats.cow_copies += 1;

        Ok(new_block_id)
    }

    /// 检查是否有足够的空闲块
    pub fn can_allocate(&self, count: usize) -> bool {
        self.free_list.len() >= count
    }

    /// 获取空闲块数量
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// 获取已分配块数量
    pub fn num_allocated_blocks(&self) -> usize {
        self.stats.allocated_blocks
    }

    /// 获取统计信息
    pub fn stats(&self) -> &AllocatorStats {
        &self.stats
    }

    /// 获取配置
    pub fn config(&self) -> &AllocatorConfig {
        &self.config
    }

    /// 重置分配器（用于测试）
    pub fn reset(&mut self) {
        self.free_list.clear();
        for block_id in 0..self.config.total_blocks as PhysicalBlockId {
            self.free_list.push_back(block_id);
        }
        self.ref_counts.clear();
        self.stats = AllocatorStats {
            total_blocks: self.config.total_blocks,
            free_blocks: self.config.total_blocks,
            allocated_blocks: 0,
            shared_blocks: 0,
            total_allocations: 0,
            total_frees: 0,
            cow_copies: 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_allocator() -> BlockAllocator {
        let config = AllocatorConfig {
            total_blocks: 100,
            block_size: 16,
            enable_cow: true,
        };
        BlockAllocator::new(config)
    }

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = create_test_allocator();

        // 分配
        let block1 = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free_blocks(), 99);
        assert_eq!(allocator.get_refcount(block1), 1);

        // 释放
        allocator.free(block1);
        assert_eq!(allocator.num_free_blocks(), 100);
        assert_eq!(allocator.get_refcount(block1), 0);
    }

    #[test]
    fn test_batch_allocate() {
        let mut allocator = create_test_allocator();

        let blocks = allocator.allocate_batch(10).unwrap();
        assert_eq!(blocks.len(), 10);
        assert_eq!(allocator.num_free_blocks(), 90);

        // 尝试分配超过剩余数量
        assert!(allocator.allocate_batch(91).is_none());
    }

    #[test]
    fn test_refcount() {
        let mut allocator = create_test_allocator();

        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.get_refcount(block), 1);

        // 增加引用
        allocator.incref(block).unwrap();
        assert_eq!(allocator.get_refcount(block), 2);
        assert_eq!(allocator.stats().shared_blocks, 1);

        // 第一次 free，引用计数降为 1
        allocator.free(block);
        assert_eq!(allocator.get_refcount(block), 1);
        assert_eq!(allocator.stats().shared_blocks, 0);

        // 第二次 free，引用计数降为 0，归还到空闲链表
        allocator.free(block);
        assert_eq!(allocator.get_refcount(block), 0);
        assert_eq!(allocator.num_free_blocks(), 100);
    }

    #[test]
    fn test_copy_on_write() {
        let mut allocator = create_test_allocator();

        let block = allocator.allocate().unwrap();

        // 没有共享，不需要复制
        let result = allocator.copy_on_write(block).unwrap();
        assert_eq!(result, block);

        // 增加引用，模拟共享
        allocator.incref(block).unwrap();
        assert_eq!(allocator.get_refcount(block), 2);

        // CoW 应该分配新块
        let new_block = allocator.copy_on_write(block).unwrap();
        assert_ne!(new_block, block);
        assert_eq!(allocator.get_refcount(block), 1); // 原块引用减1
        assert_eq!(allocator.get_refcount(new_block), 1);
        assert_eq!(allocator.stats().cow_copies, 1);
    }

    #[test]
    fn test_can_allocate() {
        let mut allocator = create_test_allocator();

        assert!(allocator.can_allocate(50));
        assert!(allocator.can_allocate(100));
        assert!(!allocator.can_allocate(101));

        // 分配50个块
        allocator.allocate_batch(50);
        assert!(allocator.can_allocate(50));
        assert!(!allocator.can_allocate(51));
    }

    #[test]
    fn test_stats() {
        let mut allocator = create_test_allocator();

        let blocks = allocator.allocate_batch(10).unwrap();
        assert_eq!(allocator.stats().total_allocations, 10);
        assert_eq!(allocator.stats().allocated_blocks, 10);

        allocator.free_batch(&blocks);
        assert_eq!(allocator.stats().total_frees, 10);
        assert_eq!(allocator.stats().allocated_blocks, 0);
    }
}
