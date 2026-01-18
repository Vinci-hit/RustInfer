//! Memory Management Module - 管理 KV Cache 的物理块分配与共享
//!
//! 本模块实现了类似 vLLM 的 PagedAttention 内存管理机制，包括：
//!
//! # 核心组件
//!
//! 1. **BlockAllocator**: 管理物理块的分配和释放
//!    - 空闲链表 (Free List)
//!    - 引用计数 (Reference Counting)
//!    - Copy-on-Write (CoW)
//!
//! 2. **BlockTableManager**: 管理每个 Sequence 的块映射
//!    - 逻辑块 -> 物理块的映射
//!    - Slot 级别的精确追踪
//!
//! 3. **RadixTree**: 前缀树实现块共享
//!    - 前缀匹配
//!    - 跨 Sequence 共享
//!    - 引用计数管理
//!
//! # 架构设计
//!
//! ```text
//! MemoryManager
//!   ├─ BlockAllocator      (物理块池管理)
//!   ├─ BlockTableManager   (Sequence 映射表)
//!   └─ RadixTree           (前缀共享索引)
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use infer_scheduler::memory::{MemoryManager, MemoryConfig};
//!
//! // 初始化内存管理器
//! let config = MemoryConfig {
//!     total_blocks: 1000,
//!     block_size: 16,
//!     enable_cow: true,
//!     enable_prefix_cache: true,
//! };
//! let mut memory_manager = MemoryManager::new(config);
//!
//! // 为新 Sequence 分配块
//! memory_manager.allocate_for_sequence("seq1", 2)?;
//!
//! // 添加 Slot
//! memory_manager.append_slots("seq1", 10)?;
//!
//! // 前缀匹配
//! let prefix_match = memory_manager.match_prefix(&[1, 2, 3]);
//! ```

pub mod allocator;
pub mod block_table;
pub mod radix_tree;
pub mod eviction;

// Re-export core types
pub use allocator::{AllocatorConfig, AllocatorStats, BlockAllocator};
pub use block_table::{
    BlockTable, BlockTableManager, PhysicalBlockId, SequenceId,
};
pub use radix_tree::{PrefixMatch, RadixTree};
pub use eviction::{EvictionList, EvictionStats};

/// 内存管理器配置
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// 总物理块数量
    pub total_blocks: usize,

    /// 每个块的大小（Slot 数量）
    pub block_size: usize,

    /// 是否启用 Copy-on-Write
    pub enable_cow: bool,

    /// 是否启用前缀缓存
    pub enable_prefix_cache: bool,

    /// 前缀缓存最大容量（token 数）
    /// None 表示无限制
    pub prefix_cache_capacity: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            total_blocks: 1000,
            block_size: 16,
            enable_cow: true,
            enable_prefix_cache: true,
            prefix_cache_capacity: Some(10000), // 10k tokens
        }
    }
}

/// 统一的内存管理器
///
/// 整合了 BlockAllocator、BlockTableManager 和 RadixTree，
/// 提供简化的高层 API
pub struct MemoryManager {
    /// 配置
    config: MemoryConfig,

    /// 物理块分配器
    allocator: BlockAllocator,

    /// BlockTable 管理器
    block_tables: BlockTableManager,

    /// 前缀树（用于 Prompt 缓存）
    radix_tree: Option<RadixTree>,
}

impl MemoryManager {
    /// 创建新的内存管理器
    pub fn new(config: MemoryConfig) -> Self {
        let allocator_config = AllocatorConfig {
            total_blocks: config.total_blocks,
            block_size: config.block_size,
            enable_cow: config.enable_cow,
        };

        let allocator = BlockAllocator::new(allocator_config);
        let block_tables = BlockTableManager::new(config.block_size, config.enable_cow);

        let radix_tree = if config.enable_prefix_cache {
            if let Some(capacity) = config.prefix_cache_capacity {
                Some(RadixTree::with_eviction(capacity))
            } else {
                Some(RadixTree::new())
            }
        } else {
            None
        };

        Self {
            config,
            allocator,
            block_tables,
            radix_tree,
        }
    }

    /// 为新的 Sequence 分配初始块
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `num_blocks` - 需要分配的块数量
    pub fn allocate_for_sequence(
        &mut self,
        seq_id: SequenceId,
        num_blocks: usize,
    ) -> Result<(), String> {
        // 检查是否有足够的空闲块
        if !self.allocator.can_allocate(num_blocks) {
            return Err(format!(
                "Out of memory: need {} blocks, only {} available",
                num_blocks,
                self.allocator.num_free_blocks()
            ));
        }

        // 分配物理块
        let blocks = self
            .allocator
            .allocate_batch(num_blocks)
            .ok_or_else(|| "Failed to allocate blocks".to_string())?;

        // 创建 BlockTable
        let table = self.block_tables.create_table(seq_id.clone());
        for block_id in blocks {
            table.append_block(block_id);
        }

        Ok(())
    }

    /// 为 Sequence 追加 Slot
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `num_slots` - 要追加的 Slot 数量
    pub fn append_slots(&mut self, seq_id: &SequenceId, num_slots: usize) -> Result<(), String> {
        let table = self
            .block_tables
            .get_table_mut(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        for _ in 0..num_slots {
            // 检查是否需要分配新块（在追加 slot 之前检查）
            let needed_blocks_after_append = (table.num_slots + 1 + table.block_size - 1) / table.block_size;
            let allocated_blocks = table.num_blocks_allocated();

            if needed_blocks_after_append > allocated_blocks {
                // 需要分配新块
                let block_id = self
                    .allocator
                    .allocate()
                    .ok_or_else(|| "Out of memory: cannot allocate new block".to_string())?;
                table.append_block(block_id);
            }

            // 追加 Slot
            table.append_slot()?;
        }

        Ok(())
    }

    /// 释放 Sequence 占用的所有块
    pub fn free_sequence(&mut self, seq_id: &SequenceId) -> Result<(), String> {
        let table = self
            .block_tables
            .remove_table(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        // 释放所有物理块
        self.allocator.free_batch(table.get_physical_blocks());

        Ok(())
    }

    /// Fork Sequence（用于 Beam Search）
    ///
    /// # Arguments
    /// * `src_seq_id` - 源 Sequence ID
    /// * `new_seq_id` - 新 Sequence ID
    pub fn fork_sequence(
        &mut self,
        src_seq_id: &SequenceId,
        new_seq_id: SequenceId,
    ) -> Result<(), String> {
        // Fork BlockTable
        self.block_tables.fork_table(src_seq_id, new_seq_id.clone())?;

        // 增加物理块的引用计数
        let table = self
            .block_tables
            .get_table(&new_seq_id)
            .ok_or_else(|| format!("Forked sequence {} not found", new_seq_id))?;

        self.allocator.incref_batch(table.get_physical_blocks())?;

        Ok(())
    }

    /// Copy-on-Write: 为即将写入的块创建副本
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `block_index` - 要写入的逻辑块索引
    pub fn copy_on_write_block(
        &mut self,
        seq_id: &SequenceId,
        block_index: usize,
    ) -> Result<(), String> {
        let table = self
            .block_tables
            .get_table_mut(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let old_block_id = table
            .get_physical_blocks()
            .get(block_index)
            .copied()
            .ok_or_else(|| format!("Block index {} out of range", block_index))?;

        // 执行 CoW
        let new_block_id = self.allocator.copy_on_write(old_block_id)?;

        // 如果分配了新块，更新 BlockTable
        if new_block_id != old_block_id {
            table.blocks[block_index] = new_block_id;
        }

        Ok(())
    }

    /// 前缀匹配（如果启用了前缀缓存）
    pub fn match_prefix(&mut self, tokens: &[i32]) -> Option<PrefixMatch> {
        self.radix_tree.as_mut().map(|tree| tree.match_prefix(tokens))
    }

    /// 插入前缀到缓存（如果启用了前缀缓存）
    pub fn insert_prefix(&mut self, tokens: &[i32], blocks: &[PhysicalBlockId]) {
        if let Some(tree) = self.radix_tree.as_mut() {
            tree.insert(tokens, blocks);
        }
    }

    /// 增加前缀引用计数
    pub fn incref_prefix(&mut self, tokens: &[i32]) {
        if let Some(tree) = self.radix_tree.as_mut() {
            tree.incref_prefix(tokens);
        }
    }

    /// 减少前缀引用计数
    pub fn decref_prefix(&mut self, tokens: &[i32]) {
        if let Some(tree) = self.radix_tree.as_mut() {
            tree.decref_prefix(tokens);
        }
    }

    /// 获取 Sequence 的 BlockTable
    pub fn get_block_table(&self, seq_id: &SequenceId) -> Option<&BlockTable> {
        self.block_tables.get_table(seq_id)
    }

    /// 获取分配器统计信息
    pub fn allocator_stats(&self) -> &AllocatorStats {
        self.allocator.stats()
    }

    /// 获取配置
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// 获取活跃的 Sequence 列表
    pub fn active_sequences(&self) -> Vec<&SequenceId> {
        self.block_tables.active_sequences()
    }

    /// 检查是否可以分配指定数量的块
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.allocator.can_allocate(num_blocks)
    }

    /// 获取空闲块数量
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free_blocks()
    }

    /// 获取已分配块数量
    pub fn num_allocated_blocks(&self) -> usize {
        self.allocator.num_allocated_blocks()
    }

    /// 获取 RadixTree 统计（如果启用）
    pub fn radix_tree_stats(&self) -> Option<(usize, u64, u64)> {
        self.radix_tree.as_ref().map(|tree| {
            (
                tree.num_nodes(),
                tree.total_matches(),
                tree.total_inserts(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manager() -> MemoryManager {
        let config = MemoryConfig {
            total_blocks: 100,
            block_size: 16,
            enable_cow: true,
            enable_prefix_cache: true,
            prefix_cache_capacity: Some(10000),
        };
        MemoryManager::new(config)
    }

    #[test]
    fn test_allocate_and_free_sequence() {
        let mut manager = create_test_manager();

        // 分配 Sequence
        assert!(manager.allocate_for_sequence("seq1".to_string(), 2).is_ok());
        assert_eq!(manager.num_free_blocks(), 98);

        // 获取 BlockTable
        let table = manager.get_block_table(&"seq1".to_string()).unwrap();
        assert_eq!(table.num_blocks_allocated(), 2);

        // 释放 Sequence
        assert!(manager.free_sequence(&"seq1".to_string()).is_ok());
        assert_eq!(manager.num_free_blocks(), 100);
    }

    #[test]
    fn test_append_slots() {
        let mut manager = create_test_manager();

        // 分配 Sequence
        manager.allocate_for_sequence("seq1".to_string(), 1).unwrap();

        // 追加 16 个 Slot（正好填满第一个块）
        assert!(manager.append_slots(&"seq1".to_string(), 16).is_ok());

        let table = manager.get_block_table(&"seq1".to_string()).unwrap();
        assert_eq!(table.num_slots, 16);
        assert_eq!(table.num_blocks_allocated(), 1);

        // 追加第 17 个 Slot，应该自动分配新块
        assert!(manager.append_slots(&"seq1".to_string(), 1).is_ok());

        let table = manager.get_block_table(&"seq1".to_string()).unwrap();
        assert_eq!(table.num_slots, 17);
        assert_eq!(table.num_blocks_allocated(), 2);
    }

    #[test]
    fn test_fork_sequence() {
        let mut manager = create_test_manager();

        // 分配 Sequence
        manager.allocate_for_sequence("seq1".to_string(), 2).unwrap();
        manager.append_slots(&"seq1".to_string(), 20).unwrap();

        // Fork
        assert!(manager
            .fork_sequence(&"seq1".to_string(), "seq2".to_string())
            .is_ok());

        // 验证两个 Sequence 共享物理块
        let table1 = manager.get_block_table(&"seq1".to_string()).unwrap();
        let table2 = manager.get_block_table(&"seq2".to_string()).unwrap();
        assert_eq!(table1.get_physical_blocks(), table2.get_physical_blocks());

        // 验证引用计数
        let block_id = table1.get_physical_blocks()[0];
        assert_eq!(manager.allocator.get_refcount(block_id), 2);
    }

    #[test]
    fn test_copy_on_write() {
        let mut manager = create_test_manager();

        // 分配并 Fork
        manager.allocate_for_sequence("seq1".to_string(), 1).unwrap();
        manager.fork_sequence(&"seq1".to_string(), "seq2".to_string()).unwrap();

        let old_block_id = manager
            .get_block_table(&"seq2".to_string())
            .unwrap()
            .get_physical_blocks()[0];

        // CoW
        assert!(manager.copy_on_write_block(&"seq2".to_string(), 0).is_ok());

        let new_block_id = manager
            .get_block_table(&"seq2".to_string())
            .unwrap()
            .get_physical_blocks()[0];

        // 应该分配了新块
        assert_ne!(old_block_id, new_block_id);

        // 原块引用计数应该减 1
        assert_eq!(manager.allocator.get_refcount(old_block_id), 1);
        assert_eq!(manager.allocator.get_refcount(new_block_id), 1);
    }

    #[test]
    fn test_prefix_cache() {
        let mut manager = create_test_manager();

        // 插入前缀
        let tokens = vec![1, 2, 3, 4];
        let blocks = vec![10, 11, 12, 13];
        manager.insert_prefix(&tokens, &blocks);

        // 匹配前缀
        let result = manager.match_prefix(&vec![1, 2, 3]).unwrap();
        assert_eq!(result.matched_tokens, 3);
        assert_eq!(result.matched_blocks, vec![10, 11, 12]);

        // 统计
        let (nodes, matches, inserts) = manager.radix_tree_stats().unwrap();
        assert!(nodes > 0);
        assert_eq!(matches, 1);
        assert_eq!(inserts, 1);
    }

    #[test]
    fn test_out_of_memory() {
        let config = MemoryConfig {
            total_blocks: 10,
            block_size: 16,
            enable_cow: false,
            enable_prefix_cache: false,
            prefix_cache_capacity: None,
        };
        let mut manager = MemoryManager::new(config);

        // 分配所有块
        assert!(manager.allocate_for_sequence("seq1".to_string(), 10).is_ok());

        // 尝试再分配应该失败
        assert!(manager.allocate_for_sequence("seq2".to_string(), 1).is_err());
    }

    #[test]
    fn test_active_sequences() {
        let mut manager = create_test_manager();

        manager.allocate_for_sequence("seq1".to_string(), 1).unwrap();
        manager.allocate_for_sequence("seq2".to_string(), 1).unwrap();

        let active = manager.active_sequences();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&&"seq1".to_string()));
        assert!(active.contains(&&"seq2".to_string()));
    }
}
