//! Block Table - 记录每个 Sequence 占用的物理 Block
//!
//! BlockTable 维护了一个 Sequence 从逻辑 Slot 到物理 Block 的映射关系。
//! 在 PagedAttention 中，每个 Block 包含固定数量的 Slot（例如 16 个）。
//!
//! # 核心概念
//!
//! - **Slot**: 一个 Token 的 KV Cache 存储位置
//! - **Block**: 一组连续的 Slot，是分配的最小单位（例如 16 个 Slot）
//! - **LogicalBlock**: Sequence 角度的逻辑块号（0, 1, 2, ...）
//! - **PhysicalBlock**: Worker 实际内存中的物理块 ID
//!
//! # 映射关系
//!
//! ```text
//! Sequence (100 tokens, block_size=16)
//!   ├─ Logical Block 0 (slot 0-15)   -> Physical Block 3
//!   ├─ Logical Block 1 (slot 16-31)  -> Physical Block 7
//!   ├─ Logical Block 2 (slot 32-47)  -> Physical Block 12
//!   └─ ...
//! ```

use std::collections::HashMap;

/// 物理块 ID
pub type PhysicalBlockId = u32;

/// 逻辑块号
pub type LogicalBlockId = usize;

/// Sequence ID
pub type SequenceId = String;

/// BlockTable - 维护单个 Sequence 的块映射
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Sequence ID
    pub seq_id: SequenceId,

    /// 逻辑块号 -> 物理块 ID 的映射
    /// Vec 的索引就是逻辑块号
    pub blocks: Vec<PhysicalBlockId>,

    /// Block 大小（每个 Block 包含的 Slot 数量）
    pub block_size: usize,

    /// 当前已使用的 Slot 数量
    pub num_slots: usize,

    /// 是否支持 Copy-on-Write（用于 Beam Search）
    pub enable_cow: bool,
}

impl BlockTable {
    /// 创建新的 BlockTable
    pub fn new(seq_id: SequenceId, block_size: usize, enable_cow: bool) -> Self {
        Self {
            seq_id,
            blocks: Vec::new(),
            block_size,
            num_slots: 0,
            enable_cow,
        }
    }

    /// 添加一个物理块
    ///
    /// # Arguments
    /// * `physical_block_id` - 物理块 ID
    pub fn append_block(&mut self, physical_block_id: PhysicalBlockId) {
        self.blocks.push(physical_block_id);
    }

    /// 追加一个 Slot
    ///
    /// 如果当前最后一个 Block 已满，调用者需要先分配新 Block
    pub fn append_slot(&mut self) -> Result<(), String> {
        let required_blocks = (self.num_slots + 1 + self.block_size - 1) / self.block_size;

        if required_blocks > self.blocks.len() {
            return Err(format!(
                "Need to allocate new block: have {} blocks, need {}",
                self.blocks.len(),
                required_blocks
            ));
        }

        self.num_slots += 1;
        Ok(())
    }

    /// 获取当前需要的块数量
    pub fn num_blocks_needed(&self) -> usize {
        if self.num_slots == 0 {
            0
        } else {
            (self.num_slots + self.block_size - 1) / self.block_size
        }
    }

    /// 获取已分配的块数量
    pub fn num_blocks_allocated(&self) -> usize {
        self.blocks.len()
    }

    /// 检查最后一个 Block 是否已满
    pub fn is_last_block_full(&self) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        self.num_slots % self.block_size == 0
    }

    /// 获取最后一个 Block 的可用 Slot 数量
    pub fn last_block_free_slots(&self) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        let used_in_last = self.num_slots % self.block_size;
        if used_in_last == 0 {
            0 // 最后一个块已满
        } else {
            self.block_size - used_in_last
        }
    }

    /// 获取所有物理块 ID
    pub fn get_physical_blocks(&self) -> &[PhysicalBlockId] {
        &self.blocks
    }

    /// 获取指定 Slot 对应的物理块 ID 和块内偏移
    ///
    /// # Returns
    /// (physical_block_id, slot_offset_in_block)
    pub fn get_slot_mapping(&self, slot_index: usize) -> Option<(PhysicalBlockId, usize)> {
        if slot_index >= self.num_slots {
            return None;
        }

        let logical_block = slot_index / self.block_size;
        let slot_offset = slot_index % self.block_size;

        self.blocks
            .get(logical_block)
            .map(|&physical_id| (physical_id, slot_offset))
    }

    /// 克隆 BlockTable（用于 Beam Search 分支）
    ///
    /// 注意：克隆的 BlockTable 与原表共享物理块，
    /// 如果启用 CoW，需要在写入前复制物理块
    pub fn fork(&self, new_seq_id: SequenceId) -> Self {
        Self {
            seq_id: new_seq_id,
            blocks: self.blocks.clone(),
            block_size: self.block_size,
            num_slots: self.num_slots,
            enable_cow: self.enable_cow,
        }
    }

    /// 清空 BlockTable
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.num_slots = 0;
    }
}

/// BlockTableManager - 管理多个 Sequence 的 BlockTable
#[derive(Debug)]
pub struct BlockTableManager {
    /// Sequence ID -> BlockTable 的映射
    tables: HashMap<SequenceId, BlockTable>,

    /// 默认 Block 大小
    default_block_size: usize,

    /// 是否启用 CoW
    enable_cow: bool,
}

impl BlockTableManager {
    /// 创建新的 BlockTableManager
    pub fn new(default_block_size: usize, enable_cow: bool) -> Self {
        Self {
            tables: HashMap::new(),
            default_block_size,
            enable_cow,
        }
    }

    /// 为新 Sequence 创建 BlockTable
    pub fn create_table(&mut self, seq_id: SequenceId) -> &mut BlockTable {
        self.tables.entry(seq_id.clone()).or_insert_with(|| {
            BlockTable::new(seq_id, self.default_block_size, self.enable_cow)
        })
    }

    /// 获取 Sequence 的 BlockTable
    pub fn get_table(&self, seq_id: &SequenceId) -> Option<&BlockTable> {
        self.tables.get(seq_id)
    }

    /// 获取 Sequence 的可变 BlockTable
    pub fn get_table_mut(&mut self, seq_id: &SequenceId) -> Option<&mut BlockTable> {
        self.tables.get_mut(seq_id)
    }

    /// 删除 Sequence 的 BlockTable
    pub fn remove_table(&mut self, seq_id: &SequenceId) -> Option<BlockTable> {
        self.tables.remove(seq_id)
    }

    /// 克隆 Sequence 的 BlockTable（用于 Beam Search）
    pub fn fork_table(&mut self, src_seq_id: &SequenceId, new_seq_id: SequenceId) -> Result<(), String> {
        let src_table = self
            .tables
            .get(src_seq_id)
            .ok_or_else(|| format!("Source sequence {} not found", src_seq_id))?
            .fork(new_seq_id.clone());

        self.tables.insert(new_seq_id, src_table);
        Ok(())
    }

    /// 获取所有活跃的 Sequence ID
    pub fn active_sequences(&self) -> Vec<&SequenceId> {
        self.tables.keys().collect()
    }

    /// 获取 BlockTable 数量
    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_basic() {
        let mut table = BlockTable::new("seq1".to_string(), 16, false);

        // 分配第一个块
        table.append_block(10);
        assert_eq!(table.num_blocks_allocated(), 1);
        assert_eq!(table.num_blocks_needed(), 0);

        // 添加 slot
        for _ in 0..16 {
            assert!(table.append_slot().is_ok());
        }
        assert_eq!(table.num_slots, 16);
        assert!(table.is_last_block_full());

        // 需要第二个块
        assert_eq!(table.num_blocks_needed(), 1);
        assert!(table.append_slot().is_err()); // 没有新块，应该失败

        // 分配第二个块
        table.append_block(25);
        assert!(table.append_slot().is_ok());
        assert_eq!(table.num_slots, 17);
        assert_eq!(table.last_block_free_slots(), 15);
    }

    #[test]
    fn test_slot_mapping() {
        let mut table = BlockTable::new("seq1".to_string(), 16, false);
        table.append_block(100);
        table.append_block(200);

        for _ in 0..20 {
            let _ = table.append_slot();
        }

        // Slot 0-15 在 Block 100
        assert_eq!(table.get_slot_mapping(0), Some((100, 0)));
        assert_eq!(table.get_slot_mapping(15), Some((100, 15)));

        // Slot 16-19 在 Block 200
        assert_eq!(table.get_slot_mapping(16), Some((200, 0)));
        assert_eq!(table.get_slot_mapping(19), Some((200, 3)));

        // Slot 20 不存在
        assert_eq!(table.get_slot_mapping(20), None);
    }

    #[test]
    fn test_block_table_manager() {
        let mut manager = BlockTableManager::new(16, false);

        // 创建表
        let table = manager.create_table("seq1".to_string());
        table.append_block(10);
        assert_eq!(manager.num_tables(), 1);

        // Fork 表
        assert!(manager.fork_table(&"seq1".to_string(), "seq2".to_string()).is_ok());
        assert_eq!(manager.num_tables(), 2);

        // 验证 fork 的表共享物理块
        let table2 = manager.get_table(&"seq2".to_string()).unwrap();
        assert_eq!(table2.get_physical_blocks(), &[10]);

        // 删除表
        manager.remove_table(&"seq1".to_string());
        assert_eq!(manager.num_tables(), 1);
    }
}
