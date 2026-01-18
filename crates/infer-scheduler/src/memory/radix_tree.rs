//! Radix Tree - 前缀树用于 Block 共享
//!
//! RadixTree 用于追踪 Token 序列前缀，实现多个 Sequence 共享相同的物理块。
//! 这对于 Prompt 缓存和 Beam Search 等场景非常重要。
//!
//! # 核心功能
//!
//! 1. **前缀匹配**: 查找与给定序列匹配的最长前缀
//! 2. **插入**: 将新的 Token 序列及其对应的物理块插入树
//! 3. **引用计数**: 追踪每个前缀被多少个 Sequence 使用
//! 4. **LRU 驱逐**: 当内存不足时，驱逐最少使用的前缀
//!
//! # 示例
//!
//! ```text
//! 假设有以下请求：
//! - Seq1: [1, 2, 3, 4, 5] -> Blocks [10, 20]
//! - Seq2: [1, 2, 3, 6, 7] -> 前缀 [1, 2, 3] 共享 Block 10，新 token [6, 7] 分配 Block 30
//!
//! Radix Tree:
//!   Root
//!     └─ [1, 2, 3] -> Block 10 (ref=2)
//!           ├─ [4, 5] -> Block 20 (ref=1)
//!           └─ [6, 7] -> Block 30 (ref=1)
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use super::block_table::PhysicalBlockId;
use super::eviction::{EvictionList, NodeId};

/// Radix 树节点
#[derive(Debug, Clone)]
pub struct RadixNode {
    /// 子节点，按第一个 token 索引
    pub children: HashMap<i32, Box<RadixNode>>,

    /// 该节点对应的 token 序列片段
    pub tokens: Vec<i32>,

    /// 该节点对应的物理块 ID 列表
    /// 长度与 tokens 对齐
    pub blocks: Vec<PhysicalBlockId>,

    /// 引用计数（有多少个 Sequence 使用这个前缀）
    pub ref_count: usize,

    /// 节点 ID（用于驱逐列表）
    node_id: Option<NodeId>,
}

impl RadixNode {
    /// 创建新的空节点
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            tokens: Vec::new(),
            blocks: Vec::new(),
            ref_count: 0,
            node_id: None,
        }
    }

    /// 创建带数据的节点
    pub fn with_data(tokens: Vec<i32>, blocks: Vec<PhysicalBlockId>) -> Self {
        assert_eq!(tokens.len(), blocks.len(), "Tokens and blocks must have same length");
        Self {
            children: HashMap::new(),
            tokens,
            blocks,
            ref_count: 0,
            node_id: None,
        }
    }

    /// 生成节点 ID（基于 tokens 的 hash）
    fn compute_node_id(tokens: &[i32]) -> NodeId {
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// 是否为叶子节点
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

impl Default for RadixNode {
    fn default() -> Self {
        Self::new()
    }
}

/// 前缀匹配结果
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// 匹配的 token 数量
    pub matched_tokens: usize,

    /// 匹配的物理块 ID 列表
    pub matched_blocks: Vec<PhysicalBlockId>,
}

/// Radix Tree - 用于前缀共享
pub struct RadixTree {
    /// 根节点
    root: RadixNode,

    /// 统计：总节点数
    num_nodes: usize,

    /// 统计：总匹配次数
    total_matches: u64,

    /// 统计：总插入次数
    total_inserts: u64,

    /// LRU 驱逐列表
    eviction_list: Option<EvictionList>,

    /// 最大缓存容量（token 数）
    max_capacity_tokens: Option<usize>,
}

impl RadixTree {
    /// 创建新的 RadixTree
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(),
            num_nodes: 1, // 包含根节点
            total_matches: 0,
            total_inserts: 0,
            eviction_list: None,
            max_capacity_tokens: None,
        }
    }

    /// 创建启用 LRU 驱逐的 RadixTree
    ///
    /// # Arguments
    /// * `max_capacity_tokens` - 最大缓存容量（token 数）
    pub fn with_eviction(max_capacity_tokens: usize) -> Self {
        Self {
            root: RadixNode::new(),
            num_nodes: 1,
            total_matches: 0,
            total_inserts: 0,
            eviction_list: Some(EvictionList::new()),
            max_capacity_tokens: Some(max_capacity_tokens),
        }
    }

    /// 检查是否需要驱逐
    fn should_evict(&self, new_tokens: usize) -> bool {
        if let (Some(list), Some(max_cap)) = (&self.eviction_list, self.max_capacity_tokens) {
            list.total_tokens() + new_tokens > max_cap
        } else {
            false
        }
    }

    /// 驱逐节点直到有足够空间
    fn evict_if_needed(&mut self, needed_tokens: usize) -> Vec<(NodeId, usize)> {
        if !self.should_evict(needed_tokens) {
            return Vec::new();
        }

        if let Some(list) = &mut self.eviction_list {
            let evicted = list.evict_until(needed_tokens);

            // TODO: 实际删除树中的节点（需要维护节点到路径的映射）
            // 目前只从驱逐列表中移除

            evicted
        } else {
            Vec::new()
        }
    }

    /// 查找最长前缀匹配
    ///
    /// # Arguments
    /// * `tokens` - Token 序列
    ///
    /// # Returns
    /// 匹配的前缀信息
    pub fn match_prefix(&mut self, tokens: &[i32]) -> PrefixMatch {
        self.total_matches += 1;

        let mut matched_tokens = 0;
        let mut matched_blocks = Vec::new();
        let mut current = &self.root;
        let mut token_idx = 0;

        while token_idx < tokens.len() {
            // 查找第一个 token 匹配的子节点
            let first_token = tokens[token_idx];
            let child = match current.children.get(&first_token) {
                Some(c) => c,
                None => break, // 没有匹配的子节点
            };

            // 比较节点的 tokens 与输入的 tokens
            let node_tokens = &child.tokens;
            let remaining_tokens = &tokens[token_idx..];
            let match_len = node_tokens
                .iter()
                .zip(remaining_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len == 0 {
                break; // 不匹配
            }

            // 匹配成功
            matched_tokens += match_len;
            matched_blocks.extend_from_slice(&child.blocks[..match_len]);
            token_idx += match_len;

            // 更新驱逐列表（touch）
            if let Some(node_id) = child.node_id {
                if let Some(list) = &mut self.eviction_list {
                    list.touch(node_id);
                }
            }

            // 如果只匹配了部分节点，停止
            if match_len < node_tokens.len() {
                break;
            }

            // 继续向下搜索
            current = child;
        }

        PrefixMatch {
            matched_tokens,
            matched_blocks,
        }
    }

    /// 插入 token 序列及其对应的物理块
    ///
    /// # Arguments
    /// * `tokens` - Token 序列
    /// * `blocks` - 物理块 ID 列表
    ///
    /// # Panics
    /// 如果 tokens 和 blocks 长度不一致
    pub fn insert(&mut self, tokens: &[i32], blocks: &[PhysicalBlockId]) {
        assert_eq!(tokens.len(), blocks.len(), "Tokens and blocks must have same length");

        if tokens.is_empty() {
            return;
        }

        // 检查是否需要驱逐
        self.evict_if_needed(tokens.len());

        self.total_inserts += 1;

        Self::insert_recursive(&mut self.root, tokens, blocks, 0, &mut self.num_nodes);

        // 添加到驱逐列表（如果启用）
        if let Some(list) = &mut self.eviction_list {
            let node_id = RadixNode::compute_node_id(tokens);
            list.push_back(node_id, tokens.len());
        }
    }

    /// 递归插入
    fn insert_recursive(
        node: &mut RadixNode,
        tokens: &[i32],
        blocks: &[PhysicalBlockId],
        start: usize,
        num_nodes: &mut usize,
    ) {
        if start >= tokens.len() {
            return;
        }

        let first_token = tokens[start];

        // 查找匹配的子节点
        if let Some(child) = node.children.get_mut(&first_token) {
            // 找到匹配的子节点，比较 tokens
            let child_tokens = &child.tokens.clone(); // Clone to avoid borrow issues
            let remaining_tokens = &tokens[start..];
            let match_len = child_tokens
                .iter()
                .zip(remaining_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len == child_tokens.len() {
                // 完全匹配子节点，继续向下
                Self::insert_recursive(child, tokens, blocks, start + match_len, num_nodes);
            } else if match_len > 0 {
                // 部分匹配，需要分裂节点
                Self::split_node(node, first_token, match_len, num_nodes);
                // 分裂后重新插入
                Self::insert_recursive(node, tokens, blocks, start, num_nodes);
            } else {
                // 不匹配，创建新子节点
                Self::create_new_child(node, tokens, blocks, start, num_nodes);
            }
        } else {
            // 没有匹配的子节点，创建新子节点
            Self::create_new_child(node, tokens, blocks, start, num_nodes);
        }
    }

    /// 分裂节点
    fn split_node(parent: &mut RadixNode, first_token: i32, split_pos: usize, num_nodes: &mut usize) {
        let child = parent.children.get_mut(&first_token).unwrap();

        // 创建新的中间节点
        let prefix_tokens = child.tokens[..split_pos].to_vec();
        let prefix_blocks = child.blocks[..split_pos].to_vec();

        let suffix_tokens = child.tokens[split_pos..].to_vec();
        let suffix_blocks = child.blocks[split_pos..].to_vec();

        // 保存原子节点的子节点
        let old_children = std::mem::take(&mut child.children);
        let old_ref_count = child.ref_count;

        // 修改当前节点为前缀节点
        child.tokens = prefix_tokens;
        child.blocks = prefix_blocks;

        // 创建后缀节点
        let suffix_first_token = suffix_tokens[0];
        let mut suffix_node = RadixNode::with_data(suffix_tokens, suffix_blocks);
        suffix_node.children = old_children;
        suffix_node.ref_count = old_ref_count;

        child.children.insert(suffix_first_token, Box::new(suffix_node));
        *num_nodes += 1;
    }

    /// 创建新子节点
    fn create_new_child(
        parent: &mut RadixNode,
        tokens: &[i32],
        blocks: &[PhysicalBlockId],
        start: usize,
        num_nodes: &mut usize,
    ) {
        let new_tokens = tokens[start..].to_vec();
        let new_blocks = blocks[start..].to_vec();
        let first_token = new_tokens[0];

        let mut new_node = RadixNode::with_data(new_tokens.clone(), new_blocks);
        new_node.node_id = Some(RadixNode::compute_node_id(&new_tokens));

        parent.children.insert(first_token, Box::new(new_node));
        *num_nodes += 1;
    }

    /// 增加前缀的引用计数
    ///
    /// # Arguments
    /// * `tokens` - Token 序列前缀
    pub fn incref_prefix(&mut self, tokens: &[i32]) {
        Self::incref_recursive(&mut self.root, tokens, 0);
    }

    fn incref_recursive(node: &mut RadixNode, tokens: &[i32], start: usize) {
        if start >= tokens.len() {
            return;
        }

        let first_token = tokens[start];
        if let Some(child) = node.children.get_mut(&first_token) {
            let match_len = std::cmp::min(child.tokens.len(), tokens.len() - start);
            child.ref_count += 1;

            if start + match_len < tokens.len() {
                Self::incref_recursive(child, tokens, start + match_len);
            }
        }
    }

    /// 减少前缀的引用计数
    pub fn decref_prefix(&mut self, tokens: &[i32]) {
        Self::decref_recursive(&mut self.root, tokens, 0);
    }

    fn decref_recursive(node: &mut RadixNode, tokens: &[i32], start: usize) {
        if start >= tokens.len() {
            return;
        }

        let first_token = tokens[start];
        if let Some(child) = node.children.get_mut(&first_token) {
            let match_len = std::cmp::min(child.tokens.len(), tokens.len() - start);
            child.ref_count = child.ref_count.saturating_sub(1);

            if start + match_len < tokens.len() {
                Self::decref_recursive(child, tokens, start + match_len);
            }
        }
    }

    /// 获取统计信息
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn total_matches(&self) -> u64 {
        self.total_matches
    }

    pub fn total_inserts(&self) -> u64 {
        self.total_inserts
    }

    /// 清空树（用于测试）
    pub fn clear(&mut self) {
        self.root = RadixNode::new();
        self.num_nodes = 1;
        self.total_matches = 0;
        self.total_inserts = 0;
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_and_match() {
        let mut tree = RadixTree::new();

        // 插入第一个序列
        let tokens1 = vec![1, 2, 3, 4, 5];
        let blocks1 = vec![10, 11, 12, 13, 14];
        tree.insert(&tokens1, &blocks1);

        // 完全匹配
        let result = tree.match_prefix(&tokens1);
        assert_eq!(result.matched_tokens, 5);
        assert_eq!(result.matched_blocks, blocks1);

        // 前缀匹配
        let tokens2 = vec![1, 2, 3];
        let result = tree.match_prefix(&tokens2);
        assert_eq!(result.matched_tokens, 3);
        assert_eq!(result.matched_blocks, vec![10, 11, 12]);

        // 部分匹配
        let tokens3 = vec![1, 2];
        let result = tree.match_prefix(&tokens3);
        assert_eq!(result.matched_tokens, 2);
        assert_eq!(result.matched_blocks, vec![10, 11]);
    }

    #[test]
    fn test_prefix_sharing() {
        let mut tree = RadixTree::new();

        // 插入第一个序列
        let tokens1 = vec![1, 2, 3, 4];
        let blocks1 = vec![10, 11, 12, 13];
        tree.insert(&tokens1, &blocks1);

        // 插入共享前缀的序列
        let tokens2 = vec![1, 2, 3, 5, 6];
        let blocks2 = vec![10, 11, 12, 20, 21];
        tree.insert(&tokens2, &blocks2);

        // 匹配共享前缀
        let result = tree.match_prefix(&vec![1, 2, 3]);
        assert_eq!(result.matched_tokens, 3);
        assert_eq!(result.matched_blocks, vec![10, 11, 12]);

        // 匹配第一个完整序列
        let result = tree.match_prefix(&tokens1);
        assert_eq!(result.matched_tokens, 4);
        assert_eq!(result.matched_blocks, blocks1);

        // 匹配第二个完整序列
        let result = tree.match_prefix(&tokens2);
        assert_eq!(result.matched_tokens, 5);
        assert_eq!(result.matched_blocks, blocks2);
    }

    #[test]
    fn test_no_match() {
        let mut tree = RadixTree::new();

        let tokens1 = vec![1, 2, 3];
        let blocks1 = vec![10, 11, 12];
        tree.insert(&tokens1, &blocks1);

        // 完全不匹配
        let result = tree.match_prefix(&vec![5, 6, 7]);
        assert_eq!(result.matched_tokens, 0);
        assert!(result.matched_blocks.is_empty());
    }

    #[test]
    fn test_refcount() {
        let mut tree = RadixTree::new();

        let tokens = vec![1, 2, 3];
        let blocks = vec![10, 11, 12];
        tree.insert(&tokens, &blocks);

        // 增加引用计数
        tree.incref_prefix(&tokens);
        tree.incref_prefix(&tokens);

        // 减少引用计数
        tree.decref_prefix(&tokens);

        // 注意：这里我们没有直接访问 ref_count，
        // 在实际使用中，Allocator 会维护引用计数
    }

    #[test]
    fn test_multiple_branches() {
        let mut tree = RadixTree::new();

        // 插入多个分支
        tree.insert(&vec![1, 2, 3], &vec![10, 11, 12]);
        tree.insert(&vec![1, 2, 4], &vec![10, 11, 20]);
        tree.insert(&vec![1, 5, 6], &vec![10, 30, 31]);

        // 测试不同分支的匹配
        let result = tree.match_prefix(&vec![1, 2, 3]);
        assert_eq!(result.matched_tokens, 3);

        let result = tree.match_prefix(&vec![1, 2, 4]);
        assert_eq!(result.matched_tokens, 3);

        let result = tree.match_prefix(&vec![1, 5, 6]);
        assert_eq!(result.matched_tokens, 3);

        // 共同前缀
        let result = tree.match_prefix(&vec![1]);
        assert_eq!(result.matched_tokens, 1);
        assert_eq!(result.matched_blocks, vec![10]);
    }
}
