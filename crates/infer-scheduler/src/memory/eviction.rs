//! LRU Eviction for RadixTree Prefix Cache
//!
//! 当内存不足时，RadixTree 可以驱逐最少使用的前缀节点来释放物理块。
//!
//! # 设计
//!
//! ```text
//! EvictionList (LRU 双向链表)
//!   HEAD (oldest)                         TAIL (newest)
//!     │                                         │
//!     ▼                                         ▼
//!   Node A  ←→  Node B  ←→  Node C  ←→  Node D
//!   (evict)                            (keep)
//! ```
//!
//! # 使用
//!
//! ```rust,ignore
//! let mut eviction_list = EvictionList::new();
//!
//! // 添加可驱逐的节点
//! eviction_list.push_back(node_id, num_tokens);
//!
//! // 访问节点（移到队尾）
//! eviction_list.touch(node_id);
//!
//! // 驱逐最老的节点
//! if let Some((node_id, num_tokens)) = eviction_list.pop_front() {
//!     // 释放该节点占用的物理块
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// 可驱逐节点的 ID（通常是前缀的 hash 或唯一标识）
pub type NodeId = u64;

/// LRU 节点信息
#[derive(Debug, Clone)]
struct LruNode {
    /// 节点 ID
    node_id: NodeId,

    /// 该节点占用的 token 数量
    num_tokens: usize,

    /// 最后访问时间
    last_access: Instant,

    /// 访问次数
    access_count: u64,
}

/// LRU 驱逐列表
///
/// 使用 VecDeque 实现简化的 LRU，牺牲部分性能换取安全性和简洁性。
/// 对于生产环境，可以参考 kv_cache/lru_list.rs 使用原始指针实现 O(1) 操作。
#[derive(Debug)]
pub struct EvictionList {
    /// LRU 队列（队首是最老的，队尾是最新的）
    queue: VecDeque<LruNode>,

    /// 快速查找 NodeId 在队列中的位置
    index_map: HashMap<NodeId, usize>,

    /// 可驱逐节点的总 token 数
    total_tokens: usize,

    /// 统计：驱逐次数
    eviction_count: u64,

    /// 统计：命中次数（touch 调用）
    hit_count: u64,
}

impl EvictionList {
    /// 创建新的驱逐列表
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            index_map: HashMap::new(),
            total_tokens: 0,
            eviction_count: 0,
            hit_count: 0,
        }
    }

    /// 获取可驱逐节点数量
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// 获取可驱逐的总 token 数
    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// 检查节点是否在驱逐列表中
    #[inline]
    pub fn contains(&self, node_id: NodeId) -> bool {
        self.index_map.contains_key(&node_id)
    }

    /// 添加节点到队尾（最新）
    ///
    /// # Arguments
    /// * `node_id` - 节点 ID
    /// * `num_tokens` - 节点包含的 token 数量
    pub fn push_back(&mut self, node_id: NodeId, num_tokens: usize) {
        // 如果已存在，先移除
        if self.contains(node_id) {
            self.remove(node_id);
        }

        let node = LruNode {
            node_id,
            num_tokens,
            last_access: Instant::now(),
            access_count: 1,
        };

        let new_index = self.queue.len();
        self.queue.push_back(node);
        self.index_map.insert(node_id, new_index);
        self.total_tokens += num_tokens;
    }

    /// 驱逐队首（最老的）节点
    ///
    /// # Returns
    /// (node_id, num_tokens, age)
    pub fn pop_front(&mut self) -> Option<(NodeId, usize, Duration)> {
        let node = self.queue.pop_front()?;
        let age = node.last_access.elapsed();

        // 更新索引映射
        self.index_map.remove(&node.node_id);
        self.rebuild_index_map();

        self.total_tokens = self.total_tokens.saturating_sub(node.num_tokens);
        self.eviction_count += 1;

        Some((node.node_id, node.num_tokens, age))
    }

    /// 标记节点为最近访问（移到队尾）
    ///
    /// # Arguments
    /// * `node_id` - 要访问的节点 ID
    pub fn touch(&mut self, node_id: NodeId) -> bool {
        let Some(&index) = self.index_map.get(&node_id) else {
            return false;
        };

        // 移除并重新添加到队尾
        if let Some(mut node) = self.queue.remove(index) {
            node.last_access = Instant::now();
            node.access_count += 1;

            let num_tokens = node.num_tokens;
            self.index_map.remove(&node_id);
            self.total_tokens = self.total_tokens.saturating_sub(num_tokens);

            self.queue.push_back(node);
            self.index_map.insert(node_id, self.queue.len() - 1);
            self.total_tokens += num_tokens;

            self.hit_count += 1;
            self.rebuild_index_map();
            true
        } else {
            false
        }
    }

    /// 移除特定节点
    ///
    /// # Arguments
    /// * `node_id` - 要移除的节点 ID
    pub fn remove(&mut self, node_id: NodeId) -> bool {
        let Some(&index) = self.index_map.get(&node_id) else {
            return false;
        };

        if let Some(node) = self.queue.remove(index) {
            self.index_map.remove(&node_id);
            self.total_tokens = self.total_tokens.saturating_sub(node.num_tokens);
            self.rebuild_index_map();
            true
        } else {
            false
        }
    }

    /// 批量驱逐节点直到释放足够的 tokens
    ///
    /// # Arguments
    /// * `target_tokens` - 需要释放的 token 数量
    ///
    /// # Returns
    /// 被驱逐的节点列表 (node_id, num_tokens)
    pub fn evict_until(&mut self, target_tokens: usize) -> Vec<(NodeId, usize)> {
        let mut evicted = Vec::new();
        let mut freed_tokens = 0;

        while freed_tokens < target_tokens && !self.is_empty() {
            if let Some((node_id, num_tokens, _age)) = self.pop_front() {
                freed_tokens += num_tokens;
                evicted.push((node_id, num_tokens));
            } else {
                break;
            }
        }

        evicted
    }

    /// 获取队首节点（最老的）但不移除
    pub fn peek_front(&self) -> Option<(NodeId, usize, Duration)> {
        self.queue.front().map(|node| {
            let age = node.last_access.elapsed();
            (node.node_id, node.num_tokens, age)
        })
    }

    /// 清空列表
    pub fn clear(&mut self) {
        self.queue.clear();
        self.index_map.clear();
        self.total_tokens = 0;
    }

    /// 获取统计信息
    pub fn stats(&self) -> EvictionStats {
        EvictionStats {
            evictable_nodes: self.len(),
            evictable_tokens: self.total_tokens,
            eviction_count: self.eviction_count,
            hit_count: self.hit_count,
        }
    }

    /// 重建索引映射（在队列修改后调用）
    fn rebuild_index_map(&mut self) {
        self.index_map.clear();
        for (index, node) in self.queue.iter().enumerate() {
            self.index_map.insert(node.node_id, index);
        }
    }
}

impl Default for EvictionList {
    fn default() -> Self {
        Self::new()
    }
}

/// 驱逐统计信息
#[derive(Debug, Clone, Default)]
pub struct EvictionStats {
    /// 可驱逐的节点数
    pub evictable_nodes: usize,

    /// 可驱逐的 token 数
    pub evictable_tokens: usize,

    /// 累计驱逐次数
    pub eviction_count: u64,

    /// 累计命中次数（touch 调用）
    pub hit_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(2, 20);
        list.push_back(3, 30);

        assert_eq!(list.len(), 3);
        assert_eq!(list.total_tokens(), 60);

        // 驱逐顺序应该是 1, 2, 3
        let (node_id, tokens, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 1);
        assert_eq!(tokens, 10);

        let (node_id, tokens, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 2);
        assert_eq!(tokens, 20);

        assert_eq!(list.total_tokens(), 30);
    }

    #[test]
    fn test_touch() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(2, 20);
        list.push_back(3, 30);

        // 访问节点 1，将其移到队尾
        assert!(list.touch(1));

        // 现在驱逐顺序应该是 2, 3, 1
        let (node_id, _, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 2);

        let (node_id, _, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 3);

        let (node_id, _, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 1);
    }

    #[test]
    fn test_remove() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(2, 20);
        list.push_back(3, 30);

        // 移除中间节点
        assert!(list.remove(2));
        assert_eq!(list.len(), 2);
        assert_eq!(list.total_tokens(), 40);

        // 驱逐顺序应该是 1, 3
        let (node_id, _, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 1);

        let (node_id, _, _) = list.pop_front().unwrap();
        assert_eq!(node_id, 3);
    }

    #[test]
    fn test_evict_until() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(2, 20);
        list.push_back(3, 30);
        list.push_back(4, 40);

        // 驱逐直到释放至少 50 tokens
        let evicted = list.evict_until(50);

        // 应该驱逐 node 1 (10) 和 node 2 (20) 和 node 3 (30) = 60 tokens
        assert_eq!(evicted.len(), 3);
        assert_eq!(evicted[0], (1, 10));
        assert_eq!(evicted[1], (2, 20));
        assert_eq!(evicted[2], (3, 30));

        assert_eq!(list.len(), 1);
        assert_eq!(list.total_tokens(), 40);
    }

    #[test]
    fn test_contains() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(2, 20);

        assert!(list.contains(1));
        assert!(list.contains(2));
        assert!(!list.contains(3));

        list.remove(1);
        assert!(!list.contains(1));
    }

    #[test]
    fn test_duplicate_push() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(1, 20); // 重复添加，应该更新

        assert_eq!(list.len(), 1);
        assert_eq!(list.total_tokens(), 20); // 应该使用新值
    }

    #[test]
    fn test_peek() {
        let mut list = EvictionList::new();

        assert!(list.peek_front().is_none());

        list.push_back(1, 10);
        list.push_back(2, 20);

        let (node_id, tokens, _) = list.peek_front().unwrap();
        assert_eq!(node_id, 1);
        assert_eq!(tokens, 10);

        // peek 不应该移除节点
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut list = EvictionList::new();

        list.push_back(1, 10);
        list.push_back(2, 20);
        list.touch(1);
        list.pop_front();

        let stats = list.stats();
        assert_eq!(stats.evictable_nodes, 1);
        assert_eq!(stats.evictable_tokens, 10);
        assert_eq!(stats.eviction_count, 1);
        assert_eq!(stats.hit_count, 1);
    }
}
