//! Eviction policy for cached blocks (stub).
//!
//! When fully implemented, supports LRU and LFU eviction strategies.

use crate::cache::traits::PhysicalBlockId;

/// Eviction policy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used.
    Lru,
    /// Least Frequently Used (stub).
    Lfu,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::Lru
    }
}

/// Evictor tracks block access patterns and determines eviction order.
///
/// **Current status:** LRU is implemented in PagedBlockAllocator's eviction_order.
/// This module provides the policy abstraction for future expansion.
pub struct Evictor {
    policy: EvictionPolicy,
}

impl Evictor {
    pub fn new(policy: EvictionPolicy) -> Self {
        Self { policy }
    }

    /// Record that a block was accessed (touch for LRU/LFU).
    pub fn touch(&mut self, _block: PhysicalBlockId) {
        // Integrated into PagedBlockAllocator for now.
    }

    /// Get the eviction policy.
    pub fn policy(&self) -> EvictionPolicy {
        self.policy
    }
}

impl Default for Evictor {
    fn default() -> Self {
        Self::new(EvictionPolicy::Lru)
    }
}
