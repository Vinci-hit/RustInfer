//! Slot-based (non-paged) KV manager.
//!
//! Each sequence occupies a dedicated slot. The worker manages KV tensors internally.
//! This is equivalent to the current `SlotPool` behavior.

use crate::cache::kv_manager::{KvAllocation, KvManager};
use crate::error::{Result, SchedulerError};

/// Non-paged KV manager: assigns integer slot IDs to sequences.
///
/// Each slot represents a full KV cache "lane" on the worker.
/// The worker is responsible for resizing/managing the actual GPU tensors.
pub struct SlotKvManager {
    /// Bitmap: true = occupied.
    slots: Vec<bool>,
    /// Max sequence length per slot (for `available_tokens` reporting).
    max_seq_len: usize,
}

impl SlotKvManager {
    /// Create a new slot manager with the given number of slots.
    pub fn new(max_slots: usize, max_seq_len: usize) -> Self {
        Self {
            slots: vec![false; max_slots],
            max_seq_len,
        }
    }

    /// Number of free slots.
    pub fn num_free(&self) -> usize {
        self.slots.iter().filter(|&&s| !s).count()
    }

    /// Total number of slots.
    pub fn total_slots(&self) -> usize {
        self.slots.len()
    }

    /// Allocate one slot, returning its id.
    fn alloc_one(&mut self) -> Option<u32> {
        for (i, occupied) in self.slots.iter_mut().enumerate() {
            if !*occupied {
                *occupied = true;
                return Some(i as u32);
            }
        }
        None
    }

    /// Free a slot by id.
    fn free_one(&mut self, slot: u32) {
        let idx = slot as usize;
        if idx < self.slots.len() {
            self.slots[idx] = false;
        }
    }
}

impl KvManager for SlotKvManager {
    fn allocate(&mut self, _num_tokens: usize) -> Result<KvAllocation> {
        match self.alloc_one() {
            Some(slot_id) => Ok(KvAllocation::Slot(slot_id)),
            None => Err(SchedulerError::CacheExhausted {
                needed: 1,
                available: 0,
            }),
        }
    }

    fn extend(&mut self, _alloc: &mut KvAllocation, _additional_tokens: usize) -> Result<()> {
        // Slot mode: worker handles KV tensor growth internally. No-op on scheduler side.
        Ok(())
    }

    fn free(&mut self, alloc: KvAllocation) {
        match alloc {
            KvAllocation::Slot(slot_id) => self.free_one(slot_id),
            KvAllocation::Blocks(_) => {
                tracing::error!("SlotKvManager::free called with Blocks variant — bug");
            }
        }
    }

    fn available_tokens(&self) -> usize {
        self.num_free() * self.max_seq_len
    }

    fn total_capacity_tokens(&self) -> usize {
        self.total_slots() * self.max_seq_len
    }

    fn mode_name(&self) -> &'static str {
        "slot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free() {
        let mut mgr = SlotKvManager::new(4, 2048);
        assert_eq!(mgr.num_free(), 4);

        let a0 = mgr.allocate(100).unwrap();
        assert!(a0.is_slot());
        assert_eq!(a0.as_slot(), 0);
        assert_eq!(mgr.num_free(), 3);

        let a1 = mgr.allocate(100).unwrap();
        assert_eq!(a1.as_slot(), 1);

        mgr.free(a0);
        assert_eq!(mgr.num_free(), 3);

        // Re-alloc gives slot 0 back (first-fit).
        let a0_again = mgr.allocate(100).unwrap();
        assert_eq!(a0_again.as_slot(), 0);
    }

    #[test]
    fn exhaustion_returns_error() {
        let mut mgr = SlotKvManager::new(2, 2048);
        let _a0 = mgr.allocate(100).unwrap();
        let _a1 = mgr.allocate(100).unwrap();

        let result = mgr.allocate(100);
        assert!(result.is_err());
    }

    #[test]
    fn extend_is_noop() {
        let mut mgr = SlotKvManager::new(4, 2048);
        let mut alloc = mgr.allocate(100).unwrap();
        // Extending in slot mode is always OK (no-op).
        assert!(mgr.extend(&mut alloc, 500).is_ok());
    }
}
