//! Multi-priority waiting queue.
//!
//! Orders requests by priority (higher first), then by arrival time (FCFS within tier).

use std::collections::VecDeque;

use crate::domain::inference_session::lifecycle::{InferenceSession, Queued, RequestId};

/// Multi-priority waiting queue.
///
/// Internally maintains a VecDeque sorted by (priority DESC, arrival_time ASC).
/// Preempted requests are re-inserted at the front of their priority tier.
///
/// Removal is *lazy* (#5): [`remove`](Self::remove) takes the session out of its
/// slot but leaves a `None` tombstone in place instead of shifting the tail —
/// `VecDeque::remove` is O(n) memmove, so under high-QPS cancellation that cost
/// dominated. Tombstones are skipped by every reader and reclaimed in bulk by
/// [`compact`](Self::compact), which runs opportunistically whenever holes pile
/// up. Live element ordering is never disturbed, so FIFO-within-priority and the
/// preempt push-front boost are preserved exactly.
pub struct WaitingQueue {
    /// Slots ordered by priority (highest first), then FIFO within same priority.
    /// `None` entries are tombstones left by lazy removal, skipped by all readers.
    queue: VecDeque<Option<InferenceSession<Queued>>>,
    /// Count of live (`Some`) slots — keeps `len`/`is_empty` O(1) despite holes.
    live: usize,
}

impl WaitingQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            live: 0,
        }
    }

    /// Number of waiting requests (live slots only, tombstones excluded).
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether the queue is empty (no live slots).
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Push a new request (appended respecting priority order).
    pub fn push(&mut self, seq: InferenceSession<Queued>) {
        let priority = seq.meta.priority;
        // Find insertion point: before the first live request with lower priority.
        let pos = self
            .queue
            .iter()
            .position(|slot| slot.as_ref().is_some_and(|s| s.meta.priority < priority));
        match pos {
            Some(idx) => self.queue.insert(idx, Some(seq)),
            None => self.queue.push_back(Some(seq)),
        }
        self.live += 1;
        self.maybe_compact();
    }

    /// Push a preempted request to the front of its priority tier (priority boost).
    pub fn push_front(&mut self, seq: InferenceSession<Queued>) {
        let priority = seq.meta.priority;
        // Insert before the first live request with same or lower priority.
        let pos = self
            .queue
            .iter()
            .position(|slot| slot.as_ref().is_some_and(|s| s.meta.priority <= priority));
        match pos {
            Some(idx) => self.queue.insert(idx, Some(seq)),
            None => self.queue.push_back(Some(seq)),
        }
        self.live += 1;
        self.maybe_compact();
    }

    /// Pop the highest-priority request from the front, discarding any leading
    /// tombstones along the way.
    pub fn pop_front(&mut self) -> Option<InferenceSession<Queued>> {
        while let Some(slot) = self.queue.pop_front() {
            if let Some(seq) = slot {
                self.live -= 1;
                return Some(seq);
            }
        }
        None
    }

    /// Peek at the front live request without removing.
    pub fn front(&self) -> Option<&InferenceSession<Queued>> {
        self.queue.iter().find_map(|slot| slot.as_ref())
    }

    /// Iterate over all live waiting sequences (highest priority first).
    pub fn iter(&self) -> impl Iterator<Item = &InferenceSession<Queued>> {
        self.queue.iter().filter_map(|slot| slot.as_ref())
    }

    /// Remove a specific request by ID (e.g., for cancellation).
    ///
    /// Lazy: leaves a `None` tombstone instead of shifting the tail (O(1) after
    /// the scan vs `VecDeque::remove`'s O(n) memmove). Tombstones are reclaimed
    /// by [`compact`](Self::compact).
    pub fn remove(&mut self, id: &RequestId) -> Option<InferenceSession<Queued>> {
        let pos = self
            .queue
            .iter()
            .position(|slot| slot.as_ref().is_some_and(|s| s.meta.id == *id))?;
        let seq = self.queue[pos].take();
        if seq.is_some() {
            self.live -= 1;
            self.maybe_compact();
        }
        seq
    }

    /// Total tokens across all live waiting requests (sum of prompt lengths).
    pub fn total_tokens(&self) -> usize {
        self.iter().map(|s| s.meta.input_ids.len()).sum()
    }

    /// Drop tombstones when they outnumber live entries (amortized O(1) per
    /// removal). The threshold keeps the backing deque from growing without
    /// bound under churn while avoiding a compaction on every single removal.
    fn maybe_compact(&mut self) {
        if self.queue.len() > 2 * self.live + 8 {
            self.compact();
        }
    }

    /// Physically drop all tombstones, preserving live order.
    fn compact(&mut self) {
        self.queue.retain(|slot| slot.is_some());
        debug_assert_eq!(self.queue.len(), self.live);
    }
}

impl Default for WaitingQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::{Priority, RequestMeta, SamplingParams};
    use std::sync::Arc;
    use std::time::Instant;

    fn make_seq(id: &str, priority: i32) -> InferenceSession<Queued> {
        let meta = Arc::new(RequestMeta {
            id: RequestId::new_v4(),
            external_id: id.to_string(),
            sequence_id: crate::domain::inference_session::lifecycle::SequenceId(1),
            input_ids: vec![1, 2, 3],
            max_tokens: 10,
            sampling: SamplingParams::default(),
            priority: Priority(priority),
            stream: false,
            stop_sequences: vec![],
            ignore_eos: false,
            diffusion: None,
            arrival_time: Instant::now(),
        });
        InferenceSession::new(meta, RequestHandle::noop())
    }

    #[test]
    fn fifo_same_priority() {
        let mut q = WaitingQueue::new();
        q.push(make_seq("a", 0));
        q.push(make_seq("b", 0));
        q.push(make_seq("c", 0));

        assert_eq!(q.pop_front().unwrap().meta.external_id, "a");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "b");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "c");
    }

    #[test]
    fn higher_priority_first() {
        let mut q = WaitingQueue::new();
        q.push(make_seq("low", -1));
        q.push(make_seq("normal", 0));
        q.push(make_seq("high", 1));

        assert_eq!(q.pop_front().unwrap().meta.external_id, "high");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "normal");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "low");
    }

    #[test]
    fn push_front_for_preempted() {
        let mut q = WaitingQueue::new();
        q.push(make_seq("a", 0));
        q.push(make_seq("b", 0));

        // Preempted request pushed to front of same priority.
        q.push_front(make_seq("preempted", 0));

        assert_eq!(q.pop_front().unwrap().meta.external_id, "preempted");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "a");
    }

    #[test]
    fn lazy_remove_skips_tombstone_and_returns_session() {
        let mut q = WaitingQueue::new();
        let a = make_seq("a", 0);
        let b = make_seq("b", 0);
        let c = make_seq("c", 0);
        let b_id = b.meta.id.clone();
        q.push(a);
        q.push(b);
        q.push(c);

        // Remove the middle element: returns the real session, len drops.
        let removed = q.remove(&b_id).expect("b present");
        assert_eq!(removed.meta.external_id, "b");
        assert_eq!(q.len(), 2);
        assert!(!q.is_empty());

        // Order of survivors preserved; tombstone is skipped on pop.
        assert_eq!(q.pop_front().unwrap().meta.external_id, "a");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "c");
        assert!(q.pop_front().is_none());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn remove_missing_id_is_none() {
        let mut q = WaitingQueue::new();
        q.push(make_seq("a", 0));
        let absent = RequestId::new_v4();
        assert!(q.remove(&absent).is_none());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn front_and_total_tokens_skip_tombstones() {
        let mut q = WaitingQueue::new();
        let a = make_seq("a", 0);
        let a_id = a.meta.id.clone();
        q.push(a);
        q.push(make_seq("b", 0));

        q.remove(&a_id);
        // Front now sees the survivor, not the hole.
        assert_eq!(q.front().unwrap().meta.external_id, "b");
        // Each make_seq has 3 input ids; only the live one counts.
        assert_eq!(q.total_tokens(), 3);
    }

    #[test]
    fn churn_compacts_and_preserves_order() {
        let mut q = WaitingQueue::new();
        // Insert many, remove most, ensure tombstones get reclaimed and the
        // live ordering survives compaction.
        let mut ids = Vec::new();
        for i in 0..64 {
            let s = make_seq(&format!("s{i}"), 0);
            ids.push((i, s.meta.id.clone()));
            q.push(s);
        }
        // Remove every even index.
        for (i, id) in &ids {
            if i % 2 == 0 {
                assert!(q.remove(id).is_some());
            }
        }
        assert_eq!(q.len(), 32);
        // Surviving odd-indexed sequences come out in original order.
        let mut expected = 1;
        while let Some(seq) = q.pop_front() {
            assert_eq!(seq.meta.external_id, format!("s{expected}"));
            expected += 2;
        }
        assert_eq!(expected, 65);
    }

    #[test]
    fn priority_order_survives_removal() {
        let mut q = WaitingQueue::new();
        let normal = make_seq("normal", 0);
        let normal_id = normal.meta.id.clone();
        q.push(make_seq("low", -1));
        q.push(normal);
        q.push(make_seq("high", 1));

        q.remove(&normal_id);
        assert_eq!(q.pop_front().unwrap().meta.external_id, "high");
        assert_eq!(q.pop_front().unwrap().meta.external_id, "low");
        assert!(q.pop_front().is_none());
    }
}
