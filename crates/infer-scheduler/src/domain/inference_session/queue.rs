//! Multi-priority waiting queue.
//!
//! Orders requests by priority (higher first), then by arrival time (FCFS within tier).

use std::collections::VecDeque;

use crate::domain::inference_session::lifecycle::{RequestId, InferenceSession, Queued};

/// Multi-priority waiting queue.
///
/// Internally maintains a VecDeque sorted by (priority DESC, arrival_time ASC).
/// Preempted requests are re-inserted at the front of their priority tier.
pub struct WaitingQueue {
    /// Sequences ordered by priority (highest first), then FIFO within same priority.
    queue: VecDeque<InferenceSession<Queued>>,
}

impl WaitingQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Number of waiting requests.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Push a new request (appended respecting priority order).
    pub fn push(&mut self, seq: InferenceSession<Queued>) {
        let priority = seq.meta.priority;
        // Find insertion point: after all sequences with >= priority.
        let pos = self.queue.iter().position(|s| s.meta.priority < priority);
        match pos {
            Some(idx) => self.queue.insert(idx, seq),
            None => self.queue.push_back(seq),
        }
    }

    /// Push a preempted request to the front of its priority tier (priority boost).
    pub fn push_front(&mut self, seq: InferenceSession<Queued>) {
        let priority = seq.meta.priority;
        // Insert before the first sequence with same or lower priority.
        let pos = self.queue.iter().position(|s| s.meta.priority <= priority);
        match pos {
            Some(idx) => self.queue.insert(idx, seq),
            None => self.queue.push_back(seq),
        }
    }

    /// Pop the highest-priority request from the front.
    pub fn pop_front(&mut self) -> Option<InferenceSession<Queued>> {
        self.queue.pop_front()
    }

    /// Peek at the front request without removing.
    pub fn front(&self) -> Option<&InferenceSession<Queued>> {
        self.queue.front()
    }

    /// Iterate over all waiting sequences (highest priority first).
    pub fn iter(&self) -> impl Iterator<Item = &InferenceSession<Queued>> {
        self.queue.iter()
    }

    /// Remove a specific request by ID (e.g., for cancellation).
    pub fn remove(&mut self, id: &RequestId) -> Option<InferenceSession<Queued>> {
        let pos = self.queue.iter().position(|s| s.meta.id == *id)?;
        self.queue.remove(pos)
    }

    /// Total tokens across all waiting requests (sum of prompt lengths).
    pub fn total_tokens(&self) -> usize {
        self.queue.iter().map(|s| s.meta.input_ids.len()).sum()
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
    use crate::domain::inference_session::lifecycle::{RequestMeta, SamplingParams, Priority};
    use std::sync::Arc;
    use std::time::Instant;

    fn make_seq(id: &str, priority: i32) -> InferenceSession<Queued> {
        let meta = Arc::new(RequestMeta {
            id: RequestId::new_v4(), external_id: id.to_string(),
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
}
