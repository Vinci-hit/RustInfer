//! Request-owned recurrent slots. Row order may change independently of ownership.
use super::*;
use crate::domain::cache::{LinearBatch, LinearLayerState};
use std::collections::{HashMap, HashSet};

pub(super) struct RecurrentState<T: Dtype, D: LlmBackend> {
    pub layers: Vec<LinearLayerState<T, D>>,
    pub batch: Option<LinearBatch<D>>,
    slots: HashMap<u64, (usize, i32)>,
    free: Vec<usize>,
    step: Vec<(u64, i32)>,
    /// Ordinary decode issues advance logical history before their async
    /// output is collected. Retain their owners until collection so a late
    /// failure cannot leave reusable, partially executed recurrent history.
    decode_in_flight: Vec<u64>,
    capacity: usize,
}

impl<T: Dtype, D: LlmBackend> RecurrentState<T, D> {
    pub(super) fn snapshot(
        &self,
        req: &StepRequest,
    ) -> OpResult<crate::domain::cache::LinearSnapshot<T, D>> {
        let slots = req
            .seqs
            .iter()
            .map(|s| self.slots[&s.sequence_id].0)
            .collect::<Vec<_>>();
        crate::domain::cache::LinearSnapshot::capture(&self.layers, &slots)
    }

    pub(super) fn retain_step(&mut self, req: &StepRequest) {
        self.step = req
            .seqs
            .iter()
            .map(|s| (s.sequence_id, s.kv_len_after))
            .collect();
    }

    pub(super) fn has_decode_in_flight(&self) -> bool {
        !self.decode_in_flight.is_empty()
    }
    pub fn has_live_sequences(&self) -> bool {
        !self.slots.is_empty()
    }

    pub fn new(
        layout: &crate::domain::cache::CacheLayout,
        capacity: usize,
        device: &D,
    ) -> OpResult<Self> {
        Ok(Self {
            layers: layout
                .linear_dims()
                .iter()
                .map(|&d| LinearLayerState::new(d, capacity, device))
                .collect::<OpResult<_>>()?,
            batch: None,
            slots: HashMap::new(),
            free: (0..capacity).rev().collect(),
            step: Vec::new(),
            decode_in_flight: Vec::with_capacity(capacity),
            capacity,
        })
    }
    fn release(&mut self, id: u64) {
        if let Some((slot, _)) = self.slots.remove(&id) {
            self.free.push(slot);
        }
    }
    pub fn complete(&mut self, success: bool) {
        for (id, len) in std::mem::take(&mut self.step) {
            if success {
                if let Some(entry) = self.slots.get_mut(&id) {
                    entry.1 = len;
                }
            } else {
                // A failed kernel chain may have changed some layers: invalidate
                // the entire request rather than permit a corrupted continuation.
                self.release(id);
            }
        }
    }

    pub fn record_decode_issue(&mut self, req: &StepRequest) {
        self.decode_in_flight.clear();
        self.decode_in_flight
            .extend(req.seqs.iter().map(|seq| seq.sequence_id));
    }

    pub fn collect_decode(&mut self, success: bool) {
        while let Some(id) = self.decode_in_flight.pop() {
            if !success {
                self.release(id);
            }
        }
    }
}

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> Runtime<T, D, M> {
    pub fn has_recurrent_state(&self) -> bool {
        self.recurrent.is_some()
    }

    /// Resolve an already prepared ordinary step at its execution boundary.
    /// Synchronous steps call this after sampling; ABC calls it after the
    /// entire issue has been enqueued. The latter advances logical history
    /// for ordered, overlapping execution, not a claim of GPU completion.
    /// Capture/forward helpers never commit history themselves.
    pub(super) fn finish_recurrent_step<R>(&mut self, result: OpResult<R>) -> OpResult<R> {
        if let Some(state) = &mut self.recurrent {
            state.complete(result.is_ok());
        }
        result
    }

    /// Call after completion, cancellation, or preemption. Reuse zeroes the slot.
    pub fn release_sequence(&mut self, id: u64) {
        self.visual.release(id);
        if let Some(state) = &mut self.recurrent {
            state.release(id);
        }
    }

    /// Serving reconciles ownership with both active decode and partial prefill requests.
    pub fn retain_sequences(&mut self, live: impl IntoIterator<Item = u64>) {
        let live: HashSet<_> = live.into_iter().collect();
        self.visual.retain(&live);
        if let Some(state) = &mut self.recurrent {
            let expired: Vec<_> = state
                .slots
                .keys()
                .filter(|id| !live.contains(id))
                .copied()
                .collect();
            for id in expired {
                state.release(id);
            }
        }
    }

    pub(super) fn prepare_recurrent(
        &mut self,
        req: &StepRequest,
        plan: &BatchPlan,
    ) -> OpResult<()> {
        if !req.draft_tokens.is_empty() {
            return Err(OpError::unsupported(
                "recurrent execution",
                "verification outside a snapshot transaction",
            ));
        }
        self.prepare_recurrent_verification(req, plan)
    }

    /// The speculative caller must snapshot before execution and resolve the
    /// retained prefix before completing this prepared history.
    pub(super) fn prepare_recurrent_verification(
        &mut self,
        req: &StepRequest,
        plan: &BatchPlan,
    ) -> OpResult<()> {
        let Some(state) = &mut self.recurrent else {
            return Ok(());
        };
        let mut seen = HashSet::new();
        let mut needed = 0;
        // Validate the entire batch before resetting or assigning any state.
        for seq in &req.seqs {
            if !seen.insert(seq.sequence_id) {
                return Err(OpError::Shape("duplicate recurrent sequence id".into()));
            }
            match state.slots.get(&seq.sequence_id) {
                Some(&(_, len)) if len == seq.kv_write_start => {}
                None if seq.kv_write_start == 0 => needed += 1,
                _ => {
                    return Err(OpError::Shape(format!(
                        "sequence {}: recurrent history does not match write position {}; replay from position 0 after release",
                        seq.sequence_id, seq.kv_write_start
                    )));
                }
            }
        }
        if needed > state.free.len() {
            return Err(OpError::Shape(
                "recurrent request slot capacity exhausted".into(),
            ));
        }
        // Drain previous readers before updating the address-stable metadata
        // and before a newly assigned request clears a recycled state slot.
        self.scope.synchronize()?;
        state.step.clear();
        let result = (|| {
            let mut slots = Vec::with_capacity(req.seqs.len());
            for seq in &req.seqs {
                let slot = match state.slots.get(&seq.sequence_id) {
                    Some(&(slot, _)) => slot,
                    None => {
                        let slot = *state.free.last().unwrap();
                        for layer in &mut state.layers {
                            layer.reset_slot(slot)?;
                        }
                        state.free.pop();
                        state.slots.insert(seq.sequence_id, (slot, 0));
                        slot
                    }
                };
                slots.push(slot as i32);
                state.step.push((seq.sequence_id, seq.kv_len_after));
            }
            match &mut state.batch {
                Some(batch) => batch.update(&slots, &plan.q_lens)?,
                None => {
                    state.batch = Some(LinearBatch::with_capacity(
                        &slots,
                        &plan.q_lens,
                        state.capacity,
                        self.cap_batch,
                        self.scope.device(),
                    )?);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            state.complete(false);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::cache::CacheLayout;
    use crate::domain::plan::StopCriteria;
    use crate::infrastructure::cpu::Cpu;

    fn state_with_history() -> RecurrentState<f32, Cpu> {
        let mut state = RecurrentState::new(&CacheLayout::default(), 3, &Cpu).unwrap();
        state.slots.insert(10, (0, 4));
        state.slots.insert(20, (1, 7));
        state.free = vec![2];
        state
    }

    fn decode_request(id: u64) -> StepRequest {
        StepRequest {
            seqs: vec![SeqStep {
                sequence_id: id,
                input_ids: vec![1],
                positions: vec![4],
                kv_write_start: 4,
                kv_len_after: 5,
                block_table: vec![],
            }],
            sampling: vec![],
            stop: StopCriteria {
                eos_ids: vec![],
                generated_counts: vec![],
                max_tokens: vec![],
                ignore_eos: vec![],
            },
            draft_tokens: vec![],
        }
    }

    #[test]
    fn prepared_history_advances_only_at_explicit_completion() {
        let mut state = state_with_history();
        state.step.push((10, 5));
        assert_eq!(state.slots[&10].1, 4);
        state.complete(true);
        assert_eq!(state.slots[&10].1, 5);
        assert_eq!(state.slots[&20].1, 7);
        // A capture or repeated completion cannot advance the request twice.
        state.complete(true);
        assert_eq!(state.slots[&10].1, 5);
    }

    #[test]
    fn failed_prepared_step_invalidates_only_its_requests() {
        let mut state = state_with_history();
        state.step.push((10, 5));
        state.complete(false);
        assert!(!state.slots.contains_key(&10));
        assert_eq!(state.slots[&20].1, 7);
        assert_eq!(state.free.len(), 2);
        state.complete(false);
        assert_eq!(state.free.len(), 2);
    }

    #[test]
    fn failed_decode_collection_invalidates_overlapped_history() {
        let mut state = state_with_history();
        state.step.push((10, 5));
        state.complete(true);
        state.record_decode_issue(&decode_request(10));
        // Mixed issue can advance the same request before decode collection.
        state.step.extend([(10, 6), (20, 8)]);
        state.complete(true);
        state.collect_decode(false);
        assert!(!state.slots.contains_key(&10));
        assert_eq!(state.slots[&20].1, 8);
        assert!(state.decode_in_flight.is_empty());
        state.collect_decode(false);
        assert_eq!(state.free.len(), 2);
    }

    #[test]
    fn successful_decode_collection_preserves_newer_mixed_progress() {
        let mut state = state_with_history();
        let retained_capacity = state.decode_in_flight.capacity();
        state.step.push((10, 5));
        state.complete(true);
        state.record_decode_issue(&decode_request(10));
        state.step.extend([(10, 6), (20, 8)]);
        state.complete(true);
        state.collect_decode(true);
        assert_eq!(state.slots[&10].1, 6);
        assert_eq!(state.slots[&20].1, 8);
        assert!(state.decode_in_flight.is_empty());
        assert_eq!(state.decode_in_flight.capacity(), retained_capacity);
    }
}
