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
    capacity: usize,
}

impl<T: Dtype, D: LlmBackend> RecurrentState<T, D> {
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
}

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> Runtime<T, D, M> {
    pub fn has_recurrent_state(&self) -> bool {
        self.recurrent.is_some()
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
        let Some(state) = &mut self.recurrent else {
            return Ok(());
        };
        if !req.draft_tokens.is_empty() {
            return Err(OpError::unsupported(
                "linear attention",
                "speculative execution",
            ));
        }
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
