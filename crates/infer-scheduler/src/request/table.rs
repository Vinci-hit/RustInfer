//! Authoritative scheduler request table.
//!
//! This table is the single owner of scheduler-side request lifecycle state. It
//! keeps the type-state `Sequence<S>` model while centralizing identity indexes
//! and location invariants that used to be spread across `SchedulerEngine`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::cache::kv_manager::KvAllocation;
use crate::cache::traits::{PhysicalBlockId, PrefixMatch};
use crate::error::{Result, SchedulerError};
use crate::request::handle::{ClientId, RequestHandle};
use crate::request::lifecycle::{
    Decoding, InFlightPrefillSegment, Prefilling, Queued, RequestId, RequestMeta,
    Sequence, SequenceId,
};
use crate::request::queue::WaitingQueue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestLocation {
    Waiting,
    Prefilling,
    Decoding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalReason {
    Finished,
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct TerminalRecord {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub reason: TerminalReason,
}

#[derive(Debug, Clone)]
pub enum PrefillStartOutcome {
    Scheduled {
        request_id: RequestId,
        sequence_id: SequenceId,
        segment: InFlightPrefillSegment,
    },
    DecodeReady {
        request_id: RequestId,
        sequence_id: SequenceId,
    },
}

#[derive(Debug, Clone)]
pub enum PrefillAckOutcome {
    Continue {
        request_id: RequestId,
        sequence_id: SequenceId,
        remaining_tokens: usize,
    },
    MovedToDecoding {
        request_id: RequestId,
        sequence_id: SequenceId,
    },
}

#[derive(Debug, Clone)]
pub struct TokenAppendOutcome {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub client_id: ClientId,
    pub stream: bool,
    pub token_id: i32,
    pub worker_finished: bool,
}

pub enum FailedOutcome {
    RemovedPrefilling {
        request_id: RequestId,
        sequence_id: SequenceId,
        sequence: Sequence<Prefilling>,
    },
    RemovedDecoding {
        request_id: RequestId,
        sequence_id: SequenceId,
        sequence: Sequence<Decoding>,
    },
    NotFound {
        sequence_id: SequenceId,
    },
}

#[derive(Debug)]
pub enum CancelOutcome {
    RemovedWaiting {
        request_id: RequestId,
        sequence_id: SequenceId,
    },
    RemovedPrefilling {
        request_id: RequestId,
        sequence_id: SequenceId,
        kv_alloc: KvAllocation,
    },
    RemovedDecoding {
        request_id: RequestId,
        sequence_id: SequenceId,
        prompt_tokens: Vec<i32>,
        kv_alloc: KvAllocation,
    },
    NotFound,
}

#[derive(Default)]
pub struct RequestTable {
    by_request: HashMap<RequestId, SequenceId>,
    locations: HashMap<SequenceId, RequestLocation>,
    waiting: WaitingQueue,
    prefilling: Vec<Sequence<Prefilling>>,
    decoding: Vec<Sequence<Decoding>>,
    terminal: Vec<TerminalRecord>,
}

impl RequestTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn waiting(&self) -> &WaitingQueue {
        &self.waiting
    }

    pub fn location_for_request(&self, request_id: &RequestId) -> Option<RequestLocation> {
        let sequence_id = self.sequence_id_for(request_id)?;
        self.locations.get(&sequence_id).copied()
    }

    pub fn prefilling(&self) -> &[Sequence<Prefilling>] {
        &self.prefilling
    }

    pub fn decoding(&self) -> &[Sequence<Decoding>] {
        &self.decoding
    }

    pub fn terminal(&self) -> &[TerminalRecord] {
        &self.terminal
    }

    pub fn contains_request(&self, request_id: &RequestId) -> bool {
        self.by_request.contains_key(request_id)
    }

    pub fn sequence_id_for(&self, request_id: &RequestId) -> Option<SequenceId> {
        self.by_request.get(request_id).copied()
    }

    pub fn active_count(&self) -> usize {
        self.by_request.len()
    }

    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.prefilling.is_empty() || !self.decoding.is_empty()
    }

    pub fn insert_new(&mut self, meta: Arc<RequestMeta>, handle: RequestHandle) -> Result<()> {
        let request_id = meta.id.clone();
        let sequence_id = meta.sequence_id;
        if self.by_request.contains_key(&request_id) {
            return Err(SchedulerError::Internal(format!(
                "duplicate request_id={}",
                request_id
            )));
        }
        if self.locations.contains_key(&sequence_id) {
            return Err(SchedulerError::Internal(format!(
                "duplicate sequence_id={}",
                sequence_id
            )));
        }

        let seq = Sequence::new(meta, handle);
        self.waiting.push(seq);
        self.by_request.insert(request_id, sequence_id);
        self.locations.insert(sequence_id, RequestLocation::Waiting);
        debug_assert!(self.validate_consistency().is_ok());
        Ok(())
    }

    pub fn prefilling_continuations(&self) -> Vec<(RequestId, usize)> {
        self.prefilling
            .iter()
            .filter(|seq| !seq.has_inflight())
            .filter_map(|seq| {
                let remaining = seq.remaining_tokens();
                (remaining > 0).then(|| (seq.meta.id.clone(), remaining))
            })
            .collect()
    }

    pub fn take_waiting(&mut self, request_id: &RequestId) -> Result<Sequence<Queued>> {
        let seq = self
            .waiting
            .remove(request_id)
            .ok_or_else(|| SchedulerError::Internal(format!("waiting request not found: {}", request_id)))?;
        self.expect_location(seq.meta.sequence_id, RequestLocation::Waiting)?;
        self.locations.remove(&seq.meta.sequence_id);
        Ok(seq)
    }

    pub fn restore_waiting_front(&mut self, seq: Sequence<Queued>) -> Result<()> {
        let request_id = seq.meta.id.clone();
        let sequence_id = seq.meta.sequence_id;
        if self.locations.contains_key(&sequence_id) {
            return Err(SchedulerError::Internal(format!(
                "cannot restore waiting request with occupied sequence_id={}",
                sequence_id
            )));
        }
        if self.by_request.get(&request_id).copied() != Some(sequence_id) {
            return Err(SchedulerError::Internal(format!(
                "cannot restore waiting request with stale request index: {}",
                request_id
            )));
        }
        self.waiting.push_front(seq);
        self.locations.insert(sequence_id, RequestLocation::Waiting);
        debug_assert!(self.validate_consistency().is_ok());
        Ok(())
    }

    pub fn commit_prefill_start(
        &mut self,
        seq: Sequence<Queued>,
        kv_alloc: KvAllocation,
        prefix_match: PrefixMatch,
        scheduled_len: usize,
    ) -> Result<PrefillStartOutcome> {
        if scheduled_len == 0 {
            return Err(SchedulerError::Internal(format!(
                "scheduled_len must be > 0 for {}",
                seq.meta.id
            )));
        }
        let request_id = seq.meta.id.clone();
        let sequence_id = seq.meta.sequence_id;
        if self.by_request.get(&request_id).copied() != Some(sequence_id) {
            return Err(SchedulerError::Internal(format!(
                "prefill start has stale request index: {}",
                request_id
            )));
        }
        if self.locations.contains_key(&sequence_id) {
            return Err(SchedulerError::Internal(format!(
                "prefill start sequence already located: {}",
                sequence_id
            )));
        }

        let mut prefilling = seq.start_prefill(kv_alloc);
        prefilling.state.num_computed_tokens = prefix_match
            .num_cached_tokens
            .min(prefilling.state.prompt_len);

        if prefilling.is_complete() {
            self.locations.insert(sequence_id, RequestLocation::Decoding);
            self.decoding.push(prefilling.start_decode());
            debug_assert!(self.validate_consistency().is_ok());
            return Ok(PrefillStartOutcome::DecodeReady {
                request_id,
                sequence_id,
            });
        }

        let start = prefilling.state.num_computed_tokens;
        let end = (start + scheduled_len).min(prefilling.state.prompt_len);
        if start >= end {
            return Err(SchedulerError::Internal(format!(
                "invalid prefill segment for {}: [{}..{})",
                request_id, start, end
            )));
        }
        prefilling.set_inflight(start, end);
        let segment = prefilling
            .state
            .inflight
            .expect("set_inflight must populate inflight segment");
        self.prefilling.push(prefilling);
        self.locations.insert(sequence_id, RequestLocation::Prefilling);
        debug_assert!(self.validate_consistency().is_ok());
        Ok(PrefillStartOutcome::Scheduled {
            request_id,
            sequence_id,
            segment,
        })
    }

    pub fn set_prefill_inflight(
        &mut self,
        request_id: &RequestId,
        scheduled_len: usize,
    ) -> Result<InFlightPrefillSegment> {
        if scheduled_len == 0 {
            return Err(SchedulerError::Internal(format!(
                "scheduled_len must be > 0 for {}",
                request_id
            )));
        }
        let sequence_id = self
            .sequence_id_for(request_id)
            .ok_or_else(|| SchedulerError::Internal(format!("request not found: {}", request_id)))?;
        self.expect_location(sequence_id, RequestLocation::Prefilling)?;
        let seq = self
            .prefilling
            .iter_mut()
            .find(|seq| seq.meta.sequence_id == sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("prefilling sequence not found: {}", sequence_id)))?;
        if seq.has_inflight() {
            return Err(SchedulerError::Internal(format!(
                "prefilling sequence already has inflight segment: {}",
                sequence_id
            )));
        }
        let remaining = seq.remaining_tokens();
        if remaining == 0 {
            return Err(SchedulerError::Internal(format!(
                "prefilling sequence has no remaining tokens: {}",
                sequence_id
            )));
        }
        let start = seq.state.num_computed_tokens;
        let end = (start + scheduled_len).min(seq.state.prompt_len);
        if start >= end {
            return Err(SchedulerError::Internal(format!(
                "invalid continuation segment for {}: [{}..{})",
                request_id, start, end
            )));
        }
        seq.set_inflight(start, end);
        Ok(seq.state.inflight.expect("set_inflight must populate segment"))
    }

    pub fn ack_prefill(&mut self, sequence_id: SequenceId) -> Result<PrefillAckOutcome> {
        self.expect_location(sequence_id, RequestLocation::Prefilling)?;
        let idx = self
            .prefilling
            .iter()
            .position(|seq| seq.meta.sequence_id == sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("prefilling sequence not found: {}", sequence_id)))?;
        let mut seq = self.prefilling.remove(idx);
        let Some(inflight) = seq.ack_inflight() else {
            return Err(SchedulerError::Internal(format!(
                "prefill ack without inflight segment: {}",
                sequence_id
            )));
        };
        let request_id = seq.meta.id.clone();
        if inflight.is_final || seq.is_complete() {
            self.locations.insert(sequence_id, RequestLocation::Decoding);
            self.decoding.push(seq.start_decode());
            debug_assert!(self.validate_consistency().is_ok());
            Ok(PrefillAckOutcome::MovedToDecoding {
                request_id,
                sequence_id,
            })
        } else {
            let remaining = seq.remaining_tokens();
            self.prefilling.push(seq);
            debug_assert!(self.validate_consistency().is_ok());
            Ok(PrefillAckOutcome::Continue {
                request_id,
                sequence_id,
                remaining_tokens: remaining,
            })
        }
    }

    pub fn append_generated_token(
        &mut self,
        sequence_id: SequenceId,
        token_id: i32,
        worker_finished: bool,
    ) -> Result<TokenAppendOutcome> {
        self.expect_location(sequence_id, RequestLocation::Decoding)?;
        let seq = self
            .decoding
            .iter_mut()
            .find(|seq| seq.meta.sequence_id == sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("decoding sequence not found: {}", sequence_id)))?;
        seq.append_token(token_id);
        Ok(TokenAppendOutcome {
            request_id: seq.meta.id.clone(),
            sequence_id,
            client_id: ClientId(seq.handle.client_id.0.clone()),
            stream: seq.meta.stream,
            token_id,
            worker_finished,
        })
    }

    pub fn extend_decode_kv(
        &mut self,
        sequence_id: SequenceId,
        blocks: Vec<PhysicalBlockId>,
    ) -> Result<()> {
        self.expect_location(sequence_id, RequestLocation::Decoding)?;
        let seq = self
            .decoding
            .iter_mut()
            .find(|seq| seq.meta.sequence_id == sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("decoding sequence not found: {}", sequence_id)))?;
        match &mut seq.state.kv_alloc {
            KvAllocation::Blocks(existing) => {
                existing.extend(blocks);
                Ok(())
            }
            KvAllocation::Slot(_) => Err(SchedulerError::Internal(format!(
                "extend_decode_kv called for slot allocation sequence_id={}",
                sequence_id
            ))),
        }
    }

    pub fn finish_decoding(&mut self, sequence_id: SequenceId) -> Result<Sequence<Decoding>> {
        self.expect_location(sequence_id, RequestLocation::Decoding)?;
        let idx = self
            .decoding
            .iter()
            .position(|seq| seq.meta.sequence_id == sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("decoding sequence not found: {}", sequence_id)))?;
        let seq = self.decoding.remove(idx);
        self.remove_active(seq.meta.id.clone(), sequence_id, TerminalReason::Finished)?;
        debug_assert!(self.validate_consistency().is_ok());
        Ok(seq)
    }

    pub fn running_sequence_ids(&self) -> Vec<SequenceId> {
        self.prefilling
            .iter()
            .map(|seq| seq.meta.sequence_id)
            .chain(self.decoding.iter().map(|seq| seq.meta.sequence_id))
            .collect()
    }

    pub fn fail_sequence(&mut self, sequence_id: SequenceId, message: &str) -> Result<FailedOutcome> {
        let Some(location) = self.locations.get(&sequence_id).copied() else {
            return Ok(FailedOutcome::NotFound { sequence_id });
        };
        match location {
            RequestLocation::Waiting => Ok(FailedOutcome::NotFound { sequence_id }),
            RequestLocation::Prefilling => {
                let idx = self
                    .prefilling
                    .iter()
                    .position(|seq| seq.meta.sequence_id == sequence_id)
                    .ok_or_else(|| SchedulerError::Internal(format!("prefilling sequence not found: {}", sequence_id)))?;
                let seq = self.prefilling.remove(idx);
                let request_id = seq.meta.id.clone();
                self.remove_active(
                    request_id.clone(),
                    sequence_id,
                    TerminalReason::Failed(message.to_string()),
                )?;
                Ok(FailedOutcome::RemovedPrefilling {
                    request_id,
                    sequence_id,
                    sequence: seq,
                })
            }
            RequestLocation::Decoding => {
                let idx = self
                    .decoding
                    .iter()
                    .position(|seq| seq.meta.sequence_id == sequence_id)
                    .ok_or_else(|| SchedulerError::Internal(format!("decoding sequence not found: {}", sequence_id)))?;
                let seq = self.decoding.remove(idx);
                let request_id = seq.meta.id.clone();
                self.remove_active(
                    request_id.clone(),
                    sequence_id,
                    TerminalReason::Failed(message.to_string()),
                )?;
                Ok(FailedOutcome::RemovedDecoding {
                    request_id,
                    sequence_id,
                    sequence: seq,
                })
            }
        }
    }

    pub fn take_prefilling_by_request(
        &mut self,
        request_id: &RequestId,
        reason: TerminalReason,
    ) -> Result<Option<Sequence<Prefilling>>> {
        let Some(sequence_id) = self.sequence_id_for(request_id) else {
            return Ok(None);
        };
        if self.locations.get(&sequence_id).copied() != Some(RequestLocation::Prefilling) {
            return Ok(None);
        }
        let idx = self
            .prefilling
            .iter()
            .position(|seq| seq.meta.sequence_id == sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("prefilling request not found: {}", request_id)))?;
        let seq = self.prefilling.remove(idx);
        self.remove_active(request_id.clone(), sequence_id, reason)?;
        Ok(Some(seq))
    }

    pub fn cancel_request(&mut self, request_id: &RequestId) -> Result<CancelOutcome> {
        let Some(sequence_id) = self.sequence_id_for(request_id) else {
            return Ok(CancelOutcome::NotFound);
        };
        let location = *self
            .locations
            .get(&sequence_id)
            .ok_or_else(|| SchedulerError::Internal(format!("request index has no location: {}", request_id)))?;
        match location {
            RequestLocation::Waiting => {
                let seq = self
                    .waiting
                    .remove(request_id)
                    .ok_or_else(|| SchedulerError::Internal(format!("waiting request not found: {}", request_id)))?;
                self.remove_active(request_id.clone(), sequence_id, TerminalReason::Cancelled)?;
                Ok(CancelOutcome::RemovedWaiting {
                    request_id: seq.meta.id.clone(),
                    sequence_id,
                })
            }
            RequestLocation::Prefilling => {
                let idx = self
                    .prefilling
                    .iter()
                    .position(|seq| seq.meta.sequence_id == sequence_id)
                    .ok_or_else(|| SchedulerError::Internal(format!("prefilling request not found: {}", request_id)))?;
                let seq = self.prefilling.remove(idx);
                let kv_alloc = seq.state.kv_alloc;
                self.remove_active(request_id.clone(), sequence_id, TerminalReason::Cancelled)?;
                Ok(CancelOutcome::RemovedPrefilling {
                    request_id: seq.meta.id.clone(),
                    sequence_id,
                    kv_alloc,
                })
            }
            RequestLocation::Decoding => {
                let idx = self
                    .decoding
                    .iter()
                    .position(|seq| seq.meta.sequence_id == sequence_id)
                    .ok_or_else(|| SchedulerError::Internal(format!("decoding request not found: {}", request_id)))?;
                let seq = self.decoding.remove(idx);
                let prompt_tokens = seq.meta.input_ids.clone();
                let kv_alloc = seq.state.kv_alloc;
                self.remove_active(request_id.clone(), sequence_id, TerminalReason::Cancelled)?;
                Ok(CancelOutcome::RemovedDecoding {
                    request_id: seq.meta.id.clone(),
                    sequence_id,
                    prompt_tokens,
                    kv_alloc,
                })
            }
        }
    }

    pub fn validate_consistency(&self) -> Result<()> {
        let mut seen_requests = HashSet::new();
        let mut seen_sequences = HashSet::new();

        for seq in self.waiting.iter() {
            self.validate_sequence_location(seq.meta.id.clone(), seq.meta.sequence_id, RequestLocation::Waiting, &mut seen_requests, &mut seen_sequences)?;
        }
        for seq in &self.prefilling {
            self.validate_sequence_location(seq.meta.id.clone(), seq.meta.sequence_id, RequestLocation::Prefilling, &mut seen_requests, &mut seen_sequences)?;
            if !seq.has_inflight() && seq.remaining_tokens() == 0 {
                return Err(SchedulerError::Internal(format!(
                    "prefilling sequence has no work and no inflight: {}",
                    seq.meta.id
                )));
            }
            if let Some(inflight) = seq.state.inflight {
                if inflight.segment_start >= inflight.segment_end
                    || inflight.segment_end > seq.state.prompt_len
                    || inflight.segment_start < seq.state.num_computed_tokens
                {
                    return Err(SchedulerError::Internal(format!(
                        "invalid inflight segment for {}: {:?}",
                        seq.meta.id, inflight
                    )));
                }
            }
        }
        for seq in &self.decoding {
            self.validate_sequence_location(seq.meta.id.clone(), seq.meta.sequence_id, RequestLocation::Decoding, &mut seen_requests, &mut seen_sequences)?;
        }

        if seen_requests.len() != self.by_request.len() {
            return Err(SchedulerError::Internal(format!(
                "request index count mismatch: seen={} index={}",
                seen_requests.len(),
                self.by_request.len()
            )));
        }
        if seen_sequences.len() != self.locations.len() {
            return Err(SchedulerError::Internal(format!(
                "sequence location count mismatch: seen={} locations={}",
                seen_sequences.len(),
                self.locations.len()
            )));
        }
        Ok(())
    }

    fn validate_sequence_location(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
        expected: RequestLocation,
        seen_requests: &mut HashSet<RequestId>,
        seen_sequences: &mut HashSet<SequenceId>,
    ) -> Result<()> {
        if !seen_requests.insert(request_id.clone()) {
            return Err(SchedulerError::Internal(format!("duplicate request in table: {}", request_id)));
        }
        if !seen_sequences.insert(sequence_id) {
            return Err(SchedulerError::Internal(format!("duplicate sequence in table: {}", sequence_id)));
        }
        if self.by_request.get(&request_id).copied() != Some(sequence_id) {
            return Err(SchedulerError::Internal(format!("request index mismatch: {}", request_id)));
        }
        if self.locations.get(&sequence_id).copied() != Some(expected) {
            return Err(SchedulerError::Internal(format!("location mismatch for sequence_id={}", sequence_id)));
        }
        Ok(())
    }

    fn expect_location(&self, sequence_id: SequenceId, expected: RequestLocation) -> Result<()> {
        match self.locations.get(&sequence_id).copied() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(SchedulerError::Internal(format!(
                "sequence_id={} location mismatch: expected {:?}, got {:?}",
                sequence_id, expected, actual
            ))),
            None => Err(SchedulerError::Internal(format!(
                "sequence_id={} has no active location",
                sequence_id
            ))),
        }
    }

    fn remove_active(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        reason: TerminalReason,
    ) -> Result<()> {
        match self.by_request.remove(&request_id) {
            Some(id) if id == sequence_id => {}
            Some(id) => {
                return Err(SchedulerError::Internal(format!(
                    "request index mismatch on removal: {} -> {}, expected {}",
                    request_id, id, sequence_id
                )));
            }
            None => {
                return Err(SchedulerError::Internal(format!(
                    "request index missing on removal: {}",
                    request_id
                )));
            }
        }
        self.locations.remove(&sequence_id).ok_or_else(|| {
            SchedulerError::Internal(format!("location missing on removal: {}", sequence_id))
        })?;
        self.terminal.push(TerminalRecord {
            request_id,
            sequence_id,
            reason,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::traits::PrefixMatch;
    use crate::request::lifecycle::{Priority, SamplingParams};
    use std::time::Instant;

    fn meta(request_id: &str, sequence_id: u64, prompt_len: usize, max_tokens: usize) -> Arc<RequestMeta> {
        Arc::new(RequestMeta {
            id: RequestId(request_id.to_string()),
            sequence_id: SequenceId(sequence_id),
            input_ids: (0..prompt_len as i32).collect(),
            max_tokens,
            sampling: SamplingParams::default(),
            priority: Priority::default(),
            stream: false,
            stop_sequences: vec![],
            diffusion: None,
            arrival_time: Instant::now(),
        })
    }

    fn no_prefix() -> PrefixMatch {
        PrefixMatch::none()
    }

    #[test]
    fn insert_rejects_duplicate_request_id() {
        let mut table = RequestTable::new();
        table.insert_new(meta("req", 1, 4, 8), RequestHandle::noop()).unwrap();
        assert!(table.insert_new(meta("req", 2, 4, 8), RequestHandle::noop()).is_err());
        assert!(table.validate_consistency().is_ok());
    }

    #[test]
    fn insert_rejects_duplicate_sequence_id() {
        let mut table = RequestTable::new();
        table.insert_new(meta("req-a", 1, 4, 8), RequestHandle::noop()).unwrap();
        assert!(table.insert_new(meta("req-b", 1, 4, 8), RequestHandle::noop()).is_err());
        assert!(table.validate_consistency().is_ok());
    }

    #[test]
    fn waiting_prefilling_decoding_complete_lifecycle() {
        let mut table = RequestTable::new();
        let request_id = RequestId("req".to_string());
        table.insert_new(meta("req", 7, 4, 8), RequestHandle::noop()).unwrap();
        let queued = table.take_waiting(&request_id).unwrap();
        let outcome = table
            .commit_prefill_start(queued, KvAllocation::Blocks(vec![]), no_prefix(), 4)
            .unwrap();
        assert!(matches!(outcome, PrefillStartOutcome::Scheduled { .. }));
        let ack = table.ack_prefill(SequenceId(7)).unwrap();
        assert!(matches!(ack, PrefillAckOutcome::MovedToDecoding { .. }));
        let token = table.append_generated_token(SequenceId(7), 42, true).unwrap();
        assert_eq!(token.token_id, 42);
        let seq = table.finish_decoding(SequenceId(7)).unwrap();
        assert_eq!(seq.state.output_tokens, vec![42]);
        assert_eq!(table.active_count(), 0);
        assert!(table.validate_consistency().is_ok());
    }

    #[test]
    fn cancel_waiting_removes_indexes() {
        let mut table = RequestTable::new();
        let request_id = RequestId("req".to_string());
        table.insert_new(meta("req", 1, 4, 8), RequestHandle::noop()).unwrap();
        let outcome = table.cancel_request(&request_id).unwrap();
        assert!(matches!(outcome, CancelOutcome::RemovedWaiting { .. }));
        assert_eq!(table.active_count(), 0);
        assert!(table.validate_consistency().is_ok());
    }

    #[test]
    fn full_prefix_hit_moves_directly_to_decoding() {
        let mut table = RequestTable::new();
        let request_id = RequestId("req".to_string());
        table.insert_new(meta("req", 3, 4, 8), RequestHandle::noop()).unwrap();
        let queued = table.take_waiting(&request_id).unwrap();
        let outcome = table
            .commit_prefill_start(
                queued,
                KvAllocation::Blocks(vec![]),
                PrefixMatch {
                    num_cached_tokens: 4,
                    cached_blocks: vec![],
                    last_block_hash: None,
                },
                1,
            )
            .unwrap();
        assert!(matches!(outcome, PrefillStartOutcome::DecodeReady { .. }));
        assert_eq!(table.prefilling().len(), 0);
        assert_eq!(table.decoding().len(), 1);
        assert!(table.validate_consistency().is_ok());
    }
}
