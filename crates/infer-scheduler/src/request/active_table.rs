//! Scheduler-side Active Request Table.
//!
//! This is a lifecycle ledger. It mirrors request state owned by the Scheduler,
//! but it does not drive Worker decode steps.

use std::collections::HashMap;

use crate::cache::kv_manager::KvAllocation;
use crate::request::lifecycle::RequestId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveRequestStatus {
    Waiting,
    Prefilling,
    Decoding,
    Cancelling,
    Finished,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ActiveRequestRecord {
    pub request_id: RequestId,
    pub model_instance_id: String,
    pub worker_id: Option<String>,
    pub kv_alloc: Option<KvAllocation>,
    pub status: ActiveRequestStatus,
    pub prompt_len: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
}

#[derive(Default)]
pub struct ActiveRequestTable {
    records: HashMap<RequestId, ActiveRequestRecord>,
}

impl ActiveRequestTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_waiting(&mut self, request_id: RequestId, prompt_len: usize, max_tokens: usize) {
        self.records.insert(
            request_id.clone(),
            ActiveRequestRecord {
                request_id,
                model_instance_id: "default".to_string(),
                worker_id: None,
                kv_alloc: None,
                status: ActiveRequestStatus::Waiting,
                prompt_len,
                generated_tokens: 0,
                max_tokens,
            },
        );
    }

    pub fn mark_prefilling(&mut self, request_id: &RequestId, kv_alloc: KvAllocation) {
        if let Some(record) = self.records.get_mut(request_id) {
            record.status = ActiveRequestStatus::Prefilling;
            record.kv_alloc = Some(kv_alloc);
        }
    }

    pub fn mark_decoding(&mut self, request_id: &RequestId) {
        if let Some(record) = self.records.get_mut(request_id) {
            record.status = ActiveRequestStatus::Decoding;
        }
    }

    pub fn record_generated_token(&mut self, request_id: &RequestId) {
        if let Some(record) = self.records.get_mut(request_id) {
            record.generated_tokens += 1;
            record.status = ActiveRequestStatus::Decoding;
        }
    }

    pub fn mark_cancelling(&mut self, request_id: &RequestId) {
        if let Some(record) = self.records.get_mut(request_id) {
            record.status = ActiveRequestStatus::Cancelling;
        }
    }

    pub fn mark_failed(&mut self, request_id: &RequestId) {
        if let Some(record) = self.records.get_mut(request_id) {
            record.status = ActiveRequestStatus::Failed;
        }
    }

    pub fn finish(&mut self, request_id: &RequestId) -> Option<ActiveRequestRecord> {
        self.records.remove(request_id).map(|mut record| {
            record.status = ActiveRequestStatus::Finished;
            record
        })
    }

    pub fn get(&self, request_id: &RequestId) -> Option<&ActiveRequestRecord> {
        self.records.get(request_id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
