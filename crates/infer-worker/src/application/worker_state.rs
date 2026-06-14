use std::collections::HashMap;

/// Per-sequence prefill state held between chunked prefill segments.
pub struct PrefillSeq {
    pub kv_len: usize,
    pub block_table: Vec<u32>,
}

pub type PrefillSeqMap = HashMap<u64, PrefillSeq>;

/// Per-sequence decode state held by the worker between iterations.
pub struct ActiveSeq {
    pub last_token: i32,
    pub kv_len: usize,
    pub block_table: Vec<u32>,
    pub max_tokens: usize,
    pub generated_count: usize,
    /// When true, EOS tokens do not terminate the sequence; it decodes
    /// all the way to `max_tokens` (fixed-length benchmarking).
    pub ignore_eos: bool,
}

pub type ActiveSeqMap = HashMap<u64, ActiveSeq>;

/// Physical row order of buffer A (`BatchWorkspace::input_ids`).
///
/// `ActiveSeqMap` owns per-sequence facts; this owns the device-row order.
/// Keeping them separate avoids deriving A's order from HashMap iteration or
/// per-step sorting after the GPU has compacted rows in-place.
#[derive(Debug, Default)]
pub struct DecodeRows {
    rows: Vec<u64>,
}

impl DecodeRows {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Drop rows whose sequence is no longer active.
    pub fn retain_active(&mut self, active: &ActiveSeqMap) {
        self.rows.retain(|sid| active.contains_key(sid));
    }

    /// Active sequences not yet materialized in A. These are copied through
    /// buffer B and appended to A before the next decode forward.
    pub fn pending_admissions(&self, active: &ActiveSeqMap) -> Vec<u64> {
        let mut pending: Vec<u64> = active
            .keys()
            .copied()
            .filter(|sid| !self.rows.contains(sid))
            .collect();
        pending.sort_unstable();
        pending
    }

    pub fn append_admissions(&mut self, admissions: &[u64]) {
        self.rows.extend_from_slice(admissions);
    }

    pub fn replace_rows(&mut self, rows: Vec<u64>) {
        self.rows = rows;
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }
}
