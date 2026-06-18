use std::collections::{HashMap, HashSet};

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

impl ActiveSeq {
    /// Create a new `ActiveSeq` with `block_table` pre-allocated to hold
    /// up to `max_tokens` additional decode steps, avoiding per-step
    /// `Vec::push` reallocation on long sequences.
    pub fn new(
        last_token: i32,
        mut block_table: Vec<u32>,
        max_tokens: usize,
        ignore_eos: bool,
    ) -> Self {
        let remaining = max_tokens.saturating_sub(1); // first token already counted
        block_table.reserve(remaining);
        Self {
            last_token,
            kv_len: block_table.len(),
            block_table,
            max_tokens,
            generated_count: 1,
            ignore_eos,
        }
    }
}

pub type ActiveSeqMap = HashMap<u64, ActiveSeq>;

/// Physical row order of buffer A (`BatchWorkspace::input_ids`).
///
/// `ActiveSeqMap` owns per-sequence facts; this owns the device-row order.
/// Keeping them separate avoids deriving A's order from HashMap iteration or
/// per-step sorting after the GPU has compacted rows in-place.
#[derive(Debug, Default)]
pub struct DecodeRows {
    /// Device-row order of buffer A.
    rows: Vec<u64>,
    /// Membership mirror of `rows` for O(1) `pending_admissions` lookups.
    /// Kept in sync by every mutator; `pending_admissions` would otherwise be
    /// O(active × rows) via `Vec::contains`.
    present: HashSet<u64>,
}

impl DecodeRows {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            present: HashSet::new(),
        }
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
        self.present.retain(|sid| active.contains_key(sid));
    }

    /// Active sequences not yet materialized in A. These are copied through
    /// buffer B and appended to A before the next decode forward.
    pub fn pending_admissions(&self, active: &ActiveSeqMap) -> Vec<u64> {
        let mut pending: Vec<u64> = active
            .keys()
            .copied()
            .filter(|sid| !self.present.contains(sid))
            .collect();
        pending.sort_unstable();
        pending
    }

    pub fn append_admissions(&mut self, admissions: &[u64]) {
        self.rows.extend_from_slice(admissions);
        self.present.extend(admissions.iter().copied());
    }

    pub fn replace_rows(&mut self, rows: Vec<u64>) {
        self.present = rows.iter().copied().collect();
        self.rows = rows;
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.present.clear();
    }
}
