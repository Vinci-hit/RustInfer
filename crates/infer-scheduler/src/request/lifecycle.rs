//! Type-state request lifecycle.
//!
//! Requests are represented as `Sequence<S>` where `S` is the state-specific data.
//! State transitions consume the old value and produce a new type — making invalid
//! state access a compile-time error.

use std::sync::Arc;
use std::time::Instant;

use crate::cache::kv_manager::KvAllocation;
use crate::cache::traits::PrefixMatch;
use crate::request::handle::RequestHandle;

// ═══════════════════════════════════════════════════════════════════════════════
//  Request Identity & Metadata
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique request identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(pub String);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Priority level (higher = more important).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub i32);

impl Default for Priority {
    fn default() -> Self {
        Self(0)
    }
}

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
        }
    }
}

/// Shared immutable request metadata (Arc'd, never changes after creation).
#[derive(Debug, Clone)]
pub struct RequestMeta {
    pub id: RequestId,
    pub input_ids: Vec<i32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub priority: Priority,
    pub stream: bool,
    pub stop_sequences: Vec<Vec<i32>>,
    pub arrival_time: Instant,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Sequence<S> — parameterized by state data
// ═══════════════════════════════════════════════════════════════════════════════

/// A sequence at a specific lifecycle stage.
///
/// The type parameter `S` is the state-specific data struct.
/// Different states expose different methods — enforced at compile time.
pub struct Sequence<S> {
    /// Shared immutable metadata.
    pub meta: Arc<RequestMeta>,
    /// Response channel back to the HTTP server.
    pub handle: RequestHandle,
    /// State-specific data.
    pub state: S,
}

// ─── State structs ───────────────────────────────────────────────────────────

/// Data for a queued (waiting) request.
pub struct Queued {
    /// Prefix match result (computed on enqueue for cache-aware scheduling).
    pub prefix_match: Option<PrefixMatch>,
}

/// Data for a sequence actively being prefilled.
pub struct Prefilling {
    /// KV allocation for this sequence.
    pub kv_alloc: KvAllocation,
    /// How many tokens have been prefilled so far (for chunked prefill).
    pub num_computed_tokens: usize,
    /// Total prompt length.
    pub prompt_len: usize,
    /// Time prefill was first scheduled.
    pub prefill_start: Instant,
}

/// Data for a sequence in the decode phase.
pub struct Decoding {
    /// KV allocation (grows as new tokens are generated in paged mode).
    pub kv_alloc: KvAllocation,
    /// All generated token IDs so far.
    pub output_tokens: Vec<i32>,
    /// Current sequence position (= prompt_len + output_tokens.len()).
    pub seq_position: usize,
    /// Prompt length.
    pub prompt_len: usize,
    /// Time of first generated token (TTFT measurement).
    pub first_token_time: Instant,
    /// Number of preemptions this sequence has experienced.
    pub preemption_count: u32,
}

/// Reason a sequence finished.
#[derive(Debug, Clone)]
pub enum FinishReason {
    Eos,
    MaxTokens,
    StopSequence,
    Error(String),
    Cancelled,
}

/// Data for a finished sequence.
pub struct Finished {
    pub finish_reason: FinishReason,
    pub output_tokens: Vec<i32>,
    pub metrics: CompletionMetrics,
}

/// Per-request completion metrics.
#[derive(Debug, Clone)]
pub struct CompletionMetrics {
    pub ttft: std::time::Duration,
    pub e2e_latency: std::time::Duration,
    pub num_output_tokens: u32,
    pub num_preemptions: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  State Transitions — ownership transfer guarantees validity
// ═══════════════════════════════════════════════════════════════════════════════

impl Sequence<Queued> {
    /// Create a new queued sequence.
    pub fn new(meta: Arc<RequestMeta>, handle: RequestHandle) -> Self {
        Self {
            meta,
            handle,
            state: Queued { prefix_match: None },
        }
    }

    /// Set prefix match result.
    pub fn with_prefix_match(mut self, prefix_match: PrefixMatch) -> Self {
        self.state.prefix_match = Some(prefix_match);
        self
    }

    /// Transition: Queued → Prefilling.
    pub fn start_prefill(self, kv_alloc: KvAllocation) -> Sequence<Prefilling> {
        let prompt_len = self.meta.input_ids.len();
        Sequence {
            meta: self.meta,
            handle: self.handle,
            state: Prefilling {
                kv_alloc,
                num_computed_tokens: 0,
                prompt_len,
                prefill_start: Instant::now(),
            },
        }
    }

    /// Get the request ID.
    pub fn id(&self) -> &RequestId {
        &self.meta.id
    }
}

impl Sequence<Prefilling> {
    /// Transition: Prefilling → Decoding (when prefill is complete).
    pub fn start_decode(self) -> Sequence<Decoding> {
        debug_assert!(
            self.state.num_computed_tokens >= self.state.prompt_len,
            "start_decode called before prefill complete: {} < {}",
            self.state.num_computed_tokens,
            self.state.prompt_len,
        );
        Sequence {
            meta: self.meta,
            handle: self.handle,
            state: Decoding {
                kv_alloc: self.state.kv_alloc,
                output_tokens: Vec::new(),
                seq_position: self.state.prompt_len,
                prompt_len: self.state.prompt_len,
                first_token_time: Instant::now(),
                preemption_count: 0,
            },
        }
    }

    /// Advance: chunk completed, more to go.
    pub fn advance_chunk(&mut self, tokens_processed: usize) {
        self.state.num_computed_tokens += tokens_processed;
    }

    /// Check if prefill is complete.
    pub fn is_complete(&self) -> bool {
        self.state.num_computed_tokens >= self.state.prompt_len
    }

    /// Remaining tokens to prefill.
    pub fn remaining_tokens(&self) -> usize {
        self.state.prompt_len.saturating_sub(self.state.num_computed_tokens)
    }

    /// Get the request ID.
    pub fn id(&self) -> &RequestId {
        &self.meta.id
    }
}

impl Sequence<Decoding> {
    /// Transition: Decoding → Finished.
    pub fn finish(self, reason: FinishReason) -> Sequence<Finished> {
        let ttft = self.state.first_token_time.duration_since(self.meta.arrival_time);
        let e2e = self.meta.arrival_time.elapsed();
        let num_tokens = self.state.output_tokens.len() as u32;

        Sequence {
            meta: self.meta,
            handle: self.handle,
            state: Finished {
                finish_reason: reason,
                output_tokens: self.state.output_tokens,
                metrics: CompletionMetrics {
                    ttft,
                    e2e_latency: e2e,
                    num_output_tokens: num_tokens,
                    num_preemptions: self.state.preemption_count,
                },
            },
        }
    }

    /// Append a generated token.
    pub fn append_token(&mut self, token_id: i32) {
        self.state.output_tokens.push(token_id);
        self.state.seq_position += 1;
    }

    /// Number of tokens generated so far.
    pub fn num_generated(&self) -> usize {
        self.state.output_tokens.len()
    }

    /// Whether max_tokens has been reached.
    pub fn reached_max_tokens(&self) -> bool {
        self.state.output_tokens.len() >= self.meta.max_tokens
    }

    /// Preempt: Decoding → Queued (for recompute strategy).
    /// Returns the queued sequence and the KV allocation to be freed.
    pub fn preempt_recompute(self) -> (Sequence<Queued>, KvAllocation) {
        let kv_alloc = self.state.kv_alloc;
        let seq = Sequence {
            meta: self.meta,
            handle: self.handle,
            state: Queued { prefix_match: None },
        };
        (seq, kv_alloc)
    }

    /// Get the request ID.
    pub fn id(&self) -> &RequestId {
        &self.meta.id
    }
}

impl Sequence<Finished> {
    /// Get the request ID.
    pub fn id(&self) -> &RequestId {
        &self.meta.id
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  AnySequence — runtime enum for heterogeneous storage
// ═══════════════════════════════════════════════════════════════════════════════

/// Runtime wrapper for sequences in any active lifecycle state.
pub enum AnySequence {
    Queued(Sequence<Queued>),
    Prefilling(Sequence<Prefilling>),
    Decoding(Sequence<Decoding>),
}

impl AnySequence {
    /// Get the request ID regardless of state.
    pub fn id(&self) -> &RequestId {
        match self {
            Self::Queued(s) => &s.meta.id,
            Self::Prefilling(s) => &s.meta.id,
            Self::Decoding(s) => &s.meta.id,
        }
    }
}
