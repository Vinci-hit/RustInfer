//! Per-request metrics recorder.

use std::sync::atomic::{AtomicU64, Ordering};

/// Lightweight metrics recorder.
///
/// Thread-safe via atomics. No mutex overhead on the hot path.
pub struct MetricsRecorder {
    enabled: bool,
    total_requests: AtomicU64,
    total_completions: AtomicU64,
    total_tokens_generated: AtomicU64,
    total_latency_ms: AtomicU64,
}

impl MetricsRecorder {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            total_requests: AtomicU64::new(0),
            total_completions: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
        }
    }

    /// Record a new request enqueued.
    pub fn record_enqueue(&self) {
        if self.enabled {
            self.total_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a completed request.
    pub fn record_completion(&self, latency_ms: u64, num_tokens: u32) {
        if self.enabled {
            self.total_completions.fetch_add(1, Ordering::Relaxed);
            self.total_tokens_generated.fetch_add(num_tokens as u64, Ordering::Relaxed);
            self.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
        }
    }

    /// Get snapshot of metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_completions: self.total_completions.load(Ordering::Relaxed),
            total_tokens_generated: self.total_tokens_generated.load(Ordering::Relaxed),
            total_latency_ms: self.total_latency_ms.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time metrics snapshot.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub total_completions: u64,
    pub total_tokens_generated: u64,
    pub total_latency_ms: u64,
}

impl MetricsSnapshot {
    /// Average latency per request (ms).
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_completions == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.total_completions as f64
        }
    }

    /// Average tokens per request.
    pub fn avg_tokens_per_request(&self) -> f64 {
        if self.total_completions == 0 {
            0.0
        } else {
            self.total_tokens_generated as f64 / self.total_completions as f64
        }
    }
}
