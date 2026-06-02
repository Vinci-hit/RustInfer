//! `CapacityGate` — abstract backpressure seam.
//!
//! Replaces the raw `worker_busy: bool` field in `SchedulerEngine`
//! with a mode-aware capacity gate. LLM mode is always schedulable
//! (continuous batching); Diffusion mode blocks after a batch is
//! dispatched and unblocks when the worker replies.

/// Abstract capacity gate. The engine asks "can I schedule?" and the
/// gate answers based on mode-specific in-flight tracking.
pub trait CapacityGate: Send + Sync {
    /// Whether the engine is allowed to schedule a new batch.
    fn can_schedule(&self) -> bool;

    /// Called after a batch is dispatched to the worker.
    fn on_batch_sent(&mut self);

    /// Called when the worker replies with step output.
    fn on_step_output_received(&mut self);

    /// Whether a worker poll should be included in the select! loop.
    /// Diffusion: true when a batch is in-flight; LLM: always true
    /// when there is pending work (the event loop adds its own
    /// `has_pending_work` guard separately).
    fn should_poll_worker(&self) -> bool;
}

// ─── LLM ────────────────────────────────────────────────────────────────

/// LLM: always schedulable — continuous batching means the engine
/// can always issue new prefill segments alongside ongoing decode.
pub struct LlmCapacityGate;

impl CapacityGate for LlmCapacityGate {
    fn can_schedule(&self) -> bool {
        true
    }

    fn on_batch_sent(&mut self) {
        // No-op: LLM is continuous batching; there is no "batch full" state.
    }

    fn on_step_output_received(&mut self) {
        // No-op.
    }

    fn should_poll_worker(&self) -> bool {
        true // Always interested in worker output (tokens, prefill acks).
    }
}

// ─── Diffusion ──────────────────────────────────────────────────────────

/// Diffusion: at most one batch in-flight at a time.
/// After a batch is sent the gate closes; it reopens when the
/// worker returns the batch's results.
pub struct DiffusionCapacityGate {
    in_flight: bool,
}

impl DiffusionCapacityGate {
    pub fn new() -> Self {
        Self { in_flight: false }
    }
}

impl Default for DiffusionCapacityGate {
    fn default() -> Self {
        Self::new()
    }
}

impl CapacityGate for DiffusionCapacityGate {
    fn can_schedule(&self) -> bool {
        !self.in_flight
    }

    fn on_batch_sent(&mut self) {
        self.in_flight = true;
    }

    fn on_step_output_received(&mut self) {
        self.in_flight = false;
    }

    fn should_poll_worker(&self) -> bool {
        self.in_flight
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_gate_always_allows_scheduling() {
        let mut gate = LlmCapacityGate;
        assert!(gate.can_schedule());
        gate.on_batch_sent();
        assert!(gate.can_schedule(), "LLM gate should still allow scheduling after batch sent");
        gate.on_step_output_received();
        assert!(gate.can_schedule());
    }

    #[test]
    fn diffusion_gate_blocks_after_batch_sent() {
        let mut gate = DiffusionCapacityGate::new();
        assert!(gate.can_schedule(), "should be schedulable initially");
        gate.on_batch_sent();
        assert!(!gate.can_schedule(), "should block after batch sent");
        gate.on_step_output_received();
        assert!(gate.can_schedule(), "should unblock after step output");
    }

    #[test]
    fn diffusion_gate_should_poll_worker() {
        let mut gate = DiffusionCapacityGate::new();
        assert!(!gate.should_poll_worker(), "no batch in flight initially");
        gate.on_batch_sent();
        assert!(gate.should_poll_worker(), "should poll while batch is in flight");
        gate.on_step_output_received();
        assert!(!gate.should_poll_worker(), "should not poll after step output");
    }
}
