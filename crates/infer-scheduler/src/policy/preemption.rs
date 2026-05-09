//! Preemption strategies (stub).
//!
//! Defines how to free KV resources when cache is exhausted.

use crate::config::PreemptionMode;
use crate::error::{Result, SchedulerError};
use crate::request::lifecycle::RequestId;

/// Preemption strategy executor.
pub struct PreemptionStrategy {
    mode: PreemptionMode,
}

impl PreemptionStrategy {
    pub fn new(mode: PreemptionMode) -> Self {
        Self { mode }
    }

    /// Select which running sequence to preempt.
    ///
    /// **Current status: STUB.** Returns NotImplemented for Recompute/Swap modes.
    pub fn select_victim(&self, _running_ids: &[RequestId]) -> Result<RequestId> {
        match self.mode {
            PreemptionMode::Disabled => Err(SchedulerError::PreemptionFailed(
                "Preemption is disabled".into(),
            )),
            PreemptionMode::Recompute => Err(SchedulerError::NotImplemented(
                "Recompute preemption victim selection not yet implemented".into(),
            )),
            PreemptionMode::Swap => Err(SchedulerError::NotImplemented(
                "Swap preemption not yet implemented".into(),
            )),
        }
    }

    /// Get the preemption mode.
    pub fn mode(&self) -> PreemptionMode {
        self.mode
    }
}
