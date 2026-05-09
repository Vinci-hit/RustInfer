//! Priority-aware multi-tier QoS scheduling policy (stub).

use crate::cache::traits::CacheState;
use crate::policy::traits::{BatchPlan, RunningSet, SchedulingPolicy};
use crate::request::queue::WaitingQueue;
use crate::utils::token_budget::TokenBudget;

/// Priority-aware scheduling policy (stub).
///
/// **Current status: NOT IMPLEMENTED.**
/// When implemented, this will support multiple QoS tiers with configurable
/// concurrency limits and timeout policies.
pub struct PriorityPolicy;

impl PriorityPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl SchedulingPolicy for PriorityPolicy {
    fn schedule(
        &self,
        _waiting: &WaitingQueue,
        _running: &RunningSet,
        _budget: &TokenBudget,
        _cache_state: &CacheState,
    ) -> BatchPlan {
        tracing::warn!("PriorityPolicy::schedule called but not implemented, returning empty plan");
        BatchPlan::empty()
    }

    fn name(&self) -> &'static str {
        "priority (stub)"
    }
}

impl Default for PriorityPolicy {
    fn default() -> Self {
        Self::new()
    }
}
