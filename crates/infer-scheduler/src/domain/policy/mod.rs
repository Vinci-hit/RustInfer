//! Scheduling policy domain service.
//!
//! Pluggable strategy for deciding which sessions run in the next
//! iteration. Two production implementations:
//!
//! - [`ContinuousBatchingPolicy`] — chunked-prefill / continuous
//!   batching for LLM workloads.
//! - [`DiffusionPolicy`] — batch-in / batch-out for diffusion
//!   workloads.
//!
//! [`TokenBudget`] is the value object the engine hands to the
//! policy each iteration capturing per-iter capacity limits.

pub mod continuous_batching;
pub mod diffusion;
pub mod token_budget;
pub mod traits;

pub use continuous_batching::ContinuousBatchingPolicy;
pub use diffusion::DiffusionPolicy;
pub use token_budget::TokenBudget;
pub use traits::{BatchPlan, DecodeEntry, PrefillEntry, PreemptionAction, SchedulingPolicy};
