//! Metrics recording.
//!
//! [`MetricsHandle`] is the application-layer-facing alias:
//! `Arc<MetricsRecorder>` clones cheaply and every Systems can hold
//! its own clone without a borrow path through the engine. The
//! underlying [`MetricsRecorder`] uses atomics on every field, so
//! all increment APIs take `&self` and concurrent emissions are
//! safe without further locking.

use std::sync::Arc;

pub mod recorder;

pub use recorder::MetricsRecorder;

/// Cheaply-clonable shared handle to the metrics recorder.
///
/// Application-layer Systems take this by value and clone it into
/// any spawned task that emits counters. The engine itself holds
/// one of these and hands it to the relevant Systems on construction.
pub type MetricsHandle = Arc<MetricsRecorder>;
