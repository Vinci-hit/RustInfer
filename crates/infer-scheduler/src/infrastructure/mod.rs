//! Infrastructure layer — IO and runtime concerns.
//!
//! Outermost ring of the hexagonal architecture: ZMQ transports,
//! Prometheus metrics emitters, control-plane router threads — any
//! piece of code that touches the OS, network, or process-global
//! state lives here.
//!
//! Sub-modules:
//! - `kv_cache`  — scheduler-side prefix index over worker KV slots
//! - `transport` — frontend / worker / control-plane I/O
//! - `metrics`   — Prometheus recorder + `MetricsHandle`
//!
//! ## `MetricsHandle`
//!
//! `MetricsHandle = Arc<MetricsRecorder>` is re-exported from this
//! module so any application-layer System can carry its own clone
//! and emit counters without holding a borrow on the engine. The
//! recorder is `Send + Sync` and all increment paths are
//! `&self`-only.

pub mod kv_cache;
pub mod metrics;
pub mod transport;

pub use metrics::MetricsHandle;
