//! Process-wide runtime flags read once from the environment.
//!
//! These knobs were previously queried with `std::env::var(...)` on every
//! decode/matmul step (M3). `std::env::var` locks the global environment and
//! allocates a `String` per call — measurable when it sits in the per-token
//! hot path. We cache each flag in a `OnceLock<bool>` so the hot path only
//! reads an already-resolved bool.
//!
//! All flags are presence-checked (`is_ok()`): setting the variable to any
//! value (including empty) enables it, matching the previous semantics.

use std::sync::OnceLock;

fn cached(slot: &'static OnceLock<bool>, key: &str) -> bool {
    *slot.get_or_init(|| std::env::var(key).is_ok())
}

/// `RUSTINFER_DISABLE_GRAPH` — force the eager path, never replay CUDA graphs.
pub fn disable_graph() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    cached(&F, "RUSTINFER_DISABLE_GRAPH")
}

/// `RUSTINFER_PROFILE_GPU` — accumulate per-step wall-clock profiling counters.
pub fn profile_gpu() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    cached(&F, "RUSTINFER_PROFILE_GPU")
}

/// `RUSTINFER_FORCE_GEMM` — force the cuBLAS GEMM path even for GEMV shapes.
pub fn force_gemm() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    cached(&F, "RUSTINFER_FORCE_GEMM")
}

/// `RUSTINFER_TRACE_GRAPH` — emit per-step graph capture/replay traces.
pub fn trace_graph() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    cached(&F, "RUSTINFER_TRACE_GRAPH")
}

/// `RUSTINFER_DEBUG_LAYERS` — emit per-layer debug traces from the runner.
pub fn debug_layers() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    cached(&F, "RUSTINFER_DEBUG_LAYERS")
}

/// `RUSTINFER_TRACE_DECODE_COMPACT` — emit compact decode row/token traces.
pub fn trace_decode_compact() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    cached(&F, "RUSTINFER_TRACE_DECODE_COMPACT")
}
