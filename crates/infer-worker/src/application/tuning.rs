//! Serve-loop tuning constants.
//!
//! Single home for the worker's hand-tuned magic numbers so they are named,
//! documented, and adjustable in one place instead of buried as bare literals
//! in the middle of the serve loop and the KV-relief retry path.

/// How long [`crate::application::kv_relief::alloc_with_relief`] block-polls the
/// control plane for KV relief per round before escalating (round 0 → round 1)
/// or giving up. Two rounds → worst-case `2 × RELIEF_TIMEOUT_MS` of blocking.
pub const RELIEF_TIMEOUT_MS: i64 = 500;

/// Lower bound for the `kv_cache_memory_fraction` knob: never reserve less than
/// 5% of free device memory for the KV pool (a smaller pool starves batching).
pub const KV_MEM_FRACTION_MIN: f32 = 0.05;

/// Upper bound for `kv_cache_memory_fraction`: keep at least 2% headroom for
/// activation workspaces and the CUDA-graph capture pool allocated after the
/// `cudaMemGetInfo` probe.
pub const KV_MEM_FRACTION_MAX: f32 = 0.98;
