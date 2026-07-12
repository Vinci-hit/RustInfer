//! Serve-loop tuning constants.
//!
//! Single home for the worker's hand-tuned magic numbers so they are named,
//! documented, and adjustable in one place instead of buried as bare literals
//! in the middle of the serve loop and the KV-relief retry path.

/// How long [`crate::application::kv_relief::alloc_with_relief`] block-polls the
/// control plane for KV relief per round before escalating (round 0 → round 1)
/// or giving up. Two rounds → worst-case `2 × RELIEF_TIMEOUT_MS` of blocking.
///
/// Lowered from 500ms (B3): under high QPS the worker was blocking up to 1s per
/// allocation while waiting for scheduler relief, which dominated the TTFT tail.
/// The scheduler answers AllocFailed within a few ms (LRU evict / preempt) when
/// it can answer at all, so a shorter deadline fails fast to the next round /
/// batch instead of stalling the serve loop.
pub const RELIEF_TIMEOUT_MS: i64 = 100;

/// Lower bound for the `kv_cache_memory_fraction` knob: never reserve less than
/// 5% of free device memory for the KV pool (a smaller pool starves batching).
pub const KV_MEM_FRACTION_MIN: f32 = 0.05;

/// Upper bound for `kv_cache_memory_fraction`: keep at least 2% headroom for
/// the *incremental* device allocations the prewarm pass makes after the probe
/// (recycling-pool scratch, cuDNN plans for shapes the dummy forward did not
/// touch). Since the probe now runs *after* the activation workspace exists
/// (see [`crate::application::serve_loop`]'s profiling-based sizing), the fixed
/// GiB-scale logits/activation cost is already subtracted from `free` — so this
/// fraction means what users expect (fraction of *usable* memory for KV), and
/// the residual headroom no longer has to absorb a fixed cost as a percentage.
pub const KV_MEM_FRACTION_MAX: f32 = 0.98;

/// Blocks in the throwaway profiling KV pool used during the bootstrap
/// dummy-forward memory probe. The pool only needs to hold one synthetic
/// prefill (capped to this many tokens) plus one full-width decode so the eager
/// forward commits its lazy library workspaces before the real pool is sized;
/// it is freed and replaced by the real pool immediately after the probe
/// ([`crate::application::runtime::Runtime::resize_kv_pool`]). Small on purpose:
/// its bytes are still resident when `cudaMemGetInfo` is read, so a large probe
/// pool would under-size the real pool by its own footprint. See
/// [`crate::application::runtime::Runtime::profile_forward`].
pub const PROFILE_KV_BLOCKS: usize = 512;

/// Device-memory headroom held back from the KV pool for allocations the
/// *prewarm* pass makes after the probe. The post-dummy-forward probe accounts
/// for the fixed activation workspace and one forward's worth of lazy
/// cuBLASLt/cuDNN/recycling-pool state, but `prime_graphs` + the decode/prefill/
/// mixed prewarm then touch ~90 further shapes, each of which may add a cuDNN
/// SDPA plan workspace and grow the recycling pool (bounded by its own retain
/// budget). Graph *capture* itself draws from the pre-reserved fixed arena, so
/// this reserve only has to cover those incremental per-shape allocations. Set
/// to 1 GiB so even `kv_cache_memory_fraction = 0.98` reaches `Ready` without
/// OOM on the prewarm path.
pub const PREWARM_HEADROOM_BYTES: usize = 1024 * 1024 * 1024;
