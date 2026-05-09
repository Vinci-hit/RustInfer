//! Runtime gauges — system-level metrics sampled per iteration.

/// System-level gauges snapshot.
#[derive(Debug, Clone, Default)]
pub struct SystemGauges {
    pub waiting_queue_depth: usize,
    pub running_batch_size: usize,
    pub prefilling_count: usize,
    pub decoding_count: usize,
    pub gpu_cache_utilization: f64,
    pub iteration_id: u64,
}
