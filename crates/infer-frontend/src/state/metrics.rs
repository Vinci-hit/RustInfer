use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu: CpuMetrics,
    pub memory: MemoryMetrics,
    pub gpu: Option<GpuMetrics>,
    pub cache: Option<CacheMetrics>,
    pub engine: Option<EngineMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub utilization_percent: f32,
    pub core_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub used_mb: u64,
    pub total_mb: u64,
    pub available_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub utilization_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_celsius: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hit_rate: f64,
    pub hits: u64,
    pub misses: u64,
    pub evictable_size: usize,
    pub protected_size: usize,
    pub total_cached: usize,
    pub total_capacity: usize,
    pub evictions: u64,
    pub node_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_queue_time_ms: f64,
    pub avg_prefill_time_ms: f64,
    pub avg_decode_time_ms: f64,
    pub queue_size: usize,
    pub queue_capacity: usize,
    pub concurrent_requests: usize,
}
