use axum::{Json, Extension};
use serde::Serialize;
use std::sync::Arc;
use crate::zmq_client::ZmqClient;

#[derive(Debug, Serialize)]
pub struct SystemMetrics {
    pub cpu: CpuMetrics,
    pub memory: MemoryMetrics,
    pub gpu: Option<GpuMetrics>,
    pub cache: Option<CacheMetrics>,
    pub engine: Option<EngineMetrics>,
    pub timestamp: i64,
}

#[derive(Debug, Serialize)]
pub struct CpuMetrics {
    pub utilization_percent: f32,
    pub core_count: usize,
}

#[derive(Debug, Serialize)]
pub struct MemoryMetrics {
    pub used_mb: u64,
    pub total_mb: u64,
    pub available_mb: u64,
}

#[derive(Debug, Serialize)]
pub struct GpuMetrics {
    pub device_id: i32,
    pub utilization_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_celsius: Option<f32>,
}

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
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

pub async fn get_system_metrics(
    Extension(zmq_client): Extension<Arc<ZmqClient>>,
) -> Json<SystemMetrics> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    // sysinfo 0.32 API: 使用cpus()获取所有CPU信息
    let cpu_usage: f32 = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
        / sys.cpus().len() as f32;

    let cpu = CpuMetrics {
        utilization_percent: cpu_usage,
        core_count: sys.cpus().len(),
    };

    let memory = MemoryMetrics {
        used_mb: sys.used_memory() / 1024 / 1024,
        total_mb: sys.total_memory() / 1024 / 1024,
        available_mb: sys.available_memory() / 1024 / 1024,
    };

    #[cfg(feature = "cuda")]
    let gpu = get_gpu_metrics();

    #[cfg(not(feature = "cuda"))]
    let gpu = None;

    // Try to get engine/cache metrics from the engine
    let (cache, engine) = match zmq_client.get_metrics().await {
        Ok(engine_metrics) => {
            (
                Some(CacheMetrics {
                    hit_rate: engine_metrics.hit_rate,
                    hits: engine_metrics.hits,
                    misses: engine_metrics.misses,
                    evictable_size: engine_metrics.evictable_size,
                    protected_size: engine_metrics.protected_size,
                    total_cached: engine_metrics.total_cached,
                    total_capacity: engine_metrics.total_capacity,
                    evictions: engine_metrics.evictions,
                    node_count: engine_metrics.node_count,
                }),
                Some(EngineMetrics {
                    total_requests: engine_metrics.total_requests,
                    completed_requests: engine_metrics.completed_requests,
                    failed_requests: engine_metrics.failed_requests,
                    total_tokens_generated: engine_metrics.total_tokens_generated,
                    avg_queue_time_ms: engine_metrics.avg_queue_time_ms,
                    avg_prefill_time_ms: engine_metrics.avg_prefill_time_ms,
                    avg_decode_time_ms: engine_metrics.avg_decode_time_ms,
                    queue_size: engine_metrics.queue_size,
                    queue_capacity: engine_metrics.queue_capacity,
                    concurrent_requests: engine_metrics.concurrent_requests,
                })
            )
        }
        Err(e) => {
            tracing::warn!("Failed to get engine metrics: {:?}", e);
            (None, None)
        }
    };

    Json(SystemMetrics {
        cpu,
        memory,
        gpu,
        cache,
        engine,
        timestamp: chrono::Utc::now().timestamp(),
    })
}

#[cfg(feature = "cuda")]
fn get_gpu_metrics() -> Option<GpuMetrics> {
    use nvml_wrapper::Nvml;

    let nvml = Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;

    let utilization = device.utilization_rates().ok()?;
    let memory = device.memory_info().ok()?;
    let temp = device.temperature(
        nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu
    ).ok();

    Some(GpuMetrics {
        device_id: 0,
        utilization_percent: utilization.gpu as f32,
        memory_used_mb: memory.used / 1024 / 1024,
        memory_total_mb: memory.total / 1024 / 1024,
        temperature_celsius: temp.map(|t| t as f32),
    })
}
