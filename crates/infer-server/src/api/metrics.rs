use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::inference::InferenceEngine;

#[derive(Debug, Serialize)]
pub struct SystemMetrics {
    pub cpu: CpuMetrics,
    pub memory: MemoryMetrics,
    pub gpu: Option<GpuMetrics>,
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

pub async fn get_system_metrics(
    State(_engine): State<Arc<Mutex<InferenceEngine>>>,
) -> Json<SystemMetrics> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu = CpuMetrics {
        utilization_percent: sys.global_cpu_info().cpu_usage(),
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

    Json(SystemMetrics {
        cpu,
        memory,
        gpu,
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
