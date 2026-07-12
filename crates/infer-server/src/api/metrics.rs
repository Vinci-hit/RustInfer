//! 系统 metrics 端点

use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct SystemMetrics {
    pub uptime_secs: u64,
    pub timestamp: i64,
}

/// GET /metrics
pub async fn get_system_metrics() -> Json<SystemMetrics> {
    Json(SystemMetrics {
        uptime_secs: 0, // TODO: track actual uptime
        timestamp: chrono::Utc::now().timestamp(),
    })
}
