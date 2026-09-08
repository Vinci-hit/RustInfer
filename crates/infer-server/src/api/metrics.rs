//! Prometheus exporter and the frontend's small JSON system summary.

use axum::{Json, extract::State, http::header, response::IntoResponse};
use serde::Serialize;

use crate::{error::AppError, state::SharedState};

#[derive(Debug, Serialize)]
pub struct SystemMetrics {
    pub uptime_secs: u64,
    pub timestamp: i64,
}

/// GET /metrics
pub async fn get_metrics(State(state): State<SharedState>) -> Result<impl IntoResponse, AppError> {
    state.metrics.set_scheduler_health(
        state.client.scheduler_alive(),
        state.client.scheduler_ready(),
    );
    let body = state
        .metrics
        .encode_with_scheduler(state.client.scheduler_metrics().as_ref())
        .map_err(|error| AppError::internal(anyhow::anyhow!(error)))?;
    Ok((
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    ))
}

/// GET /metrics/system: retained for the browser console's JSON polling.
pub async fn get_system_metrics(State(state): State<SharedState>) -> Json<SystemMetrics> {
    Json(SystemMetrics {
        uptime_secs: state.metrics.uptime_seconds() as u64,
        timestamp: chrono::Utc::now().timestamp(),
    })
}
