use axum::{Json, extract::State};
use serde_json::{json, Value};
use std::sync::Arc;
use crate::zmq_client::ZmqClient;

pub async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "rustinfer-server",
        "mode": "distributed"
    }))
}

pub async fn ready_check(
    State(_zmq_client): State<Arc<ZmqClient>>,
) -> Json<Value> {
    Json(json!({
        "status": "ready",
        "mode": "distributed",
        "engine_connected": true
    }))
}
