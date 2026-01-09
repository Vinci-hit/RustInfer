use axum::Json;
use serde_json::{json, Value};

pub async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "rustinfer-server"
    }))
}

pub async fn ready_check() -> Json<Value> {
    Json(json!({
        "status": "ready",
        "model_loaded": true
    }))
}
