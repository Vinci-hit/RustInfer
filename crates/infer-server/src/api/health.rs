use axum::Json;
use serde_json::{Value, json};

pub async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "rustinfer-server"
    }))
}

pub async fn ready_check() -> Json<Value> {
    Json(json!({
        "status": "ready"
    }))
}
