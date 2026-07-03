use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde_json::{Value, json};

use crate::state::SharedState;

/// Process liveness only — always 200 while the server is up.
pub async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "rustinfer-server"
    }))
}

/// Readiness = the scheduler has answered (reply or pong) recently. The ZMQ
/// DEALER connects lazily and never errors, so socket state can't be used;
/// liveness comes from `ZmqClient::scheduler_alive()`. 503 lets load balancers
/// stop routing to a server whose engine is down instead of feeding it
/// requests that hang until the request timeout.
pub async fn ready_check(State(state): State<SharedState>) -> (StatusCode, Json<Value>) {
    if state.client.scheduler_alive() {
        (StatusCode::OK, Json(json!({ "status": "ready" })))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unavailable",
                "reason": "no scheduler contact within readiness window",
            })),
        )
    }
}
