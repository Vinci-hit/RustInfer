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

/// Readiness requires a fresh compatible Pong reporting a ready model, worker,
/// and scheduler engine. Ordinary inference replies cannot renew readiness.
pub async fn ready_check(State(state): State<SharedState>) -> (StatusCode, Json<Value>) {
    readiness_response(state.client.readiness_state())
}

fn readiness_response(
    readiness: infer_protocol::scheduler_to_server::SchedulerReadiness,
) -> (StatusCode, Json<Value>) {
    use infer_protocol::scheduler_to_server::SchedulerReadiness;
    if readiness == SchedulerReadiness::Ready {
        (StatusCode::OK, Json(json!({ "status": "ready" })))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": readiness.as_str(),
                "reason": match readiness {
                    SchedulerReadiness::Loading => "model and worker startup has not completed",
                    SchedulerReadiness::Draining => "service is shutting down",
                    _ => "scheduler or worker unavailable, stale, or incompatible",
                },
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_protocol::scheduler_to_server::SchedulerReadiness;

    #[test]
    fn only_ready_state_serves_traffic() {
        for state in [
            SchedulerReadiness::Loading,
            SchedulerReadiness::Draining,
            SchedulerReadiness::Failed,
        ] {
            let (status, body) = readiness_response(state);
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body.0["status"], state.as_str());
        }
        assert_eq!(
            readiness_response(SchedulerReadiness::Ready).0,
            StatusCode::OK
        );
    }
}
