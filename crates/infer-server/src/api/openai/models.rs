//! GET /v1/models handler

use axum::{extract::State, Json};

use crate::state::SharedState;
use super::types::*;

/// GET /v1/models
pub async fn list_models(State(state): State<SharedState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_info.model_id.clone(),
            object: "model".to_string(),
            created: 0, // 模型创建时间不适用
            owned_by: state.model_info.owned_by.clone(),
        }],
    })
}
