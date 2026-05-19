//! Router 构建
//!
//! 集中注册所有路由 + middleware 层。

use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};

use crate::api;
use crate::middleware::request_id;
use crate::state::SharedState;

/// 构建完整的应用 Router
pub fn build_router(state: SharedState) -> Router {
    Router::new()
        // OpenAI 兼容端点
        .route("/v1/chat/completions", post(api::openai::chat::chat_completions))
        .route("/v1/completions", post(api::openai::completion::completions))
        .route("/v1/images/generations", post(api::openai::images::image_generations))
        .route("/v1/models", get(api::openai::models::list_models))
        // 运维端点
        .route("/health", get(api::health::health_check))
        .route("/ready", get(api::health::ready_check))
        .route("/metrics", get(api::metrics::get_system_metrics))
        // 共享状态
        .with_state(state)
        // Middleware 层（从下到上执行）
        .layer(middleware::from_fn(request_id::inject_request_id))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(tower_http::trace::TraceLayer::new_for_http())
}
