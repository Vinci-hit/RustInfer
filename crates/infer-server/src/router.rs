//! Router 构建
//!
//! 集中注册所有路由 + middleware 层。

use axum::{
    Router,
    http::{HeaderValue, Method, Uri, header},
    middleware,
    routing::{get, post},
};
use tower_http::cors::CorsLayer;

use crate::api;
use crate::middleware::request_id;
use crate::state::SharedState;

/// 构建完整的应用 Router
pub fn build_router(state: SharedState, cors_allowed_origins: &[String]) -> anyhow::Result<Router> {
    let cors = build_cors_layer(cors_allowed_origins)?;
    let router = Router::new()
        // OpenAI 兼容端点
        .route(
            "/v1/chat/completions",
            post(api::openai::chat::chat_completions),
        )
        .route(
            "/v1/completions",
            post(api::openai::completion::completions),
        )
        .route("/v1/models", get(api::openai::models::list_models))
        // 运维端点
        .route("/health", get(api::health::health_check))
        .route("/ready", get(api::health::ready_check))
        .route("/metrics", get(api::metrics::get_system_metrics))
        // 共享状态
        .with_state(state)
        // Middleware 层（从下到上执行）
        .layer(middleware::from_fn(request_id::inject_request_id))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // No CORS layer means browsers enforce the normal same-origin policy.
    // Cross-origin access is opt-in through explicit origins only.
    Ok(match cors {
        Some(cors) => router.layer(cors),
        None => router,
    })
}

fn build_cors_layer(origins: &[String]) -> anyhow::Result<Option<CorsLayer>> {
    if origins.is_empty() {
        return Ok(None);
    }

    let mut allowed = Vec::with_capacity(origins.len());
    for configured in origins {
        let origin = configured.trim();
        if origin.is_empty() || origin == "*" {
            anyhow::bail!(
                "CORS origins must be explicit http(s) origins, not {:?}",
                origin
            );
        }

        let uri: Uri = origin
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid CORS origin {:?}: {}", origin, e))?;
        let Some(scheme @ ("http" | "https")) = uri.scheme_str() else {
            anyhow::bail!(
                "CORS origin must include an http(s) scheme and host: {:?}",
                origin
            );
        };
        let Some(authority) = uri.authority() else {
            anyhow::bail!("CORS origin must include a host: {:?}", origin);
        };
        if let Some(path) = uri.path_and_query()
            && path.as_str() != "/"
        {
            anyhow::bail!("CORS origin must not include a path or query: {:?}", origin);
        }

        let canonical_origin = format!("{}://{}", scheme, authority);
        allowed.push(HeaderValue::from_str(&canonical_origin).map_err(|e| {
            anyhow::anyhow!(
                "invalid CORS origin header value {:?}: {}",
                canonical_origin,
                e
            )
        })?);
    }

    Ok(Some(
        CorsLayer::new()
            .allow_origin(allowed)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]),
    ))
}

#[cfg(test)]
mod tests {
    use super::build_cors_layer;

    #[test]
    fn cors_defaults_to_same_origin_only() {
        assert!(build_cors_layer(&[]).unwrap().is_none());
    }

    #[test]
    fn cors_accepts_explicit_http_origins() {
        let origins = vec![
            "https://console.example.com".to_string(),
            "http://localhost:3000".to_string(),
        ];
        assert!(build_cors_layer(&origins).unwrap().is_some());
    }

    #[test]
    fn cors_rejects_wildcards_and_non_origins() {
        assert!(build_cors_layer(&["*".to_string()]).is_err());
        assert!(build_cors_layer(&["https://example.com/path".to_string()]).is_err());
        assert!(build_cors_layer(&["example.com".to_string()]).is_err());
    }
}
