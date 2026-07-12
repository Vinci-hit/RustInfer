//! 统一错误处理 → OpenAI 兼容错误格式
//!
//! 所有 handler 返回 `Result<Response, AppError>`，
//! AppError 自动转换为 OpenAI 风格的 JSON 错误响应。

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// OpenAI 错误响应格式
#[derive(Debug, Serialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

#[derive(Debug, Serialize)]
pub struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// 统一应用错误类型
#[derive(Debug)]
pub enum AppError {
    /// 请求参数无效 (400)
    BadRequest(String),
    /// 模型不存在 (404)
    ModelNotFound(String),
    /// 服务过载 (429)
    TooManyRequests(String),
    /// 推理超时 (504)
    Timeout(String),
    /// 内部错误 (500)
    Internal(anyhow::Error),
}

/// Marker error: the client's bounded submit channel to the scheduler was full
/// (server overloaded). The ZMQ client wraps it in `anyhow`; [`AppError::from_submit`]
/// downcasts it to a 429 instead of a 500.
#[derive(Debug)]
pub struct ServerOverloaded;

impl std::fmt::Display for ServerOverloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "server overloaded: request submit queue full")
    }
}
impl std::error::Error for ServerOverloaded {}

/// Marker error: the per-request deadline elapsed without a scheduler reply
/// (scheduler/worker dead or overloaded). [`AppError::from_submit`] downcasts
/// it to a 504 instead of a 500 so load balancers and clients can tell a
/// timeout from a server bug.
#[derive(Debug)]
pub struct RequestTimedOut {
    pub request_id: String,
    pub secs: u64,
}

impl std::fmt::Display for RequestTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "request {} timed out after {}s waiting for the inference engine",
            self.request_id, self.secs
        )
    }
}
impl std::error::Error for RequestTimedOut {}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }

    pub fn internal(err: impl Into<anyhow::Error>) -> Self {
        Self::Internal(err.into())
    }

    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    pub fn too_many(msg: impl Into<String>) -> Self {
        Self::TooManyRequests(msg.into())
    }

    /// Map an error returned by a client call (`infer`/`infer_stream`): an
    /// overload marker becomes 429 (rate_limit_error), a request-deadline
    /// marker becomes 504 (timeout_error); anything else is 500.
    pub fn from_submit(err: anyhow::Error) -> Self {
        if err.downcast_ref::<ServerOverloaded>().is_some() {
            Self::TooManyRequests("server overloaded, please retry later".to_string())
        } else if let Some(timeout) = err.downcast_ref::<RequestTimedOut>() {
            Self::Timeout(timeout.to_string())
        } else {
            Self::Internal(err)
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            AppError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
            ),
            AppError::ModelNotFound(msg) => {
                (StatusCode::NOT_FOUND, "invalid_request_error", msg.clone())
            }
            AppError::TooManyRequests(msg) => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                msg.clone(),
            ),
            AppError::Timeout(msg) => (StatusCode::GATEWAY_TIMEOUT, "timeout_error", msg.clone()),
            AppError::Internal(err) => {
                tracing::error!("Internal error: {:?}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "The server encountered an internal error".to_string(),
                )
            }
        };

        let body = OpenAIErrorResponse {
            error: OpenAIError {
                message,
                error_type: error_type.to_string(),
                param: None,
                code: None,
            },
        };

        (status, Json(body)).into_response()
    }
}

/// 方便从 anyhow::Error 自动转换
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self::Internal(err.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn internal_errors_do_not_expose_details() {
        let response =
            AppError::internal(anyhow::anyhow!("secret database details")).into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("The server encountered an internal error"));
        assert!(!text.contains("secret database details"));
    }

    #[test]
    fn request_timeout_marker_maps_to_gateway_timeout() {
        let err = anyhow::Error::new(RequestTimedOut {
            request_id: "req-timeout".to_string(),
            secs: 3,
        });
        let response = AppError::from_submit(err).into_response();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    }
}
