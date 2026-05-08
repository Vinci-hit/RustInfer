//! 统一错误处理 → OpenAI 兼容错误格式
//!
//! 所有 handler 返回 `Result<Response, AppError>`，
//! AppError 自动转换为 OpenAI 风格的 JSON 错误响应。

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
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
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            AppError::BadRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", msg.clone())
            }
            AppError::ModelNotFound(msg) => {
                (StatusCode::NOT_FOUND, "invalid_request_error", msg.clone())
            }
            AppError::TooManyRequests(msg) => {
                (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", msg.clone())
            }
            AppError::Timeout(msg) => {
                (StatusCode::GATEWAY_TIMEOUT, "timeout_error", msg.clone())
            }
            AppError::Internal(err) => {
                tracing::error!("Internal error: {:?}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    err.to_string(),
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
