//! Request ID middleware
//!
//! 为每个请求注入 `X-Request-Id` header。
//! 如果客户端已携带该 header，则保留；否则生成新的 UUID。

use axum::{
    extract::Request,
    http::HeaderValue,
    middleware::Next,
    response::Response,
};

const REQUEST_ID_HEADER: &str = "x-request-id";

/// Request ID middleware
pub async fn inject_request_id(mut req: Request, next: Next) -> Response {
    // 如果请求中没有 X-Request-Id，生成一个
    if !req.headers().contains_key(REQUEST_ID_HEADER) {
        let id = uuid::Uuid::new_v4().to_string();
        req.headers_mut().insert(
            REQUEST_ID_HEADER,
            HeaderValue::from_str(&id).unwrap(),
        );
    }

    let request_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .cloned();

    let mut response = next.run(req).await;

    // 将 Request ID 回传到响应 header
    if let Some(id) = request_id {
        response.headers_mut().insert(REQUEST_ID_HEADER, id);
    }

    response
}
