//! Reserve capacity before Axum reads or deserializes a request body.
use crate::error::AppError;
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub type AdmissionPermit = Arc<OwnedSemaphorePermit>;

pub async fn admit(
    State(capacity): State<Arc<Semaphore>>,
    mut request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let permit = Arc::new(
        capacity
            .try_acquire_owned()
            .map_err(|_| AppError::too_many("server overloaded, please retry later"))?,
    );
    request.extensions_mut().insert(permit.clone());
    let response = next.run(request).await;
    // Handlers retain a clone through inference and SSE; failed extraction releases it here.
    drop(permit);
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Extension, Json, Router,
        body::Body,
        http::{Request, StatusCode},
        routing::post,
    };
    use tower::ServiceExt;

    fn app(capacity: Arc<Semaphore>) -> Router {
        Router::new()
            .route(
                "/",
                post(
                    |Extension(_permit): Extension<AdmissionPermit>,
                     Json(_): Json<serde_json::Value>| async { StatusCode::OK },
                ),
            )
            .layer(axum::middleware::from_fn_with_state(capacity, admit))
    }

    #[tokio::test]
    async fn overload_rejects_without_polling_body() {
        let capacity = Arc::new(Semaphore::new(1));
        let held = capacity.clone().acquire_owned().await.unwrap();
        let body = Body::from_stream(futures::stream::poll_fn(
            |_| -> std::task::Poll<Option<Result<axum::body::Bytes, std::io::Error>>> {
                panic!("overloaded request body must not be polled")
            },
        ));
        let response = app(capacity.clone())
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .body(body)
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(held);
        assert_eq!(capacity.available_permits(), 1);
    }

    #[tokio::test]
    async fn extraction_failure_and_success_release_capacity() {
        let capacity = Arc::new(Semaphore::new(1));
        for (body, expected) in [("invalid", StatusCode::BAD_REQUEST), ("{}", StatusCode::OK)] {
            let response = app(capacity.clone())
                .oneshot(
                    Request::post("/")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            assert_eq!(capacity.available_permits(), 1);
        }
    }

    #[tokio::test]
    async fn streaming_response_holds_capacity_until_body_is_dropped() {
        let capacity = Arc::new(Semaphore::new(1));
        let router = Router::new()
            .route(
                "/",
                post(|Extension(permit): Extension<AdmissionPermit>| async move {
                    Body::from_stream(async_stream::stream! {
                        let _permit = permit;
                        std::future::pending::<()>().await;
                        yield Ok::<_, std::io::Error>(axum::body::Bytes::new());
                    })
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                capacity.clone(),
                admit,
            ));
        let response = router
            .oneshot(Request::post("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(capacity.available_permits(), 0);
        drop(response);
        assert_eq!(capacity.available_permits(), 1);
    }

    #[tokio::test]
    async fn background_work_keeps_capacity_after_request_cancellation() {
        let capacity = Arc::new(Semaphore::new(1));
        let permit = Arc::new(capacity.clone().try_acquire_owned().unwrap());
        let background_permit = permit.clone();
        let (finish, wait) = std::sync::mpsc::channel();
        let task = tokio::task::spawn_blocking(move || {
            wait.recv().unwrap();
            drop(background_permit);
        });
        drop(permit);
        assert_eq!(capacity.available_permits(), 0);
        finish.send(()).unwrap();
        task.await.unwrap();
        assert_eq!(capacity.available_permits(), 1);
    }
}
