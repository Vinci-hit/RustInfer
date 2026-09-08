//! Observe the complete response body lifecycle, including SSE and disconnections.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    body::{Body, Bytes},
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use http_body::{Body as HttpBody, Frame, SizeHint};

use crate::metrics::{HttpMetrics, RequestMetrics};

pub async fn observe(
    State(registry): State<Arc<HttpMetrics>>,
    mut request: Request,
    next: Next,
) -> Response {
    let matched = request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str);
    let route = match matched {
        Some("/metrics" | "/metrics/system") => return next.run(request).await,
        Some("/v1/chat/completions") => "/v1/chat/completions",
        Some("/v1/completions") => "/v1/completions",
        Some("/v1/models") => "/v1/models",
        Some("/health") => "/health",
        Some("/ready") => "/ready",
        _ => "unmatched",
    };
    let method = match request.method().as_str() {
        "GET" => "GET",
        "POST" => "POST",
        "PUT" => "PUT",
        "DELETE" => "DELETE",
        "PATCH" => "PATCH",
        "HEAD" => "HEAD",
        "OPTIONS" => "OPTIONS",
        _ => "OTHER",
    };
    let observation = RequestMetrics {
        registry: registry.clone(),
        route,
        started: Instant::now(),
    };
    registry.inflight.with_label_values(&[route, method]).inc();
    let mut guard = RequestGuard {
        observation: observation.clone(),
        method,
        outcome: "cancelled",
        finished: false,
    };
    request.extensions_mut().insert(observation);
    let response = next.run(request).await;
    registry
        .requests
        .with_label_values(&[route, method, response.status().as_str()])
        .inc();
    let (parts, body) = response.into_parts();
    guard.outcome = "dropped";
    if body.is_end_stream() {
        guard.finish("complete");
    }
    Response::from_parts(parts, Body::new(ObservedBody { inner: body, guard }))
}

struct RequestGuard {
    observation: RequestMetrics,
    method: &'static str,
    outcome: &'static str,
    finished: bool,
}

impl RequestGuard {
    fn finish(&mut self, outcome: &'static str) {
        if !self.finished {
            self.finished = true;
            let registry = &self.observation.registry;
            let labels = &[self.observation.route, self.method];
            registry.inflight.with_label_values(labels).dec();
            registry
                .duration
                .with_label_values(labels)
                .observe(self.observation.started.elapsed().as_secs_f64());
            registry
                .body_outcomes
                .with_label_values(&[self.observation.route, self.method, outcome])
                .inc();
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.finish(self.outcome);
    }
}

struct ObservedBody {
    inner: Body,
    guard: RequestGuard,
}

impl HttpBody for ObservedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let result = Pin::new(&mut self.inner).poll_frame(cx);
        match &result {
            Poll::Ready(None) => self.guard.finish("complete"),
            Poll::Ready(Some(Err(_))) => self.guard.finish("error"),
            Poll::Ready(Some(Ok(_))) if self.inner.is_end_stream() => self.guard.finish("complete"),
            _ => {}
        }
        result
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::to_bytes,
        http::StatusCode,
        routing::{get, post},
    };
    use tower::ServiceExt;

    fn instrument(router: Router, registry: Arc<HttpMetrics>) -> Router {
        router.layer(axum::middleware::from_fn_with_state(registry, observe))
    }

    #[tokio::test]
    async fn streaming_inflight_lasts_until_body_drop() {
        let registry = Arc::new(HttpMetrics::new().unwrap());
        let router = instrument(
            Router::new().route(
                "/v1/completions",
                post(|| async {
                    Body::from_stream(futures::stream::pending::<Result<Bytes, std::io::Error>>())
                }),
            ),
            registry.clone(),
        );
        let response = router
            .oneshot(
                Request::post("/v1/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let labels = &["/v1/completions", "POST"];
        assert_eq!(registry.inflight.with_label_values(labels).get(), 1);
        assert_eq!(
            registry
                .duration
                .with_label_values(labels)
                .get_sample_count(),
            0
        );
        assert_eq!(
            registry
                .requests
                .with_label_values(&["/v1/completions", "POST", "200"])
                .get(),
            1
        );
        drop(response);
        assert_eq!(registry.inflight.with_label_values(labels).get(), 0);
        assert_eq!(
            registry
                .duration
                .with_label_values(labels)
                .get_sample_count(),
            1
        );
        assert_eq!(
            registry
                .body_outcomes
                .with_label_values(&["/v1/completions", "POST", "dropped"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn body_completion_and_body_error_are_recorded_once() {
        let registry = Arc::new(HttpMetrics::new().unwrap());
        let router = instrument(
            Router::new()
                .route("/health", get(|| async { "ok" }))
                .route(
                    "/ready",
                    get(|| async {
                        Body::from_stream(futures::stream::once(async {
                            Err::<Bytes, _>(std::io::Error::other("broken stream"))
                        }))
                    }),
                ),
            registry.clone(),
        );
        let response = router
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(to_bytes(response.into_body(), 1024).await.unwrap(), "ok");
        let response = router
            .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert!(to_bytes(response.into_body(), 1024).await.is_err());
        for (route, outcome) in [("/health", "complete"), ("/ready", "error")] {
            assert_eq!(
                registry.inflight.with_label_values(&[route, "GET"]).get(),
                0
            );
            assert_eq!(
                registry
                    .duration
                    .with_label_values(&[route, "GET"])
                    .get_sample_count(),
                1
            );
            assert_eq!(
                registry
                    .body_outcomes
                    .with_label_values(&[route, "GET", outcome])
                    .get(),
                1
            );
        }
    }

    #[tokio::test]
    async fn cancellation_before_headers_releases_inflight() {
        let registry = Arc::new(HttpMetrics::new().unwrap());
        let entered = Arc::new(tokio::sync::Notify::new());
        let notify = entered.clone();
        let router = instrument(
            Router::new().route(
                "/v1/completions",
                post(move || async move {
                    notify.notify_one();
                    std::future::pending::<StatusCode>().await
                }),
            ),
            registry.clone(),
        );
        let task = tokio::spawn(
            router.oneshot(
                Request::post("/v1/completions")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        entered.notified().await;
        assert_eq!(
            registry
                .inflight
                .with_label_values(&["/v1/completions", "POST"])
                .get(),
            1
        );
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(
            registry
                .inflight
                .with_label_values(&["/v1/completions", "POST"])
                .get(),
            0
        );
        assert_eq!(
            registry
                .body_outcomes
                .with_label_values(&["/v1/completions", "POST", "cancelled"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn overload_counts_rejection_before_reading_body() {
        let registry = Arc::new(HttpMetrics::new().unwrap());
        let router = instrument(
            Router::new().route(
                "/v1/completions",
                post(|| async { StatusCode::OK }).layer(axum::middleware::from_fn_with_state(
                    Arc::new(tokio::sync::Semaphore::new(0)),
                    crate::middleware::admission::admit,
                )),
            ),
            registry.clone(),
        );
        let body = Body::from_stream(futures::stream::poll_fn(
            |_| -> Poll<Option<Result<Bytes, std::io::Error>>> {
                panic!("overloaded body must not be read");
            },
        ));
        let response = router
            .oneshot(Request::post("/v1/completions").body(body).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        to_bytes(response.into_body(), 1024).await.unwrap();
        let encoded = String::from_utf8(registry.encode().unwrap()).unwrap();
        assert!(encoded.contains("rustinfer_admission_rejections_total{reason=\"inflight\"} 1"));
        assert_eq!(
            registry
                .requests
                .with_label_values(&["/v1/completions", "POST", "429"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn scrape_is_excluded_and_unknown_paths_and_methods_are_bounded() {
        let registry = Arc::new(HttpMetrics::new().unwrap());
        let router = instrument(
            Router::new().route("/metrics", get(|| async { "metrics" })),
            registry.clone(),
        );
        for (method, path) in [
            ("GET", "/metrics"),
            ("CUSTOMONE", "/user/one"),
            ("CUSTOMTWO", "/user/two"),
        ] {
            let request = Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            to_bytes(response.into_body(), 1024).await.unwrap();
        }
        assert_eq!(
            registry
                .requests
                .with_label_values(&["unmatched", "OTHER", "404"])
                .get(),
            2
        );
        let encoded = String::from_utf8(registry.encode().unwrap()).unwrap();
        for forbidden in [
            "CUSTOMONE",
            "CUSTOMTWO",
            "/user/one",
            "/user/two",
            "route=\"/metrics\"",
        ] {
            assert!(
                !encoded.contains(forbidden),
                "unexpected label: {forbidden}"
            );
        }
    }
}
