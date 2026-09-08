//! Per-server Prometheus registry. Labels contain bounded operational categories only.

use std::{sync::Arc, time::Instant};

use infer_protocol::scheduler_to_server::SchedulerMetricsSnapshot;
use prometheus::{
    Encoder, Gauge, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
    TextEncoder,
};

pub struct HttpMetrics {
    started: Instant,
    registry: Registry,
    uptime: Gauge,
    scheduler_alive: Gauge,
    scheduler_ready: Gauge,
    pub(crate) requests: IntCounterVec,
    pub(crate) inflight: IntGaugeVec,
    pub(crate) duration: HistogramVec,
    pub(crate) body_outcomes: IntCounterVec,
    admission_rejections: IntCounterVec,
    first_content: HistogramVec,
    streams: IntCounterVec,
}

impl HttpMetrics {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        let uptime = Gauge::new(
            "rustinfer_uptime_seconds",
            "HTTP server process uptime in seconds",
        )?;
        let scheduler_alive = Gauge::new(
            "rustinfer_scheduler_alive",
            "Whether a compatible scheduler heartbeat is fresh at scrape time",
        )?;
        let scheduler_ready = Gauge::new(
            "rustinfer_scheduler_ready",
            "Whether the scheduler is ready for inference at scrape time",
        )?;
        let requests = IntCounterVec::new(
            Opts::new(
                "rustinfer_http_requests_total",
                "HTTP responses by header status; streaming errors after headers are counted by rustinfer_streams_total",
            ),
            &["route", "method", "status"],
        )?;
        let inflight = IntGaugeVec::new(
            Opts::new(
                "rustinfer_http_requests_in_flight",
                "HTTP requests active through response body completion or disconnect",
            ),
            &["route", "method"],
        )?;
        let buckets = vec![
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ];
        let duration = HistogramVec::new(
            HistogramOpts::new("rustinfer_http_request_duration_seconds", "Time from middleware entry through response body completion, error, or cancellation").buckets(buckets.clone()),
            &["route", "method"],
        )?;
        let body_outcomes = IntCounterVec::new(
            Opts::new(
                "rustinfer_http_body_outcomes_total",
                "HTTP lifecycles ending in complete, error, dropped body, or cancellation before headers",
            ),
            &["route", "method", "outcome"],
        )?;
        let admission_rejections = IntCounterVec::new(
            Opts::new(
                "rustinfer_admission_rejections_total",
                "Requests rejected by server capacity gates",
            ),
            &["reason"],
        )?;
        let first_content = HistogramVec::new(
            HistogramOpts::new("rustinfer_stream_first_content_seconds", "Time from HTTP middleware entry to first decoded nonempty SSE content, excluding role and keepalive events").buckets(buckets),
            &["route"],
        )?;
        let streams = IntCounterVec::new(
            Opts::new(
                "rustinfer_streams_total",
                "SSE inference streams by logical completed, error, or cancelled outcome",
            ),
            &["route", "outcome"],
        )?;
        registry.register(Box::new(uptime.clone()))?;
        registry.register(Box::new(scheduler_alive.clone()))?;
        registry.register(Box::new(scheduler_ready.clone()))?;
        registry.register(Box::new(requests.clone()))?;
        registry.register(Box::new(inflight.clone()))?;
        registry.register(Box::new(duration.clone()))?;
        registry.register(Box::new(body_outcomes.clone()))?;
        registry.register(Box::new(admission_rejections.clone()))?;
        registry.register(Box::new(first_content.clone()))?;
        registry.register(Box::new(streams.clone()))?;
        for reason in ["inflight", "image"] {
            admission_rejections.with_label_values(&[reason]);
        }
        Ok(Self {
            started: Instant::now(),
            registry,
            uptime,
            scheduler_alive,
            scheduler_ready,
            requests,
            inflight,
            duration,
            body_outcomes,
            admission_rejections,
            first_content,
            streams,
        })
    }

    pub fn uptime_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn encode(&self) -> Result<Vec<u8>, prometheus::Error> {
        self.encode_with_scheduler(None)
    }

    /// Export absolute scheduler counters, including counter resets on scheduler restart.
    pub fn encode_with_scheduler(
        &self,
        scheduler: Option<&SchedulerMetricsSnapshot>,
    ) -> Result<Vec<u8>, prometheus::Error> {
        self.uptime.set(self.uptime_seconds());
        let mut families = self.registry.gather();
        families.extend(scheduler_families(scheduler));
        let mut bytes = Vec::new();
        TextEncoder::new().encode(&families, &mut bytes)?;
        Ok(bytes)
    }

    pub(crate) fn set_scheduler_health(&self, alive: bool, ready: bool) {
        self.scheduler_alive.set(u8::from(alive) as f64);
        self.scheduler_ready.set(u8::from(ready) as f64);
    }

    pub(crate) fn reject_inflight(&self) {
        self.admission_rejections
            .with_label_values(&["inflight"])
            .inc();
    }

    pub(crate) fn reject_image(&self) {
        self.admission_rejections
            .with_label_values(&["image"])
            .inc();
    }
}

/// Construct families directly so unavailable samples disappear instead of keeping
/// stale gauge values, and a scheduler restart can reset its counters independently.
fn scheduler_families(
    snapshot: Option<&SchedulerMetricsSnapshot>,
) -> Vec<prometheus::proto::MetricFamily> {
    use prometheus::proto::{Counter, Gauge, LabelPair, Metric, MetricFamily, MetricType};

    fn family(name: &str, help: &str, kind: MetricType, values: &[(&str, f64)]) -> MetricFamily {
        let mut family = MetricFamily::new();
        family.set_name(name.into());
        family.set_help(help.into());
        family.set_field_type(kind);
        let metrics = values
            .iter()
            .map(|(state, value)| {
                let mut metric = Metric::new();
                if !state.is_empty() {
                    let mut label = LabelPair::new();
                    label.set_name("state".into());
                    label.set_value((*state).into());
                    metric.set_label(vec![label]);
                }
                if kind == MetricType::COUNTER {
                    let mut counter = Counter::new();
                    counter.set_value(*value);
                    metric.set_counter(counter);
                } else {
                    let mut gauge = Gauge::new();
                    gauge.set_value(*value);
                    metric.set_gauge(gauge);
                }
                metric
            })
            .collect();
        family.set_metric(metrics);
        family
    }

    let mut families = vec![family(
        "rustinfer_scheduler_metrics_available",
        "Whether a fresh engine runtime snapshot is available",
        MetricType::GAUGE,
        &[("", u8::from(snapshot.is_some()) as f64)],
    )];
    let Some(snapshot) = snapshot else {
        return families;
    };
    families.push(family(
        "rustinfer_scheduler_requests",
        "Requests in the scheduler table; active includes queued, prefilling, and decoding",
        MetricType::GAUGE,
        &[
            ("queued", snapshot.queued_requests as f64),
            ("active", snapshot.active_requests as f64),
            ("prefilling", snapshot.prefilling_requests as f64),
            ("decoding", snapshot.decoding_requests as f64),
        ],
    ));
    families.push(family("rustinfer_scheduler_kv_tokens", "Scheduler KV token-slot budget: worker-reported used slots, projected pending prefill, and total capacity", MetricType::GAUGE, &[
        ("used", snapshot.kv_tokens_used as f64),
        ("pending", snapshot.kv_tokens_pending as f64),
        ("capacity", snapshot.kv_tokens_capacity as f64),
    ]));
    if snapshot.metrics_enabled {
        for (name, help, value) in [
            (
                "rustinfer_scheduler_requests_total",
                "Requests admitted into the scheduler table",
                snapshot.total_requests as f64,
            ),
            (
                "rustinfer_scheduler_completions_total",
                "Normal LLM completions and terminal diffusion replies enqueued to frontend transport",
                snapshot.total_completions as f64,
            ),
            (
                "rustinfer_scheduler_output_tokens_total",
                "Output tokens on normal completed LLM requests; excludes partial failed or cancelled requests",
                snapshot.total_tokens_generated as f64,
            ),
            (
                "rustinfer_scheduler_completion_duration_seconds_total",
                "Summed scheduler lifetime of recorded completions",
                snapshot.total_latency_ms as f64 / 1000.0,
            ),
        ] {
            families.push(family(name, help, MetricType::COUNTER, &[("", value)]));
        }
    }
    families
}

/// Installed before extraction so first-content timing includes upload and tokenization.
#[derive(Clone)]
pub struct RequestMetrics {
    pub(crate) registry: Arc<HttpMetrics>,
    pub(crate) route: &'static str,
    pub(crate) started: Instant,
}

impl RequestMetrics {
    pub(crate) fn start_stream(self) -> StreamMetrics {
        StreamMetrics {
            request: self,
            first_content: false,
            finished: false,
        }
    }
}

/// Records a logical stream outcome even when its generator is never polled.
pub(crate) struct StreamMetrics {
    request: RequestMetrics,
    first_content: bool,
    finished: bool,
}

impl StreamMetrics {
    pub(crate) fn first_content(&mut self) {
        if !self.first_content {
            self.first_content = true;
            self.request
                .registry
                .first_content
                .with_label_values(&[self.request.route])
                .observe(self.request.started.elapsed().as_secs_f64());
        }
    }

    pub(crate) fn finish(&mut self, success: bool) {
        self.record(if success { "completed" } else { "error" });
    }

    fn record(&mut self, outcome: &str) {
        if !self.finished {
            self.finished = true;
            self.request
                .registry
                .streams
                .with_label_values(&[self.request.route, outcome])
                .inc();
        }
    }
}

impl Drop for StreamMetrics {
    fn drop(&mut self) {
        self.record("cancelled");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_record_first_content_once_and_distinguish_terminal_outcomes() {
        let registry = Arc::new(HttpMetrics::new().unwrap());
        let request = RequestMetrics {
            registry: registry.clone(),
            route: "/v1/completions",
            started: Instant::now(),
        };
        let mut completed = request.clone().start_stream();
        completed.first_content();
        completed.first_content();
        completed.finish(true);
        drop(completed);
        let mut failed = request.clone().start_stream();
        failed.finish(false);
        drop(failed);
        drop(request.start_stream());
        assert_eq!(
            registry
                .first_content
                .with_label_values(&["/v1/completions"])
                .get_sample_count(),
            1
        );
        for outcome in ["completed", "error", "cancelled"] {
            assert_eq!(
                registry
                    .streams
                    .with_label_values(&["/v1/completions", outcome])
                    .get(),
                1
            );
        }
    }

    #[test]
    fn metrics_registry_is_per_instance_and_uptime_advances() {
        let metrics = HttpMetrics::new().unwrap();
        let another = HttpMetrics::new().unwrap();
        metrics.reject_inflight();
        assert!(metrics.uptime_seconds() > 0.0);
        let encoded = String::from_utf8(metrics.encode().unwrap()).unwrap();
        assert!(encoded.contains("# TYPE rustinfer_uptime_seconds gauge"));
        assert!(encoded.contains("rustinfer_admission_rejections_total{reason=\"inflight\"} 1"));
        assert_eq!(
            another
                .admission_rejections
                .with_label_values(&["inflight"])
                .get(),
            0
        );
    }

    #[test]
    fn scheduler_exports_disappear_when_stale_and_preserve_counter_resets() {
        let metrics = HttpMetrics::new().unwrap();
        let mut snapshot = SchedulerMetricsSnapshot {
            metrics_enabled: true,
            queued_requests: 2,
            active_requests: 5,
            prefilling_requests: 1,
            decoding_requests: 2,
            kv_tokens_used: 64,
            kv_tokens_pending: 16,
            kv_tokens_capacity: 128,
            total_requests: 9,
            total_completions: 4,
            total_tokens_generated: 80,
            total_latency_ms: 1250,
        };
        let encoded =
            String::from_utf8(metrics.encode_with_scheduler(Some(&snapshot)).unwrap()).unwrap();
        assert!(encoded.contains("rustinfer_scheduler_requests{state=\"queued\"} 2"));
        assert!(encoded.contains("rustinfer_scheduler_kv_tokens{state=\"pending\"} 16"));
        assert!(encoded.contains("# TYPE rustinfer_scheduler_requests_total counter"));
        assert!(encoded.contains("rustinfer_scheduler_requests_total 9"));
        assert!(encoded.contains("rustinfer_scheduler_completion_duration_seconds_total 1.25"));
        let stale = String::from_utf8(metrics.encode_with_scheduler(None).unwrap()).unwrap();
        assert!(stale.contains("rustinfer_scheduler_metrics_available 0"));
        assert!(!stale.contains("rustinfer_scheduler_kv_tokens"));
        assert!(!stale.contains("rustinfer_scheduler_requests_total"));
        snapshot.total_requests = 1;
        let restarted =
            String::from_utf8(metrics.encode_with_scheduler(Some(&snapshot)).unwrap()).unwrap();
        assert!(restarted.contains("rustinfer_scheduler_requests_total 1\n"));
        snapshot.metrics_enabled = false;
        let disabled =
            String::from_utf8(metrics.encode_with_scheduler(Some(&snapshot)).unwrap()).unwrap();
        assert!(disabled.contains("rustinfer_scheduler_kv_tokens"));
        assert!(!disabled.contains("rustinfer_scheduler_requests_total"));
    }
}
