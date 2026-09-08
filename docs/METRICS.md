# Operational metrics

`GET /metrics` exports Prometheus text format (`text/plain; version=0.0.4`).
HTTP counters reset when the HTTP server restarts. Each server process has its own
registry; scrape every replica separately and let Prometheus attach `instance`
and deployment labels. The browser console uses `GET /metrics/system`, which
retains the JSON uptime/timestamp summary with actual process uptime.

Example Prometheus configuration:

```yaml
scrape_configs:
  - job_name: rustinfer
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8000"] # use the configured HTTP server port
```

| Metric | Type | Meaning |
| --- | --- | --- |
| `rustinfer_uptime_seconds` | Gauge | Time since server initialization, measured with a monotonic clock. |
| `rustinfer_scheduler_alive` | Gauge | 1 when a compatible scheduler heartbeat is fresh at scrape time, otherwise 0. |
| `rustinfer_scheduler_ready` | Gauge | 1 when the server considers the scheduler ready for inference, otherwise 0. |
| `rustinfer_scheduler_metrics_available` | Gauge | 1 when a fresh scheduler engine snapshot is available; otherwise 0 and the runtime series below are omitted. |
| `rustinfer_scheduler_requests{state}` | Gauge | Authoritative scheduler request table size: `queued`, `prefilling`, `decoding`, or `active` (the sum of all three). Queue depth excludes ZMQ/HTTP transport buffers. |
| `rustinfer_scheduler_kv_tokens{state}` | Gauge | KV budget in token slots: `used` is worker-reported allocation including cached prefixes; `pending` is projected in-flight prefill allocation not yet reported; `capacity` is the worker's KV pool capacity. These values do not measure GPU bytes. |
| `rustinfer_scheduler_requests_total` | Counter | Requests admitted to the scheduler table. |
| `rustinfer_scheduler_completions_total` | Counter | Normal LLM completions and terminal diffusion replies enqueued to frontend transport. Diffusion error replies are included; failed/cancelled LLM requests are excluded. This follows the scheduler's existing completion recorder. |
| `rustinfer_scheduler_output_tokens_total` | Counter | Output tokens of normally completed LLM requests, counted upon completion; excludes partial output from failed/cancelled requests. |
| `rustinfer_scheduler_completion_duration_seconds_total` | Counter | Sum of the scheduler lifetimes of recorded completions, converted from milliseconds. Divide its rate by the completion counter rate for mean completion latency. |
| `rustinfer_http_requests_total{route,method,status}` | Counter | Responses whose HTTP headers have been produced, including validation failures and overload responses. |
| `rustinfer_http_requests_in_flight{route,method}` | Gauge | Requests from middleware entry through response body completion, body error, or cancellation. SSE remains active after headers are sent. |
| `rustinfer_http_request_duration_seconds{route,method}` | Histogram | The same complete HTTP lifetime, including upload, tokenization, inference, and response backpressure. Observed once when the request ends, including cancelled requests. |
| `rustinfer_http_body_outcomes_total{route,method,outcome}` | Counter | Body lifecycle outcome: `complete`, `error`, `dropped` before body completion, or `cancelled` before response headers. This measures server body delivery, not peer acknowledgement. |
| `rustinfer_admission_rejections_total{reason}` | Counter | Capacity gate rejections: `inflight` for the global pre-body gate, `image` for image processing capacity. Other 429 responses remain visible in HTTP status counters. |
| `rustinfer_stream_first_content_seconds{route}` | Histogram | Streaming requests only: middleware entry to the first nonempty decoded content yielded to SSE. Includes upload, tokenization, queueing, and model execution; excludes role-only and keepalive events. Requests with no content produce no sample. |
| `rustinfer_streams_total{route,outcome}` | Counter | Logical SSE inference outcomes: `completed` for a successful terminal reply, `error` for a decoder/engine/channel failure, `cancelled` when dropped before a terminal outcome. |

Histogram buckets are 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5,
10, 30, 60, 120, and 300 seconds, plus `+Inf`. Histograms expose the standard
`_bucket`, `_sum`, and `_count` series. Request-dependent series appear after
the corresponding event is first observed.

Routes are a fixed set of registered API/probe paths; unrecognized paths share
`route="unmatched"`. Unknown HTTP methods share `method="OTHER"`. Request IDs,
prompts, model paths, and arbitrary URLs never become labels. `/metrics` and
`/metrics/system` are excluded from request metrics to keep scraping and console
polling from changing traffic counts; health/readiness probes have separate
route labels. When adding a route, update the bounded route mapping in the
metrics middleware.

An HTTP 200 stream can later fail. Use `rustinfer_streams_total{outcome="error"}`
alongside HTTP status rates. First-content time is a server observation of text
delivery, not GPU-only TTFT or client-measured network latency. Nonstreaming
latency is covered by the HTTP duration histogram. Queue time, inter-token time,
GPU memory, and CUDA graph cache metrics are not exported yet: they
require explicit worker/scheduler telemetry, rather than estimated values at the
HTTP boundary.

Scheduler snapshots travel on the existing Ping/Pong channel. The event loop
refreshes its snapshot before waiting for the next event (including a one-second
idle tick); Pongs arrive approximately every three seconds. These are sampled
gauges, so short queue spikes may not be visible. A stalled engine or stale,
incompatible scheduler connection invalidates the snapshot. Check
`rustinfer_scheduler_metrics_available` before treating missing series as zero.
Scheduler counters retain their absolute values across HTTP server restarts and
reset when the scheduler restarts; if scheduler metric recording is disabled,
the counters are omitted while request/KV gauges remain available. Multiple HTTP
servers attached to the same scheduler export the same scheduler counters:
deduplicate by scheduler in your deployment scrape configuration before summing.

Example queries:

```promql
# Request rate by HTTP status (inference routes only).
sum by (status) (rate(rustinfer_http_requests_total{route=~"/v1/.*completions"}[5m]))

# p95 server first-content latency, by endpoint.
histogram_quantile(0.95, sum by (route, le) (rate(rustinfer_stream_first_content_seconds_bucket[5m])))

# Server capacity rejections per second.
sum by (reason) (rate(rustinfer_admission_rejections_total[5m]))

# Logical SSE failures per second, including failures after HTTP 200.
sum(rate(rustinfer_streams_total{outcome="error"}[5m]))
```
