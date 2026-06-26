//! Event loop — async select! based main loop.
//!
//! The event loop is mode-agnostic — it delegates to `EngineWorkflow`
//! for all scheduling and output processing decisions. LLM vs Diffusion
//! logic lives entirely in `LlmWorkflow` / `DiffusionWorkflow`.
//!
//! ## Background decode
//!
//! Worker output arrives as already-decoded `SchedulerEvent` variants
//! through the `decoded_rx` channel. MsgPack deserialization runs in
//! a dedicated background tokio task (see `engine.rs`), so the main
//! loop never blocks on decode.

use tokio::sync::mpsc;

use crate::application::engine::SchedulerEngine;
use crate::application::scheduler_event::SchedulerEvent;
use crate::error::Result;

/// Run the main event loop.
///
/// `decoded_rx` carries `SchedulerEvent` variants produced by the
/// background decode task (worker output already deserialized).
pub async fn run_event_loop(
    engine: &mut SchedulerEngine,
    mut decoded_rx: mpsc::UnboundedReceiver<SchedulerEvent>,
) -> Result<()> {
    tracing::info!("Event loop starting...");

    // Spawn periodic metrics summary (every 5s)
    use tokio::time::{Duration, interval};
    let metrics_for_summary = engine.metrics_handle();
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(5));
        ticker.tick().await; // skip first immediate tick
        let mut last = metrics_for_summary.snapshot();
        loop {
            ticker.tick().await;
            let current = metrics_for_summary.snapshot();
            let elapsed = 5.0; // interval is fixed at 5s
            let delta = current.delta(&last);
            if delta.completions > 0 {
                tracing::info!(
                    "Summary (last 5s): {} reqs, {} tokens, throughput: {:.1} tok/s, avg latency: {:.1} ms",
                    delta.completions,
                    delta.tokens_generated,
                    delta.throughput_tps(elapsed),
                    delta.avg_latency_ms(),
                );
            }
            last = current;
        }
    });

    loop {
        let event = engine.poll_next_event(&mut decoded_rx).await;

        match event {
            SchedulerEvent::NewRequest { client_id, request } => {
                let _t = std::time::Instant::now();
                engine.handle_new_request(client_id, request);
                engine.maybe_schedule().await?;
                if std::env::var_os("RUSTINFER_SCHED_TRACE").is_some() {
                    tracing::info!(us = _t.elapsed().as_micros() as u64, "SCHED_TRACE: NewRequest->dispatch");
                }
            }
            SchedulerEvent::Cancel {
                external_id,
                reason: _,
            } => {
                engine.cancel_request_by_external_id(&external_id).await?;
                engine.maybe_schedule().await?;
            }
            SchedulerEvent::WorkerLlmStep(_) | SchedulerEvent::WorkerDiffusionStep(_) => {
                let _t = std::time::Instant::now();
                engine.handle_step_output(event).await?;
                engine.maybe_schedule().await?;
                if std::env::var_os("RUSTINFER_SCHED_TRACE").is_some() {
                    tracing::info!(us = _t.elapsed().as_micros() as u64, "SCHED_TRACE: StepOutput->forward");
                }
            }
            SchedulerEvent::ControlSignal(ev) => {
                engine.on_control_event(ev).await?;
                engine.maybe_schedule().await?;
            }
            SchedulerEvent::BatchTimer => {
                engine.on_batch_timer().await?;
            }
            SchedulerEvent::FrontendShutdown => {
                tracing::info!("Frontend transport closed, shutting down");
                engine.shutdown_worker_best_effort();
                return Ok(());
            }
            SchedulerEvent::WorkerShutdown => {
                tracing::info!("Worker transport closed, shutting down");
                return Ok(());
            }
            SchedulerEvent::WorkerDecodeError(msg) => {
                tracing::error!("Worker decode error: {}", msg);
            }
            SchedulerEvent::FrontendError(msg) => {
                tracing::error!("Frontend recv error: {}", msg);
            }
        }
    }
}
