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

    loop {
        let event = engine.poll_next_event(&mut decoded_rx).await;

        match event {
            SchedulerEvent::NewRequest { client_id, request } => {
                engine.handle_new_request(client_id, request);
                if engine.can_schedule() {
                    engine.run_iteration().await?;
                }
            }
            SchedulerEvent::Cancel { external_id, reason: _ } => {
                engine.cancel_request_by_external_id(&external_id).await?;
                if engine.can_schedule() {
                    engine.run_iteration().await?;
                }
            }
            SchedulerEvent::WorkerLlmStep(_) | SchedulerEvent::WorkerDiffusionStep(_) => {
                engine.handle_step_output(event).await?;
                engine.run_iteration().await?;
            }
            SchedulerEvent::ControlSignal(ev) => {
                engine.on_control_event(ev).await?;
                if engine.can_schedule() {
                    engine.run_iteration().await?;
                }
            }
            SchedulerEvent::FrontendShutdown => {
                tracing::info!("Frontend transport closed, shutting down");
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
