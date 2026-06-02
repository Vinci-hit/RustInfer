//! Event loop — async select! based main loop.
//!
//! The design uses a two-step approach to avoid borrow checker issues with
//! `tokio::select!` needing multiple `&mut` borrows:
//! 1. Poll transports with split borrows via `poll_next_event`
//! 2. Process the event with full `&mut self` access
//!
//! The event loop is mode-agnostic — it delegates to `EngineWorkflow`
//! for all scheduling and output processing decisions. LLM vs Diffusion
//! logic lives entirely in `LlmWorkflow` / `DiffusionWorkflow`.

use crate::application::engine::SchedulerEngine;
use crate::error::Result;
use crate::infrastructure::transport::control_plane::ControlEvent;
use crate::infrastructure::transport::traits::FrontendEvent;

/// Run the main event loop.
pub async fn run_event_loop(engine: &mut SchedulerEngine) -> Result<()> {
    tracing::info!("Event loop starting...");

    loop {
        let event = engine.poll_next_event().await;

        match event {
            EngineEvent::Frontend(result) => match *result {
                Ok(FrontendEvent::Infer { client_id, request }) => {
                    engine.handle_new_request(client_id, request);
                    if engine.can_schedule() {
                        engine.run_iteration().await?;
                    }
                }
                Ok(FrontendEvent::Cancel { external_id, reason }) => {
                    tracing::debug!(
                        "Frontend cancel external_id={} reason={:?}",
                        external_id,
                        reason
                    );
                    engine.cancel_request_by_external_id(&external_id).await?;
                    if engine.can_schedule() {
                        engine.run_iteration().await?;
                    }
                }
                Err(crate::error::SchedulerError::Shutdown) => {
                    tracing::info!("Frontend transport closed, shutting down");
                    return Ok(());
                }
                Err(e) => {
                    tracing::error!("Frontend recv error: {}", e);
                }
            },
            EngineEvent::WorkerOutput(Ok(data)) => {
                engine.handle_step_output(data).await?;
                engine.run_iteration().await?;
            }
            EngineEvent::WorkerOutput(Err(crate::error::SchedulerError::Shutdown)) => {
                tracing::info!("Worker transport closed, shutting down");
                return Ok(());
            }
            EngineEvent::WorkerOutput(Err(e)) => {
                tracing::error!("Worker recv error: {}", e);
            }
            EngineEvent::Control(ev) => {
                engine.on_control_event(ev).await?;
                if engine.can_schedule() {
                    engine.run_iteration().await?;
                }
            }
        }
    }
}

/// Events that can occur in the scheduler loop.
pub(crate) enum EngineEvent {
    Frontend(Box<crate::error::Result<FrontendEvent>>),
    WorkerOutput(crate::error::Result<Vec<u8>>),
    Control(ControlEvent),
}
