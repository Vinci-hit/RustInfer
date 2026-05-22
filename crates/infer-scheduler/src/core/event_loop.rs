//! Event loop — async select! based main loop.

use crate::core::engine::SchedulerEngine;
use crate::error::Result;
use crate::policy::traits::SchedulingPolicy;
use crate::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};

/// Run the main event loop.
///
/// The design uses a two-step approach to avoid borrow checker issues with
/// `tokio::select!` needing multiple `&mut` borrows:
/// 1. Poll transports with split borrows via `poll_event`
/// 2. Process the event with full `&mut self` access
pub async fn run_event_loop<P, F, W>(engine: &mut SchedulerEngine<P, F, W>) -> Result<()>
where
    P: SchedulingPolicy,
    F: FrontendTransport,
    W: WorkerTransport,
{
    tracing::info!("Event loop starting...");

    loop {
        // Wait for next event from either frontend or worker.
        let event = engine.poll_next_event().await;

        match event {
            EngineEvent::Frontend(result) => match *result {
                Ok(FrontendEvent::Infer { client_id, request }) => {
                    engine.handle_new_request(client_id, request);
                    if !engine.worker_busy() {
                        engine.run_iteration().await?;
                    }
                }
                Ok(FrontendEvent::Cancel { request_id, reason }) => {
                    tracing::debug!(
                        "Frontend cancel request_id={} reason={:?}",
                        request_id,
                        reason
                    );
                    engine.cancel_request(request_id).await?;
                    if !engine.worker_busy() {
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
        }
    }
}

/// Events that can occur in the scheduler loop.
pub(crate) enum EngineEvent {
    Frontend(Box<crate::error::Result<FrontendEvent>>),
    WorkerOutput(crate::error::Result<Vec<u8>>),
}
