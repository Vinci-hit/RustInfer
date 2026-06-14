//! `DispatchSystem` — owns the two transports.
//!
//! Holds `Box<dyn FrontendTransport>` + `Box<dyn WorkerTransport>` so
//! the engine itself stays free of `<F, W>` generics. Bundling both
//! transports into a single System keeps the borrow shape simple:
//! `output_fns` takes `&mut dyn FrontendTransport`, the
//! engine's iteration loop takes `&mut dyn WorkerTransport`. Either
//! can be served by `dispatch.frontend_mut()` / `worker_mut()`
//! without aliasing.
//!
//! ## Public surface
//!
//! - [`DispatchSystem::new`] — construct from boxed transports
//! - [`DispatchSystem::frontend_mut`] / [`DispatchSystem::worker_mut`]
//!   — borrow access for systems that drive IO themselves
//!   (`output_fns`, the engine's `run_iteration`).
//! - [`DispatchSystem::send_batch`] — convenience wrapper used by
//!   the engine after `PlanningSystem::build_*_batch` produces wire
//!   bytes, so the iteration code stays a one-liner.

use crate::error::Result;
use crate::infrastructure::transport::traits::{FrontendTransport, WorkerTransport};

/// Outbound IO + transport-poll stage.
pub struct DispatchSystem {
    frontend: Box<dyn FrontendTransport>,
    worker: Box<dyn WorkerTransport>,
}

impl DispatchSystem {
    pub fn new(frontend: Box<dyn FrontendTransport>, worker: Box<dyn WorkerTransport>) -> Self {
        Self { frontend, worker }
    }

    /// Borrow the frontend transport. `output_fns` uses
    /// this to send responses / stream chunks during error and
    /// completion paths.
    pub fn frontend_mut(&mut self) -> &mut dyn FrontendTransport {
        &mut *self.frontend
    }

    /// Borrow the worker transport. The engine's `run_iteration`
    /// sends serialized batch commands through it.
    pub fn worker_mut(&mut self) -> &mut dyn WorkerTransport {
        &mut *self.worker
    }

    /// Send a serialized batch command to the worker.
    pub async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
        self.worker.send_batch(cmd).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::inference_session::handle::ClientId;
    use async_trait::async_trait;
    use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
    use std::sync::{Arc, Mutex};

    #[derive(Default, Clone)]
    struct CapturingFrontend {
        responses: Arc<Mutex<Vec<InferenceResponse>>>,
        chunks: Arc<Mutex<Vec<StreamChunk>>>,
    }
    #[async_trait]
    impl FrontendTransport for CapturingFrontend {
        async fn recv_event(
            &mut self,
        ) -> Result<crate::infrastructure::transport::traits::FrontendEvent> {
            Err(crate::error::SchedulerError::Shutdown)
        }
        async fn send_response(&mut self, _: &ClientId, r: InferenceResponse) -> Result<()> {
            self.responses.lock().unwrap().push(r);
            Ok(())
        }
        async fn send_stream_chunk(&mut self, _: &ClientId, c: StreamChunk) -> Result<()> {
            self.chunks.lock().unwrap().push(c);
            Ok(())
        }
    }

    #[derive(Default, Clone)]
    struct CapturingWorker {
        sent: Arc<Mutex<Vec<Vec<u8>>>>,
    }
    #[async_trait]
    impl WorkerTransport for CapturingWorker {
        async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
            self.sent.lock().unwrap().push(cmd);
            Ok(())
        }
        async fn recv_step_output(&mut self) -> Result<Vec<u8>> {
            Err(crate::error::SchedulerError::Shutdown)
        }
    }

    #[tokio::test]
    async fn send_batch_routes_to_worker_transport() {
        let worker = CapturingWorker::default();
        let dispatch = DispatchSystem::new(
            Box::new(CapturingFrontend::default()),
            Box::new(worker.clone()),
        );
        let mut dispatch = dispatch;
        dispatch.send_batch(vec![1, 2, 3]).await.unwrap();
        assert_eq!(worker.sent.lock().unwrap().clone(), vec![vec![1, 2, 3]]);
    }
}
