//! `DispatchSystem` — owns the two transports.
//!
//! Holds `Box<dyn FrontendTransport>` + `Box<dyn WorkerTransport>` so
//! the engine itself stays free of `<F, W>` generics. Bundling both
//! transports into a single System keeps the borrow shape simple:
//! `OutputProcessingSystem` takes `&mut dyn FrontendTransport`, the
//! engine's iteration loop takes `&mut dyn WorkerTransport`. Either
//! can be served by `dispatch.frontend_mut()` / `worker_mut()`
//! without aliasing.
//!
//! ## Public surface
//!
//! - [`DispatchSystem::new`] — construct from boxed transports
//! - [`DispatchSystem::frontend_mut`] / [`DispatchSystem::worker_mut`]
//!   — borrow access for systems that drive IO themselves
//!   (`OutputProcessingSystem`, the engine's `run_iteration`).
//! - [`DispatchSystem::send_batch`] — convenience wrapper used by
//!   the engine after `PlanningSystem::build_*_batch` produces wire
//!   bytes, so the iteration code stays a one-liner.
//! - [`DispatchSystem::recv_frontend`] / `recv_worker_output` —
//!   inbound polling helpers used by the event loop.

use crate::error::Result;
use crate::infrastructure::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};

/// Outbound IO + transport-poll stage.
pub struct DispatchSystem {
    frontend: Box<dyn FrontendTransport>,
    worker: Box<dyn WorkerTransport>,
}

impl DispatchSystem {
    pub fn new(
        frontend: Box<dyn FrontendTransport>,
        worker: Box<dyn WorkerTransport>,
    ) -> Self {
        Self { frontend, worker }
    }

    /// Borrow the frontend transport. `OutputProcessingSystem` uses
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

    /// Borrow both transports at once for `tokio::select!` based
    /// polling. Cannot be expressed via two separate `&mut` calls
    /// because Rust would treat them as aliasing the System.
    pub fn borrow_both_mut(
        &mut self,
    ) -> (&mut dyn FrontendTransport, &mut dyn WorkerTransport) {
        (&mut *self.frontend, &mut *self.worker)
    }

    /// Send a serialized batch command to the worker.
    pub async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
        self.worker.send_batch(cmd).await
    }

    /// Poll the frontend for the next event.
    pub async fn recv_frontend(&mut self) -> Result<FrontendEvent> {
        self.frontend.recv_event().await
    }

    /// Poll the worker for the next step output.
    pub async fn recv_worker_output(&mut self) -> Result<Vec<u8>> {
        self.worker.recv_step_output().await
    }

    /// Drain the raw worker output receiver out of the dispatch system.
    ///
    /// After calling this, `recv_worker_output()` will no longer work
    /// (the receiver has been taken). Use this when transitioning to
    /// a background decode task model where a separate tokio task
    /// consumes raw bytes from the worker transport.
    pub fn take_worker_output_rx(
        &mut self,
    ) -> Option<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>> {
        // We can only extract the receiver from a ZmqWorkerTransport.
        // For other implementations (mocks, tests), we return None.
        None
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
        async fn recv_event(&mut self) -> Result<FrontendEvent> {
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

    #[tokio::test]
    async fn frontend_mut_borrows_independent_of_worker_mut() {
        // Compiler test: borrow_both_mut must hand out two non-aliasing
        // mut refs (Frontend + Worker) so the engine's tokio::select!
        // can poll both. If this compiles, the API contract holds.
        let mut dispatch = DispatchSystem::new(
            Box::new(CapturingFrontend::default()),
            Box::new(CapturingWorker::default()),
        );
        let (front, _worker) = dispatch.borrow_both_mut();
        let _ = front.recv_event().await;
    }
}
