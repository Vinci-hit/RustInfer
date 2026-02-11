//! Worker Server - ZeroMQ-based RPC Server
//!
//! Thin shell: connects to Scheduler via ControlPlaneClient,
//! drives the state machine lifecycle, dispatches commands through handlers.
//!
//! Lifecycle:
//!   Initializing → Registering (handshake) → WaitingModel → command loop
//!
//! In the command loop, `handlers::dispatch` validates state per-command.

use anyhow::Result;

use crate::base::DeviceType;

use super::{Worker, WorkerConfig};
use super::control_plane::ControlPlaneClient;
use super::state_machine::WorkerState;
use super::handlers;

/// Worker Server — connects Worker to Scheduler over ZMQ.
pub struct WorkerServer {
    worker: Worker,
    control: ControlPlaneClient,
    rank: usize,
    world_size: usize,
    running: bool,
}

impl WorkerServer {
    /// Create a new WorkerServer and connect to the Scheduler.
    pub async fn new(
        rank: usize,
        world_size: usize,
        scheduler_url: &str,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let worker_config = WorkerConfig::cuda(rank as u32)
            .with_id(format!("worker-{}", rank))
            .with_tp(rank as u32, world_size as u32);

        #[cfg(not(feature = "cuda"))]
        let worker_config = WorkerConfig::cpu()
            .with_id(format!("worker-{}", rank))
            .with_tp(rank as u32, world_size as u32);

        let worker = Worker::new(worker_config)?;
        let control = ControlPlaneClient::connect(rank, world_size, scheduler_url).await?;

        Ok(Self {
            worker,
            control,
            rank,
            world_size,
            running: false,
        })
    }

    /// State-machine-driven event loop.
    ///
    /// 1. Initializing → Registering (handshake with Scheduler)
    /// 2. Registering → WaitingModel
    /// 3. Command loop: recv → dispatch (handlers manage further transitions) → send
    pub async fn run_loop(&mut self) -> Result<()> {
        println!("[Worker-{}] Starting main loop...", self.rank);
        self.running = true;

        // ── Phase 1: Init → Registering ──
        self.worker.transition_state(WorkerState::Registering)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // ── Phase 2: Handshake ──
        let device_type_str = match self.worker.device_type() {
            DeviceType::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => "cuda",
        };
        let device_id = match self.worker.device_type() {
            DeviceType::Cpu => 0,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => id as u32,
        };

        self.control
            .handshake(self.worker.worker_id(), device_type_str, device_id)
            .await?;

        // ── Phase 3: Registering → WaitingModel ──
        self.worker.transition_state(WorkerState::WaitingModel)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        println!(
            "[Worker-{}] Ready (state: {}). Waiting for commands...",
            self.rank,
            self.worker.state(),
        );

        // ── Phase 4: Command loop ──
        while self.running {
            let command = match self.control.recv_command().await {
                Ok(cmd) => cmd,
                Err(e) => {
                    eprintln!("[Worker-{}] recv error: {}", self.rank, e);
                    continue;
                }
            };

            // State-aware dispatch (handlers validate + transition)
            let response = handlers::dispatch(&mut self.worker, command);

            if let Err(e) = self.control.send_response(response).await {
                eprintln!("[Worker-{}] Failed to send response: {}", self.rank, e);
            }

            // If worker entered Error state, log but keep running
            // (Scheduler decides whether to restart)
            if self.worker.state().is_error() {
                eprintln!(
                    "[Worker-{}] Worker is in error state: {}",
                    self.rank,
                    self.worker.state(),
                );
            }
        }

        println!("[Worker-{}] Main loop exited (state: {})", self.rank, self.worker.state());
        Ok(())
    }

    /// Stop the server gracefully.
    pub fn shutdown(&mut self) {
        println!("[Worker-{}] Shutdown requested", self.rank);
        self.running = false;
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn is_running(&self) -> bool {
        self.running
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_worker_server_config() {
        let rank = 0;
        let world_size = 2;
        assert!(rank < world_size);
    }
}
