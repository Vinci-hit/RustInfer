//! DataPump — ZMQ data plane (PULL for receiving batch commands, PUSH for sending outputs).

use infer_protocol::scheduler_to_worker_data::BatchCommand;
use infer_protocol::worker_to_scheduler_data::{DiffusionBatchOutput, StepOutput};

/// DataPump manages the ZMQ data plane sockets.
pub struct DataPump {
    recv_socket: zmq::Socket, // PULL: receive BatchCommand from scheduler
    send_socket: zmq::Socket, // PUSH: send StepOutput / DiffusionBatchOutput to scheduler
}

impl DataPump {
    pub fn new(
        ctx: &zmq::Context,
        recv_endpoint: &str,
        send_endpoint: &str,
    ) -> Result<Self, String> {
        let recv_socket = ctx
            .socket(zmq::PULL)
            .map_err(|e| format!("PULL socket: {}", e))?;
        recv_socket
            .set_linger(1000)
            .map_err(|e| format!("set linger: {}", e))?;
        recv_socket
            .connect(recv_endpoint)
            .map_err(|e| format!("connect recv {}: {}", recv_endpoint, e))?;

        let send_socket = ctx
            .socket(zmq::PUSH)
            .map_err(|e| format!("PUSH socket: {}", e))?;
        send_socket
            .set_linger(1000)
            .map_err(|e| format!("set linger: {}", e))?;
        send_socket
            .connect(send_endpoint)
            .map_err(|e| format!("connect send {}: {}", send_endpoint, e))?;

        Ok(Self {
            recv_socket,
            send_socket,
        })
    }

    /// Blocking receive next batch command.
    pub fn recv_batch(&self) -> Result<BatchCommand, String> {
        let bytes = self
            .recv_socket
            .recv_bytes(0)
            .map_err(|e| format!("recv: {}", e))?;
        rmp_serde::from_slice(&bytes).map_err(|e| format!("deserialize BatchCommand: {}", e))
    }

    /// Borrow the underlying PULL socket, e.g. to build a `zmq::PollItem`
    /// for multiplexed polling alongside the control plane.
    pub fn recv_socket(&self) -> &zmq::Socket {
        &self.recv_socket
    }

    /// Non-blocking receive with timeout (ms). Returns None if no message.
    pub fn try_recv_batch(&self, timeout_ms: i64) -> Result<Option<BatchCommand>, String> {
        if self
            .recv_socket
            .poll(zmq::POLLIN, timeout_ms)
            .map_err(|e| format!("poll: {}", e))?
            == 0
        {
            return Ok(None);
        }
        self.recv_batch().map(Some)
    }

    /// Send LLM step output.
    pub fn send_step_output(&self, output: &StepOutput) -> Result<(), String> {
        let bytes = rmp_serde::to_vec(output).map_err(|e| format!("serialize: {}", e))?;
        self.send_socket
            .send(&bytes, 0)
            .map_err(|e| format!("send: {}", e))
    }

    /// Send diffusion batch output.
    pub fn send_diffusion_output(&self, output: &DiffusionBatchOutput) -> Result<(), String> {
        let bytes = rmp_serde::to_vec(output).map_err(|e| format!("serialize: {}", e))?;
        self.send_socket
            .send(&bytes, 0)
            .map_err(|e| format!("send: {}", e))
    }
}
