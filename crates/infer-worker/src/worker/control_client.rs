use anyhow::Result;

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::*;

/// Minimal ZMQ + MessagePack control-plane client used by the Worker lifecycle.
pub struct WorkerControlClient {
    _ctx: zmq::Context,
    socket: zmq::Socket,
    worker_id: String,
}

impl WorkerControlClient {
    pub fn connect(endpoint: &str, worker_id: String) -> Result<Self> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::DEALER)?;
        socket.set_identity(worker_id.as_bytes())?;
        socket.connect(endpoint)?;
        Ok(Self { _ctx: ctx, socket, worker_id })
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn send_hello(&self, pid: u32, hostname: String, device: String) -> Result<()> {
        self.send(WorkerControlMessage::Hello(WorkerHello {
            worker_id: self.worker_id.clone(),
            pid,
            hostname,
            device,
            protocol_version: WORKER_CONTROL_PROTOCOL_VERSION,
        }))
    }

    pub fn send_progress(&self, state: WorkerState, message: impl Into<String>) -> Result<()> {
        self.send(WorkerControlMessage::Progress(WorkerProgress {
            worker_id: self.worker_id.clone(),
            state,
            message: message.into(),
        }))
    }

    pub fn send_ready(
        &self,
        model_instance_id: String,
        model_path: String,
        model_type: String,
        device: String,
        capacity: WorkerCapacity,
    ) -> Result<()> {
        self.send(WorkerControlMessage::Ready(WorkerReady {
            worker_id: self.worker_id.clone(),
            model_instance_id,
            model_path,
            model_type,
            device,
            capacity,
        }))
    }

    pub fn send_error(&self, state: WorkerState, message: impl Into<String>) -> Result<()> {
        self.send(WorkerControlMessage::Error(WorkerError {
            worker_id: self.worker_id.clone(),
            state,
            message: message.into(),
        }))
    }

    pub fn recv_scheduler_message(&self) -> Result<SchedulerControlMessage> {
        let data = self.socket.recv_bytes(0)?;
        Ok(rmp_serde::from_slice(&data)?)
    }

    fn send(&self, msg: WorkerControlMessage) -> Result<()> {
        let data = rmp_serde::to_vec(&msg)?;
        self.socket.send(&data, 0)?;
        Ok(())
    }
}
