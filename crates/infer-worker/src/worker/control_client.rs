use anyhow::Result;

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::*;
use infer_protocol::{ControlEnvelope, RequestId};

/// Minimal ZMQ + MessagePack control-plane client used by the Worker lifecycle.
///
/// Every payload crossing the wire is wrapped in a [`ControlEnvelope`].
/// Spontaneous events (Hello, Progress, Heartbeat, …) carry
/// `RequestId::NONE`; RPC replies copy the originating request id.
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

    /// Borrow the underlying context (for handing off to ControlPump).
    pub fn context(&self) -> &zmq::Context {
        &self._ctx
    }

    /// Decompose into raw socket + context, used when handing the connected
    /// DEALER off to the runtime [`super::control_pump::ControlPump`].
    pub fn into_parts(self) -> (zmq::Context, zmq::Socket, String) {
        (self._ctx, self.socket, self.worker_id)
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

    pub fn send_memory_profile(&self, profile: WorkerMemoryProfile) -> Result<()> {
        self.send(WorkerControlMessage::MemoryProfile(profile))
    }

    pub fn send_paged_kv_ready(&self, ready: PagedKvReady) -> Result<()> {
        self.send(WorkerControlMessage::PagedKvReady(ready))
    }

    /// Receive the next scheduler envelope and unwrap the payload. The
    /// `RequestId` is dropped on the floor for bootstrap (which only sees
    /// one-way Hello/LoadModel/InitPagedKv messages); the runtime pump
    /// preserves it through [`super::control_pump`].
    pub fn recv_scheduler_message(&self) -> Result<SchedulerControlMessage> {
        let data = self.socket.recv_bytes(0)?;
        let env: ControlEnvelope<SchedulerControlMessage> = rmp_serde::from_slice(&data)?;
        Ok(env.payload)
    }

    fn send(&self, msg: WorkerControlMessage) -> Result<()> {
        let env = ControlEnvelope::oneway(msg);
        let data = rmp_serde::to_vec(&env)?;
        self.socket.send(&data, 0)?;
        Ok(())
    }
}

/// Wire-level helper exposed for the runtime control pump.
pub fn encode_worker_envelope(
    request_id: RequestId,
    payload: WorkerControlMessage,
) -> Result<Vec<u8>> {
    let env = ControlEnvelope { request_id, payload };
    Ok(rmp_serde::to_vec(&env)?)
}

/// Inverse of [`encode_worker_envelope`] for messages received from the
/// scheduler.
pub fn decode_scheduler_envelope(
    bytes: &[u8],
) -> Result<ControlEnvelope<SchedulerControlMessage>> {
    Ok(rmp_serde::from_slice(bytes)?)
}
