//! ControlPump — ZMQ DEALER socket for control plane communication.
//!
//! The DEALER socket connects to the scheduler's ROUTER.
//! Messages are framed as: [empty frame][msgpack payload].

use infer_protocol::control_envelope::{ControlEnvelope, RequestId};
use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::{
    WorkerControlMessage, WorkerHello, WorkerHeartbeat, WorkerReady,
    WorkerState, WorkerCapacity, WORKER_CONTROL_PROTOCOL_VERSION,
};

/// ControlPump manages the ZMQ DEALER socket for control plane.
pub struct ControlPump {
    pub worker_id: String,
    socket: zmq::Socket,
}

impl ControlPump {
    pub fn new(ctx: &zmq::Context, worker_id: String, endpoint: &str) -> Result<Self, String> {
        let socket = ctx.socket(zmq::DEALER).map_err(|e| format!("zmq socket: {}", e))?;
        socket.set_identity(worker_id.as_bytes()).map_err(|e| format!("set identity: {}", e))?;
        socket.set_linger(1000).map_err(|e| format!("set linger: {}", e))?;
        socket.connect(endpoint).map_err(|e| format!("connect {}: {}", endpoint, e))?;
        Ok(Self { worker_id, socket })
    }

    /// Serialize and send a control message.
    pub fn send(&self, msg: WorkerControlMessage, request_id: RequestId) -> Result<(), String> {
        let envelope = ControlEnvelope { request_id, payload: msg };
        let bytes = rmp_serde::to_vec(&envelope).map_err(|e| format!("serialize: {}", e))?;
        self.socket.send(&bytes, 0).map_err(|e| format!("send: {}", e))
    }

    /// Send initial hello.
    pub fn send_hello(&self) -> Result<(), String> {
        self.send(WorkerControlMessage::Hello(WorkerHello {
            worker_id: self.worker_id.clone(),
            pid: std::process::id(),
            hostname: hostname(),
            device: "cuda:0".into(),
            protocol_version: WORKER_CONTROL_PROTOCOL_VERSION,
        }), RequestId::NONE)
    }

    /// Send heartbeat. `active_requests` is the worker's current load
    /// (decode + waiting prefills); `kv_total_slots` / `kv_free_slots`
    /// expose the worker's `GlobalKvAllocator` occupancy so the scheduler
    /// can run RadixTree LRU eviction in response to pressure signals.
    pub fn send_heartbeat(
        &self,
        active_requests: usize,
        kv_total_slots: u32,
        kv_free_slots: u32,
    ) -> Result<(), String> {
        self.send(WorkerControlMessage::Heartbeat(WorkerHeartbeat {
            worker_id: self.worker_id.clone(),
            state: WorkerState::Running,
            active_requests,
            kv_total_slots,
            kv_free_slots,
        }), RequestId::NONE)
    }

    /// Send WorkerReady after model loaded.
    pub fn send_ready(
        &self,
        model_instance_id: String,
        model_path: String,
        model_type: String,
        capacity: WorkerCapacity,
    ) -> Result<(), String> {
        self.send(WorkerControlMessage::Ready(WorkerReady {
            worker_id: self.worker_id.clone(),
            model_instance_id,
            model_path,
            model_type,
            device: "cuda:0".into(),
            capacity,
        }), RequestId::NONE)
    }

    /// Receive next control message (blocking).
    pub fn recv(&self) -> Result<(SchedulerControlMessage, RequestId), String> {
        let bytes = self.socket.recv_bytes(0).map_err(|e| format!("recv: {}", e))?;
        let envelope: ControlEnvelope<SchedulerControlMessage> =
            rmp_serde::from_slice(&bytes).map_err(|e| format!("deserialize: {}", e))?;
        Ok((envelope.payload, envelope.request_id))
    }

    /// Non-blocking receive. Returns None if no message available.
    pub fn try_recv(&self, timeout_ms: i64) -> Result<Option<(SchedulerControlMessage, RequestId)>, String> {
        if self.socket.poll(zmq::POLLIN, timeout_ms).map_err(|e| format!("poll: {}", e))? == 0 {
            return Ok(None);
        }
        self.recv().map(Some)
    }
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}
