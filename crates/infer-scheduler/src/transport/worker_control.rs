//! Worker control plane over ZMQ ROUTER + MessagePack.

use infer_worker::worker::control_protocol::*;

use crate::error::{Result, SchedulerError, TransportError};

/// Blocks until one Worker reports Ready.
///
/// Phase 1 supports a single Worker. The protocol and ROUTER socket are chosen so
/// WorkerGroup support can be added without replacing the transport.
pub fn wait_for_worker_ready(endpoint: &str, load_model: Option<LoadModel>) -> Result<WorkerReady> {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::ROUTER).map_err(zmq_err)?;
    socket.bind(endpoint).map_err(zmq_err)?;
    tracing::info!("Worker control ROUTER bound to {}", endpoint);
    tracing::info!("Waiting for WorkerReady...");

    loop {
        let frames = socket.recv_multipart(0).map_err(zmq_err)?;
        if frames.len() < 2 {
            tracing::warn!("Ignoring malformed control message with {} frames", frames.len());
            continue;
        }

        let identity = &frames[0];
        let data = &frames[frames.len() - 1];
        let msg: WorkerControlMessage = rmp_serde::from_slice(data).map_err(|e| {
            SchedulerError::Transport(TransportError::Serialization(format!(
                "control msgpack decode: {}",
                e
            )))
        })?;

        match msg {
            WorkerControlMessage::Hello(hello) => {
                tracing::info!(
                    "WorkerHello: id={} pid={} host={} device={} protocol={}",
                    hello.worker_id,
                    hello.pid,
                    hello.hostname,
                    hello.device,
                    hello.protocol_version,
                );
                send_scheduler_hello(&socket, identity)?;
                if let Some(cmd) = &load_model {
                    send_load_model(&socket, identity, cmd)?;
                }
            }
            WorkerControlMessage::Progress(progress) => {
                tracing::info!(
                    "WorkerProgress: id={} state={:?} message={}",
                    progress.worker_id,
                    progress.state,
                    progress.message,
                );
            }
            WorkerControlMessage::Ready(ready) => {
                tracing::info!(
                    "WorkerReady: id={} model_type={} device={} max_batch_tokens={} max_batch_seqs={}",
                    ready.worker_id,
                    ready.model_type,
                    ready.device,
                    ready.capacity.max_batch_tokens,
                    ready.capacity.max_batch_seqs,
                );
                return Ok(ready);
            }
            WorkerControlMessage::Heartbeat(heartbeat) => {
                tracing::debug!(
                    "WorkerHeartbeat: id={} state={:?} active_requests={}",
                    heartbeat.worker_id,
                    heartbeat.state,
                    heartbeat.active_requests,
                );
            }
            WorkerControlMessage::Error(err) => {
                return Err(SchedulerError::WorkerError(format!(
                    "worker {} failed in {:?}: {}",
                    err.worker_id, err.state, err.message
                )));
            }
        }
    }
}

fn send_scheduler_hello(socket: &zmq::Socket, identity: &[u8]) -> Result<()> {
    let msg = SchedulerControlMessage::Hello(SchedulerHello {
        protocol_version: WORKER_CONTROL_PROTOCOL_VERSION,
        heartbeat_interval_ms: 1_000,
    });
    send_scheduler_msg(socket, identity, &msg)
}

fn send_load_model(socket: &zmq::Socket, identity: &[u8], cmd: &LoadModel) -> Result<()> {
    tracing::info!(
        "Sending LoadModel: model_instance_id={} model_type={} path={}",
        cmd.model_instance_id,
        cmd.model_type,
        cmd.model_path,
    );
    send_scheduler_msg(socket, identity, &SchedulerControlMessage::LoadModel(cmd.clone()))
}

fn send_scheduler_msg(
    socket: &zmq::Socket,
    identity: &[u8],
    msg: &SchedulerControlMessage,
) -> Result<()> {
    let data = rmp_serde::to_vec(msg).map_err(|e| {
        SchedulerError::Transport(TransportError::Serialization(format!(
            "control msgpack encode: {}",
            e
        )))
    })?;
    socket.send(identity, zmq::SNDMORE).map_err(zmq_err)?;
    socket.send(&data, 0).map_err(zmq_err)?;
    Ok(())
}

fn zmq_err(e: zmq::Error) -> SchedulerError {
    SchedulerError::Transport(TransportError::ConnectionFailed(e.to_string()))
}
