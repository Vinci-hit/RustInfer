//! Worker control plane over ZMQ ROUTER + MessagePack.

use infer_protocol::scheduler_to_worker_control::{
    InitPagedKv, LoadModel, SchedulerControlMessage, SchedulerHello,
};
use infer_protocol::worker_to_scheduler_control::{
    WorkerControlMessage, WorkerReady, WORKER_CONTROL_PROTOCOL_VERSION,
};

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
            WorkerControlMessage::MemoryProfile(profile) => {
                tracing::info!(
                    "WorkerMemoryProfile: id={} device={} free_after_dummy={} suggested_kv_budget={}",
                    profile.worker_id,
                    profile.device,
                    profile.free_mem_after_dummy_bytes,
                    profile.suggested_kv_budget_bytes,
                );
                if let Some(cmd) = &load_model
                    && let Some(block_size) = paged_block_size(cmd) {
                        let bytes_per_block = profile.layer_num as u64
                            * 2
                            * block_size as u64
                            * profile.kv_head_num as u64
                            * profile.head_size as u64
                            * profile.dtype_size as u64;
                        if bytes_per_block == 0 {
                            return Err(SchedulerError::WorkerError(
                                "invalid worker memory profile: bytes_per_block=0".into(),
                            ));
                        }
                        let num_blocks = (profile.suggested_kv_budget_bytes / bytes_per_block) as u32;
                        if num_blocks == 0 {
                            return Err(SchedulerError::WorkerError(format!(
                                "insufficient KV budget: budget={} bytes_per_block={}",
                                profile.suggested_kv_budget_bytes, bytes_per_block,
                            )));
                        }
                        let max_blocks_per_seq = (cmd.max_model_len as u32).div_ceil(block_size).max(1);
                        send_init_paged_kv(&socket, identity, InitPagedKv {
                            model_instance_id: cmd.model_instance_id.clone(),
                            block_size,
                            initial_num_blocks: num_blocks,
                            max_num_blocks: num_blocks,
                            max_blocks_per_seq,
                            decode_block_request_blocks: 1,
                            decode_block_prefetch_margin: 4,
                        })?;
                    }
            }
            WorkerControlMessage::PagedKvReady(ready) => {
                tracing::info!(
                    "PagedKvReady: id={} blocks={}/{} block_size={} bytes={}",
                    ready.worker_id,
                    ready.initial_num_blocks,
                    ready.max_num_blocks,
                    ready.block_size,
                    ready.bytes_allocated,
                );
            }
            WorkerControlMessage::NeedBlocks(need) => {
                tracing::debug!(
                    "NeedBlocks before ready gate: id={} seq={} current={} required={} request={}",
                    need.worker_id,
                    need.sequence_id,
                    need.current_blocks,
                    need.required_blocks,
                    need.request_blocks,
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

fn send_init_paged_kv(socket: &zmq::Socket, identity: &[u8], init: InitPagedKv) -> Result<()> {
    tracing::info!(
        "Sending InitPagedKv: model_instance_id={} block_size={} blocks={}",
        init.model_instance_id,
        init.block_size,
        init.initial_num_blocks,
    );
    send_scheduler_msg(socket, identity, &SchedulerControlMessage::InitPagedKv(init))
}

fn paged_block_size(cmd: &LoadModel) -> Option<u32> {
    let mode = cmd.kv_cache_mode.as_deref()?;
    let rest = mode.strip_prefix("paged:")?;
    rest.parse::<u32>().ok().filter(|&v| v > 0)
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
