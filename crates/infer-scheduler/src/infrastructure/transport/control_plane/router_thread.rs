//! Std thread that owns the control-plane ZMQ ROUTER socket.
//!
//! The router thread is the single point that:
//!   * binds the ROUTER socket and `inproc` wakeup pair,
//!   * decodes incoming `ControlEnvelope<WorkerControlMessage>` frames,
//!   * mutates the [`Registry`] (registers workers, updates `last_seen`),
//!   * resolves [`RequestId`]-correlated replies via [`PendingCalls`],
//!   * surfaces uncorrelated events via the engine event channel,
//!   * encodes and sends scheduler-originated frames.
//!
//! Everything else in the control-plane module is a passive helper that
//! either talks to the router thread (engine side) or is read by it
//! (registry view).

use std::sync::Arc;
use std::time::{Duration, Instant};

use infer_protocol::ControlEnvelope;
use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::{WorkerControlMessage, WorkerState};
use tokio::sync::mpsc;

use super::codec::{decode_worker, encode_scheduler};
use super::handle::{ControlError, ControlEvent, RouterCommand, WorkerId};
use super::pending_calls::PendingCalls;
use super::registry::{RegistrationRefused, Registry};

/// Configuration plumbed through to the router thread.
pub(crate) struct RouterConfig {
    /// ZMQ ROUTER endpoint string. Currently unused — the router
    /// thread inherits an already-bound socket from the bootstrap
    /// path. Kept for diagnostics and a future "rebind on poison"
    /// recovery flow.
    #[allow(dead_code)]
    pub(crate) endpoint: String,
    pub(crate) wakeup_endpoint: String,
}

/// Drive the running-phase router loop on the **current thread**, reusing an
/// already-bound ROUTER socket. Used by the bootstrap path which binds the
/// socket itself, runs the handshake, then converts itself into the router
/// loop without ever moving the !Send socket across threads.
pub(crate) fn run_in_place(
    ctx: zmq::Context,
    socket: zmq::Socket,
    cfg: RouterConfig,
    pending: Arc<PendingCalls>,
    registry: Registry,
    cmd_rx: mpsc::UnboundedReceiver<RouterCommand>,
    event_tx: mpsc::UnboundedSender<ControlEvent>,
) -> Result<(), ControlError> {
    drive_router(ctx, socket, cfg, pending, registry, cmd_rx, event_tx)
}

fn drive_router(
    ctx: zmq::Context,
    socket: zmq::Socket,
    cfg: RouterConfig,
    pending: Arc<PendingCalls>,
    mut registry: Registry,
    cmd_rx: mpsc::UnboundedReceiver<RouterCommand>,
    event_tx: mpsc::UnboundedSender<ControlEvent>,
) -> Result<(), ControlError> {
    // Inproc wakeup so the engine can prod the router whenever a RouterCommand
    // is queued — without this the router thread would block in zmq_poll()
    // and never see new outbound work.
    let wakeup_rx = ctx
        .socket(zmq::PAIR)
        .map_err(|e| ControlError::Router(format!("wakeup PAIR: {}", e)))?;
    wakeup_rx
        .bind(&cfg.wakeup_endpoint)
        .map_err(|e| ControlError::Router(format!("bind wakeup {}: {}", cfg.wakeup_endpoint, e)))?;
    wakeup_rx
        .set_rcvtimeo(0)
        .map_err(|e| ControlError::Router(format!("wakeup rcvtimeo: {}", e)))?;

    // Bridge: drain the tokio mpsc into a sync mpsc and signal the wakeup
    // socket. Identical pattern to zmq_transport.rs.
    let (sync_tx, sync_rx) = std::sync::mpsc::channel::<RouterCommand>();
    let ctx_clone = ctx.clone();
    let wakeup_endpoint = cfg.wakeup_endpoint.clone();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let barrier2 = barrier.clone();
    std::thread::Builder::new()
        .name("control-router-bridge".into())
        .spawn(move || {
            let wakeup_tx = ctx_clone
                .socket(zmq::PAIR)
                .expect("control bridge wakeup socket");
            wakeup_tx
                .connect(&wakeup_endpoint)
                .expect("control bridge wakeup connect");
            barrier2.wait();
            bridge_loop(cmd_rx, sync_tx, wakeup_tx);
        })
        .map_err(|e| ControlError::Router(format!("spawn bridge: {}", e)))?;
    barrier.wait();

    // Periodic deadline sweep cadence. Cheap; runs whenever the poll wakes.
    let mut last_sweep = Instant::now();
    let sweep_interval = Duration::from_millis(100);

    'main: loop {
        // Use a finite poll timeout so the deadline sweep runs even if no
        // traffic arrives.
        let timeout_ms: i64 = sweep_interval.as_millis() as i64;
        let mut items = [
            socket.as_poll_item(zmq::POLLIN),
            wakeup_rx.as_poll_item(zmq::POLLIN),
        ];
        if let Err(e) = zmq::poll(&mut items, timeout_ms) {
            tracing::error!("control router poll: {:?}", e);
            break;
        }

        // 1. Drain inbound frames.
        loop {
            match recv_frame(&socket) {
                Ok(Some((identity, env))) => {
                    handle_inbound(&mut registry, &pending, &event_tx, identity, env);
                }
                Ok(None) => break, // EAGAIN
                Err(e) => {
                    tracing::error!("control router recv: {:?}", e);
                    break;
                }
            }
        }

        // 2. Consume wakeup notifications and drain outbound commands.
        while wakeup_rx.recv_bytes(zmq::DONTWAIT).is_ok() {}
        while let Ok(cmd) = sync_rx.try_recv() {
            match cmd {
                RouterCommand::Shutdown => {
                    pending.shutdown();
                    tracing::info!("control router received Shutdown");
                    break 'main;
                }
                RouterCommand::SendTo { worker, env } => {
                    if let Err(e) = send_to(&socket, &worker, &env) {
                        tracing::error!("send_to {}: {:?}", worker, e);
                    }
                }
                RouterCommand::Broadcast { env } => {
                    let workers = registry.current_workers();
                    for w in workers {
                        if let Err(e) = send_to(&socket, &w, &env) {
                            tracing::error!("broadcast to {}: {:?}", w, e);
                        }
                    }
                }
                RouterCommand::CallOne {
                    worker,
                    env,
                    deadline: _,
                } => {
                    if let Err(e) = send_to(&socket, &worker, &env) {
                        tracing::error!("call_one {}: {:?}", worker, e);
                        // Resolve the pending entry with the error immediately.
                        // We need to forge a "decode error path" — easiest is to
                        // mark the pending call as router error via complete()
                        // with a fake reply. Instead use a dedicated path:
                        // pending entry will time out at its deadline; the
                        // engine sees Timeout. Acceptable failure mode.
                    }
                }
                RouterCommand::CallAll { env, deadline: _ } => {
                    let workers = registry.current_workers();
                    pending.set_expected(env.request_id, workers.clone());
                    for w in workers {
                        if let Err(e) = send_to(&socket, &w, &env) {
                            tracing::error!("call_all to {}: {:?}", w, e);
                        }
                    }
                }
            }
        }

        // 3. Periodic deadline sweep.
        let now = Instant::now();
        if now.saturating_duration_since(last_sweep) >= sweep_interval {
            let n = pending.sweep_expired(now);
            if n > 0 {
                tracing::debug!("expired {} pending control RPCs", n);
            }
            last_sweep = now;
        }
    }

    pending.shutdown();
    Ok(())
}

fn bridge_loop(
    mut cmd_rx: mpsc::UnboundedReceiver<RouterCommand>,
    sync_tx: std::sync::mpsc::Sender<RouterCommand>,
    wakeup_tx: zmq::Socket,
) {
    loop {
        match cmd_rx.blocking_recv() {
            Some(cmd) => {
                let is_shutdown = matches!(cmd, RouterCommand::Shutdown);
                if sync_tx.send(cmd).is_err() {
                    return;
                }
                let _ = wakeup_tx.send(&[1u8][..], zmq::DONTWAIT);
                if is_shutdown {
                    return;
                }
            }
            None => return,
        }
    }
}

/// Receive one full ROUTER frame. Returns `Ok(None)` on EAGAIN.
fn recv_frame(
    socket: &zmq::Socket,
) -> Result<Option<(Vec<u8>, ControlEnvelope<WorkerControlMessage>)>, ControlError> {
    let identity = match socket.recv_bytes(zmq::DONTWAIT) {
        Ok(b) => b,
        Err(zmq::Error::EAGAIN) => return Ok(None),
        Err(e) => return Err(ControlError::Router(format!("recv identity: {}", e))),
    };
    // ROUTER may send a delimiter frame between identity and payload depending
    // on whether the peer is a DEALER (no delimiter) or REQ (delimiter). We
    // walk frames until the last one (which carries the payload).
    let mut last = identity.clone();
    while socket.get_rcvmore().unwrap_or(false) {
        match socket.recv_bytes(0) {
            Ok(b) => last = b,
            Err(e) => return Err(ControlError::Router(format!("recv body: {}", e))),
        }
    }
    // If the peer is a DEALER we walked: identity → payload. If it's a REQ:
    // identity → "" → payload. Either way `last` is the payload.
    if std::ptr::eq(last.as_slice().as_ptr(), identity.as_slice().as_ptr()) {
        // single frame received → caller spoke without a payload, ignore.
        tracing::warn!("ROUTER received single-frame message; dropping");
        return Ok(None);
    }
    let env = decode_worker(&last)?;
    Ok(Some((identity, env)))
}

fn handle_inbound(
    registry: &mut Registry,
    pending: &PendingCalls,
    event_tx: &mpsc::UnboundedSender<ControlEvent>,
    identity: Vec<u8>,
    env: ControlEnvelope<WorkerControlMessage>,
) {
    let now = Instant::now();
    // Map state for liveness; default to Running for unknown variants.
    let state = liveness_state_of(&env.payload);
    let worker = match registry.intern(&identity, now, state) {
        Ok(w) => w,
        Err(RegistrationRefused::Reconnect) => {
            tracing::error!(
                "Refusing reconnect from previously evicted worker (identity len={})",
                identity.len()
            );
            return;
        }
    };

    // RPC reply path: if the envelope has a non-zero RequestId, route to
    // PendingCalls; do not surface as an event.
    if env.request_id.is_correlated() {
        pending.complete(env.request_id, worker, env.payload);
        return;
    }

    // Spontaneous events. Surface to the engine.
    let event = match env.payload {
        WorkerControlMessage::Heartbeat(hb) => ControlEvent::Heartbeat { worker, hb },
        WorkerControlMessage::StepError(err) => ControlEvent::StepError { worker, err },
        WorkerControlMessage::AllocFailed(req) => ControlEvent::AllocFailed { worker, req },
        WorkerControlMessage::Error(e) => ControlEvent::WorkerError {
            worker,
            message: e.message,
            fatal: matches!(e.state, WorkerState::Error),
        },
        // Bootstrap-phase progress messages: the bootstrap state machine
        // consumes them; once Running, they're surplus and just logged.
        WorkerControlMessage::Hello(_)
        | WorkerControlMessage::Progress(_)
        | WorkerControlMessage::Ready(_)
        | WorkerControlMessage::MemoryProfile(_)
        | WorkerControlMessage::PagedKvReady(_) => {
            tracing::debug!("ignoring bootstrap-phase message in Running plane");
            return;
        }
        // RPC replies must carry a non-zero id; if we get here with a reply
        // shape and id=NONE, something is wrong with the worker side.
        WorkerControlMessage::Pong
        | WorkerControlMessage::CancelAck(_)
        | WorkerControlMessage::DrainAck(_)
        | WorkerControlMessage::UnloadAck(_) => {
            tracing::warn!("RPC reply received with RequestId::NONE; dropping");
            return;
        }
    };

    if event_tx.send(event).is_err() {
        tracing::warn!("control event receiver dropped; router shutting down");
    }
}

fn liveness_state_of(msg: &WorkerControlMessage) -> WorkerState {
    match msg {
        WorkerControlMessage::Heartbeat(hb) => hb.state,
        WorkerControlMessage::Hello(_) => WorkerState::Connecting,
        WorkerControlMessage::Progress(p) => p.state,
        WorkerControlMessage::Ready(_) => WorkerState::Ready,
        WorkerControlMessage::MemoryProfile(_) => WorkerState::ProfilingMemory,
        WorkerControlMessage::PagedKvReady(_) => WorkerState::AllocatingRuntime,
        WorkerControlMessage::Error(e) => e.state,
        _ => WorkerState::Running,
    }
}

fn send_to(
    socket: &zmq::Socket,
    worker: &WorkerId,
    env: &ControlEnvelope<SchedulerControlMessage>,
) -> Result<(), ControlError> {
    let bytes = encode_scheduler(env)?;
    socket
        .send(worker.as_bytes(), zmq::SNDMORE)
        .map_err(|e| ControlError::Router(format!("send identity: {}", e)))?;
    socket
        .send(&bytes, 0)
        .map_err(|e| ControlError::Router(format!("send payload: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Router thread integration tests live in `mod.rs` because they need to
    //! orchestrate the full `ControlPlane<Bootstrapping>` → `Running`
    //! transition with a fake worker DEALER.
}
