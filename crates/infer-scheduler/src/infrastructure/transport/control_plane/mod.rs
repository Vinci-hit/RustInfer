//! Control plane: persistent scheduler↔worker control channel.
//!
//! [`ControlPlane`] owns the ZMQ ROUTER socket (in a dedicated std thread)
//! and exposes [`ControlPlaneCmdTx`] (cheap-clone send side) plus
//! [`ControlPlaneEventRx`] (single-consumer event stream).
//!
//! ```text
//!   engine ──cmd_tx──▶ tokio::mpsc<RouterCommand>
//!                                │
//!     std router thread ◀────────┘
//!     │ owns ROUTER socket
//!     │ encode/decode ControlEnvelope<…>
//!     │ pending_calls.complete(rid, …)  for RPC replies
//!     │ event_tx.send(ControlEvent::…)  for spontaneous events
//!     └──▶ engine.control_events
//! ```
//!
//! Bootstrap (Hello / LoadModel / InitPagedKv / WorkerReady) and runtime
//! (FreeKvIndices / Cancel / Heartbeat / …) all flow through the
//! same socket, so worker ZMQ identity is preserved across the transition.

use std::sync::Arc;
use std::time::{Duration, Instant};

use infer_protocol::scheduler_to_worker_control::{
    InitPagedKv, LoadModel, SchedulerControlMessage, SchedulerHello,
};
use infer_protocol::worker_to_scheduler_control::{
    PagedKvReady, WORKER_CONTROL_PROTOCOL_VERSION, WorkerControlMessage, WorkerError,
    WorkerHeartbeat, WorkerHello, WorkerMemoryProfile, WorkerReady, WorkerState,
};
use infer_protocol::{ControlEnvelope, RequestId};
use tokio::sync::{mpsc, oneshot};

mod codec;
pub mod handle;
mod liveness;
pub(crate) mod pending_calls;
mod registry;
mod router_thread;
pub mod worker_group;

pub use handle::{
    ControlError, ControlEvent, ControlPlaneCmdTx, ControlPlaneEventRx, ControlResult,
    TryFromControlReply, WorkerId,
};
pub use worker_group::{WorkerGroup, WorkerGroupState};

use handle::RouterCommand;
use pending_calls::PendingCalls;
use registry::Registry;

// ─────────────────────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ControlPlaneConfig {
    /// Heartbeat cadence advertised in [`SchedulerHello`]. Workers MUST emit
    /// `WorkerHeartbeat` at least this frequently.
    pub heartbeat_interval: Duration,

    /// Liveness threshold. A worker whose `last_seen` ages past this triggers
    /// [`ControlEvent::WorkerLost`].
    pub heartbeat_timeout: Duration,

    /// Default deadline applied to `call_one` / `call_all` RPCs.
    pub default_rpc_deadline: Duration,

    /// Internal: inproc endpoint used between the bridge thread and the
    /// router thread. Tests override this; production keeps the default.
    pub wakeup_endpoint: String,
}

impl Default for ControlPlaneConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_millis(1_000),
            heartbeat_timeout: Duration::from_millis(5_000),
            default_rpc_deadline: Duration::from_secs(30),
            wakeup_endpoint: format!("inproc://control-router-wakeup-{}", uuid::Uuid::new_v4()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ControlPlane (running)
// ─────────────────────────────────────────────────────────────────────────────

/// Running control plane. Construct via [`ControlPlane::bootstrap`].
///
/// Holds the join handles for the router thread and liveness watchdog plus
/// the shutdown side of each. Dropping it triggers a graceful shutdown.
pub struct ControlPlane {
    cmd_tx_inner: mpsc::UnboundedSender<RouterCommand>,
    pending: Arc<PendingCalls>,
    default_rpc_deadline: Duration,
    event_rx: Option<ControlPlaneEventRx>,
    workers_initial: Vec<WorkerId>,

    /// Cooperative shutdown for the liveness watchdog. Sent on Drop.
    liveness_cancel: Option<oneshot::Sender<()>>,

    /// Router thread join handle. Joined on Drop after Shutdown is sent.
    router_join: Option<std::thread::JoinHandle<()>>,
}

impl ControlPlane {
    /// Bind the ROUTER socket, run the bootstrap handshake, and return the
    /// running plane plus the [`WorkerGroup`] derived from the worker(s) that
    /// reported Ready.
    ///
    /// In single-rank deployments the bootstrap waits for exactly one
    /// worker to emit `WorkerReady`. The same socket continues to
    /// serve the running phase, so worker identity survives the
    /// transition.
    pub async fn bootstrap(
        endpoint: &str,
        load_model: Option<LoadModel>,
        cfg: ControlPlaneConfig,
    ) -> ControlResult<(Self, WorkerGroup)> {
        // Channels:
        //   cmd: engine → router (RouterCommand)
        //   event: router → engine (ControlEvent)
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<RouterCommand>();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<ControlEvent>();

        // Pending RPC table is shared by the engine (registers calls) and the
        // router thread (resolves replies). Build it here so cmd_tx and the
        // router both see the same Arc.
        let pending = PendingCalls::new();

        // Bootstrap dance runs in its own std thread to avoid leaking ZMQ
        // socket affinity into the tokio runtime. We hand the socket off to
        // the running router thread once the dance completes.
        let (boot_tx, boot_rx) = std::sync::mpsc::sync_channel::<BootstrapOutcome>(1);
        let endpoint = endpoint.to_string();
        let cfg_clone = cfg.clone();
        let pending_for_router = pending.clone();

        std::thread::Builder::new()
            .name("control-bootstrap".into())
            .spawn(move || {
                run_bootstrap_then_router(
                    endpoint,
                    load_model,
                    cfg_clone,
                    pending_for_router,
                    cmd_rx,
                    event_tx,
                    boot_tx,
                );
            })
            .map_err(|e| ControlError::Router(format!("spawn bootstrap thread: {}", e)))?;

        // Block until bootstrap reports its outcome. We use a short polling
        // loop (rather than blocking_recv on a tokio channel) because the
        // surrounding context is async and we want to yield while waiting.
        let outcome = await_bootstrap(boot_rx).await?;

        // Spawn the liveness watchdog now that we have the registry view.
        let liveness_cancel = liveness::spawn(
            cfg.heartbeat_interval,
            cfg.heartbeat_timeout,
            outcome.registry_view,
            cmd_tx.clone(),
            outcome.event_tx_for_liveness,
        );

        Ok((
            Self {
                cmd_tx_inner: cmd_tx,
                pending,
                default_rpc_deadline: cfg.default_rpc_deadline,
                event_rx: Some(ControlPlaneEventRx { rx: event_rx }),
                workers_initial: outcome.workers,
                liveness_cancel: Some(liveness_cancel),
                router_join: Some(outcome.router_join),
            },
            outcome.worker_group,
        ))
    }

    /// Cheap-to-clone command side.
    pub fn cmd_tx(&self) -> ControlPlaneCmdTx {
        ControlPlaneCmdTx {
            tx: self.cmd_tx_inner.clone(),
            pending: self.pending.clone(),
            default_rpc_deadline: self.default_rpc_deadline,
        }
    }

    /// Take the single-consumer event stream. Panics if called twice.
    pub fn take_event_rx(&mut self) -> ControlPlaneEventRx {
        self.event_rx
            .take()
            .expect("ControlPlane::take_event_rx called twice")
    }

    /// Snapshot of workers known to the plane after bootstrap.
    pub fn workers(&self) -> Vec<WorkerId> {
        self.workers_initial.clone()
    }
}

impl Drop for ControlPlane {
    fn drop(&mut self) {
        // Best-effort graceful shutdown.
        let _ = self.cmd_tx_inner.send(RouterCommand::Shutdown);
        if let Some(c) = self.liveness_cancel.take() {
            let _ = c.send(());
        }
        if let Some(j) = self.router_join.take() {
            // Don't block forever — give the router thread a chance to flush.
            let _ = j.join();
        }
        self.pending.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bootstrap implementation
// ─────────────────────────────────────────────────────────────────────────────

struct BootstrapOutcome {
    worker_group: WorkerGroup,
    workers: Vec<WorkerId>,
    registry_view: Arc<std::sync::RwLock<registry::RegistryView>>,
    event_tx_for_liveness: mpsc::UnboundedSender<ControlEvent>,
    router_join: std::thread::JoinHandle<()>,
}

async fn await_bootstrap(
    rx: std::sync::mpsc::Receiver<BootstrapOutcome>,
) -> ControlResult<BootstrapOutcome> {
    // Park on a blocking task so we don't busy-spin on the runtime.
    tokio::task::spawn_blocking(move || {
        rx.recv()
            .map_err(|_| ControlError::Router("bootstrap thread exited before completion".into()))
    })
    .await
    .map_err(|e| ControlError::Router(format!("bootstrap join: {}", e)))?
}

/// Bootstrap dance: bind ROUTER, drive the Hello/LoadModel/InitPagedKv state
/// machine until a worker reports Ready, then continue running by re-using
/// the same socket inside `router_thread::run_in_place`.
fn run_bootstrap_then_router(
    endpoint: String,
    load_model: Option<LoadModel>,
    cfg: ControlPlaneConfig,
    pending: Arc<PendingCalls>,
    cmd_rx: mpsc::UnboundedReceiver<RouterCommand>,
    event_tx: mpsc::UnboundedSender<ControlEvent>,
    boot_tx: std::sync::mpsc::SyncSender<BootstrapOutcome>,
) {
    let ctx = zmq::Context::new();
    let socket = match ctx.socket(zmq::ROUTER) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("control bootstrap socket: {}", e);
            return;
        }
    };
    if let Err(e) = socket.set_sndhwm(0) {
        tracing::error!("set_sndhwm: {}", e);
        return;
    }
    if let Err(e) = socket.set_rcvhwm(0) {
        tracing::error!("set_rcvhwm: {}", e);
        return;
    }
    if let Err(e) = socket.bind(&endpoint) {
        tracing::error!("control bootstrap bind {}: {}", endpoint, e);
        return;
    }
    tracing::info!("Control plane ROUTER bound to {} (bootstrap)", endpoint);
    tracing::info!("Waiting for WorkerReady...");

    let mut registry = Registry::new();

    let ready = match drive_handshake(&socket, &load_model, &mut registry) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("control bootstrap failed: {:?}", e);
            return;
        }
    };

    tracing::info!(
        "WorkerReady: id={} model_type={} device={} max_batch_tokens={} max_batch_seqs={} max_total_kv_tokens={:?}",
        ready.worker_id,
        ready.model_type,
        ready.device,
        ready.capacity.max_batch_tokens,
        ready.capacity.max_batch_seqs,
        ready.capacity.max_total_kv_tokens,
    );

    let worker_group = WorkerGroup::from_single_ready(ready);
    let workers = registry.current_workers();
    let registry_view = registry.view.clone();

    // Hand-off pattern: this thread becomes the router thread. We spawn a
    // tiny shim thread whose only purpose is to act as a `JoinHandle` carrier
    // (the running router lives in *this* thread, but JoinHandle for the
    // current thread can't be constructed). The shim parks until the router
    // loop exits.
    let cfg_for_router = cfg.clone();
    let event_tx_for_router = event_tx.clone();
    let (router_done_tx, router_done_rx) = std::sync::mpsc::sync_channel::<()>(1);
    let shim_join = match std::thread::Builder::new()
        .name("control-router-shim".into())
        .spawn(move || {
            let _ = router_done_rx.recv();
        }) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("spawn control-router-shim: {}", e);
            return;
        }
    };

    let outcome = BootstrapOutcome {
        worker_group,
        workers,
        registry_view,
        event_tx_for_liveness: event_tx,
        router_join: shim_join,
    };
    if boot_tx.send(outcome).is_err() {
        tracing::error!("bootstrap caller dropped; aborting");
        let _ = router_done_tx.send(());
        return;
    }

    let router_cfg = router_thread::RouterConfig {
        endpoint: endpoint.clone(),
        wakeup_endpoint: cfg_for_router.wakeup_endpoint.clone(),
    };
    if let Err(e) = router_thread::run_in_place(
        ctx,
        socket,
        router_cfg,
        pending,
        registry,
        cmd_rx,
        event_tx_for_router,
    ) {
        tracing::error!("control router exited with error: {:?}", e);
    }
    let _ = router_done_tx.send(());
}

/// Drive Hello / LoadModel / InitPagedKv ↔ WorkerReady until exactly one
/// worker reports Ready.
fn drive_handshake(
    socket: &zmq::Socket,
    load_model: &Option<LoadModel>,
    registry: &mut Registry,
) -> ControlResult<WorkerReady> {
    loop {
        let identity = match socket.recv_bytes(0) {
            Ok(b) => b,
            Err(e) => {
                return Err(ControlError::Router(format!(
                    "bootstrap recv identity: {}",
                    e
                )));
            }
        };
        let mut last = identity.clone();
        while socket.get_rcvmore().unwrap_or(false) {
            last = socket
                .recv_bytes(0)
                .map_err(|e| ControlError::Router(format!("bootstrap recv body: {}", e)))?;
        }
        if last.as_ptr() == identity.as_ptr() {
            tracing::warn!("bootstrap: dropped single-frame message");
            continue;
        }
        let env: ControlEnvelope<WorkerControlMessage> = rmp_serde::from_slice(&last)
            .map_err(|e| ControlError::Decode(format!("bootstrap envelope: {}", e)))?;

        let now = Instant::now();
        let state = match &env.payload {
            WorkerControlMessage::Hello(_) => WorkerState::Connecting,
            WorkerControlMessage::Progress(p) => p.state,
            WorkerControlMessage::Ready(_) => WorkerState::Ready,
            WorkerControlMessage::MemoryProfile(_) => WorkerState::ProfilingMemory,
            WorkerControlMessage::PagedKvReady(_) => WorkerState::AllocatingRuntime,
            WorkerControlMessage::Heartbeat(hb) => hb.state,
            WorkerControlMessage::Error(e) => e.state,
            _ => WorkerState::Running,
        };
        if let Err(_e) = registry.intern(&identity, now, state) {
            tracing::error!("bootstrap: refusing reconnect from evicted identity");
            continue;
        }

        match env.payload {
            WorkerControlMessage::Hello(hello) => {
                tracing::info!(
                    "WorkerHello: id={} pid={} host={} device={} protocol={}",
                    hello.worker_id,
                    hello.pid,
                    hello.hostname,
                    hello.device,
                    hello.protocol_version,
                );
                // Enforce, don't just log: a mismatched worker build would
                // otherwise handshake fine and fail mid-batch with opaque
                // msgpack decode errors. Fail the bootstrap loudly instead.
                if hello.protocol_version != WORKER_CONTROL_PROTOCOL_VERSION {
                    return Err(ControlError::Router(format!(
                        "worker {} speaks control protocol v{} but scheduler requires v{}; \
                         rebuild/redeploy the mismatched side",
                        hello.worker_id, hello.protocol_version, WORKER_CONTROL_PROTOCOL_VERSION,
                    )));
                }
                send_scheduler_msg(
                    socket,
                    &identity,
                    &SchedulerControlMessage::Hello(SchedulerHello {
                        protocol_version: WORKER_CONTROL_PROTOCOL_VERSION,
                        heartbeat_interval_ms: 1_000,
                    }),
                )?;
                if let Some(cmd) = load_model {
                    tracing::info!(
                        "Sending LoadModel: model_instance_id={} model_type={} path={}",
                        cmd.model_instance_id,
                        cmd.model_type,
                        cmd.model_path,
                    );
                    send_scheduler_msg(
                        socket,
                        &identity,
                        &SchedulerControlMessage::LoadModel(cmd.clone()),
                    )?;
                }
            }
            WorkerControlMessage::Progress(p) => {
                tracing::info!(
                    "WorkerProgress: id={} state={:?} message={}",
                    p.worker_id,
                    p.state,
                    p.message
                );
            }
            WorkerControlMessage::MemoryProfile(profile) => {
                handle_memory_profile(socket, &identity, &profile, load_model)?;
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
            WorkerControlMessage::Heartbeat(hb) => {
                tracing::debug!(
                    "WorkerHeartbeat (bootstrap): id={} state={:?} active={}",
                    hb.worker_id,
                    hb.state,
                    hb.active_requests
                );
            }
            WorkerControlMessage::Ready(ready) => {
                return Ok(ready);
            }
            WorkerControlMessage::Error(e) => {
                return Err(ControlError::Router(format!(
                    "worker {} fatal in {:?}: {}",
                    e.worker_id, e.state, e.message
                )));
            }
            other => {
                tracing::warn!("bootstrap: unexpected message {:?}", other);
            }
        }
    }
}

fn handle_memory_profile(
    socket: &zmq::Socket,
    identity: &[u8],
    profile: &WorkerMemoryProfile,
    load_model: &Option<LoadModel>,
) -> ControlResult<()> {
    tracing::info!(
        "WorkerMemoryProfile: id={} device={} free_after_dummy={} suggested_kv_budget={}",
        profile.worker_id,
        profile.device,
        profile.free_mem_after_dummy_bytes,
        profile.suggested_kv_budget_bytes,
    );
    let Some(cmd) = load_model else { return Ok(()) };
    let Some(block_size) = paged_block_size(cmd) else {
        return Ok(());
    };
    let bytes_per_block = profile.layer_num as u64
        * 2
        * block_size as u64
        * profile.kv_head_num as u64
        * profile.head_size as u64
        * profile.dtype_size as u64;
    if bytes_per_block == 0 {
        return Err(ControlError::Router(
            "invalid worker memory profile: bytes_per_block=0".into(),
        ));
    }
    let num_blocks = (profile.suggested_kv_budget_bytes / bytes_per_block) as u32;
    if num_blocks == 0 {
        return Err(ControlError::Router(format!(
            "insufficient KV budget: budget={} bytes_per_block={}",
            profile.suggested_kv_budget_bytes, bytes_per_block,
        )));
    }
    let max_blocks_per_seq = (cmd.max_model_len as u32).div_ceil(block_size).max(1);
    let init = InitPagedKv {
        model_instance_id: cmd.model_instance_id.clone(),
        block_size,
        initial_num_blocks: num_blocks,
        max_num_blocks: num_blocks,
        max_blocks_per_seq,
        decode_block_request_blocks: 1,
        decode_block_prefetch_margin: 4,
    };
    tracing::info!(
        "Sending InitPagedKv: model_instance_id={} block_size={} blocks={}",
        init.model_instance_id,
        init.block_size,
        init.initial_num_blocks,
    );
    send_scheduler_msg(
        socket,
        identity,
        &SchedulerControlMessage::InitPagedKv(init),
    )
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
) -> ControlResult<()> {
    let env = ControlEnvelope::oneway(msg.clone());
    let bytes = rmp_serde::to_vec(&env)
        .map_err(|e| ControlError::Encode(format!("bootstrap envelope: {}", e)))?;
    socket
        .send(identity, zmq::SNDMORE)
        .map_err(|e| ControlError::Router(format!("bootstrap send identity: {}", e)))?;
    socket
        .send(&bytes, 0)
        .map_err(|e| ControlError::Router(format!("bootstrap send payload: {}", e)))?;
    Ok(())
}

// Suppress unused-import warnings during in-progress wiring.
#[allow(dead_code)]
fn _force_imports(
    _: WorkerHello,
    _: WorkerHeartbeat,
    _: WorkerError,
    _: PagedKvReady,
    _: RequestId,
) {
}
