//! Rank-local runtime peers for single-process tensor parallelism.
//!
//! A model owns `Rc`-backed forward scratch and is therefore intentionally not
//! `Send`.  A follower is consequently constructed *inside* its worker thread
//! by [`spawn_follower`]; only this non-generic command handle crosses thread
//! boundaries.  The leader dispatches a command to every follower before it
//! starts the same local operation, then waits for all completions.  That
//! ordering lets all CUDA ranks enter NCCL collectives concurrently while one
//! logical serve loop remains the sole owner of batching decisions.

use std::collections::HashSet;
use std::fmt;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::domain::dtype::Dtype;
use crate::domain::model::DecoderModel;
use crate::domain::plan::StepRequest;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};

use super::{MixedStepTicket, RaggedRowKind, Runtime};

/// An owned, backend-independent operation mirrored to every TP follower.
///
/// Keep this enum non-generic: rank threads may own different concrete CUDA
/// device/model values, while the controller only needs to replicate the
/// logical runtime operation and its host-side inputs.
#[derive(Debug, Clone)]
pub enum RuntimePeerCommand {
    Step(Box<StepRequest>),
    ProfileForward,
    ResizeKvPool {
        num_blocks: usize,
    },
    IssueDecodeAbc {
        req: Box<StepRequest>,
        a_valid_prefix: usize,
        generated_counts: Vec<u32>,
        max_tokens: Vec<u32>,
        ignore_eos: Vec<bool>,
        eos_ids: Vec<i32>,
        async_next_slots: Option<Vec<u32>>,
        reuse_device_control: bool,
    },
    FinalizeDecodeAbc {
        batch: usize,
    },
    IssueFusedAbc {
        req: Box<StepRequest>,
        row_kind: Vec<RaggedRowKind>,
        next_slots: Option<Vec<u32>>,
        c_prefix_rows: usize,
        overlapped: bool,
    },
    FinalizeFusedAbc {
        req: Box<StepRequest>,
        row_kind: Vec<RaggedRowKind>,
    },
    Shutdown,
}

impl RuntimePeerCommand {
    fn name(&self) -> &'static str {
        match self {
            Self::Step(_) => "step",
            Self::ProfileForward => "profile_forward",
            Self::ResizeKvPool { .. } => "resize_kv_pool",
            Self::IssueDecodeAbc { .. } => "issue_decode_abc",
            Self::FinalizeDecodeAbc { .. } => "finalize_decode_abc",
            Self::IssueFusedAbc {
                overlapped: false, ..
            } => "issue_fused_abc",
            Self::IssueFusedAbc {
                overlapped: true, ..
            } => "issue_fused_abc_overlapped",
            Self::FinalizeFusedAbc { .. } => "finalize_fused_abc",
            Self::Shutdown => "shutdown",
        }
    }

    fn phase(&self) -> PeerCommandPhase {
        match self {
            Self::IssueDecodeAbc { .. } => PeerCommandPhase::Begin(PeerPipelineKind::DecodeAbc),
            Self::FinalizeDecodeAbc { .. } => PeerCommandPhase::End(PeerPipelineKind::DecodeAbc),
            Self::IssueFusedAbc { .. } => PeerCommandPhase::Begin(PeerPipelineKind::FusedAbc),
            Self::FinalizeFusedAbc { .. } => PeerCommandPhase::End(PeerPipelineKind::FusedAbc),
            Self::Step(_) | Self::ProfileForward | Self::ResizeKvPool { .. } | Self::Shutdown => {
                PeerCommandPhase::Standalone
            }
        }
    }

    fn watchdog_operation(&self) -> &'static str {
        match self.phase() {
            PeerCommandPhase::Begin(PeerPipelineKind::DecodeAbc) => "decode_abc_pipeline",
            PeerCommandPhase::Begin(PeerPipelineKind::FusedAbc) => "fused_abc_pipeline",
            PeerCommandPhase::Standalone | PeerCommandPhase::End(_) => self.name(),
        }
    }

    fn valid_pipeline_transition(&self, pending: &[PeerPipelineKind]) -> bool {
        match (pending, self) {
            ([], Self::Step(_) | Self::ProfileForward | Self::ResizeKvPool { .. }) => true,
            ([], Self::IssueDecodeAbc { .. } | Self::IssueFusedAbc { .. }) => true,
            ([PeerPipelineKind::DecodeAbc], Self::FinalizeDecodeAbc { .. }) => true,
            (
                [PeerPipelineKind::DecodeAbc],
                Self::IssueFusedAbc {
                    overlapped: true, ..
                },
            ) => true,
            (
                [PeerPipelineKind::DecodeAbc, PeerPipelineKind::FusedAbc],
                Self::FinalizeDecodeAbc { .. },
            ) => true,
            ([PeerPipelineKind::FusedAbc], Self::FinalizeFusedAbc { .. }) => true,
            (_, Self::Shutdown) => false,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeerPipelineKind {
    DecodeAbc,
    FusedAbc,
}

impl PeerPipelineKind {
    fn name(self) -> &'static str {
        match self {
            Self::DecodeAbc => "decode_abc",
            Self::FusedAbc => "fused_abc",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeerCommandPhase {
    Standalone,
    Begin(PeerPipelineKind),
    End(PeerPipelineKind),
}

#[derive(Debug)]
struct RuntimePeerEnvelope {
    sequence: u64,
    command: RuntimePeerCommand,
}

#[derive(Debug, Clone)]
struct RuntimePeerFailure {
    rank: usize,
    operation: &'static str,
    message: String,
}

impl RuntimePeerFailure {
    fn from_op_error(rank: usize, operation: &'static str, error: &OpError) -> Self {
        Self {
            rank,
            operation,
            message: error.to_string(),
        }
    }

    fn into_op_error(self) -> OpError {
        // Once one replica disagrees with the leader, issuing another command
        // could enter a different collective sequence and deadlock the group.
        // Treat every peer-side failure as group-fatal, even if its original
        // local classification was a recoverable shape error.
        OpError::Fatal(format!(
            "TP follower rank {} failed {}: {}",
            self.rank, self.operation, self.message
        ))
    }
}

#[derive(Debug)]
struct RuntimePeerCompletion {
    sequence: u64,
    outcome: Result<(), RuntimePeerFailure>,
}

enum WatchdogCommand {
    Arm {
        sequence: u64,
        operation: &'static str,
        deadline: Instant,
    },
    Disarm {
        sequence: u64,
    },
    Trip {
        sequence: u64,
        rank: usize,
        operation: &'static str,
        message: String,
    },
    Shutdown,
}

/// One permanent fail-stop watchdog for a TP group.
///
/// Blocking CUDA/NCCL calls cannot be safely cancelled from another thread.
/// On a missed group deadline the only race-free recovery is therefore to
/// terminate the worker and let the scheduler detect the lost heartbeat.
pub struct RuntimePeerWatchdog {
    command_tx: Sender<WatchdogCommand>,
    join: Option<JoinHandle<()>>,
    shutdown_on_drop: bool,
}

impl fmt::Debug for RuntimePeerWatchdog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimePeerWatchdog")
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(crate) struct RuntimePeerFailureNotifier {
    command_tx: Sender<WatchdogCommand>,
}

impl RuntimePeerFailureNotifier {
    fn trip(&self, sequence: u64, rank: usize, operation: &'static str, message: String) {
        let _ = self.command_tx.send(WatchdogCommand::Trip {
            sequence,
            rank,
            operation,
            message,
        });
    }
}

impl RuntimePeerWatchdog {
    pub fn fail_stop() -> OpResult<Self> {
        Self::spawn_with_action(|message| {
            tracing::error!(reason = %message, "TP watchdog terminating worker");
            std::process::exit(1);
        })
    }

    fn spawn_with_action<F>(action: F) -> OpResult<Self>
    where
        F: Fn(String) + Send + 'static,
    {
        let (command_tx, command_rx) = mpsc::channel::<WatchdogCommand>();
        let join = thread::Builder::new()
            .name("runtime-tp-watchdog".into())
            .spawn(move || run_watchdog(command_rx, action))
            .map_err(|error| OpError::Kernel(format!("spawn TP watchdog: {error}")))?;
        Ok(Self {
            command_tx,
            join: Some(join),
            shutdown_on_drop: true,
        })
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn notifier(&self) -> RuntimePeerFailureNotifier {
        RuntimePeerFailureNotifier {
            command_tx: self.command_tx.clone(),
        }
    }

    pub fn arm(&self, sequence: u64, operation: &'static str, deadline: Instant) -> OpResult<()> {
        self.command_tx
            .send(WatchdogCommand::Arm {
                sequence,
                operation,
                deadline,
            })
            .map_err(|_| OpError::Fatal("TP watchdog command channel disconnected".into()))
    }

    pub fn disarm(&self, sequence: u64) -> OpResult<()> {
        self.command_tx
            .send(WatchdogCommand::Disarm { sequence })
            .map_err(|_| OpError::Fatal("TP watchdog command channel disconnected".into()))
    }

    fn trip_controller(&self, sequence: u64, operation: &'static str, message: String) {
        let _ = self.command_tx.send(WatchdogCommand::Trip {
            sequence,
            rank: 0,
            operation,
            message,
        });
    }

    fn abandon(mut self) {
        // Keep the watchdog armed while an in-flight/fatal rank thread is
        // detached. Dropping this sender is harmless; follower notifier clones
        // keep the receiver connected until the deadline or an immediate trip.
        self.shutdown_on_drop = false;
        self.join.take();
    }
}

impl Drop for RuntimePeerWatchdog {
    fn drop(&mut self) {
        if !self.shutdown_on_drop {
            return;
        }
        let _ = self.command_tx.send(WatchdogCommand::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn run_watchdog<F>(command_rx: Receiver<WatchdogCommand>, action: F)
where
    F: Fn(String),
{
    let mut armed: Option<(u64, &'static str, Instant)> = None;
    loop {
        let command = match armed {
            Some((sequence, operation, deadline)) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    action(format!(
                        "TP operation {operation} sequence {sequence} exceeded its deadline"
                    ));
                    return;
                }
                match command_rx.recv_timeout(remaining) {
                    Ok(command) => command,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        action(format!(
                            "TP operation {operation} sequence {sequence} exceeded its deadline"
                        ));
                        return;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        action(format!(
                            "TP watchdog disconnected while {operation} sequence {sequence} was active"
                        ));
                        return;
                    }
                }
            }
            None => match command_rx.recv() {
                Ok(command) => command,
                Err(_) => return,
            },
        };

        match command {
            WatchdogCommand::Arm {
                sequence,
                operation,
                deadline,
            } => {
                if let Some((active, active_operation, _)) = armed {
                    action(format!(
                        "TP watchdog received {operation} sequence {sequence} while {active_operation} sequence {active} was active"
                    ));
                    return;
                }
                armed = Some((sequence, operation, deadline));
            }
            WatchdogCommand::Disarm { sequence } => match armed {
                Some((active, _, _)) if active == sequence => armed = None,
                Some((active, operation, _)) => {
                    action(format!(
                        "TP watchdog disarm sequence {sequence} does not match active {operation} sequence {active}"
                    ));
                    return;
                }
                None => {
                    action(format!(
                        "TP watchdog received disarm for inactive sequence {sequence}"
                    ));
                    return;
                }
            },
            WatchdogCommand::Trip {
                sequence,
                rank,
                operation,
                message,
            } => {
                action(format!(
                    "TP rank {rank} failed {operation} sequence {sequence}: {message}"
                ));
                return;
            }
            WatchdogCommand::Shutdown => return,
        }
    }
}

/// Non-generic owner of one rank thread and its ordered command channels.
pub struct RuntimePeerHandle {
    rank: usize,
    command_tx: Sender<RuntimePeerEnvelope>,
    completion_rx: Receiver<RuntimePeerCompletion>,
    startup_rx: Option<Receiver<Result<(), RuntimePeerFailure>>>,
    join: Option<JoinHandle<()>>,
    ready: bool,
}

impl fmt::Debug for RuntimePeerHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimePeerHandle")
            .field("rank", &self.rank)
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}

impl RuntimePeerHandle {
    fn wait_ready(&mut self, timeout: Duration) -> OpResult<()> {
        if self.ready {
            return Ok(());
        }
        let startup_rx = self.startup_rx.take().ok_or_else(|| {
            OpError::Fatal(format!(
                "TP follower rank {} lost its startup receiver",
                self.rank
            ))
        })?;
        match startup_rx.recv_timeout(timeout) {
            Ok(Ok(())) => {
                self.ready = true;
                Ok(())
            }
            Ok(Err(error)) => Err(error.into_op_error()),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(OpError::Fatal(format!(
                "TP follower rank {} startup timed out after {:.3}s",
                self.rank,
                timeout.as_secs_f64()
            ))),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(OpError::Fatal(format!(
                "TP follower rank {} exited during startup",
                self.rank
            ))),
        }
    }

    fn send(&self, sequence: u64, command: RuntimePeerCommand) -> OpResult<()> {
        self.command_tx
            .send(RuntimePeerEnvelope { sequence, command })
            .map_err(|_| {
                OpError::Fatal(format!(
                    "TP follower rank {} command channel disconnected",
                    self.rank
                ))
            })
    }

    fn recv(&self, timeout: Duration) -> OpResult<RuntimePeerCompletion> {
        match self.completion_rx.recv_timeout(timeout) {
            Ok(completion) => Ok(completion),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(OpError::Fatal(format!(
                "TP follower rank {} command timed out after {:.3}s",
                self.rank,
                timeout.as_secs_f64()
            ))),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(OpError::Fatal(format!(
                "TP follower rank {} completion channel disconnected",
                self.rank
            ))),
        }
    }
}

/// Spawn one follower without moving a `Runtime` across threads.
///
/// `factory(init)` executes in the new rank thread.  This is important for
/// CUDA/NCCL initialization and also permits `Runtime`/model implementations
/// containing `Rc` and other deliberately `!Send` state.  Startup completion
/// is observed later by [`RuntimePeerGroup::wait_ready`], so callers can spawn
/// every rank and initialize the leader concurrently with blocking
/// `ncclCommInitRank` calls.
pub fn spawn_follower<T, D, M, I, F>(
    rank: usize,
    factory: F,
    init: I,
) -> OpResult<RuntimePeerHandle>
where
    T: Dtype + 'static,
    D: LlmBackend + 'static,
    M: DecoderModel<T, D> + 'static,
    I: Send + 'static,
    F: FnOnce(I) -> OpResult<Runtime<T, D, M>> + Send + 'static,
{
    spawn_follower_inner(rank, factory, init, None)
}

/// Production variant that trips the group watchdog as soon as a follower
/// fails, even if rank 0 is blocked inside its matching CUDA/NCCL operation.
#[cfg(feature = "cuda")]
pub(crate) fn spawn_monitored_follower<T, D, M, I, F>(
    rank: usize,
    factory: F,
    init: I,
    failure_notifier: RuntimePeerFailureNotifier,
) -> OpResult<RuntimePeerHandle>
where
    T: Dtype + 'static,
    D: LlmBackend + 'static,
    M: DecoderModel<T, D> + 'static,
    I: Send + 'static,
    F: FnOnce(I) -> OpResult<Runtime<T, D, M>> + Send + 'static,
{
    spawn_follower_inner(rank, factory, init, Some(failure_notifier))
}

fn spawn_follower_inner<T, D, M, I, F>(
    rank: usize,
    factory: F,
    init: I,
    failure_notifier: Option<RuntimePeerFailureNotifier>,
) -> OpResult<RuntimePeerHandle>
where
    T: Dtype + 'static,
    D: LlmBackend + 'static,
    M: DecoderModel<T, D> + 'static,
    I: Send + 'static,
    F: FnOnce(I) -> OpResult<Runtime<T, D, M>> + Send + 'static,
{
    let (command_tx, command_rx) = mpsc::channel();
    let (completion_tx, completion_rx) = mpsc::channel();
    let (startup_tx, startup_rx) = mpsc::sync_channel(1);
    let join = thread::Builder::new()
        .name(format!("runtime-tp-rank-{rank}"))
        .spawn(move || {
            let mut runtime = match factory(init) {
                Ok(runtime) => runtime,
                Err(error) => {
                    if let Some(notifier) = &failure_notifier {
                        notifier.trip(0, rank, "startup", error.to_string());
                    }
                    let _ = startup_tx.send(Err(RuntimePeerFailure::from_op_error(
                        rank, "startup", &error,
                    )));
                    return;
                }
            };
            if startup_tx.send(Ok(())).is_err() {
                return;
            }
            run_follower(
                rank,
                &mut runtime,
                command_rx,
                completion_tx,
                failure_notifier,
            );
        })
        .map_err(|error| OpError::Kernel(format!("spawn TP follower rank {rank}: {error}")))?;

    Ok(RuntimePeerHandle {
        rank,
        command_tx,
        completion_rx,
        startup_rx: Some(startup_rx),
        join: Some(join),
        ready: false,
    })
}

fn run_follower<T, D, M>(
    rank: usize,
    runtime: &mut Runtime<T, D, M>,
    command_rx: Receiver<RuntimePeerEnvelope>,
    completion_tx: Sender<RuntimePeerCompletion>,
    failure_notifier: Option<RuntimePeerFailureNotifier>,
) where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    let mut mixed_ticket: Option<MixedStepTicket> = None;
    while let Ok(envelope) = command_rx.recv() {
        let operation = envelope.command.name();
        let result = match envelope.command {
            RuntimePeerCommand::Step(req) => runtime.step(&req).map(|_| ()),
            RuntimePeerCommand::ProfileForward => runtime.profile_forward(),
            RuntimePeerCommand::ResizeKvPool { num_blocks } => runtime.resize_kv_pool(num_blocks),
            RuntimePeerCommand::IssueDecodeAbc {
                req,
                a_valid_prefix,
                generated_counts,
                max_tokens,
                ignore_eos,
                eos_ids,
                async_next_slots,
                reuse_device_control,
            } => runtime.issue_decode_abc(
                &req,
                a_valid_prefix,
                &generated_counts,
                &max_tokens,
                &ignore_eos,
                &eos_ids,
                async_next_slots.as_deref(),
                reuse_device_control,
            ),
            RuntimePeerCommand::FinalizeDecodeAbc { batch } => {
                runtime.finalize_decode_abc(batch).map(|_| ())
            }
            RuntimePeerCommand::IssueFusedAbc {
                req,
                row_kind,
                next_slots,
                c_prefix_rows,
                overlapped,
            } => {
                if mixed_ticket.is_some() {
                    Err(OpError::Fatal(
                        "TP follower received a second fused issue before finalize".into(),
                    ))
                } else {
                    let ticket = if overlapped {
                        runtime.issue_fused_abc_overlapped(
                            &req,
                            &row_kind,
                            next_slots.as_deref(),
                            c_prefix_rows,
                        )
                    } else {
                        runtime.issue_fused_abc(&req, &row_kind, next_slots.as_deref())
                    };
                    ticket.map(|ticket| mixed_ticket = Some(ticket))
                }
            }
            RuntimePeerCommand::FinalizeFusedAbc { req, row_kind } => match mixed_ticket.take() {
                Some(ticket) => runtime
                    .finalize_fused_abc(ticket, &req, &row_kind)
                    .map(|_| ()),
                None => Err(OpError::Fatal(
                    "TP follower received fused finalize without an issued step".into(),
                )),
            },
            RuntimePeerCommand::Shutdown => break,
        };
        let fatal = result.as_ref().is_err_and(|error| error.is_fatal());
        if let Err(error) = &result {
            tracing::error!(
                rank,
                sequence = envelope.sequence,
                operation,
                error = %error,
                "TP follower command failed"
            );
            if let Some(notifier) = &failure_notifier {
                notifier.trip(envelope.sequence, rank, operation, error.to_string());
            }
        }
        let outcome =
            result.map_err(|error| RuntimePeerFailure::from_op_error(rank, operation, &error));
        if completion_tx
            .send(RuntimePeerCompletion {
                sequence: envelope.sequence,
                outcome,
            })
            .is_err()
        {
            break;
        }
        if fatal {
            break;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingPeerCall {
    sequence: u64,
    deadline: Instant,
    watchdog_sequence: u64,
    phase: PeerCommandPhase,
}

#[derive(Debug, Clone, Copy)]
struct PendingPipelineSpan {
    watchdog_sequence: u64,
    deadline: Instant,
}

/// Ordered set of TP follower ranks attached to the leader runtime.
#[derive(Debug)]
pub struct RuntimePeerGroup {
    peers: Vec<RuntimePeerHandle>,
    next_sequence: u64,
    in_flight: Option<u64>,
    pending_pipelines: Vec<PeerPipelineKind>,
    pipeline_span: Option<PendingPipelineSpan>,
    poisoned: Option<String>,
    startup_timeout: Duration,
    timeout: Duration,
    watchdog: Option<RuntimePeerWatchdog>,
}

impl RuntimePeerGroup {
    pub fn new(peers: Vec<RuntimePeerHandle>) -> OpResult<Self> {
        Self::with_timeout(peers, Duration::from_secs(120))
    }

    pub fn with_timeout(peers: Vec<RuntimePeerHandle>, timeout: Duration) -> OpResult<Self> {
        Self::build(peers, timeout, timeout, None)
    }

    pub fn with_watchdog(
        peers: Vec<RuntimePeerHandle>,
        startup_timeout: Duration,
        operation_timeout: Duration,
        watchdog: RuntimePeerWatchdog,
    ) -> OpResult<Self> {
        Self::build(peers, startup_timeout, operation_timeout, Some(watchdog))
    }

    fn build(
        mut peers: Vec<RuntimePeerHandle>,
        startup_timeout: Duration,
        operation_timeout: Duration,
        watchdog: Option<RuntimePeerWatchdog>,
    ) -> OpResult<Self> {
        if peers.is_empty() {
            return Err(OpError::Shape(
                "TP runtime peer group requires at least one follower".into(),
            ));
        }
        if startup_timeout.is_zero() {
            return Err(OpError::Shape(
                "TP runtime peer startup timeout must be greater than zero".into(),
            ));
        }
        if operation_timeout.is_zero() {
            return Err(OpError::Shape(
                "TP runtime peer operation timeout must be greater than zero".into(),
            ));
        }
        peers.sort_unstable_by_key(|peer| peer.rank);
        let mut ranks = HashSet::with_capacity(peers.len());
        for peer in &peers {
            if !ranks.insert(peer.rank) {
                return Err(OpError::Shape(format!(
                    "duplicate TP follower rank {}",
                    peer.rank
                )));
            }
        }
        Ok(Self {
            peers,
            next_sequence: 1,
            in_flight: None,
            pending_pipelines: Vec::new(),
            pipeline_span: None,
            poisoned: None,
            startup_timeout,
            timeout: operation_timeout,
            watchdog,
        })
    }

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn follower_ranks(&self) -> Vec<usize> {
        self.peers.iter().map(|peer| peer.rank).collect()
    }

    /// Wait until every thread-local factory has constructed its Runtime.
    pub fn wait_ready(&mut self) -> OpResult<()> {
        if let Some(message) = &self.poisoned {
            return Err(OpError::Fatal(message.clone()));
        }
        let deadline = Instant::now()
            .checked_add(self.startup_timeout)
            .ok_or_else(|| OpError::Fatal("TP startup deadline overflowed Instant".into()))?;
        if let Some(watchdog) = &self.watchdog {
            watchdog.arm(0, "startup", deadline)?;
        }
        for peer in &mut self.peers {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let error = OpError::Fatal(format!(
                    "TP follower startup timed out after {:.3}s",
                    self.startup_timeout.as_secs_f64()
                ));
                if let Some(watchdog) = &self.watchdog {
                    watchdog.trip_controller(0, "startup", error.to_string());
                }
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
            if let Err(error) = peer.wait_ready(remaining) {
                if let Some(watchdog) = &self.watchdog {
                    watchdog.trip_controller(0, "startup", error.to_string());
                }
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
        }
        if let Some(watchdog) = &self.watchdog {
            watchdog.disarm(0)?;
        }
        Ok(())
    }

    pub(crate) fn dispatch(&mut self, command: RuntimePeerCommand) -> OpResult<PendingPeerCall> {
        if let Some(message) = &self.poisoned {
            return Err(OpError::Fatal(message.clone()));
        }
        if let Some(sequence) = self.in_flight {
            return Err(OpError::Fatal(format!(
                "TP runtime peer command {sequence} is still in flight"
            )));
        }
        if self.peers.iter().any(|peer| !peer.ready) {
            return Err(OpError::Fatal(
                "TP runtime peer group used before wait_ready".into(),
            ));
        }

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        if !command.valid_pipeline_transition(&self.pending_pipelines) {
            let error = OpError::Fatal(format!(
                "invalid TP async pipeline transition: pending={:?}, command={}",
                self.pending_pipelines,
                command.name()
            ));
            if let Some(watchdog) = &self.watchdog {
                let watchdog_sequence = self
                    .pipeline_span
                    .map(|span| span.watchdog_sequence)
                    .unwrap_or(sequence);
                watchdog.trip_controller(watchdog_sequence, "async_pipeline", error.to_string());
            }
            self.poisoned = Some(error.to_string());
            return Err(error);
        }
        let phase = command.phase();
        let operation = command.watchdog_operation();
        let new_deadline = || {
            Instant::now()
                .checked_add(self.timeout)
                .ok_or_else(|| OpError::Fatal("TP runtime peer deadline overflowed Instant".into()))
        };
        let (deadline, watchdog_sequence, arm_watchdog) = match phase {
            PeerCommandPhase::Standalone if self.pending_pipelines.is_empty() => {
                let deadline = new_deadline()?;
                (deadline, sequence, true)
            }
            PeerCommandPhase::Standalone => {
                let span = self
                    .pipeline_span
                    .expect("pending TP pipelines must own a watchdog span");
                let error = OpError::Fatal(format!(
                    "TP async pipeline must be finalized before dispatching {}",
                    command.name()
                ));
                if let Some(watchdog) = &self.watchdog {
                    watchdog.trip_controller(
                        span.watchdog_sequence,
                        "async_pipeline",
                        error.to_string(),
                    );
                }
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
            PeerCommandPhase::Begin(kind) if self.pending_pipelines.contains(&kind) => {
                let span = self
                    .pipeline_span
                    .expect("pending TP pipelines must own a watchdog span");
                let error = OpError::Fatal(format!(
                    "TP {} issue was dispatched twice without finalize",
                    kind.name()
                ));
                if let Some(watchdog) = &self.watchdog {
                    watchdog.trip_controller(
                        span.watchdog_sequence,
                        kind.name(),
                        error.to_string(),
                    );
                }
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
            PeerCommandPhase::Begin(_) if self.pending_pipelines.is_empty() => {
                let deadline = new_deadline()?;
                (deadline, sequence, true)
            }
            PeerCommandPhase::Begin(_) => {
                let span = self
                    .pipeline_span
                    .expect("pending TP pipelines must own a watchdog span");
                (span.deadline, span.watchdog_sequence, false)
            }
            PeerCommandPhase::End(kind) if self.pending_pipelines.contains(&kind) => {
                let span = self
                    .pipeline_span
                    .expect("pending TP pipelines must own a watchdog span");
                (span.deadline, span.watchdog_sequence, false)
            }
            PeerCommandPhase::End(kind) => {
                let error = OpError::Fatal(format!(
                    "TP {} finalize was dispatched without a pending issue",
                    kind.name()
                ));
                if let Some(span) = self.pipeline_span
                    && let Some(watchdog) = &self.watchdog
                {
                    watchdog.trip_controller(
                        span.watchdog_sequence,
                        kind.name(),
                        error.to_string(),
                    );
                }
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
        };
        if arm_watchdog && let Some(watchdog) = &self.watchdog {
            watchdog.arm(watchdog_sequence, operation, deadline)?;
        }
        if let PeerCommandPhase::Begin(kind) = phase {
            if self.pending_pipelines.is_empty() {
                self.pipeline_span = Some(PendingPipelineSpan {
                    watchdog_sequence,
                    deadline,
                });
            }
            self.pending_pipelines.push(kind);
        }
        for peer in &self.peers {
            if let Err(error) = peer.send(sequence, command.clone()) {
                if let Some(watchdog) = &self.watchdog {
                    watchdog.trip_controller(watchdog_sequence, operation, error.to_string());
                }
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
        }
        self.in_flight = Some(sequence);
        Ok(PendingPeerCall {
            sequence,
            deadline,
            watchdog_sequence,
            phase,
        })
    }

    pub(crate) fn wait(&mut self, pending: PendingPeerCall) -> OpResult<()> {
        if self.in_flight != Some(pending.sequence) {
            let error = OpError::Fatal(format!(
                "TP runtime peer completion {} does not match in-flight {:?}",
                pending.sequence, self.in_flight
            ));
            if let Some(watchdog) = &self.watchdog {
                watchdog.trip_controller(
                    pending.watchdog_sequence,
                    "peer completion",
                    error.to_string(),
                );
            }
            self.poisoned = Some(error.to_string());
            return Err(error);
        }

        let mut first_error: Option<OpError> = None;
        for peer in &self.peers {
            let remaining = pending.deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if first_error.is_none() {
                    first_error = Some(OpError::Fatal(format!(
                        "TP command {} timed out after {:.3}s",
                        pending.sequence,
                        self.timeout.as_secs_f64()
                    )));
                }
                break;
            }
            let recv_result = peer.recv(remaining);
            match recv_result {
                Ok(completion) if completion.sequence == pending.sequence => {
                    if let Err(error) = completion.outcome
                        && first_error.is_none()
                    {
                        first_error = Some(error.into_op_error());
                    }
                }
                Ok(completion) => {
                    if first_error.is_none() {
                        first_error = Some(OpError::Fatal(format!(
                            "TP follower rank {} completed sequence {}, expected {}",
                            peer.rank, completion.sequence, pending.sequence
                        )));
                    }
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        let should_disarm = if first_error.is_none() {
            match pending.phase {
                PeerCommandPhase::Standalone => true,
                PeerCommandPhase::Begin(_) => false,
                PeerCommandPhase::End(kind) => {
                    let index = self
                        .pending_pipelines
                        .iter()
                        .position(|pending| *pending == kind)
                        .expect("finalized TP pipeline must still be pending");
                    self.pending_pipelines.remove(index);
                    if self.pending_pipelines.is_empty() {
                        self.pipeline_span = None;
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            matches!(pending.phase, PeerCommandPhase::Standalone)
        };
        if should_disarm
            && let Some(watchdog) = &self.watchdog
            && let Err(error) = watchdog.disarm(pending.watchdog_sequence)
            && first_error.is_none()
        {
            first_error = Some(error);
        }
        self.in_flight = None;
        if let Some(error) = first_error {
            if let Some(watchdog) = &self.watchdog {
                watchdog.trip_controller(
                    pending.watchdog_sequence,
                    "peer completion",
                    error.to_string(),
                );
            }
            self.poisoned = Some(error.to_string());
            Err(error)
        } else {
            Ok(())
        }
    }

    pub(crate) fn poison_leader(&mut self, operation: &'static str, error: &OpError) -> OpError {
        let error = OpError::Fatal(format!("TP leader rank 0 failed {operation}: {error}"));
        if let Some(watchdog) = &self.watchdog {
            let sequence = self
                .pipeline_span
                .map(|span| span.watchdog_sequence)
                .unwrap_or_else(|| self.next_sequence.saturating_sub(1).max(1));
            watchdog.trip_controller(sequence, operation, error.to_string());
        }
        self.poisoned = Some(error.to_string());
        error
    }

    fn safe_to_join_on_drop(&self) -> bool {
        self.in_flight.is_none()
            && self.pending_pipelines.is_empty()
            && self.poisoned.is_none()
            && self.peers.iter().all(|peer| peer.ready)
    }
}

impl Drop for RuntimePeerGroup {
    fn drop(&mut self) {
        for peer in &self.peers {
            let _ = peer.send(0, RuntimePeerCommand::Shutdown);
        }

        // Joining an idle, fully initialized group gives deterministic normal
        // teardown.  A peer stuck in NCCL startup/an in-flight collective must
        // instead be aborted by the process/group failure path; blocking Drop
        // here would hide the original failure forever.
        let safe_to_join = self.safe_to_join_on_drop();
        if safe_to_join {
            for peer in &mut self.peers {
                if let Some(join) = peer.join.take() {
                    let _ = join.join();
                }
            }
        }
        if safe_to_join {
            self.watchdog.take();
        } else if let Some(watchdog) = self.watchdog.take() {
            watchdog.abandon();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::plan::StopCriteria;

    fn empty_step_request() -> StepRequest {
        StepRequest {
            seqs: Vec::new(),
            sampling: Vec::new(),
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: Vec::new(),
                max_tokens: Vec::new(),
                ignore_eos: Vec::new(),
            },
            draft_tokens: Vec::new(),
        }
    }

    #[test]
    fn watchdog_deadline_trips_while_controller_is_blocked() {
        let (trip_tx, trip_rx) = mpsc::sync_channel(1);
        let watchdog = RuntimePeerWatchdog::spawn_with_action(move |message| {
            let _ = trip_tx.send(message);
        })
        .unwrap();
        watchdog
            .arm(
                7,
                "blocked_test",
                Instant::now() + Duration::from_millis(20),
            )
            .unwrap();

        let message = trip_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(message.contains("blocked_test sequence 7"));
        assert!(message.contains("deadline"));
    }

    #[test]
    fn follower_failure_trips_watchdog_immediately() {
        let (trip_tx, trip_rx) = mpsc::sync_channel(1);
        let watchdog = RuntimePeerWatchdog::spawn_with_action(move |message| {
            let _ = trip_tx.send(message);
        })
        .unwrap();
        watchdog
            .arm(9, "step", Instant::now() + Duration::from_secs(1))
            .unwrap();
        watchdog
            .notifier()
            .trip(9, 1, "step", "synthetic failure".into());

        let message = trip_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(message.contains("rank 1 failed step sequence 9"));
        assert!(message.contains("synthetic failure"));
    }

    #[test]
    fn startup_failure_trips_watchdog_with_another_rank_stuck() {
        let (trip_tx, trip_rx) = mpsc::sync_channel(1);
        let watchdog = RuntimePeerWatchdog::spawn_with_action(move |message| {
            let _ = trip_tx.send(message);
        })
        .unwrap();

        let (command_tx1, _command_rx1) = mpsc::channel();
        let (_completion_tx1, completion_rx1) = mpsc::channel();
        let (startup_tx1, startup_rx1) = mpsc::sync_channel(1);
        startup_tx1
            .send(Err(RuntimePeerFailure {
                rank: 1,
                operation: "startup",
                message: "synthetic startup failure".into(),
            }))
            .unwrap();
        let failed = RuntimePeerHandle {
            rank: 1,
            command_tx: command_tx1,
            completion_rx: completion_rx1,
            startup_rx: Some(startup_rx1),
            join: None,
            ready: false,
        };

        let (command_tx2, _command_rx2) = mpsc::channel();
        let (_completion_tx2, completion_rx2) = mpsc::channel();
        let (stuck_startup_tx, stuck_startup_rx) = mpsc::sync_channel(1);
        let stuck = RuntimePeerHandle {
            rank: 2,
            command_tx: command_tx2,
            completion_rx: completion_rx2,
            startup_rx: Some(stuck_startup_rx),
            join: None,
            ready: false,
        };

        let mut group = RuntimePeerGroup::with_watchdog(
            vec![failed, stuck],
            Duration::from_secs(1),
            Duration::from_secs(1),
            watchdog,
        )
        .unwrap();
        let error = group.wait_ready().unwrap_err();
        assert!(error.is_fatal());
        let message = trip_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(message.contains("rank 0 failed startup sequence 0"));
        assert!(message.contains("synthetic startup failure"));
        drop(group);
        drop(stuck_startup_tx);
    }

    #[test]
    fn peer_timeout_is_measured_from_dispatch() {
        let (command_tx, command_rx) = mpsc::channel::<RuntimePeerEnvelope>();
        let (completion_tx, completion_rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::sync_channel(1);
        let (received_tx, received_rx) = mpsc::sync_channel(1);
        let join = thread::spawn(move || {
            startup_tx.send(Ok(())).unwrap();
            let envelope = command_rx.recv().unwrap();
            received_tx.send(()).unwrap();
            thread::sleep(Duration::from_millis(120));
            let _ = completion_tx.send(RuntimePeerCompletion {
                sequence: envelope.sequence,
                outcome: Ok(()),
            });
        });
        let follower = RuntimePeerHandle {
            rank: 1,
            command_tx,
            completion_rx,
            startup_rx: Some(startup_rx),
            join: Some(join),
            ready: false,
        };
        let timeout = Duration::from_millis(80);
        let mut group = RuntimePeerGroup::with_timeout(vec![follower], timeout).unwrap();
        group.wait_ready().unwrap();

        let pending = group.dispatch(RuntimePeerCommand::ProfileForward).unwrap();
        received_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        thread::sleep(Duration::from_millis(60));
        let error = group.wait(pending).unwrap_err();

        assert!(error.is_fatal());
        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn overlapped_pipelines_share_watchdog_until_both_finalize() {
        let (command_tx, command_rx) = mpsc::channel::<RuntimePeerEnvelope>();
        let (completion_tx, completion_rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::sync_channel(1);
        let join = thread::spawn(move || {
            startup_tx.send(Ok(())).unwrap();
            for _ in 0..4 {
                let envelope = command_rx.recv().unwrap();
                completion_tx
                    .send(RuntimePeerCompletion {
                        sequence: envelope.sequence,
                        outcome: Ok(()),
                    })
                    .unwrap();
            }
        });
        let follower = RuntimePeerHandle {
            rank: 1,
            command_tx,
            completion_rx,
            startup_rx: Some(startup_rx),
            join: Some(join),
            ready: false,
        };
        let mut group = RuntimePeerGroup::new(vec![follower]).unwrap();
        group.wait_ready().unwrap();

        let issue = group
            .dispatch(RuntimePeerCommand::IssueDecodeAbc {
                req: Box::new(empty_step_request()),
                a_valid_prefix: 0,
                generated_counts: Vec::new(),
                max_tokens: Vec::new(),
                ignore_eos: Vec::new(),
                eos_ids: Vec::new(),
                async_next_slots: None,
                reuse_device_control: false,
            })
            .unwrap();
        let watchdog_sequence = issue.watchdog_sequence;
        group.wait(issue).unwrap();
        let span = group
            .pipeline_span
            .expect("issue must keep the logical pipeline pending");
        assert_eq!(group.pending_pipelines, vec![PeerPipelineKind::DecodeAbc]);
        assert_eq!(span.watchdog_sequence, watchdog_sequence);
        assert!(!group.safe_to_join_on_drop());

        let fused_issue = group
            .dispatch(RuntimePeerCommand::IssueFusedAbc {
                req: Box::new(empty_step_request()),
                row_kind: Vec::new(),
                next_slots: None,
                c_prefix_rows: 0,
                overlapped: true,
            })
            .unwrap();
        assert_eq!(fused_issue.watchdog_sequence, watchdog_sequence);
        assert_eq!(fused_issue.deadline, span.deadline);
        group.wait(fused_issue).unwrap();
        assert_eq!(
            group.pending_pipelines,
            vec![PeerPipelineKind::DecodeAbc, PeerPipelineKind::FusedAbc]
        );

        let decode_finalize = group
            .dispatch(RuntimePeerCommand::FinalizeDecodeAbc { batch: 0 })
            .unwrap();
        assert_eq!(decode_finalize.watchdog_sequence, watchdog_sequence);
        assert_eq!(decode_finalize.deadline, span.deadline);
        group.wait(decode_finalize).unwrap();
        assert_eq!(group.pending_pipelines, vec![PeerPipelineKind::FusedAbc]);
        assert!(!group.safe_to_join_on_drop());

        let fused_finalize = group
            .dispatch(RuntimePeerCommand::FinalizeFusedAbc {
                req: Box::new(empty_step_request()),
                row_kind: Vec::new(),
            })
            .unwrap();
        assert_eq!(fused_finalize.watchdog_sequence, watchdog_sequence);
        assert_eq!(fused_finalize.deadline, span.deadline);
        group.wait(fused_finalize).unwrap();
        assert!(group.pending_pipelines.is_empty());
        assert!(group.pipeline_span.is_none());
        assert!(group.safe_to_join_on_drop());
    }
}
