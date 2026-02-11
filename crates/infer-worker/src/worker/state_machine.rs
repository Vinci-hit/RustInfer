//! Worker State Machine
//!
//! Defines the 9-state lifecycle of a Worker process and validates transitions.
//!
//! ```text
//! S1: Initializing ──→ S2: Registering ──→ S3: WaitingModel
//!                                               │
//!     ┌─────────────────────────────────────────┘
//!     ↓
//! S4: LoadingModel ──→ S5: Profiling ──→ S6: WaitingKVConfig
//!                                               │
//!     ┌─────────────────────────────────────────┘
//!     ↓
//! S7: InitializingKV ──→ S8: Running ──→ S9: Unloading ──→ S3 (cycle)
//!
//! Any state ──→ Error
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// WorkerState
// ---------------------------------------------------------------------------

/// Worker lifecycle state (lightweight, no heavy data).
///
/// Heavy resources (Model, KVCache) live in the Worker struct.
/// The state machine only tracks which phase the worker is in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerState {
    /// S1: Process just started, initializing CUDA context
    Initializing,

    /// S2: Sending device info to Scheduler, waiting for ack
    Registering,

    /// S3: Idle, waiting for LoadModel command
    WaitingModel,

    /// S4: Loading model weights onto device
    LoadingModel,

    /// S5: Running dummy forward to profile peak memory
    Profiling,

    /// S6: Waiting for Scheduler to send KV cache config
    WaitingKVConfig,

    /// S7: Allocating KV cache blocks on device
    InitializingKV,

    /// S8: Normal operation — processing Forward requests
    Running,

    /// S9: Releasing model, KV cache, and NCCL resources
    Unloading,

    /// Error state (recoverable via Scheduler-initiated restart)
    Error(String),
}

impl WorkerState {
    /// Check whether a transition from `self` to `target` is valid.
    pub fn can_transition_to(&self, target: &WorkerState) -> bool {
        // Any state can transition to Error
        if matches!(target, WorkerState::Error(_)) {
            return true;
        }

        matches!(
            (self, target),
            // Happy path
            (WorkerState::Initializing, WorkerState::Registering)
                | (WorkerState::Registering, WorkerState::WaitingModel)
                | (WorkerState::WaitingModel, WorkerState::LoadingModel)
                | (WorkerState::LoadingModel, WorkerState::Profiling)
                | (WorkerState::Profiling, WorkerState::WaitingKVConfig)
                | (WorkerState::WaitingKVConfig, WorkerState::InitializingKV)
                | (WorkerState::InitializingKV, WorkerState::Running)
                // Running stays Running on Forward (no transition needed)
                | (WorkerState::Running, WorkerState::Unloading)
                // Unload cycle back to WaitingModel
                | (WorkerState::Unloading, WorkerState::WaitingModel)
        )
    }

    /// Attempt to transition to `target`. Returns `Err` if invalid.
    pub fn transition(&mut self, target: WorkerState) -> Result<(), StateTransitionError> {
        if !self.can_transition_to(&target) {
            return Err(StateTransitionError {
                from: self.clone(),
                to: target,
            });
        }
        *self = target;
        Ok(())
    }

    /// Whether the worker is in a state that accepts Forward requests.
    pub fn can_forward(&self) -> bool {
        matches!(self, WorkerState::Running)
    }

    /// Whether the worker is in a terminal/error state.
    pub fn is_error(&self) -> bool {
        matches!(self, WorkerState::Error(_))
    }

    /// Convert to protocol WorkerState for status reporting.
    pub fn to_protocol(&self) -> infer_protocol::WorkerState {
        match self {
            WorkerState::Initializing => infer_protocol::WorkerState::Initializing,
            WorkerState::Registering => infer_protocol::WorkerState::Initializing,
            WorkerState::WaitingModel => infer_protocol::WorkerState::Idle,
            WorkerState::LoadingModel => infer_protocol::WorkerState::LoadingModel,
            WorkerState::Profiling => infer_protocol::WorkerState::Profiling,
            WorkerState::WaitingKVConfig => infer_protocol::WorkerState::Idle,
            WorkerState::InitializingKV => infer_protocol::WorkerState::InitializingKVCache,
            WorkerState::Running => infer_protocol::WorkerState::Inferencing,
            WorkerState::Unloading => infer_protocol::WorkerState::UnloadingModel,
            WorkerState::Error(_) => infer_protocol::WorkerState::Error,
        }
    }

    /// Human-readable label (for logging / TUI).
    pub fn label(&self) -> &str {
        match self {
            WorkerState::Initializing => "Initializing",
            WorkerState::Registering => "Registering",
            WorkerState::WaitingModel => "WaitingModel",
            WorkerState::LoadingModel => "LoadingModel",
            WorkerState::Profiling => "Profiling",
            WorkerState::WaitingKVConfig => "WaitingKVConfig",
            WorkerState::InitializingKV => "InitializingKV",
            WorkerState::Running => "Running",
            WorkerState::Unloading => "Unloading",
            WorkerState::Error(_) => "Error",
        }
    }
}

impl fmt::Display for WorkerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerState::Error(reason) => write!(f, "Error({})", reason),
            other => write!(f, "{}", other.label()),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerEvent
// ---------------------------------------------------------------------------

/// Events that drive state transitions.
///
/// Handlers process these events and call `state.transition()`.
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    /// CUDA context ready, RPC client created
    InitCompleted,

    /// Scheduler acknowledged registration
    RegisterAcked,

    /// Scheduler sent LoadModel command
    LoadModelRequested,

    /// Model weights loaded successfully
    ModelLoaded,

    /// Profile completed, results sent to Scheduler
    ProfileCompleted,

    /// Scheduler sent InitKVCache command
    InitKVCacheRequested,

    /// KV cache blocks allocated
    KVCacheInitialized,

    /// Scheduler sent Unload command
    UnloadRequested,

    /// Resources released, ready for next model
    UnloadCompleted,

    /// An error occurred
    ErrorOccurred(String),
}

impl WorkerEvent {
    /// Return the expected target state for this event.
    pub fn target_state(&self) -> WorkerState {
        match self {
            WorkerEvent::InitCompleted => WorkerState::Registering,
            WorkerEvent::RegisterAcked => WorkerState::WaitingModel,
            WorkerEvent::LoadModelRequested => WorkerState::LoadingModel,
            WorkerEvent::ModelLoaded => WorkerState::Profiling,
            WorkerEvent::ProfileCompleted => WorkerState::WaitingKVConfig,
            WorkerEvent::InitKVCacheRequested => WorkerState::InitializingKV,
            WorkerEvent::KVCacheInitialized => WorkerState::Running,
            WorkerEvent::UnloadRequested => WorkerState::Unloading,
            WorkerEvent::UnloadCompleted => WorkerState::WaitingModel,
            WorkerEvent::ErrorOccurred(reason) => WorkerState::Error(reason.clone()),
        }
    }
}

impl fmt::Display for WorkerEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerEvent::ErrorOccurred(reason) => write!(f, "ErrorOccurred({})", reason),
            other => write!(f, "{:?}", other),
        }
    }
}

// ---------------------------------------------------------------------------
// StateTransitionError
// ---------------------------------------------------------------------------

/// Error returned when an invalid state transition is attempted.
#[derive(Debug, Clone)]
pub struct StateTransitionError {
    pub from: WorkerState,
    pub to: WorkerState,
}

impl fmt::Display for StateTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid state transition: {} -> {}",
            self.from, self.to
        )
    }
}

impl std::error::Error for StateTransitionError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happy_path() {
        let mut state = WorkerState::Initializing;

        // Full lifecycle: Init → Register → WaitModel → Load → Profile
        //   → WaitKV → InitKV → Running → Unload → WaitModel
        let events = vec![
            WorkerEvent::InitCompleted,
            WorkerEvent::RegisterAcked,
            WorkerEvent::LoadModelRequested,
            WorkerEvent::ModelLoaded,
            WorkerEvent::ProfileCompleted,
            WorkerEvent::InitKVCacheRequested,
            WorkerEvent::KVCacheInitialized,
            WorkerEvent::UnloadRequested,
            WorkerEvent::UnloadCompleted,
        ];

        let expected = vec![
            WorkerState::Registering,
            WorkerState::WaitingModel,
            WorkerState::LoadingModel,
            WorkerState::Profiling,
            WorkerState::WaitingKVConfig,
            WorkerState::InitializingKV,
            WorkerState::Running,
            WorkerState::Unloading,
            WorkerState::WaitingModel,
        ];

        for (event, exp) in events.into_iter().zip(expected.into_iter()) {
            let target = event.target_state();
            state.transition(target).unwrap();
            assert_eq!(state, exp);
        }
    }

    #[test]
    fn test_forward_in_running() {
        let state = WorkerState::Running;
        assert!(state.can_forward());
    }

    #[test]
    fn test_forward_in_wrong_state() {
        let state = WorkerState::WaitingModel;
        assert!(!state.can_forward());
    }

    #[test]
    fn test_error_from_any_state() {
        let states = vec![
            WorkerState::Initializing,
            WorkerState::Registering,
            WorkerState::WaitingModel,
            WorkerState::LoadingModel,
            WorkerState::Profiling,
            WorkerState::WaitingKVConfig,
            WorkerState::InitializingKV,
            WorkerState::Running,
            WorkerState::Unloading,
        ];

        for s in states {
            assert!(s.can_transition_to(&WorkerState::Error("test".into())));
        }
    }

    #[test]
    fn test_invalid_transition() {
        let mut state = WorkerState::Initializing;
        let result = state.transition(WorkerState::Running);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.from, WorkerState::Initializing);
        assert_eq!(err.to, WorkerState::Running);
        assert!(err.to_string().contains("invalid state transition"));
    }

    #[test]
    fn test_reload_cycle() {
        let mut state = WorkerState::Running;

        // Running → Unload → WaitModel → LoadModel → ... → Running
        state.transition(WorkerState::Unloading).unwrap();
        state.transition(WorkerState::WaitingModel).unwrap();
        state.transition(WorkerState::LoadingModel).unwrap();
        state.transition(WorkerState::Profiling).unwrap();
        state.transition(WorkerState::WaitingKVConfig).unwrap();
        state.transition(WorkerState::InitializingKV).unwrap();
        state.transition(WorkerState::Running).unwrap();

        assert_eq!(state, WorkerState::Running);
    }

    #[test]
    fn test_to_protocol_mapping() {
        assert!(matches!(
            WorkerState::Running.to_protocol(),
            infer_protocol::WorkerState::Inferencing
        ));
        assert!(matches!(
            WorkerState::WaitingModel.to_protocol(),
            infer_protocol::WorkerState::Idle
        ));
        assert!(matches!(
            WorkerState::Error("oom".into()).to_protocol(),
            infer_protocol::WorkerState::Error
        ));
    }

    #[test]
    fn test_display() {
        assert_eq!(WorkerState::Running.to_string(), "Running");
        assert_eq!(
            WorkerState::Error("OOM".into()).to_string(),
            "Error(OOM)"
        );
    }
}
