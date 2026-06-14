//! Engine workflow trait and ResourceContext.
//!
//! The `EngineWorkflow` trait is the deepest seam: mode-specific
//! scheduling, output processing, and control-event handling all
//! live behind this one interface. The engine becomes "receive event,
//! dispatch to workflow, send result."

use async_trait::async_trait;

use crate::application::dispatch::DispatchSystem;
use crate::application::outcomes::ControlOutcome;
use crate::application::scheduler_event::SchedulerEvent;
use crate::config::SchedulerConfig;
use crate::domain::inference_session::table::RequestTable;
use crate::domain::kv_budget::KvBudget;
use crate::error::Result;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::metrics::MetricsRecorder;
use crate::infrastructure::transport::codec::MsgPackCodec;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, WorkerGroup, WorkerId,
};

/// Resources shared across all workflow implementations.
///
/// Flat struct (not sub-aggregates) to avoid nested `&mut` reborrow
/// pain. The engine destructure exactly what each callee needs.
pub struct ResourceContext<'a> {
    pub requests: &'a mut RequestTable,
    pub radix: &'a mut RadixTree,
    pub kv_budget: &'a mut KvBudget,
    pub metrics: &'a MetricsRecorder,
    pub codec: &'a MsgPackCodec,
    pub config: &'a SchedulerConfig,
    pub control_cmd: &'a ControlPlaneCmdTx,
    pub worker_group: &'a WorkerGroup,
    pub default_worker: &'a WorkerId,
}

/// Mode-specific scheduling and output processing.
#[async_trait]
pub trait EngineWorkflow: Send + Sync {
    /// Whether a new batch can be scheduled right now.
    /// LLM: false while a prefill batch is awaiting worker acknowledgement.
    /// Diffusion: false while a batch is in-flight.
    fn can_schedule(&self, requests: &RequestTable) -> bool;

    /// Whether the event loop should poll the worker transport.
    /// LLM: always true when there is pending work.
    /// Diffusion: true when a batch is in-flight.
    fn has_in_flight_batch(&self) -> bool;

    /// Schedule and dispatch one batch.
    async fn try_schedule(
        &mut self,
        ctx: &mut ResourceContext<'_>,
        dispatch: &mut DispatchSystem,
    ) -> Result<()>;

    /// Process a decoded worker step output.
    async fn handle_step_output(
        &mut self,
        ctx: &mut ResourceContext<'_>,
        dispatch: &mut DispatchSystem,
        event: SchedulerEvent,
    ) -> Result<()>;

    /// Handle a control-plane event. Returns a `ControlOutcome` that
    /// the engine then dispatches (fail sessions, terminate, etc.).
    fn handle_control_event(
        &self,
        event: ControlEvent,
        ctx: &mut ResourceContext<'_>,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
        default_worker: &WorkerId,
    ) -> ControlOutcome;
}

mod diffusion;
mod llm;

pub use diffusion::DiffusionWorkflow;
pub use llm::LlmWorkflow;
