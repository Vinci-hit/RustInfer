//! Compile-time serving extension. Ordinary serving has no draft state.
use crate::application::runtime::Runtime;
use crate::application::worker_state::{ActiveSeqMap, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::transport::{control_pump::ControlPump, data_pump::DataPump};
use half::bf16;
use infer_protocol::scheduler_to_worker_data::PrefillBatchCmd;

pub struct ServingStep<'a, M: DecoderModel<bf16, Cuda>> {
    pub runner: &'a mut Runtime<bf16, Cuda, M>,
    pub active: &'a mut ActiveSeqMap,
    pub prefilling: &'a mut PrefillSeqMap,
    pub allocator: &'a mut GlobalKvAllocator,
    pub control: &'a ControlPump,
    pub data: &'a DataPump,
    pub eos_ids: &'a [i32],
    pub prefills: &'a mut Vec<PrefillBatchCmd>,
}

pub trait ServingExecution<M: DecoderModel<bf16, Cuda>> {
    const SPECULATIVE: bool;
    fn prepare(&mut self, _runner: &Runtime<bf16, Cuda, M>) -> OpResult<()> {
        Ok(())
    }
    fn step(&mut self, ctx: ServingStep<'_, M>) -> OpResult<()>;
}

pub struct OrdinaryExecution;
impl<M: DecoderModel<bf16, Cuda>> ServingExecution<M> for OrdinaryExecution {
    const SPECULATIVE: bool = false;
    fn step(&mut self, _ctx: ServingStep<'_, M>) -> OpResult<()> {
        Err(OpError::unsupported(
            "ordinary execution",
            "speculative serving entrypoint",
        ))
    }
}
