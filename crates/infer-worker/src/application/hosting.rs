use crate::application::runtime::{Runtime, run_layers_for_tap};
use crate::domain::component::LayerRange;
use crate::domain::dtype::Dtype;
use crate::domain::model::{DecoderModel, ModelDims};
use crate::domain::plan::{HiddenTap, StepOutput, StepRequest};
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;

pub trait ErasedRuntime<T: Dtype, D: LlmBackend>: Send {
    fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput>;
    fn run_layers(&mut self, range: LayerRange, req: &StepRequest) -> OpResult<HiddenTap>;
    fn prime_graphs(&mut self) -> OpResult<()>;
    fn dims(&self) -> &ModelDims;
}

impl<T, D, M> ErasedRuntime<T, D> for Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D> + Send,
{
    fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput> {
        Runtime::step(self, req)
    }

    fn run_layers(&mut self, range: LayerRange, req: &StepRequest) -> OpResult<HiddenTap> {
        run_layers_for_tap(self, range, req)
    }

    fn prime_graphs(&mut self) -> OpResult<()> {
        Runtime::prime_graphs(self)
    }

    fn dims(&self) -> &ModelDims {
        &self.dims
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Draft,
    Target,
    EagleHead,
    PipelinePeer,
}

pub struct ModelHost<T: Dtype, D: LlmBackend> {
    pub primary: Box<dyn ErasedRuntime<T, D>>,
    pub aux: Vec<(ModelRole, Box<dyn ErasedRuntime<T, D>>)>,
    pub topology: crate::domain::exec::TopologyShape,
}
