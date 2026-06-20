use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::kv::KvView;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;

pub struct Hidden<T: Dtype, D: LlmBackend> {
    pub stream: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend> Hidden<T, D> {
    pub fn num_tokens(&self) -> usize {
        self.stream.shape().as_slice().first().copied().unwrap_or(0)
    }

    pub fn tap_stream(&self, deep: bool) -> OpResult<Tensor<T, D>> {
        let _ = deep;
        Ok(self.stream.clone())
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StageKind {
    Embed,
    Norm,
    Attention,
    Ffn,
    LmHead,
    DecoderBlock,
}

#[derive(Clone, Copy, Debug)]
pub struct LayerRange {
    pub start: usize,
    pub end: usize,
}

impl LayerRange {
    pub fn all(num_layers: usize) -> Self {
        Self {
            start: 0,
            end: num_layers,
        }
    }

    pub fn single(i: usize) -> Self {
        Self {
            start: i,
            end: i + 1,
        }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn for_pp_rank(pp_rank: usize, pp_size: usize, num_layers: usize) -> Self {
        let pp_size = pp_size.max(1);
        let start = num_layers * pp_rank / pp_size;
        let end = num_layers * (pp_rank + 1) / pp_size;
        Self { start, end }
    }
}

pub trait Component<T: Dtype, D: LlmBackend> {
    fn kind(&self) -> StageKind;
    fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        kv: Option<&mut KvView<'_, T, D>>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
}
