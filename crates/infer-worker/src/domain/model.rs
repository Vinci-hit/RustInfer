//! Decoder model trait — the sliceable embed / decode_layers / finalize
//! contract every LLM the runtime drives implements.

use super::component::{Hidden, LayerRange, StageKind};
use super::dtype::Dtype as V2Dtype;
use super::kv::KvView;
use super::ports::OpResult;
use super::ports::backend::LlmBackend;
use super::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub qkv_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_num: usize,
    pub head_dim: usize,
    pub kv_head_num: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub num_shared_experts: usize,
}

impl Default for ModelDims {
    fn default() -> Self {
        Self {
            dim: 0,
            q_dim: 0,
            kv_dim: 0,
            qkv_dim: 0,
            intermediate_size: 0,
            vocab_size: 0,
            head_num: 0,
            head_dim: 0,
            kv_head_num: 0,
            num_layers: 0,
            num_experts: 0,
            experts_per_tok: 0,
            moe_intermediate_size: 0,
            num_shared_experts: 0,
        }
    }
}

impl ModelDims {
    pub fn validate(&self) -> OpResult<()> {
        if self.head_num > 0 && self.head_dim > 0 && self.q_dim != self.head_num * self.head_dim {
            return Err(crate::domain::ports::OpError::Shape(format!(
                "q_dim={} does not equal head_num*head_dim={}",
                self.q_dim,
                self.head_num * self.head_dim
            )));
        }
        if self.kv_head_num > 0
            && self.head_dim > 0
            && self.kv_dim != self.kv_head_num * self.head_dim
        {
            return Err(crate::domain::ports::OpError::Shape(format!(
                "kv_dim={} does not equal kv_head_num*head_dim={}",
                self.kv_dim,
                self.kv_head_num * self.head_dim
            )));
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
}

pub struct Logits<T: V2Dtype, D: LlmBackend>(pub Tensor<T, D>);

pub enum SampleRows<'a> {
    All,
    LastPerSeq,
    Explicit(&'a [i32]),
}

pub trait DecoderModel<T: V2Dtype, D: LlmBackend> {
    fn dims(&self) -> ModelDims;
    fn stages(&self) -> &[StageKind];

    fn embed(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()>;

    fn decode_layers(
        &self,
        range: LayerRange,
        hidden: &mut Hidden<T, D>,
        kv: &mut KvView<'_, T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()>;

    fn finalize(
        &self,
        hidden: &Hidden<T, D>,
        rows: SampleRows<'_>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<Logits<T, D>>;

    fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        kv: &mut KvView<'_, T, D>,
        rows: SampleRows<'_>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<Logits<T, D>> {
        self.embed(input_ids, hidden, ctx)?;
        self.decode_layers(LayerRange::all(self.dims().num_layers), hidden, kv, ctx)?;
        self.finalize(hidden, rows, ctx)
    }
}
