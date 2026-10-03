use super::block_quant_projection::BlockQuantProjection;
use super::linear::Linear;
use infer_core::component::Hidden;
use infer_core::dtype::Dtype;
use infer_core::exec::{ExecDevice, ExecScope, RankPair, StepCtx};
use infer_core::ports::backend::LlmBackend;
use infer_core::ports::{CollectiveOps, CommAxis, OpError, OpResult, ReduceOp, VocabOps};
use infer_core::quantized::BlockQuantWeight;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

#[derive(Clone)]
pub enum EmbeddingWeight<T: Dtype, D: LlmBackend> {
    Dense(Tensor<T, D>),
    BlockQuant(BlockQuantWeight<D>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingParallelism {
    Replicated {
        tp: RankPair,
    },
    Vocab {
        tp: RankPair,
        vocab_start: usize,
        global_vocab_size: usize,
    },
}

impl EmbeddingParallelism {
    pub const SINGLE: Self = Self::Replicated {
        tp: RankPair { rank: 0, size: 1 },
    };

    pub const fn tp(self) -> RankPair {
        match self {
            Self::Replicated { tp } | Self::Vocab { tp, .. } => tp,
        }
    }
}

impl Default for EmbeddingParallelism {
    fn default() -> Self {
        Self::SINGLE
    }
}

/// Token embedding table. Initializes the residual stream (`hidden.stream`)
/// from input token ids. Not a `Component` — embedding runs once at the model
/// boundary (`DecoderModel::embed`), not inside the per-layer stage list.
pub struct Embed<T: Dtype, D: LlmBackend> {
    weight: EmbeddingWeight<T, D>,
    parallelism: EmbeddingParallelism,
}

impl<T: Dtype, D: LlmBackend> Embed<T, D> {
    pub fn new(table: Tensor<T, D>) -> Self {
        Self {
            weight: EmbeddingWeight::Dense(table),
            parallelism: EmbeddingParallelism::default(),
        }
    }

    pub fn from_block_quant(weight: BlockQuantWeight<D>) -> Self {
        Self {
            weight: EmbeddingWeight::BlockQuant(weight),
            parallelism: EmbeddingParallelism::SINGLE,
        }
    }

    pub fn weight(&self) -> &EmbeddingWeight<T, D> {
        &self.weight
    }
    pub fn shape(&self) -> Shape {
        match &self.weight {
            EmbeddingWeight::Dense(w) => *w.shape(),
            EmbeddingWeight::BlockQuant(w) => Shape::from_slice(&w.layout().shape()),
        }
    }
    pub fn device(&self) -> &D {
        match &self.weight {
            EmbeddingWeight::Dense(w) => w.device(),
            EmbeddingWeight::BlockQuant(w) => w.device(),
        }
    }
    pub fn as_dense(&self) -> Option<&Tensor<T, D>> {
        match &self.weight {
            EmbeddingWeight::Dense(w) => Some(w),
            EmbeddingWeight::BlockQuant(_) => None,
        }
    }
    pub fn require_dense(&self) -> OpResult<&Tensor<T, D>> {
        self.as_dense().ok_or_else(|| {
            OpError::unsupported(
                self.device().name(),
                "dense embedding access on block-quantized weights",
            )
        })
    }
    pub fn shallow_clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            parallelism: self.parallelism,
        }
    }

    /// Explicit tied-weight construction for TP1. Never inferred from shape.
    /// Existing sharded dense loaders retain their vocabulary-parallel path.
    pub fn shared_linear(&self) -> OpResult<Linear<T, D>> {
        if self.parallelism != EmbeddingParallelism::SINGLE {
            return Err(OpError::unsupported(
                "embedding",
                "shared_linear requires replicated TP1",
            ));
        }
        match &self.weight {
            EmbeddingWeight::Dense(w) => Ok(Linear::new(w.clone(), None)),
            EmbeddingWeight::BlockQuant(w) => {
                Linear::from_block_quant(BlockQuantProjection::try_new(vec![w.clone()])?, None)
            }
        }
    }

    pub fn with_parallelism(mut self, parallelism: EmbeddingParallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn parallelism(&self) -> EmbeddingParallelism {
        self.parallelism
    }

    pub fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        if let EmbeddingWeight::BlockQuant(w) = &self.weight {
            if self.parallelism != EmbeddingParallelism::SINGLE
                || ctx.scope().topology().tp != (RankPair { rank: 0, size: 1 })
            {
                return Err(OpError::unsupported(
                    "block-quantized embedding",
                    "tensor parallelism",
                ));
            }
            if input_ids.ndim() != 1
                || hidden.stream.shape().as_slice() != [input_ids.numel(), w.layout().shape()[1]]
            {
                return Err(OpError::Shape(
                    "block-quantized embedding input/output shape mismatch".into(),
                ));
            }
            let device = <D as ExecDevice>::device_id(ctx.scope().device());
            if [input_ids.device(), hidden.stream.device(), w.device()]
                .into_iter()
                .any(|d| <D as ExecDevice>::device_id(d) != device)
            {
                return Err(OpError::Shape(
                    "block-quantized embedding device mismatch".into(),
                ));
            }
            return D::embedding_block_quant(ctx.scope(), w, input_ids, &mut hidden.stream);
        }
        let table = self.require_dense()?;
        let component_tp = self.parallelism.tp();
        let scope_tp = ctx.scope().topology().tp;
        if component_tp != scope_tp {
            return Err(OpError::Shape(format!(
                "Embedding TP rank {}/{} does not match execution scope rank {}/{}",
                component_tp.rank, component_tp.size, scope_tp.rank, scope_tp.size
            )));
        }

        match self.parallelism {
            EmbeddingParallelism::Replicated { .. } => {
                D::embedding(ctx.scope(), table, input_ids, &mut hidden.stream)
            }
            EmbeddingParallelism::Vocab {
                tp,
                vocab_start,
                global_vocab_size,
            } => {
                let table_shape = table.shape().as_slice();
                let local_vocab = table_shape.first().copied().ok_or_else(|| {
                    OpError::Shape("vocab-parallel Embedding table must be rank 2".into())
                })?;
                let expected_global = local_vocab.checked_mul(tp.size).ok_or_else(|| {
                    OpError::Shape("vocab-parallel Embedding vocabulary size overflows".into())
                })?;
                let expected_start = local_vocab.checked_mul(tp.rank).ok_or_else(|| {
                    OpError::Shape("vocab-parallel Embedding shard offset overflows".into())
                })?;
                if table_shape.len() != 2
                    || expected_global != global_vocab_size
                    || vocab_start != expected_start
                {
                    return Err(OpError::Shape(format!(
                        "vocab-parallel Embedding rank {}/{} expected local table [{}, dim] at vocab_start {}, got shape {:?}, start {}, global {}",
                        tp.rank,
                        tp.size,
                        global_vocab_size.checked_div(tp.size).unwrap_or(0),
                        expected_start,
                        table_shape,
                        vocab_start,
                        global_vocab_size
                    )));
                }
                if tp.size > 1 && <D as CollectiveOps>::comm(ctx.scope(), CommAxis::Tp).is_none() {
                    return Err(OpError::Kernel(format!(
                        "vocab-parallel Embedding rank {}/{} requires a TP communicator",
                        tp.rank, tp.size
                    )));
                }
                <D as VocabOps>::vocab_embedding(
                    ctx.scope(),
                    table,
                    input_ids,
                    &mut hidden.stream,
                    vocab_start,
                    global_vocab_size,
                )?;
                if tp.size > 1 {
                    <D as CollectiveOps>::all_reduce(
                        ctx.scope(),
                        CommAxis::Tp,
                        ReduceOp::Sum,
                        &mut hidden.stream,
                    )?;
                }
                Ok(())
            }
        }
    }
}
