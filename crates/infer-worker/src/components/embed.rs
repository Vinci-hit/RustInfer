use crate::domain::component::Hidden;
use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecScope, RankPair, StepCtx};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{CollectiveOps, CommAxis, OpError, OpResult, ReduceOp, VocabOps};
use crate::domain::tensor::Tensor;

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
    pub table: Tensor<T, D>,
    parallelism: EmbeddingParallelism,
}

impl<T: Dtype, D: LlmBackend> Embed<T, D> {
    pub fn new(table: Tensor<T, D>) -> Self {
        Self {
            table,
            parallelism: EmbeddingParallelism::default(),
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
                D::embedding(ctx.scope(), &self.table, input_ids, &mut hidden.stream)
            }
            EmbeddingParallelism::Vocab {
                tp,
                vocab_start,
                global_vocab_size,
            } => {
                let table_shape = self.table.shape().as_slice();
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
                    &self.table,
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
