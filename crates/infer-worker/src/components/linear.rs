use crate::domain::dtype::quant::QuantScheme;
use crate::domain::dtype::{Dtype, Fp8E4m3};
use crate::domain::exec::{ExecScope, RankPair, StepCtx};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{CollectiveOps, CommAxis, OpError, OpResult, ReduceOp};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

/// A linear layer's weight: dense, int4 group-quantized, or block-scaled FP8.
///
/// Quantization is an *attribute of the weight*, not a separate layer type, so
/// every `Linear` (attention qkv/o, lm_head, MLP gate/up/down) transparently
/// dispatches through the matching backend operation with no change to callers.
#[allow(clippy::large_enum_variant)] // Dense and AWQ layouts stay allocation-free after loading.
pub enum LinearWeight<T: Dtype, D: LlmBackend> {
    /// Dense `[N, K]` weight in the activation dtype `T`.
    Dense(Tensor<T, D>),
    /// compressed-tensors `pack-quantized` (llm-compressor AWQ W4A16):
    ///   - `packed`: `[N, K/8]` int32 — 8 int4 per word, sequential along K
    ///   - `zeros`:  `[N/8, K/group]` int32 — zero points packed along N
    ///   - `scales`: `[N, K/group]` in `T` — per-group scale
    Awq {
        packed: Tensor<i32, D>,
        zeros: Tensor<i32, D>,
        scales: Tensor<T, D>,
        scheme: QuantScheme,
    },
    /// Native block-scaled E4M3 weight. The inverse scale grid is FP32 to
    /// match the FP32 accumulator used by the native kernel.
    Fp8Block {
        weight: Tensor<Fp8E4m3, D>,
        weight_scale_inv: Tensor<f32, D>,
        block: [usize; 2],
    },
}

/// How this linear's logical `[N, K]` weight is laid out across TP ranks.
///
/// The default is a complete, single-rank weight.  The model keeps using one
/// `Linear` type; the loader only changes this metadata and the physical weight
/// shape when tensor parallelism is requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearParallelism {
    /// Every rank owns the complete weight (TP1 by default; embeddings and the
    /// LM head may also remain replicated in a multi-rank model).
    Replicated { tp: RankPair },
    /// The output-feature (`N`) axis is split. Each rank produces a disjoint
    /// output slice, so the forward itself needs no collective.
    Column { tp: RankPair },
    /// The input-feature (`K`) axis is split. Each rank produces a partial sum
    /// that must be all-reduced before adding bias.
    Row { tp: RankPair },
    /// Vocabulary rows are split. Each rank computes local logits and gathers
    /// them along the final dimension for the existing full-vocab sampler.
    Vocab {
        tp: RankPair,
        global_out_features: usize,
    },
}

impl LinearParallelism {
    pub const SINGLE: Self = Self::Replicated {
        tp: RankPair { rank: 0, size: 1 },
    };

    pub const fn tp(self) -> RankPair {
        match self {
            Self::Replicated { tp }
            | Self::Column { tp }
            | Self::Row { tp }
            | Self::Vocab { tp, .. } => tp,
        }
    }
}

impl Default for LinearParallelism {
    fn default() -> Self {
        Self::SINGLE
    }
}

impl<T: Dtype, D: LlmBackend> LinearWeight<T, D> {
    /// Output feature count `N` (logical weight rows). Dense and FP8 weights
    /// are `[N, K]`; AWQ packed weights are `[N, K/8]`, so every layout keeps
    /// `N` as its leading dimension.
    pub fn out_features(&self) -> usize {
        match self {
            LinearWeight::Dense(w) => w.shape().as_slice()[0],
            LinearWeight::Awq { packed, .. } => packed.shape().as_slice()[0],
            LinearWeight::Fp8Block { weight, .. } => weight.shape().as_slice()[0],
        }
    }

    /// Borrow the dense weight tensor, or `None` if quantized. For callers that
    /// need the full `[N, K]` shape (e.g. MoE router shape validation) and only
    /// ever run on full-precision weights.
    pub fn as_dense(&self) -> Option<&Tensor<T, D>> {
        match self {
            LinearWeight::Dense(w) => Some(w),
            LinearWeight::Awq { .. } | LinearWeight::Fp8Block { .. } => None,
        }
    }
}

/// Linear layer: `output = input @ weight^T + optional bias`.
///
/// The weight may be dense, int4-quantized, or block-scaled FP8 (see
/// [`LinearWeight`]); `forward` dispatches to the matching kernel. Bias is
/// always full-precision `T`.
pub struct Linear<T: Dtype, D: LlmBackend> {
    pub weight: LinearWeight<T, D>,
    pub bias: Option<Tensor<T, D>>,
    parallelism: LinearParallelism,
}

impl<T: Dtype, D: LlmBackend> Linear<T, D> {
    /// Full-precision linear from a dense `[N, K]` weight.
    pub fn new(weight: Tensor<T, D>, bias: Option<Tensor<T, D>>) -> Self {
        Self {
            weight: LinearWeight::Dense(weight),
            bias,
            parallelism: LinearParallelism::default(),
        }
    }

    /// int4 group-quantized (`pack-quantized`) linear. See [`LinearWeight::Awq`]
    /// for the expected tensor layout.
    pub fn from_awq(
        packed: Tensor<i32, D>,
        zeros: Tensor<i32, D>,
        scales: Tensor<T, D>,
        scheme: QuantScheme,
        bias: Option<Tensor<T, D>>,
    ) -> Self {
        Self {
            weight: LinearWeight::Awq {
                packed,
                zeros,
                scales,
                scheme,
            },
            bias,
            parallelism: LinearParallelism::default(),
        }
    }

    /// Native block-scaled FP8 linear. `weight` remains raw E4M3 on-device;
    /// the backend consumes the FP32 inverse-scale grid during matmul.
    pub fn from_fp8_block(
        weight: Tensor<Fp8E4m3, D>,
        weight_scale_inv: Tensor<f32, D>,
        block: [usize; 2],
        bias: Option<Tensor<T, D>>,
    ) -> Self {
        Self {
            weight: LinearWeight::Fp8Block {
                weight,
                weight_scale_inv,
                block,
            },
            bias,
            parallelism: LinearParallelism::default(),
        }
    }

    /// Attach the sharding semantics chosen by the weight loader.
    pub fn with_parallelism(mut self, parallelism: LinearParallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn parallelism(&self) -> LinearParallelism {
        self.parallelism
    }

    /// Output feature count `N` (logical weight rows).
    pub fn out_features(&self) -> usize {
        self.weight.out_features()
    }

    pub fn forward(
        &self,
        input: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let linear_tp = self.parallelism.tp();
        let scope_tp = ctx.scope().topology().tp;
        if linear_tp != scope_tp {
            return Err(OpError::Shape(format!(
                "Linear TP rank {}/{} does not match execution scope rank {}/{}",
                linear_tp.rank, linear_tp.size, scope_tp.rank, scope_tp.size
            )));
        }
        let collective = match self.parallelism {
            LinearParallelism::Row { tp } if tp.size > 1 => Some((tp, "row-parallel")),
            LinearParallelism::Vocab { tp, .. } if tp.size > 1 => Some((tp, "vocab-parallel")),
            _ => None,
        };
        if let Some((tp, kind)) = collective
            && <D as CollectiveOps>::comm(ctx.scope(), CommAxis::Tp).is_none()
        {
            return Err(OpError::Kernel(format!(
                "{kind} Linear rank {}/{} requires a TP communicator",
                tp.rank, tp.size,
            )));
        }

        if let LinearParallelism::Vocab {
            tp,
            global_out_features,
        } = self.parallelism
            && tp.size > 1
        {
            let gathered_features = self.out_features().checked_mul(tp.size).ok_or_else(|| {
                OpError::Shape("vocab-parallel Linear output size overflows".into())
            })?;
            if gathered_features != global_out_features {
                return Err(OpError::Shape(format!(
                    "vocab-parallel Linear local output {} x TP{} produces {}, expected global output {}",
                    self.out_features(),
                    tp.size,
                    gathered_features,
                    global_out_features
                )));
            }
            let input_shape = input.shape().as_slice();
            let output_shape = output.shape().as_slice();
            if input_shape.len() != 2 || output_shape != [input_shape[0], global_out_features] {
                return Err(OpError::Shape(format!(
                    "vocab-parallel Linear expected output [{}, {}], got {:?}",
                    input_shape.first().copied().unwrap_or(0),
                    global_out_features,
                    output_shape
                )));
            }
            let mut local = D::alloc_tensor(
                Shape::from_slice(&[input_shape[0], self.out_features()]),
                input.device(),
            )?;
            self.matmul_local(input, &mut local, ctx)?;
            self.add_bias(&mut local, ctx)?;
            <D as CollectiveOps>::all_gather(ctx.scope(), CommAxis::Tp, 1, &local, output)?;
            return Ok(());
        }

        self.matmul_local(input, output, ctx)?;

        if matches!(self.parallelism, LinearParallelism::Row { tp } if tp.size > 1) {
            <D as CollectiveOps>::all_reduce(ctx.scope(), CommAxis::Tp, ReduceOp::Sum, output)?;
        }
        self.add_bias(output, ctx)
    }

    fn matmul_local(
        &self,
        input: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        match &self.weight {
            LinearWeight::Dense(w) => D::matmul(ctx.scope(), input, w, output)?,
            LinearWeight::Awq {
                packed,
                zeros,
                scales,
                scheme,
            } => D::matmul_quant(
                ctx.scope(),
                input,
                packed,
                output,
                scales,
                Some(zeros),
                scheme,
            )?,
            LinearWeight::Fp8Block {
                weight,
                weight_scale_inv,
                block,
            } => D::matmul_fp8_block(ctx.scope(), input, weight, output, weight_scale_inv, *block)?,
        }
        Ok(())
    }

    fn add_bias(&self, output: &mut Tensor<T, D>, ctx: &StepCtx<'_, D>) -> OpResult<()> {
        if let Some(bias) = &self.bias {
            D::broadcast_add_inplace(ctx.scope(), output, bias)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::exec::{HostScope, StepCtx, TopologyShape};
    use crate::domain::plan::{BatchKind, BatchPlan};
    use crate::domain::ports::OpError;
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn linear_defaults_to_replicated_tp1() {
        let weight = Tensor::from_host_slice(&[1.0f32], [1, 1], &Cpu).unwrap();
        let linear = Linear::new(weight, None);

        assert_eq!(linear.parallelism(), LinearParallelism::SINGLE);
        assert_eq!(linear.parallelism().tp(), RankPair { rank: 0, size: 1 });
    }

    #[test]
    fn row_parallel_tp2_fails_before_compute_without_a_communicator() {
        let linear = Linear::new(
            Tensor::from_host_slice(&[2.0f32], [1, 1], &Cpu).unwrap(),
            None,
        )
        .with_parallelism(LinearParallelism::Row {
            tp: RankPair { rank: 0, size: 2 },
        });
        let input = Tensor::from_host_slice(&[3.0f32], [1, 1], &Cpu).unwrap();
        let mut output = Tensor::from_host_slice(&[0.0f32], [1, 1], &Cpu).unwrap();
        let scope = HostScope::new(Cpu).with_topology(TopologyShape {
            tp: RankPair { rank: 0, size: 2 },
            ..TopologyShape::SINGLE
        });
        let plan = BatchPlan {
            kind: BatchKind::DecodeOnly,
            num_tokens: 1,
            batch: 1,
            q_lens: vec![1],
            kv_lens: vec![1],
            seq_positions: vec![0],
            rope_positions: vec![0],
            max_blocks_per_seq: 1,
            block_size: 1,
            total_q_tiles: 0,
        };
        let ctx = StepCtx::new(&scope, &plan);

        let err = linear.forward(&input, &mut output, &ctx).unwrap_err();
        assert!(err.to_string().contains("requires a TP communicator"));
        assert_eq!(output.to_host_vec().unwrap(), vec![0.0]);
    }

    #[test]
    fn fp8_block_weight_dispatches_to_dedicated_op() {
        let cpu = Cpu;
        let raw_weight = Tensor::from_host_slice(
            &[
                Fp8E4m3(0x38),
                Fp8E4m3(0x38),
                Fp8E4m3(0x38),
                Fp8E4m3(0x38),
                Fp8E4m3(0x38),
                Fp8E4m3(0x38),
            ],
            [2, 3],
            &cpu,
        )
        .unwrap();
        let scales = Tensor::from_host_slice(&[1.0f32], [1, 1], &cpu).unwrap();
        let linear: Linear<f32, Cpu> = Linear::from_fp8_block(raw_weight, scales, [128, 128], None);

        assert_eq!(linear.out_features(), 2);
        assert!(linear.weight.as_dense().is_none());

        let input = Tensor::from_host_slice(&[1.0f32, 2.0, 3.0], [1, 3], &cpu).unwrap();
        let mut output = Tensor::from_host_slice(&[0.0f32; 2], [1, 2], &cpu).unwrap();
        let scope = HostScope::new(cpu);
        let plan = BatchPlan {
            kind: BatchKind::DecodeOnly,
            num_tokens: 1,
            batch: 1,
            q_lens: vec![1],
            kv_lens: vec![1],
            seq_positions: vec![0],
            rope_positions: vec![0],
            max_blocks_per_seq: 1,
            block_size: 1,
            total_q_tiles: 0,
        };
        let ctx = StepCtx::new(&scope, &plan);

        let err = linear
            .forward(&input, &mut output, &ctx)
            .expect_err("CPU has no native FP8 kernel");
        assert!(matches!(
            err,
            OpError::Unsupported {
                backend: "cpu",
                op: "matmul_fp8_block"
            }
        ));
    }
}
