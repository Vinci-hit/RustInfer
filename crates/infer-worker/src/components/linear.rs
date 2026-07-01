use crate::domain::dtype::Dtype;
use crate::domain::dtype::quant::QuantScheme;
use crate::domain::exec::StepCtx;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;

/// A linear layer's weight, either full-precision or int4 group-quantized.
///
/// Quantization is an *attribute of the weight*, not a separate layer type, so
/// every `Linear` (attention qkv/o, lm_head, MLP gate/up/down) transparently
/// gains int4 support with no change to callers. Extending to GPTQ / FP8 later
/// means adding a variant here — the [`Linear::forward`] dispatch is the only
/// place that needs to learn the new format.
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
}

impl<T: Dtype, D: LlmBackend> LinearWeight<T, D> {
    /// Output feature count `N` (logical weight rows). For both the dense and
    /// packed layouts the leading dim is already `N` (`packed` is `[N, K/8]`),
    /// so this reads `shape[0]` in either case.
    pub fn out_features(&self) -> usize {
        match self {
            LinearWeight::Dense(w) => w.shape().as_slice()[0],
            LinearWeight::Awq { packed, .. } => packed.shape().as_slice()[0],
        }
    }

    /// Borrow the dense weight tensor, or `None` if quantized. For callers that
    /// need the full `[N, K]` shape (e.g. MoE router shape validation) and only
    /// ever run on full-precision weights.
    pub fn as_dense(&self) -> Option<&Tensor<T, D>> {
        match self {
            LinearWeight::Dense(w) => Some(w),
            LinearWeight::Awq { .. } => None,
        }
    }
}

/// Linear layer: `output = input @ weight^T + optional bias`.
///
/// The weight may be dense or int4-quantized (see [`LinearWeight`]); `forward`
/// dispatches to the matching kernel. Bias is always full-precision `T`.
pub struct Linear<T: Dtype, D: LlmBackend> {
    pub weight: LinearWeight<T, D>,
    pub bias: Option<Tensor<T, D>>,
}

impl<T: Dtype, D: LlmBackend> Linear<T, D> {
    /// Full-precision linear from a dense `[N, K]` weight.
    pub fn new(weight: Tensor<T, D>, bias: Option<Tensor<T, D>>) -> Self {
        Self {
            weight: LinearWeight::Dense(weight),
            bias,
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
        }
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
        }
        if let Some(bias) = &self.bias {
            D::broadcast_add_inplace(ctx.scope(), output, bias)?;
        }
        Ok(())
    }
}
