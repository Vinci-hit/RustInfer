//! Reusable layer building blocks for LLM models.
//!
//! These structs hold weights and dispatch through OpBackend.
//! They are generic over `<T: Dtype, D: OpBackend>`.

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::Dtype;
use crate::domain::tensor::Tensor;

/// Linear layer: output = input @ weight^T + optional bias
pub struct Linear<T: Dtype, D: OpBackend> {
    pub weight: Tensor<T, D>,
    pub bias: Option<Tensor<T, D>>,
}

impl<T: Dtype, D: OpBackend> Linear<T, D> {
    pub fn new(weight: Tensor<T, D>, bias: Option<Tensor<T, D>>) -> Self {
        Self { weight, bias }
    }

    /// Forward: `output = input @ weight^T + bias` (bias is row-broadcast).
    ///
    /// `bias` is `[N]` (or `[1, N]`); we broadcast-add it across all rows of
    /// the `[M, N]` output. Plain `add_inplace` would treat the two tensors
    /// as element-wise of identical numel and run off the end of `bias`.
    pub fn forward(&self, input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::matmul(input, &self.weight, output)?;
        if let Some(bias) = &self.bias {
            D::broadcast_add_inplace(output, bias)?;
        }
        Ok(())
    }
}

/// Quantized linear layer (AWQ int4): output = dequant(input × weight_packed)
pub struct QuantLinear<A: Dtype, W: Dtype, D: OpBackend> {
    pub weight_packed: Tensor<W, D>,
    pub scales: Tensor<A, D>,
    pub zeros: Option<Tensor<W, D>>,
    pub group_size: usize,
}

impl<A: Dtype, W: Dtype, D: OpBackend> QuantLinear<A, W, D> {
    pub fn forward(&self, input: &Tensor<A, D>, output: &mut Tensor<A, D>) -> OpResult<()> {
        D::matmul_quant(input, &self.weight_packed, output, &self.scales, self.zeros.as_ref(), self.group_size)
    }
}

/// RMSNorm layer
pub struct RMSNorm<T: Dtype, D: OpBackend> {
    pub weight: Tensor<T, D>,
    pub eps: f32,
}

impl<T: Dtype, D: OpBackend> RMSNorm<T, D> {
    pub fn new(weight: Tensor<T, D>, eps: f32) -> Self { Self { weight, eps } }

    pub fn forward(&self, input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::rmsnorm(input, &self.weight, output, self.eps)
    }

    pub fn forward_inplace(&self, x: &mut Tensor<T, D>) -> OpResult<()> {
        D::rmsnorm_inplace(x, &self.weight, self.eps)
    }
}

/// Embedding table
pub struct Embedding<T: Dtype, D: OpBackend> {
    pub table: Tensor<T, D>,
}

impl<T: Dtype, D: OpBackend> Embedding<T, D> {
    pub fn forward(&self, indices: &Tensor<i32, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::embedding(&self.table, indices, output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Diffusion layer types
// ═══════════════════════════════════════════════════════════════════════════════

/// Conv2D layer (for VAE decoder/encoder)
pub struct Conv2D<T: Dtype, D: OpBackend> {
    pub weight: Tensor<T, D>,   // [Cout, Cin, Kh, Kw]
    pub bias: Option<Tensor<T, D>>,
    pub stride: usize,
    pub padding: usize,
}

impl<T: Dtype, D: OpBackend> Conv2D<T, D> {
    pub fn new(weight: Tensor<T, D>, bias: Option<Tensor<T, D>>, stride: usize, padding: usize) -> Self {
        Self { weight, bias, stride, padding }
    }

    pub fn forward(&self, input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::conv2d(input, &self.weight, self.bias.as_ref(), output, self.stride, self.padding)
    }
}

/// GroupNorm layer (for VAE)
pub struct GroupNorm<T: Dtype, D: OpBackend> {
    pub weight: Tensor<T, D>,
    pub bias: Tensor<T, D>,
    pub num_groups: usize,
    pub eps: f32,
}

impl<T: Dtype, D: OpBackend> GroupNorm<T, D> {
    pub fn new(weight: Tensor<T, D>, bias: Tensor<T, D>, num_groups: usize, eps: f32) -> Self {
        Self { weight, bias, num_groups, eps }
    }

    pub fn forward(&self, input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::groupnorm(input, &self.weight, &self.bias, output, self.num_groups, self.eps)
    }

    /// Fused GroupNorm + SiLU activation.
    pub fn forward_silu(&self, input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::groupnorm_silu(input, &self.weight, &self.bias, output, self.num_groups, self.eps)
    }
}

/// LayerNorm layer (for DiT — different from RMSNorm: uses mean+variance)
pub struct LayerNorm<T: Dtype, D: OpBackend> {
    pub weight: Tensor<T, D>,
    pub bias: Tensor<T, D>,
    pub eps: f32,
}

impl<T: Dtype, D: OpBackend> LayerNorm<T, D> {
    pub fn new(weight: Tensor<T, D>, bias: Tensor<T, D>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> {
        D::layernorm(input, &self.weight, &self.bias, output, self.eps)
    }
}

