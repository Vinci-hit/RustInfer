//! Caller-owned target readout. Layer capture never modifies the residual stream.
use super::{component::Hidden, dtype::Dtype, exec::StepCtx, model::ModelDims};
use super::{
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
    types::Shape,
};

#[derive(Clone, Debug)]
pub enum FeatureSpec {
    FinalNormalized,
    DecoderLayers(Vec<usize>),
}

impl FeatureSpec {
    pub fn width(&self, dim: usize) -> OpResult<usize> {
        let count = match self {
            Self::FinalNormalized => 1,
            Self::DecoderLayers(ids) => {
                if ids.is_empty() || ids.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(OpError::Shape(
                        "feature layers must be nonempty and strictly increasing".into(),
                    ));
                }
                ids.len()
            }
        };
        dim.checked_mul(count)
            .filter(|&width| width > 0)
            .ok_or_else(|| OpError::Shape("invalid target feature width".into()))
    }
}

pub trait LayerObserver<T: Dtype, D: LlmBackend> {
    fn is_active(&self) -> bool;
    fn capture(
        &mut self,
        block: usize,
        hidden: &Hidden<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
}

pub struct NoopObserver;
impl<T: Dtype, D: LlmBackend> LayerObserver<T, D> for NoopObserver {
    fn is_active(&self) -> bool {
        false
    }
    fn capture(&mut self, _: usize, _: &Hidden<T, D>, _: &StepCtx<'_, D>) -> OpResult<()> {
        Ok(())
    }
}

pub struct LayerFeatures<T: Dtype, D: LlmBackend> {
    ids: Vec<usize>,
    layers: Vec<Tensor<T, D>>,
    concat: [Tensor<T, D>; 2],
    dim: usize,
    capacity: usize,
}

impl<T: Dtype, D: LlmBackend> LayerFeatures<T, D> {
    pub fn validate(&self, dims: ModelDims, rows: usize, device: &D) -> OpResult<()> {
        if self.dim != dims.dim
            || rows == 0
            || rows > self.capacity
            || self.ids.iter().any(|&i| i >= dims.num_layers)
            || infer_core::device::Device::device_id(self.layers[0].device())
                != infer_core::device::Device::device_id(device)
        {
            return Err(OpError::Shape(
                "target feature dimensions/layers/capacity mismatch".into(),
            ));
        }
        Ok(())
    }
    fn bank(&self, bank: usize, rows: usize, cols: usize) -> OpResult<Tensor<T, D>> {
        self.concat[bank]
            .narrow(0, 0, rows * cols)?
            .view_contiguous(Shape::from_slice(&[rows, cols]))
    }
    pub fn concatenate(&self, rows: usize, scope: &D::Scope) -> OpResult<()> {
        if rows == 0 || rows > self.capacity {
            return Err(OpError::Shape("invalid feature row count".into()));
        }
        D::copy_tensor(
            scope,
            &self.layers[0].narrow(0, 0, rows)?,
            &mut self.bank(0, rows, self.dim)?,
        )?;
        for i in 1..self.layers.len() {
            D::concat_cols(
                scope,
                &self.bank((i - 1) % 2, rows, i * self.dim)?,
                &self.layers[i].narrow(0, 0, rows)?,
                &mut self.bank(i % 2, rows, (i + 1) * self.dim)?,
            )?;
        }
        Ok(())
    }
    fn rows(&self, rows: usize) -> OpResult<Tensor<T, D>> {
        self.bank(
            (self.layers.len() - 1) % 2,
            rows,
            self.layers.len() * self.dim,
        )
    }
}

impl<T: Dtype, D: LlmBackend> LayerObserver<T, D> for LayerFeatures<T, D> {
    fn is_active(&self) -> bool {
        true
    }
    fn capture(
        &mut self,
        block: usize,
        hidden: &Hidden<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        if let Ok(i) = self.ids.binary_search(&block) {
            let mut dst = self.layers[i].narrow(0, 0, ctx.plan().num_tokens)?;
            match &hidden.pending {
                Some(delta) => D::add(ctx.scope(), &hidden.stream, delta, &mut dst)?,
                None => D::copy_tensor(ctx.scope(), &hidden.stream, &mut dst)?,
            }
        }
        Ok(())
    }
}

// Startup-owned workspace; inline variants avoid an extra ownership layer.
#[allow(clippy::large_enum_variant)]
pub enum TargetFeatures<T: Dtype, D: LlmBackend> {
    FinalNormalized(Tensor<T, D>),
    Layers(LayerFeatures<T, D>),
}

impl<T: Dtype, D: LlmBackend> TargetFeatures<T, D> {
    pub fn new(spec: FeatureSpec, dim: usize, capacity: usize, device: &D) -> OpResult<Self> {
        let width = spec.width(dim)?;
        if dim == 0 || capacity == 0 {
            return Err(OpError::Shape("empty target feature workspace".into()));
        }
        match spec {
            FeatureSpec::FinalNormalized => Ok(Self::FinalNormalized(Tensor::zeros(
                [capacity, dim],
                device,
            )?)),
            FeatureSpec::DecoderLayers(ids) => {
                let size = capacity
                    .checked_mul(width)
                    .ok_or_else(|| OpError::Shape("feature workspace overflow".into()))?;
                let layers = ids
                    .iter()
                    .map(|_| Tensor::zeros([capacity, dim], device))
                    .collect::<OpResult<_>>()?;
                Ok(Self::Layers(LayerFeatures {
                    ids,
                    layers,
                    concat: [
                        Tensor::zeros([size], device)?,
                        Tensor::zeros([size], device)?,
                    ],
                    dim,
                    capacity,
                }))
            }
        }
    }
    pub fn rows(&self, rows: usize) -> OpResult<Tensor<T, D>> {
        match self {
            Self::FinalNormalized(t) => t.narrow(0, 0, rows),
            Self::Layers(l) => l.rows(rows),
        }
    }
}
