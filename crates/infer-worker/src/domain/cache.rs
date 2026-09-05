//! Model execution caches. Logical layers map to compact, separate KV and
//! recurrent arrays. Views borrow caller-owned state; they never reset it.

use std::collections::HashSet;

use super::component::LayerRange;
use super::dtype::Dtype;
use super::kv::{KvIndexTensors, KvView, PagedKvPool};
use super::plan::{BatchKind, BatchPlan};
use super::ports::backend::LlmBackend;
use super::ports::{OpError, OpResult};
use super::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearDims {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
}

impl LinearDims {
    pub fn validate(self) -> OpResult<()> {
        if self.num_key_heads == 0
            || self.num_value_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.conv_kernel_dim == 0
            || !self.num_value_heads.is_multiple_of(self.num_key_heads)
        {
            return Err(OpError::Shape(format!(
                "invalid linear dimensions: {self:?}"
            )));
        }
        let key = self.num_key_heads.checked_mul(self.key_head_dim);
        let value = self.num_value_heads.checked_mul(self.value_head_dim);
        let conv = key
            .and_then(|k| k.checked_mul(2))
            .zip(value)
            .and_then(|(k, v)| k.checked_add(v));
        let state = value.and_then(|v| v.checked_mul(self.key_head_dim));
        if conv
            .and_then(|c| c.checked_mul(self.conv_kernel_dim))
            .is_none()
            || state.is_none()
        {
            return Err(OpError::Shape("linear dimensions overflow".into()));
        }
        Ok(())
    }

    pub fn key_dim(self) -> usize {
        self.num_key_heads * self.key_head_dim
    }
    pub fn value_dim(self) -> usize {
        self.num_value_heads * self.value_head_dim
    }
    pub fn conv_dim(self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerCacheId {
    Full(usize),
    Linear(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum LayerCacheSpec {
    Full { kv_dim: usize },
    Linear(LinearDims),
}

/// Built once from the actual mixer sequence, then kept immutable by Decoder.
#[derive(Debug, Clone, Default)]
pub struct CacheLayout {
    layers: Vec<LayerCacheId>,
    num_full: usize,
    kv_dim: Option<usize>,
    linear_dims: Vec<LinearDims>,
}

impl CacheLayout {
    pub fn new(specs: impl IntoIterator<Item = LayerCacheSpec>) -> OpResult<Self> {
        let mut layout = Self::default();
        for spec in specs {
            let id = match spec {
                LayerCacheSpec::Full { kv_dim } => {
                    if kv_dim == 0 || layout.kv_dim.is_some_and(|dim| dim != kv_dim) {
                        return Err(OpError::Shape(
                            "full layers must share a nonzero KV width".into(),
                        ));
                    }
                    layout.kv_dim = Some(kv_dim);
                    let id = LayerCacheId::Full(layout.num_full);
                    layout.num_full += 1;
                    id
                }
                LayerCacheSpec::Linear(dims) => {
                    dims.validate()?;
                    let id = LayerCacheId::Linear(layout.linear_dims.len());
                    layout.linear_dims.push(dims);
                    id
                }
            };
            layout.layers.push(id);
        }
        Ok(layout)
    }

    pub fn layers(&self) -> &[LayerCacheId] {
        &self.layers
    }
    pub fn num_full_layers(&self) -> usize {
        self.num_full
    }
    pub fn linear_dims(&self) -> &[LinearDims] {
        &self.linear_dims
    }
    pub fn has_linear(&self) -> bool {
        !self.linear_dims.is_empty()
    }
}

/// Checked host metadata and its device indices. Preparing this once per batch
/// avoids device-to-host validation inside each layer. Slot identity belongs to
/// the caller and must survive batch reordering and prefill chunk boundaries.
pub struct LinearBatch<D: LlmBackend> {
    state_slots: Tensor<i32, D>,
    cu_seqlens: Tensor<i32, D>,
    q_lens: Vec<i32>,
    num_tokens: usize,
    num_slots: usize,
}

impl<D: LlmBackend> LinearBatch<D> {
    pub fn new(slots: &[i32], q_lens: &[i32], num_slots: usize, device: &D) -> OpResult<Self> {
        if slots.is_empty() || slots.len() != q_lens.len() || num_slots == 0 {
            return Err(OpError::Shape(
                "linear batch requires slots and matching q_lens".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(slots.len());
        let mut cu = Vec::with_capacity(q_lens.len() + 1);
        cu.push(0i32);
        for (&slot, &len) in slots.iter().zip(q_lens) {
            if slot < 0 || slot as usize >= num_slots || !seen.insert(slot) {
                return Err(OpError::Shape(format!(
                    "linear batch: invalid or duplicate slot {slot}"
                )));
            }
            if len < 0 {
                return Err(OpError::Shape("linear batch: negative q_len".into()));
            }
            cu.push(
                cu.last()
                    .unwrap()
                    .checked_add(len)
                    .ok_or_else(|| OpError::Shape("linear batch token count exceeds i32".into()))?,
            );
        }
        let num_tokens = *cu.last().unwrap() as usize;
        if num_tokens == 0 {
            return Err(OpError::Shape("linear batch must contain tokens".into()));
        }
        Ok(Self {
            state_slots: Tensor::from_host_slice(slots, [slots.len()], device)?,
            cu_seqlens: Tensor::from_host_slice(&cu, [cu.len()], device)?,
            q_lens: q_lens.to_vec(),
            num_tokens,
            num_slots,
        })
    }

    pub fn validate_plan(&self, plan: &BatchPlan) -> OpResult<()> {
        if matches!(plan.kind, BatchKind::Spec { .. }) {
            return Err(OpError::unsupported(
                "linear attention",
                "speculative execution",
            ));
        }
        if plan.batch != self.q_lens.len()
            || plan.q_lens != self.q_lens
            || plan.num_tokens != self.num_tokens
            || (matches!(plan.kind, BatchKind::DecodeOnly)
                && self.q_lens.iter().any(|&len| len != 1))
        {
            return Err(OpError::Shape(
                "linear batch metadata does not match BatchPlan".into(),
            ));
        }
        Ok(())
    }
}

pub struct LinearLayerState<T: Dtype, D: LlmBackend> {
    dims: LinearDims,
    conv: Tensor<T, D>,
    ssm: Tensor<f32, D>,
}

impl<T: Dtype, D: LlmBackend> LinearLayerState<T, D> {
    /// Allocate initially zero state. Reusing a sequence requires the caller to
    /// explicitly reset its slot or supply a fresh state allocation.
    pub fn new(dims: LinearDims, num_slots: usize, device: &D) -> OpResult<Self> {
        dims.validate()?;
        if num_slots == 0 {
            return Err(OpError::Shape(
                "linear state requires at least one slot".into(),
            ));
        }
        Ok(Self {
            dims,
            conv: Tensor::zeros([num_slots, dims.conv_dim(), dims.conv_kernel_dim], device)?,
            ssm: Tensor::zeros(
                [
                    num_slots,
                    dims.num_value_heads,
                    dims.key_head_dim,
                    dims.value_head_dim,
                ],
                device,
            )?,
        })
    }

    pub fn conv(&self) -> &Tensor<T, D> {
        &self.conv
    }
    pub fn ssm(&self) -> &Tensor<f32, D> {
        &self.ssm
    }

    pub fn view<'a>(&'a mut self, batch: &'a LinearBatch<D>) -> LinearLayerStateView<'a, T, D> {
        LinearLayerStateView {
            dims: self.dims,
            conv: &mut self.conv,
            ssm: &mut self.ssm,
            batch,
        }
    }

    fn validate(&self, dims: LinearDims, batch: &LinearBatch<D>) -> OpResult<()> {
        validate_linear_state(self.dims, dims, &self.conv, &self.ssm, batch)
    }
}

pub struct LinearLayerStateView<'a, T: Dtype, D: LlmBackend> {
    dims: LinearDims,
    pub(crate) conv: &'a mut Tensor<T, D>,
    pub(crate) ssm: &'a mut Tensor<f32, D>,
    batch: &'a LinearBatch<D>,
}

impl<T: Dtype, D: LlmBackend> LinearLayerStateView<'_, T, D> {
    pub fn validate(&self, dims: LinearDims, plan: &BatchPlan) -> OpResult<()> {
        self.batch.validate_plan(plan)?;
        validate_linear_state(self.dims, dims, self.conv, self.ssm, self.batch)
    }

    /// Split borrows so operators can read indices while updating both states.
    #[allow(clippy::type_complexity)] // Two mutable states and their two index tensors.
    pub(crate) fn parts(
        &mut self,
    ) -> (
        &mut Tensor<T, D>,
        &mut Tensor<f32, D>,
        &Tensor<i32, D>,
        &Tensor<i32, D>,
    ) {
        (
            self.conv,
            self.ssm,
            &self.batch.state_slots,
            &self.batch.cu_seqlens,
        )
    }
}

fn validate_linear_state<T: Dtype, D: LlmBackend>(
    actual: LinearDims,
    expected: LinearDims,
    conv: &Tensor<T, D>,
    ssm: &Tensor<f32, D>,
    batch: &LinearBatch<D>,
) -> OpResult<()> {
    if actual != expected
        || conv.shape().as_slice()
            != [
                batch.num_slots,
                expected.conv_dim(),
                expected.conv_kernel_dim,
            ]
        || ssm.shape().as_slice()
            != [
                batch.num_slots,
                expected.num_value_heads,
                expected.key_head_dim,
                expected.value_head_dim,
            ]
    {
        return Err(OpError::Shape(
            "linear layer state does not match model/batch dimensions".into(),
        ));
    }
    Ok(())
}

pub enum LayerCacheView<'a, T: Dtype, D: LlmBackend> {
    Full(KvView<'a, T, D>),
    Linear(LinearLayerStateView<'a, T, D>),
}

/// Always covers all compact cache layers of the model. Range selection is a
/// property of decoder execution, never an offset applied to these arrays.
pub struct ModelCacheView<'a, T: Dtype, D: LlmBackend> {
    kv: KvView<'a, T, D>,
    linear: &'a mut [LinearLayerState<T, D>],
    batch: Option<&'a LinearBatch<D>>,
}

impl<'a, T: Dtype, D: LlmBackend> ModelCacheView<'a, T, D> {
    pub fn full(pool: &'a mut PagedKvPool<T, D>, index: &'a KvIndexTensors<D>) -> Self {
        let range = LayerRange::all(pool.num_layers());
        Self {
            kv: pool.view(range, index),
            linear: &mut [],
            batch: None,
        }
    }

    pub fn hybrid(
        pool: &'a mut PagedKvPool<T, D>,
        index: &'a KvIndexTensors<D>,
        linear: &'a mut [LinearLayerState<T, D>],
        batch: &'a LinearBatch<D>,
    ) -> Self {
        let range = LayerRange::all(pool.num_layers());
        Self {
            kv: pool.view(range, index),
            linear,
            batch: Some(batch),
        }
    }

    /// Check all cache geometry before any layer can mutate persistent state.
    pub fn validate(
        &self,
        layout: &CacheLayout,
        range: LayerRange,
        plan: &BatchPlan,
    ) -> OpResult<()> {
        if range.start > range.end || range.end > layout.layers.len() {
            return Err(OpError::Shape("decoder layer range out of bounds".into()));
        }
        if self.kv.layers.len() != layout.num_full || self.linear.len() != layout.linear_dims.len()
        {
            return Err(OpError::Shape(
                "cache layer counts do not match model layout".into(),
            ));
        }
        if let Some(kv_dim) = layout.kv_dim {
            if self.kv.kv_dim != kv_dim
                || self.kv.block_size != plan.block_size
                || self.kv.num_blocks == 0
            {
                return Err(OpError::Shape(
                    "KV cache dimensions do not match model/plan".into(),
                ));
            }
            let shape = [self.kv.num_blocks, self.kv.block_size, kv_dim];
            if self.kv.layers.iter().any(|layer| {
                layer.k.shape().as_slice() != shape
                    || layer.v.shape().as_slice() != shape
                    || !layer.k.is_contiguous()
                    || !layer.v.is_contiguous()
            }) {
                return Err(OpError::Shape(
                    "KV layer tensors do not match pool geometry".into(),
                ));
            }
            validate_kv_plan(self.kv.index, plan, layout.has_linear())?;
        }
        if layout.has_linear() {
            let batch = self
                .batch
                .ok_or_else(|| OpError::Shape("missing linear batch metadata".into()))?;
            batch.validate_plan(plan)?;
            for (state, &dims) in self.linear.iter().zip(&layout.linear_dims) {
                state.validate(dims, batch)?;
            }
        }
        Ok(())
    }

    pub fn layer(&mut self, id: LayerCacheId) -> OpResult<LayerCacheView<'_, T, D>> {
        match id {
            LayerCacheId::Full(i) if i < self.kv.layers.len() => {
                Ok(LayerCacheView::Full(self.kv.single_layer(i)))
            }
            LayerCacheId::Linear(i) if i < self.linear.len() => {
                let batch = self
                    .batch
                    .ok_or_else(|| OpError::Shape("missing linear batch metadata".into()))?;
                Ok(LayerCacheView::Linear(self.linear[i].view(batch)))
            }
            _ => Err(OpError::Shape(format!("cache layer {id:?} out of bounds"))),
        }
    }
}

/// Validate host metadata and device tensor capacities without synchronizing.
/// The caller remains responsible for uploading matching, valid index values.
fn validate_kv_plan<D: LlmBackend>(
    index: &KvIndexTensors<D>,
    plan: &BatchPlan,
    hybrid: bool,
) -> OpResult<()> {
    if plan.block_size == 0
        || plan.max_blocks_per_seq == 0
        || plan.q_lens.len() != plan.batch
        || plan.kv_lens.len() != plan.batch
        || plan.seq_positions.len() != plan.batch
        || plan.rope_positions.len() < plan.num_tokens
        || plan.total_q_tiles < 0
    {
        return Err(OpError::Shape("invalid KV batch metadata".into()));
    }
    let max_seq = plan
        .block_size
        .checked_mul(plan.max_blocks_per_seq)
        .ok_or_else(|| OpError::Shape("KV sequence capacity overflows".into()))?;
    let mut tokens = 0usize;
    let mut tiles = 0usize;
    for ((&q, &kv), &pos) in plan
        .q_lens
        .iter()
        .zip(&plan.kv_lens)
        .zip(&plan.seq_positions)
    {
        if q < 0
            || kv < 0
            || pos < 0
            || kv as usize > max_seq
            || pos as usize > max_seq
            || (hybrid && pos.checked_add(q) != Some(kv))
        {
            return Err(OpError::Shape("invalid KV sequence range".into()));
        }
        tokens = tokens
            .checked_add(q as usize)
            .ok_or_else(|| OpError::Shape("KV token count overflows".into()))?;
        tiles += (q as usize).div_ceil(super::plan::RAGGED_Q_TILE as usize);
    }
    // Full-only graph plans contain bucket placeholders; live lengths reside
    // on the device. Hybrid execution currently requires an ordinary tape.
    if tokens > plan.num_tokens
        || tokens > i32::MAX as usize
        || (hybrid && tiles != plan.total_q_tiles as usize)
    {
        return Err(OpError::Shape("KV token/tile count mismatch".into()));
    }
    let table_len = plan
        .batch
        .checked_mul(plan.max_blocks_per_seq)
        .ok_or_else(|| OpError::Shape("KV block table capacity overflows".into()))?;
    let cu_len = plan
        .batch
        .checked_add(1)
        .ok_or_else(|| OpError::Shape("KV batch size overflows".into()))?;
    for (name, tensor, needed) in [
        ("block_tables", &index.block_tables, table_len),
        ("cu_q_lens", &index.cu_q_lens, cu_len),
        ("kv_lens", &index.kv_lens, plan.batch),
        ("seq_positions", &index.seq_positions, plan.batch),
        ("seq_lens_step", &index.seq_lens_step, plan.batch),
        ("rope_positions", &index.rope_positions, plan.num_tokens),
        ("block2req", &index.block2req, plan.total_q_tiles as usize),
        ("block2tile", &index.block2tile, plan.total_q_tiles as usize),
        ("valid_q_tiles", &index.valid_q_tiles, 1),
        ("valid_suffix_q_tiles", &index.valid_suffix_q_tiles, 1),
    ] {
        if tensor.numel() < needed || !tensor.is_contiguous() {
            return Err(OpError::Shape(format!(
                "KV {name} must be contiguous with at least {needed} elements"
            )));
        }
    }
    Ok(())
}
