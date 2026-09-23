//! Eager, single-sequence diagnostic runner. Token IDs in, last-row logits out.
//! Uses the same decoder and cache types as serving, without scheduler or graphs.
//! Single-token continuation uses causal ragged attention too: the legacy CUDA
//! decode kernel introduces additional BF16 rounding (see docs/GGUF_FORWARD.md).
use super::{Qwen3_5Model, Qwen35GgufLoader, weights};
use crate::domain::{
    cache::{LinearBatch, LinearLayerState, ModelCacheView},
    component::{Hidden, LayerRange},
    dtype::Dtype,
    exec::{ExecScope, StepCtx},
    features::LayerObserver,
    forward_scratch::ForwardScratch,
    gdn_scratch::GdnScratch,
    kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool},
    model::{DecoderModel, SampleRows},
    plan::{BatchKind, BatchPlan},
    ports::{OpBackend, OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};
use std::{collections::HashMap, time::Instant};

#[derive(Clone, Debug)]
pub struct LayerStats {
    pub layer: usize,
    pub max_abs: f32,
    /// Last token residual (stream + pending, rounded to activation dtype).
    pub last_residual: Vec<f32>,
}

#[derive(Debug)]
pub struct ProbeOutput {
    pub position: usize,
    pub logits: Vec<f32>,
    pub layers: Vec<LayerStats>,
    /// Wall time includes launches, readback and optional per-layer diagnostics.
    pub elapsed_seconds: f64,
}
impl ProbeOutput {
    /// Descending logit, breaking ties by smaller token ID.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<_> = self.logits.iter().copied().enumerate().collect();
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scores.truncate(k);
        scores
    }
}

pub struct GgufProbe<T: Dtype, D: LlmBackend> {
    model: Qwen3_5Model<T, D>,
    scope: D::Scope,
    kv: PagedKvPool<T, D>,
    states: Vec<LinearLayerState<T, D>>,
    context: usize,
    capacity: usize,
    position: usize,
    poisoned: bool,
    fatal: bool,
}

impl<T: Dtype, D: LlmBackend + OpBackend> GgufProbe<T, D> {
    /// Loads the main model once and allocates one sequence's persistent state.
    /// The reader may be dropped after return. Context comes from the validated
    /// loader; capacity bounds the maximum tokens accepted by a single step.
    pub fn load(loader: &Qwen35GgufLoader<'_>, scope: D::Scope, capacity: usize) -> OpResult<Self> {
        let context = loader.config().context_length;
        if capacity == 0 || capacity > context {
            return Err(OpError::Shape(
                "GGUF probe capacity must be in 1..=context".into(),
            ));
        }
        if scope.topology().tp.size != 1 || scope.topology().pp.size != 1 {
            return Err(OpError::unsupported("GGUF probe", "parallel execution"));
        }
        let device = scope.device();
        let mut model = loader.load::<T, D>(device)?;
        let dims = model.dims();
        model.install_scratch(ForwardScratch::new(device, dims, capacity, 1)?);
        model.install_gdn_scratch(GdnScratch::new(
            device,
            dims.dim,
            loader.config().linear,
            capacity,
        )?)?;
        let kv = PagedKvPool {
            layers: (0..model.cache_layout().num_full_layers())
                .map(|_| {
                    Ok(PagedKvLayer {
                        k: Tensor::zeros([context, 1, dims.kv_dim], device)?,
                        v: Tensor::zeros([context, 1, dims.kv_dim], device)?,
                    })
                })
                .collect::<OpResult<_>>()?,
            num_blocks: context,
            block_size: 1,
            kv_dim: dims.kv_dim,
            quant: KvQuantTier::None,
            seq_kv_len: HashMap::new(),
        };
        let states = model
            .cache_layout()
            .linear_dims()
            .iter()
            .map(|&dims| LinearLayerState::new(dims, 1, device))
            .collect::<OpResult<_>>()?;
        Ok(Self {
            model,
            scope,
            kv,
            states,
            context,
            capacity,
            position: 0,
            poisoned: false,
            fatal: false,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }

    /// Start an independent sequence without reloading weights. Old KV rows
    /// are unreachable because all subsequent lengths start again at zero.
    pub fn reset(&mut self) -> OpResult<()> {
        if self.fatal {
            return Err(OpError::Fatal(
                "cannot reset a poisoned CUDA/device context".into(),
            ));
        }
        self.poisoned = true;
        let result = (|| {
            for state in &mut self.states {
                state.reset_slot(0)?;
            }
            self.scope.device().synchronize()
        })();
        if let Err(e) = result {
            self.fatal = e.is_fatal();
            return Err(e);
        }
        self.position = 0;
        self.poisoned = false;
        Ok(())
    }

    /// Appends tokens and returns logits predicting the next token. Failed
    /// validation leaves state usable; a failed execution requires reset.
    /// Trace reads each layer's stream AND deferred residual without altering it.
    pub fn step(&mut self, ids: &[i32], trace: bool) -> OpResult<ProbeOutput> {
        if self.poisoned {
            return Err(OpError::Shape(
                "GGUF probe requires reset after failed execution".into(),
            ));
        }
        if ids.is_empty()
            || ids.len() > self.capacity
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.model.dims().vocab_size)
        {
            return Err(OpError::Shape(
                "GGUF probe invalid token IDs or step size".into(),
            ));
        }
        let end = self
            .position
            .checked_add(ids.len())
            .filter(|&n| n <= self.context)
            .ok_or_else(|| OpError::Shape("GGUF probe context exhausted".into()))?;
        let result = self.execute(ids, end, trace);
        match result {
            Ok(out) => {
                self.position = end;
                Ok(out)
            }
            Err(e) => {
                self.poisoned = true;
                self.fatal = e.is_fatal();
                Err(e)
            }
        }
    }

    fn execute(&mut self, ids: &[i32], end: usize, trace: bool) -> OpResult<ProbeOutput> {
        let now = Instant::now();
        let device = self.scope.device();
        let dims = self.model.dims();
        let q_lens = vec![ids.len() as i32];
        let positions: Vec<_> = (self.position as i32..end as i32).collect();
        let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&q_lens);
        let plan = BatchPlan {
            // A single new token is also a causal ragged query with a prefix.
            // Keep prefill and decode on the same attention arithmetic path.
            kind: BatchKind::Ragged,
            num_tokens: ids.len(),
            batch: 1,
            q_lens: q_lens.clone(),
            kv_lens: vec![end as i32],
            seq_positions: vec![self.position as i32],
            rope_positions: positions.clone(),
            max_blocks_per_seq: self.context,
            block_size: 1,
            total_q_tiles: req.len() as i32,
        };
        let ints = |values: Vec<i32>| {
            let len = values.len();
            weights::upload(values, &[len], device)
        };
        let index = KvIndexTensors {
            decode_rows: None,
            block_tables: weights::upload(
                (0..self.context as i32).collect(),
                &[1, self.context],
                device,
            )?,
            cu_q_lens: ints(cu)?,
            kv_lens: ints(vec![end as i32])?,
            seq_positions: ints(vec![self.position as i32])?,
            seq_lens_step: ints(q_lens.clone())?,
            rope_positions: ints(positions)?,
            block2req: ints(req.clone())?,
            block2tile: ints(tile)?,
            valid_q_tiles: ints(vec![req.len() as i32])?,
            valid_suffix_q_tiles: ints(vec![req.len() as i32])?,
        };
        let batch = LinearBatch::new(&[0], &q_lens, 1, device)?;
        let mut cache = ModelCacheView::hybrid(&mut self.kv, &index, &mut self.states, &batch);
        let input = ints(ids.to_vec())?;
        let mut hidden = Hidden {
            stream: Tensor::zeros([ids.len(), dims.dim], device)?,
            pending: None,
        };
        let ctx = StepCtx::new(&self.scope, &plan);
        let mut observer = FiniteLayers {
            enabled: trace,
            stats: vec![],
        };
        self.model.embed(&input, &mut hidden, &ctx)?;
        self.model.decoder.decode_layers_observed(
            LayerRange::all(dims.num_layers),
            &mut hidden,
            &mut cache,
            &ctx,
            &mut observer,
        )?;
        let logits = self
            .model
            .finalize(&hidden, SampleRows::LastPerSeq, &ctx)?
            .0
            .to_host_vec()?;
        let logits: Vec<_> = logits.iter().map(|x| T::read_f64(x) as f32).collect();
        if logits.len() != dims.vocab_size || logits.iter().any(|v| !v.is_finite()) {
            return Err(OpError::Kernel(
                "GGUF probe invalid/non-finite logits".into(),
            ));
        }
        Ok(ProbeOutput {
            position: end,
            logits,
            layers: observer.stats,
            elapsed_seconds: now.elapsed().as_secs_f64(),
        })
    }
}

struct FiniteLayers {
    enabled: bool,
    stats: Vec<LayerStats>,
}
impl<T: Dtype, D: LlmBackend> LayerObserver<T, D> for FiniteLayers {
    fn is_active(&self) -> bool {
        self.enabled
    }
    fn capture(&mut self, layer: usize, hidden: &Hidden<T, D>, _: &StepCtx<'_, D>) -> OpResult<()> {
        if !self.enabled {
            return Ok(());
        }
        let mut max_abs = 0.0f32;
        let width = hidden.stream.shape().as_slice()[1];
        let mut last_residual = vec![0.0f32; width];
        for tensor in std::iter::once(&hidden.stream).chain(hidden.pending.iter()) {
            let host = tensor.to_host_vec()?;
            for (a, b) in last_residual.iter_mut().zip(&host[host.len() - width..]) {
                *a += T::read_f64(b) as f32;
            }
            for value in host {
                let v = T::read_f64(&value) as f32;
                if !v.is_finite() {
                    return Err(OpError::Kernel(format!(
                        "GGUF probe non-finite activation at layer {layer}"
                    )));
                }
                max_abs = max_abs.max(v.abs());
            }
        }
        for x in &mut last_residual {
            *x = T::read_f64(&T::write_f64(f64::from(*x))) as f32;
        }
        self.stats.push(LayerStats {
            layer,
            max_abs,
            last_residual,
        });
        Ok(())
    }
}
