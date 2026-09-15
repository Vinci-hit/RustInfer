//! Block-parallel proposal and confirmed target-feature cache ownership.
use super::DraftProposer;
use crate::application::execution::{ExecutionMetrics, ExecutionPlan, Phase, WorkspaceUse};
use crate::domain::{
    cache::ModelCacheView,
    draft::BlockDraft,
    draft_scratch::DraftWorkspace,
    dtype::Dtype,
    exec::{ExecScope, StepCtx},
    kv::{KvQuantTier, PagedKvLayer, PagedKvPool},
    model::ModelDims,
    plan::{BatchKind, MaskMode},
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};

pub struct BlockProposer<T: Dtype, D: LlmBackend, H: BlockDraft<T, D>> {
    head: H,
    kv: PagedKvPool<T, D>,
    workspace: DraftWorkspace<T, D>,
    noise_ids: Vec<i32>,
    logits: Tensor<T, D>,
    argmax_ws: Tensor<f32, D>,
    metrics: ExecutionMetrics,
    len: usize,
    max_context: usize,
    capacity: usize,
    poisoned: bool,
}

impl<T: Dtype, D: LlmBackend, H: BlockDraft<T, D>> BlockProposer<T, D, H> {
    pub fn new(mut head: H, max_context: usize, capacity: usize, device: &D) -> OpResult<Self> {
        if max_context == 0
            || max_context > i32::MAX as usize
            || capacity < 2
            || capacity > max_context
            || head.cache_layout().has_linear()
            || head.block_size() < 2
            || infer_core::device::Device::device_id(head.device())
                != infer_core::device::Device::device_id(device)
        {
            return Err(OpError::Shape(
                "invalid block proposer capacity/device/cache".into(),
            ));
        }
        head.prepare(capacity)?;
        let dims = head.dims();
        let block_size = 16;
        let num_blocks = max_context.div_ceil(block_size);
        let kv = PagedKvPool {
            layers: (0..head.cache_layout().num_full_layers())
                .map(|_| {
                    Ok(PagedKvLayer {
                        k: Tensor::zeros([num_blocks, block_size, dims.kv_dim], device)?,
                        v: Tensor::zeros([num_blocks, block_size, dims.kv_dim], device)?,
                    })
                })
                .collect::<OpResult<_>>()?,
            num_blocks,
            block_size,
            kv_dim: dims.kv_dim,
            quant: KvQuantTier::None,
            seq_kv_len: Default::default(),
        };
        let rows = head.block_size().min(capacity) - 1;
        Ok(Self {
            workspace: DraftWorkspace::new(
                ModelDims {
                    vocab_size: 0,
                    ..dims
                },
                max_context,
                capacity,
                device,
            )?,
            noise_ids: vec![head.mask_token_id(); rows + 1],
            logits: Tensor::zeros([rows, dims.vocab_size], device)?,
            argmax_ws: Tensor::zeros([rows * 512], device)?,
            head,
            kv,
            metrics: ExecutionMetrics::from_env("proposer"),
            len: 0,
            max_context,
            capacity,
            poisoned: false,
        })
    }

    fn validate(&self, scope: &D::Scope) -> OpResult<()> {
        if self.poisoned
            || scope.topology().tp.size != 1
            || infer_core::device::Device::device_id(scope.device())
                != infer_core::device::Device::device_id(self.head.device())
        {
            return Err(OpError::Shape(
                "block proposer requires a healthy TP1 session on its device".into(),
            ));
        }
        Ok(())
    }
    fn finish<R>(&mut self, result: OpResult<R>, scope: &D::Scope) -> OpResult<R> {
        if result.is_err() {
            self.poisoned = true;
            let _ = scope.synchronize();
        }
        result
    }
}

impl<T: Dtype, D: LlmBackend, H: BlockDraft<T, D>> DraftProposer<T, D> for BlockProposer<T, D, H> {
    fn dims(&self) -> ModelDims {
        self.head.dims()
    }
    fn feature_width(&self) -> usize {
        self.head.feature_width()
    }
    fn context_len(&self) -> usize {
        self.len
    }
    fn committed_len(&self) -> usize {
        self.len
    }
    fn prepare_metrics(&self, scope: &D::Scope) -> OpResult<()> {
        self.metrics.prepare_gpu(scope)
    }
    fn reset(&mut self) {
        self.len = 0;
        self.poisoned = false;
    }
    fn observe_with_input(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
        _device_input: Option<&Tensor<i32, D>>,
    ) -> OpResult<()> {
        self.validate(scope)?;
        let n = ids.len();
        if n == 0
            || n > self.capacity
            || start != self.len
            || start
                .checked_add(n)
                .is_none_or(|end| end > self.max_context)
            || ids
                .iter()
                .any(|&t| t < 0 || t as usize >= self.head.dims().vocab_size)
            || hidden.shape().as_slice() != [n, self.head.feature_width()]
            || infer_core::device::Device::device_id(hidden.device())
                != infer_core::device::Device::device_id(scope.device())
        {
            return Err(OpError::Shape(
                "block proposer target feature history/layout mismatch".into(),
            ));
        }
        let result = ExecutionPlan::eager(Phase::CatchUp, 1, n, WorkspaceUse::Proposer).execute(
            &self.metrics,
            |_| {
                self.workspace.prepare_observe_index(n, start)?;
                let index = self.workspace.index(0, n)?;
                self.head.cache_features(
                    hidden,
                    &mut ModelCacheView::full(&mut self.kv, &index),
                    &StepCtx::new(scope, &self.workspace.plan),
                )?;
                scope.synchronize()
            },
        );
        self.finish(result, scope)?;
        self.len += n;
        Ok(())
    }
    fn draft_with_device(
        &mut self,
        pending: i32,
        count: usize,
        scope: &D::Scope,
    ) -> OpResult<(Vec<i32>, Tensor<i32, D>)> {
        self.validate(scope)?;
        let n = count
            .checked_add(1)
            .ok_or_else(|| OpError::Shape("block width overflow".into()))?;
        if self.len == 0
            || n > self.noise_ids.len()
            || n > self.capacity
            || self
                .len
                .checked_add(n)
                .is_none_or(|end| end > self.max_context)
            || pending < 0
            || pending as usize >= self.head.dims().vocab_size
        {
            return Err(OpError::Shape(
                "block draft exceeds history/capacity/vocabulary".into(),
            ));
        }
        let result = ExecutionPlan::eager(Phase::Draft, 1, count, WorkspaceUse::Proposer).execute(
            &self.metrics,
            |_| {
                // Restore every mask on each round, including after a shorter
                // block or rejected suffix. The target tape overwrites these IDs.
                self.noise_ids[..n].fill(self.head.mask_token_id());
                self.noise_ids[0] = pending;
                self.workspace
                    .prepare_observe(&self.noise_ids[..n], self.len)?;
                if count > 0 {
                    self.workspace.plan.kind = BatchKind::Spec {
                        mask: MaskMode::Full,
                        mask_handle: None,
                    };
                    let index = self.workspace.prepared_index(0, n, scope)?;
                    let ctx = StepCtx::new(scope, &self.workspace.plan);
                    let mut logits = self.logits.narrow(0, 0, count)?;
                    self.head.forward_block(
                        &self.workspace.input(n)?,
                        &mut ModelCacheView::full(&mut self.kv, &index),
                        &mut logits,
                        &ctx,
                    )?;
                    D::argmax_into(
                        &ctx,
                        &logits,
                        &mut self.workspace.drafts(count)?,
                        &self.argmax_ws,
                        None,
                    )?;
                }
                scope.synchronize()?;
                let ids = if count == 0 {
                    vec![]
                } else {
                    self.workspace.drafts(count)?.to_host_vec()?
                };
                if ids
                    .iter()
                    .any(|&t| t < 0 || t as usize >= self.head.dims().vocab_size)
                {
                    return Err(OpError::Shape(
                        "block draft produced an invalid token".into(),
                    ));
                }
                Ok((ids, self.workspace.input(n)?))
            },
        );
        // len never changes here: temporary block K/V is unreachable next
        // round until confirmed target features overwrite the corresponding rows.
        self.finish(result, scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        cache::{CacheLayout, LayerCacheSpec, LayerCacheView},
        exec::HostScope,
        ports::FusedOps,
    };
    use crate::infrastructure::cpu::Cpu;
    use std::{cell::Cell, rc::Rc};

    struct FailingHead {
        layout: CacheLayout,
        fail: Rc<Cell<bool>>,
    }
    impl BlockDraft<f32, Cpu> for FailingHead {
        fn dims(&self) -> ModelDims {
            ModelDims {
                dim: 2,
                kv_dim: 2,
                vocab_size: 8,
                num_layers: 1,
                ..Default::default()
            }
        }
        fn device(&self) -> &Cpu {
            &Cpu
        }
        fn cache_layout(&self) -> &CacheLayout {
            &self.layout
        }
        fn feature_width(&self) -> usize {
            2
        }
        fn block_size(&self) -> usize {
            4
        }
        fn mask_token_id(&self) -> i32 {
            7
        }
        fn prepare(&mut self, _capacity: usize) -> OpResult<()> {
            Ok(())
        }
        fn cache_features(
            &self,
            features: &Tensor<f32, Cpu>,
            cache: &mut ModelCacheView<'_, f32, Cpu>,
            ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<()> {
            let LayerCacheView::Full(mut kv) = cache.layer(self.layout.layers()[0])? else {
                unreachable!()
            };
            // Simulate a layer that fails after modifying persistent device state.
            Cpu::scatter_kv_paged(ctx, features, features, &mut kv.layer_mut(0), 2)?;
            if self.fail.get() {
                return Err(OpError::Kernel("injected context failure".into()));
            }
            Ok(())
        }
        fn forward_block(
            &self,
            _ids: &Tensor<i32, Cpu>,
            _cache: &mut ModelCacheView<'_, f32, Cpu>,
            _logits: &mut Tensor<f32, Cpu>,
            _ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<()> {
            Err(OpError::Kernel("injected block failure".into()))
        }
    }

    #[test]
    fn partial_context_writes_and_draft_failure_require_reset_before_reuse() {
        let fail = Rc::new(Cell::new(false));
        let head = FailingHead {
            layout: CacheLayout::new([LayerCacheSpec::Full { kv_dim: 2 }]).unwrap(),
            fail: fail.clone(),
        };
        let scope = HostScope::new(Cpu);
        let mut proposer = BlockProposer::new(head, 16, 4, &Cpu).unwrap();
        let features = Tensor::from_host_slice(&[1.0, 2.0], [1, 2], &Cpu).unwrap();
        proposer.observe(&[1], 0, &features, &scope).unwrap();
        fail.set(true);
        assert!(proposer.observe(&[2], 1, &features, &scope).is_err());
        assert_eq!(proposer.context_len(), 1);
        assert_eq!(
            &proposer.kv.layers[0].k.to_host_vec().unwrap()[2..4],
            &[1.0, 2.0]
        );
        fail.set(false);
        assert!(proposer.observe(&[2], 1, &features, &scope).is_err());
        assert!(proposer.draft_with_device(2, 0, &scope).is_err());
        proposer.reset();
        proposer.observe(&[1], 0, &features, &scope).unwrap();
        assert!(proposer.draft_with_device(2, 1, &scope).is_err());
        assert_eq!(proposer.context_len(), 1);
        assert!(proposer.draft_with_device(2, 0, &scope).is_err());
        proposer.reset();
        proposer.observe(&[1], 0, &features, &scope).unwrap();
        assert_eq!(
            proposer
                .draft_with_device(2, 0, &scope)
                .unwrap()
                .1
                .to_host_vec()
                .unwrap(),
            [2]
        );
    }
}
