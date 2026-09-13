//! Conditioned draft cache ownership, autoregressive drafting, and target-feature catch-up.
use super::prefill::ShiftedFeatureAlignment;
use crate::application::execution::{ExecutionMetrics, ExecutionPlan, Phase, WorkspaceUse};
use crate::domain::cache::ModelCacheView;
use crate::domain::draft::ConditionedDraft;
use crate::domain::draft_scratch::DraftWorkspace;
use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecScope, StepCtx};
use crate::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use crate::domain::plan::BatchPlan;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// One sequence's draft state. Speculative writes never advance `alignment`.
/// Catch-up overwrites them using actual target hidden states after verification,
/// even when every draft was accepted. No target cache or sampling policy here.
pub struct ConditionedProposer<T: Dtype, D: LlmBackend, H: ConditionedDraft<T, D>> {
    metrics: ExecutionMetrics,
    head: H,
    kv: PagedKvPool<T, D>,
    alignment: ShiftedFeatureAlignment<T, D>,
    max_context: usize,
    max_tokens: usize,
    workspace: DraftWorkspace<T, D>,
    projected_features: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend, H: ConditionedDraft<T, D>> ConditionedProposer<T, D, H> {
    pub fn new(mut head: H, max_context: usize, max_tokens: usize, device: &D) -> OpResult<Self> {
        if max_context == 0
            || max_context > i32::MAX as usize
            || max_tokens == 0
            || max_tokens > max_context
            || head.cache_layout().has_linear()
            || infer_core::device::Device::device_id(head.device())
                != infer_core::device::Device::device_id(device)
        {
            return Err(OpError::Shape(
                "invalid conditioned draft proposer capacity or cache layout".into(),
            ));
        }
        head.prepare(max_tokens, 1)?;
        let kv_dim = head.dims().kv_dim;
        let block_size = 16;
        let num_blocks = max_context.div_ceil(block_size);
        let kv = PagedKvPool {
            layers: (0..head.cache_layout().num_full_layers())
                .map(|_| {
                    Ok(PagedKvLayer {
                        k: Tensor::zeros([num_blocks, block_size, kv_dim], device)?,
                        v: Tensor::zeros([num_blocks, block_size, kv_dim], device)?,
                    })
                })
                .collect::<OpResult<_>>()?,
            num_blocks,
            block_size,
            kv_dim,
            quant: KvQuantTier::None,
            seq_kv_len: Default::default(),
        };
        let alignment =
            ShiftedFeatureAlignment::with_capacity(max_tokens, head.dims().dim, device)?;
        let workspace = DraftWorkspace::new(
            crate::domain::model::ModelDims {
                vocab_size: head.logits_vocab_size(),
                ..head.dims()
            },
            max_context,
            max_tokens,
            device,
        )?;
        let projected_features = Tensor::zeros([max_tokens, head.dims().dim], device)?;
        Ok(Self {
            projected_features,
            metrics: ExecutionMetrics::from_env("proposer"),
            head,
            kv,
            alignment,
            workspace,
            max_context,
            max_tokens,
        })
    }

    pub(crate) fn prepare_metrics(&self, scope: &D::Scope) -> OpResult<()> {
        self.metrics.prepare_gpu(scope)
    }

    pub fn reset(&mut self) {
        self.alignment.reset();
    }

    /// The target has L cached tokens; this head has L-1 committed pairs.
    pub fn committed_len(&self) -> usize {
        self.alignment.pending().map_or(0, |(p, _)| p as usize)
    }

    /// Synchronize target outputs before calling. Alignment copies and catch-up
    /// execute in order on the supplied scope and complete before commit.
    pub fn observe(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
    ) -> OpResult<()> {
        self.observe_with_input(ids, start, hidden, scope, None)
    }

    /// Continuation catch-up can consume the same tape as target verification.
    pub(crate) fn observe_with_input(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
        device_input: Option<&Tensor<i32, D>>,
    ) -> OpResult<()> {
        let result = ExecutionPlan::eager(Phase::CatchUp, 1, ids.len(), WorkspaceUse::Proposer)
            .execute(&self.metrics.clone(), |_| {
                self.observe_with_input_local(ids, start, hidden, scope, device_input)
            });
        if result.is_err() {
            let _ = scope.synchronize();
        }
        result
    }

    fn observe_with_input_local(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
        device_input: Option<&Tensor<i32, D>>,
    ) -> OpResult<()> {
        self.validate(ids, start, scope)?;
        if let Some(input) = device_input
            && (self.alignment.pending().is_none()
                || input.shape().as_slice() != [ids.len()]
                || !input.is_contiguous()
                || infer_core::device::Device::device_id(input.device())
                    != infer_core::device::Device::device_id(scope.device()))
        {
            return Err(OpError::Shape(
                "invalid conditioned draft catch-up device tape".into(),
            ));
        }
        if hidden.shape().as_slice() != [ids.len(), self.head.feature_width()]
            || infer_core::device::Device::device_id(hidden.device())
                != infer_core::device::Device::device_id(scope.device())
        {
            return Err(OpError::Shape(
                "conditioned draft target hidden shape/device mismatch".into(),
            ));
        }
        // Feature fusion is a head operation; sequence alignment always stores
        // the head's H-wide conditioning, independently of target readout width.
        self.workspace.prepare_observe_index(ids.len(), start)?;
        let mut projected = self.projected_features.narrow(0, 0, ids.len())?;
        self.head.project_features_into(
            hidden,
            &mut projected,
            &StepCtx::new(scope, &self.workspace.plan),
        )?;
        let positions = self.workspace.positions(start, ids.len());
        let chunk = self
            .alignment
            .prepare_on(ids, positions, &projected, scope)?;
        let result = (|| {
            if !chunk.next_token_ids.is_empty() {
                let input = match device_input {
                    Some(input) => {
                        self.workspace.prepare_observe_index(
                            chunk.next_token_ids.len(),
                            chunk.positions[0] as usize,
                        )?;
                        input.clone()
                    }
                    None => {
                        self.workspace
                            .prepare_observe(&chunk.next_token_ids, chunk.positions[0] as usize)?;
                        self.workspace.input(chunk.next_token_ids.len())?
                    }
                };
                run_head(
                    &self.head,
                    &mut self.kv,
                    &input,
                    &chunk.target_hidden,
                    &mut self.workspace.hidden(0, chunk.next_token_ids.len())?,
                    &self
                        .workspace
                        .prepared_index(0, chunk.next_token_ids.len(), scope)?,
                    &self.workspace.plan,
                    scope,
                )?;
            }
            Ok(())
        })();
        // Includes alignment copies and the catch-up forward, even on errors.
        let completed = ExecutionPlan::eager(Phase::Wait, 1, ids.len(), WorkspaceUse::Proposer)
            .execute(&self.metrics, |_| scope.synchronize());
        result?;
        completed?;
        chunk.commit();
        Ok(())
    }

    pub fn draft(&mut self, pending: i32, count: usize, scope: &D::Scope) -> OpResult<Vec<i32>> {
        self.draft_with_device(pending, count, scope)
            .map(|(ids, _)| ids)
    }

    /// Host verification metadata and its matching device input tape. The tape
    /// remains valid until the next proposer operation; callers consume it now.
    pub(crate) fn draft_with_device(
        &mut self,
        pending: i32,
        count: usize,
        scope: &D::Scope,
    ) -> OpResult<(Vec<i32>, Tensor<i32, D>)> {
        ExecutionPlan::eager(Phase::Draft, 1, count, WorkspaceUse::Proposer)
            .execute(&self.metrics.clone(), |_| {
                self.draft_with_device_local(pending, count, scope)
            })
    }

    fn draft_with_device_local(
        &mut self,
        pending: i32,
        count: usize,
        scope: &D::Scope,
    ) -> OpResult<(Vec<i32>, Tensor<i32, D>)> {
        let (position, hidden) = self
            .alignment
            .pending()
            .ok_or_else(|| OpError::Shape("conditioned draft requires target prefill".into()))?;
        self.validate(&[pending], position as usize, scope)?;
        if count > self.max_tokens
            || (position as usize)
                .checked_add(count)
                .is_none_or(|end| end > self.max_context)
        {
            return Err(OpError::Shape("draft exceeds capacity".into()));
        }
        let mut conditioning = hidden.clone();
        self.workspace
            .prepare_draft(pending, position as usize, count)?;
        let result = (|| {
            // The head reads pending from the packed control upload. Preserve
            // it beside the draft outputs for target's contiguous input tape.
            D::copy_tensor(
                scope,
                &self.workspace.token(0)?,
                &mut self.workspace.input(1)?,
            )?;
            for i in 0..count {
                self.workspace.set_decode_position(position as usize + i);
                let mut next_hidden = self.workspace.hidden(i % 2, 1)?;
                run_head(
                    &self.head,
                    &mut self.kv,
                    &self.workspace.token(i)?,
                    &conditioning,
                    &mut next_hidden,
                    &self.workspace.prepared_index(i, 1, scope)?,
                    &self.workspace.plan,
                    scope,
                )?;
                let ctx = StepCtx::new(scope, &self.workspace.plan);
                self.head
                    .project_logits_into(&next_hidden, &mut self.workspace.logits, &ctx)?;
                // The next embedding reads this device token directly. Only the
                // completed draft vector crosses to the CPU, once per round.
                D::argmax_into(
                    &ctx,
                    &self.workspace.logits,
                    &mut self.workspace.token(i + 1)?,
                    &self.workspace.argmax_ws,
                    None,
                )?;
                if let Some(map) = self.head.token_map() {
                    D::remap_ids_inplace(scope, &mut self.workspace.token(i + 1)?, map)?;
                }
                conditioning = next_hidden;
            }
            // Downloads use the device's default stream, which need not be the
            // caller's execution stream. Fence once after the complete chain.
            ExecutionPlan::eager(Phase::Wait, 1, count, WorkspaceUse::Proposer)
                .execute(&self.metrics, |_| scope.synchronize())?;
            self.workspace.drafts(count)?.to_host_vec()
        })();
        // Workspace owns all intermediates; drain before reuse, including errors.
        if result.is_err() {
            let _ = scope.synchronize();
        }
        let draft = result?;
        if draft
            .iter()
            .any(|&id| id < 0 || id as usize >= self.head.dims().vocab_size)
        {
            return Err(OpError::Shape(
                "invalid conditioned draft argmax output".into(),
            ));
        }
        Ok((draft, self.workspace.input(count + 1)?))
    }

    fn validate(&self, ids: &[i32], start: usize, scope: &D::Scope) -> OpResult<()> {
        if scope.topology().tp.size != 1
            || ids.is_empty()
            || ids.len() > self.max_tokens
            || infer_core::device::Device::device_id(self.head.device())
                != infer_core::device::Device::device_id(scope.device())
            || start
                .checked_add(ids.len())
                .is_none_or(|end| end > self.max_context)
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.head.dims().vocab_size)
        {
            return Err(OpError::Shape(
                "conditioned draft requires valid text tokens, capacity, and TP1".into(),
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_head<T: Dtype, D: LlmBackend, H: ConditionedDraft<T, D>>(
    head: &H,
    kv: &mut PagedKvPool<T, D>,
    ids: &Tensor<i32, D>,
    hidden: &Tensor<T, D>,
    output: &mut Tensor<T, D>,
    index: &KvIndexTensors<D>,
    plan: &BatchPlan,
    scope: &D::Scope,
) -> OpResult<()> {
    head.forward_hidden_into(
        ids,
        hidden,
        &mut ModelCacheView::full(kv, index),
        output,
        &StepCtx::new(scope, plan),
    )
}
