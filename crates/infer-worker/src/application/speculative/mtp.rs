//! MTP cache ownership, autoregressive drafting, and target-feature catch-up.
use super::prefill::MtpPrefill;
use crate::components::mtp::{MtpHead, MtpInput};
use crate::domain::cache::ModelCacheView;
use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecScope, StepCtx};
use crate::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use crate::domain::model::DecoderReadout;
use crate::domain::plan::{BatchKind, BatchPlan};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// One sequence's draft state. Speculative writes never advance `alignment`.
/// Catch-up overwrites them using actual target hidden states after verification,
/// even when every draft was accepted. No target cache or sampling policy here.
pub struct MtpProposer<T: Dtype, D: LlmBackend, H: DecoderReadout<T, D>> {
    head: MtpHead<T, D, H>,
    kv: PagedKvPool<T, D>,
    alignment: MtpPrefill<T, D>,
    max_context: usize,
    max_tokens: usize,
}

impl<T: Dtype, D: LlmBackend, H: DecoderReadout<T, D>> MtpProposer<T, D, H> {
    pub fn new(
        mut head: MtpHead<T, D, H>,
        max_context: usize,
        max_tokens: usize,
        device: &D,
    ) -> OpResult<Self> {
        if max_context == 0
            || max_context > i32::MAX as usize
            || max_tokens == 0
            || max_tokens > max_context
            || head.cache_layout().has_linear()
            || infer_core::device::Device::device_id(head.device())
                != infer_core::device::Device::device_id(device)
        {
            return Err(OpError::Shape(
                "invalid MTP proposer capacity or cache layout".into(),
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
        Ok(Self {
            head,
            kv,
            alignment: MtpPrefill::default(),
            max_context,
            max_tokens,
        })
    }

    pub fn reset(&mut self) {
        self.alignment.reset();
    }

    /// The target has L cached tokens; this head has L-1 committed pairs.
    pub fn committed_len(&self) -> usize {
        self.alignment.pending().map_or(0, |(p, _)| p as usize)
    }

    /// Synchronize target outputs before calling. Default-stream alignment
    /// copies are drained before launching on the supplied execution scope.
    pub fn observe(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
    ) -> OpResult<()> {
        self.validate(ids, start, scope)?;
        if hidden.shape().as_slice() != [ids.len(), self.head.dims().dim]
            || infer_core::device::Device::device_id(hidden.device())
                != infer_core::device::Device::device_id(scope.device())
        {
            return Err(OpError::Shape(
                "MTP target hidden shape/device mismatch".into(),
            ));
        }
        scope.synchronize()?;
        let positions = (start as i32..(start + ids.len()) as i32).collect::<Vec<_>>();
        let chunk = self.alignment.prepare(ids, &positions, hidden)?;
        infer_core::device::MemoryPort::synchronize(scope.device())?;
        if !chunk.next_token_ids.is_empty() {
            run_head(
                &self.head,
                &mut self.kv,
                &chunk.next_token_ids,
                chunk.positions[0] as usize,
                &chunk.target_hidden,
                scope,
            )?;
        }
        scope.synchronize()?;
        chunk.commit();
        Ok(())
    }

    pub fn draft(&mut self, pending: i32, count: usize, scope: &D::Scope) -> OpResult<Vec<i32>> {
        let (position, hidden) = self
            .alignment
            .pending()
            .ok_or_else(|| OpError::Shape("MTP requires target prefill".into()))?;
        self.validate(&[pending], position as usize, scope)?;
        if count > self.max_tokens
            || (position as usize)
                .checked_add(count)
                .is_none_or(|end| end > self.max_context)
        {
            return Err(OpError::Shape("MTP draft exceeds capacity".into()));
        }
        let mut conditioning = hidden.clone();
        let mut token = pending;
        let mut draft = Vec::with_capacity(count);
        for i in 0..count {
            let (next_hidden, plan) = run_head(
                &self.head,
                &mut self.kv,
                &[token],
                position as usize + i,
                &conditioning,
                scope,
            )?;
            let ctx = StepCtx::new(scope, &plan);
            let mut logits = Tensor::zeros([1, self.head.dims().vocab_size], scope.device())?;
            self.head
                .project_logits_into(&next_hidden, &mut logits, &ctx)?;
            let predicted = D::argmax(&ctx, &logits)?;
            if predicted.len() != 1
                || predicted[0] < 0
                || predicted[0] as usize >= self.head.dims().vocab_size
            {
                return Err(OpError::Shape("invalid MTP argmax output".into()));
            }
            token = predicted[0];
            draft.push(token);
            conditioning = next_hidden;
        }
        Ok(draft)
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
                "MTP requires valid text tokens, capacity, and TP1".into(),
            ));
        }
        Ok(())
    }
}

fn run_head<T: Dtype, D: LlmBackend, H: DecoderReadout<T, D>>(
    head: &MtpHead<T, D, H>,
    kv: &mut PagedKvPool<T, D>,
    ids: &[i32],
    start: usize,
    hidden: &Tensor<T, D>,
    scope: &D::Scope,
) -> OpResult<(Tensor<T, D>, BatchPlan)> {
    let n = ids.len();
    let device = scope.device();
    let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], device);
    let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&[n as i32]);
    let positions = (start as i32..(start + n) as i32).collect::<Vec<_>>();
    let plan = BatchPlan {
        kind: if n == 1 {
            BatchKind::DecodeOnly
        } else {
            BatchKind::Ragged
        },
        num_tokens: n,
        batch: 1,
        q_lens: vec![n as i32],
        kv_lens: vec![(start + n) as i32],
        seq_positions: vec![start as i32],
        rope_positions: positions.clone(),
        max_blocks_per_seq: kv.num_blocks,
        block_size: kv.block_size,
        total_q_tiles: req.len() as i32,
    };
    let index = KvIndexTensors {
        block_tables: Tensor::from_host_slice(
            &(0..kv.num_blocks as i32).collect::<Vec<_>>(),
            [1, kv.num_blocks],
            device,
        )?,
        cu_q_lens: ints(&cu)?,
        kv_lens: ints(&plan.kv_lens)?,
        seq_positions: ints(&plan.seq_positions)?,
        seq_lens_step: ints(&plan.q_lens)?,
        rope_positions: ints(&positions)?,
        block2req: ints(&req)?,
        block2tile: ints(&tile)?,
        valid_q_tiles: ints(&[req.len() as i32])?,
        valid_suffix_q_tiles: ints(&[req.len() as i32])?,
    };
    let input = ints(ids)?;
    let mut output = Tensor::zeros([n, head.dims().dim], device)?;
    let result = head.forward_hidden_into(
        MtpInput {
            next_token_ids: &input,
            target_hidden: hidden,
        },
        &mut ModelCacheView::full(kv, &index),
        &mut output,
        &StepCtx::new(scope, &plan),
    );
    // These local control tensors must outlive all GPU readers, including errors.
    let completed = scope.synchronize();
    result?;
    completed?;
    Ok((output, plan))
}
