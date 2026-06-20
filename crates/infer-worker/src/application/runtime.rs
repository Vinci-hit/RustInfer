use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::domain::component::{Hidden, LayerRange};
use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecDevice as Device, ExecScope};
use crate::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use crate::domain::model::{DecoderModel, ModelDims, SampleRows};
use crate::domain::plan::{BatchKind, BatchPlan, SampledToken, StepOutput, StepRequest};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::sampler::Sampler;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

pub struct Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    pub model: M,
    pub kv_pool: PagedKvPool<T, D>,
    pub kv_index: KvIndexTensors<D>,
    pub hidden: Hidden<T, D>,
    pub scope: <D as Device>::Scope,
    pub sampler: Box<dyn Sampler<T, D>>,
    pub dims: ModelDims,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,
    pub max_seq_len: usize,
    pub cap_num_tokens: usize,
    pub cap_batch: usize,
    pub capture_sizes: Vec<usize>,
    pub graph: Option<GraphRunner<D>>,
}

impl<T, D, M> Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: M,
        scope: <D as Device>::Scope,
        sampler: Box<dyn Sampler<T, D>>,
        num_blocks: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        cap_num_tokens: usize,
        cap_batch: usize,
        capture_sizes: Vec<usize>,
    ) -> OpResult<Self> {
        let dims = model.dims();
        dims.validate()?;
        if block_size == 0 {
            return Err(OpError::Shape("Runtime::new: block_size=0".into()));
        }
        if max_blocks_per_seq == 0 {
            return Err(OpError::Shape("Runtime::new: max_blocks_per_seq=0".into()));
        }
        if cap_num_tokens == 0 || cap_batch == 0 {
            return Err(OpError::Shape(format!(
                "Runtime::new: invalid caps tokens={} batch={}",
                cap_num_tokens, cap_batch
            )));
        }

        let device = scope.device();
        let mut layers = Vec::with_capacity(dims.num_layers);
        for _ in 0..dims.num_layers {
            layers.push(PagedKvLayer {
                k: D::alloc_tensor(
                    Shape::from_slice(&[num_blocks, block_size, dims.kv_dim]),
                    device,
                )?,
                v: D::alloc_tensor(
                    Shape::from_slice(&[num_blocks, block_size, dims.kv_dim]),
                    device,
                )?,
            });
        }

        let kv_pool = PagedKvPool {
            layers,
            num_blocks,
            block_size,
            kv_dim: dims.kv_dim,
            quant: KvQuantTier::None,
            seq_kv_len: std::collections::HashMap::new(),
        };

        let cap_total_q_tiles = ((cap_num_tokens + crate::domain::plan::RAGGED_Q_TILE as usize
            - 1)
            / crate::domain::plan::RAGGED_Q_TILE as usize)
            .max(1)
            .max(cap_batch);
        let alloc_i32 = |n: usize| D::alloc_tensor::<i32>(Shape::from_slice(&[n.max(1)]), device);
        let kv_index = KvIndexTensors {
            block_tables: alloc_i32(cap_batch * max_blocks_per_seq)?,
            cu_q_lens: alloc_i32(cap_batch + 1)?,
            kv_lens: alloc_i32(cap_batch)?,
            seq_positions: alloc_i32(cap_batch)?,
            seq_lens_step: alloc_i32(cap_batch)?,
            rope_positions: alloc_i32(cap_num_tokens)?,
            block2req: alloc_i32(cap_total_q_tiles)?,
            block2tile: alloc_i32(cap_total_q_tiles)?,
        };

        let hidden = Hidden {
            stream: D::alloc_tensor(Shape::from_slice(&[cap_num_tokens, dims.dim]), device)?,
        };

        Ok(Self {
            model,
            kv_pool,
            kv_index,
            hidden,
            scope,
            sampler,
            dims,
            block_size,
            max_blocks_per_seq,
            max_seq_len,
            cap_num_tokens,
            cap_batch,
            capture_sizes,
            graph: None,
        })
    }

    pub fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput> {
        let plan = self.build_plan(req)?;
        self.upload_index(&plan, req)?;
        match self.decide(&plan) {
            GraphDecision::Eager => self.step_eager(&plan, req),
            GraphDecision::Graph(slot) => self.step_graph(slot, &plan, req),
        }
    }

    pub fn step_eager(
        &mut self,
        plan: &crate::domain::plan::BatchPlan,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        let input_ids = self.input_ids_tensor(req, plan)?;
        let mut hidden = Hidden {
            stream: self.hidden.stream.view_raw(
                Shape::from_slice(&[plan.num_tokens, self.dims.dim]),
                Shape::from_slice(&[plan.num_tokens.max(1), self.dims.dim]).contiguous_strides(),
                0,
                true,
            ),
        };
        let ctx = crate::domain::exec::StepCtx::new(&self.scope, plan);
        let _guard = self.scope.enter();
        let mut kv = self
            .kv_pool
            .view(LayerRange::all(self.dims.num_layers), &self.kv_index);

        self.model.embed(&input_ids, &mut hidden, &ctx)?;
        self.model.decode_layers(
            LayerRange::all(self.dims.num_layers),
            &mut hidden,
            &mut kv,
            &ctx,
        )?;
        drop(kv);

        let logits = self.model.finalize(&hidden, SampleRows::All, &ctx)?;
        let sids: Vec<u64> = req.seqs.iter().map(|seq| seq.sequence_id).collect();
        let (tokens, accepted, speculative_len) = if req.draft_tokens.is_empty() {
            let sampled = self.sampler.sample(&logits.0, &req.sampling, &ctx)?;
            let mut tokens: Vec<Vec<SampledToken>> = sampled
                .tokens
                .into_iter()
                .map(|token| vec![token])
                .collect();
            tokens.resize_with(plan.batch, Vec::new);
            let accepted: Vec<u32> = plan.q_lens.iter().map(|&q| q.max(0) as u32).collect();
            let speculative_len = vec![0; plan.batch];
            (tokens, accepted, speculative_len)
        } else {
            let draft_tokens = flatten_draft_tokens(req, plan)?;
            let draft_probs = D::alloc_tensor::<f32>(
                Shape::from_slice(&[logits.0.numel().max(1)]),
                logits.0.device(),
            )?;
            let verify =
                self.sampler
                    .verify(&logits.0, &draft_tokens, &draft_probs, &req.sampling, &ctx)?;
            if verify.accepted_count.len() != plan.batch {
                return Err(OpError::Shape(format!(
                    "Runtime::step: verify accepted_count {} != batch {}",
                    verify.accepted_count.len(),
                    plan.batch
                )));
            }
            let speculative_len: Vec<u32> = req
                .draft_tokens
                .iter()
                .map(|tokens| tokens.len() as u32)
                .collect();
            for (i, (&accepted, &spec)) in verify
                .accepted_count
                .iter()
                .zip(speculative_len.iter())
                .enumerate()
            {
                if accepted > spec {
                    return Err(OpError::Shape(format!(
                        "Runtime::step: seq[{}] accepted {} > spec {}",
                        i, accepted, spec
                    )));
                }
            }
            let tokens = spec_tokens(req, &verify.accepted_count, verify.bonus_token)?;
            for (seq, &spec) in req.seqs.iter().zip(speculative_len.iter()) {
                if spec > 0 {
                    self.kv_pool
                        .seq_kv_len
                        .insert(seq.sequence_id, seq.kv_len_after as u32);
                }
            }
            (tokens, verify.accepted_count, speculative_len)
        };
        // Non-speculative steps do NOT touch the pool's per-seq length map: the
        // worker (`ActiveSeq`) owns the length for ordinary decode/prefill, and
        // `build_plan` trusts the caller-provided `kv_write_start`. Only the
        // speculative commit maintains `seq_kv_len` (for `KvEdit::truncate`).
        // This keeps the map empty under normal operation, so out-of-band
        // eviction (cancel/preempt/drain) can never orphan an entry.
        if !req.draft_tokens.is_empty() {
            self.kv_pool
                .edit()
                .apply_step(&sids, &accepted, &speculative_len)?;
        }

        let finished = finished_flags(req, &tokens);
        for (sid, done) in sids.iter().zip(finished.iter()) {
            if *done {
                self.kv_pool.seq_kv_len.remove(sid);
            }
        }

        Ok(StepOutput {
            tokens,
            accepted,
            finished,
            hidden_tap: None,
        })
    }

    pub fn decide(&self, plan: &crate::domain::plan::BatchPlan) -> GraphDecision {
        self.graph
            .as_ref()
            .map_or(GraphDecision::Eager, |graph| graph.decide(plan))
    }

    pub fn prime_graphs(&mut self) -> OpResult<()> {
        self.graph = None;
        if self.capture_sizes.is_empty() || !self.scope.supports_graphs() {
            return Ok(());
        }
        self.graph = Some(GraphRunner::new(
            self.capture_sizes.clone(),
            self.cap_batch,
        )?);
        Ok(())
    }

    fn step_graph(
        &mut self,
        slot: GraphSlotId,
        plan: &crate::domain::plan::BatchPlan,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        let Some(graph) = self.graph.as_ref() else {
            return self.step_eager(plan, req);
        };
        if graph.slot_size(slot).is_none() {
            return Err(OpError::Shape(format!(
                "Runtime::step: graph slot {} is out of range",
                slot.0
            )));
        }

        // V2 graph capture/replay is gated by `ExecScope::supports_graphs`.
        // Until a scope installs real replay, the safe behavior is eager.
        self.step_eager(plan, req)
    }

    fn build_plan(&self, req: &StepRequest) -> OpResult<BatchPlan> {
        let batch = req.seqs.len();
        if batch == 0 {
            return Err(OpError::Shape("Runtime::step: empty request".into()));
        }
        if batch > self.cap_batch {
            return Err(OpError::Shape(format!(
                "Runtime::step: batch {} > cap {}",
                batch, self.cap_batch
            )));
        }
        validate_step_request_vectors(req, batch)?;

        let mut q_lens = Vec::with_capacity(batch);
        let mut kv_lens = Vec::with_capacity(batch);
        let mut seq_positions = Vec::with_capacity(batch);
        let mut rope_positions = Vec::new();
        let mut total_tokens = 0usize;
        for (i, seq) in req.seqs.iter().enumerate() {
            if seq.input_ids.is_empty() {
                return Err(OpError::Shape(format!("Runtime::step: seq[{}] empty", i)));
            }
            if seq.input_ids.len() != seq.positions.len() {
                return Err(OpError::Shape(format!(
                    "Runtime::step: seq[{}] input_ids={} positions={}",
                    i,
                    seq.input_ids.len(),
                    seq.positions.len()
                )));
            }
            if seq.block_table.len() > self.max_blocks_per_seq {
                return Err(OpError::Shape(format!(
                    "Runtime::step: seq[{}] block_table {} > max {}",
                    i,
                    seq.block_table.len(),
                    self.max_blocks_per_seq
                )));
            }
            total_tokens += seq.input_ids.len();
            q_lens.push(seq.input_ids.len() as i32);
            if seq.kv_write_start < 0 || seq.kv_len_after < 0 {
                return Err(OpError::Shape(format!(
                    "Runtime::step: seq[{}] negative kv range start={} after={}",
                    i, seq.kv_write_start, seq.kv_len_after
                )));
            }
            // The worker (`ActiveSeq`) is authoritative for KV positions — it owns
            // each sequence's block table and length. Validate internal
            // consistency only; do NOT cross-check a redundant pool-side length,
            // which would spuriously reject a reused sequence id whose pool entry
            // was left behind by an out-of-band eviction (cancel/preempt/drain).
            let start = seq.kv_write_start as u32;
            let expected_after = start.checked_add(seq.input_ids.len() as u32).ok_or_else(|| {
                OpError::Shape(format!(
                    "Runtime::step: seq[{}] kv_len overflow start={} q={}",
                    i,
                    start,
                    seq.input_ids.len()
                ))
            })?;
            if seq.kv_len_after as u32 != expected_after {
                return Err(OpError::Shape(format!(
                    "Runtime::step: seq[{}] kv_len_after {} != start {} + q_len {}",
                    i,
                    seq.kv_len_after,
                    start,
                    seq.input_ids.len()
                )));
            }
            kv_lens.push(expected_after as i32);
            seq_positions.push(start as i32);
            rope_positions.extend_from_slice(&seq.positions);
        }
        if total_tokens > self.cap_num_tokens {
            return Err(OpError::Shape(format!(
                "Runtime::step: total_tokens {} > cap {}",
                total_tokens, self.cap_num_tokens
            )));
        }
        if !req.draft_tokens.is_empty() {
            if req.draft_tokens.len() != batch {
                return Err(OpError::Shape(format!(
                    "Runtime::step: draft_tokens {} != batch {}",
                    req.draft_tokens.len(),
                    batch
                )));
            }
            for (i, (draft, seq)) in req.draft_tokens.iter().zip(req.seqs.iter()).enumerate() {
                if draft.len() != seq.input_ids.len() {
                    return Err(OpError::Shape(format!(
                        "Runtime::step: seq[{}] draft_tokens {} != input_ids {}",
                        i,
                        draft.len(),
                        seq.input_ids.len()
                    )));
                }
            }
        }

        let (_cu_q_lens, block2req, _block2tile) = BatchPlan::plan_ragged_tiles(&q_lens);
        let kind = if !req.draft_tokens.is_empty() {
            BatchKind::Spec {
                mask: crate::domain::plan::MaskMode::Causal,
                mask_handle: None,
            }
        } else if q_lens.iter().all(|&q| q == 1) {
            BatchKind::DecodeOnly
        } else {
            BatchKind::Ragged
        };
        Ok(BatchPlan {
            kind,
            num_tokens: total_tokens,
            batch,
            q_lens,
            kv_lens,
            seq_positions,
            rope_positions,
            max_blocks_per_seq: self.max_blocks_per_seq,
            block_size: self.block_size,
            total_q_tiles: block2req.len() as i32,
        })
    }

    fn input_ids_tensor(&self, req: &StepRequest, plan: &BatchPlan) -> OpResult<Tensor<i32, D>> {
        let mut input_ids = Vec::with_capacity(plan.num_tokens);
        for seq in &req.seqs {
            input_ids.extend_from_slice(&seq.input_ids);
        }
        Tensor::from_host_slice(
            &input_ids,
            Shape::from_slice(&[plan.num_tokens]),
            self.scope.device(),
        )
    }

    fn upload_index(&mut self, plan: &BatchPlan, req: &StepRequest) -> OpResult<()> {
        let (cu_q_lens, block2req, block2tile) = BatchPlan::plan_ragged_tiles(&plan.q_lens);
        let mut block_tables = vec![0i32; plan.batch * self.max_blocks_per_seq];
        for (i, seq) in req.seqs.iter().enumerate() {
            let row = i * self.max_blocks_per_seq;
            for (j, &block) in seq.block_table.iter().enumerate() {
                block_tables[row + j] = block as i32;
            }
        }
        let seq_lens_step = plan.q_lens.clone();

        let device = self.scope.device();
        unsafe {
            upload_i32_prefix(device, &self.kv_index.block_tables, &block_tables)?;
            upload_i32_prefix(device, &self.kv_index.cu_q_lens, &cu_q_lens)?;
            upload_i32_prefix(device, &self.kv_index.kv_lens, &plan.kv_lens)?;
            upload_i32_prefix(device, &self.kv_index.seq_positions, &plan.seq_positions)?;
            upload_i32_prefix(device, &self.kv_index.seq_lens_step, &seq_lens_step)?;
            upload_i32_prefix(device, &self.kv_index.rope_positions, &plan.rope_positions)?;
            upload_i32_prefix(device, &self.kv_index.block2req, &block2req)?;
            upload_i32_prefix(device, &self.kv_index.block2tile, &block2tile)?;
        }
        Ok(())
    }
}

fn flatten_draft_tokens(req: &StepRequest, plan: &BatchPlan) -> OpResult<Vec<i32>> {
    if req.draft_tokens.len() != plan.batch {
        return Err(OpError::Shape(format!(
            "flatten_draft_tokens: draft_tokens {} != batch {}",
            req.draft_tokens.len(),
            plan.batch
        )));
    }
    let mut out = Vec::with_capacity(plan.num_tokens);
    for (i, draft) in req.draft_tokens.iter().enumerate() {
        let expected = plan.q_lens[i].max(0) as usize;
        if draft.len() != expected {
            return Err(OpError::Shape(format!(
                "flatten_draft_tokens: seq[{}] draft {} != q_len {}",
                i,
                draft.len(),
                expected
            )));
        }
        out.extend_from_slice(draft);
    }
    Ok(out)
}

fn spec_tokens(
    req: &StepRequest,
    accepted: &[u32],
    bonus: Vec<SampledToken>,
) -> OpResult<Vec<Vec<SampledToken>>> {
    if accepted.len() != req.seqs.len() || bonus.len() != req.seqs.len() {
        return Err(OpError::Shape(format!(
            "spec_tokens: accepted={} bonus={} batch={}",
            accepted.len(),
            bonus.len(),
            req.seqs.len()
        )));
    }
    let mut out = Vec::with_capacity(req.seqs.len());
    for i in 0..req.seqs.len() {
        let draft = req
            .draft_tokens
            .get(i)
            .ok_or_else(|| OpError::Shape(format!("spec_tokens: missing draft row {}", i)))?;
        let n = accepted[i] as usize;
        if n > draft.len() {
            return Err(OpError::Shape(format!(
                "spec_tokens: accepted {} > draft {} for row {}",
                n,
                draft.len(),
                i
            )));
        }
        let mut row = draft[..n]
            .iter()
            .map(|&token_id| SampledToken {
                token_id,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
            .collect::<Vec<_>>();
        row.push(bonus[i].clone());
        out.push(row);
    }
    Ok(out)
}

fn finished_flags(req: &StepRequest, tokens: &[Vec<SampledToken>]) -> Vec<bool> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let ignore_eos = req.stop.ignore_eos.get(i).copied().unwrap_or(false);
            let hit_eos = !ignore_eos
                && row
                    .iter()
                    .any(|token| req.stop.eos_ids.contains(&token.token_id));
            let generated =
                req.stop.generated_counts.get(i).copied().unwrap_or(0) + row.len() as u32;
            let max = req.stop.max_tokens.get(i).copied().unwrap_or(u32::MAX);
            hit_eos || generated >= max
        })
        .collect()
}

fn validate_step_request_vectors(req: &StepRequest, batch: usize) -> OpResult<()> {
    validate_optional_batch_len("sampling", req.sampling.len(), batch)?;
    validate_optional_batch_len("generated_counts", req.stop.generated_counts.len(), batch)?;
    validate_optional_batch_len("max_tokens", req.stop.max_tokens.len(), batch)?;
    validate_optional_batch_len("ignore_eos", req.stop.ignore_eos.len(), batch)?;
    Ok(())
}

fn validate_optional_batch_len(name: &str, len: usize, batch: usize) -> OpResult<()> {
    if len != 0 && len != batch {
        return Err(OpError::Shape(format!(
            "Runtime::step: {} len {} != batch {}",
            name, len, batch
        )));
    }
    Ok(())
}

unsafe fn upload_i32_prefix<D: Device>(
    device: &D,
    dst: &Tensor<i32, D>,
    host: &[i32],
) -> OpResult<()> {
    if host.is_empty() {
        return Ok(());
    }
    if host.len() > dst.numel() {
        return Err(OpError::Shape(format!(
            "upload_i32_prefix: host {} > dst {}",
            host.len(),
            dst.numel()
        )));
    }
    let bytes = std::mem::size_of_val(host);
    let ptr = unsafe { NonNull::new_unchecked(dst.data_ptr_mut() as *mut u8) };
    unsafe { device.upload_async(ptr, host.as_ptr() as *const u8, bytes) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphDecision {
    Graph(GraphSlotId),
    Eager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphSlotId(pub usize);

pub struct GraphRunner<D: LlmBackend> {
    capture_sizes: Vec<usize>,
    _d: PhantomData<D>,
}

impl<D: LlmBackend> Default for GraphRunner<D> {
    fn default() -> Self {
        Self {
            capture_sizes: Vec::new(),
            _d: PhantomData,
        }
    }
}

impl<D: LlmBackend> GraphRunner<D> {
    pub fn new(mut capture_sizes: Vec<usize>, cap_batch: usize) -> OpResult<Self> {
        capture_sizes.sort_unstable();
        capture_sizes.dedup();
        if capture_sizes.is_empty() {
            return Err(OpError::Shape(
                "GraphRunner::new: capture_sizes is empty".into(),
            ));
        }
        if capture_sizes[0] == 0 {
            return Err(OpError::Shape(
                "GraphRunner::new: capture size 0 is invalid".into(),
            ));
        }
        if let Some(&max_size) = capture_sizes.last() {
            if max_size > cap_batch {
                return Err(OpError::Shape(format!(
                    "GraphRunner::new: capture size {} > cap_batch {}",
                    max_size, cap_batch
                )));
            }
        }
        Ok(Self {
            capture_sizes,
            _d: PhantomData,
        })
    }

    pub fn capture_sizes(&self) -> &[usize] {
        &self.capture_sizes
    }

    pub fn decide(&self, plan: &crate::domain::plan::BatchPlan) -> GraphDecision {
        if !plan.is_decode_only() {
            return GraphDecision::Eager;
        }
        self.slot_for_batch(plan.batch)
            .map_or(GraphDecision::Eager, GraphDecision::Graph)
    }

    pub fn slot_size(&self, slot: GraphSlotId) -> Option<usize> {
        self.capture_sizes.get(slot.0).copied()
    }

    fn slot_for_batch(&self, batch: usize) -> Option<GraphSlotId> {
        if batch == 0 {
            return None;
        }
        let idx = match self.capture_sizes.binary_search(&batch) {
            Ok(exact) => exact,
            Err(insert_point) if insert_point < self.capture_sizes.len() => insert_point,
            Err(_) => return None,
        };
        Some(GraphSlotId(idx))
    }
}

pub fn run_layers_for_tap<T, D, M>(
    runtime: &mut Runtime<T, D, M>,
    range: LayerRange,
    req: &StepRequest,
) -> OpResult<crate::domain::plan::HiddenTap>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    let plan = runtime.build_plan(req)?;
    runtime.upload_index(&plan, req)?;
    let input_ids = runtime.input_ids_tensor(req, &plan)?;
    let mut hidden = Hidden {
        stream: runtime.hidden.stream.view_raw(
            Shape::from_slice(&[plan.num_tokens, runtime.dims.dim]),
            Shape::from_slice(&[plan.num_tokens.max(1), runtime.dims.dim]).contiguous_strides(),
            0,
            true,
        ),
    };
    let ctx = crate::domain::exec::StepCtx::new(&runtime.scope, &plan);
    let _guard = runtime.scope.enter();
    let mut kv = runtime.kv_pool.view(range, &runtime.kv_index);
    runtime.model.embed(&input_ids, &mut hidden, &ctx)?;
    runtime
        .model
        .decode_layers(range, &mut hidden, &mut kv, &ctx)?;
    Ok(crate::domain::plan::HiddenTap {
        at_layer: range.end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::sampler_stack::GreedySampler;
    use crate::domain::component::{Hidden, LayerRange, StageKind};
    use crate::domain::exec::{HostScope, StepCtx};
    use crate::domain::kv::KvView;
    use crate::domain::model::{DecoderModel, Logits, ModelDims, SampleRows};
    use crate::domain::plan::{BatchKind, SeqStep, StopCriteria};
    use crate::infrastructure::cpu::Cpu;

    struct TinyDecoder;

    impl DecoderModel<f32, Cpu> for TinyDecoder {
        fn dims(&self) -> ModelDims {
            ModelDims {
                dim: 1,
                q_dim: 1,
                kv_dim: 1,
                qkv_dim: 3,
                intermediate_size: 1,
                vocab_size: 4,
                head_num: 1,
                head_dim: 1,
                kv_head_num: 1,
                num_layers: 0,
                num_experts: 0,
                experts_per_tok: 0,
                moe_intermediate_size: 0,
                num_shared_experts: 0,
            }
        }

        fn stages(&self) -> &[StageKind] {
            &[]
        }

        fn embed(
            &self,
            input_ids: &Tensor<i32, Cpu>,
            hidden: &mut Hidden<f32, Cpu>,
            _ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<()> {
            let values = input_ids
                .to_host_vec()?
                .into_iter()
                .map(|id| id as f32)
                .collect::<Vec<_>>();
            hidden.stream.upload_from_host(&values)
        }

        fn decode_layers(
            &self,
            _range: LayerRange,
            _hidden: &mut Hidden<f32, Cpu>,
            _kv: &mut KvView<'_, f32, Cpu>,
            _ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<()> {
            Ok(())
        }

        fn finalize(
            &self,
            hidden: &Hidden<f32, Cpu>,
            _rows: SampleRows<'_>,
            _ctx: &StepCtx<'_, Cpu>,
        ) -> OpResult<Logits<f32, Cpu>> {
            let rows = hidden.num_tokens();
            let mut logits = vec![0.0f32; rows * 4];
            for row in 0..rows {
                logits[row * 4 + ((row + 1) % 4)] = 10.0;
            }
            Tensor::from_host_slice(
                &logits,
                Shape::from_slice(&[rows, 4]),
                hidden.stream.device(),
            )
            .map(Logits)
        }
    }

    fn tok(token_id: i32) -> SampledToken {
        SampledToken {
            token_id,
            logprob: 0.0,
            top_logprobs: Vec::new(),
        }
    }

    fn req_for_batch(batch: usize) -> StepRequest {
        StepRequest {
            seqs: (0..batch)
                .map(|i| SeqStep {
                    sequence_id: i as u64,
                    input_ids: vec![1],
                    positions: vec![0],
                    kv_write_start: 0,
                    kv_len_after: 1,
                    block_table: vec![0],
                })
                .collect(),
            sampling: vec![Default::default(); batch],
            stop: StopCriteria {
                eos_ids: vec![9],
                generated_counts: vec![0; batch],
                max_tokens: vec![16; batch],
                ignore_eos: vec![false; batch],
            },
            draft_tokens: Vec::new(),
        }
    }

    #[test]
    fn finished_flags_checks_all_emitted_tokens() {
        let req = req_for_batch(2);
        let finished = finished_flags(&req, &[vec![tok(1), tok(9)], vec![tok(2)]]);

        assert_eq!(finished, vec![true, false]);
    }

    #[test]
    fn finished_flags_honors_ignore_eos_and_max_tokens() {
        let mut req = req_for_batch(2);
        req.stop.ignore_eos[0] = true;
        req.stop.generated_counts[1] = 15;

        let finished = finished_flags(&req, &[vec![tok(9)], vec![tok(3)]]);

        assert_eq!(finished, vec![false, true]);
    }

    #[test]
    fn validate_step_request_vectors_allows_empty_defaults() {
        let mut req = req_for_batch(2);
        req.sampling.clear();
        req.stop.generated_counts.clear();
        req.stop.max_tokens.clear();
        req.stop.ignore_eos.clear();

        validate_step_request_vectors(&req, 2).unwrap();
    }

    #[test]
    fn validate_step_request_vectors_rejects_partial_vectors() {
        let mut req = req_for_batch(2);
        req.stop.max_tokens.pop();

        let err = validate_step_request_vectors(&req, 2).unwrap_err();

        assert!(format!("{err:?}").contains("max_tokens len 1 != batch 2"));
    }

    #[test]
    fn graph_runner_picks_smallest_decode_slot() {
        let runner = GraphRunner::<Cpu>::new(vec![4, 1, 2, 2], 4).unwrap();
        assert_eq!(runner.capture_sizes(), &[1, 2, 4]);

        let mut plan = BatchPlan {
            kind: BatchKind::DecodeOnly,
            num_tokens: 3,
            batch: 3,
            q_lens: vec![1, 1, 1],
            kv_lens: vec![1, 1, 1],
            seq_positions: vec![0, 0, 0],
            rope_positions: vec![0, 0, 0],
            max_blocks_per_seq: 4,
            block_size: 1,
            total_q_tiles: 3,
        };
        assert_eq!(runner.decide(&plan), GraphDecision::Graph(GraphSlotId(2)));

        plan.batch = 5;
        assert_eq!(runner.decide(&plan), GraphDecision::Eager);

        plan.batch = 1;
        plan.kind = BatchKind::Ragged;
        assert_eq!(runner.decide(&plan), GraphDecision::Eager);
    }

    #[test]
    fn runtime_prime_graphs_unsupported_scope_stays_eager() {
        let cpu = Cpu;
        let scope = HostScope::new(cpu);
        let mut runtime = Runtime::new(
            TinyDecoder,
            scope,
            Box::new(GreedySampler),
            4,
            1,
            4,
            4,
            2,
            2,
            vec![1, 2],
        )
        .unwrap();

        runtime.prime_graphs().unwrap();

        let req = req_for_batch(2);
        let plan = runtime.build_plan(&req).unwrap();
        assert!(runtime.graph.is_none());
        assert_eq!(runtime.decide(&plan), GraphDecision::Eager);
    }

    #[test]
    fn runtime_step_runs_tiny_decoder_and_commits_kv() {
        let cpu = Cpu;
        let scope = HostScope::new(cpu);
        let mut runtime = Runtime::new(
            TinyDecoder,
            scope,
            Box::new(GreedySampler),
            4,
            1,
            4,
            4,
            2,
            2,
            Vec::new(),
        )
        .unwrap();
        let req = StepRequest {
            seqs: vec![
                SeqStep {
                    sequence_id: 10,
                    input_ids: vec![7],
                    positions: vec![0],
                    kv_write_start: 0,
                    kv_len_after: 1,
                    block_table: vec![0],
                },
                SeqStep {
                    sequence_id: 11,
                    input_ids: vec![8],
                    positions: vec![0],
                    kv_write_start: 0,
                    kv_len_after: 1,
                    block_table: vec![1],
                },
            ],
            sampling: vec![Default::default(); 2],
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: vec![0, 0],
                max_tokens: vec![8, 8],
                ignore_eos: vec![false, false],
            },
            draft_tokens: Vec::new(),
        };

        let out = runtime.step(&req).unwrap();

        assert_eq!(out.accepted, vec![1, 1]);
        assert_eq!(out.tokens[0][0].token_id, 1);
        assert_eq!(out.tokens[1][0].token_id, 2);
        // Non-speculative steps leave the pool's per-seq length map empty
        // (the worker owns the length); only the speculative path populates it.
        assert!(runtime.kv_pool.seq_kv_len.is_empty());
    }
}
