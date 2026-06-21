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

#[cfg(feature = "cuda")]
use half::bf16;
#[cfg(feature = "cuda")]
use crate::infrastructure::cuda::{
    kernels::gather_merge::{
        append_decode_admissions_into, merge_compact_decode_into, MergeCompactDecodeArgs,
    },
    Cuda,
};

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
    /// Persistent decode input-id buffer (capacity `cap_batch`). Graph replay
    /// requires the forward to read from a fixed device address; this buffer is
    /// rewritten in place each step instead of allocating a fresh tensor.
    pub input_ids_buf: Tensor<i32, D>,
    /// ABC GPU-resident decode pipeline buffers. Buffer A is `input_ids_buf`.
    pub abc: AbcBuffers<D>,
}

/// Address-stable buffers for the ABC GPU-resident decode pipeline.
///
/// A = `Runtime::input_ids_buf` (next token per row). B = `new_token_dev`
/// (first tokens of freshly-admitted seqs). C = `argmax_out_dev` (in-graph
/// argmax output). The compact-merge kernel consumes C, applies stop
/// criteria, compacts surviving rows to the front of A, and emits the
/// active/finished side-bands below. All buffers are sized to `cap_batch`
/// so their device addresses never change (required for CUDA-graph capture).
///
/// Allocated in `Runtime::new`; wired into the decode step in later stages.
#[allow(dead_code)]
pub struct AbcBuffers<D: Device> {
    /// B: first decode tokens of newly-admitted sequences (uploaded each step).
    pub new_token_dev: Tensor<i32, D>,
    /// C: in-graph argmax output, one token id per row.
    pub argmax_out_dev: Tensor<i32, D>,
    /// Two-phase argmax scratch (`cap_batch * 512` f32 — kernel uses 512 bf16/row).
    pub argmax_ws: Tensor<f32, D>,
    /// Per-row stop metadata (uploaded each step).
    pub generated_counts_dev: Tensor<i32, D>,
    pub max_tokens_dev: Tensor<i32, D>,
    pub ignore_eos_dev: Tensor<i32, D>,
    /// EOS id list (small; `eos_len` passed alongside).
    pub eos_ids_dev: Tensor<i32, D>,
    /// Compact-merge outputs (device).
    pub active_src_rows_dev: Tensor<i32, D>,
    pub finished_src_rows_dev: Tensor<i32, D>,
    pub finished_tokens_dev: Tensor<i32, D>,
    pub active_tokens_dev: Tensor<i32, D>,
    /// `[active_n, finished_n, old_batch]`.
    pub counts_dev: Tensor<i32, D>,
    // Host mirrors for the single small D2H after each step.
    pub argmax_out_host: Vec<i32>,
    pub counts_host: Vec<i32>,
    pub active_src_rows_host: Vec<i32>,
    pub active_tokens_host: Vec<i32>,
    pub finished_src_rows_host: Vec<i32>,
    pub finished_tokens_host: Vec<i32>,
    /// Persistent host staging for buffer B (admission tokens). Must outlive the
    /// copy-in stream's DMA, so it cannot be a per-step local.
    pub new_token_host: Vec<i32>,
    /// Whether a copy-out has been recorded on So at least once. Gates the
    /// `compute_wait_copy_out` guard so the first step does not wait on an
    /// event that was never recorded.
    pub copy_out_recorded: bool,
    /// Whether the host map mirrors + B staging have been page-locked (pinned)
    /// yet. Done lazily on the first decode step so the Si/So copies are truly
    /// async (pageable host memory would make `cudaMemcpyAsync` host-synchronous
    /// and serialize the pipeline).
    pub pinned: bool,
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

        let input_ids_buf =
            D::alloc_tensor::<i32>(Shape::from_slice(&[cap_batch.max(1)]), device)?;

        // ── ABC pipeline buffers (Stage 1: allocated, wired in later stages) ──
        let cb = cap_batch.max(1);
        let abc = AbcBuffers {
            new_token_dev: alloc_i32(cb)?,
            argmax_out_dev: alloc_i32(cb)?,
            argmax_ws: D::alloc_tensor::<f32>(Shape::from_slice(&[cb * 512]), device)?,
            generated_counts_dev: alloc_i32(cb)?,
            max_tokens_dev: alloc_i32(cb)?,
            ignore_eos_dev: alloc_i32(cb)?,
            eos_ids_dev: alloc_i32(64)?,
            active_src_rows_dev: alloc_i32(cb)?,
            finished_src_rows_dev: alloc_i32(cb)?,
            finished_tokens_dev: alloc_i32(cb)?,
            active_tokens_dev: alloc_i32(cb)?,
            counts_dev: alloc_i32(3)?,
            argmax_out_host: vec![0; cb],
            counts_host: vec![0; 3],
            active_src_rows_host: vec![0; cb],
            active_tokens_host: vec![0; cb],
            finished_src_rows_host: vec![0; cb],
            finished_tokens_host: vec![0; cb],
            new_token_host: vec![0; cb],
            copy_out_recorded: false,
            pinned: false,
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
            input_ids_buf,
            abc,
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
        self.run_layers(plan, &input_ids)?;
        self.sample_tail(plan, req)
    }

    /// Embed + all decoder layers into the persistent `hidden` buffer. This is
    /// the capturable region of a decode step: every device buffer it touches
    /// (input ids, hidden, KV pool, KV index) is a fixed allocation, so the
    /// kernel sequence is replayable as a CUDA graph. Under graph capture the
    /// scratch tensors allocated inside the model come from the capture arena.
    fn run_layers(
        &mut self,
        plan: &crate::domain::plan::BatchPlan,
        input_ids: &Tensor<i32, D>,
    ) -> OpResult<()> {
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
        self.model.embed(input_ids, &mut hidden, &ctx)?;
        self.model.decode_layers(
            LayerRange::all(self.dims.num_layers),
            &mut hidden,
            &mut kv,
            &ctx,
        )?;
        Ok(())
    }

    /// Finalize (logits) + sample/verify + KV commit. Always eager: the capture
    /// arena is disabled here, so these allocations use the normal allocator and
    /// the (data-dependent, variable-shape) sampling never enters a graph.
    fn sample_tail(
        &mut self,
        plan: &crate::domain::plan::BatchPlan,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        let hidden = Hidden {
            stream: self.hidden.stream.view_raw(
                Shape::from_slice(&[plan.num_tokens, self.dims.dim]),
                Shape::from_slice(&[plan.num_tokens.max(1), self.dims.dim]).contiguous_strides(),
                0,
                true,
            ),
        };
        let ctx = crate::domain::exec::StepCtx::new(&self.scope, plan);
        let _guard = self.scope.enter();
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

    /// Capturable decode region for the graph path: forward (embed + decoder
    /// layers) + finalize (lm_head → logits) + in-graph argmax into buffer C
    /// (`abc.argmax_out_dev`). Every allocation here is served from the capture
    /// arena, so the whole chain replays from a single graph with the argmax
    /// result already on-device in C — no eager finalize/sample afterwards.
    fn forward_finalize_argmax(
        &mut self,
        plan: &crate::domain::plan::BatchPlan,
        input_ids: &Tensor<i32, D>,
    ) -> OpResult<()> {
        self.run_layers(plan, input_ids)?;
        let hidden = Hidden {
            stream: self.hidden.stream.view_raw(
                Shape::from_slice(&[plan.num_tokens, self.dims.dim]),
                Shape::from_slice(&[plan.num_tokens.max(1), self.dims.dim]).contiguous_strides(),
                0,
                true,
            ),
        };
        let ctx = crate::domain::exec::StepCtx::new(&self.scope, plan);
        let _guard = self.scope.enter();
        let logits = self.model.finalize(&hidden, SampleRows::All, &ctx)?;
        D::argmax_into(
            &ctx,
            &logits.0,
            &mut self.abc.argmax_out_dev,
            &self.abc.argmax_ws,
        )
    }

    /// Build the decode `StepOutput` from buffer C (the in-graph argmax result),
    /// instead of an eager finalize+sample. Greedy, one token per decode row;
    /// `logprob` is unused downstream. Mirrors `sample_tail`'s non-speculative
    /// path for finished-flag + KV-length bookkeeping.
    fn decode_output_from_c(
        &mut self,
        plan: &crate::domain::plan::BatchPlan,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        let batch = plan.batch;
        let c_view = self.abc.argmax_out_dev.view_raw(
            Shape::from_slice(&[batch]),
            Shape::from_slice(&[batch.max(1)]).contiguous_strides(),
            0,
            true,
        );
        let ids = c_view.to_host_vec()?;
        let sids: Vec<u64> = req.seqs.iter().map(|seq| seq.sequence_id).collect();
        let mut tokens: Vec<Vec<SampledToken>> = ids
            .iter()
            .map(|&token_id| {
                vec![SampledToken {
                    token_id,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                }]
            })
            .collect();
        tokens.resize_with(batch, Vec::new);
        let accepted: Vec<u32> = plan.q_lens.iter().map(|&q| q.max(0) as u32).collect();
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
        let Some(slot_batch) = graph.slot_size(slot) else {
            return Err(OpError::Shape(format!(
                "Runtime::step: graph slot {} is out of range",
                slot.0
            )));
        };
        // A captured graph hard-codes the batch it was traced with. Only replay
        // when the live decode batch matches the slot exactly (decode ⇒
        // num_tokens == batch); otherwise fall back to eager rather than feed a
        // mismatched batch through the wrong graph. (Padding to the capture size
        // is a future optimization.)
        if slot_batch != plan.batch || plan.num_tokens != plan.batch {
            return self.step_eager(plan, req);
        }
        let key = plan.batch as u64;

        // Refresh the persistent input-id buffer in place (decode: one new token
        // per sequence). The graph's `embed` reads from this fixed address.
        let mut ids = Vec::with_capacity(plan.num_tokens);
        for seq in &req.seqs {
            ids.extend_from_slice(&seq.input_ids);
        }
        unsafe {
            upload_i32_prefix(self.scope.device(), &self.input_ids_buf, &ids)?;
        }
        let input_ids = self.input_ids_buf.view_raw(
            Shape::from_slice(&[plan.num_tokens]),
            Shape::from_slice(&[plan.num_tokens]).contiguous_strides(),
            0,
            true,
        );

        if self.scope.graph_ready(key) {
            // Hot path: pure replay. `upload_index` (in `step`) and the id
            // refresh above already rewrote every input buffer this graph reads.
            self.scope.graph_launch(key)?;
        } else {
            // Cold path. First run one EAGER forward at this exact shape so the
            // libraries that lazily plan/benchmark on a cold shape (cuDNN SDPA
            // plan cache, cuBLASLt algo selection) populate their shape-keyed
            // caches — those code paths do mallocs/private-stream launches that
            // are illegal under stream capture. This eager pass also produces a
            // correct result; the KV scatter it performs writes the same values
            // at the same paged positions the replay will, so the immediately
            // following capture+launch is idempotent on KV state.
            self.forward_finalize_argmax(plan, &input_ids)?;
            self.scope.synchronize()?;

            // Now trace the (warm) forward+finalize+argmax into a graph and run once.
            self.scope.graph_capture_begin()?;
            if let Err(e) = self.forward_finalize_argmax(plan, &input_ids) {
                // Close the capture so the stream is left in a usable state.
                let _ = self.scope.graph_capture_end(key);
                return Err(e);
            }
            self.scope.graph_capture_end(key)?;
            tracing::info!("[graph] captured decode graph (forward+argmax) for batch={}", plan.batch);
            self.scope.graph_launch(key)?;
        }
        // The graph already produced the per-row argmax in buffer C; just read
        // it back (no eager finalize/sample). The decode graph path is always
        // greedy q_len=1 (speculative q_len>1 fell back to eager above).
        self.scope.synchronize()?;
        self.decode_output_from_c(plan, req)
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

/// One row of an ABC compact-merge result: the original (pre-compaction)
/// device row it came from, plus the token the step produced for it.
#[derive(Debug, Clone)]
pub struct DecodeRowToken {
    pub src_row: usize,
    pub token_id: i32,
}

/// Outcome of one ABC decode step, with surviving and finished rows kept
/// SEPARATE (the merge marks them apart on-device). The caller advances the
/// `active` rows and reclaims each `finished` row's KV — including the slot
/// allocated for it this step — on return.
#[derive(Debug, Clone)]
pub struct DecodeCompactOutput {
    pub active: Vec<DecodeRowToken>,
    pub finished: Vec<DecodeRowToken>,
}

#[cfg(feature = "cuda")]
impl<M> Runtime<bf16, Cuda, M>
where
    M: DecoderModel<bf16, Cuda>,
{
    /// ABC GPU-resident decode step (buffer A = `input_ids_buf`).
    ///
    /// Precondition: rows `0..a_valid_prefix` of A already hold the correct
    /// input token — they are the longest prefix of this step's row order that
    /// is unchanged from the prior step, so the previous step's compact merge
    /// already wrote their tokens on-device. The divergent suffix (fresh
    /// admissions, or rows shifted by an out-of-band eviction) is uploaded
    /// through buffer B and appended into A.
    ///
    /// Make the compute stream wait on the in-flight decode step's copy-out
    /// (ev_out) before any further compute-stream write to buffer A. The serve
    /// loop calls this before running a prefill while a decode step is pending:
    /// a graph-eligible (all q=1) prefill uploads into `input_ids_buf` on the
    /// compute stream and would otherwise race the pending step's async
    /// copy-out read of A. Enqueue-only — no host sync.
    pub fn guard_buffer_a_against_pending_copyout(&self) -> OpResult<()> {
        if self.abc.copy_out_recorded {
            let cfg = self.scope.device().config.clone();
            cfg.compute_wait_copy_out()?;
        }
        Ok(())
    }

    /// ISSUE half of the 1-deep decode pipeline: enqueues forward + finalize +
    /// in-graph argmax (graph replay when primed) into buffer C, then the
    /// compact-merge kernel (stop criteria on-device, compacts survivors to the
    /// front of A so the next step reuses them without a host upload, emits
    /// active/finished source-row maps), then an ASYNC copy-out (So) of the
    /// maps into the host mirrors. Does NOT synchronize — the caller runs
    /// `finalize_decode_abc` one step later so the NEXT step's compute overlaps
    /// this step's host commit/send. Only one step may be in flight at a time
    /// (the host map mirrors and buffer C are single-buffered).
    #[allow(clippy::too_many_arguments)]
    pub fn issue_decode_abc(
        &mut self,
        req: &StepRequest,
        a_valid_prefix: usize,
        generated_counts: &[u32],
        max_tokens: &[u32],
        ignore_eos: &[bool],
        eos_ids: &[i32],
    ) -> OpResult<()> {
        let plan = self.build_plan(req)?;
        let batch = plan.batch;
        if plan.num_tokens != batch || plan.q_lens.iter().any(|&q| q != 1) {
            return Err(OpError::Shape(
                "step_decode_abc: requires pure decode (q_len=1 per row)".into(),
            ));
        }
        if generated_counts.len() != batch
            || max_tokens.len() != batch
            || ignore_eos.len() != batch
        {
            return Err(OpError::Shape(format!(
                "step_decode_abc: metadata lens gen={} max={} ignore={} batch={}",
                generated_counts.len(),
                max_tokens.len(),
                ignore_eos.len(),
                batch
            )));
        }
        if eos_ids.len() > self.abc.eos_ids_dev.numel() {
            return Err(OpError::Shape(format!(
                "step_decode_abc: eos_ids {} > capacity {}",
                eos_ids.len(),
                self.abc.eos_ids_dev.numel()
            )));
        }

        // Lazily page-lock the host staging on the first step so the Si/So
        // copies are truly async (pageable memory makes them host-synchronous).
        if !self.abc.pinned {
            let cfg = self.scope.device().config.clone();
            let _guard = self.scope.enter();
            cfg.pin_host_i32(&self.abc.new_token_host)?;
            cfg.pin_host_i32(&self.abc.counts_host)?;
            cfg.pin_host_i32(&self.abc.active_src_rows_host)?;
            cfg.pin_host_i32(&self.abc.active_tokens_host)?;
            cfg.pin_host_i32(&self.abc.finished_src_rows_host)?;
            cfg.pin_host_i32(&self.abc.finished_tokens_host)?;
            self.abc.pinned = true;
        }

        self.upload_index(&plan, req)?;

        let cfg = self.scope.device().config.clone();

        // Guard A against this step's append/merge overwriting it before the
        // prior step's copy-out (So) finished reading A_{n-1}. (ev_out)
        if self.abc.copy_out_recorded {
            cfg.compute_wait_copy_out()?;
        }

        // ── Refresh A: only the divergent suffix (rows >= a_valid_prefix). ──
        // Rows 0..vp already hold the right token from the prior step's merge.
        // B is uploaded on the copy-in stream (Si) so it can overlap compute;
        // the append (compute) waits on ev_in before reading B.
        let vp = a_valid_prefix.min(batch);
        if vp < batch {
            let n = batch - vp;
            // WAR guard: the prior step's copy-in DMA may still be reading
            // `new_token_host`. Drain Si before the CPU overwrites it. Cheap
            // (Si is a tiny copy, usually long idle by now) and only on
            // admission steps. Mandatory for correctness once this staging
            // buffer is page-locked (pinned), where the H2D is truly async.
            cfg.synchronize_copy_in()?;
            for (dst, seq) in self.abc.new_token_host[..n]
                .iter_mut()
                .zip(req.seqs[vp..].iter())
            {
                *dst = seq.input_ids[0];
            }
            let _guard = self.scope.enter();
            unsafe {
                cfg.upload_h2d_copy_in(
                    self.abc.new_token_dev.data_ptr_mut() as *mut std::ffi::c_void,
                    self.abc.new_token_host.as_ptr() as *const std::ffi::c_void,
                    n * std::mem::size_of::<i32>(),
                )?;
            }
            cfg.record_copy_in()?; // ev_in on Si
            cfg.compute_wait_copy_in()?; // compute waits ev_in before reading B
            let stream = ExecScope::stream(&self.scope).0;
            append_decode_admissions_into(
                &mut self.input_ids_buf,
                &self.abc.new_token_dev,
                vp,
                n,
                stream,
            )?;
        }

        // ── forward + finalize + in-graph argmax → buffer C ──
        let input_ids = self.input_ids_buf.view_raw(
            Shape::from_slice(&[batch]),
            Shape::from_slice(&[batch]).contiguous_strides(),
            0,
            true,
        );
        let graph_key = batch as u64;
        let use_graph = match self.decide(&plan) {
            GraphDecision::Graph(slot) => {
                self.graph.as_ref().and_then(|g| g.slot_size(slot)) == Some(batch)
            }
            GraphDecision::Eager => false,
        };
        if use_graph {
            if self.scope.graph_ready(graph_key) {
                self.scope.graph_launch(graph_key)?;
            } else {
                // Cold path: warm the lazy library caches with one eager pass,
                // then capture + replay (mirrors `step_graph`).
                self.forward_finalize_argmax(&plan, &input_ids)?;
                self.scope.synchronize()?;
                self.scope.graph_capture_begin()?;
                if let Err(e) = self.forward_finalize_argmax(&plan, &input_ids) {
                    let _ = self.scope.graph_capture_end(graph_key);
                    return Err(e);
                }
                self.scope.graph_capture_end(graph_key)?;
                tracing::info!(
                    "[graph] captured decode graph (forward+argmax) for batch={}",
                    batch
                );
                self.scope.graph_launch(graph_key)?;
            }
        } else {
            self.forward_finalize_argmax(&plan, &input_ids)?;
        }

        // ── upload stop metadata + run the compact merge (C → A) ──
        let gen_i32: Vec<i32> = generated_counts.iter().map(|&x| x as i32).collect();
        let max_i32: Vec<i32> = max_tokens.iter().map(|&x| x as i32).collect();
        let ign_i32: Vec<i32> = ignore_eos.iter().map(|&b| i32::from(b)).collect();
        let device = self.scope.device();
        unsafe {
            upload_i32_prefix(device, &self.abc.generated_counts_dev, &gen_i32)?;
            upload_i32_prefix(device, &self.abc.max_tokens_dev, &max_i32)?;
            upload_i32_prefix(device, &self.abc.ignore_eos_dev, &ign_i32)?;
            if !eos_ids.is_empty() {
                upload_i32_prefix(device, &self.abc.eos_ids_dev, eos_ids)?;
            }
        }
        let stream = ExecScope::stream(&self.scope).0;
        {
            let _guard = self.scope.enter();
            let mut a = self.input_ids_buf.view_raw(
                Shape::from_slice(&[batch]),
                Shape::from_slice(&[batch]).contiguous_strides(),
                0,
                true,
            );
            merge_compact_decode_into(MergeCompactDecodeArgs {
                a_out: &mut a,
                c_prev: &self.abc.argmax_out_dev,
                generated_counts: &self.abc.generated_counts_dev,
                max_tokens: &self.abc.max_tokens_dev,
                ignore_eos: &self.abc.ignore_eos_dev,
                eos_ids: &self.abc.eos_ids_dev,
                eos_len: eos_ids.len(),
                old_batch: batch,
                active_src_rows: &self.abc.active_src_rows_dev,
                finished_src_rows: &self.abc.finished_src_rows_dev,
                finished_tokens: &self.abc.finished_tokens_dev,
                counts: &self.abc.counts_dev,
                stream,
            })?;
        }
        // A now holds the committed/compacted tokens (compute). Mark it so the
        // copy-out stream may begin downloading once compute reaches here. (ev_a)
        cfg.record_compute_a()?;

        // ── download the compaction maps on the copy-out stream (So) ──
        // So waits ev_a, then the D2H runs (and may overlap the next step's
        // forward). Fixed `batch`-sized chunks avoid a dependency on the (still
        // on-device) counts. ev_out gates the next step's A overwrite.
        cfg.copy_out_wait_compute_a()?;
        let bytes = batch * std::mem::size_of::<i32>();
        let elem = std::mem::size_of::<i32>();
        unsafe {
            cfg.download_d2h_copy_out(
                self.abc.counts_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.counts_dev.data_ptr() as *const std::ffi::c_void,
                3 * elem,
            )?;
            cfg.download_d2h_copy_out(
                self.abc.active_src_rows_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.active_src_rows_dev.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
            // Active tokens live in A[0..active] after the compaction.
            cfg.download_d2h_copy_out(
                self.abc.active_tokens_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.input_ids_buf.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
            cfg.download_d2h_copy_out(
                self.abc.finished_src_rows_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.finished_src_rows_dev.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
            cfg.download_d2h_copy_out(
                self.abc.finished_tokens_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.finished_tokens_dev.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
        }
        cfg.record_copy_out()?; // ev_out on So
        self.abc.copy_out_recorded = true;
        // NOTE: no synchronize here — the So download runs asynchronously and is
        // collected by `finalize_decode_abc`, which the caller invokes one step
        // later so the next step's compute overlaps this step's host commit.
        Ok(())
    }

    /// Collect the result of the in-flight `issue_decode_abc` step: drain the
    /// copy-out stream, read the now-valid host map mirrors, and surface the
    /// surviving vs finished rows SEPARATELY (the merge split them on-device:
    /// compacted survivors in A[0..active] + active_src_rows; finished rows in
    /// finished_src_rows / finished_tokens). The caller advances the actives
    /// and reclaims each finished row's previous-step KV. A coverage check
    /// (each row returned exactly once, all covered) catches merge-kernel
    /// desync before it can corrupt host row bookkeeping. `batch` must be the
    /// `order.len()` the matching `issue_decode_abc` ran with.
    pub fn finalize_decode_abc(&mut self, batch: usize) -> OpResult<DecodeCompactOutput> {
        let cfg = self.scope.device().config.clone();
        cfg.synchronize_copy_out()?; // host mirrors valid before we read them

        let active_n = self.abc.counts_host[0].max(0) as usize;
        let finished_n = self.abc.counts_host[1].max(0) as usize;
        let old_n = self.abc.counts_host[2].max(0) as usize;
        if old_n != batch || active_n + finished_n != batch {
            return Err(OpError::Kernel(format!(
                "step_decode_abc: compact counts invalid active={} finished={} old={} batch={}",
                active_n, finished_n, old_n, batch
            )));
        }
        let mut seen = vec![false; batch];
        let mut mark = |src: i32| -> OpResult<usize> {
            let row = src as usize;
            if row >= batch {
                return Err(OpError::Kernel(format!(
                    "step_decode_abc: src_row {} >= batch {}",
                    row, batch
                )));
            }
            if seen[row] {
                return Err(OpError::Kernel(format!(
                    "step_decode_abc: src_row {} returned twice",
                    row
                )));
            }
            seen[row] = true;
            Ok(row)
        };
        let mut active = Vec::with_capacity(active_n);
        for k in 0..active_n {
            let row = mark(self.abc.active_src_rows_host[k])?;
            active.push(DecodeRowToken {
                src_row: row,
                token_id: self.abc.active_tokens_host[k],
            });
        }
        let mut finished = Vec::with_capacity(finished_n);
        for j in 0..finished_n {
            let row = mark(self.abc.finished_src_rows_host[j])?;
            finished.push(DecodeRowToken {
                src_row: row,
                token_id: self.abc.finished_tokens_host[j],
            });
        }
        if seen.iter().any(|covered| !*covered) {
            return Err(OpError::Kernel(
                "step_decode_abc: compaction did not cover every row".into(),
            ));
        }

        Ok(DecodeCompactOutput { active, finished })
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
