//! Runtime — the generic step engine over `LlmBackend`.
//!
//! Split by concern (each submodule adds `impl Runtime` blocks):
//! - [`plan`]       — `StepRequest` validation → `BatchPlan` + control-index upload
//! - [`graph_exec`] — CUDA-graph capture/replay lifecycles + `GraphRunner` policy
//! - [`abc_decode`] — ABC pipelined pure-decode issue/finalize halves (CUDA)
//! - [`mixed_abc`]  — fused mixed-batch (decode prefix + prefill) ABC path (CUDA)
//!
//! This file keeps the `Runtime` struct and its address-stable buffers,
//! construction, the eager step path, and the shared free helpers.

use std::ptr::NonNull;

use crate::domain::component::{Hidden, LayerRange};
use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecDevice as Device, ExecScope};
use crate::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
use crate::domain::model::{DecoderModel, ModelDims, SampleRows};
use crate::domain::plan::{
    BatchPlan, SampledToken, SeqStep, StepOutput, StepRequest, StopCriteria,
};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::sampler::Sampler;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

mod abc_decode;
mod graph_exec;
mod mixed_abc;
mod plan;

pub use graph_exec::{GraphDecision, GraphRunner, GraphSlotId};
pub use mixed_abc::MixedStepTicket;

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
    /// Persistent prefill input-id staging (capacity `cap_num_tokens`) + its
    /// host mirror. Eager forwards upload into this fixed buffer with an async
    /// H2D (no per-step `cudaStreamSynchronize`), instead of allocating a fresh
    /// `from_host_slice` tensor whose throwaway host Vec forced a full sync.
    pub prefill_ids_buf: Tensor<i32, D>,
    prefill_ids_host: Vec<i32>,
    /// Persistent host staging for the paged block tables (capacity
    /// `cap_batch * max_blocks_per_seq`). Rewritten in place every step and
    /// async-uploaded into `kv_index.block_tables`. Lives for the runtime's
    /// lifetime so the `upload_async` (cudaMemcpyAsync) contract holds — the
    /// previous code allocated+zeroed a fresh `Vec` per step and uploaded from
    /// it, which (a) churned 1–4 MiB/step in the decode hot loop, (b) was
    /// pageable so the "async" copy silently degraded to a synchronous one, and
    /// (c) freed the host buffer before the stream consumed it. Pinned lazily
    /// alongside the ABC buffers so the copy is genuinely async on the decode
    /// path.
    block_tables_host: Vec<i32>,
    /// ABC GPU-resident decode pipeline buffers. Buffer A is `input_ids_buf`.
    pub abc: AbcBuffers<D>,
    /// Async-decode device control plane scratch (built lazily on first async
    /// step). `compact_extend_control` gathers the compacted block tables /
    /// kv_lens into these (an in-place gather would race), and the caller copies
    /// them back into `kv_index`. `new_slots_dev` holds the next step's KV slot
    /// per output row, uploaded O(batch) each async step (vs the O(batch*seqlen)
    /// host block-table rebuild the async path eliminates).
    async_ctrl: Option<AsyncControlBuffers<D>>,
    /// Count of distinct single-seq prefill graphs captured so far (Stage A
    /// keys each prefill graph by exact `num_tokens`). Bounds graph memory:
    /// once it reaches `PREFILL_GRAPH_BUDGET`, further uncaptured prefill
    /// lengths run eager instead of capturing a new graph.
    prefill_graphs_captured: usize,
    /// Exact-shape mixed ABC graphs captured so far. Bounded because mixed
    /// q_len layouts can be much more diverse than decode batch sizes.
    mixed_graphs_captured: usize,
    /// Mixed graph capture is a bootstrap-only activity. Serving may replay a
    /// ready bucket, but an unseen bucket falls back to eager instead of doing
    /// eager warmup + synchronize + stream capture on the request path.
    mixed_graph_capture_enabled: bool,
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
    /// Mixed/ragged row kind, one value per logical row.
    pub row_kind_dev: Tensor<i32, D>,
    /// Mixed graph: last token row in the flat token tape for every logical row.
    pub last_token_rows_dev: Tensor<i32, D>,
    /// Compact-merge outputs (device).
    pub active_src_rows_dev: Tensor<i32, D>,
    pub finished_src_rows_dev: Tensor<i32, D>,
    pub finished_tokens_dev: Tensor<i32, D>,
    pub active_tokens_dev: Tensor<i32, D>,
    pub prefill_final_src_rows_dev: Tensor<i32, D>,
    pub prefill_final_tokens_dev: Tensor<i32, D>,
    /// Decode: `[active_n, finished_n, old_batch]`.
    /// Mixed: `[active_n, finished_n, prefill_final_n, old_rows]`.
    pub counts_dev: Tensor<i32, D>,
    // Host mirrors for the single small D2H after each step.
    pub argmax_out_host: Vec<i32>,
    pub counts_host: Vec<i32>,
    pub active_src_rows_host: Vec<i32>,
    pub active_tokens_host: Vec<i32>,
    pub finished_src_rows_host: Vec<i32>,
    pub finished_tokens_host: Vec<i32>,
    pub prefill_final_src_rows_host: Vec<i32>,
    pub prefill_final_tokens_host: Vec<i32>,
    pub row_kind_host: Vec<i32>,
    pub last_token_rows_host: Vec<i32>,
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

/// Scratch buffers for the device-resident decode control plane. Allocated
/// lazily on the first decode step that builds it (they cost
/// `cap_batch * (max_blocks_per_seq + 1)` i32s).
struct AsyncControlBuffers<D: Device> {
    /// Gather target for the compacted block tables ([cap_batch * mbps]).
    block_tables_scratch: Tensor<i32, D>,
    /// Gather target for the compacted kv_lens ([cap_batch]).
    kv_lens_scratch: Tensor<i32, D>,
    /// Next-step KV slot per output row, device side ([cap_batch]). Uploaded
    /// O(batch) each step from a small pageable Vec (the copy is host-sync for
    /// pageable memory, so no pinned staging is needed for this tiny transfer).
    new_slots_dev: Tensor<i32, D>,
}

/// RAII guard that restores the default (decode) GEMM mode when an eager prefill
/// forward returns — including early `?` returns. The `bool` records whether the
/// mode was actually flipped on, so the no-prefill case is a cheap no-op.
pub(super) struct PrefillGemmGuard<D: LlmBackend>(pub(super) bool, pub(super) std::marker::PhantomData<D>);
impl<D: LlmBackend> Drop for PrefillGemmGuard<D> {
    fn drop(&mut self) {
        if self.0 {
            D::set_prefill_gemm_mode(false);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaggedRowKind {
    Decode,
    PrefillFinal,
    PrefillCont,
    Pad,
}

impl RaggedRowKind {
    pub fn as_i32(self) -> i32 {
        match self {
            Self::Decode => 0,
            Self::PrefillFinal => 1,
            Self::PrefillCont => 2,
            Self::Pad => 3,
        }
    }

    pub(super) fn emits_token(self) -> bool {
        matches!(self, Self::Decode | Self::PrefillFinal)
    }
}

impl<T, D, M> Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut model: M,
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

        // Worst-case ragged tiles for ANY valid batch: every req contributes ≥1
        // tile (≤ cap_batch) PLUS the extra tiles a long prefill adds
        // (≤ ⌈cap_num_tokens / TILE⌉). A mixed batch (many q=1 decode rows + a
        // long prefill chunk) hits both at once, so the cap is their SUM — not
        // their max (the old `.max()` under-sized block2req/block2tile for mixed
        // batches and for the bucketed mixed graph's padded `cap_batch + ⌈B/TILE⌉`
        // tile grid).
        let tile = crate::domain::plan::RAGGED_Q_TILE as usize;
        let cap_total_q_tiles = (cap_batch + (cap_num_tokens + tile - 1) / tile).max(1);
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
            valid_q_tiles: alloc_i32(1)?,
            valid_suffix_q_tiles: alloc_i32(1)?,
        };

        let hidden = Hidden {
            stream: D::alloc_tensor(Shape::from_slice(&[cap_num_tokens, dims.dim]), device)?,
            pending: None,
        };

        let input_ids_buf = D::alloc_tensor::<i32>(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let prefill_ids_buf =
            D::alloc_tensor::<i32>(Shape::from_slice(&[cap_num_tokens.max(1)]), device)?;
        let prefill_ids_host = vec![0i32; cap_num_tokens.max(1)];
        let block_tables_host = vec![0i32; (cap_batch * max_blocks_per_seq).max(1)];

        // ── ABC pipeline buffers (Stage 1: allocated, wired in later stages) ──
        let cb = cap_batch.max(1);
        // Greedy prefill argmax processes every token row of the forward, so the
        // argmax output/scratch must hold up to `cap_num_tokens` rows, not just
        // the decode batch. Decode only ever uses the first `batch` slots.
        let argmax_cap = cb.max(cap_num_tokens.max(1));
        let abc = AbcBuffers {
            new_token_dev: alloc_i32(cb)?,
            argmax_out_dev: alloc_i32(argmax_cap)?,
            argmax_ws: D::alloc_tensor::<f32>(Shape::from_slice(&[argmax_cap * 512]), device)?,
            generated_counts_dev: alloc_i32(cb)?,
            max_tokens_dev: alloc_i32(cb)?,
            ignore_eos_dev: alloc_i32(cb)?,
            eos_ids_dev: alloc_i32(64)?,
            row_kind_dev: alloc_i32(cb)?,
            last_token_rows_dev: alloc_i32(cb)?,
            active_src_rows_dev: alloc_i32(cb)?,
            finished_src_rows_dev: alloc_i32(cb)?,
            finished_tokens_dev: alloc_i32(cb)?,
            active_tokens_dev: alloc_i32(cb)?,
            prefill_final_src_rows_dev: alloc_i32(cb)?,
            prefill_final_tokens_dev: alloc_i32(cb)?,
            counts_dev: alloc_i32(5)?,
            argmax_out_host: vec![0; cb],
            counts_host: vec![0; 5],
            active_src_rows_host: vec![0; cb],
            active_tokens_host: vec![0; cb],
            finished_src_rows_host: vec![0; cb],
            finished_tokens_host: vec![0; cb],
            prefill_final_src_rows_host: vec![0; cb],
            prefill_final_tokens_host: vec![0; cb],
            row_kind_host: vec![0; cb],
            last_token_rows_host: vec![0; cb],
            new_token_host: vec![0; cb],
            copy_out_recorded: false,
            pinned: false,
        };

        // Preallocate the address-stable per-layer forward scratch and install
        // it into the model's sublayers. Eliminates the ~11 device allocations
        // per layer (cudaMalloc/cudaFree/memset storm) on the eager forward and
        // bakes fixed scratch addresses into captured decode graphs.
        let scratch = crate::domain::forward_scratch::ForwardScratch::<T, D>::new(
            device,
            dims,
            cap_num_tokens,
            cb,
        )?;
        model.install_scratch(scratch);

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
            async_ctrl: None,
            input_ids_buf,
            prefill_ids_buf,
            prefill_ids_host,
            block_tables_host,
            abc,
            prefill_graphs_captured: 0,
            mixed_graphs_captured: 0,
            mixed_graph_capture_enabled: false,
        })
    }

    pub fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput> {
        let plan = self.build_plan(req)?;
        self.upload_index(&plan, req)?;
        match self.decide(&plan) {
            GraphDecision::Eager => self.step_eager(&plan, req),
            GraphDecision::Graph(slot) => self.step_graph(slot, &plan, req),
            GraphDecision::PrefillGraph(num_tokens) => {
                self.step_prefill_graph(num_tokens, &plan, req)
            }
        }
    }

    pub fn step_eager(
        &mut self,
        plan: &crate::domain::plan::BatchPlan,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        // Eager prefill (num_tokens > batch) routes bf16 GEMMs to the build-free
        // chunked path so each distinct prompt length skips the cuBLASLt cache
        // build (~9-18ms off TTFT). A guard restores the default on any return so
        // the decode graph cold-path warmup still builds the cuBLASLt cache.
        let prefill_gemm = plan.num_tokens > plan.batch;
        if prefill_gemm {
            D::set_prefill_gemm_mode(true);
        }
        let _gemm_guard = PrefillGemmGuard::<D>(prefill_gemm, std::marker::PhantomData);
        let input_ids = self.input_ids_tensor(req, plan)?;
        let _trace = std::env::var_os("RUSTINFER_TTFT_TRACE").is_some();
        let _t0 = std::time::Instant::now();
        self.run_layers(plan, &input_ids)?;
        if _trace {
            let _ = self.scope.synchronize();
            tracing::info!(
                "[ttft-trace] run_layers (36L fwd, synced) = {:.2}ms",
                _t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let _t1 = std::time::Instant::now();
        let out = self.sample_tail(plan, req);
        if _trace {
            tracing::info!(
                "[ttft-trace] sample_tail (finalize+argmax+D2H) = {:.2}ms",
                _t1.elapsed().as_secs_f64() * 1e3
            );
        }
        out
    }

    /// Embed + all decoder layers into the persistent `hidden` buffer. This is
    /// the capturable region of a decode step: every device buffer it touches
    /// (input ids, hidden, KV pool, KV index) is a fixed allocation, so the
    /// kernel sequence is replayable as a CUDA graph. Under graph capture the
    /// scratch tensors allocated inside the model come from the capture arena.
    pub(super) fn run_layers(
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
            pending: None,
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
    pub(super) fn sample_tail(
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
            pending: None,
        };
        let ctx = crate::domain::exec::StepCtx::new(&self.scope, plan);
        let _guard = self.scope.enter();
        // Greedy first-token sampling only needs the last row of each sequence,
        // so project just those (`LastPerSeq`) — this keeps the lm_head GEMM at
        // M=batch (a warm decode shape) and off the cold per-prompt-length path
        // that paid a ~37ms cuBLASLt heuristic on prefill. Speculative verify
        // scores every draft position, so it still needs all rows.
        let sample_rows = if req.draft_tokens.is_empty() {
            SampleRows::LastPerSeq
        } else {
            SampleRows::All
        };
        let logits = self.model.finalize(&hidden, sample_rows, &ctx)?;
        let sids: Vec<u64> = req.seqs.iter().map(|seq| seq.sequence_id).collect();
        let (tokens, accepted, speculative_len) = if req.draft_tokens.is_empty() {
            // Greedy fast path: reuse the ABC argmax workspace instead of having
            // `Sampler::sample` allocate a fresh `[rows]` output + `[rows*512]`
            // scratch every call. The CUDA backend's `argmax` (used by
            // `GreedySampler::sample`) otherwise pays two `Tensor::zeros`
            // (cudaMallocAsync + memset) per prefill, which directly inflated
            // TTFT at low QPS after the buffer-pipeline refactor.
            //
            // `finalize(LastPerSeq)` already projected exactly one logits row per
            // sequence (the last token of each), in `q_lens` order. So
            // `argmax_into` yields one id per sequence and seq `i` maps directly
            // to row `i` — no per-sequence `offset + q_len - 1` indexing.
            let logits_rows = logits.0.shape().as_slice()[0];
            if logits_rows > self.abc.argmax_out_dev.numel() {
                return Err(OpError::Shape(format!(
                    "sample_tail: logits rows {} exceeds argmax_out capacity {}",
                    logits_rows,
                    self.abc.argmax_out_dev.numel()
                )));
            }
            // Pass a `[logits_rows]`-sized view of the (capacity `argmax_cap`)
            // output buffer: `argmax_into` writes exactly one id per logits row,
            // and the reference path requires the output numel to match.
            let mut argmax_out = self.abc.argmax_out_dev.view_raw(
                Shape::from_slice(&[logits_rows]),
                Shape::from_slice(&[logits_rows.max(1)]).contiguous_strides(),
                0,
                true,
            );
            D::argmax_into(&ctx, &logits.0, &mut argmax_out, &self.abc.argmax_ws, None)?;
            let ids = argmax_out.to_host_vec()?;
            let mut tokens: Vec<Vec<SampledToken>> = Vec::with_capacity(plan.batch);
            for i in 0..plan.q_lens.len() {
                let token_id = *ids.get(i).ok_or_else(|| {
                    OpError::Shape(format!(
                        "sample_tail: sampled row {} out of argmax range {}",
                        i,
                        ids.len()
                    ))
                })?;
                tokens.push(vec![SampledToken {
                    token_id,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                }]);
            }
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
    pub(super) fn forward_finalize_argmax(
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
            pending: None,
        };
        let ctx = crate::domain::exec::StepCtx::new(&self.scope, plan);
        let _guard = self.scope.enter();
        let logits = self.model.finalize(&hidden, SampleRows::All, &ctx)?;
        // Pass a `[logits_rows]` view of the capacity-sized C buffer: the host
        // reference `argmax_into` requires the output numel to match the row
        // count (the CUDA kernel just writes the first rows either way).
        let logits_rows = logits.0.shape().as_slice()[0];
        if logits_rows > self.abc.argmax_out_dev.numel() {
            return Err(OpError::Shape(format!(
                "forward_finalize_argmax: logits rows {} exceed C capacity {}",
                logits_rows,
                self.abc.argmax_out_dev.numel()
            )));
        }
        let mut c_view = self.abc.argmax_out_dev.view_raw(
            Shape::from_slice(&[logits_rows]),
            Shape::from_slice(&[logits_rows.max(1)]).contiguous_strides(),
            0,
            true,
        );
        D::argmax_into(&ctx, &logits.0, &mut c_view, &self.abc.argmax_ws, None)
    }

    /// Build the decode `StepOutput` from buffer C (the in-graph argmax result),
    /// instead of an eager finalize+sample. Greedy, one token per decode row;
    /// `logprob` is unused downstream. Mirrors `sample_tail`'s non-speculative
    /// path for finished-flag + KV-length bookkeeping.
    pub(super) fn decode_output_from_c(
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

    /// Run one worst-case synthetic forward so the bootstrap memory probe (in
    /// `serve_loop`) measures the true resident footprint before the real KV
    /// pool is sized. The eager forward commits the fixed activation workspace
    /// (already allocated in `Runtime::new`) *and* forces the lazy device
    /// allocations the first live forward would otherwise make — cuBLASLt algo
    /// workspaces, cuDNN SDPA plans, the recycling pool's scratch. After this
    /// returns and the caller `synchronize()`s + probes, `cudaMemGetInfo`'s
    /// `free` reflects everything except the KV pool, so the pool can be sized
    /// from what is genuinely left instead of a fraction-of-free guess taken
    /// before the workspace existed.
    ///
    /// Must be called with `self.graph == None` (i.e. before `prime_graphs`):
    /// `decide()` then routes every `step()` through the eager path, so this
    /// captures no CUDA graph and bakes no KV-base pointer that the subsequent
    /// `resize_kv_pool` would invalidate.
    ///
    /// The synthetic request is a single ragged prefill of `profile_len` tokens
    /// (the largest that both the activation workspace `cap_num_tokens` and the
    /// throwaway profiling KV pool can hold), mirroring `prewarm_prefill_shapes`.
    /// Blocks `0..profile_len` are scratch — this runs before any admission.
    pub fn profile_forward(&mut self) -> OpResult<()> {
        debug_assert!(
            self.graph.is_none(),
            "profile_forward must run before prime_graphs (eager only)"
        );
        // Largest prefill the fixed workspace and the profiling pool both hold.
        // `num_blocks - 1` leaves the graph-scratch block untouched, matching
        // the `pool_blocks = num_blocks + 1` convention the real pool uses.
        let profile_len = self
            .cap_num_tokens
            .min(self.max_seq_len)
            .min(self.max_blocks_per_seq)
            .min(self.kv_pool.num_blocks.saturating_sub(1))
            .max(1);
        let req = StepRequest {
            seqs: vec![SeqStep {
                sequence_id: 0,
                input_ids: vec![1; profile_len],
                positions: (0..profile_len as i32).collect(),
                kv_write_start: 0,
                kv_len_after: profile_len as i32,
                block_table: (0..profile_len as u32).collect(),
            }],
            sampling: vec![Default::default(); 1],
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: vec![0; 1],
                max_tokens: vec![u32::MAX; 1],
                ignore_eos: vec![true; 1],
            },
            draft_tokens: Vec::new(),
        };
        self.step(&req)?;
        self.scope.synchronize()?;
        Ok(())
    }

    /// Replace the KV pool's per-layer `k`/`v` tensors with a fresh set sized for
    /// `num_blocks`, freeing the previous (profiling) pool first. Called once at
    /// bootstrap between `profile_forward` and `prime_graphs`, so no captured
    /// graph yet references the old KV base and `seq_kv_len` is empty (no live
    /// sequence state to migrate). Dropping the old `layers` before allocating
    /// the new ones returns the profiling pool's bytes to the device so the
    /// (typically larger) real pool can reuse them.
    pub fn resize_kv_pool(&mut self, num_blocks: usize) -> OpResult<()> {
        debug_assert!(
            self.graph.is_none(),
            "resize_kv_pool must run before prime_graphs (no graph may reference the old KV base)"
        );
        let device = self.scope.device();
        let block_size = self.kv_pool.block_size;
        let kv_dim = self.kv_pool.kv_dim;
        let num_layers = self.kv_pool.layers.len();
        // Free the profiling pool first (drop its device tensors), so its bytes
        // are available for the new allocation on a memory-tight device.
        self.kv_pool.layers.clear();
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(PagedKvLayer {
                k: D::alloc_tensor(
                    Shape::from_slice(&[num_blocks, block_size, kv_dim]),
                    device,
                )?,
                v: D::alloc_tensor(
                    Shape::from_slice(&[num_blocks, block_size, kv_dim]),
                    device,
                )?,
            });
        }
        self.kv_pool.layers = layers;
        self.kv_pool.num_blocks = num_blocks;
        self.kv_pool.seq_kv_len.clear();
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

pub(super) fn validate_mixed_src_row(src: i32, batch: usize, label: &str) -> OpResult<usize> {
    let row = src as usize;
    if row >= batch {
        return Err(OpError::Kernel(format!(
            "step_fused_abc_eager: {} src_row {} >= batch {}",
            label, row, batch
        )));
    }
    Ok(row)
}

pub(super) fn u32_to_i32_saturating(x: u32) -> i32 {
    x.min(i32::MAX as u32) as i32
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

pub(super) fn finished_flags(req: &StepRequest, tokens: &[Vec<SampledToken>]) -> Vec<bool> {
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

pub(super) fn validate_step_request_vectors(req: &StepRequest, batch: usize) -> OpResult<()> {
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

pub(super) unsafe fn upload_i32_prefix<D: Device>(
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

/// Upload `host` into the front of `dst` and zero the remaining tail, so the
/// full buffer holds only this step's data. Required for per-sequence control
/// buffers (`cu_q_lens`/`seq_lens`/`seq_positions`/`kv_lens`) that the attention
/// kernels iterate over at capacity: a stale tail from a prior larger batch
/// would otherwise be read as phantom sequences. The pad is tiny (≤ cap_batch).
pub(super) unsafe fn upload_i32_full_zeropad<D: Device>(
    device: &D,
    dst: &Tensor<i32, D>,
    host: &[i32],
) -> OpResult<()> {
    let cap = dst.numel();
    if host.len() > cap {
        return Err(OpError::Shape(format!(
            "upload_i32_full_zeropad: host {} > dst {}",
            host.len(),
            cap
        )));
    }
    let mut padded = vec![0i32; cap];
    padded[..host.len()].copy_from_slice(host);
    let bytes = std::mem::size_of_val(padded.as_slice());
    let ptr = unsafe { NonNull::new_unchecked(dst.data_ptr_mut() as *mut u8) };
    unsafe { device.upload_async(ptr, padded.as_ptr() as *const u8, bytes) }
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
        pending: None,
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

    /// The PRODUCTION serving path — the ABC pipelined decode issue/finalize
    /// halves — runs end-to-end on the CPU backend through the
    /// `DecodePipelineOps` host reference implementation. Before the port,
    /// this path was `bf16 + Cuda` only and had no off-GPU test at all.
    #[test]
    fn abc_decode_pipeline_runs_on_cpu_backend() {
        let cpu = Cpu;
        let scope = HostScope::new(cpu);
        let mut runtime = Runtime::new(
            TinyDecoder,
            scope,
            Box::new(GreedySampler),
            8,
            1,
            8,
            8,
            4,
            4,
            Vec::new(),
        )
        .unwrap();

        // Two decode rows; TinyDecoder's argmax yields token (row + 1) % 4 →
        // C = [1, 2]. Row 0 hits max_tokens (0 + 1 >= 1) and finishes; row 1
        // survives and compacts to the front of A.
        let mut req = req_for_batch(2);
        req.stop.max_tokens = vec![1, 16];
        // a_valid_prefix = 0: both rows seed A through the B-admissions upload.
        runtime
            .issue_decode_abc(
                &req,
                0,
                &[0, 0],
                &[1, 16],
                &[false, false],
                &[9],
                Some(&[4, 5]),
                false,
            )
            .unwrap();
        let compact = runtime.finalize_decode_abc(2).unwrap();

        assert_eq!(compact.active.len(), 1);
        assert_eq!(compact.active[0].src_row, 1);
        assert_eq!(compact.active[0].token_id, 2);
        assert_eq!(compact.finished.len(), 1);
        assert_eq!(compact.finished[0].src_row, 0);
        assert_eq!(compact.finished[0].token_id, 1);

        // The async control builder ran on the host reference too: the next
        // step's control plane holds ONE survivor with its appended slot.
        let kv_lens = runtime.kv_index.kv_lens.to_host_vec().unwrap();
        assert_eq!(kv_lens[0], 2, "survivor advanced to kv_len 2");
        let block_tables = runtime.kv_index.block_tables.to_host_vec().unwrap();
        assert_eq!(
            block_tables[1], 4,
            "survivor's next-step slot appended at its old length"
        );
    }
}
