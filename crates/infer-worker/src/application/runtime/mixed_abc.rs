//! Fused mixed-batch (decode prefix + prefill rows) through the ABC data
//! model: eager and bucketed-graph paths, issue/finalize halves, and the
//! bootstrap mixed-graph prewarm (CUDA).

use crate::domain::component::Hidden;
use crate::domain::dtype::Dtype;
use crate::domain::exec::ExecScope;
use crate::domain::model::{DecoderModel, SampleRows};
use crate::domain::plan::{
    BatchKind, BatchPlan, SampledToken, SeqStep, StepOutput, StepRequest, StopCriteria,
};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::pipeline_ops::{CompactExtendControlArgs, MergeCompactMixedArgs};
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

use super::{
    PrefillGemmGuard, RaggedRowKind, Runtime, u32_to_i32_saturating, upload_i32_prefix,
    validate_mixed_src_row,
};

/// Tag bit OR'd into mixed-graph keys so they never collide with decode /
/// prefill graph keys in the scope's graph map.
const MIXED_GRAPH_KEY_TAG: u64 = 1 << 41;
/// Bucketed mixed graphs are keyed by row/decode-prefix/token/tile buckets, not
/// exact q_len layout. Keep the budget high enough for the common online QPS
/// buckets without falling back to eager during a timed run.
const MIXED_GRAPH_BUDGET: usize = 128;
/// Round mixed graph token shapes to this multiple. 64 keeps LM-head/FFN GEMM
/// shapes stable without padding every mixed step all the way to
/// `max_batch_tokens`.
const MIXED_GRAPH_TOKEN_BUCKET: usize = 64;
/// Round mixed graph ragged tile grids to this multiple. This keeps capture
/// cardinality bounded while avoiding the previous `tile_capacity` grid, which
/// launched hundreds of no-op tiles for small mixed batches.
const MIXED_GRAPH_TILE_BUCKET: usize = 32;
/// Number of common mixed graph buckets to capture at bootstrap. Keep this
/// below the total mixed graph budget so rare live buckets can still be learned.
/// 48 truncated the case grid inside the 192-token bucket, so every live mixed
/// step above ~192 tokens (any real prompt riding a loaded decode batch) missed
/// the graph and ran eager — the qps32 ITL p99 tail. 112 covers the full
/// prefix(≤128) × token-bucket(≤384) grid (104 cases).
const MIXED_GRAPH_PREWARM_MAX: usize = 112;
const MIXED_GRAPH_PREWARM_MAX_DECODE_PREFIX: usize = 128;
const MIXED_GRAPH_PREWARM_TOKEN_BUCKETS: &[usize] = &[64, 128, 192, 256, 320, 384];

/// In-flight mixed step handle: `issue_fused_abc` → overlapped host work →
/// `finalize_fused_abc`. Owns the issue's `BatchPlan` so finalize decodes the
/// host mirrors against the exact issued shape.
pub struct MixedStepTicket {
    plan: BatchPlan,
    ran_graph: bool,
    trace: bool,
    t0: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MixedGraphShape {
    rows: usize,
    tokens: usize,
    tiles: i32,
    decode_prefix: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MixedGraphWarmupCase {
    decode_prefix: usize,
    token_bucket: usize,
    prefill_len: usize,
}

fn mixed_graph_warmup_cases(
    capture_sizes: &[usize],
    cap_batch: usize,
    cap_num_tokens: usize,
    max_prefill_len: usize,
    limit: usize,
) -> Vec<MixedGraphWarmupCase> {
    if limit == 0 {
        return Vec::new();
    }
    let mut prefixes: Vec<usize> = capture_sizes
        .iter()
        .copied()
        .filter(|&p| {
            p > 0
                && p <= MIXED_GRAPH_PREWARM_MAX_DECODE_PREFIX
                && p < cap_batch
                && ceil_capture_slot(capture_sizes, p + 1).is_some()
        })
        .collect();
    prefixes.sort_unstable();
    prefixes.dedup();

    let mut out = Vec::with_capacity(limit.min(64));
    for &token_bucket in MIXED_GRAPH_PREWARM_TOKEN_BUCKETS {
        if token_bucket > cap_num_tokens {
            continue;
        }
        for &decode_prefix in &prefixes {
            if out.len() >= limit {
                return out;
            }
            if decode_prefix + 2 > token_bucket {
                continue;
            }
            let prefill_len = token_bucket - decode_prefix;
            if prefill_len > max_prefill_len {
                continue;
            }
            out.push(MixedGraphWarmupCase {
                decode_prefix,
                token_bucket,
                prefill_len,
            });
        }
    }
    out
}

fn mixed_graph_shape(
    plan: &BatchPlan,
    row_kind: &[RaggedRowKind],
    cap_batch: usize,
    cap_num_tokens: usize,
    tile_capacity: i32,
    capture_sizes: &[usize],
) -> Option<MixedGraphShape> {
    if !matches!(plan.kind, BatchKind::Ragged) {
        return None;
    }
    if plan.batch == 0 || plan.batch > cap_batch || plan.num_tokens > cap_num_tokens {
        return None;
    }
    let has_decode = row_kind.iter().any(|&k| k == RaggedRowKind::Decode);
    let has_prefill = row_kind
        .iter()
        .any(|&k| matches!(k, RaggedRowKind::PrefillFinal | RaggedRowKind::PrefillCont));
    if !has_decode || !has_prefill {
        return None;
    }
    let actual_decode_prefix = plan.q_lens.iter().take_while(|&&q| q == 1).count();
    let decode_prefix = floor_capture_slot(capture_sizes, actual_decode_prefix)?;
    let rows = ceil_capture_slot(capture_sizes, plan.batch)?;
    if decode_prefix == 0 || decode_prefix >= rows || rows > cap_batch {
        return None;
    }
    let tokens = round_up_to_bucket(plan.num_tokens, MIXED_GRAPH_TOKEN_BUCKET)?;
    let actual_tiles = usize::try_from(plan.total_q_tiles).ok()?;
    let tile_capacity = usize::try_from(tile_capacity).ok()?;
    let tiles = round_up_to_bucket(actual_tiles.max(1), MIXED_GRAPH_TILE_BUCKET)?;
    if tokens > cap_num_tokens || tiles > tile_capacity {
        return None;
    }
    Some(MixedGraphShape {
        rows,
        tokens,
        tiles: tiles as i32,
        decode_prefix,
    })
}

fn floor_capture_slot(capture_sizes: &[usize], n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    capture_sizes.iter().copied().filter(|&s| s <= n).max()
}

fn ceil_capture_slot(capture_sizes: &[usize], n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    capture_sizes.iter().copied().filter(|&s| s >= n).min()
}

fn round_up_to_bucket(n: usize, bucket: usize) -> Option<usize> {
    if bucket == 0 {
        return None;
    }
    Some(n.div_ceil(bucket) * bucket)
}

fn mixed_graph_key(shape: MixedGraphShape, eos_len: usize, next_control: bool) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x100000001b3);
    };
    mix(shape.rows as u64);
    mix(shape.tokens as u64);
    mix(shape.tiles as u64);
    mix(shape.decode_prefix as u64);
    mix(eos_len as u64);
    mix(next_control as u64);
    MIXED_GRAPH_KEY_TAG | (h & ((1u64 << 40) - 1))
}

impl<T, D, M> Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    /// Capture the common mixed (decode-prefix + one prefill admission) ABC
    /// graph buckets before the worker reports Ready. Live mixed batches used to
    /// discover these buckets inline on the serve thread, so the first request
    /// at each `(decode_prefix, token_bucket)` paid eager warmup + synchronize +
    /// stream capture in the timed path.
    ///
    /// The synthetic request mirrors the scheduler's common online shape: a
    /// decode prefix already rounded to a capture slot, followed by one prefill
    /// final row. It uses real `step_fused_abc_eager`, with `next_slots`, so the
    /// captured region and graph key match the high-QPS ABC path.
    pub fn prewarm_mixed_graphs(&mut self, eos_ids: &[i32]) -> OpResult<usize> {
        if self.graph.is_none() || !self.scope.supports_graphs() {
            return Ok(0);
        }
        let prev_capture_enabled = self.mixed_graph_capture_enabled;
        self.mixed_graph_capture_enabled = true;
        let result = self.prewarm_mixed_graphs_inner(eos_ids);
        self.mixed_graph_capture_enabled = prev_capture_enabled;
        result
    }

    fn prewarm_mixed_graphs_inner(&mut self, eos_ids: &[i32]) -> OpResult<usize> {
        let max_prefill_len = self
            .max_seq_len
            .min(self.cap_num_tokens)
            .min(self.max_blocks_per_seq.saturating_mul(self.block_size));
        let cases = mixed_graph_warmup_cases(
            &self.capture_sizes,
            self.cap_batch,
            self.cap_num_tokens,
            max_prefill_len,
            MIXED_GRAPH_PREWARM_MAX,
        );
        let mut warmed = 0usize;
        let mut seen = std::collections::HashSet::new();
        for case in cases {
            let Some((req, row_kind, next_slots, key)) =
                self.mixed_graph_warmup_request(case, eos_ids)?
            else {
                continue;
            };
            if !seen.insert(key) || self.scope.graph_ready(key) {
                continue;
            }
            self.step_fused_abc_eager(&req, &row_kind, Some(&next_slots))?;
            warmed += 1;
            if self.mixed_graphs_captured >= MIXED_GRAPH_BUDGET {
                break;
            }
        }
        self.scope.synchronize()?;
        Ok(warmed)
    }

    fn mixed_graph_warmup_request(
        &self,
        case: MixedGraphWarmupCase,
        eos_ids: &[i32],
    ) -> OpResult<Option<(StepRequest, Vec<RaggedRowKind>, Vec<u32>, u64)>> {
        if case.prefill_len < 2 {
            return Ok(None);
        }
        let batch = case.decode_prefix + 1;
        if batch > self.cap_batch || case.decode_prefix + case.prefill_len > self.cap_num_tokens {
            return Ok(None);
        }

        let prefill_blocks = case.prefill_len.div_ceil(self.block_size);
        if prefill_blocks > self.max_blocks_per_seq {
            return Ok(None);
        }
        let needed_blocks = case
            .decode_prefix
            .saturating_add(prefill_blocks)
            .saturating_add(batch);
        if needed_blocks > self.kv_pool.num_blocks || needed_blocks > u32::MAX as usize {
            return Ok(None);
        }

        let mut next_block = 0u32;
        let mut seqs = Vec::with_capacity(batch);
        let mut row_kind = Vec::with_capacity(batch);
        for i in 0..case.decode_prefix {
            let block = next_block;
            next_block += 1;
            seqs.push(SeqStep {
                sequence_id: 10_000_000 + i as u64,
                input_ids: vec![1],
                positions: vec![0],
                kv_write_start: 0,
                kv_len_after: 1,
                block_table: vec![block],
            });
            row_kind.push(RaggedRowKind::Decode);
        }

        let prefill_start = next_block;
        next_block += prefill_blocks as u32;
        seqs.push(SeqStep {
            sequence_id: 20_000_000 + case.token_bucket as u64,
            input_ids: vec![1; case.prefill_len],
            positions: (0..case.prefill_len as i32).collect(),
            kv_write_start: 0,
            kv_len_after: case.prefill_len as i32,
            block_table: (prefill_start..next_block).collect(),
        });
        row_kind.push(RaggedRowKind::PrefillFinal);

        let next_slots: Vec<u32> = (0..batch)
            .map(|_| {
                let block = next_block;
                next_block += 1;
                block
            })
            .collect();

        let req = StepRequest {
            sampling: vec![Default::default(); batch],
            stop: StopCriteria {
                eos_ids: eos_ids.to_vec(),
                generated_counts: vec![0; batch],
                max_tokens: vec![u32::MAX; batch],
                ignore_eos: vec![true; batch],
            },
            draft_tokens: Vec::new(),
            seqs,
        };
        let plan = self.build_plan(&req)?;
        let Some(shape) = self.graph.as_ref().and_then(|graph| {
            mixed_graph_shape(
                &plan,
                &row_kind,
                self.cap_batch,
                self.cap_num_tokens,
                self.mixed_graph_tile_capacity(),
                graph.capture_sizes(),
            )
        }) else {
            return Ok(None);
        };
        let key = mixed_graph_key(shape, eos_ids.len(), true);
        Ok(Some((req, row_kind, next_slots, key)))
    }

    /// ABC GPU-resident decode step (buffer A = `input_ids_buf`).
    ///
    /// Precondition: rows `0..a_valid_prefix` of A already hold the correct
    /// input token — they are the longest prefix of this step's row order that
    /// is unchanged from the prior step, so the previous step's compact merge
    /// already wrote their tokens on-device. The divergent suffix (fresh
    /// admissions, or rows shifted by an out-of-band eviction) is uploaded
    /// through buffer B and appended into A.
    ///
    /// Forward + sample a fused (mixed prefill + decode) batch, always eager.
    /// A fused step mixes q>1 prefill-chunk rows with q=1 decode rows in one
    /// ragged forward. It must NOT take the captured decode-graph path (that path
    /// is driven by the ABC pipeline with fixed buffers, not this request-driven
    /// forward), so even an all-q=1 fused batch is routed through `step_eager`.
    /// The ragged/varlen attention handles any per-row q distribution; the eager
    /// finalize samples one token per row in `req.seqs` order.
    pub fn step_fused_eager(&mut self, req: &StepRequest) -> OpResult<StepOutput> {
        // Eager fused forward. Route ALL transient scratch through the bump arena
        // (zero cudaMalloc/cudaFree) instead of the recycling pool, which churns
        // cudaMalloc on every new ragged shape and cudaFree (device-sync) once
        // over its retain budget. Arena pointers are no-op on free
        // (arena_contains filter); on exhaustion allocs fall back to the pool.
        //
        // NOTE: this path is currently ~2.4x slower than the decode baseline at
        // high QPS. Root cause (measured): the fused batch routes the q=1 DECODE
        // rows through the CuTe ragged PREFILL attention kernel (1/128 tile
        // utilization). The CUDA-graph attempt was a dead end (graphs the slow
        // attention). The real fix is to SPLIT the fused attention by row type —
        // q=1 decode rows → fast cuDNN decode SDPA, q>1 prefill rows → cuDNN
        // varlen+causal SDPA (needs a varlen-q cuDNN graph; current graph
        // hardcodes q-seqlen=1) or the CuTe ragged kernel. See memory
        // fused-mixed-batch-and-graph.
        D::pipeline_arena_begin(&self.scope);
        let result = (|| {
            let plan = self.build_plan(req)?;
            self.upload_index(&plan, req)?;
            self.step_eager(&plan, req)
        })();
        D::pipeline_arena_end(&self.scope);
        result
    }

    /// Mixed/ragged forward through the ABC data model — issue half.
    ///
    /// Hot path: an exact-shape mixed CUDA graph captures forward + finalize +
    /// selected-row argmax + mixed merge + optional next-control compact. Eager
    /// remains the fallback and keeps the same flat-token ABC contract.
    ///
    /// Launches the forward and enqueues the result copy-out WITHOUT syncing,
    /// so the caller can do host work (e.g. send the prior step's tokens)
    /// while the GPU runs. Exactly one `finalize_fused_abc` must follow each
    /// successful issue before any other step is issued.
    pub fn issue_fused_abc(
        &mut self,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
        next_slots: Option<&[u32]>,
    ) -> OpResult<MixedStepTicket> {
        let plan = self.validate_mixed_abc_request(req, row_kind)?;
        if let Some(slots) = next_slots {
            self.upload_mixed_next_slots(slots)?;
        }

        let trace = std::env::var_os("RUSTINFER_MIXED_TRACE").is_some();
        let t0 = std::time::Instant::now();
        let ran_graph = self.try_run_mixed_abc_graph(&plan, req, row_kind, next_slots.is_some())?;
        if !ran_graph {
            self.upload_index(&plan, req)?;
            let input_ids = self.input_ids_tensor(req, &plan)?;
            self.upload_mixed_abc_metadata(req, row_kind, plan.batch)?;
            D::pipeline_arena_begin(&self.scope);
            let result =
                self.run_mixed_abc_eager_region(&plan, req, &input_ids, next_slots.is_some());
            D::pipeline_arena_end(&self.scope);
            result?;
        }
        self.copy_out_mixed_abc(plan.batch)?;
        Ok(MixedStepTicket {
            plan,
            ran_graph,
            trace,
            t0,
        })
    }

    /// Collect a mixed step issued by `issue_fused_abc`: drain the copy-out
    /// (this is where the host blocks for the forward) and decode the host
    /// mirrors into a `StepOutput`. `req`/`row_kind` must be the same values
    /// the issue ran with.
    pub fn finalize_fused_abc(
        &mut self,
        ticket: MixedStepTicket,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
    ) -> OpResult<StepOutput> {
        D::pipeline_synchronize_copy_out(&self.scope)?;
        if ticket.trace {
            // `elapsed` spans issue→sync, so it includes any host work the
            // caller overlapped between the two halves.
            tracing::info!(
                "[mixed-trace] mode={} rows={} tokens={} tiles={} elapsed={:.2}ms",
                if ticket.ran_graph { "graph" } else { "eager" },
                ticket.plan.batch,
                ticket.plan.num_tokens,
                ticket.plan.total_q_tiles,
                ticket.t0.elapsed().as_secs_f64() * 1e3
            );
        }
        self.finalize_mixed_abc(&ticket.plan, req, row_kind)
    }

    /// Synchronous mixed step: issue + finalize back-to-back. Used by the
    /// bootstrap prewarm; serving uses the split halves to overlap host work
    /// with the fused forward.
    pub fn step_fused_abc_eager(
        &mut self,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
        next_slots: Option<&[u32]>,
    ) -> OpResult<StepOutput> {
        let ticket = self.issue_fused_abc(req, row_kind, next_slots)?;
        self.finalize_fused_abc(ticket, req, row_kind)
    }

    /// Largest mixed-graph token bucket the bootstrap prewarm covers. The
    /// fused-step packer bounds each step's prefill admission to this so live
    /// mixed steps replay a prewarmed graph instead of falling back to eager.
    /// `None` when graphs are unavailable (packer falls back to the raw cap).
    pub fn mixed_step_token_budget(&self) -> Option<usize> {
        if self.graph.is_none() || !self.scope.supports_graphs() {
            return None;
        }
        MIXED_GRAPH_PREWARM_TOKEN_BUCKETS
            .iter()
            .copied()
            .filter(|&b| b <= self.cap_num_tokens)
            .max()
    }

    fn validate_mixed_abc_request(
        &self,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
    ) -> OpResult<BatchPlan> {
        let plan = self.build_plan(req)?;
        if !req.draft_tokens.is_empty() {
            return Err(OpError::Shape(
                "step_fused_abc_eager: speculative mixed batches are not supported".into(),
            ));
        }
        if row_kind.len() != plan.batch {
            return Err(OpError::Shape(format!(
                "step_fused_abc_eager: row_kind {} != batch {}",
                row_kind.len(),
                plan.batch
            )));
        }
        if req.stop.eos_ids.len() > self.abc.eos_ids_dev.numel() {
            return Err(OpError::Shape(format!(
                "step_fused_abc_eager: eos_ids {} > capacity {}",
                req.stop.eos_ids.len(),
                self.abc.eos_ids_dev.numel()
            )));
        }
        Ok(plan)
    }

    fn try_run_mixed_abc_graph(
        &mut self,
        plan: &BatchPlan,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
        next_control: bool,
    ) -> OpResult<bool> {
        if self.graph.is_none() || !self.scope.supports_graphs() {
            return Ok(false);
        }
        let Some(shape) = self.graph.as_ref().and_then(|graph| {
            mixed_graph_shape(
                plan,
                row_kind,
                self.cap_batch,
                self.cap_num_tokens,
                self.mixed_graph_tile_capacity(),
                graph.capture_sizes(),
            )
        }) else {
            return Ok(false);
        };
        let key = mixed_graph_key(shape, req.stop.eos_ids.len(), next_control);
        if self.scope.graph_ready(key) {
            self.upload_index_with_suffix_prefix(plan, req, Some(shape.decode_prefix))?;
            self.upload_input_ids_bucket(req, plan.num_tokens, shape.tokens)?;
            self.upload_mixed_abc_metadata(req, row_kind, shape.rows)?;
            self.scope.graph_launch(key)?;
            return Ok(true);
        }
        if !self.mixed_graph_capture_enabled {
            return Ok(false);
        }
        if self.mixed_graphs_captured >= MIXED_GRAPH_BUDGET {
            return Ok(false);
        }

        let graph_plan = self.mixed_graph_bucket_plan(plan, shape);
        let input_ids = self.upload_input_ids_bucket(req, plan.num_tokens, shape.tokens)?;
        self.upload_index_with_suffix_prefix(plan, req, Some(shape.decode_prefix))?;
        self.upload_mixed_abc_metadata(req, row_kind, shape.rows)?;

        // Warm only the capturable forward/finalize kernels. Do not run merge or
        // compact-extend here, because compact-extend mutates this step's control
        // plane into the next step's control plane.
        self.forward_finalize_argmax_all_selected(&graph_plan, &input_ids)?;
        self.scope.synchronize()?;

        // Re-seed every device input the graph reads. Warmup should be
        // idempotent, but keeping this explicit makes future mutation safer.
        self.upload_index_with_suffix_prefix(plan, req, Some(shape.decode_prefix))?;
        self.upload_input_ids_bucket(req, plan.num_tokens, shape.tokens)?;
        self.upload_mixed_abc_metadata(req, row_kind, shape.rows)?;
        self.scope.graph_capture_begin()?;
        if let Err(e) =
            self.run_mixed_abc_graph_region(&graph_plan, req.stop.eos_ids.len(), next_control)
        {
            let _ = self.scope.graph_capture_end(key);
            return Err(e);
        }
        self.scope.graph_capture_end(key)?;
        self.mixed_graphs_captured += 1;
        tracing::info!(
            "[graph] captured mixed ABC graph bucket rows={} tokens={} tiles={} decode_prefix={} actual_rows={} actual_tokens={} actual_tiles={} next_control={} ({}/{})",
            shape.rows,
            shape.tokens,
            shape.tiles,
            shape.decode_prefix,
            plan.batch,
            plan.num_tokens,
            plan.total_q_tiles,
            next_control,
            self.mixed_graphs_captured,
            MIXED_GRAPH_BUDGET
        );
        self.scope.graph_launch(key)?;
        Ok(true)
    }

    fn run_mixed_abc_eager_region(
        &mut self,
        plan: &BatchPlan,
        req: &StepRequest,
        input_ids: &Tensor<i32, D>,
        next_control: bool,
    ) -> OpResult<()> {
        let prefill_gemm = plan.num_tokens > plan.batch;
        if prefill_gemm {
            D::set_prefill_gemm_mode(true);
        }
        let _gemm_guard = PrefillGemmGuard::<D>(prefill_gemm, std::marker::PhantomData);
        self.run_layers(plan, input_ids)?;
        self.forward_argmax_last_per_seq(plan)?;
        let control_rows = self.mixed_next_control_rows(plan.batch);
        self.run_mixed_merge_and_next_control(
            plan,
            req.stop.eos_ids.len(),
            next_control,
            control_rows,
        )
    }

    fn run_mixed_abc_graph_region(
        &mut self,
        plan: &BatchPlan,
        eos_len: usize,
        next_control: bool,
    ) -> OpResult<()> {
        let input_ids = self.prefill_ids_buf.view_raw(
            Shape::from_slice(&[plan.num_tokens]),
            Shape::from_slice(&[plan.num_tokens.max(1)]).contiguous_strides(),
            0,
            true,
        );
        self.forward_finalize_argmax_all_selected(plan, &input_ids)?;
        self.run_mixed_merge_and_next_control(plan, eos_len, next_control, plan.batch)
    }

    fn mixed_graph_tile_capacity(&self) -> i32 {
        let tile = crate::domain::plan::RAGGED_Q_TILE as usize;
        (self.cap_batch + self.cap_num_tokens.div_ceil(tile)).max(1) as i32
    }

    fn mixed_graph_bucket_plan(&self, actual: &BatchPlan, shape: MixedGraphShape) -> BatchPlan {
        let mut q_lens = vec![0i32; shape.rows];
        for q in q_lens.iter_mut().take(shape.decode_prefix) {
            *q = 1;
        }
        if shape.decode_prefix < shape.rows {
            // Keep the host-side attention branch stable: the leading q==1 run
            // is the decode prefix, and a non-1 row after it forces the ragged
            // suffix path. Real q lengths live in the device control tensors.
            q_lens[shape.decode_prefix] = 2;
        }
        BatchPlan {
            kind: BatchKind::Ragged,
            num_tokens: shape.tokens,
            batch: shape.rows,
            q_lens,
            kv_lens: vec![0; shape.rows],
            seq_positions: vec![0; shape.rows],
            rope_positions: vec![0; shape.tokens],
            max_blocks_per_seq: actual.max_blocks_per_seq,
            block_size: actual.block_size,
            total_q_tiles: shape.tiles,
        }
    }

    fn forward_argmax_last_per_seq(&mut self, plan: &BatchPlan) -> OpResult<()> {
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
        let logits = self.model.finalize(&hidden, SampleRows::LastPerSeq, &ctx)?;
        let logits_rows = logits.0.shape().as_slice()[0];
        if logits_rows != plan.batch {
            return Err(OpError::Shape(format!(
                "step_fused_abc_eager: logits rows {} != batch {}",
                logits_rows, plan.batch
            )));
        }
        if logits_rows > self.abc.argmax_out_dev.numel() {
            return Err(OpError::Shape(format!(
                "step_fused_abc_eager: logits rows {} exceeds C capacity {}",
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

    fn forward_finalize_argmax_all_selected(
        &mut self,
        plan: &BatchPlan,
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
        let mut c_view = self.abc.argmax_out_dev.view_raw(
            Shape::from_slice(&[plan.batch]),
            Shape::from_slice(&[plan.batch.max(1)]).contiguous_strides(),
            0,
            true,
        );
        let selected = self.abc.last_token_rows_dev.view_raw(
            Shape::from_slice(&[plan.batch]),
            Shape::from_slice(&[plan.batch.max(1)]).contiguous_strides(),
            0,
            true,
        );
        D::argmax_into(
            &ctx,
            &logits.0,
            &mut c_view,
            &self.abc.argmax_ws,
            Some(&selected),
        )
    }

    fn run_mixed_merge_and_next_control(
        &mut self,
        plan: &BatchPlan,
        eos_len: usize,
        next_control: bool,
        control_rows: usize,
    ) -> OpResult<()> {
        let mut a_view = self.input_ids_buf.view_raw(
            Shape::from_slice(&[plan.batch]),
            Shape::from_slice(&[plan.batch.max(1)]).contiguous_strides(),
            0,
            true,
        );
        D::merge_compact_mixed(
            &self.scope,
            MergeCompactMixedArgs {
                a_out: &mut a_view,
                c_prev: &self.abc.argmax_out_dev,
                row_kind: &self.abc.row_kind_dev,
                generated_counts: &self.abc.generated_counts_dev,
                max_tokens: &self.abc.max_tokens_dev,
                ignore_eos: &self.abc.ignore_eos_dev,
                eos_ids: &self.abc.eos_ids_dev,
                eos_len,
                old_rows: plan.batch,
                active_src_rows: &mut self.abc.active_src_rows_dev,
                active_tokens: &mut self.abc.active_tokens_dev,
                finished_src_rows: &mut self.abc.finished_src_rows_dev,
                finished_tokens: &mut self.abc.finished_tokens_dev,
                prefill_final_src_rows: &mut self.abc.prefill_final_src_rows_dev,
                prefill_final_tokens: &mut self.abc.prefill_final_tokens_dev,
                counts: &mut self.abc.counts_dev,
            },
        )?;
        if next_control {
            self.run_compact_extend_control(control_rows)?;
        }
        Ok(())
    }

    fn mixed_next_control_rows(&self, batch: usize) -> usize {
        self.next_capture_slot(batch)
            .unwrap_or(batch)
            .min(self.cap_batch)
            .max(batch)
    }

    fn copy_out_mixed_abc(&mut self, batch: usize) -> OpResult<()> {
        self.ensure_abc_pinned()?;
        D::pipeline_record_compute_a(&self.scope)?;
        D::pipeline_copy_out_wait_compute_a(&self.scope)?;
        let elem = std::mem::size_of::<i32>();
        let row_bytes = batch * elem;
        unsafe {
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.counts_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.counts_dev.data_ptr() as *const std::ffi::c_void,
                4 * elem,
            )?;
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.argmax_out_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.argmax_out_dev.data_ptr() as *const std::ffi::c_void,
                row_bytes,
            )?;
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.active_src_rows_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.active_src_rows_dev.data_ptr() as *const std::ffi::c_void,
                row_bytes,
            )?;
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.finished_src_rows_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.finished_src_rows_dev.data_ptr() as *const std::ffi::c_void,
                row_bytes,
            )?;
        }
        D::pipeline_record_copy_out(&self.scope)?;
        // No sync here: the D2H runs async on the copy-out stream and is
        // collected by `finalize_fused_abc`, so the caller can overlap host
        // work with the fused forward. `copy_out_recorded` gates the next
        // step's A overwrite via `compute_wait_copy_out`.
        self.abc.copy_out_recorded = true;
        Ok(())
    }

    fn upload_mixed_abc_metadata(
        &mut self,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
        rows_to_upload: usize,
    ) -> OpResult<()> {
        let batch = req.seqs.len();
        if rows_to_upload < batch || rows_to_upload > self.cap_batch {
            return Err(OpError::Shape(format!(
                "upload_mixed_abc_metadata: rows_to_upload {} not in [{}..={}]",
                rows_to_upload, batch, self.cap_batch
            )));
        }
        let mut gen_i32 = Vec::with_capacity(rows_to_upload);
        let mut max_i32 = Vec::with_capacity(rows_to_upload);
        let mut ign_i32 = Vec::with_capacity(rows_to_upload);
        let mut token_offset = 0i32;
        for i in 0..batch {
            gen_i32.push(u32_to_i32_saturating(
                req.stop.generated_counts.get(i).copied().unwrap_or(0),
            ));
            max_i32.push(u32_to_i32_saturating(
                req.stop.max_tokens.get(i).copied().unwrap_or(u32::MAX),
            ));
            ign_i32.push(i32::from(
                req.stop.ignore_eos.get(i).copied().unwrap_or(false),
            ));
            self.abc.row_kind_host[i] = row_kind[i].as_i32();
            let q = req.seqs[i].input_ids.len().max(1) as i32;
            // `token_offset` indexes into the flat token tape, bounded by
            // `max_batch_tokens` in practice. Use a checked add so a malformed
            // oversized batch fails loudly instead of wrapping `i32` and pointing
            // `last_token_rows` at the wrong tape row (which would sample and
            // return the wrong token to the user).
            token_offset = token_offset.checked_add(q).ok_or_else(|| {
                OpError::Shape(format!(
                    "upload_mixed_abc_metadata: token_offset overflow at row {} (q={})",
                    i, q
                ))
            })?;
            self.abc.last_token_rows_host[i] = token_offset - 1;
        }
        for i in batch..rows_to_upload {
            gen_i32.push(0);
            max_i32.push(i32::MAX);
            ign_i32.push(1);
            self.abc.row_kind_host[i] = RaggedRowKind::Pad.as_i32();
            self.abc.last_token_rows_host[i] = 0;
        }
        let device = self.scope.device();
        unsafe {
            upload_i32_prefix(device, &self.abc.generated_counts_dev, &gen_i32)?;
            upload_i32_prefix(device, &self.abc.max_tokens_dev, &max_i32)?;
            upload_i32_prefix(device, &self.abc.ignore_eos_dev, &ign_i32)?;
            upload_i32_prefix(
                device,
                &self.abc.row_kind_dev,
                &self.abc.row_kind_host[..rows_to_upload],
            )?;
            upload_i32_prefix(
                device,
                &self.abc.last_token_rows_dev,
                &self.abc.last_token_rows_host[..rows_to_upload],
            )?;
            if !req.stop.eos_ids.is_empty() {
                upload_i32_prefix(device, &self.abc.eos_ids_dev, &req.stop.eos_ids)?;
            }
        }
        Ok(())
    }

    fn upload_mixed_next_slots(&mut self, next_slots: &[u32]) -> OpResult<()> {
        if next_slots.is_empty() {
            return Ok(());
        }
        if next_slots.len() > self.cap_batch {
            return Err(OpError::Shape(format!(
                "upload_mixed_next_slots: next_slots {} > cap_batch {}",
                next_slots.len(),
                self.cap_batch
            )));
        }
        self.ensure_async_ctrl()?;
        let slots_i32: Vec<i32> = next_slots.iter().map(|&s| s as i32).collect();
        let ctrl = self.async_ctrl.as_ref().expect("async_ctrl ensured");
        unsafe {
            upload_i32_prefix(self.scope.device(), &ctrl.new_slots_dev, &slots_i32)?;
        }
        Ok(())
    }

    fn run_compact_extend_control(&mut self, control_rows: usize) -> OpResult<()> {
        self.ensure_async_ctrl()?;
        let mbps = self.max_blocks_per_seq;
        let cap_batch = control_rows.clamp(1, self.cap_batch);
        let _guard = self.scope.enter();
        let ctrl = self.async_ctrl.as_mut().expect("async_ctrl ensured");
        D::compact_extend_control(
            &self.scope,
            CompactExtendControlArgs {
                block_tables: &mut self.kv_index.block_tables,
                block_tables_scratch: &mut ctrl.block_tables_scratch,
                kv_lens: &mut self.kv_index.kv_lens,
                kv_lens_scratch: &mut ctrl.kv_lens_scratch,
                seq_positions_out: &mut self.kv_index.seq_positions,
                seq_lens_step_out: &mut self.kv_index.seq_lens_step,
                rope_positions_out: &mut self.kv_index.rope_positions,
                cu_q_lens_out: &mut self.kv_index.cu_q_lens,
                block2req_out: &mut self.kv_index.block2req,
                block2tile_out: &mut self.kv_index.block2tile,
                active_src_rows: &self.abc.active_src_rows_dev,
                counts: &self.abc.counts_dev,
                new_slots: &ctrl.new_slots_dev,
                mbps,
                cap_batch,
            },
        )
    }

    fn finalize_mixed_abc(
        &mut self,
        plan: &BatchPlan,
        req: &StepRequest,
        row_kind: &[RaggedRowKind],
    ) -> OpResult<StepOutput> {
        let batch = plan.batch;
        let counts = &self.abc.counts_host[..4];
        let active_n = counts[0].max(0) as usize;
        let finished_n = counts[1].max(0) as usize;
        let prefill_final_n = counts[2].max(0) as usize;
        let old_n = counts[3].max(0) as usize;
        if old_n < batch
            || old_n > self.cap_batch
            || active_n + finished_n > batch
            || prefill_final_n > batch
        {
            return Err(OpError::Kernel(format!(
                "step_fused_abc_eager: mixed counts invalid active={} finished={} prefill_final={} old={} batch={} cap={}",
                active_n, finished_n, prefill_final_n, old_n, batch, self.cap_batch
            )));
        }

        let active_src = &self.abc.active_src_rows_host[..batch];
        let finished_src = &self.abc.finished_src_rows_host[..batch];
        let row_tokens = &self.abc.argmax_out_host[..batch];

        let mut row_results: Vec<Option<(i32, bool)>> = vec![None; batch];
        let mut seen = vec![false; batch];
        for k in 0..active_n {
            let row = validate_mixed_src_row(active_src[k], batch, "active")?;
            if !row_kind[row].emits_token() {
                return Err(OpError::Kernel(format!(
                    "step_fused_abc_eager: active row {} has non-emitting kind {:?}",
                    row, row_kind[row]
                )));
            }
            if seen[row] {
                return Err(OpError::Kernel(format!(
                    "step_fused_abc_eager: row {} returned twice",
                    row
                )));
            }
            seen[row] = true;
            row_results[row] = Some((row_tokens[row], false));
        }
        for k in 0..finished_n {
            let row = validate_mixed_src_row(finished_src[k], batch, "finished")?;
            if !row_kind[row].emits_token() {
                return Err(OpError::Kernel(format!(
                    "step_fused_abc_eager: finished row {} has non-emitting kind {:?}",
                    row, row_kind[row]
                )));
            }
            if seen[row] {
                return Err(OpError::Kernel(format!(
                    "step_fused_abc_eager: row {} returned twice",
                    row
                )));
            }
            seen[row] = true;
            row_results[row] = Some((row_tokens[row], true));
        }
        for (row, kind) in row_kind.iter().copied().enumerate() {
            if kind.emits_token() && row_results[row].is_none() {
                return Err(OpError::Kernel(format!(
                    "step_fused_abc_eager: emitting row {} missing from mixed merge",
                    row
                )));
            }
        }

        let mut tokens: Vec<Vec<SampledToken>> = Vec::with_capacity(batch);
        let mut finished = Vec::with_capacity(batch);
        for result in row_results {
            if let Some((token_id, done)) = result {
                tokens.push(vec![SampledToken {
                    token_id,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                }]);
                finished.push(done);
            } else {
                tokens.push(Vec::new());
                finished.push(false);
            }
        }
        for (seq, done) in req.seqs.iter().zip(finished.iter()) {
            if *done {
                self.kv_pool.seq_kv_len.remove(&seq.sequence_id);
            }
        }
        Ok(StepOutput {
            tokens,
            accepted: plan.q_lens.iter().map(|&q| q.max(0) as u32).collect(),
            finished,
            hidden_tap: None,
        })
    }

    /// Build the next pure-decode control plane after a mixed ABC step.
    ///
    /// `step_fused_abc_eager` leaves this step's block tables / kv_lens in
    /// `kv_index` and mixed survivors in `abc.active_src_rows_dev`. Given one
    /// fresh KV slot per survivor, the same compact-extend kernel used by the
    /// decode ABC path can gather survivors into the compacted front and append
    /// those slots, so the next pure decode step can skip host control upload.
    pub fn prepare_mixed_next_decode_control(&mut self, next_slots: &[u32]) -> OpResult<()> {
        let active_n = self.abc.counts_host[0].max(0) as usize;
        if next_slots.len() != active_n {
            return Err(OpError::Shape(format!(
                "prepare_mixed_next_decode_control: next_slots {} != active {}",
                next_slots.len(),
                active_n
            )));
        }
        if active_n == 0 {
            return Ok(());
        }
        self.upload_mixed_next_slots(next_slots)?;
        let control_rows = self.mixed_next_control_rows(active_n);
        self.run_compact_extend_control(control_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_graph_shape_buckets_tokens_and_rows() {
        let plan = BatchPlan {
            kind: BatchKind::Ragged,
            num_tokens: 321,
            batch: 21,
            q_lens: vec![1; 16]
                .into_iter()
                .chain([65, 64, 64, 64, 63])
                .collect(),
            kv_lens: vec![1; 21],
            seq_positions: vec![0; 21],
            rope_positions: vec![0; 321],
            max_blocks_per_seq: 96,
            block_size: 1,
            total_q_tiles: 21,
        };
        let mut kinds = vec![RaggedRowKind::Decode; 16];
        kinds.extend([RaggedRowKind::PrefillCont; 4]);
        kinds.push(RaggedRowKind::PrefillFinal);

        let shape =
            mixed_graph_shape(&plan, &kinds, 32, 512, 36, &[1, 2, 4, 8, 16, 24, 32]).unwrap();

        assert_eq!(
            shape,
            MixedGraphShape {
                rows: 24,
                tokens: 384,
                tiles: 32,
                decode_prefix: 16,
            }
        );
    }

    #[test]
    fn mixed_graph_shape_buckets_decode_prefix_down() {
        let plan = BatchPlan {
            kind: BatchKind::Ragged,
            num_tokens: 62,
            batch: 31,
            q_lens: vec![1; 30].into_iter().chain([32]).collect(),
            kv_lens: vec![1; 31],
            seq_positions: vec![0; 31],
            rope_positions: vec![0; 62],
            max_blocks_per_seq: 96,
            block_size: 1,
            total_q_tiles: 31,
        };
        let mut kinds = vec![RaggedRowKind::Decode; 30];
        kinds.push(RaggedRowKind::PrefillFinal);

        let shape =
            mixed_graph_shape(&plan, &kinds, 32, 512, 36, &[1, 2, 4, 8, 16, 24, 32]).unwrap();

        assert_eq!(
            shape,
            MixedGraphShape {
                rows: 32,
                tokens: 64,
                tiles: 32,
                decode_prefix: 24,
            }
        );
    }

    #[test]
    fn mixed_graph_key_ignores_exact_q_lens_with_same_bucket() {
        let a = MixedGraphShape {
            rows: 32,
            tokens: 384,
            tiles: 36,
            decode_prefix: 16,
        };
        let b = MixedGraphShape {
            rows: 32,
            tokens: 384,
            tiles: 36,
            decode_prefix: 16,
        };
        let c = MixedGraphShape {
            rows: 32,
            tokens: 448,
            tiles: 36,
            decode_prefix: 16,
        };

        assert_eq!(mixed_graph_key(a, 2, true), mixed_graph_key(b, 2, true));
        assert_ne!(mixed_graph_key(a, 2, true), mixed_graph_key(c, 2, true));
    }

    #[test]
    fn mixed_graph_warmup_cases_cover_common_buckets() {
        let cases = mixed_graph_warmup_cases(&[1, 2, 4, 8, 16, 24, 32], 32, 512, 512, 10);

        assert_eq!(
            cases,
            vec![
                MixedGraphWarmupCase {
                    decode_prefix: 1,
                    token_bucket: 64,
                    prefill_len: 63,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 2,
                    token_bucket: 64,
                    prefill_len: 62,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 4,
                    token_bucket: 64,
                    prefill_len: 60,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 8,
                    token_bucket: 64,
                    prefill_len: 56,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 16,
                    token_bucket: 64,
                    prefill_len: 48,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 24,
                    token_bucket: 64,
                    prefill_len: 40,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 1,
                    token_bucket: 128,
                    prefill_len: 127,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 2,
                    token_bucket: 128,
                    prefill_len: 126,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 4,
                    token_bucket: 128,
                    prefill_len: 124,
                },
                MixedGraphWarmupCase {
                    decode_prefix: 8,
                    token_bucket: 128,
                    prefill_len: 120,
                },
            ]
        );
        assert!(cases.iter().all(|case| case.decode_prefix < 32));
    }
}
