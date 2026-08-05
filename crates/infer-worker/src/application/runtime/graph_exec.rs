//! CUDA-graph execution: the capture/replay lifecycles for decode and
//! single-seq prefill graphs, plus the `GraphRunner` replay policy.

use std::marker::PhantomData;

use crate::domain::dtype::Dtype;
use crate::domain::exec::ExecScope;
use crate::domain::model::DecoderModel;
use crate::domain::plan::{BatchKind, SeqStep, StepOutput, StepRequest, StopCriteria};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::types::Shape;

use super::{Runtime, upload_i32_prefix};

/// Tag bit OR'd into prefill-graph keys (keyed by `num_tokens`) so they never
/// collide with decode-graph keys (keyed by `batch` ≤ `cap_batch`) in the
/// scope's graph map. `1 << 40` is far above any realistic token count.
const PREFILL_GRAPH_KEY_TAG: u64 = 1 << 40;
/// Max distinct single-seq prefill lengths to capture graphs for (Stage A keys
/// each prefill graph by exact `num_tokens`). Bounds graph memory; beyond this,
/// uncaptured prefill lengths run eager.
const PREFILL_GRAPH_BUDGET: usize = 16;
/// Default max prompt length (tokens) eligible for a single-seq prefill graph.
/// Set to 0 (disabled): the eager prefill path now routes its bf16 GEMMs to the
/// `algo=nullptr` cuBLASLt runtime-selected kernel (532f5c's path), which is
/// ~2ms/forward FASTER than a captured graph replaying a capturable-but-slower
/// cached algo. The graph machinery is retained (Stage B) but off by default;
/// raise this to re-enable single-seq prefill graphs.
const PREFILL_GRAPH_MAX_TOKENS: usize = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphDecision {
    Graph(GraphSlotId),
    /// Single-seq prefill graph, keyed by exact `num_tokens` (the captured
    /// region is `run_layers`; `sample_tail` stays eager).
    PrefillGraph(usize),
    Eager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphSlotId(pub usize);

pub struct GraphRunner<D: LlmBackend> {
    capture_sizes: Vec<usize>,
    /// Max prompt length (tokens) eligible for a single-seq prefill graph.
    /// `0` disables prefill graphs (decode graphs only).
    prefill_graph_max: usize,
    _d: PhantomData<D>,
}

impl<D: LlmBackend> Default for GraphRunner<D> {
    fn default() -> Self {
        Self {
            capture_sizes: Vec::new(),
            prefill_graph_max: 0,
            _d: PhantomData,
        }
    }
}

impl<D: LlmBackend> GraphRunner<D> {
    pub fn new(
        mut capture_sizes: Vec<usize>,
        cap_batch: usize,
        prefill_graph_max: usize,
    ) -> OpResult<Self> {
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
        if let Some(&max_size) = capture_sizes.last()
            && max_size > cap_batch
        {
            return Err(OpError::Shape(format!(
                "GraphRunner::new: capture size {} > cap_batch {}",
                max_size, cap_batch
            )));
        }
        Ok(Self {
            capture_sizes,
            prefill_graph_max,
            _d: PhantomData,
        })
    }

    pub fn capture_sizes(&self) -> &[usize] {
        &self.capture_sizes
    }

    pub fn decide(&self, plan: &crate::domain::plan::BatchPlan) -> GraphDecision {
        if plan.is_decode_only() {
            return self
                .slot_for_batch(plan.batch)
                .map_or(GraphDecision::Eager, GraphDecision::Graph);
        }
        // Single-seq prefill graph (Stage A): exact-`num_tokens` key. Only plain
        // ragged prefill of one sequence is eligible — bursts (batch > 1) and
        // speculative/spec masks still run eager (Stage B generalizes via
        // bucketing + dummy-tail padding).
        if self.prefill_graph_max > 0
            && plan.batch == 1
            && plan.num_tokens >= 2
            && plan.num_tokens <= self.prefill_graph_max
            && matches!(plan.kind, BatchKind::Ragged)
        {
            return GraphDecision::PrefillGraph(plan.num_tokens);
        }
        GraphDecision::Eager
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

impl<T, D, M> Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    pub fn decide(&self, plan: &crate::domain::plan::BatchPlan) -> GraphDecision {
        self.graph
            .as_ref()
            .map_or(GraphDecision::Eager, |graph| graph.decide(plan))
    }

    pub fn prime_graphs(&mut self) -> OpResult<()> {
        let pending = self.dispatch_peer_command(super::RuntimePeerCommand::PrimeGraphs)?;
        let local = self.prime_graphs_local();
        let followers = self.wait_peer_command(pending);
        super::complete_replicated(&mut self.peers, "prime_graphs", local, followers)
    }

    fn prime_graphs_local(&mut self) -> OpResult<()> {
        self.graph = None;
        if self.capture_sizes.is_empty() || !self.scope.supports_graphs() {
            return Ok(());
        }
        self.graph = Some(GraphRunner::new(
            self.capture_sizes.clone(),
            self.cap_batch,
            PREFILL_GRAPH_MAX_TOKENS,
        )?);
        Ok(())
    }

    /// Eagerly capture a decode CUDA graph for every configured capture size,
    /// so the first *live* decode at each batch size replays a ready graph
    /// instead of paying an inline `forward + capture` on the serve thread.
    ///
    /// Without this, each size is traced lazily on its first hit during
    /// serving (`step_graph` cold path): that step runs a full eager forward,
    /// a `synchronize`, then a capture pass — blocking the serve loop, and with
    /// it every prefill queued behind it. A cold-vs-warm QPS sweep pins this as
    /// the dominant source of TTFT/TPOT tail spikes (p99 TTFT ~235–471ms cold
    /// vs ~25–45ms once all graphs are cached). Running every capture here, at
    /// bootstrap before `Ready`, moves that one-time cost off the hot path.
    ///
    /// Each warmup step is a synthetic 1-token decode against scratch KV blocks
    /// `0..batch`. The graph bakes only the batch dimension; KV positions /
    /// lengths are read from address-stable control buffers at replay, so a
    /// capture traced at `kv_len = 1` replays correctly at any real length.
    /// The scratch blocks hold throwaway state: this runs before any request is
    /// admitted, and real sequences allocate from the free list and overwrite
    /// whatever lands there. The decode kernels address the KV pool by raw
    /// block id, so no allocator interaction or ownership check is needed.
    /// Smallest decode capture size `>= batch` (and `<= cap_batch`), or `None` if
    /// `batch` is 0 or exceeds the largest capture size. The fused path pads its
    /// cuDNN decode-prefix up to this slot so the prewarmed (per-capture-size)
    /// cuDNN SDPA plan is reused instead of paying a ~370ms HEURISTICS_CHOICE
    /// build on every novel decode-row count. See [[eager-fused-mixed-tail-regression]].
    pub fn next_capture_slot(&self, batch: usize) -> Option<usize> {
        if batch == 0 {
            return None;
        }
        self.capture_sizes
            .iter()
            .copied()
            .filter(|&s| s >= batch && s <= self.cap_batch)
            .min()
    }

    pub fn prewarm_decode_graphs(&mut self) -> OpResult<()> {
        if self.graph.is_none() {
            return Ok(());
        }
        for batch in self.capture_sizes.clone() {
            if batch == 0 || batch > self.cap_batch {
                continue;
            }
            let seqs: Vec<SeqStep> = (0..batch)
                .map(|i| SeqStep {
                    sequence_id: i as u64,
                    input_ids: vec![1],
                    positions: vec![0],
                    kv_write_start: 0,
                    kv_len_after: 1,
                    block_table: vec![i as u32],
                })
                .collect();
            let req = StepRequest {
                seqs,
                sampling: vec![Default::default(); batch],
                stop: StopCriteria {
                    eos_ids: Vec::new(),
                    generated_counts: vec![0; batch],
                    max_tokens: vec![u32::MAX; batch],
                    ignore_eos: vec![true; batch],
                },
                draft_tokens: Vec::new(),
            };
            // Drives the `step_graph` cold path for this exact size → capture.
            self.step(&req)?;
            if self.scope.topology().tp.size > 1 {
                // NCCL treats graph launch as a collective. The mirrored cold
                // Step above must finish capture/instantiate on every rank
                // before any rank launches the graph. Returning from that peer
                // command is the group barrier; this second Step validates and
                // warms the synchronized replay before the worker becomes Ready.
                self.step(&req)?;
            }
        }
        self.scope.synchronize()?;
        Ok(())
    }

    /// Warm the prefill path at a grid of prompt lengths, so the first *live*
    /// prefill of a given length does not pay one-time per-shape costs inline on
    /// the serve thread (and stall any decode it is interleaved with).
    ///
    /// Decode-graph prewarm (`prewarm_decode_graphs`) fixes the decode side, but
    /// a cold-vs-warm sweep shows a residual TTFT p99 tail (~235-365ms cold,
    /// ~40ms once warm) that tracks first-encounter prefill shapes: CUDA
    /// caching-allocator growth as new token counts allocate new workspace
    /// sizes (inline `cudaMalloc`), `lm_head` first-touch, and any shape-keyed
    /// library state the eager prefill path builds. Running a synthetic prefill
    /// per grid length here pays all of that at bootstrap instead.
    ///
    /// Each warmup is a single synthetic sequence of `len` tokens routed through
    /// the real `step()` prefill path (`num_tokens > batch`), so it exercises the
    /// exact GEMM mode, ragged attention, and sampling the live path uses. The
    /// allocator rounds sizes to bins, so a coarse grid covers nearby lengths;
    /// scratch KV blocks `0..len` are throwaway (overwritten by real requests).
    pub fn prewarm_prefill_shapes(&mut self, lengths: &[usize]) -> OpResult<()> {
        let max_len = self
            .max_seq_len
            .min(self.cap_num_tokens)
            .min(self.max_blocks_per_seq);
        for &len in lengths {
            if len == 0 || len > max_len {
                continue;
            }
            let req = StepRequest {
                seqs: vec![SeqStep {
                    sequence_id: 0,
                    input_ids: vec![1; len],
                    positions: (0..len as i32).collect(),
                    kv_write_start: 0,
                    kv_len_after: len as i32,
                    block_table: (0..len as u32).collect(),
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
            if self.scope.topology().tp.size > 1 && self.scope.supports_graphs() {
                // Separate first replay from capture for the same reason as
                // decode graph prewarm: a captured NCCL sequence may launch
                // only after every rank has finished graph instantiation.
                self.step(&req)?;
            }
        }
        self.scope.synchronize()?;
        Ok(())
    }

    pub(super) fn step_graph(
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
        // Pad the live decode batch UP to the captured slot: `slot_for_batch`
        // rounds `plan.batch` up to the smallest capture size >= batch, and
        // replaying that (>= batch) graph is correct — the zero-padded control
        // tail rows [batch, slot) are inert (q_len=0/kv_len=0 ⇒ no KV scatter,
        // no attention read) and their forward output in C[batch..slot] is never
        // read (`decode_output_from_c` reads only the real `plan.batch` rows).
        // Requires a pure decode step (num_tokens == batch); ragged/spec steps
        // (num_tokens > batch) still run eager. See `issue_decode_abc` for the
        // same fix on the hot ABC decode path.
        if plan.num_tokens != plan.batch || slot_batch < plan.batch {
            return self.step_eager(plan, req);
        }
        let key = slot_batch as u64;

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
        } else if slot_batch == plan.batch {
            // Cold path, exact shape. First run one EAGER forward at this exact
            // shape so the libraries that lazily plan/benchmark on a cold shape
            // (cuDNN SDPA plan cache, cuBLASLt algo selection) populate their
            // shape-keyed caches — those code paths do mallocs/private-stream
            // launches that are illegal under stream capture. This eager pass
            // also produces a correct result; the KV scatter it performs writes
            // the same values at the same paged positions the replay will, so
            // the immediately following capture+launch is idempotent on KV state.
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
            tracing::info!(
                "[graph] captured decode graph (forward+argmax) for batch={}",
                plan.batch
            );
            if self.scope.topology().tp.size == 1 {
                self.scope.graph_launch(key)?;
            }
        } else {
            // Padded slot whose graph is not yet captured (boot prewarm off):
            // can't capture a `slot_batch` graph from a `batch`-sized plan, so
            // fall back to a correct eager step at the real batch.
            return self.step_eager(plan, req);
        }
        // The graph already produced the per-row argmax in buffer C; just read
        // it back (no eager finalize/sample). The decode graph path is always
        // greedy q_len=1 (speculative q_len>1 fell back to eager above).
        // `decode_output_from_c` does a `to_host_vec()` on C, which itself
        // syncs the compute stream — so the extra `scope.synchronize()` that
        // used to live here was redundant on the warm path and added ~1
        // round-trip of latency per decode step. On the cold path the
        // synchronize before capture is still required (and already issued
        // above before `graph_capture_begin`).
        self.decode_output_from_c(plan, req)
    }

    /// Single-sequence prefill via CUDA graph (Stage A). The captured region is
    /// `run_layers` (embed + decoder layers → the persistent `hidden` buffer) —
    /// the launch-heavy ~Nlayers×Nkernels chain whose per-launch CPU dispatch
    /// dominates a short-prompt prefill (~8ms for a 6-token prompt is almost
    /// entirely launch/setup, not compute). `sample_tail` stays EAGER: the
    /// finalize (`LastPerSeq`) + argmax + the small data-dependent D2H run
    /// outside the graph, exactly as for an eager prefill.
    ///
    /// Keyed by exact `num_tokens` (tagged so it can't collide with a decode
    /// graph keyed by `batch`). Replay reuses the address-stable control buffers
    /// (`upload_index`, already run in `step`) and the fixed `prefill_ids_buf`,
    /// so a later same-length prefill with different blocks/positions replays
    /// correctly — the kernels read block tables / kv lens at runtime.
    ///
    /// IMPORTANT: this path does NOT enable `prefill_gemm_mode`. The eager
    /// `(N,K)` GEMM cache is capture-illegal (it builds/probes lazily). With the
    /// flag off, the cold warmup pass below builds the capturable per-`(M,N,K)`
    /// cuBLASLt cache for `M == num_tokens`, which the capture then replays.
    pub(super) fn step_prefill_graph(
        &mut self,
        num_tokens: usize,
        plan: &crate::domain::plan::BatchPlan,
        req: &StepRequest,
    ) -> OpResult<StepOutput> {
        // Fall back to eager if graphs are unavailable on this scope.
        if self.graph.is_none() || !self.scope.supports_graphs() {
            return self.step_eager(plan, req);
        }
        let key = PREFILL_GRAPH_KEY_TAG | num_tokens as u64;
        // Fixed-address input ids (uploaded into the persistent prefill buffer).
        let input_ids = self.input_ids_tensor(req, plan)?;

        if self.scope.graph_ready(key) {
            // Hot path: pure replay. `upload_index` + the id upload above already
            // rewrote every input buffer the graph reads.
            self.scope.graph_launch(key)?;
        } else if self.prefill_graphs_captured >= PREFILL_GRAPH_BUDGET {
            // Graph budget spent: run eager (still correct, just not captured).
            self.run_layers(plan, &input_ids)?;
            return self.sample_tail(plan, req);
        } else {
            // Cold path (mirrors the decode cold path). One eager `run_layers`
            // first warms the lazily-built, capture-illegal library caches
            // (cuBLASLt per-`(M,N,K)` algo selection / capturability probe) and
            // writes the seq's KV at its paged positions. Then capture the
            // (now warm) `run_layers` and replay it — the replay re-scatters the
            // same KV at the same positions, so it is idempotent on KV state.
            self.run_layers(plan, &input_ids)?;
            self.scope.synchronize()?;
            self.scope.graph_capture_begin()?;
            if let Err(e) = self.run_layers(plan, &input_ids) {
                let _ = self.scope.graph_capture_end(key);
                return Err(e);
            }
            self.scope.graph_capture_end(key)?;
            self.prefill_graphs_captured += 1;
            tracing::info!(
                "[graph] captured prefill graph (run_layers) for num_tokens={} ({}/{} budget)",
                num_tokens,
                self.prefill_graphs_captured,
                PREFILL_GRAPH_BUDGET
            );
            if self.scope.topology().tp.size == 1 {
                self.scope.graph_launch(key)?;
            }
        }
        // `hidden` now holds the replayed forward output; finalize + sample eager.
        self.sample_tail(plan, req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::plan::BatchPlan;
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn graph_runner_picks_smallest_decode_slot() {
        let runner = GraphRunner::<Cpu>::new(vec![4, 1, 2, 2], 4, 0).unwrap();
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
}
