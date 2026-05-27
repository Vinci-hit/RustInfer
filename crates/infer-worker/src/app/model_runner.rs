//! ModelRunner — orchestrates full inference: load → forward → sample.
//!
//! Application layer:
//! - Domain : LlmModel, PagedKvPool, BatchPlan, ForwardContext
//! - Infra  : Device (Cpu/Cuda), tensor allocation
//!
//! All KV addressing is paged. Single-sequence generation runs as a
//! `batch=1` plan with a contiguous `block_table = [0, 1, 2, ...]`.

use crate::domain::batch::{BatchKind, BatchPlan, PagedKvLayer, PagedKvPool};
use crate::domain::batch_workspace::{BatchWorkspace, WsSeqStep};
use crate::domain::forward_workspace::{ForwardWorkspace, ModelDims};
use crate::domain::model::{ForwardContext, LlmModel};
use crate::domain::ports::{MemoryPort, OpBackend, OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};

#[cfg(feature = "cuda")]
use crate::app::cuda_graph_runner::{CudaGraphRunner, GraphDecision};
#[cfg(feature = "cuda")]
use crate::infra::cuda::{Cuda, kernels::argmax_batched::argmax_batched_decode_into};

/// Per-sequence step description fed by callers (host side). The runner
/// converts a slice of these into a device-resident `BatchPlan`.
#[derive(Debug, Clone)]
pub struct SeqStep {
    /// Tokens this step feeds (prefill: full prompt; decode: 1 token).
    pub input_ids: Vec<i32>,
    /// Absolute RoPE positions for each input token.
    pub positions: Vec<i32>,
    /// First cache row this step writes (== current kv_len BEFORE the step).
    pub kv_write_start: i32,
    /// KV length AFTER this step writes (= kv_write_start + input_ids.len()).
    pub kv_len_after: i32,
    /// Physical block ids for this seq, length must equal
    /// `runner.max_blocks_per_seq`. Unused trailing entries can be 0.
    pub block_table: Vec<u32>,
}

pub struct ModelRunner<T: Dtype, D: OpBackend, M: LlmModel<T, D>> {
    pub model: M,
    pub kv_pool: PagedKvPool<T, D>,
    pub forward_ws: ForwardWorkspace<T, D>,
    pub batch_ws: BatchWorkspace<D>,
    pub device: D,
    pub num_layers: usize,
    pub kv_dim: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,
    pub cap_num_tokens: usize,
    pub cap_batch: usize,
    pub capture_sizes: Vec<usize>,
    /// CUDA Graph runner — populated by `prime_graphs_cuda` (cuda only).
    /// While `None`, `step_batch` always falls back to eager execution.
    #[cfg(feature = "cuda")]
    pub graph_runner: Option<CudaGraphRunner>,

    /// Wall-clock time (ns) spent inside `step_batch_with_graph` (full
    /// wrapper: build_plan + launch + D2H). Includes GPU sync.
    #[cfg(feature = "cuda")]
    pub prof_step_wall_ns: u64,
    /// GPU-side time (ns) of `cudaGraphLaunch` (measured by cudaEvent
    /// pair around the launch; excludes D2H and host bookkeeping).
    #[cfg(feature = "cuda")]
    pub prof_graph_gpu_ns: u64,
    /// Number of decode steps profiled.
    #[cfg(feature = "cuda")]
    pub prof_step_count: u64,
}

impl<T: Dtype, D: OpBackend, M: LlmModel<T, D>> ModelRunner<T, D, M> {
    /// Create a runner. CUDA-Graph capture is **not** done here; call
    /// `prime_graphs_cuda(...)` after construction (CUDA only).
    pub fn new(
        model: M,
        device: D,
        num_blocks: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        cap_num_tokens: usize,
        cap_batch: usize,
        flash_decode_capacity_f32: usize,
        capture_sizes: Vec<usize>,
    ) -> OpResult<Self> {
        let num_layers = model.num_layers();
        let kv_dim = model.kv_dim();
        if block_size * max_blocks_per_seq < max_seq_len {
            return Err(OpError::Shape(format!(
                "ModelRunner::new: block_size({})*max_blocks_per_seq({}) = {} < max_seq_len({})",
                block_size, max_blocks_per_seq,
                block_size * max_blocks_per_seq, max_seq_len,
            )));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = D::alloc_tensor::<T>(
                Shape::from_slice(&[num_blocks, block_size, kv_dim]),
                &device,
            )?;
            let v = D::alloc_tensor::<T>(
                Shape::from_slice(&[num_blocks, block_size, kv_dim]),
                &device,
            )?;
            layers.push(PagedKvLayer { k, v });
        }
        let kv_pool = PagedKvPool { layers, num_blocks, block_size, kv_dim };

        let dims = ModelDims {
            dim: model.dim(),
            q_dim: model.q_dim(),
            kv_dim: model.kv_dim(),
            qkv_dim: model.q_dim() + 2 * model.kv_dim(),
            intermediate_size: model.intermediate_size(),
            vocab_size: model.vocab_size(),
            head_num: model.head_num(),
            head_dim: model.head_dim(),
        };
        let forward_ws = ForwardWorkspace::<T, D>::new(
            &device, dims, cap_num_tokens, cap_batch, flash_decode_capacity_f32,
        )?;
        let batch_ws = BatchWorkspace::<D>::new(
            &device, cap_num_tokens, cap_batch, max_blocks_per_seq,
        )?;

        Ok(Self {
            model,
            kv_pool,
            forward_ws,
            batch_ws,
            device,
            num_layers,
            kv_dim,
            max_seq_len,
            block_size,
            max_blocks_per_seq,
            cap_num_tokens,
            cap_batch,
            capture_sizes,
            #[cfg(feature = "cuda")]
            graph_runner: None,
            #[cfg(feature = "cuda")]
            prof_step_wall_ns: 0,
            #[cfg(feature = "cuda")]
            prof_graph_gpu_ns: 0,
            #[cfg(feature = "cuda")]
            prof_step_count: 0,
        })
    }

    /// Run one forward step over a ragged batch of sequences.
    ///
    /// All plan tensors come from the long-lived `BatchWorkspace`; all
    /// forward intermediates from `ForwardWorkspace`. Zero `cudaMalloc`
    /// per step. Async H2D for plan metadata; one `argmax_batched` D2H
    /// at the end.
    ///
    /// **Eager** path. CUDA users seeking graph-replay should call
    /// `step_batch_with_graph` instead — it auto-dispatches between
    /// graph and eager based on batch shape.
    pub fn step_batch(&mut self, seqs: &[SeqStep]) -> OpResult<Vec<i32>> {
        if seqs.is_empty() {
            return Ok(Vec::new());
        }
        self.step_batch_eager(seqs)
    }

    /// Eager step (no graphs). Always available on every backend.
    fn step_batch_eager(&mut self, seqs: &[SeqStep]) -> OpResult<Vec<i32>> {
        // Adapt SeqStep → WsSeqStep (workspace's own type).
        let ws_seqs: Vec<WsSeqStep> = seqs.iter().map(|s| WsSeqStep {
            input_ids: s.input_ids.clone(),
            positions: s.positions.clone(),
            kv_write_start: s.kv_write_start,
            kv_len_after: s.kv_len_after,
            block_table: s.block_table.clone(),
        }).collect();

        let (input_ids_dev, mut plan) = self.batch_ws.build_plan(&ws_seqs, &self.device)?;
        // Workspace doesn't know the runner's block_size; patch it in.
        plan.block_size = self.block_size;
        let batch = plan.batch;

        // Profiling hook: same shape as the graph path so A/B numbers
        // share metric definitions.
        #[cfg(feature = "cuda")]
        let prof = std::env::var("RUSTINFER_PROFILE_GPU").is_ok();
        #[cfg(feature = "cuda")]
        let wall_t0 = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let mut ev_t0: crate::infra::cuda::ffi::cudaEvent_t = std::ptr::null_mut();
        #[cfg(feature = "cuda")]
        let mut ev_t1: crate::infra::cuda::ffi::cudaEvent_t = std::ptr::null_mut();
        #[cfg(feature = "cuda")]
        if prof {
            // SAFETY: cuda feature gates a Cuda-specific code path; the
            // generic `D` is Cuda here when this branch is taken at the
            // call site. The events are stream-scoped and harmless on
            // non-Cuda builds (this whole block is excluded then).
            // We only care about the case where D is Cuda — see callers.
            unsafe {
                use crate::infra::cuda::ffi as cf;
                cf::cudaEventCreate(&mut ev_t0);
                cf::cudaEventCreate(&mut ev_t1);
                // Stream comes from the output tensor's device handle.
                // For non-Cuda backends this branch is never taken.
            }
        }

        let logits = {
            let mut ctx = ForwardContext {
                kv_pool: &mut self.kv_pool,
                plan: &plan,
                workspace: &mut self.forward_ws,
            };
            self.model.forward(&input_ids_dev, &mut ctx)?
        };
        let result = D::argmax_batched(&logits, &plan.cu_q_lens, batch);

        #[cfg(feature = "cuda")]
        if prof {
            // Best-effort: capture wall-clock since model.forward + argmax
            // includes its own stream sync inside argmax_batched.
            self.prof_step_wall_ns += wall_t0.elapsed().as_nanos() as u64;
            // We don't have a clean GPU-only timing here without threading
            // events through every kernel; report 0 to signal "eager
            // path (graph not used)". Use the graph path for GPU timing.
            self.prof_step_count += 1;
            unsafe {
                if !ev_t0.is_null() {
                    crate::infra::cuda::ffi::cudaEventDestroy(ev_t0);
                }
                if !ev_t1.is_null() {
                    crate::infra::cuda::ffi::cudaEventDestroy(ev_t1);
                }
            }
        }

        result
    }

    /// Convenience: prefill a single prompt then greedily decode up to
    /// `max_new_tokens` new tokens. Stops early on any token in `eos_ids`.
    /// Uses physical blocks `[0, 1, ..., max_blocks_per_seq - 1]`.
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_ids: &[i32],
    ) -> OpResult<Vec<i32>> {
        let debug = std::env::var("RUSTINFER_DEBUG_LAYERS").is_ok();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let num_prompt = prompt_ids.len();
        if num_prompt == 0 {
            return Err(OpError::Shape("empty prompt".into()));
        }
        let block_table: Vec<u32> = (0..self.max_blocks_per_seq as u32).collect();

        let prefill_seq = SeqStep {
            input_ids: prompt_ids.to_vec(),
            positions: (0..num_prompt as i32).collect(),
            kv_write_start: 0,
            kv_len_after: num_prompt as i32,
            block_table: block_table.clone(),
        };
        if debug {
            eprintln!("[runner] prefill: num_tokens={} kv_len_after={}", num_prompt, num_prompt);
        }
        let mut last = self.step_batch(&[prefill_seq])?[0];
        if debug { eprintln!("[runner] prefill argmax → token {}", last); }
        generated.push(last);
        if eos_ids.contains(&last) {
            return Ok(generated);
        }

        for i in 0..max_new_tokens.saturating_sub(1) {
            let kv_write_start = (num_prompt + i) as i32;
            let kv_len_after = (num_prompt + i + 1) as i32;
            let step = SeqStep {
                input_ids: vec![last],
                positions: vec![kv_write_start],
                kv_write_start,
                kv_len_after,
                block_table: block_table.clone(),
            };
            let new = self.step_batch(&[step])?[0];
            if debug {
                eprintln!(
                    "[runner] decode step {:>2}: in={:>6} pos={} kv_len={} → token {}",
                    i, last, kv_write_start, kv_len_after, new,
                );
            }
            last = new;
            generated.push(last);
            if eos_ids.contains(&last) {
                break;
            }
        }
        Ok(generated)
    }
}

// ─── CUDA-only: graph priming + graph-aware step ───────────────────────────
#[cfg(feature = "cuda")]
impl<T: Dtype, M: LlmModel<T, Cuda>> ModelRunner<T, Cuda, M> {
    /// Capture all decode-only graphs in `capture_sizes`.
    ///
    /// For each `size` (in reverse — largest first for memory-friendly
    /// allocator behaviour):
    ///
    ///   1. Build a dummy decode-only `WsSeqStep` of `size` sequences,
    ///      each with `input_ids=[0]`, `positions=[0]`, kv_write_start=0,
    ///      kv_len_after=1, and a block_table that points entirely at the
    ///      LAST physical block (used as a graph-only scratch block — its
    ///      contents are deliberately discarded between captures).
    ///   2. Run 2 eager warmup forwards to settle cuBLAS/cuDNN algos.
    ///   3. Capture forward + argmax_batched_decode_into into the graph.
    ///
    /// After this returns, `step_batch_with_graph` will route any
    /// decode-only step with `batch ≤ max_capture_size` through the
    /// captured graph instead of eager kernels.
    ///
    /// **NOTE**: this assumes the LAST physical block (id `num_blocks-1`)
    /// is reserved by the runner as a graph scratch block — production
    /// allocations must avoid it.
    pub fn prime_graphs_cuda(&mut self) -> OpResult<()> {
        if self.capture_sizes.is_empty() {
            return Ok(());
        }
        // Drop sizes exceeding the batch capacity — those would overflow
        // `BatchWorkspace::build_plan` during capture.
        let usable_sizes: Vec<usize> = self
            .capture_sizes
            .iter()
            .copied()
            .filter(|&s| s <= self.cap_batch)
            .collect();
        if usable_sizes.is_empty() {
            eprintln!(
                "[graph] cap_batch={} too small for any capture size in {:?}; skipping",
                self.cap_batch, self.capture_sizes,
            );
            return Ok(());
        }
        if usable_sizes.len() != self.capture_sizes.len() {
            eprintln!(
                "[graph] capping capture sizes to {:?} (cap_batch={})",
                usable_sizes, self.cap_batch,
            );
        }
        let scratch_block = (self.kv_pool.num_blocks - 1) as u32;
        let block_table: Vec<u32> = vec![scratch_block; self.max_blocks_per_seq];

        let mut graph_runner = CudaGraphRunner::new(usable_sizes.clone());

        // Build the runner exactly once; wrap in Option so we can `take`
        // around the closure (needed to satisfy borrow checker — the
        // closure borrows `self` mutably, so the runner can't sit in
        // `self.graph_runner` while we're inside.

        // Block_table is the same for all dummy seqs; produce SeqSteps
        // for the maximum capture size, slice for smaller ones.
        let max_size = *usable_sizes.last().unwrap();
        let dummy_steps: Vec<SeqStep> = (0..max_size).map(|_| SeqStep {
            input_ids: vec![0],
            positions: vec![0],
            kv_write_start: 0,
            kv_len_after: 1,
            block_table: block_table.clone(),
        }).collect();

        // The capture loop needs `&CudaConfig`. The runner's stream lives
        // inside `Arc<CudaConfig>`, so a cheap clone gives us a handle
        // independent of `self` and avoids aliasing during the closure.
        let cuda_config = self.device.config.clone();

        graph_runner.warmup_and_capture_all(
            &*cuda_config,
            2,
            |size, is_capture| {
                if is_capture {
                    // Capture pass: forward + argmax ONLY (no H2D memcpy).
                    // Device buffers already hold valid data from warmup.
                    self.run_decode_forward_only(size)
                } else {
                    // Warmup pass: full path including H2D upload.
                    self.run_decode_only_step(&dummy_steps[..size])
                }
            },
        )?;

        self.graph_runner = Some(graph_runner);
        Ok(())
    }

    /// Run a single decode-only forward into `forward_ws.argmax_out_dev`.
    /// Includes build_plan (H2D upload). Used for warmup passes and eager fallback.
    fn run_decode_only_step(&mut self, seqs: &[SeqStep]) -> OpResult<()> {
        let ws_seqs: Vec<WsSeqStep> = seqs.iter().map(|s| WsSeqStep {
            input_ids: s.input_ids.clone(),
            positions: s.positions.clone(),
            kv_write_start: s.kv_write_start,
            kv_len_after: s.kv_len_after,
            block_table: s.block_table.clone(),
        }).collect();

        let (input_ids_dev, mut plan) = self.batch_ws.build_plan(&ws_seqs, &self.device)?;
        plan.block_size = self.block_size;

        let logits = {
            let mut ctx = ForwardContext {
                kv_pool: &mut self.kv_pool,
                plan: &plan,
                workspace: &mut self.forward_ws,
            };
            self.model.forward(&input_ids_dev, &mut ctx)?
        };
        // Decode-only: logits is [batch, vocab]. Use the graph-friendly
        // argmax (zero alloc, zero D2H, writes into forward_ws.argmax_out_dev).
        argmax_batched_decode_into(&logits, self.forward_ws.argmax_out_dev_mut())
    }

    /// Forward + argmax ONLY — no H2D upload.
    ///
    /// Used during CUDA Graph capture: device buffers already hold valid
    /// data from the preceding warmup pass. By skipping `build_plan`'s
    /// `upload_async` calls, we keep cudaMemcpyAsync operations OUT of
    /// the captured graph. The graph will contain only kernel launches.
    fn run_decode_forward_only(&mut self, batch_size: usize) -> OpResult<()> {
        let (input_ids_dev, mut plan) =
            self.batch_ws.get_last_plan_views(batch_size, self.block_size)?;
        plan.block_size = self.block_size;

        let logits = {
            let mut ctx = ForwardContext {
                kv_pool: &mut self.kv_pool,
                plan: &plan,
                workspace: &mut self.forward_ws,
            };
            self.model.forward(&input_ids_dev, &mut ctx)?
        };
        argmax_batched_decode_into(&logits, self.forward_ws.argmax_out_dev_mut())
    }

    /// Decode-only graph-aware step.
    ///
    /// - If every seq has q_len=1 AND batch ≤ max_capture_size AND graphs
    ///   are primed: pad up to the next captured size (extra rows point at
    ///   the scratch block + position 0), launch the graph, D2H-read the
    ///   first `batch` argmax outputs, return.
    /// - Otherwise: fall back to `step_batch_eager` (which does its own
    ///   D2H-sync inside `argmax_batched`).
    pub fn step_batch_with_graph(&mut self, seqs: &[SeqStep]) -> OpResult<Vec<i32>> {
        if seqs.is_empty() {
            return Ok(Vec::new());
        }
        // Escape hatch for A/B benchmarking against eager.
        if std::env::var("RUSTINFER_DISABLE_GRAPH").is_ok() {
            return self.step_batch_eager(seqs);
        }
        let all_decode = seqs.iter().all(|s| s.input_ids.len() == 1);
        let batch = seqs.len();
        let primed = self.graph_runner.is_some();
        let max_cap = self.graph_runner.as_ref().map(|g| g.max_capture_size()).unwrap_or(0);

        if !primed || !all_decode || batch > max_cap {
            return self.step_batch_eager(seqs);
        }

        // Pick the next captured size ≥ batch.
        let decision = self.graph_runner.as_ref().unwrap().decide(batch);
        let (slot, padded_size) = match decision {
            GraphDecision::Replay { slot, padded_size, .. } => (slot, padded_size),
            GraphDecision::Eager => return self.step_batch_eager(seqs),
        };

        // Pad up to padded_size with scratch-block dummy seqs.
        let scratch_block = (self.kv_pool.num_blocks - 1) as u32;
        let pad_block_table: Vec<u32> = vec![scratch_block; self.max_blocks_per_seq];
        let mut padded: Vec<SeqStep> = seqs.to_vec();
        for _ in batch..padded_size {
            padded.push(SeqStep {
                input_ids: vec![0],
                positions: vec![0],
                kv_write_start: 0,
                kv_len_after: 1,
                block_table: pad_block_table.clone(),
            });
        }

        // 1. Async-upload the (padded) plan into batch_ws.
        let ws_seqs: Vec<WsSeqStep> = padded.iter().map(|s| WsSeqStep {
            input_ids: s.input_ids.clone(),
            positions: s.positions.clone(),
            kv_write_start: s.kv_write_start,
            kv_len_after: s.kv_len_after,
            block_table: s.block_table.clone(),
        }).collect();
        let _ = self.batch_ws.build_plan(&ws_seqs, &self.device)?;

        // 2. Launch the captured graph.
        if std::env::var("RUSTINFER_TRACE_GRAPH").is_ok() {
            eprintln!("[graph] replay slot={:?} batch={}->{}", slot, batch, padded_size);
        }

        // Profiling: enable with RUSTINFER_PROFILE_GPU=1. We wrap the
        // graph launch with a cudaEvent pair to measure pure GPU time.
        // The wall-clock around the whole step_batch_with_graph call is
        // measured outside the launch (build_plan + D2H included).
        let prof = std::env::var("RUSTINFER_PROFILE_GPU").is_ok();
        let wall_t0 = std::time::Instant::now();
        let mut ev_t0: crate::infra::cuda::ffi::cudaEvent_t = std::ptr::null_mut();
        let mut ev_t1: crate::infra::cuda::ffi::cudaEvent_t = std::ptr::null_mut();
        if prof {
            unsafe {
                crate::infra::cuda::ffi::cudaEventCreate(&mut ev_t0);
                crate::infra::cuda::ffi::cudaEventCreate(&mut ev_t1);
                crate::infra::cuda::ffi::cudaEventRecord(ev_t0, self.device.config.stream);
            }
        }
        self.device.config.launch(slot)?;
        if prof {
            unsafe {
                crate::infra::cuda::ffi::cudaEventRecord(ev_t1, self.device.config.stream);
            }
        }

        // 3. Synchronous D2H of the argmax_out_dev (just `padded_size` ints).
        // We only return the first `batch` of them.
        let host = self.forward_ws.argmax_out_dev().to_host_vec()?;

        if prof {
            unsafe {
                crate::infra::cuda::ffi::cudaEventSynchronize(ev_t1);
                let mut ms: f32 = 0.0;
                crate::infra::cuda::ffi::cudaEventElapsedTime(&mut ms, ev_t0, ev_t1);
                self.prof_graph_gpu_ns += (ms as f64 * 1.0e6) as u64;
                crate::infra::cuda::ffi::cudaEventDestroy(ev_t0);
                crate::infra::cuda::ffi::cudaEventDestroy(ev_t1);
            }
            self.prof_step_wall_ns += wall_t0.elapsed().as_nanos() as u64;
            self.prof_step_count += 1;
        }
        Ok(host.into_iter().take(batch).collect())
    }

    /// Same shape as `generate`, but routes decode steps through
    /// `step_batch_with_graph` so primed CUDA graphs are used.
    ///
    /// Prefill (multi-token) always goes through eager — it's never
    /// decode-only by definition.
    pub fn generate_with_graph(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_ids: &[i32],
    ) -> OpResult<Vec<i32>> {
        let debug = std::env::var("RUSTINFER_DEBUG_LAYERS").is_ok();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let num_prompt = prompt_ids.len();
        if num_prompt == 0 {
            return Err(OpError::Shape("empty prompt".into()));
        }
        let block_table: Vec<u32> = (0..self.max_blocks_per_seq as u32).collect();

        let prefill_seq = SeqStep {
            input_ids: prompt_ids.to_vec(),
            positions: (0..num_prompt as i32).collect(),
            kv_write_start: 0,
            kv_len_after: num_prompt as i32,
            block_table: block_table.clone(),
        };
        // Prefill is multi-token → eager.
        let mut last = self.step_batch_eager(&[prefill_seq])?[0];
        if debug { eprintln!("[runner] prefill argmax → token {}", last); }
        generated.push(last);
        if eos_ids.contains(&last) {
            return Ok(generated);
        }

        for i in 0..max_new_tokens.saturating_sub(1) {
            let kv_write_start = (num_prompt + i) as i32;
            let kv_len_after = (num_prompt + i + 1) as i32;
            let step = SeqStep {
                input_ids: vec![last],
                positions: vec![kv_write_start],
                kv_write_start,
                kv_len_after,
                block_table: block_table.clone(),
            };
            // Decode → graph (auto-falls-back to eager if not primed).
            let new = self.step_batch_with_graph(&[step])?[0];
            if debug {
                eprintln!(
                    "[runner] graph-decode {:>2}: in={:>6} pos={} kv_len={} → token {}",
                    i, last, kv_write_start, kv_len_after, new,
                );
            }
            last = new;
            generated.push(last);
            if eos_ids.contains(&last) {
                break;
            }
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::cpu::Cpu;
    use crate::models::layers::{Linear, RMSNorm, Embedding};
    use crate::models::llama3::{Llama3Model, Llama3Layer};

    fn tiny_llama3() -> Llama3Model<f32, Cpu> {
        let dim = 16;
        let head_num = 2;
        let kv_head_num = 2;
        let head_dim = 8;
        let intermediate = 32;
        let vocab = 64;
        let q_dim = head_num * head_dim;
        let kv_dim = kv_head_num * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let max_seq = 32;

        use crate::domain::tensor::Tensor;

        let make_weight = |rows: usize, cols: usize| -> Tensor<f32, Cpu> {
            let data: Vec<f32> = (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
            Tensor::<f32, Cpu>::from_slice(&data, [rows, cols])
        };
        let make_norm = |dim: usize| -> Tensor<f32, Cpu> {
            Tensor::<f32, Cpu>::from_slice(&vec![1.0f32; dim], [dim])
        };

        let layer = Llama3Layer {
            input_layernorm: RMSNorm::new(make_norm(dim), 1e-5),
            post_attention_layernorm: RMSNorm::new(make_norm(dim), 1e-5),
            qkv_proj: Linear::new(make_weight(qkv_dim, dim), None),
            o_proj: Linear::new(make_weight(dim, q_dim), None),
            gate_up_proj: Linear::new(make_weight(2 * intermediate, dim), None),
            down_proj: Linear::new(make_weight(dim, intermediate), None),
        };

        let sin_cache = Tensor::<f32, Cpu>::zeros_cpu([max_seq, head_dim]);
        let cos_cache = {
            let data = vec![1.0f32; max_seq * head_dim];
            Tensor::<f32, Cpu>::from_slice(&data, [max_seq, head_dim])
        };

        Llama3Model {
            embed_tokens: Embedding { table: make_weight(vocab, dim) },
            layers: vec![layer],
            norm: RMSNorm::new(make_norm(dim), 1e-5),
            lm_head: Linear::new(make_weight(vocab, dim), None),
            sin_cache,
            cos_cache,
            head_num,
            kv_head_num,
            head_dim,
            dim,
            kv_dim,
            intermediate_size: intermediate,
            vocab_size: vocab,
        }
    }

    #[test]
    fn e2e_cpu_forward_and_argmax() {
        // 4 blocks × 8 tokens = 32 tokens capacity, 1 seq.
        let model = tiny_llama3();
        let mut runner = ModelRunner::new(model, Cpu, 4, 8, 4, 32, 4, 1, 1, vec![]).unwrap();
        let prompt = &[1i32, 5, 10];
        let tokens = runner.generate(prompt, 3, &[]).unwrap();
        assert_eq!(tokens.len(), 3);
        for &t in &tokens {
            assert!(t >= 0 && t < 64, "token {} out of vocab range", t);
        }
    }

    /// Ragged batch (CPU naive): 2 sequences, different prompt lengths,
    /// each routed to its own physical blocks. Must match the per-seq
    /// serial reference's first decode token.
    #[test]
    fn ragged_batch_matches_serial() {
        // Pool: 8 physical blocks × 4 tokens = 32 tokens; max 4 blocks/seq.
        let model = tiny_llama3();
        let mut runner_batch = ModelRunner::new(model, Cpu, 8, 4, 4, 16, 8, 2, 1, vec![]).unwrap();

        let p0: Vec<i32> = vec![1, 2, 3, 4, 5];
        let p1: Vec<i32> = vec![10, 20, 30];

        // Seq 0 uses blocks [0,1,2,3]; seq 1 uses blocks [4,5,6,7].
        let bt0: Vec<u32> = vec![0, 1, 2, 3];
        let bt1: Vec<u32> = vec![4, 5, 6, 7];

        let batched = vec![
            SeqStep {
                input_ids: p0.clone(),
                positions: (0..p0.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: p0.len() as i32,
                block_table: bt0,
            },
            SeqStep {
                input_ids: p1.clone(),
                positions: (0..p1.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: p1.len() as i32,
                block_table: bt1,
            },
        ];
        let batched_first = runner_batch.step_batch(&batched).unwrap();
        assert_eq!(batched_first.len(), 2);

        // Reference: serial per-prompt runners, each with its own pool.
        let model_ref0 = tiny_llama3();
        let mut runner_ref0 = ModelRunner::new(model_ref0, Cpu, 4, 4, 4, 16, 8, 1, 1, vec![]).unwrap();
        let r0 = runner_ref0.generate(&p0, 1, &[]).unwrap();

        let model_ref1 = tiny_llama3();
        let mut runner_ref1 = ModelRunner::new(model_ref1, Cpu, 4, 4, 4, 16, 8, 1, 1, vec![]).unwrap();
        let r1 = runner_ref1.generate(&p1, 1, &[]).unwrap();

        assert_eq!(batched_first[0], r0[0],
            "ragged batch seq 0 first-token mismatch: batch={} serial={}", batched_first[0], r0[0]);
        assert_eq!(batched_first[1], r1[0],
            "ragged batch seq 1 first-token mismatch: batch={} serial={}", batched_first[1], r1[0]);
    }
}
