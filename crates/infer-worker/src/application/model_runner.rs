//! ModelRunner — orchestrates full inference: load → forward → sample.
//!
//! Application layer:
//! - Domain : LlmModel, PagedKvPool, BatchPlan, ForwardContext
//! - Infra  : Device (Cpu/Cuda), tensor allocation
//!
//! All KV addressing is paged. Single-sequence generation runs as a
//! `batch=1` plan with a contiguous `block_table = [0, 1, 2, ...]`.

use crate::application::batch_workspace::BatchWorkspace;
use crate::application::forward_workspace::{ForwardWorkspace, ModelDims};
use crate::domain::batch::{PagedKvLayer, PagedKvPool};
use crate::domain::model::{ForwardContext, LlmModel};
use crate::domain::ports::{OpBackend, OpError, OpResult};

use crate::domain::types::{Dtype, Shape};

#[cfg(feature = "cuda")]
mod cuda_decode;

#[cfg(feature = "cuda")]
pub use cuda_decode::DecodeCompactOutput;

#[cfg(feature = "cuda")]
use crate::application::cuda_graph_runner::CudaGraphRunner;

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
    /// Physical block ids for this seq. The row must cover every block
    /// touched by `kv_len_after`; unused trailing entries are optional.
    pub block_table: Vec<u32>,
}

pub struct ModelRunner<T: Dtype, D: OpBackend, M: LlmModel<T, D, ForwardWorkspace<T, D>>> {
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
    /// True after this runner has recorded at least one copy-out event on
    /// CUDA's decode copy-out stream.
    #[cfg(feature = "cuda")]
    pub decode_copy_out_recorded: bool,
}

impl<T: Dtype, D: OpBackend, M: LlmModel<T, D, ForwardWorkspace<T, D>>> ModelRunner<T, D, M> {
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
        if block_size == 0 {
            return Err(OpError::Shape("ModelRunner::new: block_size=0".into()));
        }
        if max_blocks_per_seq == 0 {
            return Err(OpError::Shape(
                "ModelRunner::new: max_blocks_per_seq=0".into(),
            ));
        }
        if max_seq_len == 0 {
            return Err(OpError::Shape("ModelRunner::new: max_seq_len=0".into()));
        }
        if block_size.saturating_mul(max_blocks_per_seq) < max_seq_len {
            return Err(OpError::Shape(format!(
                "ModelRunner::new: block_size({})*max_blocks_per_seq({}) = {} < max_seq_len({})",
                block_size,
                max_blocks_per_seq,
                block_size.saturating_mul(max_blocks_per_seq),
                max_seq_len,
            )));
        }
        if block_size != 1 && device.name() == "cuda" {
            // The worker-owned `GlobalKvAllocator` (Phase 1) and the
            // RadixTree handle (Phase 5) both rely on `block_table[seq][i]`
            // being the i-th token's global KV index, which only holds when
            // block_size == 1. Other values still *function* at the kernel
            // level (`pos / block_size`) — CPU tests exercise paging with
            // block_size > 1 — but the worker-owned KV-reuse / release
            // invariants break silently on the production Cuda path (M9).
            // Reject there rather than limp on with corrupt KV bookkeeping.
            return Err(OpError::Shape(format!(
                "ModelRunner::new: block_size={} != 1 is unsupported on the worker-owned (cuda) KV path",
                block_size,
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
        let kv_pool = PagedKvPool {
            layers,
            num_blocks,
            block_size,
            kv_dim,
        };

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
            &device,
            dims,
            cap_num_tokens,
            cap_batch,
            flash_decode_capacity_f32,
        )?;
        let batch_ws =
            BatchWorkspace::<D>::new(&device, cap_num_tokens, cap_batch, max_blocks_per_seq)?;

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
            #[cfg(feature = "cuda")]
            decode_copy_out_recorded: false,
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
        self.validate_steps(seqs)?;

        // `build_plan` consumes `SeqStep` directly (H1): no per-seq adapter
        // clone of input_ids/positions/block_table.
        let (input_ids_dev, mut plan) = self.batch_ws.build_plan(seqs, &self.device)?;
        // Workspace doesn't know the runner's block_size; patch it in.
        plan.block_size = self.block_size;
        let batch = plan.batch;

        // Profiling hook: wall-clock only. The earlier cudaEvent-based GPU
        // timing here was dead code (events were created and destroyed but
        // never recorded) and relied on an unsound "D is Cuda" assumption in
        // a generic method (H2). GPU-side timing lives in the graph path.
        #[cfg(feature = "cuda")]
        let prof = crate::env_flags::profile_gpu();
        #[cfg(feature = "cuda")]
        let wall_t0 = std::time::Instant::now();

        let logits = {
            let mut ctx = ForwardContext {
                kv_pool: &mut self.kv_pool,
                plan: &plan,
                workspace: &mut self.forward_ws,
            };
            self.model.forward(&input_ids_dev, &mut ctx)?
        };
        let (out_dev, workspace, rows) = self.forward_ws.argmax_args();
        let result = D::argmax_batched(&logits, &plan.cu_q_lens, batch, out_dev, workspace, rows);

        // Sticky CUDA error check at the step boundary (C1/C2): any kernel
        // launch failure during forward sets the sticky error, which
        // `cudaGetLastError` surfaces here as `OpError::Kernel` rather than
        // letting a NaN/garbage token be returned silently. Only meaningful
        // on the Cuda backend; a no-op global query otherwise.
        #[cfg(feature = "cuda")]
        if self.device.name() == "cuda" {
            crate::infrastructure::cuda::error::check_last_error("step_batch_eager forward")?;
        }

        #[cfg(feature = "cuda")]
        if prof {
            self.prof_step_wall_ns += wall_t0.elapsed().as_nanos() as u64;
            self.prof_step_count += 1;
        }

        result
    }

    fn validate_steps(&self, seqs: &[SeqStep]) -> OpResult<()> {
        for (i, s) in seqs.iter().enumerate() {
            if s.input_ids.is_empty() {
                return Err(OpError::Shape(format!("SeqStep[{}]: empty input_ids", i)));
            }
            if s.input_ids.len() != s.positions.len() {
                return Err(OpError::Shape(format!(
                    "SeqStep[{}]: input_ids ({}) != positions ({})",
                    i,
                    s.input_ids.len(),
                    s.positions.len(),
                )));
            }
            if s.kv_write_start < 0 || s.kv_len_after < 0 {
                return Err(OpError::Shape(format!(
                    "SeqStep[{}]: negative kv range start={} len_after={}",
                    i, s.kv_write_start, s.kv_len_after,
                )));
            }

            let q_len = s.input_ids.len();
            let kv_write_start = s.kv_write_start as usize;
            let kv_len_after = s.kv_len_after as usize;
            let expected_kv_len = kv_write_start
                .checked_add(q_len)
                .ok_or_else(|| OpError::Shape(format!("SeqStep[{}]: kv length overflow", i)))?;
            if kv_len_after != expected_kv_len {
                return Err(OpError::Shape(format!(
                    "SeqStep[{}]: kv_len_after ({}) != kv_write_start ({}) + q_len ({})",
                    i, kv_len_after, kv_write_start, q_len,
                )));
            }
            if kv_len_after > self.max_seq_len {
                return Err(OpError::Shape(format!(
                    "SeqStep[{}]: kv_len_after ({}) > max_seq_len ({})",
                    i, kv_len_after, self.max_seq_len,
                )));
            }

            for (j, &pos) in s.positions.iter().enumerate() {
                if pos < 0 || pos as usize >= self.max_seq_len {
                    return Err(OpError::Shape(format!(
                        "SeqStep[{}]: position[{}]={} outside [0,{})",
                        i, j, pos, self.max_seq_len,
                    )));
                }
            }

            let required_blocks = kv_len_after.div_ceil(self.block_size);
            if s.block_table.len() < required_blocks {
                return Err(OpError::Shape(format!(
                    "SeqStep[{}]: block_table ({}) < required blocks ({}) for kv_len_after {}",
                    i,
                    s.block_table.len(),
                    required_blocks,
                    kv_len_after,
                )));
            }
            if s.block_table.len() > self.max_blocks_per_seq {
                return Err(OpError::Shape(format!(
                    "SeqStep[{}]: block_table ({}) > max_blocks_per_seq ({})",
                    i,
                    s.block_table.len(),
                    self.max_blocks_per_seq,
                )));
            }
            for (j, &block) in s.block_table.iter().enumerate() {
                if block as usize >= self.kv_pool.num_blocks {
                    return Err(OpError::Shape(format!(
                        "SeqStep[{}]: block_table[{}]={} outside KV pool blocks {}",
                        i, j, block, self.kv_pool.num_blocks,
                    )));
                }
            }
        }
        Ok(())
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
        let debug = crate::env_flags::debug_layers();
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
            tracing::debug!(
                num_tokens = num_prompt,
                kv_len_after = num_prompt,
                "prefill"
            );
        }
        let mut last = self.step_batch(&[prefill_seq])?[0];
        if debug {
            tracing::debug!(token = last, "prefill argmax");
        }
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
                tracing::debug!(
                    step = i,
                    input = last,
                    pos = kv_write_start,
                    kv_len = kv_len_after,
                    token = new,
                    "decode step"
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
    use crate::domain::Tensor;
    use crate::infrastructure::cpu::Cpu;
    use crate::models::layers::{Embedding, Linear, RMSNorm};
    use crate::models::llama3::{Llama3Layer, Llama3Model};

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

        let make_weight = |rows: usize, cols: usize| -> Tensor<f32, Cpu> {
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
                .collect();
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
            embed_tokens: Embedding {
                table: make_weight(vocab, dim),
            },
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

    #[test]
    fn rejects_invalid_step_before_forward() {
        let model = tiny_llama3();
        let mut runner = ModelRunner::new(model, Cpu, 4, 4, 4, 16, 4, 1, 1, vec![]).unwrap();
        let err = runner
            .step_batch(&[SeqStep {
                input_ids: vec![1, 2],
                positions: vec![0, 1],
                kv_write_start: 0,
                kv_len_after: 3,
                block_table: vec![0],
            }])
            .unwrap_err();
        assert!(
            matches!(err, OpError::Shape(ref msg) if msg.contains("kv_len_after")),
            "unexpected error: {:?}",
            err
        );
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
        let mut runner_ref0 =
            ModelRunner::new(model_ref0, Cpu, 4, 4, 4, 16, 8, 1, 1, vec![]).unwrap();
        let r0 = runner_ref0.generate(&p0, 1, &[]).unwrap();

        let model_ref1 = tiny_llama3();
        let mut runner_ref1 =
            ModelRunner::new(model_ref1, Cpu, 4, 4, 4, 16, 8, 1, 1, vec![]).unwrap();
        let r1 = runner_ref1.generate(&p1, 1, &[]).unwrap();

        assert_eq!(
            batched_first[0], r0[0],
            "ragged batch seq 0 first-token mismatch: batch={} serial={}",
            batched_first[0], r0[0]
        );
        assert_eq!(
            batched_first[1], r1[0],
            "ragged batch seq 1 first-token mismatch: batch={} serial={}",
            batched_first[1], r1[0]
        );
    }
}
