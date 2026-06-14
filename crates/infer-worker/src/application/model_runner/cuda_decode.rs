use super::{ModelRunner, SeqStep};
use crate::application::batch_workspace::WsSeqStep;
use crate::application::cuda_graph_runner::{CudaGraphRunner, GraphDecision};
use crate::application::forward_workspace::ForwardWorkspace;
use crate::domain::model::{ForwardContext, LlmModel};
use crate::domain::ports::{OpError, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::infrastructure::cuda::{
    Cuda,
    kernels::argmax_batched::argmax_batched_decode_into,
    kernels::gather_merge::{
        MergeCompactDecodeArgs, append_decode_admissions_into, merge_compact_decode_into,
    },
};

#[derive(Debug, Clone)]
pub struct DecodeRowToken {
    pub src_row: usize,
    pub token_id: i32,
}

#[derive(Debug, Clone)]
pub struct DecodeCompactOutput {
    pub active: Vec<DecodeRowToken>,
    pub finished: Vec<DecodeRowToken>,
}

impl<T: Dtype, M: LlmModel<T, Cuda, ForwardWorkspace<T, Cuda>>> ModelRunner<T, Cuda, M> {
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

        // Block_table is the same for all dummy seqs; produce SeqSteps
        // for the maximum capture size, slice for smaller ones.
        let max_size = *usable_sizes.last().unwrap();
        let dummy_steps: Vec<SeqStep> = (0..max_size)
            .map(|_| SeqStep {
                input_ids: vec![0],
                positions: vec![0],
                kv_write_start: 0,
                kv_len_after: 1,
                block_table: block_table.clone(),
            })
            .collect();

        // The capture loop needs `&CudaConfig`. The runner's stream lives
        // inside `Arc<CudaConfig>`, so a cheap clone gives us a handle
        // independent of `self` and avoids aliasing during the closure.
        let cuda_config = self.device.config.clone();

        graph_runner.warmup_and_capture_all(&*cuda_config, 2, |size, is_capture| {
            if is_capture {
                // Capture pass: forward + argmax ONLY (no H2D memcpy).
                // Device buffers already hold valid data from warmup.
                self.run_decode_forward_only(size)
            } else {
                // Warmup pass: full path including H2D upload.
                self.run_decode_only_step(&dummy_steps[..size])
            }
        })?;

        self.graph_runner = Some(graph_runner);
        Ok(())
    }

    /// Run a single decode-only forward into `forward_ws.argmax_out_dev`.
    /// Includes build_plan (H2D upload). Used for warmup passes and eager fallback.
    fn run_decode_only_step(&mut self, seqs: &[SeqStep]) -> OpResult<()> {
        let ws_seqs: Vec<WsSeqStep> = seqs
            .iter()
            .map(|s| WsSeqStep {
                input_ids: s.input_ids.clone(),
                positions: s.positions.clone(),
                kv_write_start: s.kv_write_start,
                kv_len_after: s.kv_len_after,
                block_table: s.block_table.clone(),
            })
            .collect();

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
        let (out_dev, workspace, _rows) = self.forward_ws.argmax_args();
        argmax_batched_decode_into(&logits, out_dev, workspace)
    }

    /// Forward + argmax ONLY — no H2D upload.
    ///
    /// Used during CUDA Graph capture: device buffers already hold valid
    /// data from the preceding warmup pass. By skipping `build_plan`'s
    /// `upload_async` calls, we keep cudaMemcpyAsync operations OUT of
    /// the captured graph. The graph will contain only kernel launches.
    fn run_decode_forward_only(&mut self, batch_size: usize) -> OpResult<()> {
        let (input_ids_dev, mut plan) = self
            .batch_ws
            .get_last_plan_views(batch_size, self.block_size)?;
        plan.block_size = self.block_size;

        let logits = {
            let mut ctx = ForwardContext {
                kv_pool: &mut self.kv_pool,
                plan: &plan,
                workspace: &mut self.forward_ws,
            };
            self.model.forward(&input_ids_dev, &mut ctx)?
        };
        let (out_dev, workspace, _rows) = self.forward_ws.argmax_args();
        argmax_batched_decode_into(&logits, out_dev, workspace)
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
        self.validate_steps(seqs)?;
        // Escape hatch for A/B benchmarking against eager.
        if std::env::var("RUSTINFER_DISABLE_GRAPH").is_ok() {
            return self.step_batch_eager(seqs);
        }
        let all_decode = seqs.iter().all(|s| s.input_ids.len() == 1);
        let batch = seqs.len();
        let primed = self.graph_runner.is_some();
        let max_cap = self
            .graph_runner
            .as_ref()
            .map(|g| g.max_capture_size())
            .unwrap_or(0);

        if !primed || !all_decode || batch > max_cap {
            return self.step_batch_eager(seqs);
        }

        // Pick the next captured size >= batch.
        let decision = self.graph_runner.as_ref().unwrap().decide(batch);
        let (slot, padded_size) = match decision {
            GraphDecision::Replay {
                slot, padded_size, ..
            } => (slot, padded_size),
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
        let ws_seqs: Vec<WsSeqStep> = padded
            .iter()
            .map(|s| WsSeqStep {
                input_ids: s.input_ids.clone(),
                positions: s.positions.clone(),
                kv_write_start: s.kv_write_start,
                kv_len_after: s.kv_len_after,
                block_table: s.block_table.clone(),
            })
            .collect();
        let _ = self.batch_ws.build_plan(&ws_seqs, &self.device)?;

        // 2. Launch the captured graph.
        if std::env::var("RUSTINFER_TRACE_GRAPH").is_ok() {
            eprintln!(
                "[graph] replay slot={:?} batch={}->{}",
                slot, batch, padded_size
            );
        }

        // Profiling: enable with RUSTINFER_PROFILE_GPU=1. We wrap the
        // graph launch with a cudaEvent pair to measure pure GPU time.
        // The wall-clock around the whole step_batch_with_graph call is
        // measured outside the launch (build_plan + D2H included).
        let prof = std::env::var("RUSTINFER_PROFILE_GPU").is_ok();
        let wall_t0 = std::time::Instant::now();
        let mut ev_t0: crate::infrastructure::cuda::ffi::cudaEvent_t = std::ptr::null_mut();
        let mut ev_t1: crate::infrastructure::cuda::ffi::cudaEvent_t = std::ptr::null_mut();
        if prof {
            unsafe {
                let r0 = crate::infrastructure::cuda::ffi::cudaEventCreate(&mut ev_t0);
                let r1 = crate::infrastructure::cuda::ffi::cudaEventCreate(&mut ev_t1);
                if r0 != crate::infrastructure::cuda::ffi::cudaError_cudaSuccess
                    || r1 != crate::infrastructure::cuda::ffi::cudaError_cudaSuccess
                {
                    tracing::debug!("cudaEventCreate failed - skipping graph profiling");
                    ev_t0 = std::ptr::null_mut();
                    ev_t1 = std::ptr::null_mut();
                } else {
                    crate::infrastructure::cuda::ffi::cudaEventRecord(
                        ev_t0,
                        self.device.config.stream,
                    );
                }
            }
        }
        self.device.config.launch(slot)?;
        let prof_ok = prof && !ev_t0.is_null() && !ev_t1.is_null();
        if prof_ok {
            unsafe {
                crate::infrastructure::cuda::ffi::cudaEventRecord(ev_t1, self.device.config.stream);
            }
        }

        // 3. Synchronous D2H of the argmax_out_dev (just `padded_size` ints).
        // We only return the first `batch` of them.
        let host = self.forward_ws.argmax_out_dev().to_host_vec()?;

        if prof_ok {
            unsafe {
                crate::infrastructure::cuda::ffi::cudaEventSynchronize(ev_t1);
                let mut ms: f32 = 0.0;
                crate::infrastructure::cuda::ffi::cudaEventElapsedTime(&mut ms, ev_t0, ev_t1);
                self.prof_graph_gpu_ns += (ms as f64 * 1.0e6) as u64;
                crate::infrastructure::cuda::ffi::cudaEventDestroy(ev_t0);
                crate::infrastructure::cuda::ffi::cudaEventDestroy(ev_t1);
            }
            self.prof_step_wall_ns += wall_t0.elapsed().as_nanos() as u64;
            self.prof_step_count += 1;
        }
        Ok(host.into_iter().take(batch).collect())
    }

    /// True when a decode-only CUDA graph replay is available for `batch`.
    pub fn can_replay_decode_graph(&self, batch: usize) -> bool {
        if batch == 0 || std::env::var("RUSTINFER_DISABLE_GRAPH").is_ok() {
            return false;
        }
        let Some(graph_runner) = self.graph_runner.as_ref() else {
            return false;
        };
        if batch == 1 {
            return graph_runner.captured_slot_for(1).is_some();
        }
        matches!(graph_runner.decide(batch), GraphDecision::Replay { .. })
    }

    /// Copy newly-admitted decode seed tokens through buffer B and append
    /// them into stable buffer A.
    pub fn append_decode_admissions_to_a(
        &mut self,
        start_row: usize,
        tokens: &[i32],
    ) -> OpResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if start_row + tokens.len() > self.cap_batch {
            return Err(OpError::Shape(format!(
                "append_decode_admissions_to_a: rows {}..{} exceed cap_batch {}",
                start_row,
                start_row + tokens.len(),
                self.cap_batch,
            )));
        }
        if tokens.len() > self.forward_ws.new_token_host_mut().len() {
            return Err(OpError::Shape(format!(
                "append_decode_admissions_to_a: tokens ({}) > B capacity ({})",
                tokens.len(),
                self.forward_ws.new_token_host_mut().len(),
            )));
        }

        let cfg = self.device.config.clone();
        let src_ptr = {
            let host = self.forward_ws.new_token_host_mut();
            host[..tokens.len()].copy_from_slice(tokens);
            host.as_ptr() as *const std::ffi::c_void
        };
        let dst_ptr = self.forward_ws.new_token_dev().data_ptr() as *mut std::ffi::c_void;
        unsafe {
            cfg.upload_h2d_copy_in(dst_ptr, src_ptr, std::mem::size_of_val(tokens))?;
        }
        cfg.record_copy_in()?;
        cfg.compute_wait_copy_in()?;

        let mut a = self.batch_ws.input_ids_dev().view_raw(
            Shape::from_slice(&[start_row + tokens.len()]),
            Shape::from_slice(&[start_row + tokens.len()]).contiguous_strides(),
            0,
            true,
        );
        let b = self.forward_ws.new_token_dev();
        append_decode_admissions_into(&mut a, b, start_row, tokens.len(), cfg.stream)
    }

    fn upload_decode_compact_metadata_copy_in(
        &mut self,
        generated_counts: &[usize],
        max_tokens: &[usize],
        ignore_eos: &[bool],
        eos_ids: &[i32],
    ) -> OpResult<()> {
        let batch = generated_counts.len();
        if batch != max_tokens.len() || batch != ignore_eos.len() {
            return Err(OpError::Shape(format!(
                "upload_decode_compact_metadata_copy_in: mismatched lens gen={} max={} ignore={}",
                generated_counts.len(),
                max_tokens.len(),
                ignore_eos.len(),
            )));
        }
        if batch > self.cap_batch {
            return Err(OpError::Shape(format!(
                "upload_decode_compact_metadata_copy_in: batch ({}) > cap ({})",
                batch, self.cap_batch,
            )));
        }
        if eos_ids.len() > self.forward_ws.decode_eos_ids_host_mut().len() {
            return Err(OpError::Shape(format!(
                "upload_decode_compact_metadata_copy_in: eos_ids ({}) > capacity ({})",
                eos_ids.len(),
                self.forward_ws.decode_eos_ids_host_mut().len(),
            )));
        }

        let gen_ptr = {
            let host = self.forward_ws.decode_generated_counts_host_mut();
            for (dst, &src) in host[..batch].iter_mut().zip(generated_counts.iter()) {
                *dst = src as i32;
            }
            host.as_ptr() as *const std::ffi::c_void
        };
        let max_ptr = {
            let host = self.forward_ws.decode_max_tokens_host_mut();
            for (dst, &src) in host[..batch].iter_mut().zip(max_tokens.iter()) {
                *dst = src as i32;
            }
            host.as_ptr() as *const std::ffi::c_void
        };
        let ignore_ptr = {
            let host = self.forward_ws.decode_ignore_eos_host_mut();
            for (dst, &src) in host[..batch].iter_mut().zip(ignore_eos.iter()) {
                *dst = i32::from(src);
            }
            host.as_ptr() as *const std::ffi::c_void
        };
        let eos_ptr = {
            let host = self.forward_ws.decode_eos_ids_host_mut();
            host[..eos_ids.len()].copy_from_slice(eos_ids);
            host.as_ptr() as *const std::ffi::c_void
        };

        let cfg = self.device.config.clone();
        unsafe {
            cfg.upload_h2d_copy_in(
                self.forward_ws.decode_generated_counts_dev().data_ptr() as *mut std::ffi::c_void,
                gen_ptr,
                batch * std::mem::size_of::<i32>(),
            )?;
            cfg.upload_h2d_copy_in(
                self.forward_ws.decode_max_tokens_dev().data_ptr() as *mut std::ffi::c_void,
                max_ptr,
                batch * std::mem::size_of::<i32>(),
            )?;
            cfg.upload_h2d_copy_in(
                self.forward_ws.decode_ignore_eos_dev().data_ptr() as *mut std::ffi::c_void,
                ignore_ptr,
                batch * std::mem::size_of::<i32>(),
            )?;
            cfg.upload_h2d_copy_in(
                self.forward_ws.decode_eos_ids_dev().data_ptr() as *mut std::ffi::c_void,
                eos_ptr,
                std::mem::size_of_val(eos_ids),
            )?;
        }
        cfg.record_copy_in()
    }

    /// Serving decode primitive for the ABC compact pipeline.
    ///
    /// Precondition: buffer A already contains one input token per `seqs`
    /// row. The graph reads A and writes C; the compact merge kernel writes
    /// non-finished C tokens back to A, putting active rows first and
    /// returning source-row mappings plus tokens.
    pub fn step_decode_abc_compact(
        &mut self,
        seqs: &[SeqStep],
        generated_counts: &[usize],
        max_tokens: &[usize],
        ignore_eos: &[bool],
        eos_ids: &[i32],
    ) -> OpResult<DecodeCompactOutput> {
        if seqs.is_empty() {
            return Ok(DecodeCompactOutput {
                active: Vec::new(),
                finished: Vec::new(),
            });
        }
        self.validate_steps(seqs)?;
        if seqs.iter().any(|s| s.input_ids.len() != 1) {
            return Err(OpError::Shape(
                "step_decode_abc_compact: all seqs must be decode-only q_len=1".into(),
            ));
        }
        if generated_counts.len() != seqs.len()
            || max_tokens.len() != seqs.len()
            || ignore_eos.len() != seqs.len()
        {
            return Err(OpError::Shape(format!(
                "step_decode_abc_compact: metadata lens gen={} max={} ignore={} seqs={}",
                generated_counts.len(),
                max_tokens.len(),
                ignore_eos.len(),
                seqs.len(),
            )));
        }
        if std::env::var("RUSTINFER_DISABLE_GRAPH").is_ok() {
            return Err(OpError::unsupported(
                "cuda",
                "ABC compact decode requires graph replay",
            ));
        }
        let graph_runner = self.graph_runner.as_ref().ok_or_else(|| {
            OpError::unsupported("cuda", "ABC compact decode requires primed CUDA graphs")
        })?;
        let batch = seqs.len();
        let decision = graph_runner.decide(batch);
        let (slot, padded_size) = match decision {
            GraphDecision::Replay {
                slot, padded_size, ..
            } => (slot, padded_size),
            GraphDecision::Eager => {
                return Err(OpError::unsupported(
                    "cuda",
                    "ABC compact decode batch exceeds captured graph sizes",
                ));
            }
        };

        if padded_size > batch {
            let zeros = vec![0i32; padded_size - batch];
            self.append_decode_admissions_to_a(batch, &zeros)?;
        }

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

        let ws_seqs: Vec<WsSeqStep> = padded
            .iter()
            .map(|s| WsSeqStep {
                input_ids: s.input_ids.clone(),
                positions: s.positions.clone(),
                kv_write_start: s.kv_write_start,
                kv_len_after: s.kv_len_after,
                block_table: s.block_table.clone(),
            })
            .collect();
        let (_, mut plan) = self.batch_ws.build_decode_plan_preserve_input(
            &ws_seqs,
            &self.device,
            self.block_size,
        )?;
        plan.block_size = self.block_size;

        self.upload_decode_compact_metadata_copy_in(
            generated_counts,
            max_tokens,
            ignore_eos,
            eos_ids,
        )?;

        let cfg = self.device.config.clone();
        cfg.launch(slot)?;
        cfg.compute_wait_copy_in()?;

        {
            let mut a = self.batch_ws.input_ids_dev().view_raw(
                Shape::from_slice(&[batch]),
                Shape::from_slice(&[batch]).contiguous_strides(),
                0,
                true,
            );
            merge_compact_decode_into(MergeCompactDecodeArgs {
                a_out: &mut a,
                c_prev: self.forward_ws.argmax_out_dev(),
                generated_counts: self.forward_ws.decode_generated_counts_dev(),
                max_tokens: self.forward_ws.decode_max_tokens_dev(),
                ignore_eos: self.forward_ws.decode_ignore_eos_dev(),
                eos_ids: self.forward_ws.decode_eos_ids_dev(),
                eos_len: eos_ids.len(),
                old_batch: batch,
                active_src_rows: self.forward_ws.decode_active_src_rows_dev(),
                finished_src_rows: self.forward_ws.decode_finished_src_rows_dev(),
                finished_tokens: self.forward_ws.decode_finished_tokens_dev(),
                counts: self.forward_ws.decode_counts_dev(),
                stream: cfg.stream,
            })?;
        }

        let counts = self.forward_ws.decode_counts_dev().to_host_vec()?;
        if counts.len() < 3 {
            return Err(OpError::Kernel("compact counts buffer too small".into()));
        }
        if counts[0] < 0 || counts[1] < 0 || counts[2] < 0 {
            return Err(OpError::Kernel(format!(
                "compact counts negative: {:?}",
                &counts[..3]
            )));
        }
        let active_n = counts[0] as usize;
        let finished_n = counts[1] as usize;
        let old_n = counts[2] as usize;
        if old_n != batch || active_n + finished_n != batch {
            return Err(OpError::Kernel(format!(
                "compact counts invalid: active={} finished={} old={} batch={}",
                active_n, finished_n, old_n, batch,
            )));
        }

        let active_tokens = self
            .batch_ws
            .input_ids_dev()
            .view_raw(
                Shape::from_slice(&[active_n]),
                Shape::from_slice(&[active_n.max(1)]).contiguous_strides(),
                0,
                true,
            )
            .to_host_vec()?;
        let active_src_rows = self
            .forward_ws
            .decode_active_src_rows_dev()
            .view_raw(
                Shape::from_slice(&[active_n]),
                Shape::from_slice(&[active_n.max(1)]).contiguous_strides(),
                0,
                true,
            )
            .to_host_vec()?;
        let finished_src_rows = self
            .forward_ws
            .decode_finished_src_rows_dev()
            .view_raw(
                Shape::from_slice(&[finished_n]),
                Shape::from_slice(&[finished_n.max(1)]).contiguous_strides(),
                0,
                true,
            )
            .to_host_vec()?;
        let finished_tokens = self
            .forward_ws
            .decode_finished_tokens_dev()
            .view_raw(
                Shape::from_slice(&[finished_n]),
                Shape::from_slice(&[finished_n.max(1)]).contiguous_strides(),
                0,
                true,
            )
            .to_host_vec()?;

        let active: Vec<DecodeRowToken> = active_src_rows
            .into_iter()
            .zip(active_tokens)
            .map(|(src_row, token_id)| DecodeRowToken {
                src_row: src_row as usize,
                token_id,
            })
            .collect();
        let finished: Vec<DecodeRowToken> = finished_src_rows
            .into_iter()
            .zip(finished_tokens)
            .map(|(src_row, token_id)| DecodeRowToken {
                src_row: src_row as usize,
                token_id,
            })
            .collect();

        let mut seen = vec![false; batch];
        for row in active.iter().chain(finished.iter()) {
            if row.src_row >= batch {
                return Err(OpError::Kernel(format!(
                    "compact src_row {} out of range batch {}",
                    row.src_row, batch
                )));
            }
            if seen[row.src_row] {
                return Err(OpError::Kernel(format!(
                    "compact src_row {} returned twice",
                    row.src_row
                )));
            }
            seen[row.src_row] = true;
        }
        if seen.iter().any(|v| !*v) {
            return Err(OpError::Kernel(
                "compact result did not cover every input row".into(),
            ));
        }

        Ok(DecodeCompactOutput { active, finished })
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
        // Prefill is multi-token -> eager.
        let mut last = self.step_batch_eager(&[prefill_seq])?[0];
        if debug {
            eprintln!("[runner] prefill argmax -> token {}", last);
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
            // Decode -> graph (auto-falls-back to eager if not primed).
            let new = self.step_batch_with_graph(&[step])?[0];
            if debug {
                eprintln!(
                    "[runner] graph-decode {:>2}: in={:>6} pos={} kv_len={} -> token {}",
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

    /// ABC-buffer greedy decode for a single sequence.
    ///
    /// This demo path uses the same per-step primitive as serving:
    /// graph forward reads A, writes C, then compact merge commits
    /// surviving `C -> A`; host-visible active output is read from A. It still
    /// synchronizes once per token for EOS handling, so it is an ABC
    /// correctness path rather than the final fully overlapped pipeline.
    pub fn generate_pipelined(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_ids: &[i32],
    ) -> OpResult<Vec<i32>> {
        // Without primed graphs the merge/stream chain has no captured
        // forward to drive - defer to the graph-aware eager loop.
        if self.graph_runner.is_none() {
            return self.generate_with_graph(prompt_ids, max_new_tokens, eos_ids);
        }
        let num_prompt = prompt_ids.len();
        if num_prompt == 0 {
            return Err(OpError::Shape("empty prompt".into()));
        }
        // Decode is single-seq batch=1; bail to eager if no exact graph.
        if self
            .graph_runner
            .as_ref()
            .and_then(|g| g.captured_slot_for(1))
            .is_none()
        {
            return self.generate_with_graph(prompt_ids, max_new_tokens, eos_ids);
        }
        let debug = std::env::var("RUSTINFER_DEBUG_LAYERS").is_ok();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let block_table: Vec<u32> = (0..self.max_blocks_per_seq as u32).collect();

        // 1. Prefill (eager, multi-token). Returns the first decode token.
        let prefill_seq = SeqStep {
            input_ids: prompt_ids.to_vec(),
            positions: (0..num_prompt as i32).collect(),
            kv_write_start: 0,
            kv_len_after: num_prompt as i32,
            block_table: block_table.clone(),
        };
        let mut last = self.step_batch_eager(&[prefill_seq])?[0];
        if debug {
            eprintln!("[pipe] prefill argmax -> token {}", last);
        }
        generated.push(last);
        if eos_ids.contains(&last) {
            return Ok(generated);
        }

        self.append_decode_admissions_to_a(0, &[last])?;
        let mut generated_count = 1usize;
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
            let compact = self.step_decode_abc_compact(
                &[step],
                &[generated_count],
                &[max_new_tokens],
                &[false],
                eos_ids,
            )?;
            let (token, finished) = if let Some(row) = compact.active.first() {
                (row.token_id, false)
            } else if let Some(row) = compact.finished.first() {
                (row.token_id, true)
            } else {
                return Err(OpError::Kernel(
                    "generate_pipelined: compact returned no row".into(),
                ));
            };
            last = token;
            generated_count += 1;
            if debug {
                eprintln!(
                    "[pipe] decode {:>2}: pos={} kv_len={} -> token {}",
                    i, kv_write_start, kv_len_after, last,
                );
            }
            generated.push(last);
            if finished {
                break;
            }
        }
        // Drain all streams before returning.
        self.device.config.synchronize()?;
        Ok(generated)
    }
}
