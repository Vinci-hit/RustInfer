//! ABC GPU-resident pipelined pure-decode: the `issue_decode_abc` /
//! `finalize_decode_abc` halves of the 1-deep decode pipeline (CUDA).

use crate::domain::dtype::Dtype;
use crate::domain::exec::ExecScope;
use crate::domain::model::DecoderModel;
use crate::domain::plan::StepRequest;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::pipeline_ops::{CompactExtendControlArgs, MergeCompactDecodeArgs};
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

use super::{
    AsyncControlBuffers, DecodeCompactOutput, DecodeRowToken, GraphDecision, Runtime,
    u32_to_i32_saturating, upload_i32_prefix,
};

impl<T, D, M> Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    pub(super) fn ensure_abc_pinned(&mut self) -> OpResult<()> {
        if self.abc.pinned {
            return Ok(());
        }
        let _guard = self.scope.enter();
        D::pipeline_pin_host_i32(&self.scope, &self.abc.new_token_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.argmax_out_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.counts_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.active_src_rows_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.active_tokens_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.finished_src_rows_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.finished_tokens_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.prefill_final_src_rows_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.prefill_final_tokens_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.row_kind_host)?;
        D::pipeline_pin_host_i32(&self.scope, &self.abc.last_token_rows_host)?;
        // Pin the persistent block-table staging too — it is the largest
        // per-step H2D on the decode path and was previously a pageable
        // (host-synchronous) copy.
        D::pipeline_pin_host_i32(&self.scope, &self.block_tables_host)?;
        self.abc.pinned = true;
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
    /// Lazily allocate the async-decode control-plane scratch (only when the
    /// async path is first used). Sized to capacity so addresses are stable.
    pub(super) fn ensure_async_ctrl(&mut self) -> OpResult<()> {
        if self.async_ctrl.is_some() {
            return Ok(());
        }
        let device = self.scope.device();
        let cb = self.cap_batch.max(1);
        let mbps = self.max_blocks_per_seq.max(1);
        let block_tables_scratch =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cb * mbps]), device)?;
        let kv_lens_scratch = Tensor::<i32, D>::zeros(Shape::from_slice(&[cb]), device)?;
        let new_slots_dev = Tensor::<i32, D>::zeros(Shape::from_slice(&[cb]), device)?;
        self.async_ctrl = Some(AsyncControlBuffers {
            block_tables_scratch,
            kv_lens_scratch,
            new_slots_dev,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn issue_decode_abc(
        &mut self,
        req: &StepRequest,
        a_valid_prefix: usize,
        generated_counts: &[u32],
        max_tokens: &[u32],
        ignore_eos: &[bool],
        eos_ids: &[i32],
        // Async device control plane (high-QPS path). `Some(next_slots)` enables
        // it: after the forward+merge, build the NEXT step's per-row control on
        // device (`compact_extend_control`) instead of a host rebuild+upload.
        // `next_slots[r]` is the next-step KV slot for output row r (the Stage-1
        // speculative reservation). `reuse_device_control` skips this step's
        // `upload_index` because the previous step's `compact_extend` already
        // left this step's control on device; false on the first async step or
        // after any admission/eviction re-seeds it from the host request.
        async_next_slots: Option<&[u32]>,
        reuse_device_control: bool,
    ) -> OpResult<()> {
        let plan = self.build_plan(req)?;
        let batch = plan.batch;
        if plan.num_tokens != batch || plan.q_lens.iter().any(|&q| q != 1) {
            return Err(OpError::Shape(
                "step_decode_abc: requires pure decode (q_len=1 per row)".into(),
            ));
        }
        if generated_counts.len() != batch || max_tokens.len() != batch || ignore_eos.len() != batch
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
        self.ensure_abc_pinned()?;

        // Async path: when reusing the device-resident control plane, this
        // step's block tables / kv_lens / positions were left on device by the
        // previous step's `compact_extend_control`; skip the host rebuild +
        // upload entirely. Otherwise (proven path, or first/post-change async
        // step) seed the device control from the host request as usual.
        if async_next_slots.is_some() && reuse_device_control {
            // Control already on device. Nothing to upload.
        } else {
            self.upload_index(&plan, req)?;
        }

        // Guard A against this step's append/merge overwriting it before the
        // prior step's copy-out (So) finished reading A_{n-1}. (ev_out)
        if self.abc.copy_out_recorded {
            D::pipeline_compute_wait_copy_out(&self.scope)?;
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
            D::pipeline_synchronize_copy_in(&self.scope)?;
            for (dst, seq) in self.abc.new_token_host[..n]
                .iter_mut()
                .zip(req.seqs[vp..].iter())
            {
                *dst = seq.input_ids[0];
            }
            let _guard = self.scope.enter();
            unsafe {
                D::pipeline_upload_h2d_copy_in(
                    &self.scope,
                    self.abc.new_token_dev.data_ptr_mut() as *mut std::ffi::c_void,
                    self.abc.new_token_host.as_ptr() as *const std::ffi::c_void,
                    n * std::mem::size_of::<i32>(),
                )?;
            }
            D::pipeline_record_copy_in(&self.scope)?; // ev_in on Si
            // Compute waits ev_in before reading B.
            D::pipeline_compute_wait_copy_in(&self.scope)?;
            D::append_decode_admissions(
                &self.scope,
                &mut self.input_ids_buf,
                &self.abc.new_token_dev,
                vp,
                n,
            )?;
        }

        // ── forward + finalize + in-graph argmax → buffer C ──
        let input_ids = self.input_ids_buf.view_raw(
            Shape::from_slice(&[batch]),
            Shape::from_slice(&[batch]).contiguous_strides(),
            0,
            true,
        );
        // Pad the live decode batch UP to the next captured graph size and
        // replay that (possibly larger) graph. `slot_for_batch` rounds `batch`
        // up to the smallest capture size >= batch; replaying it is correct
        // because the per-seq control buffers are zero-padded to capacity, so
        // the tail rows [batch, slot) carry q_len=0 / kv_len=0: the KV scatter
        // writes nothing and paged attention reads nothing for them, and the
        // forward is per-row independent (no cross-row reduction), so their
        // garbage output lands only in C[batch..slot], which the merge — run at
        // the real `batch` (old_batch=batch) — never reads. This eliminates the
        // eager fallback that every NON-capture batch size used to take: an
        // eager `forward_finalize_argmax` on a cold (batch × kv_len) shape pays
        // the full lazy cuDNN-SDPA-plan + cuBLASLt-algo build (~400ms measured),
        // inline on the serve thread — the dominant wandering TTFT/TPOT tail
        // spike under load ("抖动"). With boot prewarm every slot is ready, so
        // this is a pure replay.
        let slot_batch = match self.decide(&plan) {
            GraphDecision::Graph(slot) => self.graph.as_ref().and_then(|g| g.slot_size(slot)),
            // This ABC path only runs for decode-only steps, so a prefill-graph
            // decision never reaches here; treat it as eager for exhaustiveness.
            GraphDecision::PrefillGraph(_) | GraphDecision::Eager => None,
        };
        match slot_batch {
            Some(sb) if self.scope.graph_ready(sb as u64) => {
                // Hot path: pure replay of the (>= batch) captured graph.
                self.scope.graph_launch(sb as u64)?;
            }
            // No ready graph for this shape (boot prewarm off/failed, or
            // batch > max capture size): run EAGER at the real batch. We never
            // capture inline at serve time — an inline capture is a full eager
            // forward + `synchronize` + trace pass that blocks the serve loop
            // (and every prefill queued behind it); that lazy capture WAS the
            // original TTFT/TPOT stall. Every decode graph is captured once at
            // boot by `prewarm_decode_graphs`; serving only replays or runs eager.
            _ => {
                self.forward_finalize_argmax(&plan, &input_ids)?;
            }
        }

        // ── upload stop metadata + run the compact merge (C → A) ──
        let gen_i32: Vec<i32> = generated_counts
            .iter()
            .map(|&x| u32_to_i32_saturating(x))
            .collect();
        let max_i32: Vec<i32> = max_tokens
            .iter()
            .map(|&x| u32_to_i32_saturating(x))
            .collect();
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
        {
            let _guard = self.scope.enter();
            let mut a = self.input_ids_buf.view_raw(
                Shape::from_slice(&[batch]),
                Shape::from_slice(&[batch]).contiguous_strides(),
                0,
                true,
            );
            D::merge_compact_decode(
                &self.scope,
                MergeCompactDecodeArgs {
                    a_out: &mut a,
                    c_prev: &self.abc.argmax_out_dev,
                    generated_counts: &self.abc.generated_counts_dev,
                    max_tokens: &self.abc.max_tokens_dev,
                    ignore_eos: &self.abc.ignore_eos_dev,
                    eos_ids: &self.abc.eos_ids_dev,
                    eos_len: eos_ids.len(),
                    old_batch: batch,
                    active_src_rows: &mut self.abc.active_src_rows_dev,
                    finished_src_rows: &mut self.abc.finished_src_rows_dev,
                    finished_tokens: &mut self.abc.finished_tokens_dev,
                    counts: &mut self.abc.counts_dev,
                },
            )?;
        }

        // ── Async path: build NEXT step's device control plane on-device ──
        // After the merge has produced `active_src_rows` + `counts`, gather the
        // surviving rows' block tables / kv_lens to the compacted front, append
        // each row's next-step KV slot, advance position/length, and rebuild the
        // decode tile layout — replacing the host O(batch*seq_len) rebuild +
        // upload with O(batch) device work. Runs on the compute stream after the
        // merge; the next step's forward (after the finalize sync) reads the
        // result. `new_slots` are the Stage-1 speculative reservations.
        if let Some(next_slots) = async_next_slots {
            self.ensure_async_ctrl()?;
            let device = self.scope.device();
            // Upload the next-step slots (O(batch)); the kernel uses the first
            // `active` of them (active is device-resident).
            let slots_i32: Vec<i32> = next_slots.iter().map(|&s| s as i32).collect();
            {
                let ctrl = self.async_ctrl.as_ref().expect("async_ctrl ensured");
                unsafe {
                    upload_i32_prefix(device, &ctrl.new_slots_dev, &slots_i32)?;
                }
            }
            let mbps = self.max_blocks_per_seq;
            let cap_batch = slot_batch.unwrap_or(batch).clamp(1, self.cap_batch);
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
            )?;
        }

        // A now holds the committed/compacted tokens (compute). Mark it so the
        // copy-out stream may begin downloading once compute reaches here. (ev_a)
        D::pipeline_record_compute_a(&self.scope)?;

        // ── download the compaction maps on the copy-out stream (So) ──
        // So waits ev_a, then the D2H runs (and may overlap the next step's
        // forward). Fixed `batch`-sized chunks avoid a dependency on the (still
        // on-device) counts. ev_out gates the next step's A overwrite.
        D::pipeline_copy_out_wait_compute_a(&self.scope)?;
        let bytes = batch * std::mem::size_of::<i32>();
        let elem = std::mem::size_of::<i32>();
        unsafe {
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.counts_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.counts_dev.data_ptr() as *const std::ffi::c_void,
                3 * elem,
            )?;
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.active_src_rows_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.active_src_rows_dev.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
            // Active tokens live in A[0..active] after the compaction.
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.active_tokens_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.input_ids_buf.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.finished_src_rows_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.finished_src_rows_dev.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
            D::pipeline_download_d2h_copy_out(
                &self.scope,
                self.abc.finished_tokens_host.as_mut_ptr() as *mut std::ffi::c_void,
                self.abc.finished_tokens_dev.data_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
        }
        D::pipeline_record_copy_out(&self.scope)?; // ev_out on So
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
        D::pipeline_synchronize_copy_out(&self.scope)?; // host mirrors valid before we read them

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
