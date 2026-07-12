//! `StepRequest` validation → `BatchPlan`, plus the per-step control-plane
//! index upload into the address-stable `kv_index` tensors.

use crate::domain::dtype::Dtype;
use crate::domain::exec::ExecScope;
use crate::domain::model::DecoderModel;
use crate::domain::plan::{BatchKind, BatchPlan, StepRequest};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

use super::{
    Runtime, upload_i32_full_zeropad, upload_i32_prefix, upload_i32_range,
    validate_step_request_vectors,
};

impl<T, D, M> Runtime<T, D, M>
where
    T: Dtype,
    D: LlmBackend,
    M: DecoderModel<T, D>,
{
    pub(super) fn build_plan(&self, req: &StepRequest) -> OpResult<BatchPlan> {
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
            let expected_after =
                start
                    .checked_add(seq.input_ids.len() as u32)
                    .ok_or_else(|| {
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

        // `total_q_tiles` is the only thing the plan needs from the ragged-tile
        // layout; compute it arithmetically instead of building (and discarding)
        // the three `plan_ragged_tiles` Vecs here. `upload_index` builds them
        // once, when it actually uploads them.
        let tile = crate::domain::plan::RAGGED_Q_TILE;
        let total_q_tiles: i32 = q_lens.iter().map(|&q| (q + tile - 1) / tile).sum();
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
            total_q_tiles,
        })
    }

    pub(super) fn input_ids_tensor(
        &mut self,
        req: &StepRequest,
        plan: &BatchPlan,
    ) -> OpResult<Tensor<i32, D>> {
        let n = plan.num_tokens;
        if n > self.prefill_ids_host.len() {
            return Err(OpError::Shape(format!(
                "input_ids_tensor: num_tokens {} > cap {}",
                n,
                self.prefill_ids_host.len()
            )));
        }
        // Fill the persistent host staging prefix, then async-upload into the
        // fixed device buffer (NO cudaStreamSynchronize). Same compute stream as
        // the forward, so the H2D is ordered before the kernels that read it;
        // the host buffer is reused (not freed), so no flush-sync is needed.
        let mut off = 0usize;
        for seq in &req.seqs {
            let len = seq.input_ids.len();
            self.prefill_ids_host[off..off + len].copy_from_slice(&seq.input_ids);
            off += len;
        }
        unsafe {
            upload_i32_prefix(
                self.scope.device(),
                &self.prefill_ids_buf,
                &self.prefill_ids_host[..n],
            )?;
        }
        Ok(self.prefill_ids_buf.view_raw(
            Shape::from_slice(&[n]),
            Shape::from_slice(&[n.max(1)]).contiguous_strides(),
            0,
            true,
        ))
    }

    pub(super) fn upload_input_ids_bucket(
        &mut self,
        req: &StepRequest,
        actual_tokens: usize,
        token_bucket: usize,
    ) -> OpResult<Tensor<i32, D>> {
        if actual_tokens > token_bucket {
            return Err(OpError::Shape(format!(
                "upload_input_ids_bucket: actual_tokens {} > bucket {}",
                actual_tokens, token_bucket
            )));
        }
        if token_bucket > self.prefill_ids_host.len() {
            return Err(OpError::Shape(format!(
                "upload_input_ids_bucket: bucket {} > cap {}",
                token_bucket,
                self.prefill_ids_host.len()
            )));
        }
        let mut off = 0usize;
        for seq in &req.seqs {
            let len = seq.input_ids.len();
            self.prefill_ids_host[off..off + len].copy_from_slice(&seq.input_ids);
            off += len;
        }
        debug_assert_eq!(off, actual_tokens);
        if token_bucket > off {
            self.prefill_ids_host[off..token_bucket].fill(0);
        }
        unsafe {
            upload_i32_prefix(
                self.scope.device(),
                &self.prefill_ids_buf,
                &self.prefill_ids_host[..token_bucket],
            )?;
        }
        Ok(self.prefill_ids_buf.view_raw(
            Shape::from_slice(&[token_bucket]),
            Shape::from_slice(&[token_bucket.max(1)]).contiguous_strides(),
            0,
            true,
        ))
    }

    /// Like `upload_input_ids_bucket`, but skips the first `skip_tokens` flat
    /// tape entries: those device rows are filled by an on-device gather (the
    /// overlapped fused path's C-prefix — the prior step's argmax output),
    /// so only `[skip_tokens, token_bucket)` is staged and uploaded. The
    /// request carries placeholder ids for the skipped rows.
    pub(super) fn upload_input_ids_suffix(
        &mut self,
        req: &StepRequest,
        skip_tokens: usize,
        actual_tokens: usize,
        token_bucket: usize,
    ) -> OpResult<Tensor<i32, D>> {
        if skip_tokens > actual_tokens || actual_tokens > token_bucket {
            return Err(OpError::Shape(format!(
                "upload_input_ids_suffix: skip {} actual {} bucket {}",
                skip_tokens, actual_tokens, token_bucket
            )));
        }
        if token_bucket > self.prefill_ids_host.len() {
            return Err(OpError::Shape(format!(
                "upload_input_ids_suffix: bucket {} > cap {}",
                token_bucket,
                self.prefill_ids_host.len()
            )));
        }
        let mut off = 0usize;
        for seq in &req.seqs {
            let len = seq.input_ids.len();
            if off + len > skip_tokens {
                let from = skip_tokens.saturating_sub(off);
                self.prefill_ids_host[off + from..off + len]
                    .copy_from_slice(&seq.input_ids[from..]);
            }
            off += len;
        }
        debug_assert_eq!(off, actual_tokens);
        if token_bucket > off {
            self.prefill_ids_host[off..token_bucket].fill(0);
        }
        unsafe {
            upload_i32_range(
                self.scope.device(),
                &self.prefill_ids_buf,
                skip_tokens,
                &self.prefill_ids_host[skip_tokens..token_bucket],
            )?;
        }
        Ok(self.prefill_ids_buf.view_raw(
            Shape::from_slice(&[token_bucket]),
            Shape::from_slice(&[token_bucket.max(1)]).contiguous_strides(),
            0,
            true,
        ))
    }

    pub(super) fn upload_index(&mut self, plan: &BatchPlan, req: &StepRequest) -> OpResult<()> {
        self.upload_index_with_suffix_prefix(plan, req, None)
    }

    pub(super) fn upload_index_with_suffix_prefix(
        &mut self,
        plan: &BatchPlan,
        req: &StepRequest,
        suffix_prefix_tiles: Option<usize>,
    ) -> OpResult<()> {
        let (cu_q_lens, block2req, block2tile) = BatchPlan::plan_ragged_tiles(&plan.q_lens);

        // Refresh the persistent block-table staging IN PLACE — no per-step
        // heap alloc/zero of a `batch * max_blocks_per_seq` Vec (1–4 MiB in the
        // decode hot loop), and the buffer outlives the async H2D below so the
        // `upload_async` (cudaMemcpyAsync) host-pointer contract holds.
        //
        // Each row writes only its own `block_table` entries; the kernels bound
        // every block-table read by the sequence's live length —
        // `qkv_norm_rope_scatter` walks `logical_pos/block_size` for
        // `t < seq_lens[seq]`, paged attention walks `logical_block < kv_len` —
        // so entries past a row's live length are never read and need not be
        // cleared. (Unlike the per-seq control buffers below, which the kernels
        // iterate to capacity and therefore must be zero-padded.)
        let mbps = self.max_blocks_per_seq;
        let upload_len = plan.batch * mbps;
        let host_index = self.block_tables_host_next;
        self.block_tables_host_next = (host_index + 1) % self.block_tables_host.len();
        {
            let block_tables_host = &mut self.block_tables_host[host_index];
            for (i, seq) in req.seqs.iter().enumerate() {
                let row = i * mbps;
                for (j, &block) in seq.block_table.iter().enumerate() {
                    block_tables_host[row + j] = block as i32;
                }
            }
        }

        let device = self.scope.device();
        unsafe {
            upload_i32_prefix(
                device,
                &self.kv_index.block_tables,
                &self.block_tables_host[host_index][..upload_len],
            )?;
            // The per-sequence control buffers are sized to `cap_batch` but the
            // attention/scatter kernels iterate over `seq_positions.shape()[0]`
            // (== capacity). A prefix-only upload leaves the tail holding STALE
            // values from a prior, larger batch — phantom sequences then claim
            // real tokens via stale `cu_q_lens`, re-applying RoPE in-place at the
            // wrong position and corrupting Q/K (manifests as persistent garbage
            // for every later request after one high-batch step). Upload these
            // zero-padded to full capacity so phantom rows are inert (0 length).
            upload_i32_full_zeropad(device, &self.kv_index.cu_q_lens, &cu_q_lens)?;
            upload_i32_full_zeropad(device, &self.kv_index.kv_lens, &plan.kv_lens)?;
            upload_i32_full_zeropad(device, &self.kv_index.seq_positions, &plan.seq_positions)?;
            // `seq_lens_step` is the per-row q_len; upload `plan.q_lens`
            // directly instead of cloning it each step.
            upload_i32_full_zeropad(device, &self.kv_index.seq_lens_step, &plan.q_lens)?;
            upload_i32_prefix(device, &self.kv_index.rope_positions, &plan.rope_positions)?;
            upload_i32_prefix(device, &self.kv_index.block2req, &block2req)?;
            upload_i32_prefix(device, &self.kv_index.block2tile, &block2tile)?;
            let valid_q_tiles = [block2req.len() as i32];
            let actual_decode_prefix_tiles = plan.q_lens.iter().take_while(|&&q| q == 1).count();
            let decode_prefix_tiles = suffix_prefix_tiles
                .unwrap_or(actual_decode_prefix_tiles)
                .min(block2req.len()) as i32;
            let valid_suffix_q_tiles = [(valid_q_tiles[0] - decode_prefix_tiles).max(0)];
            upload_i32_prefix(device, &self.kv_index.valid_q_tiles, &valid_q_tiles)?;
            upload_i32_prefix(
                device,
                &self.kv_index.valid_suffix_q_tiles,
                &valid_suffix_q_tiles,
            )?;
        }
        Ok(())
    }
}
