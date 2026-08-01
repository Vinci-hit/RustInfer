//! Decode-pipeline port — the ABC pipelined-decode / fused-mixed device
//! surface: compact-merge kernels, the dual-stream copy-in/copy-out event
//! choreography, host pinning, and the capture-safe scratch arena.
//!
//! Every method ships a HOST reference implementation with synchronous
//! semantics (events/pins/arena are no-ops; copies are `memcpy`; the merge
//! kernels are straight-line ports of the CUDA kernels), so the worker's
//! production decode pipeline type-checks and runs against the CPU backend in
//! tests. The CUDA backend overrides everything with the real kernels and its
//! copy-in (Si) / compute / copy-out (So) stream machinery.
//!
//! Reference semantics are pinned by the CUDA kernel unit tests in
//! the CUDA backend's gather/merge kernels — a row is *finished* when
//! `(!ignore_eos && token ∈ eos_ids) || generated + 1 >= max_tokens`; survivors
//! compact to the front of A in row order.

use crate::ports::{OpError, OpResult};
use infer_core::exec::ExecDevice;
use infer_core::tensor::Tensor;

/// Arguments for the pure-decode compact merge (C → A). Output tensors are
/// `&mut`; `counts` receives `[active, finished, old_batch]`.
pub struct MergeCompactDecodeArgs<'a, D: ExecDevice> {
    pub a_out: &'a mut Tensor<i32, D>,
    pub c_prev: &'a Tensor<i32, D>,
    pub generated_counts: &'a Tensor<i32, D>,
    pub max_tokens: &'a Tensor<i32, D>,
    pub ignore_eos: &'a Tensor<i32, D>,
    pub eos_ids: &'a Tensor<i32, D>,
    pub eos_len: usize,
    pub old_batch: usize,
    pub active_src_rows: &'a mut Tensor<i32, D>,
    pub finished_src_rows: &'a mut Tensor<i32, D>,
    pub finished_tokens: &'a mut Tensor<i32, D>,
    pub counts: &'a mut Tensor<i32, D>,
}

/// Arguments for the mixed (ragged) compact merge. Row kinds follow
/// `RaggedRowKind::as_i32`: 0 = Decode, 1 = PrefillFinal, 2 = PrefillCont,
/// 3 = Pad; only kinds 0/1 emit a token. `counts` receives
/// `[active, finished, prefill_final, old_rows]`.
pub struct MergeCompactMixedArgs<'a, D: ExecDevice> {
    pub a_out: &'a mut Tensor<i32, D>,
    pub c_prev: &'a Tensor<i32, D>,
    pub row_kind: &'a Tensor<i32, D>,
    pub generated_counts: &'a Tensor<i32, D>,
    pub max_tokens: &'a Tensor<i32, D>,
    pub ignore_eos: &'a Tensor<i32, D>,
    pub eos_ids: &'a Tensor<i32, D>,
    pub eos_len: usize,
    pub old_rows: usize,
    pub active_src_rows: &'a mut Tensor<i32, D>,
    pub active_tokens: &'a mut Tensor<i32, D>,
    pub finished_src_rows: &'a mut Tensor<i32, D>,
    pub finished_tokens: &'a mut Tensor<i32, D>,
    pub prefill_final_src_rows: &'a mut Tensor<i32, D>,
    pub prefill_final_tokens: &'a mut Tensor<i32, D>,
    pub counts: &'a mut Tensor<i32, D>,
}

/// Arguments for the device-resident next-decode control-plane builder: gather
/// the surviving rows' block tables / kv_lens to the compacted front, append
/// each survivor's next-step KV slot, advance position/length, and rebuild the
/// decode tile layout. Rows `[active, cap_batch)` are zeroed (the attention /
/// scatter kernels iterate control buffers to capacity, so a stale tail would
/// be read as phantom sequences).
pub struct CompactExtendControlArgs<'a, D: ExecDevice> {
    pub block_tables: &'a mut Tensor<i32, D>,
    pub block_tables_scratch: &'a mut Tensor<i32, D>,
    pub kv_lens: &'a mut Tensor<i32, D>,
    pub kv_lens_scratch: &'a mut Tensor<i32, D>,
    pub seq_positions_out: &'a mut Tensor<i32, D>,
    pub seq_lens_step_out: &'a mut Tensor<i32, D>,
    pub rope_positions_out: &'a mut Tensor<i32, D>,
    pub cu_q_lens_out: &'a mut Tensor<i32, D>,
    pub block2req_out: &'a mut Tensor<i32, D>,
    pub block2tile_out: &'a mut Tensor<i32, D>,
    pub active_src_rows: &'a Tensor<i32, D>,
    pub counts: &'a Tensor<i32, D>,
    pub new_slots: &'a Tensor<i32, D>,
    pub mbps: usize,
    pub cap_batch: usize,
}

fn read_i32<D: ExecDevice>(t: &Tensor<i32, D>) -> OpResult<Vec<i32>> {
    t.to_host_vec()
}

fn write_i32<D: ExecDevice>(t: &mut Tensor<i32, D>, values: &[i32]) -> OpResult<()> {
    if values.len() != t.numel() {
        return Err(OpError::Shape(format!(
            "pipeline write: {} values into {} elements",
            values.len(),
            t.numel()
        )));
    }
    t.upload_from_host(values)
}

/// Read-modify-write the first `prefix.len()` elements of `t`.
fn write_i32_prefix<D: ExecDevice>(t: &mut Tensor<i32, D>, prefix: &[i32]) -> OpResult<()> {
    if prefix.is_empty() {
        return Ok(());
    }
    let mut full = t.to_host_vec()?;
    if prefix.len() > full.len() {
        return Err(OpError::Shape(format!(
            "pipeline prefix write: {} values into {} elements",
            prefix.len(),
            full.len()
        )));
    }
    full[..prefix.len()].copy_from_slice(prefix);
    t.upload_from_host(&full)
}

fn row_finished(token: i32, generated: i32, max: i32, ignore: i32, eos: &[i32]) -> bool {
    let hit_eos = ignore == 0 && eos.contains(&token);
    hit_eos || generated.saturating_add(1) >= max
}

/// Decode-pipeline device operations. See the module docs for semantics; the
/// defaults are the synchronous host reference implementation.
pub trait DecodePipelineOps: ExecDevice + Sized {
    /// Copy new-admission seed tokens from B into A at `start..start+count`.
    fn append_decode_admissions(
        _scope: &Self::Scope,
        a_out: &mut Tensor<i32, Self>,
        b_new: &Tensor<i32, Self>,
        start: usize,
        count: usize,
    ) -> OpResult<()> {
        if count == 0 {
            return Ok(());
        }
        let b = read_i32(b_new)?;
        if count > b.len() {
            return Err(OpError::Shape(format!(
                "append_decode_admissions: count {} > B {}",
                count,
                b.len()
            )));
        }
        let mut a = read_i32(a_out)?;
        if start + count > a.len() {
            return Err(OpError::Shape(format!(
                "append_decode_admissions: {}..{} > A {}",
                start,
                start + count,
                a.len()
            )));
        }
        a[start..start + count].copy_from_slice(&b[..count]);
        write_i32(a_out, &a)
    }

    /// Commit decode output C into stable A, compacting surviving rows to the
    /// front (in row order) and splitting finished rows into the side-bands.
    fn merge_compact_decode(
        _scope: &Self::Scope,
        args: MergeCompactDecodeArgs<'_, Self>,
    ) -> OpResult<()> {
        if args.old_batch == 0 {
            return Ok(());
        }
        let c = read_i32(args.c_prev)?;
        let gen_counts = read_i32(args.generated_counts)?;
        let max = read_i32(args.max_tokens)?;
        let ign = read_i32(args.ignore_eos)?;
        let eos_all = read_i32(args.eos_ids)?;
        let eos = &eos_all[..args.eos_len.min(eos_all.len())];

        let mut a_new = Vec::new();
        let mut active_src = Vec::new();
        let mut finished_src = Vec::new();
        let mut finished_tok = Vec::new();
        for row in 0..args.old_batch {
            let token = c[row];
            if row_finished(token, gen_counts[row], max[row], ign[row], eos) {
                finished_src.push(row as i32);
                finished_tok.push(token);
            } else {
                a_new.push(token);
                active_src.push(row as i32);
            }
        }
        write_i32_prefix(args.a_out, &a_new)?;
        write_i32_prefix(args.active_src_rows, &active_src)?;
        write_i32_prefix(args.finished_src_rows, &finished_src)?;
        write_i32_prefix(args.finished_tokens, &finished_tok)?;
        write_i32_prefix(
            args.counts,
            &[
                active_src.len() as i32,
                finished_src.len() as i32,
                args.old_batch as i32,
            ],
        )
    }

    /// Commit mixed ragged output C into flat A: emitting rows (Decode /
    /// PrefillFinal) split into active vs finished; PrefillFinal rows are also
    /// listed in their own side-band regardless of finished state.
    fn merge_compact_mixed(
        _scope: &Self::Scope,
        args: MergeCompactMixedArgs<'_, Self>,
    ) -> OpResult<()> {
        if args.old_rows == 0 {
            return Ok(());
        }
        let c = read_i32(args.c_prev)?;
        let kind = read_i32(args.row_kind)?;
        let gen_counts = read_i32(args.generated_counts)?;
        let max = read_i32(args.max_tokens)?;
        let ign = read_i32(args.ignore_eos)?;
        let eos_all = read_i32(args.eos_ids)?;
        let eos = &eos_all[..args.eos_len.min(eos_all.len())];

        let mut a_new = Vec::new();
        let mut active_src = Vec::new();
        let mut active_tok = Vec::new();
        let mut finished_src = Vec::new();
        let mut finished_tok = Vec::new();
        let mut pf_src = Vec::new();
        let mut pf_tok = Vec::new();
        for row in 0..args.old_rows {
            // 0 = Decode, 1 = PrefillFinal (emit); 2 = PrefillCont, 3 = Pad.
            let emits = kind[row] == 0 || kind[row] == 1;
            if !emits {
                continue;
            }
            let token = c[row];
            if kind[row] == 1 {
                pf_src.push(row as i32);
                pf_tok.push(token);
            }
            if row_finished(token, gen_counts[row], max[row], ign[row], eos) {
                finished_src.push(row as i32);
                finished_tok.push(token);
            } else {
                a_new.push(token);
                active_src.push(row as i32);
                active_tok.push(token);
            }
        }
        write_i32_prefix(args.a_out, &a_new)?;
        write_i32_prefix(args.active_src_rows, &active_src)?;
        write_i32_prefix(args.active_tokens, &active_tok)?;
        write_i32_prefix(args.finished_src_rows, &finished_src)?;
        write_i32_prefix(args.finished_tokens, &finished_tok)?;
        write_i32_prefix(args.prefill_final_src_rows, &pf_src)?;
        write_i32_prefix(args.prefill_final_tokens, &pf_tok)?;
        write_i32_prefix(
            args.counts,
            &[
                active_src.len() as i32,
                finished_src.len() as i32,
                pf_src.len() as i32,
                args.old_rows as i32,
            ],
        )
    }

    /// Build the NEXT decode step's control plane from this step's survivors:
    /// gather block tables / kv_lens by `active_src_rows`, append
    /// `new_slots[k]` at each survivor's current length, advance
    /// positions/lengths, rebuild the decode tile layout, and zero the tail up
    /// to `cap_batch`.
    fn compact_extend_control(
        _scope: &Self::Scope,
        args: CompactExtendControlArgs<'_, Self>,
    ) -> OpResult<()> {
        if args.cap_batch == 0 {
            return Ok(());
        }
        let counts = read_i32(args.counts)?;
        let active = counts.first().copied().unwrap_or(0).max(0) as usize;
        if active > args.cap_batch {
            return Err(OpError::Shape(format!(
                "compact_extend_control: active {} > cap_batch {}",
                active, args.cap_batch
            )));
        }
        let src_rows = read_i32(args.active_src_rows)?;
        let new_slots = read_i32(args.new_slots)?;
        let bt = read_i32(args.block_tables)?;
        let kv = read_i32(args.kv_lens)?;
        let mbps = args.mbps;

        let mut bt_new = vec![0i32; args.cap_batch * mbps];
        let mut kv_new = vec![0i32; args.cap_batch];
        let mut seq_pos = vec![0i32; args.cap_batch];
        let mut seq_lens = vec![0i32; args.cap_batch];
        let mut rope = vec![0i32; args.cap_batch];
        let mut cu = vec![0i32; args.cap_batch + 1];
        let mut b2r = vec![0i32; args.cap_batch];
        let mut b2t = vec![0i32; args.cap_batch];
        for k in 0..active {
            let src = src_rows[k].max(0) as usize;
            let old_len = kv.get(src).copied().unwrap_or(0).max(0) as usize;
            if old_len >= mbps {
                return Err(OpError::Shape(format!(
                    "compact_extend_control: row {} kv_len {} >= mbps {}",
                    k, old_len, mbps
                )));
            }
            let dst = &mut bt_new[k * mbps..(k + 1) * mbps];
            dst.copy_from_slice(&bt[src * mbps..(src + 1) * mbps]);
            dst[old_len] = new_slots[k];
            kv_new[k] = (old_len + 1) as i32;
            seq_pos[k] = old_len as i32;
            seq_lens[k] = 1;
            rope[k] = old_len as i32;
            b2r[k] = k as i32;
            b2t[k] = 0;
        }
        for (k, slot) in cu.iter_mut().enumerate() {
            *slot = k.min(active) as i32;
        }

        write_i32_prefix(args.block_tables_scratch, &bt_new)?;
        write_i32_prefix(args.kv_lens_scratch, &kv_new)?;
        // Copy-back: the live control tensors get the compacted result.
        write_i32_prefix(args.block_tables, &bt_new)?;
        write_i32_prefix(args.kv_lens, &kv_new)?;
        write_i32_prefix(args.seq_positions_out, &seq_pos)?;
        write_i32_prefix(args.seq_lens_step_out, &seq_lens)?;
        write_i32_prefix(args.rope_positions_out, &rope)?;
        write_i32_prefix(args.cu_q_lens_out, &cu)?;
        write_i32_prefix(args.block2req_out, &b2r)?;
        write_i32_prefix(args.block2tile_out, &b2t)
    }

    /// Page-lock a host staging buffer so async copies are truly async.
    /// Host default: no-op.
    fn pipeline_pin_host_i32(_scope: &Self::Scope, _buf: &[i32]) -> OpResult<()> {
        Ok(())
    }

    // ── Dual-stream event choreography. Host defaults: synchronous no-ops. ──
    fn pipeline_record_copy_in(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_compute_wait_copy_in(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_record_compute_a(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_copy_out_wait_compute_a(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_record_copy_out(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_compute_wait_copy_out(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_synchronize_copy_in(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_synchronize_copy_out(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }

    /// H2D on the copy-in stream (Si). Host default: synchronous memcpy.
    ///
    /// # Safety
    /// `dst`/`src` must be valid for `bytes` and non-overlapping; on device
    /// backends `dst` must be device memory and `src` pinned host memory that
    /// outlives the copy.
    unsafe fn pipeline_upload_h2d_copy_in(
        _scope: &Self::Scope,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> OpResult<()> {
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
        Ok(())
    }

    /// D2H on the copy-out stream (So). Host default: synchronous memcpy.
    ///
    /// # Safety
    /// See [`Self::pipeline_upload_h2d_copy_in`], with directions swapped.
    unsafe fn pipeline_download_d2h_copy_out(
        _scope: &Self::Scope,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> OpResult<()> {
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
        Ok(())
    }

    /// Route transient forward scratch through the capture-safe bump arena
    /// (zero device malloc/free). Host default: no-op.
    fn pipeline_arena_begin(_scope: &Self::Scope) -> OpResult<()> {
        Ok(())
    }
    fn pipeline_arena_end(_scope: &Self::Scope) {}
}
