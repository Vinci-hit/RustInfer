//! `BatchWorkspace` — fixed-capacity, address-stable plan tensors.
//!
//! Owns:
//! - device tensors mirroring `BatchPlan` at the worst-case capacity;
//! - reusable host staging `Vec<i32>` long-lived enough to be safely used
//!   with `MemoryPort::upload_async` (no `cudaStreamSynchronize` needed).
//!
//! `build_plan(seqs)` writes the host staging buffers, async-uploads them
//! into the same device addresses across calls, and returns a `BatchPlan`
//! whose tensors are O(1) `view_raw` shares of the workspace storage.
//! Subsequent CUDA Graph capture/replay over decoders sees stable addresses.
//!
//! # Why domain (not app)
//!
//! Same rationale as `ForwardWorkspace`: it composes only domain-layer
//! types (`Tensor`, `BatchPlan`, `MemoryPort`). DDD-pure.

use super::batch::{BatchKind, BatchPlan, RAGGED_Q_TILE};
use super::ports::{MemoryPort, OpError, OpResult};
use super::tensor::Tensor;
use super::types::Shape;

/// Caller's per-sequence step description.
///
/// Mirrors `app::model_runner::SeqStep`. Kept independent so domain-level
/// tests can construct workspaces without depending on app.
#[derive(Debug, Clone)]
pub struct WsSeqStep {
    pub input_ids: Vec<i32>,
    pub positions: Vec<i32>,
    pub kv_write_start: i32,
    pub kv_len_after: i32,
    pub block_table: Vec<u32>,
}

pub struct BatchWorkspace<D: MemoryPort> {
    pub cap_num_tokens: usize,
    pub cap_batch: usize,
    pub cap_max_blocks_per_seq: usize,
    pub cap_total_q_tiles: usize,

    // Device tensors at MAX capacity (alloc once, address-stable).
    input_ids:      Tensor<i32, D>,
    rope_positions: Tensor<i32, D>,
    cu_q_lens:      Tensor<i32, D>,
    kv_lens:        Tensor<i32, D>,
    seq_positions:  Tensor<i32, D>,
    seq_lens_step:  Tensor<i32, D>,
    block_tables:   Tensor<i32, D>,
    block2req:      Tensor<i32, D>,
    block2tile:     Tensor<i32, D>,

    // Long-lived host staging — owns memory for the runner's lifetime so
    // `upload_async` is safe (the device stream consumes the copy before
    // the next `step_batch`).
    h_input_ids: Vec<i32>,
    h_rope_positions: Vec<i32>,
    h_cu_q_lens: Vec<i32>,
    h_kv_lens: Vec<i32>,
    h_seq_positions: Vec<i32>,
    h_seq_lens_step: Vec<i32>,
    h_block_tables: Vec<i32>,
    h_block2req: Vec<i32>,
    h_block2tile: Vec<i32>,
}

impl<D: MemoryPort> BatchWorkspace<D> {
    pub fn new(
        device: &D,
        cap_num_tokens: usize,
        cap_batch: usize,
        cap_max_blocks_per_seq: usize,
    ) -> OpResult<Self> {
        let cap_total_q_tiles =
            ((cap_num_tokens + RAGGED_Q_TILE as usize - 1) / RAGGED_Q_TILE as usize)
                .max(1)
                .max(cap_batch);
        let alloc1 = |n: usize| -> OpResult<Tensor<i32, D>> {
            Tensor::<i32, D>::zeros(Shape::from_slice(&[n.max(1)]), device)
        };
        Ok(Self {
            cap_num_tokens, cap_batch, cap_max_blocks_per_seq, cap_total_q_tiles,
            input_ids:      alloc1(cap_num_tokens)?,
            rope_positions: alloc1(cap_num_tokens)?,
            cu_q_lens:      alloc1(cap_batch + 1)?,
            kv_lens:        alloc1(cap_batch)?,
            seq_positions:  alloc1(cap_batch)?,
            seq_lens_step:  alloc1(cap_batch)?,
            block_tables:   alloc1(cap_batch * cap_max_blocks_per_seq.max(1))?,
            block2req:      alloc1(cap_total_q_tiles)?,
            block2tile:     alloc1(cap_total_q_tiles)?,
            h_input_ids:      vec![0; cap_num_tokens.max(1)],
            h_rope_positions: vec![0; cap_num_tokens.max(1)],
            h_cu_q_lens:      vec![0; cap_batch + 1],
            h_kv_lens:        vec![0; cap_batch.max(1)],
            h_seq_positions:  vec![0; cap_batch.max(1)],
            h_seq_lens_step:  vec![0; cap_batch.max(1)],
            h_block_tables:   vec![0; cap_batch * cap_max_blocks_per_seq.max(1)],
            h_block2req:      vec![0; cap_total_q_tiles],
            h_block2tile:     vec![0; cap_total_q_tiles],
        })
    }

    /// Fill host staging from `seqs`, async-upload into the workspace's
    /// stable device tensors, return `(input_ids_view, plan)`.
    ///
    /// Errors if any cap is exceeded — the caller can fall back to a
    /// per-step alloc path or surface the error.
    pub fn build_plan(
        &mut self,
        seqs: &[WsSeqStep],
        device: &D,
    ) -> OpResult<(Tensor<i32, D>, BatchPlan<D>)> {
        let batch = seqs.len();
        if batch == 0 {
            return Err(OpError::Shape("BatchWorkspace::build_plan: empty seqs".into()));
        }
        if batch > self.cap_batch {
            return Err(OpError::Shape(format!(
                "BatchWorkspace::build_plan: batch ({}) > cap ({})",
                batch, self.cap_batch,
            )));
        }

        // Validate + collect host metadata.
        let mut total_tokens = 0usize;
        let mut total_q_tiles = 0i32;
        let mut q_lens: Vec<i32> = Vec::with_capacity(batch);
        for (i, s) in seqs.iter().enumerate() {
            if s.input_ids.is_empty() {
                return Err(OpError::Shape("WsSeqStep: empty input_ids".into()));
            }
            if s.input_ids.len() != s.positions.len() {
                return Err(OpError::Shape(format!(
                    "WsSeqStep[{}]: input_ids ({}) != positions ({})",
                    i, s.input_ids.len(), s.positions.len(),
                )));
            }
            if s.block_table.len() > self.cap_max_blocks_per_seq {
                return Err(OpError::Shape(format!(
                    "WsSeqStep[{}]: block_table ({}) > cap ({})",
                    i, s.block_table.len(), self.cap_max_blocks_per_seq,
                )));
            }
            total_tokens += s.input_ids.len();
            q_lens.push(s.input_ids.len() as i32);
            total_q_tiles += (s.input_ids.len() as i32 + RAGGED_Q_TILE - 1) / RAGGED_Q_TILE;
        }
        if total_tokens > self.cap_num_tokens {
            return Err(OpError::Shape(format!(
                "BatchWorkspace::build_plan: total_tokens ({}) > cap ({})",
                total_tokens, self.cap_num_tokens,
            )));
        }
        if total_q_tiles as usize > self.cap_total_q_tiles {
            return Err(OpError::Shape(format!(
                "BatchWorkspace::build_plan: total_q_tiles ({}) > cap ({})",
                total_q_tiles, self.cap_total_q_tiles,
            )));
        }

        // Fill host buffers (zero the prefix that we'll upload, leave the
        // rest at last-step values — the kernel only reads what plan tells
        // it to read).
        // input_ids / rope_positions: prefix [0..total_tokens]
        // cu_q_lens: [batch+1]
        // kv_lens / seq_positions / seq_lens_step: [batch]
        // block_tables: [batch * cap_max_blocks_per_seq]
        // block2req / block2tile: [total_q_tiles]
        self.h_cu_q_lens[0] = 0;
        let mut acc = 0i32;
        for (i, s) in seqs.iter().enumerate() {
            let q_len = s.input_ids.len();
            // input_ids + rope_positions
            self.h_input_ids[acc as usize..acc as usize + q_len]
                .copy_from_slice(&s.input_ids);
            self.h_rope_positions[acc as usize..acc as usize + q_len]
                .copy_from_slice(&s.positions);
            acc += q_len as i32;
            self.h_cu_q_lens[i + 1] = acc;
            self.h_kv_lens[i] = s.kv_len_after;
            self.h_seq_positions[i] = s.kv_write_start;
            self.h_seq_lens_step[i] = q_len as i32;
            // block_tables row
            let row_off = i * self.cap_max_blocks_per_seq;
            // zero unused tail first
            for slot in &mut self.h_block_tables[row_off..row_off + self.cap_max_blocks_per_seq] {
                *slot = 0;
            }
            for (j, &phys) in s.block_table.iter().enumerate() {
                self.h_block_tables[row_off + j] = phys as i32;
            }
        }

        // Build q-tile schedule (block2req / block2tile).
        let mut tile_idx = 0usize;
        for (i, &q) in q_lens.iter().enumerate() {
            let n_tiles = (q + RAGGED_Q_TILE - 1) / RAGGED_Q_TILE;
            for t in 0..n_tiles {
                self.h_block2req[tile_idx] = i as i32;
                self.h_block2tile[tile_idx] = t;
                tile_idx += 1;
            }
        }

        // Async upload only the prefixes we just wrote (avoid unnecessary
        // bandwidth for the unused tail).
        unsafe {
            self.upload_prefix(device, &self.input_ids, &self.h_input_ids[..total_tokens])?;
            self.upload_prefix(device, &self.rope_positions, &self.h_rope_positions[..total_tokens])?;
            self.upload_prefix(device, &self.cu_q_lens, &self.h_cu_q_lens[..batch + 1])?;
            self.upload_prefix(device, &self.kv_lens, &self.h_kv_lens[..batch])?;
            self.upload_prefix(device, &self.seq_positions, &self.h_seq_positions[..batch])?;
            self.upload_prefix(device, &self.seq_lens_step, &self.h_seq_lens_step[..batch])?;
            self.upload_prefix(device, &self.block_tables,
                &self.h_block_tables[..batch * self.cap_max_blocks_per_seq])?;
            if total_q_tiles > 0 {
                self.upload_prefix(device, &self.block2req,
                    &self.h_block2req[..total_q_tiles as usize])?;
                self.upload_prefix(device, &self.block2tile,
                    &self.h_block2tile[..total_q_tiles as usize])?;
            }
        }

        // Build plan with views matching the active sizes.
        let kind = if q_lens.iter().all(|&q| q == 1) {
            BatchKind::DecodeOnly
        } else {
            BatchKind::Ragged
        };
        let plan = BatchPlan {
            kind,
            num_tokens: total_tokens,
            batch,
            rope_positions: Self::view_n(&self.rope_positions, total_tokens),
            cu_q_lens:      Self::view_n(&self.cu_q_lens, batch + 1),
            kv_lens:        Self::view_n(&self.kv_lens, batch),
            seq_positions:  Self::view_n(&self.seq_positions, batch),
            seq_lens_step:  Self::view_n(&self.seq_lens_step, batch),
            block_tables:   Self::view_n(&self.block_tables, batch * self.cap_max_blocks_per_seq),
            max_blocks_per_seq: self.cap_max_blocks_per_seq,
            block_size: 0,  // caller fills (the runner knows the pool block size)
            block2req:  Self::view_n(&self.block2req, total_q_tiles.max(1) as usize),
            block2tile: Self::view_n(&self.block2tile, total_q_tiles.max(1) as usize),
            total_q_tiles,
        };
        let input_ids_view = Self::view_n(&self.input_ids, total_tokens);
        Ok((input_ids_view, plan))
    }

    /// Return plan views for a decode-only batch WITHOUT uploading.
    ///
    /// Used during CUDA Graph capture: the device buffers already contain
    /// valid data from the preceding warmup pass, so we only need the plan
    /// metadata (views into stable addresses). This keeps H2D memcpys
    /// OUT of the captured graph.
    pub fn get_last_plan_views(
        &self,
        batch_size: usize,
        block_size: usize,
    ) -> OpResult<(Tensor<i32, D>, BatchPlan<D>)> {
        if batch_size == 0 || batch_size > self.cap_batch {
            return Err(OpError::Shape(format!(
                "get_last_plan_views: batch_size ({}) out of range [1, {}]",
                batch_size, self.cap_batch,
            )));
        }
        // Decode-only: each seq has q_len=1, so total_tokens = batch_size.
        let total_tokens = batch_size;
        let total_q_tiles = batch_size as i32; // 1 tile per seq for q_len=1

        let kind = BatchKind::DecodeOnly;
        let plan = BatchPlan {
            kind,
            num_tokens: total_tokens,
            batch: batch_size,
            rope_positions: Self::view_n(&self.rope_positions, total_tokens),
            cu_q_lens:      Self::view_n(&self.cu_q_lens, batch_size + 1),
            kv_lens:        Self::view_n(&self.kv_lens, batch_size),
            seq_positions:  Self::view_n(&self.seq_positions, batch_size),
            seq_lens_step:  Self::view_n(&self.seq_lens_step, batch_size),
            block_tables:   Self::view_n(&self.block_tables, batch_size * self.cap_max_blocks_per_seq),
            max_blocks_per_seq: self.cap_max_blocks_per_seq,
            block_size,
            block2req:  Self::view_n(&self.block2req, total_q_tiles.max(1) as usize),
            block2tile: Self::view_n(&self.block2tile, total_q_tiles.max(1) as usize),
            total_q_tiles,
        };
        let input_ids_view = Self::view_n(&self.input_ids, total_tokens);
        Ok((input_ids_view, plan))
    }

    /// Set `plan.block_size` after construction (the runner knows it).
    pub fn block_size(&self) -> usize { 0 } // sentinel; runner sets in plan

    fn view_n(t: &Tensor<i32, D>, n: usize) -> Tensor<i32, D> {
        let strides = Shape::from_slice(&[n.max(1)]).contiguous_strides();
        t.view_raw(Shape::from_slice(&[n]), strides, 0, true)
    }

    /// Async-upload `host[..]` into the prefix of `dev`. Caller asserts
    /// `host.len() <= dev.numel()`.
    unsafe fn upload_prefix(
        &self,
        device: &D,
        dev: &Tensor<i32, D>,
        host: &[i32],
    ) -> OpResult<()> {
        if host.is_empty() { return Ok(()); }
        let bytes = host.len() * std::mem::size_of::<i32>();
        let dst = unsafe {
            std::ptr::NonNull::new_unchecked(dev.data_ptr_mut() as *mut u8)
        };
        unsafe {
            device.upload_async(
                dst,
                host.as_ptr() as *const u8,
                bytes,
            )?;
        }
        Ok(())
    }
}
