//! Startup-owned MTP control/readout buffers, complementing ForwardScratch.
//! All views have stable storage. Host metadata is packed into one upload per
//! round, and draft token IDs stay on device between head invocations.
use super::dtype::Dtype;
use super::kv::KvIndexTensors;
use super::model::ModelDims;
use super::plan::{BatchKind, BatchPlan, RAGGED_Q_TILE};
use super::ports::OpResult;
use super::ports::backend::LlmBackend;
use super::tensor::Tensor;

pub(crate) struct MtpWorkspace<T: Dtype, D: LlmBackend> {
    control: Tensor<i32, D>,
    control_host: Vec<i32>,
    blocks: Tensor<i32, D>,
    ids: Tensor<i32, D>,
    pending_offset: usize,
    hidden: [Tensor<T, D>; 2],
    positions_host: Vec<i32>,
    pub logits: Tensor<T, D>,
    pub argmax_ws: Tensor<f32, D>,
    pub plan: BatchPlan,
}

impl<T: Dtype, D: LlmBackend> MtpWorkspace<T, D> {
    pub fn new(dims: ModelDims, context: usize, capacity: usize, device: &D) -> OpResult<Self> {
        let blocks = context.div_ceil(16);
        let tiles = capacity.div_ceil(RAGGED_Q_TILE as usize);
        let control_size = (capacity * 10 + 1).max(7 + capacity + tiles * 2);
        Ok(Self {
            control: Tensor::zeros([control_size], device)?,
            control_host: vec![0; control_size],
            blocks: Tensor::from_host_slice(
                &(0..blocks as i32).collect::<Vec<_>>(),
                [1, blocks],
                device,
            )?,
            ids: Tensor::zeros([capacity + 1], device)?,
            pending_offset: 0,
            hidden: [
                Tensor::zeros([capacity, dims.dim], device)?,
                Tensor::zeros([capacity, dims.dim], device)?,
            ],
            positions_host: vec![0; capacity],
            logits: Tensor::zeros([1, dims.vocab_size], device)?,
            argmax_ws: Tensor::zeros([512], device)?,
            plan: BatchPlan {
                kind: BatchKind::DecodeOnly,
                num_tokens: 1,
                batch: 1,
                q_lens: vec![1],
                kv_lens: vec![1],
                seq_positions: vec![0],
                rope_positions: Vec::with_capacity(capacity),
                max_blocks_per_seq: blocks,
                block_size: 16,
                total_q_tiles: 1,
            },
        })
    }
    pub fn positions(&mut self, start: usize, n: usize) -> &[i32] {
        for (i, p) in self.positions_host[..n].iter_mut().enumerate() {
            *p = (start + i) as i32;
        }
        &self.positions_host[..n]
    }
    fn set_plan(&mut self, start: usize, n: usize) {
        self.plan.kind = if n == 1 {
            BatchKind::DecodeOnly
        } else {
            BatchKind::Ragged
        };
        self.plan.num_tokens = n;
        self.plan.q_lens[0] = n as i32;
        self.plan.kv_lens[0] = (start + n) as i32;
        self.plan.seq_positions[0] = start as i32;
        self.plan.rope_positions.clear();
        self.plan
            .rope_positions
            .extend(start as i32..(start + n) as i32);
        self.plan.total_q_tiles = n.div_ceil(RAGGED_Q_TILE as usize) as i32;
    }
    pub fn set_decode_position(&mut self, start: usize) {
        self.set_plan(start, 1);
    }
    pub fn prepare_draft(&mut self, pending: i32, start: usize, count: usize) -> OpResult<()> {
        for i in 0..count {
            let p = (start + i) as i32;
            self.control_host[i * 10..(i + 1) * 10].copy_from_slice(&[
                0,
                1,
                p + 1,
                p,
                1,
                p,
                0,
                0,
                1,
                1,
            ]);
        }
        self.pending_offset = count * 10;
        self.control_host[self.pending_offset] = pending;
        self.control
            .narrow(0, 0, self.pending_offset + 1)?
            .upload_from_host(&self.control_host[..=self.pending_offset])
    }
    pub fn prepare_observe(&mut self, ids: &[i32], start: usize) -> OpResult<()> {
        let n = ids.len();
        self.set_plan(start, n);
        let tiles = self.plan.total_q_tiles as usize;
        self.control_host[..5].copy_from_slice(&[
            0,
            n as i32,
            (start + n) as i32,
            start as i32,
            n as i32,
        ]);
        for i in 0..n {
            self.control_host[5 + i] = (start + i) as i32;
        }
        self.control_host[5 + n..5 + n + tiles].fill(0);
        for i in 0..tiles {
            self.control_host[5 + n + tiles + i] = i as i32;
        }
        self.control_host[5 + n + 2 * tiles..7 + n + 2 * tiles].fill(tiles as i32);
        let used = 7 + n + 2 * tiles;
        self.control
            .narrow(0, 0, used)?
            .upload_from_host(&self.control_host[..used])?;
        self.input(n)?.upload_from_host(ids)
    }
    pub fn index(&self, slot: usize, n: usize) -> OpResult<KvIndexTensors<D>> {
        let tiles = n.div_ceil(RAGGED_Q_TILE as usize);
        let base = slot * 10;
        let v = |offset, len| self.control.narrow(0, base + offset, len);
        Ok(KvIndexTensors {
            block_tables: self.blocks.clone(),
            cu_q_lens: v(0, 2)?,
            kv_lens: v(2, 1)?,
            seq_positions: v(3, 1)?,
            seq_lens_step: v(4, 1)?,
            rope_positions: v(5, n)?,
            block2req: v(5 + n, tiles)?,
            block2tile: v(5 + n + tiles, tiles)?,
            valid_q_tiles: v(5 + n + tiles * 2, 1)?,
            valid_suffix_q_tiles: v(6 + n + tiles * 2, 1)?,
        })
    }
    pub fn input(&self, n: usize) -> OpResult<Tensor<i32, D>> {
        self.ids.narrow(0, 0, n)
    }
    pub fn token(&self, i: usize) -> OpResult<Tensor<i32, D>> {
        if i == 0 {
            self.control.narrow(0, self.pending_offset, 1)
        } else {
            self.ids.narrow(0, i, 1)
        }
    }
    pub fn drafts(&self, n: usize) -> OpResult<Tensor<i32, D>> {
        self.ids.narrow(0, 1, n)
    }
    pub fn hidden(&self, slot: usize, n: usize) -> OpResult<Tensor<T, D>> {
        self.hidden[slot].narrow(0, 0, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn packed_indices_cross_tile_boundaries_and_reuse_storage() {
        let tile = RAGGED_Q_TILE as usize;
        let mut ws = MtpWorkspace::<f32, _>::new(
            ModelDims {
                dim: 4,
                vocab_size: 8,
                ..Default::default()
            },
            256,
            tile * 2 + 1,
            &Cpu,
        )
        .unwrap();
        let control = ws.control.data_ptr();
        let blocks = ws.blocks.data_ptr();
        let hidden = ws.hidden(0, 1).unwrap().data_ptr();
        for n in [1, tile, tile + 1, tile * 2 + 1, 2, 1] {
            ws.prepare_observe(&vec![3; n], 7).unwrap();
            let ix = ws.index(0, n).unwrap();
            assert_eq!(ix.cu_q_lens.to_host_vec().unwrap(), [0, n as i32]);
            assert_eq!(ix.kv_lens.to_host_vec().unwrap(), [(7 + n) as i32]);
            assert_eq!(
                ix.rope_positions.to_host_vec().unwrap(),
                (7..(7 + n) as i32).collect::<Vec<_>>()
            );
            let tiles = n.div_ceil(tile);
            assert_eq!(
                ix.block2tile.to_host_vec().unwrap(),
                (0..tiles as i32).collect::<Vec<_>>()
            );
            assert_eq!(ix.valid_q_tiles.to_host_vec().unwrap(), [tiles as i32]);
            assert_eq!(
                ix.valid_suffix_q_tiles.to_host_vec().unwrap(),
                [tiles as i32]
            );
            assert_eq!(ix.block2req.to_host_vec().unwrap(), vec![0; tiles]);
            assert_eq!(ws.control.data_ptr(), control);
            assert_eq!(ws.blocks.data_ptr(), blocks);
            assert_eq!(ws.hidden(0, 1).unwrap().data_ptr(), hidden);
        }
        ws.prepare_draft(5, 29, 3).unwrap();
        for i in 0..3 {
            let ix = ws.index(i, 1).unwrap();
            assert_eq!(ix.kv_lens.to_host_vec().unwrap(), [30 + i as i32]);
            assert_eq!(ix.rope_positions.to_host_vec().unwrap(), [29 + i as i32]);
            assert_eq!(ix.valid_q_tiles.to_host_vec().unwrap(), [1]);
        }
        assert_eq!(ws.token(0).unwrap().to_host_vec().unwrap(), [5]);
        assert_ne!(
            ws.hidden(0, 1).unwrap().data_ptr(),
            ws.hidden(1, 1).unwrap().data_ptr()
        );
    }
}
