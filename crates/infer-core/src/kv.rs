use std::collections::HashMap;

use crate::component::LayerRange;
use infer_core::dtype::Dtype;
use infer_core::dtype::quant::QuantScheme;
use infer_core::exec::ExecDevice as Device;
use crate::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum KvQuantTier {
    None,
    PerTensor(QuantScheme),
    PerBlock { scheme: QuantScheme },
}

pub struct PagedKvLayer<T: Dtype, D: Device> {
    pub k: Tensor<T, D>,
    pub v: Tensor<T, D>,
}

pub struct KvIndexTensors<D: Device> {
    pub block_tables: Tensor<i32, D>,
    pub cu_q_lens: Tensor<i32, D>,
    pub kv_lens: Tensor<i32, D>,
    pub seq_positions: Tensor<i32, D>,
    pub seq_lens_step: Tensor<i32, D>,
    pub rope_positions: Tensor<i32, D>,
    pub block2req: Tensor<i32, D>,
    pub block2tile: Tensor<i32, D>,
}

pub struct PagedKvPool<T: Dtype, D: Device> {
    pub layers: Vec<PagedKvLayer<T, D>>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub kv_dim: usize,
    pub quant: KvQuantTier,
    pub seq_kv_len: HashMap<SeqId, u32>,
}

impl<T: Dtype, D: Device> PagedKvPool<T, D> {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn view<'a>(
        &'a mut self,
        range: LayerRange,
        index: &'a KvIndexTensors<D>,
    ) -> KvView<'a, T, D> {
        assert!(range.start <= range.end, "invalid layer range");
        assert!(range.end <= self.layers.len(), "layer range out of bounds");
        KvView {
            layers: &mut self.layers[range.start..range.end],
            index,
            num_blocks: self.num_blocks,
            block_size: self.block_size,
            kv_dim: self.kv_dim,
            quant: &self.quant,
        }
    }

    pub fn edit(&mut self) -> KvEdit<'_, T, D> {
        KvEdit { pool: self }
    }
}

pub struct KvView<'a, T: Dtype, D: Device> {
    pub layers: &'a mut [PagedKvLayer<T, D>],
    pub index: &'a KvIndexTensors<D>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub kv_dim: usize,
    pub quant: &'a KvQuantTier,
}

impl<'a, T: Dtype, D: Device> KvView<'a, T, D> {
    pub fn single_layer(&mut self, layer_idx: usize) -> KvView<'_, T, D> {
        let layer = std::slice::from_mut(&mut self.layers[layer_idx]);
        KvView {
            layers: layer,
            index: self.index,
            num_blocks: self.num_blocks,
            block_size: self.block_size,
            kv_dim: self.kv_dim,
            quant: self.quant,
        }
    }

    pub fn layer_mut(&mut self, layer_idx: usize) -> LayerKv<'_, T, D> {
        let layer = &mut self.layers[layer_idx];
        LayerKv {
            k: &mut layer.k,
            v: &mut layer.v,
            index: self.index,
        }
    }

    pub fn layer(&self, layer_idx: usize) -> (&Tensor<T, D>, &Tensor<T, D>) {
        let layer = &self.layers[layer_idx];
        (&layer.k, &layer.v)
    }
}

pub struct LayerKv<'a, T: Dtype, D: Device> {
    pub k: &'a mut Tensor<T, D>,
    pub v: &'a mut Tensor<T, D>,
    pub index: &'a KvIndexTensors<D>,
}

pub struct KvEdit<'a, T: Dtype, D: Device> {
    pub pool: &'a mut PagedKvPool<T, D>,
}

impl<'a, T: Dtype, D: Device> KvEdit<'a, T, D> {
    pub fn append(&mut self, sid: SeqId, n: u32) -> OpResult<()> {
        let len = self.pool.seq_kv_len.entry(sid).or_insert(0);
        *len = len
            .checked_add(n)
            .ok_or_else(|| OpError::Shape(format!("kv_len overflow for seq {}", sid)))?;
        Ok(())
    }

    pub fn truncate(&mut self, sid: SeqId, to: u32) -> OpResult<Vec<u32>> {
        self.pool.seq_kv_len.insert(sid, to);
        Ok(Vec::new())
    }

    pub fn apply_step(
        &mut self,
        sids: &[SeqId],
        accepted: &[u32],
        speculative_len: &[u32],
    ) -> OpResult<Vec<u32>> {
        if sids.len() != accepted.len() {
            return Err(OpError::Shape(format!(
                "KvEdit::apply_step: sids={} accepted={}",
                sids.len(),
                accepted.len()
            )));
        }
        let mut freed = Vec::new();
        for (i, (&sid, &accepted_count)) in sids.iter().zip(accepted.iter()).enumerate() {
            let spec = speculative_len.get(i).copied().unwrap_or(0);
            if spec > 0 {
                if accepted_count > spec {
                    return Err(OpError::Shape(format!(
                        "KvEdit::apply_step: accepted {} > speculative {} for seq {}",
                        accepted_count, spec, sid
                    )));
                }
                let current = self.pool.seq_kv_len.get(&sid).copied().unwrap_or(0);
                if current < spec {
                    return Err(OpError::Shape(format!(
                        "KvEdit::apply_step: current kv_len {} < speculative {} for seq {}",
                        current, spec, sid
                    )));
                }
                let base = current - spec;
                let keep = base + accepted_count;
                if keep < current {
                    freed.extend(self.truncate(sid, keep)?);
                }
            } else {
                self.append(sid, accepted_count)?;
            }
        }
        Ok(freed)
    }
}

pub type SeqId = u64;
