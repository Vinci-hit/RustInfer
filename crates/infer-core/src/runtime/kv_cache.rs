use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::model::config::RuntimeModelConfig;
use crate::tensor::Tensor;

pub struct KvCache {
    cache: Vec<(Tensor, Tensor)>,
}

impl KvCache {
    pub fn new(config: &RuntimeModelConfig, device: &DeviceType) -> Result<Self> {
        let cache_shape = vec![
            config.seq_len,
            config.kv_head_num * config.head_size,
        ];

        let float_type = config.runtime_float_dtype(*device)?;

        let mut kv_cache = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            let k_cache = Tensor::new(&cache_shape, float_type, *device)?;
            let v_cache = Tensor::new(&cache_shape, float_type, *device)?;
            kv_cache.push((k_cache, v_cache));
        }

        Ok(KvCache { cache: kv_cache })
    }

    pub fn slice_kv_cache(
        &mut self,
        layer_idx: usize,
        start_pos: i32,
        len: usize,
        kv_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (k_cache_full, v_cache_full) = self.get_mut(layer_idx)?;
        let max_seq_len = k_cache_full.shape()[0];

        if start_pos as usize + len > max_seq_len {
            return Err(anyhow::anyhow!(
                "KV cache slice out of bounds: pos {} + len {} > max {}",
                start_pos, len, max_seq_len
            ));
        }

        let k_slice = k_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;
        let v_slice = v_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;
        Ok((k_slice, v_slice))
    }

    pub fn get(&self, layer_id: usize) -> Result<(&Tensor, &Tensor)> {
        let (k, v) = self.cache.get(layer_id)
            .ok_or_else(|| anyhow::anyhow!("Layer {} out of bounds for KV cache", layer_id))?;
        Ok((k, v))
    }

    pub fn get_mut(&mut self, layer_id: usize) -> Result<(&mut Tensor, &mut Tensor)> {
        let (k, v) = self.cache.get_mut(layer_id)
            .ok_or_else(|| anyhow::anyhow!("Layer {} out of bounds for KV cache", layer_id))?;
        Ok((k, v))
    }
}
