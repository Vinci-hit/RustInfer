use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::model::common::config::RuntimeModelConfig;
use crate::tensor::Tensor;

const DEFAULT_INITIAL_KV_CACHE_LEN: usize = 2048;

pub struct KvCache {
    cache: Vec<(Tensor, Tensor)>,
    capacity: usize,
    max_capacity: usize,
    kv_dim: usize,
    dtype: DataType,
    device: DeviceType,
}

impl KvCache {
    pub fn new(config: &RuntimeModelConfig, device: &DeviceType) -> Result<Self> {
        let kv_dim = config.kv_head_num * config.head_size;
        let dtype = config.runtime_float_dtype(*device)?;
        let max_capacity = config.seq_len;
        let capacity = DEFAULT_INITIAL_KV_CACHE_LEN.min(max_capacity).max(1);
        let cache_shape = [capacity, kv_dim];

        let mut kv_cache = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            let k_cache = Tensor::new(&cache_shape, dtype, *device)?;
            let v_cache = Tensor::new(&cache_shape, dtype, *device)?;
            kv_cache.push((k_cache, v_cache));
        }

        Ok(KvCache {
            cache: kv_cache,
            capacity,
            max_capacity,
            kv_dim,
            dtype,
            device: *device,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Ensure KV cache can store tokens in range `[0, required_len)`.
    ///
    /// Returns `true` if underlying tensors were reallocated. Callers that use
    /// CUDA Graphs must invalidate captured graphs when this returns true,
    /// because graph nodes capture old K/V cache pointers.
    pub fn ensure_capacity(&mut self, required_len: usize) -> Result<bool> {
        if required_len <= self.capacity {
            return Ok(false);
        }
        if required_len > self.max_capacity {
            return Err(Error::InvalidArgument(format!(
                "KV cache required length {} exceeds max capacity {}",
                required_len, self.max_capacity
            )).into());
        }

        let mut new_capacity = self.capacity.max(1);
        while new_capacity < required_len {
            new_capacity = new_capacity.saturating_mul(2);
            if new_capacity >= self.max_capacity {
                new_capacity = self.max_capacity;
                break;
            }
        }
        self.grow_to(new_capacity)?;
        Ok(true)
    }

    fn grow_to(&mut self, new_capacity: usize) -> Result<()> {
        if new_capacity <= self.capacity {
            return Ok(());
        }
        if new_capacity > self.max_capacity {
            return Err(Error::InvalidArgument(format!(
                "KV cache grow target {} exceeds max capacity {}",
                new_capacity, self.max_capacity
            )).into());
        }

        let new_shape = [new_capacity, self.kv_dim];
        let old_shape = [self.capacity, self.kv_dim];
        let mut new_cache = Vec::with_capacity(self.cache.len());

        for (k_old, v_old) in &self.cache {
            let k_new = Tensor::new(&new_shape, self.dtype, self.device)?;
            let v_new = Tensor::new(&new_shape, self.dtype, self.device)?;

            if self.capacity > 0 {
                let k_src = k_old.slice(&[0, 0], &old_shape)?;
                let v_src = v_old.slice(&[0, 0], &old_shape)?;
                let mut k_dst = k_new.slice(&[0, 0], &old_shape)?;
                let mut v_dst = v_new.slice(&[0, 0], &old_shape)?;
                k_dst.copy_from(&k_src)?;
                v_dst.copy_from(&v_src)?;
            }

            new_cache.push((k_new, v_new));
        }

        self.cache = new_cache;
        self.capacity = new_capacity;
        Ok(())
    }

    pub fn slice_kv_cache(
        &mut self,
        layer_idx: usize,
        start_pos: i32,
        len: usize,
        kv_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let start = usize::try_from(start_pos).map_err(|_| {
            Error::InvalidArgument(format!("KV cache start_pos {} is negative", start_pos))
        })?;
        let required_len = start.checked_add(len).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "KV cache range overflow: pos {} + len {}",
                start_pos, len
            ))
        })?;
        self.ensure_capacity(required_len)?;

        if kv_dim != self.kv_dim {
            return Err(Error::InvalidArgument(format!(
                "KV cache kv_dim mismatch: requested {}, cache {}",
                kv_dim, self.kv_dim
            )).into());
        }

        let (k_cache_full, v_cache_full) = self.get_mut(layer_idx)?;
        let k_slice = k_cache_full.slice(&[start, 0], &[len, kv_dim])?;
        let v_slice = v_cache_full.slice(&[start, 0], &[len, kv_dim])?;
        Ok((k_slice, v_slice))
    }

    pub fn get(&self, layer_id: usize) -> Result<(&Tensor, &Tensor)> {
        let (k, v) = self.cache.get(layer_id)
            .ok_or_else(|| Error::InvalidArgument(format!("Layer {} out of bounds for KV cache", layer_id)))?;
        Ok((k, v))
    }

    pub fn get_mut(&mut self, layer_id: usize) -> Result<(&mut Tensor, &mut Tensor)> {
        let (k, v) = self.cache.get_mut(layer_id)
            .ok_or_else(|| Error::InvalidArgument(format!("Layer {} out of bounds for KV cache", layer_id)))?;
        Ok((k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};

    fn test_config(seq_len: usize) -> RuntimeModelConfig {
        RuntimeModelConfig {
            dim: 8,
            intermediate_size: 16,
            layer_num: 1,
            head_num: 2,
            kv_head_num: 1,
            seq_len,
            vocab_size: 32,
            kv_dim: 4,
            kv_mul: 2,
            head_size: 4,
            q_dim: 8,
            is_shared_weight: true,
            torch_dtype: "float32".to_string(),
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tokenizer_vocab_size: 32,
            immediate_dim: None,
            quant_config: None,
            rope_scaling: None,
        }
    }

    #[test]
    fn kv_cache_grows_and_preserves_prefix() -> Result<()> {
        // seq_len must be >= DEFAULT_INITIAL_KV_CACHE_LEN so the initial
        // capacity equals the default (the constructor clamps to seq_len).
        let max_capacity = DEFAULT_INITIAL_KV_CACHE_LEN * 2;
        let cfg = test_config(max_capacity);
        let mut cache = KvCache::new(&cfg, &DeviceType::Cpu)?;
        assert_eq!(cache.capacity(), DEFAULT_INITIAL_KV_CACHE_LEN);

        {
            let (k, v) = cache.get_mut(0)?;
            k.as_f32_mut()?.as_slice_mut()?[0] = 42.0;
            v.as_f32_mut()?.as_slice_mut()?[0] = 24.0;
        }

        assert!(cache.ensure_capacity(DEFAULT_INITIAL_KV_CACHE_LEN + 1)?);
        // Growth doubles 2048 -> 4096, which equals max_capacity.
        assert_eq!(cache.capacity(), max_capacity);

        let (k, v) = cache.get(0)?;
        assert_eq!(k.dtype(), DataType::F32);
        assert_eq!(k.shape(), &[max_capacity, 4]);
        assert_eq!(k.as_f32()?.as_slice()?[0], 42.0);
        assert_eq!(v.as_f32()?.as_slice()?[0], 24.0);
        Ok(())
    }
}
