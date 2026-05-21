use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::model::common::config::RuntimeModelConfig;
use crate::tensor::Tensor;

/// One model layer's global paged KV storage.
pub struct PagedKvLayer {
    pub k: Tensor,
    pub v: Tensor,
}

/// Worker-owned physical Paged KV pool.
///
/// The scheduler owns logical block allocation. This pool owns the actual GPU/CPU
/// tensors for all physical blocks. Runtime decode block requests only extend a
/// sequence's block table; they do not allocate per-request KV tensors.
pub struct PagedKvPool {
    block_size: usize,
    num_blocks: usize,
    max_blocks_per_seq: usize,
    layer_num: usize,
    kv_dim: usize,
    dtype: DataType,
    device: DeviceType,
    bytes_allocated: usize,
    layers: Vec<PagedKvLayer>,
}

impl PagedKvPool {
    pub fn new(
        config: &RuntimeModelConfig,
        device: DeviceType,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(Error::InvalidArgument("PagedKvPool block_size must be > 0".into()).into());
        }
        if num_blocks == 0 {
            return Err(Error::InvalidArgument("PagedKvPool num_blocks must be > 0".into()).into());
        }

        let kv_dim = config.kv_head_num * config.head_size;
        let dtype = config.runtime_float_dtype(device)?;
        let shape = [num_blocks, block_size, kv_dim];
        let mut layers = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            layers.push(PagedKvLayer {
                k: Tensor::new(&shape, dtype, device)?,
                v: Tensor::new(&shape, dtype, device)?,
            });
        }

        let bytes_allocated = Self::bytes_for_pool(
            config.layer_num,
            num_blocks,
            block_size,
            kv_dim,
            dtype.size_in_bytes(),
        );

        Ok(Self {
            block_size,
            num_blocks,
            max_blocks_per_seq: config.seq_len.div_ceil(block_size),
            layer_num: config.layer_num,
            kv_dim,
            dtype,
            device,
            bytes_allocated,
            layers,
        })
    }

    pub fn bytes_for_pool(
        layer_num: usize,
        num_blocks: usize,
        block_size: usize,
        kv_dim: usize,
        dtype_size: usize,
    ) -> usize {
        layer_num * 2 * num_blocks * block_size * kv_dim * dtype_size
    }

    pub fn bytes_per_block_all_layers(config: &RuntimeModelConfig, device: DeviceType, block_size: usize) -> Result<usize> {
        if block_size == 0 {
            return Err(Error::InvalidArgument("block_size must be > 0".into()).into());
        }
        let kv_dim = config.kv_head_num * config.head_size;
        let dtype_size = config.runtime_float_dtype(device)?.size_in_bytes();
        Ok(Self::bytes_for_pool(config.layer_num, 1, block_size, kv_dim, dtype_size))
    }

    /// Number of physical blocks fitting into `kv_budget_bytes`.
    /// The caller should already pass a budget such as 95% of post-profile free memory.
    pub fn num_blocks_from_budget(
        config: &RuntimeModelConfig,
        device: DeviceType,
        block_size: usize,
        kv_budget_bytes: usize,
    ) -> Result<usize> {
        let bytes_per_block = Self::bytes_per_block_all_layers(config, device, block_size)?;
        Ok(kv_budget_bytes / bytes_per_block)
    }

    pub fn block_size(&self) -> usize { self.block_size }
    pub fn num_blocks(&self) -> usize { self.num_blocks }
    pub fn max_blocks_per_seq(&self) -> usize { self.max_blocks_per_seq }
    pub fn layer_num(&self) -> usize { self.layer_num }
    pub fn kv_dim(&self) -> usize { self.kv_dim }
    pub fn dtype(&self) -> DataType { self.dtype }
    pub fn device(&self) -> DeviceType { self.device }
    pub fn bytes_allocated(&self) -> usize { self.bytes_allocated }
    pub fn layers(&self) -> &[PagedKvLayer] { &self.layers }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(seq_len: usize) -> RuntimeModelConfig {
        RuntimeModelConfig {
            dim: 8,
            intermediate_size: 16,
            layer_num: 2,
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
    fn bytes_per_block_matches_formula() -> Result<()> {
        let cfg = test_config(128);
        let bytes = PagedKvPool::bytes_per_block_all_layers(&cfg, DeviceType::Cpu, 16)?;
        // layer_num * K/V * block_size * kv_dim * f32_size
        assert_eq!(bytes, 2 * 2 * 16 * 4 * 4);
        Ok(())
    }

    #[test]
    fn budget_to_num_blocks() -> Result<()> {
        let cfg = test_config(128);
        let per_block = PagedKvPool::bytes_per_block_all_layers(&cfg, DeviceType::Cpu, 16)?;
        assert_eq!(PagedKvPool::num_blocks_from_budget(&cfg, DeviceType::Cpu, 16, per_block * 7 + 1)?, 7);
        Ok(())
    }

    #[test]
    fn allocates_global_pool_per_layer() -> Result<()> {
        let cfg = test_config(128);
        let pool = PagedKvPool::new(&cfg, DeviceType::Cpu, 16, 8)?;
        assert_eq!(pool.block_size(), 16);
        assert_eq!(pool.num_blocks(), 8);
        assert_eq!(pool.max_blocks_per_seq(), 8);
        assert_eq!(pool.layer_num(), 2);
        assert_eq!(pool.kv_dim(), 4);
        assert_eq!(pool.layers().len(), 2);
        assert_eq!(pool.layers()[0].k.shape(), &[8, 16, 4]);
        assert_eq!(pool.layers()[0].v.shape(), &[8, 16, 4]);
        assert_eq!(pool.bytes_allocated(), 2 * 2 * 8 * 16 * 4 * 4);
        Ok(())
    }
}
