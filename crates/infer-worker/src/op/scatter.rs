use crate::base::error::Result;
use crate::base::DeviceType;
use crate::op::{kernels, Op, OpContext};

/// Scatter operator: writes K,V to paged KV cache based on slot_mapping
///
/// This is used for KV cache updates in the decoding phase using paged attention.
/// The operator handles both Key and Value tensors simultaneously.
///
/// # Context
/// * `ctx.inputs[0]` (key): Key tensor with shape [batch_size, kv_dim]
/// * `ctx.inputs[1]` (value): Value tensor with shape [batch_size, kv_dim]
/// * `ctx.inputs[2]` (slot_mapping): Slot mapping tensor [batch_size] with target positions
/// * `ctx.outputs[0]` (k_cache): Key cache tensor [num_blocks, block_size, kv_dim]
/// * `ctx.outputs[1]` (v_cache): Value cache tensor [num_blocks, block_size, kv_dim]
#[derive(Debug, Clone, Copy)]
pub struct Scatter;

impl Scatter {
    /// Creates a new Scatter operator instance.
    pub fn new() -> Self {
        Scatter
    }
}

impl Default for Scatter {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for Scatter {
    fn name(&self) -> &'static str {
        "Scatter"
    }

    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        // 支持两种模式：
        // 模式1（遗留）: 2输入, 1输出 - 单个张量scatter
        // 模式2（新）: 3输入, 2输出 - KV一起scatter到paged cache

        match (ctx.inputs.len(), ctx.outputs.len()) {
            (2, 1) => {
                // 旧模式：用于单个张量scatter（后向兼容）
                self.scatter_single(ctx)
            }
            (3, 2) => {
                // 新模式：用于KV cache写入
                self.scatter_kv(ctx)
            }
            _ => Err(anyhow::anyhow!(
                "Scatter requires either (2 inputs, 1 output) or (3 inputs, 2 outputs), got ({}, {})",
                ctx.inputs.len(), ctx.outputs.len()
            )),
        }
    }
}

impl Scatter {
    /// 旧模式：scatter单个张量
    fn scatter_single(&self, ctx: &mut OpContext) -> Result<()> {
        let src = ctx.inputs[0];
        let pos = ctx.inputs[1];
        let dst = &mut ctx.outputs[0];

        // Check device compatibility
        if src.device() != dst.device() {
            return Err(anyhow::anyhow!(
                "Device mismatch for Scatter: src={:?}, dst={:?}",
                src.device(), dst.device()
            ));
        }

        // Dispatch to kernel based on device
        match src.device() {
            DeviceType::Cpu => {
                return Err(anyhow::anyhow!(
                    "Scatter operator is only supported on CUDA devices"
                ));
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::scatter(dst, src, pos, ctx.cuda_config)?;
            }
        }

        Ok(())
    }

    /// 新模式：将K,V分别写入到paged KV cache
    fn scatter_kv(&self, ctx: &mut OpContext) -> Result<()> {
        let key = ctx.inputs[0];
        let value = ctx.inputs[1];
        let slot_mapping = ctx.inputs[2];

        // Check device compatibility
        if key.device() != value.device() || key.device() != slot_mapping.device() {
            return Err(anyhow::anyhow!(
                "Device mismatch for Scatter KV: key={:?}, value={:?}, slot_mapping={:?}",
                key.device(), value.device(), slot_mapping.device()
            ));
        }

        // Validate shapes
        if key.shape()[0] != value.shape()[0] || key.shape()[1] != value.shape()[1] {
            return Err(anyhow::anyhow!(
                "Key and Value shapes must match: key={:?}, value={:?}",
                key.shape(), value.shape()
            ));
        }

        if slot_mapping.shape()[0] != key.shape()[0] {
            return Err(anyhow::anyhow!(
                "Slot mapping length must match batch size: slot_mapping={:?}, batch_size={}",
                slot_mapping.shape(), key.shape()[0]
            ));
        }

        // Validate output count
        if ctx.outputs.len() != 2 {
            return Err(anyhow::anyhow!(
                "Scatter KV requires exactly 2 outputs (k_cache, v_cache), got {}",
                ctx.outputs.len()
            ));
        }

        // Split outputs to get mutable references
        let (k_cache_ref, v_cache_ref) = ctx.outputs.split_at_mut(1);
        let k_cache = &mut k_cache_ref[0];
        let v_cache = &mut v_cache_ref[0];

        // Check cache device compatibility
        if key.device() != k_cache.device() || key.device() != v_cache.device() {
            return Err(anyhow::anyhow!(
                "Device mismatch for caches: key={:?}, k_cache={:?}, v_cache={:?}",
                key.device(), k_cache.device(), v_cache.device()
            ));
        }

        // Dispatch to kernel based on device
        match key.device() {
            DeviceType::Cpu => {
                return Err(anyhow::anyhow!(
                    "Scatter KV operator is only supported on CUDA devices"
                ));
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::scatter_kv(key, value, slot_mapping, k_cache, v_cache, ctx.cuda_config)?;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Scatter {
    /// Scatter is stateless; nothing to move to CUDA.
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
//  Unit Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;
    use half::bf16;

    /// Helper to assert BF16 results are close
    fn assert_bf16_close(a: &[bf16], b: &[bf16], tol: f32) {
        assert_eq!(a.len(), b.len(), "BF16 slices have different lengths");
        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (val_a.to_f32() - val_b.to_f32()).abs();
            assert!(
                diff < tol,
                "BF16 mismatch at index {}: a = {}, b = {}, diff = {}",
                i, val_a.to_f32(), val_b.to_f32(), diff
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_scatter_bf16_cuda_batch() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;

        // Test multiple kvdim and max_seq_len combinations
        for (max_seq_len, kvdim, pos_value) in [(16, 128, 5), (32, 256, 10), (64, 512, 30), (128, 768, 64)] {
            // Prepare source data (1 row to scatter)
            let src_data: Vec<bf16> = (0..kvdim)
                .map(|i| bf16::from_f32(((i * 7) % 100) as f32 * 0.1 + 10.0))
                .collect();

            // Create source tensor on GPU
            let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
            src.as_bf16_mut()?.buffer_mut().copy_from_host(&src_data)?;

            // Create position tensor on CPU (as expected by operator)
            let mut pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            pos.as_i32_mut()?.as_slice_mut()?[0] = pos_value;
            let pos_gpu = pos.to_cuda(0)?;

            // Create destination tensor on GPU (initialized with different values)
            let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
            let dst_data: Vec<bf16> = (0..(max_seq_len * kvdim))
                .map(|i| bf16::from_f32(((i * 13) % 100) as f32 * 0.01))
                .collect();
            dst.as_bf16_mut()?.buffer_mut().copy_from_host(&dst_data)?;

            // Execute scatter
            let scatter_op = Scatter::new();
            let cuda_config = crate::cuda::CudaConfig::new()?;
            scatter_op.forward(&mut OpContext::new(&[&src, &pos_gpu], &mut [&mut dst], Some(&cuda_config)))?;

            // Copy result back and verify
            let result_tensor = dst.to_cpu()?;
            let result = result_tensor.as_bf16()?.as_slice()?;

            // Verify the scattered row matches source
            let scattered_row_start = pos_value as usize * kvdim;
            let scattered_row_end = scattered_row_start + kvdim;
            let scattered_row = &result[scattered_row_start..scattered_row_end];

            assert_bf16_close(scattered_row, &src_data, 1e-3);

            // Verify other rows are unchanged
            for row in 0..max_seq_len {
                if row != pos_value as usize {
                    let row_start = row * kvdim;
                    let row_end = row_start + kvdim;
                    let row_data = &result[row_start..row_end];
                    let expected_row: Vec<bf16> = (row_start..row_end)
                        .map(|i| bf16::from_f32(((i * 13) % 100) as f32 * 0.01))
                        .collect();
                    assert_bf16_close(row_data, &expected_row, 1e-3);
                }
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_scatter_bf16_edge_cases() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;
        let max_seq_len = 32;
        let kvdim = 256;

        // Test edge positions: first, middle, last
        for pos_value in [0, max_seq_len / 2, max_seq_len - 1] {
            // Prepare unique source data
            let src_data: Vec<bf16> = (0..kvdim)
                .map(|i| bf16::from_f32((i as f32) * 0.1 + (pos_value as f32) * 100.0))
                .collect();

            let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
            src.as_bf16_mut()?.buffer_mut().copy_from_host(&src_data)?;

            let mut pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            pos.as_i32_mut()?.as_slice_mut()?[0] = pos_value as i32;
            let pos_gpu = pos.to_cuda(0)?;

            // Initialize dst with zeros
            let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
            let dst_data = vec![bf16::from_f32(0.0); max_seq_len * kvdim];
            dst.as_bf16_mut()?.buffer_mut().copy_from_host(&dst_data)?;

            // Execute scatter
            let scatter_op = Scatter::new();
            let cuda_config = crate::cuda::CudaConfig::new()?;
            scatter_op.forward(&mut OpContext::new(&[&src, &pos_gpu], &mut [&mut dst], Some(&cuda_config)))?;

            // Verify
            let result_tensor = dst.to_cpu()?;
            let result = result_tensor.as_bf16()?.as_slice()?;

            // Check scattered row
            let scattered_row_start = pos_value as usize * kvdim;
            let scattered_row_end = scattered_row_start + kvdim;
            let scattered_row = &result[scattered_row_start..scattered_row_end];

            assert_bf16_close(scattered_row, &src_data, 1e-3);

            // Check all other rows are still zero
            for row in 0..max_seq_len {
                if row != pos_value {
                    let row_start = row * kvdim;
                    let row_end = row_start + kvdim;
                    let row_data = &result[row_start..row_end];

                    for &val in row_data {
                        assert!(
                            val.to_f32().abs() < 1e-5,
                            "Row {} should be zero but got non-zero value", row
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_scatter_bf16_correctness() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;
        let max_seq_len = 64;
        let kvdim = 512;

        // Test multiple scatter operations (simulating KV cache updates)
        let mut dst = Tensor::new(&[max_seq_len, kvdim], dtype, device)?;
        let dst_data = vec![bf16::from_f32(-1.0); max_seq_len * kvdim]; // Initialize with -1.0
        dst.as_bf16_mut()?.buffer_mut().copy_from_host(&dst_data)?;

        let scatter_op = Scatter::new();
        let cuda_config = crate::cuda::CudaConfig::new()?;

        // Scatter to multiple positions
        for pos_value in [0, 5, 10, 20, 63] {
            // Create unique source data for this position (using smaller values for BF16)
            let src_data: Vec<bf16> = (0..kvdim)
                .map(|i| bf16::from_f32((pos_value as f32) * 10.0 + (i as f32) * 0.01))
                .collect();

            let mut src = Tensor::new(&[1, kvdim], dtype, device)?;
            src.as_bf16_mut()?.buffer_mut().copy_from_host(&src_data)?;

            let mut pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            pos.as_i32_mut()?.as_slice_mut()?[0] = pos_value;
            let pos_gpu = pos.to_cuda(0)?;

            // Execute scatter
            scatter_op.forward(&mut OpContext::new(&[&src, &pos_gpu], &mut [&mut dst], Some(&cuda_config)))?;

            // Verify this position was updated correctly
            let result_tensor = dst.to_cpu()?;
            let result = result_tensor.as_bf16()?.as_slice()?;

            let scattered_row_start = pos_value as usize * kvdim;
            let scattered_row_end = scattered_row_start + kvdim;
            let scattered_row = &result[scattered_row_start..scattered_row_end];

            assert_bf16_close(scattered_row, &src_data, 1e-2);
        }

        // Final verification: check that all scattered positions have correct values
        // and unscattered positions still have -1.0
        let final_result = dst.to_cpu()?;
        let final_data = final_result.as_bf16()?.as_slice()?;

        for row in 0..max_seq_len {
            let row_start = row * kvdim;
            let row_end = row_start + kvdim;
            let row_data = &final_data[row_start..row_end];

            if [0, 5, 10, 20, 63].contains(&row) {
                // Should contain scattered data, not -1.0
                // Check that first value is close to expected (row * 10.0)
                let first_val = row_data[0].to_f32();
                let expected_first = (row as f32) * 10.0;
                assert!(
                    (first_val - expected_first).abs() < 3.0, // Allow larger tolerance for BF16 precision
                    "Row {}: expected first value ~{}, got {}", row, expected_first, first_val
                );
            } else {
                // Should still be -1.0
                for &val in row_data {
                    assert!(
                        (val.to_f32() + 1.0).abs() < 0.1,
                        "Row {} should still be -1.0", row
                    );
                }
            }
        }

        Ok(())
    }
}