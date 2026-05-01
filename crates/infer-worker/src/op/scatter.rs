use crate::base::error::Result;
use crate::base::DeviceType;
use crate::tensor::Tensor;
use crate::OpConfig;

use super::kernels;

/// Scatter operator: copies src[0, :] to dst[pos, :]
#[derive(Debug, Clone, Copy)]
pub struct Scatter;

impl Scatter {
    pub fn new() -> Self {
        Scatter
    }

    /// 执行 scatter: dst[pos, :] = src[0, :]
    pub fn forward(
        &self,
        src: &Tensor,
        pos: &Tensor,
        dst: &mut Tensor,
        #[allow(unused_variables)]
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        match src.device() {
            DeviceType::Cpu => {
                let pos_val = pos.as_i32()?.as_slice()?[0] as usize;
                let kvdim = src.shape()[1];

                match src.dtype() {
                    crate::base::DataType::F32 => {
                        let src_slice = src.as_f32()?.as_slice()?;
                        let dst_slice = dst.as_f32_mut()?.as_slice_mut()?;
                        let dst_start = pos_val * kvdim;
                        dst_slice[dst_start..dst_start + kvdim]
                            .copy_from_slice(&src_slice[..kvdim]);
                    }
                    crate::base::DataType::BF16 => {
                        let src_slice = src.as_bf16()?.as_slice()?;
                        let dst_slice = dst.as_bf16_mut()?.as_slice_mut()?;
                        let dst_start = pos_val * kvdim;
                        dst_slice[dst_start..dst_start + kvdim]
                            .copy_from_slice(&src_slice[..kvdim]);
                    }
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Scatter CPU: unsupported dtype {:?}", src.dtype()
                        ));
                    }
                }
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::scatter(dst, src, pos, cuda_config)?;
            }
        }

        Ok(())
    }
}

impl Default for Scatter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl Scatter {
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}

/// 融合 scatter: 同时写入 K 和 V cache
#[allow(unused_variables)]
pub fn scatter_kv(
    dst_k: &mut Tensor,
    src_k: &Tensor,
    dst_v: &mut Tensor,
    src_v: &Tensor,
    pos: &Tensor,
    #[allow(unused_variables)]
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    match dst_k.device() {
        DeviceType::Cpu => kernels::cpu::scatter_kv(dst_k, src_k, dst_v, src_v, pos),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::scatter_kv(dst_k, src_k, dst_v, src_v, pos, cuda_config),
    }
}

/// 批量 scatter: 将 B 行 K/V 写入 B 个不同的 KV cache 的各自 position。
///
/// 用于 batch decode: B 个 seq 各产生 1 行 KV，需要写入各自独立的 cache。
///
/// CUDA 路径：一次 kernel launch 并行处理 B 行。需要调用方提供两块 device
/// 指针数组 buffer（容量 ≥ batch_size * sizeof(u64) bytes），以及 device 上的
/// `positions_dev` 张量。
///
/// CPU 路径：循环 B 次（内存拷贝已经很快，不值得额外 kernel 化）。
///
/// # Arguments
/// * `k_caches` / `v_caches` - B 个 K/V cache 的 &mut Tensor 引用, 每个 shape [max_seq_len, kv_dim]
/// * `src_k` / `src_v` - [B, kv_dim] 新的 K/V
/// * `positions_cpu` - [B] 每个 seq 的写入位置（host slice，用于 CPU 分支）
/// * `positions_dev` - [B] I32 设备张量（用于 CUDA 分支），CPU 下可传 None
/// * `k_ptrs_dev` / `v_ptrs_dev` - 预分配的 device 指针数组 buffer（CUDA 必须提供）
#[allow(clippy::too_many_arguments)]
pub fn scatter_kv_batch(
    k_caches: &mut [&mut Tensor],
    v_caches: &mut [&mut Tensor],
    src_k: &Tensor,
    src_v: &Tensor,
    positions_cpu: &[i32],
    #[cfg(feature = "cuda")] positions_dev: Option<&Tensor>,
    #[cfg(feature = "cuda")] k_ptrs_dev: *mut u64,
    #[cfg(feature = "cuda")] v_ptrs_dev: *mut u64,
    #[allow(unused_variables)]
    cuda_config: Option<&OpConfig>,
) -> Result<()> {
    let batch_size = k_caches.len();
    assert_eq!(v_caches.len(), batch_size);
    assert_eq!(positions_cpu.len(), batch_size);
    if batch_size == 0 { return Ok(()); }

    let kv_dim = k_caches[0].shape()[1];

    match k_caches[0].device() {
        DeviceType::Cpu => {
            for i in 0..batch_size {
                let src_k_row = src_k.slice(&[i, 0], &[1, kv_dim])?;
                let src_v_row = src_v.slice(&[i, 0], &[1, kv_dim])?;
                let mut pos_tensor = Tensor::new(&[1], crate::base::DataType::I32, DeviceType::Cpu)?;
                pos_tensor.as_i32_mut()?.as_slice_mut()?[0] = positions_cpu[i];
                kernels::cpu::scatter_kv(
                    k_caches[i], &src_k_row, v_caches[i], &src_v_row, &pos_tensor,
                )?;
            }
            Ok(())
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let pos_dev = positions_dev.ok_or_else(|| crate::base::error::Error::InvalidArgument(
                "scatter_kv_batch CUDA 路径需要 positions_dev".into()
            ))?;
            if k_ptrs_dev.is_null() || v_ptrs_dev.is_null() {
                return Err(crate::base::error::Error::InvalidArgument(
                    "scatter_kv_batch CUDA 路径需要预分配的 k/v_ptrs_dev 指针数组 buffer".into()
                ).into());
            }
            kernels::cuda::scatter_kv_batch(
                k_caches, v_caches, src_k, src_v,
                pos_dev, k_ptrs_dev, v_ptrs_dev,
                cuda_config,
            )
        }
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
            scatter_op.forward(&src, &pos_gpu, &mut dst, Some(&cuda_config))?;

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
            scatter_op.forward(&src, &pos_gpu, &mut dst, Some(&cuda_config))?;

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
            scatter_op.forward(&src, &pos_gpu, &mut dst, Some(&cuda_config))?;

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

    // ========================================================================
    // scatter_kv_batch 正确性：batched 一次 kernel launch vs per-seq 循环
    // ========================================================================

    /// CPU: scatter_kv_batch 写 B 行到 B 个 cache vs 逐 seq 循环 scatter_kv
    #[test]
    fn test_scatter_kv_batch_cpu_matches_loop() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::BF16;
        let batch = 4;
        let max_seq_len = 32;
        let kv_dim = 128;

        // 构造 B 个独立 cache，填 -1.0；随机 src_k, src_v；随机 positions
        let mut rng = rand::rng();
        let positions: Vec<i32> = vec![3, 0, 17, 8]; // 各不相同

        // 构造 src_k / src_v: [B, kv_dim]
        let src_k_data: Vec<bf16> = (0..batch * kv_dim)
            .map(|_| bf16::from_f32(rand::Rng::random_range(&mut rng, -1.0f32..1.0))).collect();
        let src_v_data: Vec<bf16> = (0..batch * kv_dim)
            .map(|_| bf16::from_f32(rand::Rng::random_range(&mut rng, -1.0f32..1.0))).collect();
        let mut src_k = Tensor::new(&[batch, kv_dim], dtype, device)?;
        let mut src_v = Tensor::new(&[batch, kv_dim], dtype, device)?;
        src_k.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&src_k_data);
        src_v.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&src_v_data);

        // 两套 caches: A 走 batched, B 走循环。都用 -1.0 初值
        let init: Vec<bf16> = vec![bf16::from_f32(-1.0); max_seq_len * kv_dim];
        let mut k_caches_a: Vec<Tensor> = (0..batch).map(|_| {
            let mut t = Tensor::new(&[max_seq_len, kv_dim], dtype, device).unwrap();
            t.as_bf16_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&init);
            t
        }).collect();
        let mut v_caches_a: Vec<Tensor> = (0..batch).map(|_| {
            let mut t = Tensor::new(&[max_seq_len, kv_dim], dtype, device).unwrap();
            t.as_bf16_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&init);
            t
        }).collect();
        let mut k_caches_b: Vec<Tensor> = (0..batch).map(|_| {
            let mut t = Tensor::new(&[max_seq_len, kv_dim], dtype, device).unwrap();
            t.as_bf16_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&init);
            t
        }).collect();
        let mut v_caches_b: Vec<Tensor> = (0..batch).map(|_| {
            let mut t = Tensor::new(&[max_seq_len, kv_dim], dtype, device).unwrap();
            t.as_bf16_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&init);
            t
        }).collect();

        // A: batched
        {
            let mut k_refs: Vec<&mut Tensor> = k_caches_a.iter_mut().collect();
            let mut v_refs: Vec<&mut Tensor> = v_caches_a.iter_mut().collect();
            super::scatter_kv_batch(
                &mut k_refs, &mut v_refs,
                &src_k, &src_v, &positions,
                #[cfg(feature = "cuda")] None,
                #[cfg(feature = "cuda")] std::ptr::null_mut(),
                #[cfg(feature = "cuda")] std::ptr::null_mut(),
                None,
            )?;
        }

        // B: per-seq loop，使用 CPU scatter_kv
        for i in 0..batch {
            let src_k_row = src_k.slice(&[i, 0], &[1, kv_dim])?;
            let src_v_row = src_v.slice(&[i, 0], &[1, kv_dim])?;
            let mut pos_t = Tensor::new(&[1], DataType::I32, device)?;
            pos_t.as_i32_mut()?.as_slice_mut()?[0] = positions[i];
            crate::op::kernels::cpu::scatter_kv(
                &mut k_caches_b[i], &src_k_row, &mut v_caches_b[i], &src_v_row, &pos_t,
            )?;
        }

        // 比对
        for i in 0..batch {
            let a = k_caches_a[i].as_bf16()?.as_slice()?;
            let b = k_caches_b[i].as_bf16()?.as_slice()?;
            assert_eq!(a, b, "CPU K cache[{}] batched vs loop mismatch", i);
            let a = v_caches_a[i].as_bf16()?.as_slice()?;
            let b = v_caches_b[i].as_bf16()?.as_slice()?;
            assert_eq!(a, b, "CPU V cache[{}] batched vs loop mismatch", i);
        }
        Ok(())
    }

    /// CUDA: scatter_kv_batch kernel vs per-seq 循环 scatter_kv
    #[test]
    #[cfg(feature = "cuda")]
    fn test_scatter_kv_batch_cuda_matches_loop() -> Result<()> {
        use crate::cuda::CudaConfig;

        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;
        let batch = 6;
        let max_seq_len = 64;
        let kv_dim = 256; // 需要 %8==0

        let mut rng = rand::rng();
        let positions: Vec<i32> = vec![0, 1, 10, 20, 33, 63]; // 互不相同，都 < max_seq_len

        let src_k_data: Vec<bf16> = (0..batch * kv_dim)
            .map(|_| bf16::from_f32(rand::Rng::random_range(&mut rng, -1.0f32..1.0))).collect();
        let src_v_data: Vec<bf16> = (0..batch * kv_dim)
            .map(|_| bf16::from_f32(rand::Rng::random_range(&mut rng, -1.0f32..1.0))).collect();
        let mut src_k = Tensor::new(&[batch, kv_dim], dtype, device)?;
        let mut src_v = Tensor::new(&[batch, kv_dim], dtype, device)?;
        src_k.as_bf16_mut()?.buffer_mut().copy_from_host(&src_k_data)?;
        src_v.as_bf16_mut()?.buffer_mut().copy_from_host(&src_v_data)?;

        // 两套 caches：zeros
        let zero: Vec<bf16> = vec![bf16::from_f32(0.0); max_seq_len * kv_dim];
        let mut k_caches_a: Vec<Tensor> = Vec::new();
        let mut v_caches_a: Vec<Tensor> = Vec::new();
        let mut k_caches_b: Vec<Tensor> = Vec::new();
        let mut v_caches_b: Vec<Tensor> = Vec::new();
        for _ in 0..batch {
            let mut k = Tensor::new(&[max_seq_len, kv_dim], dtype, device)?;
            let mut v = Tensor::new(&[max_seq_len, kv_dim], dtype, device)?;
            k.as_bf16_mut()?.buffer_mut().copy_from_host(&zero)?;
            v.as_bf16_mut()?.buffer_mut().copy_from_host(&zero)?;
            k_caches_a.push(k);
            v_caches_a.push(v);

            let mut k = Tensor::new(&[max_seq_len, kv_dim], dtype, device)?;
            let mut v = Tensor::new(&[max_seq_len, kv_dim], dtype, device)?;
            k.as_bf16_mut()?.buffer_mut().copy_from_host(&zero)?;
            v.as_bf16_mut()?.buffer_mut().copy_from_host(&zero)?;
            k_caches_b.push(k);
            v_caches_b.push(v);
        }

        let cuda_cfg = CudaConfig::new()?;

        // 分配 device 指针数组 buffer
        let bytes = batch * std::mem::size_of::<u64>();
        let mut k_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut v_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut k_ptrs_dev, bytes))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut v_ptrs_dev, bytes))?;
        }

        // positions device tensor
        let mut pos_cpu_t = Tensor::new(&[batch], DataType::I32, DeviceType::Cpu)?;
        pos_cpu_t.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&positions);
        let pos_dev = pos_cpu_t.to_cuda(0)?;

        // A: batched
        {
            let mut k_refs: Vec<&mut Tensor> = k_caches_a.iter_mut().collect();
            let mut v_refs: Vec<&mut Tensor> = v_caches_a.iter_mut().collect();
            super::scatter_kv_batch(
                &mut k_refs, &mut v_refs,
                &src_k, &src_v, &positions,
                Some(&pos_dev),
                k_ptrs_dev as *mut u64,
                v_ptrs_dev as *mut u64,
                Some(&cuda_cfg),
            )?;
        }
        cuda_cfg.sync_stream()?;

        // B: per-seq 循环 scatter_kv (CUDA)
        for i in 0..batch {
            let src_k_row = src_k.slice(&[i, 0], &[1, kv_dim])?;
            let src_v_row = src_v.slice(&[i, 0], &[1, kv_dim])?;
            let mut pos_t = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            pos_t.as_i32_mut()?.as_slice_mut()?[0] = positions[i];
            let pos_gpu = pos_t.to_cuda(0)?;
            crate::op::kernels::cuda::scatter_kv(
                &mut k_caches_b[i], &src_k_row, &mut v_caches_b[i], &src_v_row, &pos_gpu,
                Some(&cuda_cfg),
            )?;
        }
        cuda_cfg.sync_stream()?;

        // 释放指针 buffer
        unsafe {
            let _ = crate::cuda::ffi::cudaFree(k_ptrs_dev);
            let _ = crate::cuda::ffi::cudaFree(v_ptrs_dev);
        }

        // 比对
        for i in 0..batch {
            let ka = k_caches_a[i].to_cpu()?;
            let kb = k_caches_b[i].to_cpu()?;
            let va = v_caches_a[i].to_cpu()?;
            let vb = v_caches_b[i].to_cpu()?;
            assert_eq!(ka.as_bf16()?.as_slice()?, kb.as_bf16()?.as_slice()?,
                "CUDA K cache[{}] batched vs loop mismatch", i);
            assert_eq!(va.as_bf16()?.as_slice()?, vb.as_bf16()?.as_slice()?,
                "CUDA V cache[{}] batched vs loop mismatch", i);
        }
        Ok(())
    }
}
