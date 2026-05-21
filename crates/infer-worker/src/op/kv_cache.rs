//! KV cache ops —— 模型层面**语义化**的 KV cache 访问。
//!
//! 模型只关心 "把本步产出的 K/V 存进每个 seq 的 KV cache"，不应关心 per-seq
//! 循环、slice_kv_cache 的几何计算、以及 kernel launch / graph capture 细节。
//! 本模块负责把这些全部封装起来。
//!
//! ## CUDA 路径的设计
//!
//! 所有控制数据都**常驻于 `BatchWorkspace`** 的 device scratch，scatter op
//! 内部 **零 malloc / 零 sync / 零 per-step H2D**：
//!
//! - `workspace.k_cache_ptrs_dev` / `v_cache_ptrs_dev`
//!     `[layer_num × max_batch_seqs]` u64 表 —— 每个 slot 在每层的 cache base
//!     指针。runner 在 batch 成员变化 / KV 扩容时调
//!     [`BatchWorkspace::fill_cache_ptrs_from_states`] 填一次；后续 graph
//!     replay 不用再动。scatter 和 attention 共用同一张表。
//!
//! - `workspace.scatter_slot_indices_dev` / `seq_positions_dev` /
//!   `seq_starts_dev` / `seq_lens_dev`
//!     `[max_batch_seqs]` i32 —— 本 step 的 per-seq 几何；runner 在 step 入口
//!     调 [`BatchWorkspace::refresh_scatter_indices`] 一次，所有层共用。
//!
//! kernel 自己用 `cache_ptrs[layer_idx * max_slots + slot] + pos * dst_stride`
//! 算写入起点，不再需要上游"预计算 dst 指针数组"。

use crate::OpConfig;
use crate::base::DeviceType;
use crate::base::error::Result;
use crate::model::runtime::{InferenceState, PagedKvPool};
use crate::tensor::Tensor;
use crate::worker::batch_workspace::BatchWorkspace;
use crate::worker::runner::WorkerBatchMeta;

/// 把本层当前 batch 的 `k` / `v` 分发写入各 seq 对应的 KV cache 段。
///
/// # 前置条件（CUDA）
/// 调用方（runner）必须已经：
/// 1. 确保 `states[*].kv_cache` 容量足以容纳本 step 所有 seq 的 `pos + len`
///    （调 `kv_cache.ensure_capacity(...)`）。扩容后调
///    [`BatchWorkspace::invalidate_batch_member_cache`]。
/// 2. 在 batch 成员或 cache 地址变化后调
///    [`BatchWorkspace::fill_cache_ptrs_from_states`]。
/// 3. 每个 step 入口调一次
///    [`BatchWorkspace::refresh_scatter_indices`]（所有层共用）。
///
/// CPU 路径无此约束，仍按 per-seq 循环直接写入 cache。
///
/// # 参数
/// - `k`, `v`：本步所有 token 的 K / V，shape `[total_tokens, kv_dim]`，
///   允许是 strided view（例如 `qkv.narrow(1, ...)`）。
/// - `layer_idx`：写入哪一层。
/// - `states`：per-seq 状态；CPU 路径就地扩容 / 写入，CUDA 路径不再访问。
/// - `meta`：本步的 batch meta（CPU 路径才用；CUDA 路径依赖 workspace 里
///   已经 refresh 过的 scratch）。
/// - `workspace`：持有 scatter 所需的 device scratch。
pub fn scatter(
    k: &Tensor,
    v: &Tensor,
    layer_idx: usize,
    states: &mut [&mut InferenceState],
    meta: &WorkerBatchMeta<'_>,
    workspace: &BatchWorkspace,
    paged_kv_pool: Option<&PagedKvPool>,
    cuda_cfg: Option<&OpConfig>,
) -> Result<()> {
    match k.device() {
        DeviceType::Cpu => {
            let _ = workspace;
            let _ = paged_kv_pool;
            let _ = cuda_cfg;
            scatter_cpu(k, v, layer_idx, states, meta)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let _ = states;
            scatter_cuda(k, v, layer_idx, meta, workspace, paged_kv_pool, cuda_cfg)
        }
    }
}

// ============================================================================
//  CPU：per-seq 循环
// ============================================================================

fn scatter_cpu(
    k: &Tensor,
    v: &Tensor,
    layer_idx: usize,
    states: &mut [&mut InferenceState],
    meta: &WorkerBatchMeta<'_>,
) -> Result<()> {
    let kv_dim = k.shape()[1];
    for seq_idx in 0..meta.num_seqs() {
        let start = meta.seq_start(seq_idx);
        let len = meta.seq_len(seq_idx);
        let pos = meta.seq_pos(seq_idx);
        let (mut k_dst, mut v_dst) = states[seq_idx]
            .kv_cache
            .slice_kv_cache(layer_idx, pos, len, kv_dim)?;
        k_dst.copy_from_on_current_stream(&k.narrow(0, start, len)?)?;
        v_dst.copy_from_on_current_stream(&v.narrow(0, start, len)?)?;
    }
    Ok(())
}

// ============================================================================
//  CUDA：一次 kernel launch 完成全 batch（零 alloc / 零 sync）
// ============================================================================

#[cfg(feature = "cuda")]
fn scatter_cuda(
    k: &Tensor,
    v: &Tensor,
    layer_idx: usize,
    meta: &WorkerBatchMeta<'_>,
    workspace: &BatchWorkspace,
    paged_kv_pool: Option<&PagedKvPool>,
    cuda_cfg: Option<&OpConfig>,
) -> Result<()> {
    use std::ffi::c_void;
    let batch = meta.num_seqs();
    if batch == 0 {
        return Ok(());
    }
    let kv_dim = k.shape()[1];
    let k_src_row_stride = k.strides()[0];
    let v_src_row_stride = v.strides()[0];

    if workspace.paged_active {
        let pool = paged_kv_pool.ok_or_else(|| crate::base::error::Error::InvalidArgument(
            "paged KV scatter requested but PagedKvPool is not initialized".into(),
        ))?;
        let layer = pool.layers().get(layer_idx).ok_or_else(|| crate::base::error::Error::InvalidArgument(format!(
            "paged KV scatter layer_idx {} out of bounds {}",
            layer_idx,
            pool.layers().len(),
        )))?;
        return unsafe {
            crate::op::kernels::cuda::kv_scatter_paged(
                k, v,
                &layer.k,
                &layer.v,
                workspace.paged_block_tables_dev,
                workspace.paged_max_blocks_per_seq,
                pool.block_size(),
                workspace.scatter_seq_positions_dev,
                workspace.scatter_seq_starts_dev,
                workspace.scatter_seq_lens_dev,
                batch,
                kv_dim,
                k_src_row_stride,
                v_src_row_stride,
                cuda_cfg,
            )
        };
    }

    // cache 是连续分配的 `[capacity, kv_dim]`，行步长 = kv_dim。
    let dst_row_stride = kv_dim;
    unsafe {
        crate::op::kernels::cuda::kv_scatter_batched(
            k, v,
            workspace.k_cache_ptrs_dev as *const *mut c_void,
            workspace.v_cache_ptrs_dev as *const *mut c_void,
            layer_idx,
            workspace.max_batch_seqs,
            workspace.scatter_slot_indices_dev,
            workspace.scatter_seq_positions_dev,
            workspace.scatter_seq_starts_dev,
            workspace.scatter_seq_lens_dev,
            batch,
            kv_dim,
            k_src_row_stride,
            v_src_row_stride,
            dst_row_stride,
            cuda_cfg,
        )
    }
}


// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::DataType;
    use crate::model::common::config::RuntimeModelConfig;
    use crate::model::runtime::InferenceState;
    use crate::worker::batch_workspace::BatchWorkspace;

    fn test_config(layer_num: usize, kv_head_num: usize, head_size: usize, seq_len: usize) -> RuntimeModelConfig {
        RuntimeModelConfig {
            dim: 16,
            intermediate_size: 32,
            layer_num,
            head_num: kv_head_num * 2,
            kv_head_num,
            seq_len,
            vocab_size: 32,
            kv_dim: kv_head_num * head_size,
            kv_mul: 2,
            head_size,
            q_dim: kv_head_num * 2 * head_size,
            is_shared_weight: true,
            torch_dtype: "bfloat16".to_string(),
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tokenizer_vocab_size: 32,
            immediate_dim: None,
            quant_config: None,
            rope_scaling: None,
        }
    }

    /// 构造一个 WorkerBatchMeta：三条 seq，长度分别 3 / 2 / 4；每条 seq 的起始
    /// 位置（pos_in_cache）分别 0 / 5 / 10。
    /// 构造一个 StepMeta + WorkerBatchMeta view 对。测试生命期内 StepMeta
    /// 由 caller 持有，view 从它借出。
    fn build_step_meta(
        q_start_loc: &[i32],
        slot_indices: &[i32],
        positions_start: &[i32],
        num_prefill: usize,
        num_decode: usize,
    ) -> crate::worker::runner::StepMeta {
        let mut m = crate::worker::runner::StepMeta::zeroed();
        m.num_prefill = num_prefill;
        m.num_decode = num_decode;
        for (i, &v) in q_start_loc.iter().enumerate() {
            m.q_start_loc[i] = v;
        }
        for (i, &v) in slot_indices.iter().enumerate() {
            m.slot_indices[i] = v;
        }
        for (i, &v) in positions_start.iter().enumerate() {
            m.positions_start[i] = v;
        }
        m
    }

    /// 公共逻辑：准备 k/v src + 若干 InferenceState，跑 `scatter`，对每 seq
    /// 检查其 kv_cache 的 [pos..pos+len] 段与 src 的 [seq_start..seq_end] 段
    /// 逐 bit 一致；未被写入的段保持 0。
    ///
    /// device / dtype / 值转换函数都由调用方提供，避免把 "CPU 走 F32, CUDA
    /// 走 BF16" 的分支散进测试里。
    fn run_scatter_and_verify<F>(device: DeviceType, dtype: DataType, to_scalar: F) -> Result<()>
    where
        F: Fn(f32) -> f32, // 只是校验用；实际 host 数据用 F32 生成即可
    {
        let _ = to_scalar; // 未使用，占位（保留 API 方便以后扩展）
        let kv_head_num = 2;
        let head_size = 4;
        let kv_dim = kv_head_num * head_size;
        let layer_num = 2;
        let seq_len_cap = 512;
        let mut cfg = test_config(layer_num, kv_head_num, head_size, seq_len_cap);
        // 让 CUDA path 选择 dtype；CPU 永远是 F32（runtime_float_dtype 决定）。
        cfg.torch_dtype = match dtype {
            DataType::F32 => "float32".to_string(),
            DataType::BF16 => "bfloat16".to_string(),
            DataType::F16 => "float16".to_string(),
            _ => "bfloat16".to_string(),
        };

        // batch 几何：3 seqs，长度 3 / 2 / 4，pos 0 / 5 / 10
        let q_start_loc = [0i32, 3, 5, 9];
        let seq_lens = [3usize, 2, 4];
        let seq_poses = [0i32, 5, 10];
        let total_tokens: usize = seq_lens.iter().sum();

        let mut positions: Vec<i32> = Vec::with_capacity(total_tokens);
        for (seq_i, &len) in seq_lens.iter().enumerate() {
            for t in 0..len {
                positions.push(seq_poses[seq_i] + t as i32);
            }
        }
        let slot_indices = [0i32, 1, 2];
        let token_ids: Vec<i32> = (0..total_tokens as i32).collect();
        let _ = token_ids;
        let step = build_step_meta(&q_start_loc, &slot_indices, &seq_poses, 3, 0);
        let meta = crate::worker::runner::WorkerBatchMeta::from_step(&step);

        // F32 源数据：K = row*100 + col；V = K + 10000（让 BF16 下 K/V 的值也能
        // 明确区分，不受 ULP round 影响；BF16 尾数 7 位，整数 +10000 偏移在
        // 测试数据的值域内完全无损）。
        let mut k_host_f32: Vec<f32> = Vec::with_capacity(total_tokens * kv_dim);
        let mut v_host_f32: Vec<f32> = Vec::with_capacity(total_tokens * kv_dim);
        for r in 0..total_tokens {
            for c in 0..kv_dim {
                let k = (r * 100 + c) as f32;
                k_host_f32.push(k);
                v_host_f32.push(k + 10000.0);
            }
        }

        // 把 F32 host 数据灌进目标 dtype 的 tensor（CPU/CUDA 各自正确路径）。
        let k_src = upload_2d(&k_host_f32, total_tokens, kv_dim, dtype, device)?;
        let v_src = upload_2d(&v_host_f32, total_tokens, kv_dim, dtype, device)?;

        // Sanity check: v_src 读回来的值必须 ≈ v_host_f32（隔离精度问题 vs 其它 bug）
        let v_read = read_as_f32(&v_src)?;
        for i in 0..v_host_f32.len() {
            let diff = (v_read[i] - v_host_f32[i]).abs();
            let tol = if matches!(dtype, DataType::F32) { 0.0 } else { 100.0 };
            assert!(diff <= tol,
                "v_src uploaded/read mismatch at {}: got {} want {}", i, v_read[i], v_host_f32[i]);
        }

        let mut state0 = InferenceState::new(&cfg, device)?;
        let mut state1 = InferenceState::new(&cfg, device)?;
        let mut state2 = InferenceState::new(&cfg, device)?;

        // workspace：CPU 路径也要，但只为填充 op 的参数槽；CUDA 路径才真正用它。
        let mut workspace = BatchWorkspace::new(&cfg, total_tokens, 8, device)?;
        #[cfg(feature = "cuda")]
        if matches!(device, DeviceType::Cuda(_)) {
            // 把 states 的 kv_cache base 指针表和 per-seq 索引表 push 到 device。
            let mut init_refs: Vec<&mut InferenceState> =
                vec![&mut state0, &mut state1, &mut state2];
            workspace.fill_cache_ptrs_from_states(&[0, 1, 2], &mut init_refs, std::ptr::null_mut())?;
            workspace.refresh_scatter_indices(&meta, std::ptr::null_mut())?;
        }

        // --- 跑 scatter on layer 0 ---
        {
            let mut refs: Vec<&mut InferenceState> =
                vec![&mut state0, &mut state1, &mut state2];
            scatter(&k_src, &v_src, 0, &mut refs, &meta, &workspace, None, None)?;
        }

        for (seq_i, len) in seq_lens.iter().copied().enumerate() {
            let pos = seq_poses[seq_i] as usize;
            let seq_start = q_start_loc[seq_i] as usize;

            let state: &InferenceState = [&state0, &state1, &state2][seq_i];
            let (k_cache, v_cache) = state.kv_cache.get(0)?;
            let k_host_cache = read_as_f32(k_cache)?;
            let v_host_cache = read_as_f32(v_cache)?;

            let cap = k_cache.shape()[0];
            assert_eq!(k_cache.shape()[1], kv_dim);

            // 写入段：BF16 下 ULP 可能到 64，但 K/V 的值域分别 [0,500) / [10000,10500)
            // 完全分开，任何 round 都不会让两者混淆；校验 tolerance = 100（BF16
            // 在 10000 附近的 ULP）足以覆盖正常 round。
            for t in 0..len {
                for c in 0..kv_dim {
                    let got_k = k_host_cache[(pos + t) * kv_dim + c];
                    let got_v = v_host_cache[(pos + t) * kv_dim + c];
                    let want_k = k_host_f32[(seq_start + t) * kv_dim + c];
                    let want_v = v_host_f32[(seq_start + t) * kv_dim + c];
                    let tol = if matches!(dtype, DataType::F32) { 0.0 } else { 100.0 };
                    assert!((got_k - want_k).abs() <= tol,
                        "K seq {} token {} dim {}: got {} want {}",
                        seq_i, t, c, got_k, want_k);
                    assert!((got_v - want_v).abs() <= tol,
                        "V seq {} token {} dim {}: got {} want {}",
                        seq_i, t, c, got_v, want_v);
                }
            }
            let _ = cap; // 生产代码不保证未写入段的内容（attention 只读 kv_lens 范围内），
                         // 因此不在这里断言未写入段的值。
        }

        // layer 1 多层独立性
        {
            let mut refs: Vec<&mut InferenceState> =
                vec![&mut state0, &mut state1, &mut state2];
            scatter(&k_src, &v_src, 1, &mut refs, &meta, &workspace, None, None)?;
        }
        for (seq_i, _len) in seq_lens.iter().copied().enumerate() {
            let pos = seq_poses[seq_i] as usize;
            let state: &InferenceState = [&state0, &state1, &state2][seq_i];
            let (k1, _) = state.kv_cache.get(1)?;
            let k_host_cache = read_as_f32(k1)?;
            let first = k_host_cache[pos * kv_dim];
            let expected = (q_start_loc[seq_i] as f32) * 100.0;
            let tol = if matches!(dtype, DataType::F32) { 0.0 } else { 100.0 };
            assert!((first - expected).abs() <= tol,
                "layer 1 first write on seq {} wrong: got {} want {}", seq_i, first, expected);
        }
        Ok(())
    }

    /// 按 dtype 把 host f32 数据上传到指定 device 的 2D tensor。
    fn upload_2d(
        data_f32: &[f32],
        rows: usize,
        cols: usize,
        dtype: DataType,
        device: DeviceType,
    ) -> Result<Tensor> {
        let mut t = Tensor::new(&[rows, cols], dtype, device)?;
        match (dtype, device) {
            (DataType::F32, DeviceType::Cpu) => {
                t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(data_f32);
            }
            (DataType::BF16, DeviceType::Cpu) => {
                let bf: Vec<half::bf16> = data_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
                t.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&bf);
            }
            #[cfg(feature = "cuda")]
            (DataType::F32, DeviceType::Cuda(_)) => {
                t.as_f32_mut()?.buffer_mut().copy_from_host(data_f32)?;
            }
            #[cfg(feature = "cuda")]
            (DataType::BF16, DeviceType::Cuda(_)) => {
                let bf: Vec<half::bf16> = data_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
                t.as_bf16_mut()?.buffer_mut().copy_from_host(&bf)?;
            }
            _ => return Err(crate::base::error::Error::InvalidArgument(
                format!("test upload_2d: unsupported (dtype {:?}, device {:?})", dtype, device)
            ).into()),
        }
        Ok(t)
    }

    /// 读 tensor 到 host Vec<f32>（BF16 / F32 统一转 F32 方便校验）。
    fn read_as_f32(t: &Tensor) -> Result<Vec<f32>> {
        let host = match t.device() {
            DeviceType::Cpu => t.clone(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => t.to_cpu()?,
        };
        match host.dtype() {
            DataType::F32 => Ok(host.as_f32()?.as_slice()?.to_vec()),
            DataType::BF16 => {
                Ok(host.as_bf16()?.as_slice()?.iter().map(|x| x.to_f32()).collect())
            }
            dt => Err(crate::base::error::Error::InvalidArgument(
                format!("read_as_f32: unsupported dtype {:?}", dt)
            ).into()),
        }
    }

    #[test]
    fn kv_scatter_cpu_matches_expected() -> Result<()> {
        run_scatter_and_verify(DeviceType::Cpu, DataType::F32, |x| x)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn kv_scatter_cuda_matches_expected() -> Result<()> {
        run_scatter_and_verify(DeviceType::Cuda(0), DataType::BF16, |x| x)
    }

    /// CUDA 专项：K/V 作为 fused qkv 的 **strided view**，scatter 结果必须
    /// 与先物理 copy 到连续 k_src/v_src 后再 scatter 的结果一致。
    #[test]
    #[cfg(feature = "cuda")]
    fn kv_scatter_cuda_accepts_strided_src() -> Result<()> {
        let kv_head_num = 2;
        let head_size = 4;
        let kv_dim = kv_head_num * head_size;
        let q_dim = kv_head_num * 2 * head_size;
        let layer_num = 1;
        let cfg = test_config(layer_num, kv_head_num, head_size, 512);
        let device = DeviceType::Cuda(0);
        let dtype = DataType::BF16;

        // batch: 2 seqs of length 3, pos 0 / 7
        let q_start_loc = [0i32, 3, 6];
        let total_tokens = 6usize;
        let positions: Vec<i32> = (0..3).chain(7..10).collect();
        let slot_indices = [0i32, 1];
        let token_ids: Vec<i32> = (0..6).collect();
        let _ = (positions, token_ids);
        let seq_start_pos = [0i32, 7];
        let step = build_step_meta(&q_start_loc, &slot_indices, &seq_start_pos, 2, 0);
        let meta = crate::worker::runner::WorkerBatchMeta::from_step(&step);

        // 构造 fused qkv [T, q_dim + 2*kv_dim]（BF16）
        let total_cols = q_dim + 2 * kv_dim;
        let mut qkv_host: Vec<half::bf16> = Vec::with_capacity(total_tokens * total_cols);
        for r in 0..total_tokens {
            for c in 0..total_cols {
                qkv_host.push(half::bf16::from_f32((r * 1000 + c) as f32 * 0.001));
            }
        }
        let mut qkv = Tensor::new(&[total_tokens, total_cols], dtype, device)?;
        qkv.as_bf16_mut()?.buffer_mut().copy_from_host(&qkv_host)?;

        // --- A. strided scatter ---
        let k_view = qkv.narrow(1, q_dim, kv_dim)?;
        let v_view = qkv.narrow(1, q_dim + kv_dim, kv_dim)?;

        let mut s0 = InferenceState::new(&cfg, device)?;
        let mut s1 = InferenceState::new(&cfg, device)?;
        let mut ws_a = BatchWorkspace::new(&cfg, total_tokens, 8, device)?;
        {
            let mut init_refs: Vec<&mut InferenceState> = vec![&mut s0, &mut s1];
            ws_a.fill_cache_ptrs_from_states(&[0, 1], &mut init_refs, std::ptr::null_mut())?;
            ws_a.refresh_scatter_indices(&meta, std::ptr::null_mut())?;
        }
        {
            let mut refs: Vec<&mut InferenceState> = vec![&mut s0, &mut s1];
            scatter(&k_view, &v_view, 0, &mut refs, &meta, &ws_a, None, None)?;
        }

        // --- B. dense scatter: 先物理 copy 到 [T, kv_dim] 再 scatter ---
        let mut k_dense_host: Vec<half::bf16> = Vec::with_capacity(total_tokens * kv_dim);
        let mut v_dense_host: Vec<half::bf16> = Vec::with_capacity(total_tokens * kv_dim);
        for r in 0..total_tokens {
            for c in 0..kv_dim {
                k_dense_host.push(qkv_host[r * total_cols + q_dim + c]);
                v_dense_host.push(qkv_host[r * total_cols + q_dim + kv_dim + c]);
            }
        }
        let mut k_dense = Tensor::new(&[total_tokens, kv_dim], dtype, device)?;
        let mut v_dense = Tensor::new(&[total_tokens, kv_dim], dtype, device)?;
        k_dense.as_bf16_mut()?.buffer_mut().copy_from_host(&k_dense_host)?;
        v_dense.as_bf16_mut()?.buffer_mut().copy_from_host(&v_dense_host)?;

        let mut s0b = InferenceState::new(&cfg, device)?;
        let mut s1b = InferenceState::new(&cfg, device)?;
        let mut ws_b = BatchWorkspace::new(&cfg, total_tokens, 8, device)?;
        {
            let mut init_refs: Vec<&mut InferenceState> = vec![&mut s0b, &mut s1b];
            ws_b.fill_cache_ptrs_from_states(&[0, 1], &mut init_refs, std::ptr::null_mut())?;
            ws_b.refresh_scatter_indices(&meta, std::ptr::null_mut())?;
        }
        {
            let mut refs_b: Vec<&mut InferenceState> = vec![&mut s0b, &mut s1b];
            scatter(&k_dense, &v_dense, 0, &mut refs_b, &meta, &ws_b, None, None)?;
        }

        // --- 比对：两路径每 seq 在写入段的内容必须逐 bit 相等 ---
        // 生产代码不保证未写入段的内容，因此这里只对 [pos, pos+len) 做比对。
        let kv_stride = kv_dim;
        for seq_i in 0..2 {
            let a: &InferenceState = [&s0, &s1][seq_i];
            let b: &InferenceState = [&s0b, &s1b][seq_i];
            let (k_a, v_a) = a.kv_cache.get(0)?;
            let (k_b, v_b) = b.kv_cache.get(0)?;
            let ka = read_as_f32(k_a)?;
            let kb = read_as_f32(k_b)?;
            let va = read_as_f32(v_a)?;
            let vb = read_as_f32(v_b)?;
            let pos = seq_start_pos[seq_i] as usize;   // 每 seq 3 个 token，起始 pos
            let len = 3usize;
            for t in 0..len {
                for c in 0..kv_dim {
                    let idx = (pos + t) * kv_stride + c;
                    assert_eq!(ka[idx], kb[idx],
                        "seq {} K token {} dim {} strided={} dense={}",
                        seq_i, t, c, ka[idx], kb[idx]);
                    assert_eq!(va[idx], vb[idx],
                        "seq {} V token {} dim {} strided={} dense={}",
                        seq_i, t, c, va[idx], vb[idx]);
                }
            }
        }
        Ok(())
    }
}
