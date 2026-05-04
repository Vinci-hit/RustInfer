use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::model::common::config::RuntimeModelConfig;
use crate::tensor::Tensor;

/// Batch forward 用的共享 workspace buffer。
///
/// 所有 buffer 按 `max_batch_tokens × dim` 预分配一次，
/// 每步通过 `Tensor::slice` 零拷贝取前 N 行使用。
/// 与单 seq 的 `InferenceState.workspace` 不同，这里是跨 seq 共享的。
pub struct BatchWorkspace {
    // ═══ 主数据流 ═══
    /// embedding 输出 / residual stream, [max_batch_tokens, dim]
    pub x: Tensor,
    /// RMSNorm 输出 (复用做 attn_out, ffn_norm_out), [max_batch_tokens, dim]
    pub rms_out: Tensor,
    /// Fused QKV 输出, [max_batch_tokens, q_dim + 2 * kv_dim]
    pub qkv_out: Tensor,
    /// Q slice (不额外分配, 从 qkv_out slice)
    // pub q: slice of qkv_out
    /// Gate+Up fused 输出, [max_batch_tokens, 2 * intermediate_size]
    pub gate_up_out: Tensor,
    /// W2 (down proj) 输出 / FFN 中间 buffer, [max_batch_tokens, dim]
    pub ffn_out: Tensor,
    /// 额外的 [max_batch_tokens, dim] buffer (用于 residual 等)
    pub intermediate: Tensor,

    // ═══ Token 级 buffer ═══
    /// 输入 token ids, [max_batch_tokens], I32
    pub input_tokens: Tensor,
    /// Input positions, [max_batch_tokens], I32 (设备上, 供 RoPE/scatter 等 CUDA kernel 使用)
    pub input_pos: Tensor,
    /// Input positions 的 host staging buffer, [max_batch_tokens], I32, CPU
    pub input_pos_cpu: Tensor,

    // ═══ Sin/Cos cache (从 InferenceState 复制或共享) ═══
    /// [max_seq_len, head_size]
    pub sin_cache: Tensor,
    pub cos_cache: Tensor,

    // ═══ 输出 ═══
    /// Logits, [max_batch_seqs, vocab_size]
    pub logits: Tensor,
    /// 裁剪到 tokenizer_vocab_size 的 logits, [max_batch_seqs, tokenizer_vocab_size]
    pub logits_trim: Tensor,

    // ═══ 每层 w1/w3 的独立连续 buffer（避免在 capture 中分配）═══
    //
    // Q/K/V 不再占独立 buffer：它们现在是 `qkv_out` 的 strided 列视图，由模型
    // 层用 `Tensor::narrow(1, ..., ...)` 零拷贝切出，见 `llama3::Attention::forward`。
    /// [max_batch_tokens, intermediate_size]
    pub w1_out: Tensor,
    /// [max_batch_tokens, intermediate_size]
    pub w3_out: Tensor,

    // ═══ batched flash-decoding 辅助 ═══
    /// [max_batch_seqs] I32, device. 每 seq 的 kv_len
    pub kv_lens_dev: Tensor,
    /// [max_batch_seqs] I32, CPU staging
    pub kv_lens_cpu: Tensor,

    // ═══ scatter_kv_batch 用的 device 指针数组 ═══
    /// Device memory，存 **所有层的 B 个 K-cache 起始指针**，shape = [layer_num, max_batch_seqs]
    /// CPU 模式下为 null。
    #[cfg(feature = "cuda")]
    pub k_cache_ptrs_dev: *mut u64,
    #[cfg(feature = "cuda")]
    pub v_cache_ptrs_dev: *mut u64,
    /// 指针数组是否已经被填充过（按 (states, layer_num) 一次性填充，之后 graph replay 复用）。
    ///
    /// Runner 不要直接写这个字段；改 batch 组合时请调用
    /// [`BatchWorkspace::invalidate_batch_member_cache`] 语义更清晰。
    #[cfg(feature = "cuda")]
    pub(crate) cache_ptrs_filled: bool,

    // ═══ scatter_kv_batch 的 per-step i32 小数组（device 常驻，step 入口 refresh）═══
    //
    // runner 在 step 入口调 `refresh_scatter_indices(meta)` 一次性上传，所有层共用。
    // op 内 **零 malloc / 零 sync / 零 per-step H2D**。
    /// [max_batch_seqs] seq i → slot id
    #[cfg(feature = "cuda")]
    pub scatter_slot_indices_dev: *mut i32,
    /// [max_batch_seqs] seq i 的起始 pos
    #[cfg(feature = "cuda")]
    pub scatter_seq_positions_dev: *mut i32,
    /// [max_batch_seqs] seq i 在源里的 token 起点
    #[cfg(feature = "cuda")]
    pub scatter_seq_starts_dev: *mut i32,
    /// [max_batch_seqs] seq i 的 token 数
    #[cfg(feature = "cuda")]
    pub scatter_seq_lens_dev: *mut i32,

    /// layer_num（初始化时由模型 config 传入）
    pub layer_num: usize,

    // ═══ 容量 ═══
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
}

// 裸指针不自动 Send，但 BatchWorkspace 只会被一个 runner 线程独占使用并跨线程移动一次。
#[cfg(feature = "cuda")]
unsafe impl Send for BatchWorkspace {}

impl BatchWorkspace {
    pub fn new(
        config: &RuntimeModelConfig,
        max_batch_tokens: usize,
        max_batch_seqs: usize,
        device: DeviceType,
    ) -> Result<Self> {
        let dim = config.dim;
        let q_dim = config.q_dim;
        let kv_dim = config.kv_dim;
        let inter = config.intermediate_size;
        let head_size = config.head_size;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.seq_len;

        let float_dtype = config.runtime_float_dtype(device)?;
        let int_dtype = DataType::I32;

        // scatter_kv_batch 用的 device 指针数组（仅 CUDA），按 [layer_num, max_batch_seqs] 分配，
        // 一次性填入所有层所有 seq 的 K/V cache 指针，后续 graph replay 无需 H2D 更新
        #[cfg(feature = "cuda")]
        let (k_cache_ptrs_dev, v_cache_ptrs_dev) = match device {
            DeviceType::Cpu => (std::ptr::null_mut::<u64>(), std::ptr::null_mut::<u64>()),
            DeviceType::Cuda(_) => {
                let bytes = config.layer_num * max_batch_seqs * std::mem::size_of::<u64>();
                let mut k_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut v_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut k_ptr, bytes))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut v_ptr, bytes))?;
                }
                (k_ptr as *mut u64, v_ptr as *mut u64)
            }
        };

        // scatter 的 per-step i32 小数组：4 × [max_batch_seqs] 各 alloc 一次，地址稳定。
        #[cfg(feature = "cuda")]
        let (
            scatter_slot_indices_dev,
            scatter_seq_positions_dev,
            scatter_seq_starts_dev,
            scatter_seq_lens_dev,
        ) = match device {
            DeviceType::Cpu => (
                std::ptr::null_mut::<i32>(),
                std::ptr::null_mut::<i32>(),
                std::ptr::null_mut::<i32>(),
                std::ptr::null_mut::<i32>(),
            ),
            DeviceType::Cuda(_) => {
                let bytes = max_batch_seqs * std::mem::size_of::<i32>();
                let mut p0: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut p1: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut p2: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut p3: *mut std::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p0, bytes))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p1, bytes))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p2, bytes))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p3, bytes))?;
                }
                (p0 as *mut i32, p1 as *mut i32, p2 as *mut i32, p3 as *mut i32)
            }
        };

        Ok(Self {
            x: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            rms_out: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            qkv_out: Tensor::new(&[max_batch_tokens, q_dim + 2 * kv_dim], float_dtype, device)?,
            gate_up_out: Tensor::new(&[max_batch_tokens, 2 * inter], float_dtype, device)?,
            ffn_out: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            intermediate: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,

            input_tokens: Tensor::new(&[max_batch_tokens], int_dtype, device)?,
            input_pos: Tensor::new(&[max_batch_tokens], int_dtype, device)?,
            input_pos_cpu: Tensor::new(&[max_batch_tokens], int_dtype, DeviceType::Cpu)?,

            sin_cache: Tensor::new(&[max_seq_len, head_size], float_dtype, device)?,
            cos_cache: Tensor::new(&[max_seq_len, head_size], float_dtype, device)?,

            logits: Tensor::new(&[max_batch_seqs, vocab_size], float_dtype, device)?,
            logits_trim: Tensor::new(&[max_batch_seqs, config.tokenizer_vocab_size], float_dtype, device)?,

            w1_out: Tensor::new(&[max_batch_tokens, inter], float_dtype, device)?,
            w3_out: Tensor::new(&[max_batch_tokens, inter], float_dtype, device)?,

            kv_lens_dev: Tensor::new(&[max_batch_seqs], int_dtype, device)?,
            kv_lens_cpu: Tensor::new(&[max_batch_seqs], int_dtype, DeviceType::Cpu)?,

            #[cfg(feature = "cuda")]
            k_cache_ptrs_dev,
            #[cfg(feature = "cuda")]
            v_cache_ptrs_dev,
            #[cfg(feature = "cuda")]
            cache_ptrs_filled: false,

            #[cfg(feature = "cuda")]
            scatter_slot_indices_dev,
            #[cfg(feature = "cuda")]
            scatter_seq_positions_dev,
            #[cfg(feature = "cuda")]
            scatter_seq_starts_dev,
            #[cfg(feature = "cuda")]
            scatter_seq_lens_dev,

            layer_num: config.layer_num,

            max_batch_tokens,
            max_batch_seqs,
        })
    }

    /// 通知 workspace "下一次 `forward_batch_decode` 的 batch 成员已变化"，
    /// 清掉所有依赖于 "具体 state 集合" 的缓存（目前是 K/V cache 指针数组）。
    ///
    /// Runner 在检测到 decode 组的 slot 集合变化时调用。
    pub fn invalidate_batch_member_cache(&mut self) {
        #[cfg(feature = "cuda")]
        {
            self.cache_ptrs_filled = false;
        }
    }

    /// 刷新本 step 的 scatter per-seq 小数组到 device。
    ///
    /// 4 个数组（`slot_indices / seq_positions / seq_starts / seq_lens`）都是
    /// `[B]` i32，规模极小（B 通常 ≤ 32），合计一次 ≤ 512 bytes H2D，
    /// 远小于 `op::kv_cache::scatter` kernel 本身的一次 launch overhead，
    /// 并且**只在 step 入口一次**，层间共用。
    ///
    /// 调用约定：runner 拿到 meta 后调一次。
    #[cfg(feature = "cuda")]
    pub fn refresh_scatter_indices(
        &mut self,
        meta: &crate::worker::runner::WorkerBatchMeta<'_>,
    ) -> crate::base::error::Result<()> {
        let b = meta.num_seqs();
        if b == 0 {
            return Ok(());
        }
        if b > self.max_batch_seqs {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "refresh_scatter_indices: batch {} > max_batch_seqs {}",
                b, self.max_batch_seqs
            )).into());
        }
        let mut slots = Vec::with_capacity(b);
        let mut poses = Vec::with_capacity(b);
        let mut starts = Vec::with_capacity(b);
        let mut lens = Vec::with_capacity(b);
        for i in 0..b {
            slots.push(meta.seq_slot(i) as i32);
            poses.push(meta.seq_pos(i));
            starts.push(meta.seq_start(i) as i32);
            lens.push(meta.seq_len(i) as i32);
        }
        let bytes = b * std::mem::size_of::<i32>();
        unsafe {
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.scatter_slot_indices_dev as *mut _,
                slots.as_ptr() as *const _,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.scatter_seq_positions_dev as *mut _,
                poses.as_ptr() as *const _,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.scatter_seq_starts_dev as *mut _,
                starts.as_ptr() as *const _,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.scatter_seq_lens_dev as *mut _,
                lens.as_ptr() as *const _,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }
        Ok(())
    }

    /// 把 `states[*].kv_cache.cache[layer][0..capacity]` 的 K/V base 指针**全部**
    /// 一次性上传到 `k_cache_ptrs_dev / v_cache_ptrs_dev`，shape = `[layer_num × B]`。
    ///
    /// 职责：只在 batch 成员改变或 KV 扩容（见 `invalidate_batch_member_cache`）
    /// 之后的**下一次 step 入口**调一次；之后 graph replay 无需再调。
    ///
    /// `slots` 指明哪一个 InferenceState 对应 batch 里哪个"slot id"（顺序即
    /// device 指针数组的行 index）；caller 通常传 `0..states.len()`。
    #[cfg(feature = "cuda")]
    pub fn fill_cache_ptrs_from_states(
        &mut self,
        states: &mut [&mut crate::model::runtime::InferenceState],
    ) -> crate::base::error::Result<()> {
        let layer_num = self.layer_num;
        let cap = self.max_batch_seqs;
        if states.len() > cap {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "fill_cache_ptrs: states {} > max_batch_seqs {}",
                states.len(), cap
            )).into());
        }
        // Host 缓冲：两个 [layer_num × max_batch_seqs] u64，按行 layer-major 排列。
        let total = layer_num * cap;
        let mut k_host: Vec<u64> = vec![0u64; total];
        let mut v_host: Vec<u64> = vec![0u64; total];
        for (slot, st) in states.iter_mut().enumerate() {
            for layer_idx in 0..layer_num {
                let (k_t, v_t) = st.kv_cache.get_mut(layer_idx)?;
                k_host[layer_idx * cap + slot] = k_t.data_ptr_mut() as u64;
                v_host[layer_idx * cap + slot] = v_t.data_ptr_mut() as u64;
            }
        }
        let bytes = total * std::mem::size_of::<u64>();
        unsafe {
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.k_cache_ptrs_dev as *mut _,
                k_host.as_ptr() as *const _,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.v_cache_ptrs_dev as *mut _,
                v_host.as_ptr() as *const _,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }
        self.cache_ptrs_filled = true;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for BatchWorkspace {
    fn drop(&mut self) {
        unsafe {
            if !self.k_cache_ptrs_dev.is_null() {
                let _ = crate::cuda::ffi::cudaFree(self.k_cache_ptrs_dev as *mut _);
            }
            if !self.v_cache_ptrs_dev.is_null() {
                let _ = crate::cuda::ffi::cudaFree(self.v_cache_ptrs_dev as *mut _);
            }
            for p in [
                self.scatter_slot_indices_dev as *mut std::ffi::c_void,
                self.scatter_seq_positions_dev as *mut std::ffi::c_void,
                self.scatter_seq_starts_dev as *mut std::ffi::c_void,
                self.scatter_seq_lens_dev as *mut std::ffi::c_void,
            ] {
                if !p.is_null() {
                    let _ = crate::cuda::ffi::cudaFree(p);
                }
            }
        }
    }
}
