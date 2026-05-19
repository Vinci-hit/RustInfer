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
    /// Attention fused 输出 buffer，`[max_batch_tokens, q_dim]`。
    /// 由 attention kernel 写入，wo 读出。Llama3 / Qwen3 共用。
    pub intermediate: Tensor,
    /// lm_head 之前的 sample buffer，`[max_batch_seqs, dim]`。
    /// 与 `intermediate` 分离以便 Qwen3 这种 `q_dim != dim` 的模型也能拿到
    /// dense view（`intermediate` 宽度 = q_dim，对它做 col-narrow 取 dim 是
    /// strided，不利于下游 Matmul）。
    pub sample_hidden: Tensor,

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

    /// Host-side staging 镜像：`[layer_num × max_batch_seqs]` u64 K/V cache 指针表。
    /// `fill_cache_ptrs_from_states` 在这里 read-modify，然后**单向 H2D**到
    /// `k_cache_ptrs_dev` / `v_cache_ptrs_dev`，避免每步都 D2H 读回。
    /// 跨 step 持久化保留，配合 `cache_ptrs_filled` 标志一并使用。
    #[cfg(feature = "cuda")]
    k_cache_ptrs_host: Vec<u64>,
    #[cfg(feature = "cuda")]
    v_cache_ptrs_host: Vec<u64>,

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

    /// Host staging for scatter control arrays. These buffers are owned by the
    /// workspace and live across runner steps, so async H2D copies never borrow
    /// stack-local vectors.
    #[cfg(feature = "cuda")]
    scatter_slot_indices_host: Vec<i32>,
    #[cfg(feature = "cuda")]
    scatter_seq_positions_host: Vec<i32>,
    #[cfg(feature = "cuda")]
    scatter_seq_starts_host: Vec<i32>,
    #[cfg(feature = "cuda")]
    scatter_seq_lens_host: Vec<i32>,

    // ═══ Flash-Decoding split-K workspace ═══
    //
    // Flash-Decoding decode-path 用 split-K 策略并行扫 KV（每 seq × 每 head 分
    // 成 num_splits 条 block），每条 block 把"partial output + log-sum-exp"写到
    // 这块 workspace，再由一个 reduction kernel 合并。存储**必须是 f32**：
    //   - partial output accumulator：几千次 BF16/FP16 FMA 的累加误差不可接受；
    //   - log-sum-exp：跨 split 合并时 `exp(lse_i - lse_max)` 对 ULP 极敏感。
    //
    // 尺寸由 `FlashAttnDecodeBatch::workspace_bytes(max_batch_seqs, q_heads, head_dim)`
    // 决定；地址跨 step 稳定，graph-capture 友好。
    #[cfg(feature = "cuda")]
    pub flash_decode_workspace_dev: *mut f32,

    // ═══ Ragged prefill 的 kernel 调度表 ═══
    //
    // Flash-Attn ragged kernel 的 grid 大小 = `total_q_tiles = Σ ceil(q_len_i / RAGGED_Q_TILE)`。
    // 每个 block 要通过下列 3 张表反查"我是哪个 request 的第几个 Q-tile"：
    //
    // - `cu_q_lens_dev`  `[max_batch_seqs + 1]` i32: `q_len` 前缀和；
    // - `block2req_dev`  `[max_q_tiles]`        i32: tile → request；
    // - `block2tile_dev` `[max_q_tiles]`        i32: tile → request 内的第几 tile。
    //
    // `max_q_tiles` 是本 workspace 所能容纳的上界 `ceil(max_batch_tokens /
    // RAGGED_Q_TILE)`。Runner 每 step 入口（只在 `num_prefill > 0` 时）调
    // [`refresh_ragged_plan`] 做一次 host-compute + 3 次小 H2D。
    #[cfg(feature = "cuda")]
    pub ragged_cu_q_lens_dev: *mut i32,
    #[cfg(feature = "cuda")]
    pub ragged_block2req_dev: *mut i32,
    #[cfg(feature = "cuda")]
    pub ragged_block2tile_dev: *mut i32,
    #[cfg(feature = "cuda")]
    pub ragged_max_q_tiles: usize,

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

        // Flash-Decoding split-K workspace —— 按 kernel 自报的字节数一次分配。
        // 用 `max_batch_seqs` 做上界，batch 未满时 kernel 只用前 batch 份；地址稳定。
        #[cfg(feature = "cuda")]
        let flash_decode_workspace_dev = match device {
            DeviceType::Cpu => std::ptr::null_mut::<f32>(),
            DeviceType::Cuda(_) => {
                let bytes = crate::op::attention::FlashAttnDecodeBatch::workspace_bytes(
                    max_batch_seqs,
                    config.head_num,
                    config.head_size,
                );
                if bytes == 0 {
                    std::ptr::null_mut::<f32>()
                } else {
                    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                    unsafe {
                        crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p, bytes))?;
                    }
                    p as *mut f32
                }
            }
        };

        // Ragged prefill 调度表：按 max_batch_tokens 为上界估算 max_q_tiles。
        #[cfg(feature = "cuda")]
        let (
            ragged_cu_q_lens_dev,
            ragged_block2req_dev,
            ragged_block2tile_dev,
            ragged_max_q_tiles,
        ) = match device {
            DeviceType::Cpu => (
                std::ptr::null_mut::<i32>(),
                std::ptr::null_mut::<i32>(),
                std::ptr::null_mut::<i32>(),
                0usize,
            ),
            DeviceType::Cuda(_) => {
                let max_tiles = max_batch_tokens
                    .div_ceil(crate::op::attention::ragged::RAGGED_Q_TILE)
                    .max(1);
                let cu_bytes = (max_batch_seqs + 1) * std::mem::size_of::<i32>();
                let tile_bytes = max_tiles * std::mem::size_of::<i32>();
                let mut p_cu: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut p_b2r: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut p_b2t: *mut std::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p_cu, cu_bytes))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p_b2r, tile_bytes))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaMalloc(&mut p_b2t, tile_bytes))?;
                }
                (
                    p_cu as *mut i32,
                    p_b2r as *mut i32,
                    p_b2t as *mut i32,
                    max_tiles,
                )
            }
        };

        Ok(Self {
            x: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            rms_out: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            qkv_out: Tensor::new(&[max_batch_tokens, q_dim + 2 * kv_dim], float_dtype, device)?,
            gate_up_out: Tensor::new(&[max_batch_tokens, 2 * inter], float_dtype, device)?,
            ffn_out: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            // attention fused 输出 buffer：宽度 = q_dim（attention kernel 期望的
            // `[T, q_dim]` 布局），与 wo 输入对齐。Llama3 q_dim==dim，Qwen3
            // q_dim>dim 时此 buffer 仍按 q_dim 分配。
            intermediate: Tensor::new(&[max_batch_tokens, q_dim], float_dtype, device)?,
            // sample buffer：单独 `[max_batch_seqs, dim]`，下游 lm_head 直接吃
            // dense view。
            sample_hidden: Tensor::new(&[max_batch_seqs, dim], float_dtype, device)?,

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
            k_cache_ptrs_host: vec![0u64; config.layer_num * max_batch_seqs],
            #[cfg(feature = "cuda")]
            v_cache_ptrs_host: vec![0u64; config.layer_num * max_batch_seqs],

            #[cfg(feature = "cuda")]
            scatter_slot_indices_dev,
            #[cfg(feature = "cuda")]
            scatter_seq_positions_dev,
            #[cfg(feature = "cuda")]
            scatter_seq_starts_dev,
            #[cfg(feature = "cuda")]
            scatter_seq_lens_dev,
            #[cfg(feature = "cuda")]
            scatter_slot_indices_host: vec![0i32; max_batch_seqs],
            #[cfg(feature = "cuda")]
            scatter_seq_positions_host: vec![0i32; max_batch_seqs],
            #[cfg(feature = "cuda")]
            scatter_seq_starts_host: vec![0i32; max_batch_seqs],
            #[cfg(feature = "cuda")]
            scatter_seq_lens_host: vec![0i32; max_batch_seqs],

            #[cfg(feature = "cuda")]
            flash_decode_workspace_dev,

            #[cfg(feature = "cuda")]
            ragged_cu_q_lens_dev,
            #[cfg(feature = "cuda")]
            ragged_block2req_dev,
            #[cfg(feature = "cuda")]
            ragged_block2tile_dev,
            #[cfg(feature = "cuda")]
            ragged_max_q_tiles,

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
    ///
    /// hot-path：4 个小 H2D 走 `cudaMemcpyAsync(stream)` 而非同步 cudaMemcpy。
    /// host 端使用 workspace-owned staging buffer，保证异步 copy 的源内存
    /// 生命周期覆盖整个 runner step。
    #[cfg(feature = "cuda")]
    pub fn refresh_scatter_indices(
        &mut self,
        meta: &crate::worker::runner::WorkerBatchMeta<'_>,
        stream: crate::cuda::ffi::cudaStream_t,
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
        for i in 0..b {
            self.scatter_slot_indices_host[i] = meta.seq_slot(i) as i32;
            self.scatter_seq_positions_host[i] = meta.seq_pos(i);
            self.scatter_seq_starts_host[i] = meta.seq_start(i) as i32;
            self.scatter_seq_lens_host[i] = meta.seq_len(i) as i32;
        }
        let bytes = b * std::mem::size_of::<i32>();
        unsafe {
            use crate::cuda::ffi::{cudaMemcpyAsync, cudaMemcpyKind};
            crate::cuda_check!(cudaMemcpyAsync(
                self.scatter_slot_indices_dev as *mut _,
                self.scatter_slot_indices_host.as_ptr() as *const _,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;
            crate::cuda_check!(cudaMemcpyAsync(
                self.scatter_seq_positions_dev as *mut _,
                self.scatter_seq_positions_host.as_ptr() as *const _,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;
            crate::cuda_check!(cudaMemcpyAsync(
                self.scatter_seq_starts_dev as *mut _,
                self.scatter_seq_starts_host.as_ptr() as *const _,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;
            crate::cuda_check!(cudaMemcpyAsync(
                self.scatter_seq_lens_dev as *mut _,
                self.scatter_seq_lens_host.as_ptr() as *const _,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
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
    /// `states[i]` **必须**对应 `slot_ids[i]`，即 server 已经按 "slot 顺序" gather
    /// 好 refs；函数内部按该 slot 下标填入指针表，和 kernel 侧的
    /// `req_to_slot_dev / slot_indices_dev` 一致。
    ///
    /// hot-path 优化：用 host-side staging 镜像（[`Self::k_cache_ptrs_host`] /
    /// `v_cache_ptrs_host`，跨 step 持久持有），避免每次都 D2H 读回 device 表。
    /// 调用方只覆盖本次列出的 slot，其它 slot 的 staging 项保留。两次小 H2D
    /// 走 `cudaMemcpyAsync(stream)` 避免同步打断 GPU pipeline。
    #[cfg(feature = "cuda")]
    pub fn fill_cache_ptrs_from_states(
        &mut self,
        slot_ids: &[usize],
        states: &mut [&mut crate::model::runtime::InferenceState],
        stream: crate::cuda::ffi::cudaStream_t,
    ) -> crate::base::error::Result<()> {
        if states.len() != slot_ids.len() {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "fill_cache_ptrs: states {} != slot_ids {}",
                states.len(), slot_ids.len()
            )).into());
        }
        let layer_num = self.layer_num;
        let cap = self.max_batch_seqs;
        for &slot in slot_ids {
            if slot >= cap {
                return Err(crate::base::error::Error::InvalidArgument(format!(
                    "fill_cache_ptrs: slot {} >= max_batch_seqs {}", slot, cap
                )).into());
            }
        }
        // 直接在 host staging 上 read-modify-write，**不读回 device**。
        // staging 跨 step 持久化，前一步未涉及的 slot 项保留旧值。
        for (idx, st) in states.iter_mut().enumerate() {
            let slot = slot_ids[idx];
            for layer_idx in 0..layer_num {
                let (k_t, v_t) = st.kv_cache.get_mut(layer_idx)?;
                self.k_cache_ptrs_host[layer_idx * cap + slot] = k_t.data_ptr_mut() as u64;
                self.v_cache_ptrs_host[layer_idx * cap + slot] = v_t.data_ptr_mut() as u64;
            }
        }
        let bytes = layer_num * cap * std::mem::size_of::<u64>();
        unsafe {
            use crate::cuda::ffi::{cudaMemcpyAsync, cudaMemcpyKind};
            crate::cuda_check!(cudaMemcpyAsync(
                self.k_cache_ptrs_dev as *mut _,
                self.k_cache_ptrs_host.as_ptr() as *const _,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;
            crate::cuda_check!(cudaMemcpyAsync(
                self.v_cache_ptrs_dev as *mut _,
                self.v_cache_ptrs_host.as_ptr() as *const _,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;
        }
        self.cache_ptrs_filled = true;
        Ok(())
    }

    /// 在 step 入口，**num_prefill > 0** 时调一次，host 侧从 meta 算出
    /// ragged kernel 所需的 3 张调度表并 H2D。
    ///
    /// 返回 `total_q_tiles`——本步 ragged kernel 的 `grid.x`。
    #[cfg(feature = "cuda")]
    pub fn refresh_ragged_plan(
        &mut self,
        meta: &crate::worker::runner::WorkerBatchMeta<'_>,
    ) -> crate::base::error::Result<i32> {
        let b = meta.num_seqs();
        if b == 0 {
            return Ok(0);
        }
        let mut q_lens: Vec<i32> = Vec::with_capacity(b);
        for i in 0..b {
            q_lens.push(meta.seq_len(i) as i32);
        }
        let (cu_q_lens, block2req, block2tile) =
            crate::op::attention::ragged::plan_ragged_tiles(&q_lens);
        let total_q_tiles = block2req.len();
        if total_q_tiles > self.ragged_max_q_tiles {
            return Err(crate::base::error::Error::InvalidArgument(format!(
                "refresh_ragged_plan: total_q_tiles {} > max {}",
                total_q_tiles, self.ragged_max_q_tiles
            ))
            .into());
        }
        unsafe {
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.ragged_cu_q_lens_dev as *mut _,
                cu_q_lens.as_ptr() as *const _,
                cu_q_lens.len() * std::mem::size_of::<i32>(),
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.ragged_block2req_dev as *mut _,
                block2req.as_ptr() as *const _,
                block2req.len() * std::mem::size_of::<i32>(),
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
            crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                self.ragged_block2tile_dev as *mut _,
                block2tile.as_ptr() as *const _,
                block2tile.len() * std::mem::size_of::<i32>(),
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }
        Ok(total_q_tiles as i32)
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
                self.flash_decode_workspace_dev as *mut std::ffi::c_void,
                self.ragged_cu_q_lens_dev as *mut std::ffi::c_void,
                self.ragged_block2req_dev as *mut std::ffi::c_void,
                self.ragged_block2tile_dev as *mut std::ffi::c_void,
            ] {
                if !p.is_null() {
                    let _ = crate::cuda::ffi::cudaFree(p);
                }
            }
        }
    }
}
