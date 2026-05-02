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

    // ═══ 每层 Q/K/V、w1/w3 的独立连续 buffer（避免在 capture 中分配）═══
    /// [max_batch_tokens, q_dim]
    pub q_out: Tensor,
    /// [max_batch_tokens, kv_dim]
    pub k_out: Tensor,
    /// [max_batch_tokens, kv_dim]
    pub v_out: Tensor,
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

            q_out: Tensor::new(&[max_batch_tokens, q_dim], float_dtype, device)?,
            k_out: Tensor::new(&[max_batch_tokens, kv_dim], float_dtype, device)?,
            v_out: Tensor::new(&[max_batch_tokens, kv_dim], float_dtype, device)?,
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
        }
    }
}
