use std::sync::atomic::{AtomicU8, AtomicU32};
use std::sync::Arc;

use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

/// 预分配的共享 GPU buffer，Server 和 Runner 通过它交换数据。
///
/// 同步协议:
///   - Server 写 input buffer → store(input_ready, total_tokens) [Release]
///   - Runner load(input_ready) [Acquire] → 读 input → 执行 → 写 output
///     → store(input_ready, 0) [Release] → store(output_ready, num_seqs) [Release]
///   - Server load(output_ready) [Acquire] → 读 output → store(output_ready, 0)
///     → 写下一步 input (此时 input_ready==0，安全)
pub struct SharedBuffers {
    // ═══ Input (Server 写, Runner 读) ═══
    pub input_token_ids: Tensor,
    pub input_positions: Tensor,
    pub input_q_start_loc: Tensor,
    pub input_context_lens: Tensor,
    pub input_slot_indices: Tensor,

    // ═══ Output (Runner 写, Server 读) ═══
    pub output_token_ids: Tensor,

    // ═══ 同步信号 ═══
    pub input_meta: InputMeta,
    pub output_meta: OutputMeta,

    // ═══ 容量上限 ═══
    pub max_batch_tokens: usize,
    pub max_seqs: usize,
}

/// Server → Runner 同步信号 (CPU 原子变量)
pub struct InputMeta {
    /// 0 = 无输入; >0 = 输入就绪, 值 = total_tokens
    pub ready: AtomicU32,
    /// 0 = DecodeOnly, 1 = MixedBatch
    pub batch_type: AtomicU8,
    /// 本步 decode 序列数
    pub num_decode_seqs: AtomicU32,
    /// 本步 prefill 序列数
    pub num_prefill_seqs: AtomicU32,
    /// 本步 prefill token 总数
    pub num_prefill_tokens: AtomicU32,
}

/// Runner → Server 同步信号
pub struct OutputMeta {
    /// 0 = 无输出; >0 = 输出就绪, 值 = num_seqs
    pub ready: AtomicU32,
}

impl SharedBuffers {
    /// 预分配所有共享 buffer
    ///
    /// # Arguments
    /// * `max_batch_tokens` - 单步最大 token 数 (如 2048)
    /// * `max_seqs` - 最大并发序列数 (如 64)
    /// * `device` - GPU 设备
    pub fn new(max_batch_tokens: usize, max_seqs: usize, device: DeviceType) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            input_token_ids: Tensor::new(&[max_batch_tokens], DataType::I32, device)?,
            input_positions: Tensor::new(&[max_batch_tokens], DataType::I32, device)?,
            input_q_start_loc: Tensor::new(&[max_seqs + 1], DataType::I32, device)?,
            input_context_lens: Tensor::new(&[max_seqs], DataType::I32, device)?,
            input_slot_indices: Tensor::new(&[max_seqs], DataType::I32, device)?,
            output_token_ids: Tensor::new(&[max_seqs], DataType::I32, device)?,
            input_meta: InputMeta {
                ready: AtomicU32::new(0),
                batch_type: AtomicU8::new(0),
                num_decode_seqs: AtomicU32::new(0),
                num_prefill_seqs: AtomicU32::new(0),
                num_prefill_tokens: AtomicU32::new(0),
            },
            output_meta: OutputMeta {
                ready: AtomicU32::new(0),
            },
            max_batch_tokens,
            max_seqs,
        }))
    }

    // ═══════════════════════════════════════════════════════════
    // H2D / D2H 方法
    //
    // Safety 说明:
    //   这些方法通过 &self 调用却写入 GPU buffer。这在 Rust 的借用规则下
    //   看起来违反了 &mut 要求,但实际上是安全的:
    //   1. GPU memory 的写入是通过 cudaMemcpy 完成的,不受 Rust 内存模型管辖
    //   2. 同步协议(input_ready / output_ready 原子信号)保证了:
    //      - Server 写 input 时 Runner 不会读 input (input_ready == 0)
    //      - Server 读 output 时 Runner 不会写 output (output_ready > 0)
    //   3. 这与 UnsafeCell 的语义类似: 外部同步保证独占
    // ═══════════════════════════════════════════════════════════

    /// 将 CPU i32 slice 写入指定 GPU tensor 的前 count 个元素。
    ///
    /// # Safety guarantee
    /// 调用者必须通过 input_ready == 0 确保 Runner 不在读此 buffer。
    pub fn write_input_i32(&self, tensor: &Tensor, src: &[i32], count: usize) -> Result<()> {
        debug_assert!(count <= src.len());
        let elem_size = std::mem::size_of::<i32>();
        let copy_bytes = count * elem_size;
        debug_assert!(copy_bytes <= tensor.buffer().len_bytes());

        let dst_ptr = tensor.buffer().as_ptr() as *mut u8;
        let src_ptr = src.as_ptr() as *const u8;

        #[cfg(feature = "cuda")]
        {
            if tensor.device().is_cuda() {
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                        dst_ptr as *mut _,
                        src_ptr as *const _,
                        copy_bytes,
                        crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ))?;
                }
                return Ok(());
            }
        }

        // CPU fallback
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_bytes);
        }
        Ok(())
    }

    /// 从 GPU tensor 的前 count 个 i32 元素 D2H copy 到 CPU Vec。
    ///
    /// # Safety guarantee
    /// 调用者必须通过 output_ready > 0 确保 Runner 不在写此 buffer。
    pub fn read_output_i32(&self, tensor: &Tensor, count: usize) -> Result<Vec<i32>> {
        let elem_size = std::mem::size_of::<i32>();
        let copy_bytes = count * elem_size;
        debug_assert!(copy_bytes <= tensor.buffer().len_bytes());

        let mut result = vec![0i32; count];
        let src_ptr = tensor.buffer().as_ptr() as *const u8;
        let dst_ptr = result.as_mut_ptr() as *mut u8;

        #[cfg(feature = "cuda")]
        {
            if tensor.device().is_cuda() {
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                        dst_ptr as *mut _,
                        src_ptr as *const _,
                        copy_bytes,
                        crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    ))?;
                }
                return Ok(result);
            }
        }

        // CPU fallback
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_bytes);
        }
        Ok(result)
    }
}
