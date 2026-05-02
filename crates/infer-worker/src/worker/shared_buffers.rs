//! 进程内共享的 Server ↔ Runner 交换区 + 同步协议。
//!
//! Server 负责 CPU batch 组装和 H2D；Runner 只按固定 device buffer 执行 forward。

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32};

use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

/// CPU 上的固定容量小 metadata buffer，跨线程通过外部 atomic 信号保证独占访问。
pub struct SyncBuf<T> {
    cell: UnsafeCell<Box<[T]>>,
}

unsafe impl<T: Send> Send for SyncBuf<T> {}
unsafe impl<T: Send> Sync for SyncBuf<T> {}

impl<T: Copy + Default> SyncBuf<T> {
    pub fn new(capacity: usize) -> Self {
        let v = vec![T::default(); capacity].into_boxed_slice();
        Self { cell: UnsafeCell::new(v) }
    }
}

impl<T> SyncBuf<T> {
    pub fn capacity(&self) -> usize {
        unsafe { (&*self.cell.get()).len() }
    }

    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        assert!(len <= self.capacity(), "SyncBuf::as_slice length {} exceeds capacity {}", len, self.capacity());
        let ptr: *const T = unsafe { (*self.cell.get()).as_ptr() };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice(&self, len: usize) -> &mut [T] {
        assert!(len <= self.capacity(), "SyncBuf::as_mut_slice length {} exceeds capacity {}", len, self.capacity());
        let ptr: *mut T = unsafe { (*self.cell.get()).as_mut_ptr() };
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

/// 预分配的 Server ↔ Runner 交换区。
pub struct SharedBuffers {
    // ── Device input（Server H2D 写，Runner 读）──
    pub input_token_ids: Tensor,
    pub input_positions: Tensor,
    pub input_q_start_loc: Tensor,
    pub input_context_lens: Tensor,
    pub input_slot_indices: Tensor,

    // Runner 仍需要最小 CPU metadata 来索引 state pool / 切 seq range；不承载 token 数据。
    pub host_positions: SyncBuf<i32>,
    pub host_q_start_loc: SyncBuf<i32>,
    pub host_slot_indices: SyncBuf<i32>,

    // ── Device output（Runner 写，Server D2H 读）──
    pub output_token_ids: Tensor,

    pub input_meta: InputMeta,
    pub output_meta: OutputMeta,

    pub max_batch_tokens: usize,
    pub max_seqs: usize,
}

pub struct InputMeta {
    /// 0 = 无输入；>0 = 输入就绪，值 = total_tokens
    pub ready: AtomicU32,
    /// 0 = DecodeOnly, 1 = MixedBatch
    pub batch_type: AtomicU8,
    pub num_decode_seqs: AtomicU32,
    pub num_prefill_seqs: AtomicU32,
    pub num_prefill_tokens: AtomicU32,
}

pub struct OutputMeta {
    /// 0 = 无输出；>0 = 输出就绪，值 = num_seqs
    pub ready: AtomicU32,
}

impl SharedBuffers {
    pub fn new(max_batch_tokens: usize, max_seqs: usize, device: DeviceType) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            input_token_ids: Tensor::new(&[max_batch_tokens], DataType::I32, device)?,
            input_positions: Tensor::new(&[max_batch_tokens], DataType::I32, device)?,
            input_q_start_loc: Tensor::new(&[max_seqs + 1], DataType::I32, device)?,
            input_context_lens: Tensor::new(&[max_seqs], DataType::I32, device)?,
            input_slot_indices: Tensor::new(&[max_seqs], DataType::I32, device)?,
            host_positions: SyncBuf::new(max_batch_tokens),
            host_q_start_loc: SyncBuf::new(max_seqs + 1),
            host_slot_indices: SyncBuf::new(max_seqs),
            output_token_ids: Tensor::new(&[max_seqs], DataType::I32, device)?,
            input_meta: InputMeta {
                ready: AtomicU32::new(0),
                batch_type: AtomicU8::new(0),
                num_decode_seqs: AtomicU32::new(0),
                num_prefill_seqs: AtomicU32::new(0),
                num_prefill_tokens: AtomicU32::new(0),
            },
            output_meta: OutputMeta { ready: AtomicU32::new(0) },
            max_batch_tokens,
            max_seqs,
        }))
    }
}
