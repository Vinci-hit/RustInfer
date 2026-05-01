//! 进程内共享的 Server ↔ Runner 交换区 + 同步协议。
//!
//! # 设计约定
//! - 所有 i32 metadata 都在 **CPU** 堆上，通过 `SyncBuf<T>` 零拷贝读写；
//!   Runner / Server 使用原生 `&[i32]` / `&mut [i32]`，没有 cudaMemcpy / Vec 分配。
//! - 唯一的跨线程同步机制是 `input_meta` / `output_meta` 里的原子信号：
//!
//! ```text
//! Server 写 input ─────────────► store(input.ready, total_tokens)      [Release]
//! Runner   load(input.ready)  ─► 读 input → 执行 → 写 output
//!          store(input.ready, 0)            [Release]
//!          store(output.ready, num_seqs)    [Release]
//! Server   load(output.ready) ─► 读 output
//!          store(output.ready, 0)           [Release]
//!          写下一步 input  (此时 input.ready == 0，安全)
//! ```
//!
//! 有了这个协议，`SyncBuf` 的 `&self` 读写方法在内部用 `UnsafeCell`，不走锁。

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32};

use crate::base::error::Result;

/// CPU 上的固定容量 buffer，跨线程通过外部 atomic 信号保证独占访问。
///
/// # Safety invariant
/// 调用者负责通过 [`InputMeta::ready`] / [`OutputMeta::ready`] 等原子信号，
/// 保证任意时刻对同一 `SyncBuf` 的读/写访问不会重叠：
/// - 仅当 `input.ready == 0` 时 Server 能写 input 类 buffer；
/// - 仅当 `input.ready > 0` 且 `output.ready == 0` 时 Runner 能读 input / 写 output；
/// - 仅当 `output.ready > 0` 时 Server 能读 output。
pub struct SyncBuf<T> {
    cell: UnsafeCell<Box<[T]>>,
}

// SAFETY: SyncBuf 的所有访问都经过外部同步协议（见类型文档）。
unsafe impl<T: Send> Send for SyncBuf<T> {}
unsafe impl<T: Send> Sync for SyncBuf<T> {}

impl<T: Copy + Default> SyncBuf<T> {
    pub fn new(capacity: usize) -> Self {
        let v = vec![T::default(); capacity].into_boxed_slice();
        Self { cell: UnsafeCell::new(v) }
    }
}

impl<T> SyncBuf<T> {
    /// 容量（元素数，不是字节）。
    pub fn capacity(&self) -> usize {
        // 从 raw pointer 拿长度而不产生对内部的 alias 借用：用 addr_of! 读 Box fat pointer 的 len 字。
        // Box<[T]> 的内存布局是 (data_ptr, len)；直接投成 &[*mut T, usize] 拿 len。
        // 更简单：UnsafeCell::get 后直接 read fat ptr 的两字，但那会 move Box —— 所以还是
        // 在 unsafe 块内用 (&*raw_ptr).len()，并在函数签名上加 allow。
        unsafe {
            let ptr = self.cell.get();
            #[allow(clippy::borrow_as_ptr)]
            let b: &Box<[T]> = &*ptr;
            b.len()
        }
    }

    /// 读视图。
    ///
    /// # Safety
    /// 调用方保证此时没有任何其他线程持有 `&mut` 视图。
    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        debug_assert!(len <= self.capacity());
        // SAFETY: UnsafeCell 保证通过 &self 能拿 &mut 内部；
        //         外部同步协议保证此时无 overlapping borrow。
        let ptr: *const T = unsafe { (*self.cell.get()).as_ptr() };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// 可写视图。
    ///
    /// # Safety
    /// 调用方保证此时没有任何其他线程持有视图（读或写）。
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice(&self, len: usize) -> &mut [T] {
        debug_assert!(len <= self.capacity());
        let ptr: *mut T = unsafe { (*self.cell.get()).as_mut_ptr() };
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

/// 预分配的 Server ↔ Runner 交换区。
pub struct SharedBuffers {
    // ── Input（Server 写，Runner 读）──
    pub input_token_ids: SyncBuf<i32>,
    pub input_positions: SyncBuf<i32>,
    pub input_q_start_loc: SyncBuf<i32>,
    pub input_context_lens: SyncBuf<i32>,
    pub input_slot_indices: SyncBuf<i32>,

    // ── Output（Runner 写，Server 读）──
    pub output_token_ids: SyncBuf<i32>,

    // ── 同步信号 ──
    pub input_meta: InputMeta,
    pub output_meta: OutputMeta,

    // ── 容量上限 ──
    pub max_batch_tokens: usize,
    pub max_seqs: usize,
}

/// Server → Runner 同步信号。
pub struct InputMeta {
    /// 0 = 无输入；>0 = 输入就绪，值 = total_tokens
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

/// Runner → Server 同步信号。
pub struct OutputMeta {
    /// 0 = 无输出；>0 = 输出就绪，值 = num_seqs
    pub ready: AtomicU32,
}

impl SharedBuffers {
    /// 按 `(max_batch_tokens, max_seqs)` 预分配所有 CPU 侧交换区。
    ///
    /// `device` 参数保留为未来可能的 pinned memory / GPU 镜像设计（当前未使用）。
    pub fn new(max_batch_tokens: usize, max_seqs: usize) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            input_token_ids: SyncBuf::new(max_batch_tokens),
            input_positions: SyncBuf::new(max_batch_tokens),
            input_q_start_loc: SyncBuf::new(max_seqs + 1),
            input_context_lens: SyncBuf::new(max_seqs),
            input_slot_indices: SyncBuf::new(max_seqs),
            output_token_ids: SyncBuf::new(max_seqs),
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
}
