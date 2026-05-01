//! GPU 线程：读 `SharedBuffers` 的 CPU 交换区 → 执行 batched forward → 写输出。
//!
//! 职责：把 scheduler 已经摆好的 step metadata 翻译成 `LlmModel` 的 API 调用，
//!       不做调度决策，不感知 ZMQ / 请求语义。

use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;
use std::time::Instant;

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::base::slice_utils::disjoint_mut;
use crate::model::llm::LlmModel;
use crate::model::runtime::InferenceState;
use crate::worker::batch_workspace::BatchWorkspace;
use crate::worker::shared_buffers::SharedBuffers;

/// GPU 线程。泛型参数 `M` 是满足 [`LlmModel`] 的具体模型实现。
pub struct ModelRunner<M: LlmModel> {
    model: M,
    states: Vec<InferenceState>,
    shared: Arc<SharedBuffers>,
    device_id: i32,

    // ── batch decode 状态 ──
    workspace: BatchWorkspace,
    /// Batch 专用 CudaConfig；持有 split-K flash-decode workspace 和 graph slot HashMap。
    #[cfg(feature = "cuda")]
    batch_cuda_cfg: crate::cuda::CudaConfig,
    /// 上一步 decode 用到的 slot 集合（排序后）。变化时需要让 workspace 的
    /// batch-member 级缓存失效 + 让 llama3 按新 batch_size 重新 capture graph。
    last_decode_slots: Vec<usize>,
}

impl<M: LlmModel> ModelRunner<M> {
    pub fn new(
        model: M,
        states: Vec<InferenceState>,
        shared: Arc<SharedBuffers>,
        device_id: i32,
    ) -> Result<Self> {
        let device = DeviceType::Cuda(device_id);
        let max_batch_tokens = shared.max_batch_tokens;
        let max_batch_seqs = shared.max_seqs;

        let mut workspace = BatchWorkspace::new(
            model.config(),
            max_batch_tokens,
            max_batch_seqs,
            device,
        )?;

        // sin/cos cache 直接从模型的 RoPE 超参算出来，不再依赖 "states[0] 借拷贝"。
        model.fill_rope_cache(&mut workspace.sin_cache, &mut workspace.cos_cache)?;

        #[cfg(feature = "cuda")]
        let batch_cuda_cfg = crate::cuda::CudaConfig::new()?
            .with_flash_decode(
                model.config().head_num,
                model.config().head_size,
                max_batch_seqs,
            )?;

        Ok(Self {
            model,
            states,
            shared,
            device_id,
            workspace,
            #[cfg(feature = "cuda")]
            batch_cuda_cfg,
            last_decode_slots: Vec::new(),
        })
    }

    /// GPU 线程主循环。外层是纯 `loop`；真实逻辑在 [`Self::run_step`]。
    ///
    /// 任何 step 里返回的错误都会被 log 后 panic（因为 Runner 线程退出意味着
    /// Server 永远等不到 output_ready，进程必须同时结束）。
    pub fn run(mut self) {
        tracing::info!("ModelRunner started on device {} (Phase 2: batched decode)", self.device_id);
        loop {
            if let Err(e) = self.run_step() {
                tracing::error!("ModelRunner fatal: {:?}", e);
                panic!("ModelRunner fatal: {:?}", e);
            }
        }
    }

    fn run_step(&mut self) -> Result<()> {
        // spin wait input ready
        let total_tokens = loop {
            let v = self.shared.input_meta.ready.load(Acquire);
            if v > 0 {
                break v as usize;
            }
            std::hint::spin_loop();
        };

        let num_decode = self.shared.input_meta.num_decode_seqs.load(Acquire) as usize;
        let num_prefill = self.shared.input_meta.num_prefill_seqs.load(Acquire) as usize;
        let num_seqs = num_decode + num_prefill;
        let step_start = Instant::now();

        // 拿一个独立 Arc clone，让 meta 的生命周期不再借自 `self`——这样下面
        // `self.run_batch_decode(&meta, ...)` 的 `&mut self` 借用不会与 meta 冲突。
        let shared = Arc::clone(&self.shared);
        // 从 CPU 共享区拿 host slice（零拷贝）。SAFETY: input_meta.ready > 0 表示
        // Server 已完成写入且 Runner 独占这些 buffer 直到本 step 末尾。
        let meta = unsafe {
            StepMeta {
                q_start_loc: shared.input_q_start_loc.as_slice(num_seqs + 1),
                slot_indices: shared.input_slot_indices.as_slice(num_seqs),
                token_ids: shared.input_token_ids.as_slice(total_tokens),
                positions: shared.input_positions.as_slice(total_tokens),
            }
        };

        // 分组：seq_len==1 → decode；seq_len>1 → prefill。
        let mut decode_order: Vec<usize> = Vec::with_capacity(num_decode);
        let mut prefill_order: Vec<usize> = Vec::with_capacity(num_prefill);
        for i in 0..num_seqs {
            if meta.seq_len(i) == 1 {
                decode_order.push(i);
            } else {
                prefill_order.push(i);
            }
        }

        let mut output_tokens = vec![0i32; num_seqs];

        if !decode_order.is_empty() {
            self.run_batch_decode(&decode_order, &meta, &mut output_tokens)?;
        }
        for &seq_idx in &prefill_order {
            let slot = meta.seq_slot(seq_idx);
            let (tokens, start_pos, seq_len) = meta.seq_prefill(seq_idx);
            let tok = self.model.forward_prefill(&mut self.states[slot], tokens, start_pos, seq_len)?;
            output_tokens[seq_idx] = tok;
        }

        // 写输出（CPU memcpy）。SAFETY: output_meta.ready==0 由主循环协议保证。
        unsafe {
            shared.output_token_ids.as_mut_slice(num_seqs).copy_from_slice(&output_tokens);
        }

        tracing::debug!(
            "Runner step done in {}us ({}d/{}p)",
            step_start.elapsed().as_micros() as u64,
            decode_order.len(),
            prefill_order.len(),
        );

        // 信号翻转
        shared.input_meta.ready.store(0, Release);
        shared.output_meta.ready.store(num_seqs as u32, Release);
        Ok(())
    }

    /// 把本步所有 decode seq 聚合成一次 `forward_batch_decode`。
    fn run_batch_decode(
        &mut self,
        decode_order: &[usize],
        meta: &StepMeta<'_>,
        output_tokens: &mut [i32],
    ) -> Result<()> {
        let slots: Vec<usize> = decode_order.iter().map(|&i| meta.seq_slot(i)).collect();
        let positions: Vec<i32> = decode_order.iter().map(|&i| meta.seq_pos(i)).collect();

        // batch 组合变化时让 workspace 的 KV 指针缓存失效。
        let mut slots_sorted = slots.clone();
        slots_sorted.sort_unstable();
        if slots_sorted != self.last_decode_slots {
            self.workspace.invalidate_batch_member_cache();
            self.last_decode_slots = slots_sorted;
        }

        let mut refs = disjoint_mut(&mut self.states, &slots)?;

        #[cfg(feature = "cuda")]
        let cuda_cfg: Option<&crate::OpConfig> = Some(&self.batch_cuda_cfg);
        #[cfg(not(feature = "cuda"))]
        let cuda_cfg = None;

        let result = self.model.forward_batch_decode(
            refs.as_mut_slice(),
            &mut self.workspace,
            &positions,
            cuda_cfg,
        )?;
        debug_assert_eq!(result.len(), decode_order.len());

        for (i, &seq_idx) in decode_order.iter().enumerate() {
            output_tokens[seq_idx] = result[i];
        }
        Ok(())
    }
}

/// 本步 metadata 的只读 view，所有字段都是 host slice。
struct StepMeta<'a> {
    q_start_loc: &'a [i32],
    slot_indices: &'a [i32],
    token_ids: &'a [i32],
    positions: &'a [i32],
}

impl<'a> StepMeta<'a> {
    fn seq_slot(&self, i: usize) -> usize {
        self.slot_indices[i] as usize
    }
    fn seq_pos(&self, i: usize) -> i32 {
        self.positions[self.q_start_loc[i] as usize]
    }
    fn seq_len(&self, i: usize) -> usize {
        (self.q_start_loc[i + 1] - self.q_start_loc[i]) as usize
    }
    fn seq_prefill(&self, i: usize) -> (&'a [i32], i32, usize) {
        let start = self.q_start_loc[i] as usize;
        let end = self.q_start_loc[i + 1] as usize;
        (&self.token_ids[start..end], self.positions[start], end - start)
    }
}
