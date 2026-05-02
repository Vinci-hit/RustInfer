//! GPU 线程：读 `SharedBuffers` 的 CPU 交换区 → 执行 batched forward → 写输出。
//!
//! 职责：把 scheduler 已经摆好的 step metadata 翻译成 `LlmModel` 的 API 调用，
//!       不做调度决策，不感知 ZMQ / 请求语义。

use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;
use std::time::Instant;

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
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

        // 拿一个独立 Arc clone，让 meta 的生命周期不再借自 `self`。
        let shared = Arc::clone(&self.shared);
        // 从 CPU 共享区拿 host slice（零拷贝）。SAFETY: input_meta.ready > 0 表示
        // Server 已完成写入且 Runner 独占这些 buffer 直到本 step 末尾。
        let meta = unsafe {
            WorkerBatchMeta {
                q_start_loc: shared.host_q_start_loc.as_slice(num_seqs + 1),
                slot_indices: shared.host_slot_indices.as_slice(num_seqs),
                token_ids: &[],
                positions: shared.host_positions.as_slice(total_tokens),
                num_decode,
                num_prefill,
            }
        };

        let slots: Vec<usize> = (0..meta.num_seqs()).map(|i| meta.seq_slot(i)).collect();
        self.prepare_state_and_workspace(&slots, &meta)?;
        self.stage_device_inputs(total_tokens, num_seqs)?;
        let mut refs = disjoint_mut(&mut self.states, &slots)?;

        #[cfg(feature = "cuda")]
        let cuda_cfg: Option<&crate::OpConfig> = Some(&self.batch_cuda_cfg);
        #[cfg(not(feature = "cuda"))]
        let cuda_cfg = None;

        let mut output_token_ids = shared.output_token_ids.clone();
        self.model.forward(
            refs.as_mut_slice(),
            &mut self.workspace,
            &meta,
            &mut output_token_ids,
            cuda_cfg,
        )?;

        tracing::debug!(
            "Runner step done in {}us ({}d/{}p)",
            step_start.elapsed().as_micros() as u64,
            num_decode,
            num_prefill,
        );

        // 信号翻转
        shared.input_meta.ready.store(0, Release);
        shared.output_meta.ready.store(num_seqs as u32, Release);
        Ok(())
    }

    fn stage_device_inputs(&mut self, total_tokens: usize, num_seqs: usize) -> Result<()> {
        let mut dst_tokens = self.workspace.input_tokens.slice(&[0], &[total_tokens])?;
        let src_tokens = self.shared.input_token_ids.slice(&[0], &[total_tokens])?;
        dst_tokens.copy_from_on_current_stream(&src_tokens)?;

        let mut dst_pos = self.workspace.input_pos.slice(&[0], &[total_tokens])?;
        let src_pos = self.shared.input_positions.slice(&[0], &[total_tokens])?;
        dst_pos.copy_from_on_current_stream(&src_pos)?;

        let dst_pos_cpu = self.workspace.input_pos_cpu.as_i32_mut()?.as_slice_mut()?;
        let src_pos_cpu = unsafe { self.shared.host_positions.as_slice(total_tokens) };
        dst_pos_cpu[..total_tokens].copy_from_slice(src_pos_cpu);

        let dst_kv = self.workspace.kv_lens_cpu.as_i32_mut()?.as_slice_mut()?;
        let q_start = unsafe { self.shared.host_q_start_loc.as_slice(num_seqs + 1) };
        let pos = unsafe { self.shared.host_positions.as_slice(total_tokens) };
        for i in 0..num_seqs {
            dst_kv[i] = pos[q_start[i] as usize];
        }
        #[cfg(feature = "cuda")]
        {
            let src = self.workspace.kv_lens_cpu.slice(&[0], &[num_seqs])?;
            let mut dst = self.workspace.kv_lens_dev.slice(&[0], &[num_seqs])?;
            dst.copy_from_on_current_stream(&src)?;
        }
        Ok(())
    }

    fn prepare_state_and_workspace(&mut self, slots: &[usize], meta: &WorkerBatchMeta<'_>) -> Result<()> {
        let mut kv_grew = false;
        for (seq_idx, &slot) in slots.iter().enumerate() {
            let required_len = meta.seq_end_pos(seq_idx)?;
            let state = self.states.get_mut(slot).ok_or_else(|| {
                Error::InvalidArgument(format!("state slot {} out of range", slot))
            })?;
            if state.kv_cache.ensure_capacity(required_len)? {
                state.invalidate_decode_graphs();
                kv_grew = true;
            }
        }

        if kv_grew {
            self.workspace.invalidate_batch_member_cache();
            #[cfg(feature = "cuda")]
            self.batch_cuda_cfg.graphs.clear();
        }

        let mut slots_sorted = slots.to_vec();
        slots_sorted.sort_unstable();
        if slots_sorted != self.last_decode_slots {
            self.workspace.invalidate_batch_member_cache();
            self.last_decode_slots = slots_sorted;
        }
        Ok(())
    }
}

/// 本步 metadata 的只读 view，所有字段都是 host slice。
pub struct WorkerBatchMeta<'a> {
    pub q_start_loc: &'a [i32],
    pub slot_indices: &'a [i32],
    pub token_ids: &'a [i32],
    pub positions: &'a [i32],
    pub num_decode: usize,
    pub num_prefill: usize,
}

impl<'a> WorkerBatchMeta<'a> {
    pub fn num_seqs(&self) -> usize {
        self.num_decode + self.num_prefill
    }

    pub fn is_decode_only(&self) -> bool {
        self.num_prefill == 0
    }

    pub fn seq_slot(&self, i: usize) -> usize {
        self.slot_indices[i] as usize
    }

    pub fn seq_start(&self, i: usize) -> usize {
        self.q_start_loc[i] as usize
    }

    pub fn seq_end(&self, i: usize) -> usize {
        self.q_start_loc[i + 1] as usize
    }

    pub fn seq_len(&self, i: usize) -> usize {
        self.seq_end(i) - self.seq_start(i)
    }

    pub fn seq_pos(&self, i: usize) -> i32 {
        self.positions[self.seq_start(i)]
    }

    pub fn seq_tokens(&self, i: usize) -> &'a [i32] {
        &self.token_ids[self.seq_start(i)..self.seq_end(i)]
    }

    pub fn seq_end_pos(&self, i: usize) -> Result<usize> {
        let end_idx = self.seq_end(i).checked_sub(1).ok_or_else(|| {
            Error::InvalidArgument(format!("empty sequence at index {}", i))
        })?;
        let last_pos = usize::try_from(self.positions[end_idx]).map_err(|_| {
            Error::InvalidArgument(format!("negative position {}", self.positions[end_idx]))
        })?;
        last_pos.checked_add(1).ok_or_else(|| {
            Error::InvalidArgument(format!("position overflow at seq {}", i)).into()
        })
    }
}
