use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;
use std::time::Instant;

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::model::llm::llama3::Llama3;
use crate::runtime::InferenceState;
use crate::worker::batch_workspace::BatchWorkspace;
use crate::worker::shared_buffers::SharedBuffers;

/// ModelRunner — GPU 线程，执行 forward + sample
///
/// Phase 2: decode 组聚合成一次 `forward_batch_decode`（真正的 batched forward）；
///          prefill 组仍串行（seq_len 不一，不好 batch）。
pub struct ModelRunner {
    model: Llama3,
    /// 每个 KV slot 一个 InferenceState
    states: Vec<InferenceState>,
    shared: Arc<SharedBuffers>,
    #[allow(dead_code)]
    device_id: i32,

    // ═══ batch decode 相关 ═══
    workspace: BatchWorkspace,
    /// 专门给 batch decode 用的 CudaConfig（持有 split-K flash-decode workspace + graph slot HashMap）
    #[cfg(feature = "cuda")]
    batch_cuda_cfg: crate::cuda::CudaConfig,
    /// 上一步 batch decode 用到的 slot 集合（排序后），用于判断组合是否变化。
    /// 组合变化时需要：workspace.cache_ptrs_filled = false + 重新 capture graph。
    last_decode_slots: Vec<usize>,
}

impl ModelRunner {
    pub fn new(
        model: Llama3,
        states: Vec<InferenceState>,
        shared: Arc<SharedBuffers>,
        device_id: i32,
    ) -> Result<Self> {
        let device = DeviceType::Cuda(device_id);
        let max_batch_tokens = shared.max_batch_tokens;
        let max_batch_seqs = shared.max_seqs;

        // Batch workspace：一次性按 max 容量分配，后续每步 slice 前缀使用。
        let mut workspace = BatchWorkspace::new(
            model.config(),
            max_batch_tokens,
            max_batch_seqs,
            device,
        )?;

        // sin/cos cache: 从 state[0] 拷贝（所有 state 的 cache 都一样，由 RoPE θ 计算）
        if !states.is_empty() {
            let sin_src = states[0]
                .workspace
                .get(&crate::model::BufferType::SinCache)
                .expect("state workspace missing SinCache");
            let cos_src = states[0]
                .workspace
                .get(&crate::model::BufferType::CosCache)
                .expect("state workspace missing CosCache");
            workspace.sin_cache.copy_from(sin_src)?;
            workspace.cos_cache.copy_from(cos_src)?;
        }

        // Batch decode 用 CudaConfig：按 max_batch_seqs 分配 split-K flash-decode workspace。
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

    /// GPU 线程主循环
    pub fn run(mut self) {
        tracing::info!("ModelRunner started on device {} (Phase 2: batched decode)", self.device_id);

        loop {
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

            tracing::debug!(
                "Runner step: total_tokens={}, decode={}, prefill={}",
                total_tokens, num_decode, num_prefill,
            );

            let start = Instant::now();

            // Runner / Server 间的 metadata buffer 都放在 CPU（见 SharedBuffers::new 注释），
            // 这里 read_output_i32 走纯 host 的 memcpy 分支，不会触发 cudaMemcpy / stream 同步。
            let q_start_loc = self
                .shared
                .read_output_i32(&self.shared.input_q_start_loc, num_seqs + 1)
                .expect("read q_start_loc");
            let slot_indices = self
                .shared
                .read_output_i32(&self.shared.input_slot_indices, num_seqs)
                .expect("read slot_indices");
            let _context_lens = self
                .shared
                .read_output_i32(&self.shared.input_context_lens, num_seqs)
                .expect("read context_lens");
            let token_ids_host = self
                .shared
                .read_output_i32(&self.shared.input_token_ids, total_tokens)
                .expect("D2H input_token_ids");
            let positions_host = self
                .shared
                .read_output_i32(&self.shared.input_positions, total_tokens)
                .expect("D2H input_positions");

            // ═══ 分组: decode (seq_len==1) vs prefill (seq_len>1) ═══
            //
            // Scheduler 的约定（见 server.rs::write_input_buffer）：前 num_decode 个是 decode，
            // 后 num_prefill 个是 prefill。这里按实际 seq_len 再校验一次。
            let mut decode_order: Vec<usize> = Vec::with_capacity(num_decode);
            let mut prefill_order: Vec<usize> = Vec::with_capacity(num_prefill);
            for seq_idx in 0..num_seqs {
                let seq_len = (q_start_loc[seq_idx + 1] - q_start_loc[seq_idx]) as usize;
                if seq_len == 1 {
                    decode_order.push(seq_idx);
                } else {
                    prefill_order.push(seq_idx);
                }
            }

            // 按 seq_idx 顺序收集结果（稍后写回 output_token_ids）
            let mut output_tokens = vec![0i32; num_seqs];

            // ═══ Decode 组：一次性 batched forward ═══
            if !decode_order.is_empty() {
                if let Err(e) = self.run_batch_decode(
                    &decode_order,
                    &q_start_loc,
                    &slot_indices,
                    &positions_host,
                    &mut output_tokens,
                ) {
                    tracing::error!("batched decode failed: {:?}", e);
                    panic!("batched decode failed");
                }
            }

            // ═══ Prefill 组：串行 ═══
            // forward_prefill 的 tokens / pos_cpu 都要求是 CPU tensor，直接用上面 D2H 好的 host 数组。
            for &seq_idx in &prefill_order {
                let seq_start = q_start_loc[seq_idx] as usize;
                let seq_end = q_start_loc[seq_idx + 1] as usize;
                let seq_len = seq_end - seq_start;
                let slot = slot_indices[seq_idx] as usize;

                let mut tokens_cpu = crate::tensor::Tensor::new(
                    &[seq_len],
                    crate::base::DataType::I32,
                    DeviceType::Cpu,
                ).expect("alloc tokens_cpu");
                tokens_cpu.as_i32_mut().unwrap().as_slice_mut().unwrap()
                    .copy_from_slice(&token_ids_host[seq_start..seq_end]);

                let mut pos_cpu = crate::tensor::Tensor::new(
                    &[1],
                    crate::base::DataType::I32,
                    DeviceType::Cpu,
                ).expect("alloc pos_cpu");
                pos_cpu.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] =
                    positions_host[seq_start];

                let tok = self
                    .model
                    .forward_prefill(&mut self.states[slot], &tokens_cpu, &pos_cpu, seq_len)
                    .expect("forward_prefill failed");
                output_tokens[seq_idx] = tok;
            }

            let elapsed_us = start.elapsed().as_micros() as u64;
            // 写 output tokens 到共享 buffer（CPU memcpy）
            self.shared
                .write_input_i32(&self.shared.output_token_ids, &output_tokens, num_seqs)
                .expect("write output_token_ids");
            tracing::debug!(
                "Runner step done in {}us, {} tokens out ({}d/{}p)",
                elapsed_us,
                output_tokens.len(),
                decode_order.len(),
                prefill_order.len(),
            );

            // 信号: 先释放 input, 再标记 output ready
            self.shared.input_meta.ready.store(0, Release);
            self.shared.output_meta.ready.store(num_seqs as u32, Release);
        }
    }

    /// 将本步所有 decode seq 聚合成一次 `forward_batch_decode`。
    ///
    /// 用 `slice::get_disjoint_mut` 从 `self.states` 拿到 B 个独立的 `&mut InferenceState`
    /// （零分配、零拷贝），组成一个 `Vec<&mut InferenceState>` 传给 batched forward。
    /// InferenceState 持有的 Tensor / KV cache buffer 地址完全稳定，不依赖 state 在 Vec 中的位置。
    fn run_batch_decode(
        &mut self,
        decode_order: &[usize],
        q_start_loc: &[i32],
        slot_indices: &[i32],
        positions_host: &[i32],
        output_tokens: &mut [i32],
    ) -> Result<()> {
        let b = decode_order.len();

        // 收集 slot 和 positions（每 seq 取其 start 处的 position 值）
        let mut slots: Vec<usize> = Vec::with_capacity(b);
        for &seq_idx in decode_order {
            slots.push(slot_indices[seq_idx] as usize);
        }
        let positions: Vec<i32> = decode_order
            .iter()
            .map(|&seq_idx| positions_host[q_start_loc[seq_idx] as usize])
            .collect();

        // 检查 batch 组合是否变化；变化时重新填 KV 指针 + 让 llama3 内部按新 batch_size 重新 capture graph。
        #[cfg(feature = "cuda")]
        {
            let mut slots_sorted = slots.clone();
            slots_sorted.sort_unstable();
            if slots_sorted != self.last_decode_slots {
                self.workspace.cache_ptrs_filled = false;
                self.last_decode_slots = slots_sorted;
            }
        }

        // 从 self.states 里取 B 个互不相同的 &mut InferenceState。
        // 标准库的 get_disjoint_mut 只支持编译期已知 N，动态 B 需自己做。
        // 安全性：slots 内部去重由上层 scheduler 保证（每个 seq 的 slot 唯一），
        //         且我们在这段里没有任何其他对 self.states 的并发访问。
        #[cfg(debug_assertions)]
        {
            let mut dup_check = slots.clone();
            dup_check.sort_unstable();
            dup_check.dedup();
            assert_eq!(dup_check.len(), slots.len(), "duplicate slot in batch decode");
        }
        let mut refs_vec: Vec<&mut InferenceState> = Vec::with_capacity(b);
        {
            let base: *mut InferenceState = self.states.as_mut_ptr();
            for &slot in &slots {
                assert!(slot < self.states.len(), "slot {} out of range", slot);
                // SAFETY: slots 互不重复（上方 debug_assert），base..base+len 的元素
                // 各自从属一个不可重叠的 &mut，这里只持有方法本地作用域，不会泄漏。
                let r: &mut InferenceState = unsafe { &mut *base.add(slot) };
                refs_vec.push(r);
            }
        }

        #[cfg(feature = "cuda")]
        let cuda_cfg: Option<&crate::OpConfig> = Some(&self.batch_cuda_cfg);
        #[cfg(not(feature = "cuda"))]
        let cuda_cfg = None;

        let result = self.model.forward_batch_decode(
            refs_vec.as_mut_slice(),
            &mut self.workspace,
            &positions,
            cuda_cfg,
        )?;
        debug_assert_eq!(result.len(), b);

        // 写结果到 output_tokens（按原 seq_idx 顺序）
        for (i, &seq_idx) in decode_order.iter().enumerate() {
            output_tokens[seq_idx] = result[i];
        }
        Ok(())
    }
}
