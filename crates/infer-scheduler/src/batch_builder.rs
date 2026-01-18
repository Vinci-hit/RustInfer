//! Batch Builder - 构建 Worker 输入
//!
//! 将 BatchPlan 和 GlobalState 转换为 Worker 需要的 ForwardParams

use crate::policy::BatchPlan;
use crate::state::GlobalState;
use infer_protocol::{ForwardParams, SamplingParams};
use anyhow::Result;

/// Batch 构建器
pub struct BatchBuilder<'a> {
    plan: &'a BatchPlan,
    state: &'a GlobalState,
    block_size: usize,
}

impl<'a> BatchBuilder<'a> {
    /// 创建新的 BatchBuilder
    pub fn new(plan: &'a BatchPlan, state: &'a GlobalState, block_size: usize) -> Self {
        Self {
            plan,
            state,
            block_size,
        }
    }

    /// 构建 ForwardParams
    ///
    /// 将 BatchPlan 中的 prefills 和 decodes 打包成 Worker 可以处理的格式
    pub fn build(&self) -> Result<ForwardParams> {
        let mut request_ids = Vec::new();
        let mut token_ids = Vec::new();
        let mut position_ids = Vec::new();
        let mut kv_cache_block_ids = Vec::new();

        // 处理 Prefill 请求
        for req_id in &self.plan.prefills {
            if let Some(seq) = self.state.get_sequence(req_id) {
                request_ids.push(req_id.clone());

                // Prefill: 输入所有 input tokens
                token_ids.push(seq.input_tokens.clone());

                // Position IDs: 0, 1, 2, ..., len-1
                let positions: Vec<i32> = (0..seq.input_tokens.len() as i32).collect();
                position_ids.push(positions);

                // KV Cache Block IDs
                let blocks: Vec<u32> = seq.allocated_blocks.iter().map(|&id| id).collect();
                kv_cache_block_ids.push(blocks);
            }
        }

        // 处理 Decode 请求
        for req_id in &self.plan.decodes {
            if let Some(seq) = self.state.get_sequence(req_id) {
                request_ids.push(req_id.clone());

                // Decode: 只输入最后一个生成的 token
                if let Some(&last_token) = seq.generated_tokens.last() {
                    token_ids.push(vec![last_token]);
                } else {
                    // 第一次 decode，输入 input 的最后一个 token
                    if let Some(&last_input) = seq.input_tokens.last() {
                        token_ids.push(vec![last_input]);
                    } else {
                        continue; // 跳过无效请求
                    }
                }

                // Position ID: 当前位置（总 token 数 - 1）
                let position = (seq.total_tokens() - 1) as i32;
                position_ids.push(vec![position]);

                // KV Cache Block IDs
                let blocks: Vec<u32> = seq.allocated_blocks.iter().map(|&id| id).collect();
                kv_cache_block_ids.push(blocks);
            }
        }

        // 是否为 Prefill 阶段（如果 batch 中有 prefill 请求）
        let is_prefill = !self.plan.prefills.is_empty();

        // 默认 SamplingParams（可以从 Sequence 中获取）
        let sampling_params = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            seed: None,
        };

        Ok(ForwardParams {
            request_ids,
            token_ids,
            position_ids,
            kv_cache_block_ids,
            is_prefill,
            sampling_params,
            return_logits: false,
        })
    }

    /// 计算 slot mapping（用于 FlashAttention）
    ///
    /// 返回 [batch_size, max_seq_len] 的 slot IDs
    pub fn build_slot_mapping(&self) -> Vec<Vec<u32>> {
        let mut slot_mapping = Vec::new();

        // Prefill requests
        for req_id in &self.plan.prefills {
            if let Some(seq) = self.state.get_sequence(req_id) {
                let mut slots = Vec::new();
                let total_tokens = seq.input_tokens.len();

                for token_idx in 0..total_tokens {
                    let block_idx = token_idx / self.block_size;
                    let slot_offset = token_idx % self.block_size;

                    if block_idx < seq.allocated_blocks.len() {
                        let physical_block_id = seq.allocated_blocks[block_idx];
                        let slot_id = physical_block_id * self.block_size as u32 + slot_offset as u32;
                        slots.push(slot_id);
                    }
                }

                slot_mapping.push(slots);
            }
        }

        // Decode requests
        for req_id in &self.plan.decodes {
            if let Some(seq) = self.state.get_sequence(req_id) {
                let total_tokens = seq.total_tokens();
                let token_idx = total_tokens - 1; // 最后一个 token 的位置

                let block_idx = token_idx / self.block_size;
                let slot_offset = token_idx % self.block_size;

                if block_idx < seq.allocated_blocks.len() {
                    let physical_block_id = seq.allocated_blocks[block_idx];
                    let slot_id = physical_block_id * self.block_size as u32 + slot_offset as u32;
                    slot_mapping.push(vec![slot_id]);
                }
            }
        }

        slot_mapping
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::BatchPlan;
    use crate::state::GlobalState;
    use infer_protocol::{InferRequest, SamplingParams, StoppingCriteria};
    use tokio::sync::mpsc;

    fn create_test_request(id: &str, tokens: Vec<u32>) -> InferRequest {
        InferRequest {
            request_id: id.to_string(),
            input_token_ids: tokens,
            sampling_params: SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                top_k: 50,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                seed: None,
            },
            stopping_criteria: StoppingCriteria {
                stop_token_ids: vec![2],
                max_new_tokens: 100,
                ignore_eos: false,
            },
            adapter_id: None,
        }
    }

    #[test]
    fn test_batch_builder_prefill() {
        let mut state = GlobalState::new(16, false);
        let req = create_test_request("req1", vec![1, 2, 3, 4, 5]);
        let (tx, _rx) = mpsc::unbounded_channel();
        let (finish_tx, _finish_rx) = mpsc::unbounded_channel();

        state.add_request(req, 0, tx, finish_tx);

        // 模拟分配显存
        if let Some(seq) = state.get_mut_sequence("req1") {
            seq.allocated_blocks = vec![0]; // 分配 block 0
        }

        let mut plan = BatchPlan::new();
        plan.add_prefill("req1".to_string());

        let builder = BatchBuilder::new(&plan, &state, 16);
        let forward_params = builder.build().unwrap();

        assert_eq!(forward_params.request_ids.len(), 1);
        assert_eq!(forward_params.request_ids[0], "req1");
        assert_eq!(forward_params.token_ids[0], vec![1, 2, 3, 4, 5]);
        assert_eq!(forward_params.position_ids[0], vec![0, 1, 2, 3, 4]);
        assert_eq!(forward_params.kv_cache_block_ids[0], vec![0]);
        assert!(forward_params.is_prefill);
    }

    #[test]
    fn test_batch_builder_decode() {
        let mut state = GlobalState::new(16, false);
        let req = create_test_request("req1", vec![1, 2, 3]);
        let (tx, _rx) = mpsc::unbounded_channel();
        let (finish_tx, _finish_rx) = mpsc::unbounded_channel();

        state.add_request(req, 0, tx, finish_tx);

        // 模拟已经在 running 并生成了一些 tokens
        if let Some(seq) = state.get_mut_sequence("req1") {
            seq.allocated_blocks = vec![0];
            seq.generated_tokens = vec![10, 11];
            seq.state = crate::policy::RequestState::Running;
        }

        let mut plan = BatchPlan::new();
        plan.add_decode("req1".to_string());

        let builder = BatchBuilder::new(&plan, &state, 16);
        let forward_params = builder.build().unwrap();

        assert_eq!(forward_params.request_ids.len(), 1);
        assert_eq!(forward_params.request_ids[0], "req1");
        assert_eq!(forward_params.token_ids[0], vec![11]); // 最后一个生成的 token
        assert_eq!(forward_params.position_ids[0], vec![4]); // total_tokens - 1 = 3 + 2 - 1 = 4
        assert!(!forward_params.is_prefill);
    }

    #[test]
    fn test_batch_builder_mixed() {
        let mut state = GlobalState::new(16, false);

        // Prefill request
        let req1 = create_test_request("req1", vec![1, 2, 3]);
        let (tx1, _rx1) = mpsc::unbounded_channel();
        let (finish_tx1, _finish_rx1) = mpsc::unbounded_channel();
        state.add_request(req1, 0, tx1, finish_tx1);
        if let Some(seq) = state.get_mut_sequence("req1") {
            seq.allocated_blocks = vec![0];
        }

        // Decode request
        let req2 = create_test_request("req2", vec![4, 5, 6]);
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let (finish_tx2, _finish_rx2) = mpsc::unbounded_channel();
        state.add_request(req2, 0, tx2, finish_tx2);
        if let Some(seq) = state.get_mut_sequence("req2") {
            seq.allocated_blocks = vec![1];
            seq.generated_tokens = vec![20];
            seq.state = crate::policy::RequestState::Running;
        }

        let mut plan = BatchPlan::new();
        plan.add_prefill("req1".to_string());
        plan.add_decode("req2".to_string());

        let builder = BatchBuilder::new(&plan, &state, 16);
        let forward_params = builder.build().unwrap();

        assert_eq!(forward_params.request_ids.len(), 2);
        assert!(forward_params.is_prefill); // 因为有 prefill 请求
    }
}
