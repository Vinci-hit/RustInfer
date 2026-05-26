//! ModelRunner — orchestrates full inference: load → forward → sample.
//!
//! This is the application-layer entry point that wires together:
//! - Domain: LlmModel, KvCache, ForwardContext
//! - Infra: Device (Cpu/Cuda), tensor allocation
//! - App: CudaGraphRunner (optional)

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use crate::domain::model::{LlmModel, ForwardContext};
use crate::domain::runtime::KvCache;

/// ModelRunner holds a model + KV caches + manages forward steps.
pub struct ModelRunner<T: Dtype, D: OpBackend, M: LlmModel<T, D>> {
    pub model: M,
    /// Per-sequence KV caches (indexed by slot).
    pub kv_caches: Vec<KvCache<T, D>>,
    /// Device handle.
    pub device: D,
    /// Model geometry.
    pub num_layers: usize,
    pub kv_dim: usize,
    pub max_seq_len: usize,
}

impl<T: Dtype, D: OpBackend, M: LlmModel<T, D>> ModelRunner<T, D, M> {
    /// Create a runner from a loaded model.
    pub fn new(model: M, device: D, max_batch_seqs: usize, max_seq_len: usize) -> OpResult<Self> {
        let num_layers = model.num_layers();
        let kv_dim = model.kv_dim();

        // Pre-allocate KV caches for each batch slot
        let initial_cap = max_seq_len.min(256);
        let mut kv_caches = Vec::with_capacity(max_batch_seqs);
        for _ in 0..max_batch_seqs {
            let mut k_layers = Vec::with_capacity(num_layers);
            let mut v_layers = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                k_layers.push(D::alloc_tensor::<T>(Shape::from_slice(&[initial_cap, kv_dim]), &device)?);
                v_layers.push(D::alloc_tensor::<T>(Shape::from_slice(&[initial_cap, kv_dim]), &device)?);
            }
            kv_caches.push(KvCache {
                k_layers, v_layers,
                capacity: initial_cap,
                max_capacity: max_seq_len,
                kv_dim,
            });
        }

        Ok(Self { model, kv_caches, device, num_layers, kv_dim, max_seq_len })
    }

    /// Run a single forward step.
    ///
    /// - `input_ids`: token IDs for this step [num_tokens]
    /// - `slot`: which KV cache slot to use (batch index)
    /// - `positions`: position of each token in the sequence
    /// - `seq_lens`: how many KV entries each sequence has so far
    ///
    /// Returns logits [num_tokens, vocab_size].
    pub fn step(
        &mut self,
        input_ids: &Tensor<i32, D>,
        slot: usize,
        positions: &[i32],
        seq_lens: &[usize],
    ) -> OpResult<Tensor<T, D>> {
        let kv = &mut self.kv_caches[slot];
        let mut ctx = ForwardContext {
            k_caches: &mut kv.k_layers,
            v_caches: &mut kv.v_layers,
            positions,
            seq_lens,
        };

        self.model.forward(input_ids, &mut ctx)
    }

    /// Convenience: run prefill + N decode steps, return generated token IDs.
    /// Uses D::argmax for sampling — works on both CPU and GPU tensors.
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
    ) -> OpResult<Vec<i32>> {
        let mut generated = Vec::new();
        let num_prompt = prompt_ids.len();

        // Prefill
        let input = crate::models::llama3::alloc_i32::<D>(prompt_ids, &self.device)?;
        let positions: Vec<i32> = (0..num_prompt as i32).collect();
        let seq_lens = vec![num_prompt];

        let logits = self.step(&input, 0, &positions, &seq_lens)?;
        let token = D::argmax(&logits, num_prompt)?;
        generated.push(token);

        // Decode
        for i in 0..max_new_tokens.saturating_sub(1) {
            let pos = (num_prompt + i + 1) as i32;
            let input = crate::models::llama3::alloc_i32::<D>(&[generated.last().copied().unwrap()], &self.device)?;
            let logits = self.step(&input, 0, &[pos], &[num_prompt + i + 1])?;
            let token = D::argmax(&logits, 1)?;
            generated.push(token);
        }

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::cpu::Cpu;
    use crate::models::layers::{Linear, RMSNorm, Embedding};
    use crate::models::llama3::{Llama3Model, Llama3Layer};

    /// Build a tiny 1-layer Llama3 model (random weights) for testing.
    fn tiny_llama3() -> Llama3Model<f32, Cpu> {
        let dim = 16;
        let head_num = 2;
        let kv_head_num = 2;
        let head_dim = 8;
        let intermediate = 32;
        let vocab = 64;
        let q_dim = head_num * head_dim;
        let kv_dim = kv_head_num * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let max_seq = 32;

        use crate::domain::tensor::Tensor;

        // Random-ish weights (all 0.01 to keep values small)
        let make_weight = |rows: usize, cols: usize| -> Tensor<f32, Cpu> {
            let data: Vec<f32> = (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
            Tensor::<f32, Cpu>::from_slice(&data, [rows, cols])
        };
        let make_norm = |dim: usize| -> Tensor<f32, Cpu> {
            Tensor::<f32, Cpu>::from_slice(&vec![1.0f32; dim], [dim])
        };

        let layer = Llama3Layer {
            input_layernorm: RMSNorm::new(make_norm(dim), 1e-5),
            post_attention_layernorm: RMSNorm::new(make_norm(dim), 1e-5),
            qkv_proj: Linear::new(make_weight(qkv_dim, dim), None),
            o_proj: Linear::new(make_weight(dim, q_dim), None),
            gate_proj: Linear::new(make_weight(intermediate, dim), None),
            up_proj: Linear::new(make_weight(intermediate, dim), None),
            down_proj: Linear::new(make_weight(dim, intermediate), None),
        };

        let sin_cache = Tensor::<f32, Cpu>::zeros([max_seq, head_dim]);
        let cos_cache = {
            // cos(0) = 1 for all positions (identity RoPE)
            let data = vec![1.0f32; max_seq * head_dim];
            Tensor::<f32, Cpu>::from_slice(&data, [max_seq, head_dim])
        };

        Llama3Model {
            embed_tokens: Embedding { table: make_weight(vocab, dim) },
            layers: vec![layer],
            norm: RMSNorm::new(make_norm(dim), 1e-5),
            lm_head: Linear::new(make_weight(vocab, dim), None),
            sin_cache,
            cos_cache,
            head_num,
            kv_head_num,
            head_dim,
            dim,
            kv_dim,
            intermediate_size: intermediate,
            vocab_size: vocab,
        }
    }

    #[test]
    fn e2e_cpu_forward_and_argmax() {
        let model = tiny_llama3();
        let mut runner = ModelRunner::new(model, Cpu, 1, 32).unwrap();

        // Prefill with 3 tokens
        let prompt = &[1i32, 5, 10];
        let tokens = runner.generate(prompt, 3).unwrap();

        // Should generate 3 tokens, each a valid vocab index in [0, 64)
        assert_eq!(tokens.len(), 3);
        for &t in &tokens {
            assert!(t >= 0 && t < 64, "token {} out of vocab range", t);
        }
    }
}
