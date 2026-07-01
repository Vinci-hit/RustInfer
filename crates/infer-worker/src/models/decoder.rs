//! Assembled decoder model — a stage list of reusable components driven over
//! the sliceable `embed` / `decode_layers(range)` / `finalize` contract.
//!
//! Llama3 and Qwen3 share this exact structure; the only difference is whether
//! each block's `Attention` carries Qwen3-style Q/K norms (set at load time).

use crate::components::decoder_block::DecoderBlock;
use crate::components::embed::Embed;
use crate::components::ffn_dense::DenseFfn;
use crate::components::lm_head::LmHead;
use crate::components::norm::RmsNorm;
use std::rc::Rc;

use crate::domain::component::{Component, Hidden, LayerRange, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::kv::KvView;
use crate::domain::model::{DecoderModel, Logits, ModelDims, SampleRows};
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

const STAGES: [StageKind; 3] = [StageKind::Embed, StageKind::DecoderBlock, StageKind::LmHead];

/// A dense autoregressive decoder assembled from components. Generic over the
/// backend `D`; the same type backs both Llama3 and Qwen3 (Qwen3 populates the
/// per-block `Attention` Q/K norms).
pub struct Decoder<T: Dtype, D: LlmBackend> {
    pub embed: Embed<T, D>,
    pub blocks: Vec<DecoderBlock<T, D, DenseFfn<T, D>>>,
    pub norm: RmsNorm<T, D>,
    pub lm_head: LmHead<T, D>,
    pub dims: ModelDims,
    /// Shared per-forward scratch (installed by `Runtime::new`); used by
    /// `finalize` for the norm + lm_head output. `None` → allocate (pooled).
    pub scratch: Option<Rc<ForwardScratch<T, D>>>,
}

impl<T: Dtype, D: LlmBackend> DecoderModel<T, D> for Decoder<T, D> {
    fn dims(&self) -> ModelDims {
        self.dims
    }

    fn stages(&self) -> &[StageKind] {
        &STAGES
    }

    fn install_scratch(
        &mut self,
        scratch: std::rc::Rc<crate::domain::forward_scratch::ForwardScratch<T, D>>,
    ) {
        for block in &mut self.blocks {
            block.attention.scratch = Some(scratch.clone());
            block.ffn.scratch = Some(scratch.clone());
        }
        self.scratch = Some(scratch);
    }

    fn embed(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.embed.forward(input_ids, hidden, ctx)
    }

    fn decode_layers(
        &self,
        range: LayerRange,
        hidden: &mut Hidden<T, D>,
        kv: &mut KvView<'_, T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        for layer_idx in range.start..range.end {
            let local = layer_idx - range.start;
            let mut layer_view = kv.single_layer(local);
            self.blocks[layer_idx].run(hidden, Some(&mut layer_view), ctx)?;
        }
        // Flush the last sublayer's deferred residual delta so `stream` holds the
        // true residual for `finalize`. This is the one residual boundary not
        // fused with a following pre-norm (all inter-sublayer boundaries are).
        if let Some(delta) = hidden.pending.take() {
            D::add_inplace(ctx.scope(), &mut hidden.stream, &delta)?;
        }
        Ok(())
    }

    fn finalize(
        &self,
        hidden: &Hidden<T, D>,
        rows: SampleRows<'_>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<Logits<T, D>> {
        let num_tokens = hidden.num_tokens();
        let dev = hidden.stream.device().clone();
        let dim = self.dims.dim;

        // Row selection. The greedy first-token path passes `LastPerSeq`: only
        // the last token of each sequence needs logits. Projecting just those
        // `batch` rows instead of all `num_tokens` (a) shrinks the lm_head GEMM
        // from [num_tokens, vocab] to [batch, vocab], and (b) — the bigger win —
        // keeps that GEMM at M=batch (a warm decode shape) instead of
        // M=num_tokens, which is cold for every distinct prompt length and
        // otherwise paid a ~37ms cuBLASLt heuristic for the 151936-wide vocab
        // projection on the TTFT critical path. Decode (every q_len==1 →
        // num_tokens==batch) gathers the identity, so we skip the gather there.
        let gather_idx: Option<Vec<i32>> = match rows {
            SampleRows::All => None,
            SampleRows::LastPerSeq => {
                let plan = ctx.plan();
                let mut idx: Vec<i32> = Vec::with_capacity(plan.q_lens.len());
                let mut off = 0i32;
                for &q in &plan.q_lens {
                    off += q.max(1);
                    idx.push(off - 1);
                }
                // All-singleton (decode) → identity gather; project directly.
                if idx.len() >= num_tokens {
                    None
                } else {
                    Some(idx)
                }
            }
            SampleRows::Explicit(sel) => {
                if sel.len() >= num_tokens {
                    None
                } else {
                    Some(sel.to_vec())
                }
            }
        };

        // Materialize the selected rows (if any) into a contiguous buffer.
        let selected: Option<Tensor<T, D>> = match &gather_idx {
            None => None,
            Some(idx) if idx.len() == 1 => {
                // Single-sequence prefill: the last row is contiguous, so a
                // narrow avoids both the gather kernel and the index upload.
                Some(hidden.stream.narrow(0, idx[0] as usize, 1)?)
            }
            Some(idx) => {
                // Scattered last rows (burst prefill): gather via the embedding
                // row-select kernel (table = residual stream, ids = last rows).
                let idx_dev = Tensor::from_host_slice(idx, Shape::from_slice(&[idx.len()]), &dev)?;
                let mut gathered =
                    D::alloc_tensor::<T>(Shape::from_slice(&[idx.len(), dim]), &dev)?;
                D::embedding(ctx.scope(), &hidden.stream, &idx_dev, &mut gathered)?;
                Some(gathered)
            }
        };

        let src: &Tensor<T, D> = selected.as_ref().unwrap_or(&hidden.stream);
        let n_rows = src.shape().as_slice()[0];

        // Reuse the address-stable scratch (norm buffer + preallocated logits)
        // when installed; otherwise allocate (pooled). Keeps `finalize`
        // allocation-free and keeps the large [rows, vocab] logits off the
        // recycling pool (where ragged lengths would grow it unbounded).
        let scratch = self.scratch.as_deref().filter(|s| s.fits(n_rows));
        let mut normed = match scratch {
            Some(s) => s.normed(n_rows),
            None => D::alloc_tensor(Shape::from_slice(&[n_rows, dim]), &dev)?,
        };
        let mut logits = match scratch {
            Some(s) => s.logits(n_rows),
            None => D::alloc_tensor(Shape::from_slice(&[n_rows, self.dims.vocab_size]), &dev)?,
        };
        self.norm.forward(src, &mut normed, ctx)?;
        self.lm_head.forward(&normed, &mut logits, ctx)?;
        Ok(Logits(logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::runtime::Runtime;
    use crate::application::sampler_stack::GreedySampler;
    use crate::components::linear::Linear;
    use crate::domain::plan::{SeqStep, StepRequest, StopCriteria};
    use crate::infrastructure::cpu::Cpu;

    const DIM: usize = 16;
    const HEAD_NUM: usize = 2;
    const HEAD_DIM: usize = 8;
    const INTER: usize = 32;
    const VOCAB: usize = 64;
    const MAX_SEQ: usize = 32;

    fn weight(rows: usize, cols: usize) -> Tensor<f32, Cpu> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        Tensor::from_host_slice(&data, Shape::from_slice(&[rows, cols]), &Cpu).unwrap()
    }

    fn ones(dim: usize) -> Tensor<f32, Cpu> {
        Tensor::from_host_slice(&vec![1.0f32; dim], Shape::from_slice(&[dim]), &Cpu).unwrap()
    }

    fn rms(weight: Tensor<f32, Cpu>) -> RmsNorm<f32, Cpu> {
        RmsNorm { weight, eps: 1e-5 }
    }

    fn lin(rows: usize, cols: usize) -> Linear<f32, Cpu> {
        Linear::new(weight(rows, cols), None)
    }

    /// 1-layer Llama-style decoder with deterministic weights (rebuildable so
    /// the batched and serial runners use byte-identical weights).
    fn tiny_decoder() -> Decoder<f32, Cpu> {
        let q_dim = HEAD_NUM * HEAD_DIM;
        let kv_dim = HEAD_NUM * HEAD_DIM;
        let qkv_dim = q_dim + 2 * kv_dim;
        let sin = Tensor::from_host_slice(
            &vec![0.0f32; MAX_SEQ * HEAD_DIM],
            Shape::from_slice(&[MAX_SEQ, HEAD_DIM]),
            &Cpu,
        )
        .unwrap();
        let cos = Tensor::from_host_slice(
            &vec![1.0f32; MAX_SEQ * HEAD_DIM],
            Shape::from_slice(&[MAX_SEQ, HEAD_DIM]),
            &Cpu,
        )
        .unwrap();
        let block = DecoderBlock {
            attention: Attention {
                input_layernorm: rms(ones(DIM)),
                qkv_proj: lin(qkv_dim, DIM),
                o_proj: lin(DIM, q_dim),
                q_norm: None,
                k_norm: None,
                sin,
                cos,
                head_num: HEAD_NUM,
                kv_head_num: HEAD_NUM,
                head_dim: HEAD_DIM,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                scratch: None,
            },
            ffn: DenseFfn {
                post_attention_layernorm: rms(ones(DIM)),
                gate_up_proj: lin(2 * INTER, DIM),
                down_proj: lin(DIM, INTER),
                scratch: None,
            },
        };
        Decoder {
            embed: Embed {
                table: weight(VOCAB, DIM),
            },
            blocks: vec![block],
            norm: rms(ones(DIM)),
            lm_head: LmHead {
                proj: lin(VOCAB, DIM),
            },
            dims: ModelDims {
                dim: DIM,
                q_dim,
                kv_dim,
                qkv_dim,
                intermediate_size: INTER,
                vocab_size: VOCAB,
                head_num: HEAD_NUM,
                head_dim: HEAD_DIM,
                kv_head_num: HEAD_NUM,
                num_layers: 1,
                num_experts: 0,
                experts_per_tok: 0,
                moe_intermediate_size: 0,
                num_shared_experts: 0,
            },
            scratch: None,
        }
    }

    use crate::components::attention::Attention;

    fn runner(num_blocks: usize, cap_batch: usize) -> Runtime<f32, Cpu, Decoder<f32, Cpu>> {
        Runtime::new(
            tiny_decoder(),
            crate::domain::exec::HostScope::new(Cpu),
            Box::new(GreedySampler),
            num_blocks,
            1,
            8,
            MAX_SEQ,
            32,
            cap_batch,
            Vec::new(),
        )
        .unwrap()
    }

    fn prefill_seq(sid: u64, ids: &[i32], blocks: &[u32]) -> SeqStep {
        SeqStep {
            sequence_id: sid,
            input_ids: ids.to_vec(),
            positions: (0..ids.len() as i32).collect(),
            kv_write_start: 0,
            kv_len_after: ids.len() as i32,
            block_table: blocks.to_vec(),
        }
    }

    fn first_token(runner: &mut Runtime<f32, Cpu, Decoder<f32, Cpu>>, seq: SeqStep) -> i32 {
        let n = seq.input_ids.len();
        let req = StepRequest {
            seqs: vec![seq],
            sampling: Vec::new(),
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: vec![0],
                max_tokens: vec![16],
                ignore_eos: vec![false],
            },
            draft_tokens: Vec::new(),
        };
        let _ = n;
        runner.step(&req).unwrap().tokens[0][0].token_id
    }

    /// Component forward correctness: a ragged 2-sequence prefill must produce
    /// the same first token per sequence as running each sequence on its own.
    /// Exercises embed → attention (paged KV, ragged) → FFN → finalize → greedy.
    #[test]
    fn component_decoder_ragged_batch_matches_serial() {
        let p0: Vec<i32> = vec![1, 2, 3, 4, 5];
        let p1: Vec<i32> = vec![10, 20, 30];

        let mut batch = runner(16, 2);
        let bt0: Vec<u32> = vec![0, 1, 2, 3, 4];
        let bt1: Vec<u32> = vec![8, 9, 10];
        let req = StepRequest {
            seqs: vec![prefill_seq(0, &p0, &bt0), prefill_seq(1, &p1, &bt1)],
            sampling: Vec::new(),
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: vec![0, 0],
                max_tokens: vec![16, 16],
                ignore_eos: vec![false, false],
            },
            draft_tokens: Vec::new(),
        };
        let out = batch.step(&req).unwrap();
        let b0 = out.tokens[0][0].token_id;
        let b1 = out.tokens[1][0].token_id;

        let mut s0 = runner(8, 1);
        let r0 = first_token(&mut s0, prefill_seq(0, &p0, &[0, 1, 2, 3, 4]));
        let mut s1 = runner(8, 1);
        let r1 = first_token(&mut s1, prefill_seq(1, &p1, &[0, 1, 2]));

        assert_eq!(b0, r0, "ragged seq0 first token {} != serial {}", b0, r0);
        assert_eq!(b1, r1, "ragged seq1 first token {} != serial {}", b1, r1);
        assert!((0..VOCAB as i32).contains(&b0));
        assert!((0..VOCAB as i32).contains(&b1));
    }
}
