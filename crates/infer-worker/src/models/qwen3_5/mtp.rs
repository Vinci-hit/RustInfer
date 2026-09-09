//! Qwen3.5 MTP checkpoint convention. Loaded explicitly, never by base build.
use super::*;
use crate::components::Embed;
use crate::components::mtp::MtpHead;
use crate::domain::model::DecoderModel;

#[derive(Debug, serde::Deserialize)]
pub struct MtpConfig {
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
}

impl<T: Dtype, D: OpBackend + LlmBackend> Qwen3_5Model<T, D> {
    /// Share immutable embedding and LM head storage with the target.
    /// The returned head has its own workspace and requires an independent KV pool.
    pub fn load_mtp(
        &self,
        loader: &WeightLoader<'_>,
        cfg: &LoadConfig,
        mtp: &MtpConfig,
    ) -> OpResult<MtpHead<T, D, Decoder<T, D>>> {
        let device = self.decoder.embed.table.device();
        if mtp.mtp_num_hidden_layers != 1
            || mtp.mtp_use_dedicated_embeddings
            || loader.tensor_parallel().size != 1
            || cfg.num_experts != 0
            || cfg.mlp_quant.is_some()
            || cfg.fp8_block.is_some()
        {
            return Err(OpError::unsupported(
                "Qwen3.5 MTP",
                "requires one dense layer, shared embeddings, TP1 and unquantized weights",
            ));
        }
        let dims = self.dims();
        if cfg.dim != dims.dim
            || cfg.head_num != dims.head_num
            || cfg.kv_head_num != dims.kv_head_num
            || cfg.head_dim != dims.head_dim
            || cfg.vocab_size != dims.vocab_size
            || cfg.intermediate_size != dims.intermediate_size
            || cfg.rotary_dim != self.rotary_dim
            || cfg.rope_theta != self.rope_theta
            || cfg.rms_norm_eps != self.decoder.norm.eps
        {
            return Err(OpError::Shape(
                "MTP configuration does not match target".into(),
            ));
        }
        let norm = |name: &str| load_norm(loader, name, cfg.dim, cfg.rms_norm_eps, device);
        let embedding_norm = norm("mtp.pre_fc_norm_embedding.weight")?;
        let hidden_norm = norm("mtp.pre_fc_norm_hidden.weight")?;
        let fc = load_linear(loader, "mtp.fc.weight", cfg.dim, 2 * cfg.dim, device)?;
        let (sin, cos) = compute_rope_cache(
            cfg.seq_len,
            cfg.rotary_dim,
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
            device,
        )?;
        let layer = "mtp.layers.0";
        let block = DecoderBlock {
            attention: Attention::Full(load_full_attention(
                loader,
                cfg,
                layer,
                norm("mtp.layers.0.input_layernorm.weight")?,
                &sin,
                &cos,
                device,
            )?),
            ffn: load_dense_ffn(loader, cfg, layer, device)?,
        };
        let embedding = &self.decoder.embed;
        let proj = &self.decoder.lm_head.proj;
        let weight = proj
            .weight
            .as_dense()
            .ok_or_else(|| OpError::unsupported("Qwen3.5 MTP", "quantized shared LM head"))?;
        let decoder = Decoder::new(
            Embed::new(embedding.table.clone()).with_parallelism(embedding.parallelism()),
            vec![block],
            norm("mtp.norm.weight")?,
            LmHead {
                proj: Linear::new(weight.clone(), proj.bias.clone())
                    .with_parallelism(proj.parallelism()),
            },
            ModelDims {
                num_layers: 1,
                ..dims
            },
        )?;
        Ok(MtpHead::new(embedding_norm, hidden_norm, fc, decoder))
    }
}

fn load_full_attention<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer: &str,
    input_layernorm: RmsNorm<T, D>,
    sin: &Tensor<T, D>,
    cos: &Tensor<T, D>,
    device: &D,
) -> OpResult<FullAttention<T, D>> {
    let q_dim = cfg.head_num * cfg.head_dim;
    let kv_dim = cfg.kv_head_num * cfg.head_dim;
    Ok(FullAttention {
        input_layernorm,
        qkv_proj: loader.load_fused_qkv_with_fp8(
            layer,
            q_dim * (1 + usize::from(cfg.attn_output_gate)),
            kv_dim,
            cfg.dim,
            None,
            device,
        )?,
        o_proj: load_linear(
            loader,
            &format!("{layer}.self_attn.o_proj.weight"),
            cfg.dim,
            q_dim,
            device,
        )?,
        q_norm: Some(load_norm(
            loader,
            &format!("{layer}.self_attn.q_norm.weight"),
            cfg.head_dim,
            cfg.rms_norm_eps,
            device,
        )?),
        k_norm: Some(load_norm(
            loader,
            &format!("{layer}.self_attn.k_norm.weight"),
            cfg.head_dim,
            cfg.rms_norm_eps,
            device,
        )?),
        sin: sin.clone(),
        cos: cos.clone(),
        head_num: cfg.head_num,
        kv_head_num: cfg.kv_head_num,
        head_dim: cfg.head_dim,
        rotary_dim: cfg.rotary_dim,
        attn_output_gate: cfg.attn_output_gate,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        scratch: None,
    })
}
fn load_dense_ffn<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer: &str,
    device: &D,
) -> OpResult<DenseFfn<T, D>> {
    Ok(DenseFfn {
        post_attention_layernorm: load_norm(
            loader,
            &format!("{layer}.post_attention_layernorm.weight"),
            cfg.dim,
            cfg.rms_norm_eps,
            device,
        )?,
        gate_up_proj: loader.load_fused_gate_up_with_fp8(
            layer,
            cfg.intermediate_size,
            cfg.dim,
            None,
            device,
        )?,
        down_proj: load_linear(
            loader,
            &format!("{layer}.mlp.down_proj.weight"),
            cfg.dim,
            cfg.intermediate_size,
            device,
        )?,
        scratch: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::speculative::prefill::MtpPrefill;
    use crate::components::mtp::MtpInput;
    use crate::domain::cache::ModelCacheView;
    use crate::domain::exec::StepCtx;
    use crate::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
    use crate::domain::plan::{BatchKind, BatchPlan};
    use crate::domain::tensor::Tensor;

    #[test]
    fn eager_session_handles_chunked_prefill_budgets_eos_and_reuse() {
        use crate::application::speculative::{MtpLimits, MtpSession};
        use crate::domain::exec::HostScope;
        use crate::infrastructure::cpu::Cpu;
        let mut cfg = super::super::tests::config();
        cfg.seq_len = 32;
        let reader = super::super::tests::checkpoint(true);
        let loader = WeightLoader::new(&reader);
        for context in [6, 32] {
            for (k, max_step) in [(0, 1), (1, 2), (3, 4)] {
                for (budget, eos) in [(1, vec![]), (8, vec![]), (8, vec![0])] {
                    let model = build::<f32, Cpu>(&loader, &cfg, &Cpu).unwrap();
                    let head = model
                        .load_mtp(
                            &loader,
                            &cfg,
                            &MtpConfig {
                                mtp_num_hidden_layers: 1,
                                mtp_use_dedicated_embeddings: false,
                            },
                        )
                        .unwrap();
                    let mut session = MtpSession::new(
                        model,
                        head,
                        HostScope::new(Cpu),
                        MtpLimits {
                            max_context: context,
                            max_step_tokens: max_step,
                            max_output_tokens: budget,
                            draft_tokens: k,
                            eos_ids: eos.clone(),
                        },
                    )
                    .unwrap();
                    assert!(session.decode().is_err());
                    assert!(session.prefill(&[-1]).is_err());
                    for prompt in [&[1, 2, 3, 4, 5][..], &[3][..]] {
                        session.reset().unwrap();
                        let mut step = session.prefill(prompt).unwrap();
                        assert!(session.prefill(prompt).is_err());
                        let mut tokens = step.tokens.clone();
                        while !step.finished {
                            step = session.decode().unwrap();
                            assert_eq!(step.accepted, step.proposed);
                            tokens.extend_from_slice(&step.tokens);
                            assert_eq!(session.cached_tokens(), session.draft_cached_tokens() + 1);
                        }
                        assert_eq!(
                            tokens,
                            vec![
                                0;
                                if eos.is_empty() {
                                    (budget as usize).min(context - prompt.len() + 1)
                                } else {
                                    1
                                }
                            ]
                        );
                        assert_eq!(session.cached_tokens(), prompt.len() + tokens.len() - 1);
                        assert!(session.decode().is_err());
                    }
                }
            }
        }
    }
    fn indices<D: crate::domain::ports::backend::LlmBackend>(
        start: usize,
        n: usize,
        blocks: usize,
        device: &D,
    ) -> (BatchPlan, KvIndexTensors<D>) {
        let positions: Vec<i32> = (start as i32..(start + n) as i32).collect();
        let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&[n as i32]);
        let plan = BatchPlan {
            kind: if n == 1 {
                BatchKind::DecodeOnly
            } else {
                BatchKind::Ragged
            },
            num_tokens: n,
            batch: 1,
            q_lens: vec![n as i32],
            kv_lens: vec![(start + n) as i32],
            seq_positions: vec![start as i32],
            rope_positions: positions.clone(),
            max_blocks_per_seq: blocks,
            block_size: 1,
            total_q_tiles: req.len() as i32,
        };
        let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], device).unwrap();
        let idx = KvIndexTensors {
            block_tables: Tensor::from_host_slice(
                &(0..blocks as i32).collect::<Vec<_>>(),
                [1, blocks],
                device,
            )
            .unwrap(),
            cu_q_lens: ints(&cu),
            kv_lens: ints(&plan.kv_lens),
            seq_positions: ints(&[start as i32]),
            seq_lens_step: ints(&[n as i32]),
            rope_positions: ints(&positions),
            block2req: ints(&req),
            block2tile: ints(&tile),
            valid_q_tiles: ints(&[req.len() as i32]),
            valid_suffix_q_tiles: ints(&[req.len() as i32]),
        };
        (plan, idx)
    }
    fn pool<T: crate::domain::dtype::Dtype, D: crate::domain::ports::backend::LlmBackend>(
        layers: usize,
        blocks: usize,
        kv_dim: usize,
        device: &D,
    ) -> PagedKvPool<T, D> {
        PagedKvPool {
            layers: (0..layers)
                .map(|_| PagedKvLayer {
                    k: Tensor::zeros([blocks, 1, kv_dim], device).unwrap(),
                    v: Tensor::zeros([blocks, 1, kv_dim], device).unwrap(),
                })
                .collect(),
            num_blocks: blocks,
            block_size: 1,
            kv_dim,
            quant: KvQuantTier::None,
            seq_kv_len: Default::default(),
        }
    }

    #[test]
    fn loaded_head_matches_chunked_execution_and_rejects_unsupported_configs() {
        use crate::infrastructure::cpu::Cpu;
        let cfg = super::super::tests::config();
        let reader = super::super::tests::checkpoint(true);
        let loader = WeightLoader::new(&reader);
        let model = build::<f32, Cpu>(&loader, &cfg, &Cpu).unwrap();
        let mtp = MtpConfig {
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
        };
        let mut head = model.load_mtp(&loader, &cfg, &mtp).unwrap();
        assert_eq!(head.cache_layout().num_full_layers(), 1);
        assert!(head.cache_layout().linear_dims().is_empty());
        assert!(
            model
                .load_mtp(
                    &loader,
                    &cfg,
                    &MtpConfig {
                        mtp_num_hidden_layers: 2,
                        ..mtp
                    },
                )
                .is_err()
        );
        assert!(
            model
                .load_mtp(
                    &loader,
                    &cfg,
                    &MtpConfig {
                        mtp_use_dedicated_embeddings: true,
                        ..mtp
                    },
                )
                .is_err()
        );
        let missing = super::super::tests::checkpoint(false);
        assert!(
            model
                .load_mtp(&WeightLoader::new(&missing), &cfg, &mtp)
                .is_err()
        );
        let mut bad_cfg = cfg.clone();
        bad_cfg.dim += 1;
        assert!(model.load_mtp(&loader, &bad_cfg, &mtp).is_err());
        head.prepare(5, 1).unwrap();
        let ids = [1, 2, 3, 4, 5];
        let target = Tensor::from_host_slice(
            &(0..40).map(|i| (i as f32 - 20.) / 11.).collect::<Vec<_>>(),
            [5, 8],
            &Cpu,
        )
        .unwrap();
        let scope = crate::domain::exec::HostScope::new(Cpu);
        let mut expected: Option<Vec<f32>> = None;
        for cuts in [vec![5], vec![1, 2, 3, 4, 5], vec![2, 4, 5]] {
            let mut kv = pool(1, 8, 4, &Cpu);
            let mut align = MtpPrefill::default();
            let mut actual = Vec::new();
            let mut start = 0;
            for end in cuts {
                let positions: Vec<_> = (start as i32..end as i32).collect();
                let chunk = align
                    .prepare(
                        &ids[start..end],
                        &positions,
                        &target.narrow(0, start, end - start).unwrap(),
                    )
                    .unwrap();
                let n = chunk.next_token_ids.len();
                if n > 0 {
                    let (plan, index) = indices(chunk.positions[0] as usize, n, 8, &Cpu);
                    let ctx = StepCtx::new(&scope, &plan);
                    let mut output = Tensor::zeros([n, 8], &Cpu).unwrap();
                    let tokens = Tensor::from_host_slice(&chunk.next_token_ids, [n], &Cpu).unwrap();
                    let before = kv.layers[0].k.to_host_vec().unwrap();
                    assert!(
                        head.forward_hidden_into(
                            MtpInput {
                                next_token_ids: &tokens,
                                target_hidden: &chunk.target_hidden
                            },
                            &mut ModelCacheView::full(&mut kv, &index),
                            &mut Tensor::zeros([n, 9], &Cpu).unwrap(),
                            &ctx,
                        )
                        .is_err()
                    );
                    assert_eq!(kv.layers[0].k.to_host_vec().unwrap(), before);
                    let mut cache = ModelCacheView::full(&mut kv, &index);
                    head.forward_hidden_into(
                        MtpInput {
                            next_token_ids: &tokens,
                            target_hidden: &chunk.target_hidden,
                        },
                        &mut cache,
                        &mut output,
                        &ctx,
                    )
                    .unwrap();
                    actual.extend(output.to_host_vec().unwrap());
                }
                chunk.commit();
                start = end;
            }
            if let Some(ref expected) = expected {
                assert_eq!(actual.len(), expected.len());
                for (&a, &b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() < 1e-5, "{a} != {b}");
                }
            } else {
                expected = Some(actual);
            }
        }
        assert!(expected.unwrap().iter().all(|x| x.is_finite()));
    }
}
