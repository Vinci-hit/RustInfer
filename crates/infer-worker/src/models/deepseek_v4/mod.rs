//! Single-sequence, host-assisted V4 architecture reference for small fixtures.
//!
//! GEMMs use the existing CPU/CUDA backend. Scalar operators and cache state
//! live on the host, with explicit synchronization; this is intentionally an
//! offline diagnostic, not a serving decoder or a performance implementation.
//! Supports unquantized Transformers 5.12.0 fixtures, plain short-context RoPE,
//! SWA/CSA/HCA, hash/top-k/shared MoE, mHC and incremental compression caches.
//! Full checkpoints, quantized KV/weights, YaRN, MTP and TP are rejected.

mod attention;
mod config;
mod hyper;
mod math;
mod moe;

pub use config::TinyConfig;

use attention::{Attention, AttentionState};
use hyper::Hyper;
use math::*;
use moe::Moe;

use crate::domain::{
    dtype::Dtype,
    exec::ExecScope,
    ports::{OpError, OpResult, backend::LlmBackend},
};
use crate::models::loader::WeightLoader;

struct Block<T: Dtype, D: LlmBackend> {
    attention: Attention<T, D>,
    moe: Moe<T, D>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_hc: Hyper,
    ffn_hc: Hyper,
}

/// Cache is explicit so independent requests cannot share compressor state.
pub struct TinyCache {
    config: TinyConfig,
    layers: Vec<AttentionState>,
    position: usize,
    poisoned: bool,
}

impl TinyCache {
    pub fn position(&self) -> usize {
        self.position
    }

    /// (attention compressed entries, indexer compressed entries), by layer.
    pub fn compressed_entries(&self) -> Vec<(usize, usize)> {
        self.layers
            .iter()
            .map(|s| (s.compressed.entries.len(), s.index.entries.len()))
            .collect()
    }
}

/// Row-major host diagnostics. Layer rows contain all mHC residual streams.
pub struct TinyOutput {
    pub logits: Vec<f32>,
    pub layers: Vec<Vec<f32>>,
}

pub struct TinyModel<T: Dtype, D: LlmBackend> {
    config: TinyConfig,
    embedding: Vec<f32>,
    blocks: Vec<Block<T, D>>,
    head_hc: Hyper,
    norm: Vec<f32>,
    head: Matrix<T, D>,
}

impl<T: Dtype, D: LlmBackend> TinyModel<T, D> {
    pub fn load(loader: &WeightLoader<'_>, cfg: TinyConfig, device: &D) -> OpResult<Self> {
        cfg.validate()?;
        if loader.tensor_parallel().size != 1 {
            return Err(OpError::Shape("V4 tiny supports TP=1 only".into()));
        }
        let dtype = if T::DATA_TYPE == <f32 as Dtype>::DATA_TYPE {
            "float32"
        } else if T::DATA_TYPE == <half::bf16 as Dtype>::DATA_TYPE {
            "bfloat16"
        } else {
            "unsupported"
        };
        if cfg.dtype != dtype {
            return Err(OpError::Shape(format!(
                "V4 fixture dtype {} does not match execution dtype {dtype}",
                cfg.dtype
            )));
        }
        let embedding = values(
            loader,
            "model.embed_tokens.weight",
            &[cfg.vocab_size, cfg.hidden_size],
        )?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            blocks.push(Block {
                attention: Attention::load(loader, &format!("{p}.self_attn"), i, &cfg, device)?,
                moe: Moe::load(loader, &format!("{p}.mlp"), i, &cfg, device)?,
                attn_norm: values(
                    loader,
                    &format!("{p}.input_layernorm.weight"),
                    &[cfg.hidden_size],
                )?,
                ffn_norm: values(
                    loader,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[cfg.hidden_size],
                )?,
                attn_hc: Hyper::load(loader, &format!("{p}.attn_hc"), &cfg, false)?,
                ffn_hc: Hyper::load(loader, &format!("{p}.ffn_hc"), &cfg, false)?,
            });
        }
        Ok(Self {
            embedding,
            blocks,
            head_hc: Hyper::load(loader, "model.hc_head", &cfg, true)?,
            norm: values(loader, "model.norm.weight", &[cfg.hidden_size])?,
            head: Matrix::load(
                loader,
                "lm_head.weight",
                cfg.vocab_size,
                cfg.hidden_size,
                device,
            )?,
            config: cfg,
        })
    }

    pub fn config(&self) -> &TinyConfig {
        &self.config
    }

    pub fn new_cache(&self) -> TinyCache {
        TinyCache {
            config: self.config.clone(),
            layers: (0..self.blocks.len())
                .map(|_| AttentionState::default())
                .collect(),
            position: 0,
            poisoned: false,
        }
    }

    /// Process any nonempty contiguous chunk of one request (prefill or decode).
    pub fn forward(
        &self,
        ids: &[usize],
        cache: &mut TinyCache,
        scope: &D::Scope,
    ) -> OpResult<TinyOutput> {
        if cache.config != self.config || cache.poisoned {
            return Err(OpError::Shape(
                "V4 cache has incompatible geometry or a failed forward; create a new cache".into(),
            ));
        }
        if ids.is_empty()
            || ids.iter().any(|&id| id >= self.config.vocab_size)
            || ids.len()
                > self
                    .config
                    .max_position_embeddings
                    .saturating_sub(cache.position)
        {
            return Err(OpError::Shape(
                "V4 input is empty, out of vocabulary, or exceeds the context limit".into(),
            ));
        }
        if scope.topology().world_size() != 1 {
            return Err(OpError::Shape(
                "V4 tiny requires a single-rank scope".into(),
            ));
        }
        let _active = scope.enter();
        cache.poisoned = true;
        let cfg = &self.config;
        let mut hidden = Vec::with_capacity(ids.len() * cfg.hc_mult * cfg.hidden_size);
        for &id in ids {
            for _ in 0..cfg.hc_mult {
                hidden.extend(
                    self.embedding[id * cfg.hidden_size..(id + 1) * cfg.hidden_size]
                        .iter()
                        .map(|&v| round::<T>(v)),
                );
            }
        }
        let mut layers = Vec::with_capacity(self.blocks.len());
        for (block, state) in self.blocks.iter().zip(&mut cache.layers) {
            let (collapsed, post, comb) = block.attn_hc.pre::<T>(&hidden, cfg);
            let x = norm::<T>(&collapsed, &block.attn_norm, cfg.rms_norm_eps);
            let x = block.attention.run(&x, cache.position, state, cfg, scope)?;
            hidden = block.attn_hc.post::<T>(&hidden, &x, &post, &comb);
            let (collapsed, post, comb) = block.ffn_hc.pre::<T>(&hidden, cfg);
            let x = norm::<T>(&collapsed, &block.ffn_norm, cfg.rms_norm_eps);
            let x = block.moe.run(&x, ids, cfg, scope)?;
            hidden = block.ffn_hc.post::<T>(&hidden, &x, &post, &comb);
            layers.push(hidden.clone());
        }
        let (collapsed, _, _) = self.head_hc.pre::<T>(&hidden, cfg);
        let logits = self
            .head
            .run(&norm::<T>(&collapsed, &self.norm, cfg.rms_norm_eps), scope)?;
        if logits
            .iter()
            .chain(layers.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(OpError::Kernel(
                "V4 reference produced non-finite output".into(),
            ));
        }
        cache.position += ids.len();
        cache.poisoned = false;
        Ok(TinyOutput { logits, layers })
    }
}
