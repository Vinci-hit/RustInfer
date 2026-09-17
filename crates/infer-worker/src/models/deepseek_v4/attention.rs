use std::collections::VecDeque;

use super::{TinyConfig, math::*};
use crate::domain::{
    dtype::Dtype,
    ports::{OpResult, backend::LlmBackend},
};
use crate::models::loader::WeightLoader;

#[derive(Default)]
pub(super) struct CompressionState {
    pending_kv: Vec<Vec<f32>>,
    pending_gate: Vec<Vec<f32>>,
    previous_kv: Vec<Vec<f32>>,
    previous_gate: Vec<Vec<f32>>,
    pub entries: Vec<Vec<f32>>,
}

#[derive(Default)]
pub(super) struct AttentionState {
    window: VecDeque<Vec<f32>>,
    pub compressed: CompressionState,
    pub index: CompressionState,
}

struct Compressor<T: Dtype, D: LlmBackend> {
    kv: Matrix<T, D>,
    gate: Matrix<T, D>,
    bias: Vec<f32>,
    norm: Vec<f32>,
    ratio: usize,
    dim: usize,
}

impl<T: Dtype, D: LlmBackend> Compressor<T, D> {
    fn load(
        loader: &WeightLoader<'_>,
        prefix: &str,
        dim: usize,
        ratio: usize,
        cfg: &TinyConfig,
        device: &D,
    ) -> OpResult<Self> {
        let width = dim * if ratio == 4 { 2 } else { 1 };
        Ok(Self {
            kv: Matrix::load(
                loader,
                &format!("{prefix}.kv_proj.weight"),
                width,
                cfg.hidden_size,
                device,
            )?,
            gate: Matrix::load(
                loader,
                &format!("{prefix}.gate_proj.weight"),
                width,
                cfg.hidden_size,
                device,
            )?,
            bias: values(loader, &format!("{prefix}.position_bias"), &[ratio, width])?,
            norm: values(loader, &format!("{prefix}.kv_norm.weight"), &[dim])?,
            ratio,
            dim,
        })
    }

    fn push(&self, kv: &[f32], gate: &[f32], state: &mut CompressionState, cfg: &TinyConfig) {
        let (r, d, width) = (self.ratio, self.dim, self.kv.output);
        let offset = state.pending_kv.len() * width;
        state.pending_kv.push(kv.to_vec());
        state.pending_gate.push(
            gate.iter()
                .enumerate()
                .map(|(j, v)| round::<T>(*v + self.bias[offset + j]))
                .collect(),
        );
        if state.pending_kv.len() < r {
            return;
        }
        let overlap = r == 4;
        let mut pooled = vec![0.0; d];
        for j in 0..d {
            let mut v = Vec::with_capacity(2 * r);
            let mut g = Vec::with_capacity(2 * r);
            if overlap {
                for (kv, gate) in state.previous_kv.iter().zip(&state.previous_gate) {
                    v.push(kv[j]);
                    g.push(gate[j]);
                }
            }
            let col = if overlap { d + j } else { j };
            for (kv, gate) in state.pending_kv.iter().zip(&state.pending_gate) {
                v.push(kv[col]);
                g.push(gate[col]);
            }
            pooled[j] = round::<T>(
                softmax(&g)
                    .iter()
                    .zip(&v)
                    .map(|(&p, &v)| round::<T>(round::<T>(p) * v))
                    .sum(),
            );
        }
        let mut entry = norm::<T>(&pooled, &self.norm, cfg.rms_norm_eps);
        rope::<T>(
            &mut entry,
            d,
            cfg.rotary_dim(),
            state.entries.len() * r,
            cfg.theta(true),
            false,
        );
        state.entries.push(entry);
        if overlap {
            state.previous_kv = std::mem::take(&mut state.pending_kv);
            state.previous_gate = std::mem::take(&mut state.pending_gate);
        } else {
            state.pending_kv.clear();
            state.pending_gate.clear();
        }
    }
}

struct Indexer<T: Dtype, D: LlmBackend> {
    compressor: Compressor<T, D>,
    query: Matrix<T, D>,
    weights: Matrix<T, D>,
}

pub(super) struct Attention<T: Dtype, D: LlmBackend> {
    qa: Matrix<T, D>,
    qb: Matrix<T, D>,
    qa_norm: Vec<f32>,
    kv: Matrix<T, D>,
    kv_norm: Vec<f32>,
    oa: Vec<Matrix<T, D>>,
    ob: Matrix<T, D>,
    sinks: Vec<f32>,
    compressor: Option<Compressor<T, D>>,
    indexer: Option<Indexer<T, D>>,
}

impl<T: Dtype, D: LlmBackend> Attention<T, D> {
    pub fn load(
        loader: &WeightLoader<'_>,
        prefix: &str,
        layer: usize,
        cfg: &TinyConfig,
        device: &D,
    ) -> OpResult<Self> {
        let ratio = cfg.ratio(layer);
        let cp = format!("{prefix}.compressor");
        let ip = format!("{cp}.indexer");
        let group_width = cfg.num_attention_heads * cfg.head_dim / cfg.o_groups;
        let oa_data = values(
            loader,
            &format!("{prefix}.o_a_proj.weight"),
            &[cfg.o_groups * cfg.o_lora_rank, group_width],
        )?;
        let mut oa = Vec::new();
        for chunk in oa_data.chunks_exact(cfg.o_lora_rank * group_width) {
            oa.push(Matrix::new(chunk, cfg.o_lora_rank, group_width, device)?);
        }
        Ok(Self {
            qa: Matrix::load(
                loader,
                &format!("{prefix}.q_a_proj.weight"),
                cfg.q_lora_rank,
                cfg.hidden_size,
                device,
            )?,
            qb: Matrix::load(
                loader,
                &format!("{prefix}.q_b_proj.weight"),
                cfg.num_attention_heads * cfg.head_dim,
                cfg.q_lora_rank,
                device,
            )?,
            qa_norm: values(
                loader,
                &format!("{prefix}.q_a_norm.weight"),
                &[cfg.q_lora_rank],
            )?,
            kv: Matrix::load(
                loader,
                &format!("{prefix}.kv_proj.weight"),
                cfg.head_dim,
                cfg.hidden_size,
                device,
            )?,
            kv_norm: values(loader, &format!("{prefix}.kv_norm.weight"), &[cfg.head_dim])?,
            oa,
            ob: Matrix::load(
                loader,
                &format!("{prefix}.o_b_proj.weight"),
                cfg.hidden_size,
                cfg.o_groups * cfg.o_lora_rank,
                device,
            )?,
            sinks: values(
                loader,
                &format!("{prefix}.sinks"),
                &[cfg.num_attention_heads],
            )?,
            compressor: if ratio == 0 {
                None
            } else {
                Some(Compressor::load(
                    loader,
                    &cp,
                    cfg.head_dim,
                    ratio,
                    cfg,
                    device,
                )?)
            },
            indexer: if ratio != 4 {
                None
            } else {
                Some(Indexer {
                    compressor: Compressor::load(loader, &ip, cfg.index_head_dim, 4, cfg, device)?,
                    query: Matrix::load(
                        loader,
                        &format!("{ip}.q_b_proj.weight"),
                        cfg.index_n_heads * cfg.index_head_dim,
                        cfg.q_lora_rank,
                        device,
                    )?,
                    weights: Matrix::load(
                        loader,
                        &format!("{ip}.scorer.weights_proj.weight"),
                        cfg.index_n_heads,
                        cfg.hidden_size,
                        device,
                    )?,
                })
            },
        })
    }

    pub fn run(
        &self,
        x: &[f32],
        start: usize,
        state: &mut AttentionState,
        cfg: &TinyConfig,
        scope: &D::Scope,
    ) -> OpResult<Vec<f32>> {
        let tokens = x.len() / cfg.hidden_size;
        let (heads, d) = (cfg.num_attention_heads, cfg.head_dim);
        let qr = norm::<T>(&self.qa.run(x, scope)?, &self.qa_norm, cfg.rms_norm_eps);
        let mut q = self.qb.run(&qr, scope)?;
        // Unweighted Q normalization rounds the reciprocal before multiplication.
        for row in q.chunks_exact_mut(d) {
            let inv = round::<T>((dot(row, row) / d as f32 + cfg.rms_norm_eps).sqrt().recip());
            for v in row {
                *v = round::<T>(*v * inv);
            }
        }
        let kv = norm::<T>(&self.kv.run(x, scope)?, &self.kv_norm, cfg.rms_norm_eps);
        let compressed = self
            .compressor
            .as_ref()
            .map(|c| Ok((c.kv.run(x, scope)?, c.gate.run(x, scope)?)))
            .transpose()?;
        let indexed = self
            .indexer
            .as_ref()
            .map(|i| {
                Ok((
                    i.compressor.kv.run(x, scope)?,
                    i.compressor.gate.run(x, scope)?,
                    i.query.run(&qr, scope)?,
                    i.weights.run(x, scope)?,
                ))
            })
            .transpose()?;
        let theta = cfg.theta(self.compressor.is_some());
        let mut output = vec![0.0; tokens * heads * d];
        for t in 0..tokens {
            let position = start + t;
            let query = &mut q[t * heads * d..(t + 1) * heads * d];
            rope::<T>(query, d, cfg.rotary_dim(), position, theta, false);
            let mut key = kv[t * d..(t + 1) * d].to_vec();
            rope::<T>(&mut key, d, cfg.rotary_dim(), position, theta, false);
            state.window.push_back(key);
            if state.window.len() > cfg.sliding_window {
                state.window.pop_front();
            }
            if let (Some(c), Some((k, g))) = (&self.compressor, &compressed) {
                let width = c.kv.output;
                c.push(
                    &k[t * width..(t + 1) * width],
                    &g[t * width..(t + 1) * width],
                    &mut state.compressed,
                    cfg,
                );
            }
            let mut picked: Vec<usize> = (0..state.compressed.entries.len()).collect();
            if let (Some(indexer), Some((k, g, iq, weights))) = (&self.indexer, &indexed) {
                let width = indexer.compressor.kv.output;
                indexer.compressor.push(
                    &k[t * width..(t + 1) * width],
                    &g[t * width..(t + 1) * width],
                    &mut state.index,
                    cfg,
                );
                let (ih, id) = (cfg.index_n_heads, cfg.index_head_dim);
                let mut iq = iq[t * ih * id..(t + 1) * ih * id].to_vec();
                rope::<T>(&mut iq, id, cfg.rotary_dim(), position, theta, false);
                let scores: Vec<f32> = state
                    .index
                    .entries
                    .iter()
                    .map(|entry| {
                        (0..ih)
                            .map(|h| {
                                dot(&iq[h * id..(h + 1) * id], entry).max(0.0) / (id as f32).sqrt()
                                    * (weights[t * ih + h] / (ih as f32).sqrt())
                            })
                            .sum()
                    })
                    .collect();
                picked = topk(&scores, cfg.index_topk);
                // HF's scatter mask keeps selected entries in chronological order.
                picked.sort_unstable();
            }
            let keys: Vec<&Vec<f32>> = state
                .window
                .iter()
                .chain(picked.iter().map(|&i| &state.compressed.entries[i]))
                .collect();
            for h in 0..heads {
                let query = &query[h * d..(h + 1) * d];
                let mut logits: Vec<f32> = keys
                    .iter()
                    .map(|key| round::<T>(round::<T>(dot(query, key)) / (d as f32).sqrt()))
                    .collect();
                logits.push(self.sinks[h]);
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for v in &mut logits {
                    *v = round::<T>(*v - max);
                }
                let probabilities = softmax(&logits);
                for j in 0..d {
                    output[(t * heads + h) * d + j] = round::<T>(
                        keys.iter()
                            .zip(&probabilities)
                            .map(|(key, &p)| round::<T>(p) * key[j])
                            .sum(),
                    );
                }
            }
            rope::<T>(
                &mut output[t * heads * d..(t + 1) * heads * d],
                d,
                cfg.rotary_dim(),
                position,
                theta,
                true,
            );
        }
        let group_width = heads * d / cfg.o_groups;
        let mut projected = vec![0.0; tokens * cfg.o_groups * cfg.o_lora_rank];
        for (g, matrix) in self.oa.iter().enumerate() {
            let rows: Vec<f32> = output
                .chunks_exact(heads * d)
                .flat_map(|row| row[g * group_width..(g + 1) * group_width].iter().copied())
                .collect();
            let result = matrix.run(&rows, scope)?;
            for t in 0..tokens {
                let offset = (t * cfg.o_groups + g) * cfg.o_lora_rank;
                projected[offset..offset + cfg.o_lora_rank]
                    .copy_from_slice(&result[t * cfg.o_lora_rank..(t + 1) * cfg.o_lora_rank]);
            }
        }
        self.ob.run(&projected, scope)
    }
}
