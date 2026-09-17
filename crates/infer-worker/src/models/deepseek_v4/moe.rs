use super::{TinyConfig, math::*};
use crate::domain::{
    dtype::Dtype,
    ports::{OpError, OpResult, backend::LlmBackend},
};
use crate::models::loader::WeightLoader;

struct Expert<T: Dtype, D: LlmBackend> {
    gate_up: Matrix<T, D>,
    down: Matrix<T, D>,
}

impl<T: Dtype, D: LlmBackend> Expert<T, D> {
    fn run(&self, x: &[f32], cfg: &TinyConfig, scope: &D::Scope) -> OpResult<Vec<f32>> {
        let projected = self.gate_up.run(x, scope)?;
        let width = cfg.moe_intermediate_size;
        let mut activated = Vec::with_capacity(projected.len() / 2);
        for row in projected.chunks_exact(2 * width) {
            for j in 0..width {
                let gate = row[j].min(cfg.swiglu_limit);
                let up = row[width + j].clamp(-cfg.swiglu_limit, cfg.swiglu_limit);
                activated.push(round::<T>(round::<T>(gate * sigmoid(gate)) * up));
            }
        }
        self.down.run(&activated, scope)
    }
}

pub(super) struct Moe<T: Dtype, D: LlmBackend> {
    router: Matrix<T, D>,
    hash: Option<Vec<usize>>,
    bias: Vec<f32>,
    experts: Vec<Expert<T, D>>,
    shared: Expert<T, D>,
}

impl<T: Dtype, D: LlmBackend> Moe<T, D> {
    pub fn load(
        loader: &WeightLoader<'_>,
        prefix: &str,
        layer: usize,
        cfg: &TinyConfig,
        device: &D,
    ) -> OpResult<Self> {
        let (e, d, m, k) = (
            cfg.n_routed_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
        );
        let gate_up = values(
            loader,
            &format!("{prefix}.experts.gate_up_proj"),
            &[e, 2 * m, d],
        )?;
        let down = values(loader, &format!("{prefix}.experts.down_proj"), &[e, d, m])?;
        let mut experts = Vec::with_capacity(e);
        for i in 0..e {
            experts.push(Expert {
                gate_up: Matrix::new(
                    &gate_up[i * 2 * m * d..(i + 1) * 2 * m * d],
                    2 * m,
                    d,
                    device,
                )?,
                down: Matrix::new(&down[i * d * m..(i + 1) * d * m], d, m, device)?,
            });
        }
        let (hash, bias) = if cfg.mlp_layer_types[layer] == "hash_moe" {
            let name = format!("{prefix}.gate.tid2eid");
            let view = loader.read_view(&name).map_err(OpError::Kernel)?;
            if view.shape() != [cfg.vocab_size, k] || view.dtype() != safetensors::Dtype::I64 {
                return Err(OpError::Shape(format!(
                    "{name}: expected I64 [{}, {k}]",
                    cfg.vocab_size
                )));
            }
            let mut hash = Vec::with_capacity(cfg.vocab_size * k);
            for bytes in view.data().chunks_exact(8) {
                let id = i64::from_le_bytes(bytes.try_into().unwrap());
                if id < 0 || id >= e as i64 {
                    return Err(OpError::Shape(format!("{name}: invalid expert {id}")));
                }
                hash.push(id as usize);
            }
            for row in hash.chunks_exact(k) {
                for (i, id) in row.iter().enumerate() {
                    if row[..i].contains(id) {
                        return Err(OpError::Shape(format!(
                            "{name}: duplicate expert in hash route"
                        )));
                    }
                }
            }
            (Some(hash), vec![0.0; e])
        } else {
            (
                None,
                values(
                    loader,
                    &format!("{prefix}.gate.e_score_correction_bias"),
                    &[e],
                )?,
            )
        };
        let mut shared_up = values(
            loader,
            &format!("{prefix}.shared_experts.gate_proj.weight"),
            &[m, d],
        )?;
        shared_up.extend(values(
            loader,
            &format!("{prefix}.shared_experts.up_proj.weight"),
            &[m, d],
        )?);
        Ok(Self {
            router: Matrix::load(loader, &format!("{prefix}.gate.weight"), e, d, device)?,
            hash,
            bias,
            experts,
            shared: Expert {
                gate_up: Matrix::new(&shared_up, 2 * m, d, device)?,
                down: Matrix::load(
                    loader,
                    &format!("{prefix}.shared_experts.down_proj.weight"),
                    d,
                    m,
                    device,
                )?,
            },
        })
    }

    pub fn run(
        &self,
        x: &[f32],
        ids: &[usize],
        cfg: &TinyConfig,
        scope: &D::Scope,
    ) -> OpResult<Vec<f32>> {
        let (d, e, k) = (
            cfg.hidden_size,
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
        );
        let logits = self.router.run(x, scope)?;
        let mut routes = vec![Vec::<(usize, f32)>::new(); e];
        for (t, row) in logits.chunks_exact(e).enumerate() {
            let scores: Vec<f32> = row
                .iter()
                .map(|&v| {
                    let softplus = v.max(0.0) + (-v.abs()).exp().ln_1p();
                    round::<T>(round::<T>(softplus).sqrt())
                })
                .collect();
            let selected = if let Some(hash) = &self.hash {
                hash[ids[t] * k..(ids[t] + 1) * k].to_vec()
            } else {
                topk(
                    &scores
                        .iter()
                        .zip(&self.bias)
                        .map(|(s, b)| round::<T>(s + b))
                        .collect::<Vec<_>>(),
                    k,
                )
            };
            let total = round::<T>(selected.iter().map(|&i| scores[i]).sum::<f32>());
            for i in selected {
                // Correction bias affects selection only, never the route weight.
                let weight = round::<T>(round::<T>(scores[i] / total) * cfg.routed_scaling_factor);
                routes[i].push((t, weight));
            }
        }
        let mut out = vec![0.0; x.len()];
        for (expert, rows) in self.experts.iter().zip(routes) {
            if rows.is_empty() {
                continue;
            }
            let input: Vec<f32> = rows
                .iter()
                .flat_map(|&(t, _)| x[t * d..(t + 1) * d].iter().copied())
                .collect();
            let result = expert.run(&input, cfg, scope)?;
            for (i, &(t, weight)) in rows.iter().enumerate() {
                for j in 0..d {
                    out[t * d + j] =
                        round::<T>(out[t * d + j] + round::<T>(result[i * d + j] * weight));
                }
            }
        }
        let shared = self.shared.run(x, cfg, scope)?;
        for (v, s) in out.iter_mut().zip(shared) {
            *v = round::<T>(*v + s);
        }
        Ok(out)
    }
}
