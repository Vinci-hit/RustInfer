use super::{TinyConfig, math::*};
use crate::domain::{dtype::Dtype, ports::OpResult};
use crate::models::loader::WeightLoader;

pub(super) struct Hyper {
    weight: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    hc: usize,
    dim: usize,
}

impl Hyper {
    pub fn load(
        loader: &WeightLoader<'_>,
        prefix: &str,
        cfg: &TinyConfig,
        head: bool,
    ) -> OpResult<Self> {
        let n = cfg.hc_mult;
        let mix = if head { n } else { (n + 2) * n };
        let field = if head { "hc_" } else { "" };
        Ok(Self {
            weight: values(
                loader,
                &format!("{prefix}.{field}fn"),
                &[mix, n * cfg.hidden_size],
            )?,
            base: values(loader, &format!("{prefix}.{field}base"), &[mix])?,
            scale: values(
                loader,
                &format!("{prefix}.{field}scale"),
                &[if head { 1 } else { 3 }],
            )?,
            hc: n,
            dim: cfg.hidden_size,
        })
    }

    /// Per row: collapsed stream, post weights and row-major combination matrix.
    pub fn pre<T: Dtype>(&self, x: &[f32], cfg: &TinyConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (n, d) = (self.hc, self.dim);
        let mut collapsed = Vec::new();
        let mut posts = Vec::new();
        let mut combinations = Vec::new();
        for row in x.chunks_exact(n * d) {
            let inv = (dot(row, row) / (n * d) as f32 + cfg.rms_norm_eps)
                .sqrt()
                .recip();
            let flat: Vec<f32> = row.iter().map(|v| v * inv).collect();
            let mix: Vec<f32> = self
                .weight
                .chunks_exact(n * d)
                .map(|w| dot(w, &flat))
                .collect();
            let pre: Vec<f32> = (0..n)
                .map(|i| sigmoid(mix[i] * self.scale[0] + self.base[i]) + cfg.hc_eps)
                .collect();
            for j in 0..d {
                collapsed.push(round::<T>((0..n).map(|i| pre[i] * row[i * d + j]).sum()));
            }
            if self.scale.len() == 1 {
                continue;
            }
            posts.extend(
                (0..n).map(|i| 2.0 * sigmoid(mix[n + i] * self.scale[1] + self.base[n + i])),
            );
            let mut comb = Vec::with_capacity(n * n);
            for i in 0..n {
                let logits: Vec<f32> = (0..n)
                    .map(|j| mix[2 * n + i * n + j] * self.scale[2] + self.base[2 * n + i * n + j])
                    .collect();
                comb.extend(softmax(&logits).into_iter().map(|v| v + cfg.hc_eps));
            }
            for iteration in 0..cfg.hc_sinkhorn_iters {
                if iteration > 0 {
                    for row in comb.chunks_exact_mut(n) {
                        let total = row.iter().sum::<f32>() + cfg.hc_eps;
                        for v in row {
                            *v /= total;
                        }
                    }
                }
                for j in 0..n {
                    let total = (0..n).map(|i| comb[i * n + j]).sum::<f32>() + cfg.hc_eps;
                    for i in 0..n {
                        comb[i * n + j] /= total;
                    }
                }
            }
            combinations.extend(comb);
        }
        (collapsed, posts, combinations)
    }

    pub fn post<T: Dtype>(
        &self,
        residual: &[f32],
        x: &[f32],
        post: &[f32],
        comb: &[f32],
    ) -> Vec<f32> {
        let (n, d) = (self.hc, self.dim);
        let mut out = vec![0.0; residual.len()];
        for t in 0..x.len() / d {
            for j in 0..n {
                for z in 0..d {
                    // HF/reference uses comb^T @ residual, not comb @ residual.
                    let mixed = round::<T>(
                        (0..n)
                            .map(|i| {
                                round::<T>(comb[t * n * n + i * n + j])
                                    * residual[(t * n + i) * d + z]
                            })
                            .sum(),
                    );
                    let branch = round::<T>(round::<T>(post[t * n + j]) * x[t * d + z]);
                    out[(t * n + j) * d + z] = round::<T>(mixed + branch);
                }
            }
        }
        out
    }
}
