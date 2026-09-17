//! Scalar reference operations. GEMMs use the selected backend; these small
//! operations intentionally synchronize through host memory during bring-up.
use crate::domain::{
    dtype::Dtype,
    exec::ExecScope,
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};
use crate::models::loader::WeightLoader;

pub(super) fn round<T: Dtype>(x: f32) -> f32 {
    T::read_f64(&T::write_f64(x as f64)) as f32
}

pub(super) fn values(loader: &WeightLoader<'_>, name: &str, shape: &[usize]) -> OpResult<Vec<f32>> {
    let view = loader.read_view(name).map_err(OpError::Kernel)?;
    if view.shape() != shape {
        return Err(OpError::Shape(format!(
            "{name}: expected {shape:?}, got {:?}",
            view.shape()
        )));
    }
    let values: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|v| half::bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
            .collect(),
        other => {
            return Err(OpError::Shape(format!(
                "{name}: expected unquantized F32/BF16, got {other:?}"
            )));
        }
    };
    if values.iter().any(|x| !x.is_finite()) {
        return Err(OpError::Shape(format!("{name}: non-finite weight")));
    }
    Ok(values)
}

pub(super) struct Matrix<T: Dtype, D: LlmBackend> {
    pub weight: Tensor<T, D>,
    pub input: usize,
    pub output: usize,
}

impl<T: Dtype, D: LlmBackend> Matrix<T, D> {
    pub fn new(data: &[f32], output: usize, input: usize, device: &D) -> OpResult<Self> {
        let data: Vec<T> = data.iter().map(|&v| T::write_f64(v as f64)).collect();
        Ok(Self {
            weight: Tensor::from_host_slice(&data, [output, input], device)?,
            input,
            output,
        })
    }

    pub fn load(
        loader: &WeightLoader<'_>,
        name: &str,
        output: usize,
        input: usize,
        device: &D,
    ) -> OpResult<Self> {
        Self::new(
            &values(loader, name, &[output, input])?,
            output,
            input,
            device,
        )
    }

    pub fn run(&self, x: &[f32], scope: &D::Scope) -> OpResult<Vec<f32>> {
        if x.is_empty() || !x.len().is_multiple_of(self.input) {
            return Err(OpError::Shape(
                "V4 reference GEMM input shape mismatch".into(),
            ));
        }
        let rows = x.len() / self.input;
        let input: Vec<T> = x.iter().map(|&v| T::write_f64(v as f64)).collect();
        let input = Tensor::from_host_slice(&input, [rows, self.input], scope.device())?;
        let mut output = Tensor::zeros([rows, self.output], scope.device())?;
        D::matmul(scope, &input, &self.weight, &mut output)?;
        // Keep input/output alive until work on the compute stream completes.
        scope.synchronize()?;
        Ok(output
            .to_host_vec()?
            .iter()
            .map(|v| T::read_f64(v) as f32)
            .collect())
    }
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub(super) fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(super) fn norm<T: Dtype>(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    x.chunks_exact(dim)
        .flat_map(|row| {
            let inv = (dot(row, row) / dim as f32 + eps).sqrt().recip();
            row.iter()
                .zip(weight)
                .map(move |(&v, &w)| round::<T>(round::<T>(v * inv) * w))
        })
        .collect()
}

pub(super) fn rope<T: Dtype>(
    x: &mut [f32],
    width: usize,
    rotary: usize,
    position: usize,
    theta: f32,
    inverse: bool,
) {
    for head in x.chunks_exact_mut(width) {
        for i in (0..rotary).step_by(2) {
            let phase = position as f32 / theta.powf(i as f32 / rotary as f32);
            let cos = round::<T>(phase.cos());
            let sin = round::<T>(phase.sin()) * if inverse { -1.0 } else { 1.0 };
            let j = width - rotary + i;
            let (a, b) = (head[j], head[j + 1]);
            head[j] = round::<T>(a * cos - b * sin);
            head[j + 1] = round::<T>(b * cos + a * sin);
        }
    }
}

pub(super) fn topk(scores: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids.truncate(k.min(ids.len()));
    ids
}
