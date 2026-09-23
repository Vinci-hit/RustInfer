use super::*;
use crate::components::block_quant_projection::BlockQuantProjection;
use crate::infrastructure::io::gguf::{GgmlType, GgufTensorView};
use crate::models::gguf_weights::block_quant_view;
use infer_core::{device::MemoryPort, quantized::BlockQuantWeight, types::Shape};
use std::ptr::NonNull;

#[derive(Clone, Copy, Debug)]
pub(super) enum Transform {
    Identity,
    Norm,
    NegativeExp,
}

pub(super) fn values(view: GgufTensorView<'_>, transform: Transform) -> OpResult<Vec<f32>> {
    let kind = view.info.ggml_type();
    let size = match kind {
        GgmlType::F32 => 4,
        GgmlType::F16 | GgmlType::BF16 => 2,
        _ => return Err(OpError::unsupported("GGUF dense tensor", kind.name())),
    };
    view.bytes
        .chunks_exact(size)
        .map(|b| {
            let x = match kind {
                GgmlType::F32 => f32::from_le_bytes(b.try_into().unwrap()),
                GgmlType::F16 => half::f16::from_le_bytes(b.try_into().unwrap()).to_f32(),
                GgmlType::BF16 => half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32(),
                _ => unreachable!(),
            };
            let y = match transform {
                Transform::Identity => x,
                // Recover the zero-centered scale before conversion to T. Rounding
                // GGUF's (1+w) directly to BF16 would erase small learned scales.
                Transform::Norm => x - 1.0,
                Transform::NegativeExp if x < 0.0 => (-x).ln(),
                Transform::NegativeExp => f32::NAN,
            };
            if !x.is_finite() || !y.is_finite() {
                return Err(OpError::Shape(format!(
                    "GGUF {}: invalid {transform:?} value {x}",
                    view.info.name()
                )));
            }
            Ok(y)
        })
        .collect()
}

/// Own both source and destination until the backend drains even on errors.
/// MemoryPort permits asynchronous uploads; a borrowed temporary is unsafe.
pub(super) fn upload<T: Dtype, D: MemoryPort>(
    data: Vec<T>,
    shape: &[usize],
    device: &D,
) -> OpResult<Tensor<T, D>> {
    let size = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .and_then(|n| n.checked_mul(T::SIZE_BYTES))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| OpError::Shape("GGUF dense allocation overflow".into()))?;
    if data.len().checked_mul(T::SIZE_BYTES) != Some(size) {
        return Err(OpError::Shape("GGUF dense upload shape mismatch".into()));
    }
    let tensor = Tensor::<T, D>::zeros(Shape::from_slice(shape), device)?;
    let dst = NonNull::new(tensor.data_ptr_mut().cast::<u8>())
        .ok_or_else(|| OpError::Kernel("GGUF null tensor allocation".into()))?;
    // SAFETY: both owned allocations cover `size`, and stay alive through sync.
    let uploaded = unsafe { device.upload(dst, data.as_ptr().cast::<u8>(), size) };
    if let Err(err) = device.synchronize() {
        std::mem::forget((data, tensor));
        return Err(OpError::Fatal(format!(
            "GGUF upload sync failed; buffers retained: {err}"
        )));
    }
    uploaded?;
    Ok(tensor)
}

pub(super) struct Weights<'a> {
    pub reader: &'a GgufReader,
}
impl Weights<'_> {
    pub fn view(&self, name: &str) -> OpResult<GgufTensorView<'_>> {
        self.reader
            .read_view(name)
            .map_err(|e| OpError::Shape(e.to_string()))
    }
    pub fn dense<T: Dtype, D: LlmBackend>(
        &self,
        name: &str,
        shape: &[usize],
        transform: Transform,
        device: &D,
    ) -> OpResult<Tensor<T, D>> {
        let host = values(self.view(name)?, transform)?
            .into_iter()
            .map(|x| T::write_f64(f64::from(x)))
            .collect();
        upload(host, shape, device)
    }
    pub fn norm<T: Dtype, D: LlmBackend>(
        &self,
        name: &str,
        dim: usize,
        eps: f32,
        device: &D,
    ) -> OpResult<RmsNorm<T, D>> {
        Ok(RmsNorm {
            weight: self.dense(name, &[dim], Transform::Norm, device)?,
            eps,
            zero_centered: true,
        })
    }
    pub fn embedding<T: Dtype, D: LlmBackend>(
        &self,
        dims: ModelDims,
        device: &D,
    ) -> OpResult<Embed<T, D>> {
        let v = self.view("token_embd.weight")?;
        if v.info.ggml_type().block_quant_format().is_some() {
            Ok(Embed::from_block_quant(BlockQuantWeight::from_host(
                block_quant_view(v)?,
                device,
            )?))
        } else {
            Ok(Embed::new(self.dense(
                "token_embd.weight",
                &[dims.vocab_size, dims.dim],
                Transform::Identity,
                device,
            )?))
        }
    }
    pub fn linear<T: Dtype, D: LlmBackend>(
        &self,
        names: &[String],
        device: &D,
    ) -> OpResult<Linear<T, D>> {
        if self
            .view(&names[0])?
            .info
            .ggml_type()
            .block_quant_format()
            .is_some()
        {
            let parts = names
                .iter()
                .map(|name| {
                    BlockQuantWeight::from_host(block_quant_view(self.view(name)?)?, device)
                })
                .collect::<OpResult<Vec<_>>>()?;
            Linear::from_block_quant(BlockQuantProjection::try_new(parts)?, None)
        } else {
            let mut host = Vec::new();
            let mut rows = 0;
            let cols = self.view(&names[0])?.info.dimensions()[0] as usize;
            for name in names {
                let v = self.view(name)?;
                rows += v.info.dimensions()[1] as usize;
                host.extend(
                    values(v, Transform::Identity)?
                        .into_iter()
                        .map(|x| T::write_f64(x as f64)),
                );
            }
            Ok(Linear::new(upload(host, &[rows, cols], device)?, None))
        }
    }
}
