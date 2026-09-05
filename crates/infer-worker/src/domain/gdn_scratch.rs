//! One preallocated workspace shared by serial GDN layers. Simultaneously live
//! projections have separate storage; no aliases with the dense decoder slots.

use std::rc::Rc;

use super::cache::LinearDims;
use super::dtype::Dtype;
use super::ports::backend::LlmBackend;
use super::ports::{OpError, OpResult};
use super::tensor::Tensor;

pub struct GdnScratch<T: Dtype, D: LlmBackend> {
    pub(crate) dim: usize,
    pub(crate) dims: LinearDims,
    cap_tokens: usize,
    buffers: GdnBuffers<T, D>,
}

pub(crate) struct GdnBuffers<T: Dtype, D: LlmBackend> {
    pub normed: Tensor<T, D>,
    pub qkv: Tensor<T, D>,
    pub conv: Tensor<T, D>,
    pub q: Tensor<T, D>,
    pub k: Tensor<T, D>,
    pub v: Tensor<T, D>,
    pub a: Tensor<T, D>,
    pub b: Tensor<T, D>,
    pub z: Tensor<T, D>,
    pub core: Tensor<T, D>,
    pub gated: Tensor<T, D>,
    pub out: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend> GdnScratch<T, D> {
    pub fn new(device: &D, dim: usize, dims: LinearDims, cap_tokens: usize) -> OpResult<Rc<Self>> {
        dims.validate()?;
        if dim == 0 || cap_tokens == 0 {
            return Err(OpError::Shape(
                "GdnScratch requires nonzero dimensions and capacity".into(),
            ));
        }
        let alloc = |cols| Tensor::zeros([cap_tokens, cols], device);
        Ok(Rc::new(Self {
            dim,
            dims,
            cap_tokens,
            buffers: GdnBuffers {
                normed: alloc(dim)?,
                qkv: alloc(dims.conv_dim())?,
                conv: alloc(dims.conv_dim())?,
                q: alloc(dims.key_dim())?,
                k: alloc(dims.key_dim())?,
                v: alloc(dims.value_dim())?,
                a: alloc(dims.num_value_heads)?,
                b: alloc(dims.num_value_heads)?,
                z: alloc(dims.value_dim())?,
                core: alloc(dims.value_dim())?,
                gated: alloc(dims.value_dim())?,
                out: alloc(dim)?,
            },
        }))
    }

    pub(crate) fn validate(&self, dim: usize, dims: LinearDims, tokens: usize) -> OpResult<()> {
        if self.dim != dim || self.dims != dims || tokens > self.cap_tokens {
            return Err(OpError::Shape(
                "GDN scratch geometry/capacity mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn buffers(&self, tokens: usize) -> OpResult<GdnBuffers<T, D>> {
        self.validate(self.dim, self.dims, tokens)?;
        let b = &self.buffers;
        Ok(GdnBuffers {
            normed: b.normed.narrow(0, 0, tokens)?,
            qkv: b.qkv.narrow(0, 0, tokens)?,
            conv: b.conv.narrow(0, 0, tokens)?,
            q: b.q.narrow(0, 0, tokens)?,
            k: b.k.narrow(0, 0, tokens)?,
            v: b.v.narrow(0, 0, tokens)?,
            a: b.a.narrow(0, 0, tokens)?,
            b: b.b.narrow(0, 0, tokens)?,
            z: b.z.narrow(0, 0, tokens)?,
            core: b.core.narrow(0, 0, tokens)?,
            gated: b.gated.narrow(0, 0, tokens)?,
            out: b.out.narrow(0, 0, tokens)?,
        })
    }
}
