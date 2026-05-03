//! Zero-copy tensor views.
//!
//! Every method in this module returns a new `Tensor` that *shares* its
//! backing [`Buffer`] with `self`. They manipulate only
//! (shape, strides, storage_offset) and are therefore O(ndim) in time and
//! zero-allocation outside of a small `Dims` on the stack.
//!
//! None of these operations ever touches element data. If a caller needs a
//! dense copy (e.g. to satisfy a kernel that demands contiguous input), it
//! must explicitly call [`Tensor::contiguous`] or
//! [`Tensor::to_owned`] afterwards.

use std::ops::Range;

use crate::base::error::{Error, Result};

use super::dims::Dims;
use super::tensor::Tensor;

impl Tensor {
    // ──────────────────────── reshape family ──────────────────────────

    /// Strict reshape: requires `self.is_contiguous()` and returns a new
    /// view with freshly computed row-major strides. Mirrors PyTorch's
    /// `Tensor::view` semantics.
    pub fn view(&self, new_shape: &[usize]) -> Result<Self> {
        let new_shape = normalize_shape_with_infer(new_shape, self.numel())?;
        if !self.is_contiguous() {
            return Err(Error::InvalidArgument(format!(
                "view: input not contiguous (shape={:?}, strides={:?}); \
                 use reshape() for an allocating fallback",
                self.shape(), self.strides(),
            )).into());
        }
        if new_shape.product() != self.numel() {
            return Err(Error::InvalidArgument(format!(
                "view: shape {:?} has {} elements, expected {}",
                new_shape.as_slice(), new_shape.product(), self.numel()
            )).into());
        }
        let new_strides = Dims::contiguous_strides_for(new_shape.as_slice());
        Ok(self.from_view_parts(new_shape, new_strides, self.storage_offset()))
    }

    /// Best-effort reshape: attempts a zero-copy `view`; if the tensor
    /// isn't contiguous, falls back to a [`Tensor::contiguous`] copy first.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        if self.is_contiguous() {
            return self.view(new_shape);
        }
        self.contiguous()?.view(new_shape)
    }

    // ──────────────────────── dim manipulation ────────────────────────

    /// Insert a size-1 dimension at `dim`. Always zero-copy.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let ndim = self.ndim();
        if dim > ndim {
            return Err(Error::InvalidArgument(format!(
                "unsqueeze: dim {} out of range [0, {}]", dim, ndim
            )).into());
        }
        // Stride for the new axis: next-axis stride, or 1 if appending.
        let stride_at_dim = if dim == ndim { 1 } else { self.strides()[dim] };
        let mut shape   = Dims::from_slice(self.shape());
        let mut strides = Dims::from_slice(self.strides());
        shape.insert(dim, 1);
        strides.insert(dim, stride_at_dim);
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }

    /// Drop the size-1 dimension at `dim`. Zero-copy.
    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidArgument(format!(
                "squeeze: dim {} out of range [0, {})", dim, ndim
            )).into());
        }
        if self.shape()[dim] != 1 {
            return Err(Error::InvalidArgument(format!(
                "squeeze: shape[{}]={} != 1", dim, self.shape()[dim]
            )).into());
        }
        let mut shape   = Dims::from_slice(self.shape());
        let mut strides = Dims::from_slice(self.strides());
        shape.remove(dim);
        strides.remove(dim);
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }

    /// Drop *all* size-1 dimensions.
    pub fn squeeze_all(&self) -> Result<Self> {
        let mut shape   = Dims::new();
        let mut strides = Dims::new();
        for (i, &d) in self.shape().iter().enumerate() {
            if d != 1 {
                shape.push(d);
                strides.push(self.strides()[i]);
            }
        }
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }

    // ───────────────────────── permute / transpose ────────────────────

    /// Zero-copy permute — only the metadata is reordered. The returned
    /// view is usually non-contiguous; call `.contiguous()` to densify.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let ndim = self.ndim();
        if perm.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "permute: perm length {} != ndim {}", perm.len(), ndim
            )).into());
        }
        let mut seen = [false; super::dims::MAX_RANK];
        for &p in perm {
            if p >= ndim || seen[p] {
                return Err(Error::InvalidArgument(format!(
                    "permute: invalid permutation {:?}", perm
                )).into());
            }
            seen[p] = true;
        }
        let mut shape   = Dims::new();
        let mut strides = Dims::new();
        for &p in perm {
            shape.push(self.shape()[p]);
            strides.push(self.strides()[p]);
        }
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }

    /// Swap two dimensions. Equivalent to `permute` with a 2-swap.
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Self> {
        let ndim = self.ndim();
        if d0 >= ndim || d1 >= ndim {
            return Err(Error::InvalidArgument(format!(
                "transpose: ({}, {}) out of range for ndim {}", d0, d1, ndim
            )).into());
        }
        if d0 == d1 {
            return Ok(self.clone());
        }
        let mut shape   = Dims::from_slice(self.shape());
        let mut strides = Dims::from_slice(self.strides());
        shape.as_mut_slice().swap(d0, d1);
        strides.as_mut_slice().swap(d0, d1);
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }

    // ───────────────────────────── slice family ────────────────────────

    /// Zero-copy 1-D narrowing: keep `length` entries of `dim` starting at
    /// `start`. Shape shrinks, strides unchanged, storage offset advances.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidArgument(format!(
                "narrow: dim {} out of range [0, {})", dim, ndim
            )).into());
        }
        let extent = self.shape()[dim];
        let end = start.checked_add(length).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "narrow: overflow on dim {} (start={} + length={})",
                dim, start, length
            ))
        })?;
        if end > extent {
            return Err(Error::InvalidArgument(format!(
                "narrow: out of bounds on dim {}: {}..{} > {}",
                dim, start, end, extent
            )).into());
        }
        let mut shape = Dims::from_slice(self.shape());
        shape[dim] = length;
        let strides = Dims::from_slice(self.strides());
        let offset = self.storage_offset() + start * self.strides()[dim];
        Ok(self.from_view_parts(shape, strides, offset))
    }

    /// Select a single index along `dim`, dropping that dimension. Zero-copy.
    pub fn select(&self, dim: usize, index: usize) -> Result<Self> {
        self.narrow(dim, index, 1)?.squeeze(dim)
    }

    /// Multi-dimensional range slice. `ranges.len()` must equal `ndim`.
    /// Equivalent to a chain of [`Tensor::narrow`] calls.
    pub fn slice_ranges(&self, ranges: &[Range<usize>]) -> Result<Self> {
        if ranges.len() != self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "slice_ranges: expected {} ranges, got {}",
                self.ndim(), ranges.len()
            )).into());
        }
        let mut out = self.clone();
        for (dim, r) in ranges.iter().enumerate() {
            if r.end < r.start {
                return Err(Error::InvalidArgument(format!(
                    "slice_ranges: invalid range {}..{} on dim {}",
                    r.start, r.end, dim
                )).into());
            }
            out = out.narrow(dim, r.start, r.end - r.start)?;
        }
        Ok(out)
    }

    /// Legacy-style offset/shape slice. Equivalent to applying
    /// [`Tensor::narrow`] along every dimension. Zero-copy, correctness-
    /// preserving for strided inputs (unlike the old implementation).
    pub fn slice(&self, offsets: &[usize], new_shape: &[usize]) -> Result<Self> {
        let ndim = self.ndim();
        if offsets.len() != ndim || new_shape.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "slice: dim mismatch — ndim={}, offsets.len={}, shape.len={}",
                ndim, offsets.len(), new_shape.len()
            )).into());
        }
        let mut out = self.clone();
        for dim in 0..ndim {
            out = out.narrow(dim, offsets[dim], new_shape[dim])?;
        }
        Ok(out)
    }

    /// Zero-copy 1-D prefix on a rank-1 tensor — the common case for
    /// "first `count` elements of a flat buffer".
    #[inline]
    pub fn view_prefix(&self, count: usize) -> Result<Self> {
        self.narrow(0, 0, count)
    }

    // ───────────────────────── split / chunk ──────────────────────────

    /// Split `dim` into `n` contiguous equal-length pieces via zero-copy
    /// narrowing. `shape[dim]` must be divisible by `n`.
    pub fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Self>> {
        if n == 0 {
            return Err(Error::InvalidArgument("chunk: n must be > 0".into()).into());
        }
        let extent = self.shape().get(dim).copied().ok_or_else(|| {
            Error::InvalidArgument(format!("chunk: dim {} out of range", dim))
        })?;
        if extent % n != 0 {
            return Err(Error::InvalidArgument(format!(
                "chunk: shape[{}]={} not divisible by n={}",
                dim, extent, n
            )).into());
        }
        let size = extent / n;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.narrow(dim, i * size, size)?);
        }
        Ok(out)
    }

    /// Split `dim` into chunks whose sizes sum to `shape[dim]`.
    pub fn split(&self, sizes: &[usize], dim: usize) -> Result<Vec<Self>> {
        let extent = self.shape().get(dim).copied().ok_or_else(|| {
            Error::InvalidArgument(format!("split: dim {} out of range", dim))
        })?;
        let total: usize = sizes.iter().sum();
        if total != extent {
            return Err(Error::InvalidArgument(format!(
                "split: sizes sum to {} but shape[{}]={}",
                total, dim, extent
            )).into());
        }
        let mut out = Vec::with_capacity(sizes.len());
        let mut off = 0;
        for &s in sizes {
            out.push(self.narrow(dim, off, s)?);
            off += s;
        }
        Ok(out)
    }

    // ───────────────────────── broadcast / expand ─────────────────────

    /// PyTorch-style `expand`: broadcast any size-1 dimension to a larger
    /// size without copying. The resulting view has stride 0 on broadcast
    /// dimensions, so every logical index on that axis maps to the same
    /// underlying element.
    ///
    /// Rules:
    /// - `new_shape.len() >= self.ndim()`; leading axes are treated as
    ///   freshly inserted size-1 dims.
    /// - For matching axes, `self.shape[i]` must equal `new_shape[i]` or be
    ///   1 (broadcastable).
    pub fn expand(&self, new_shape: &[usize]) -> Result<Self> {
        let old_ndim = self.ndim();
        let new_ndim = new_shape.len();
        if new_ndim < old_ndim {
            return Err(Error::InvalidArgument(format!(
                "expand: new_ndim {} < self.ndim {}", new_ndim, old_ndim
            )).into());
        }
        let pad = new_ndim - old_ndim;
        let mut shape   = Dims::new();
        let mut strides = Dims::new();
        for i in 0..new_ndim {
            let new_d = new_shape[i];
            if i < pad {
                // Freshly prepended axis — broadcast-only.
                shape.push(new_d);
                strides.push(0);
            } else {
                let old_d = self.shape()[i - pad];
                let old_s = self.strides()[i - pad];
                if old_d == new_d {
                    shape.push(new_d);
                    strides.push(old_s);
                } else if old_d == 1 {
                    shape.push(new_d);
                    strides.push(0);
                } else {
                    return Err(Error::InvalidArgument(format!(
                        "expand: cannot broadcast dim {} from {} to {}",
                        i, old_d, new_d
                    )).into());
                }
            }
        }
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }

    // ──────────────────────── flatten / unflatten ─────────────────────

    /// Collapse dimensions `[start, end]` into a single axis, returning a
    /// zero-copy view when the collapsed region is storage-contiguous,
    /// otherwise materialising via [`Tensor::contiguous`].
    pub fn flatten(&self, start: usize, end: usize) -> Result<Self> {
        let ndim = self.ndim();
        if start >= ndim || end >= ndim || start > end {
            return Err(Error::InvalidArgument(format!(
                "flatten: invalid range [{}, {}] for ndim {}", start, end, ndim
            )).into());
        }
        // Zero-copy fast path: collapsed region is stride-contiguous.
        let collapsible = (start..end).all(|i| {
            self.strides()[i] == self.strides()[i + 1] * self.shape()[i + 1]
        });
        if collapsible {
            let mut shape   = Dims::new();
            let mut strides = Dims::new();
            let mut merged = 1usize;
            for i in 0..ndim {
                if i >= start && i <= end {
                    merged *= self.shape()[i];
                    if i == end {
                        shape.push(merged);
                        strides.push(self.strides()[end]);
                    }
                } else {
                    shape.push(self.shape()[i]);
                    strides.push(self.strides()[i]);
                }
            }
            return Ok(self.from_view_parts(shape, strides, self.storage_offset()));
        }
        // Fallback: densify, then rebuild shape with merged axis.
        let dense = self.contiguous()?;
        let mut flat_shape = Dims::new();
        let mut merged = 1usize;
        for i in 0..ndim {
            if i >= start && i <= end {
                merged *= dense.shape()[i];
                if i == end { flat_shape.push(merged); }
            } else {
                flat_shape.push(dense.shape()[i]);
            }
        }
        dense.view(flat_shape.as_slice())
    }

    /// Opposite of [`Tensor::flatten`]: replace `shape[dim]` with
    /// `sizes[..]` (must multiply to the original extent). Zero-copy when
    /// `self` is contiguous around `dim`.
    pub fn unflatten(&self, dim: usize, sizes: &[usize]) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "unflatten: dim {} out of range", dim
            )).into());
        }
        let extent = self.shape()[dim];
        let prod: usize = sizes.iter().product();
        if prod != extent {
            return Err(Error::InvalidArgument(format!(
                "unflatten: sizes {:?} multiply to {}, expected {}",
                sizes, prod, extent
            )).into());
        }
        // Row-major inner strides for the split-out sizes.
        let inner_strides = Dims::contiguous_strides_for(sizes);
        let base_stride = self.strides()[dim];
        let mut shape   = Dims::new();
        let mut strides = Dims::new();
        for i in 0..self.ndim() {
            if i == dim {
                for (k, &s) in sizes.iter().enumerate() {
                    shape.push(s);
                    strides.push(base_stride * inner_strides[k]);
                }
            } else {
                shape.push(self.shape()[i]);
                strides.push(self.strides()[i]);
            }
        }
        Ok(self.from_view_parts(shape, strides, self.storage_offset()))
    }
}

// ─────────────────────── shape-inference helper ────────────────────────

/// Resolve a shape with at most one `-1` (encoded as `usize::MAX`) entry,
/// validating that the product matches `numel`. Currently the codebase
/// passes fully-specified shapes, so this primarily checks the product.
fn normalize_shape_with_infer(shape: &[usize], numel: usize) -> Result<Dims> {
    const INFER: usize = usize::MAX;
    let infer_count = shape.iter().filter(|&&d| d == INFER).count();
    if infer_count > 1 {
        return Err(Error::InvalidArgument(
            "reshape: at most one dimension may be -1/INFER".into()
        ).into());
    }
    if infer_count == 0 {
        return Ok(Dims::from_slice(shape));
    }
    let known: usize = shape.iter().filter(|&&d| d != INFER).product();
    if known == 0 || numel % known != 0 {
        return Err(Error::InvalidArgument(format!(
            "reshape: cannot infer dimension for shape {:?} with numel {}",
            shape, numel
        )).into());
    }
    let inferred = numel / known;
    let mut out = Dims::new();
    for &d in shape {
        out.push(if d == INFER { inferred } else { d });
    }
    Ok(out)
}
