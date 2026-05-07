//! Zero-copy tensor views.
//!
//! Every method in this module returns a new `Tensor` that *shares* its
//! backing [`crate::base::buffer::Buffer`] with `self`. They manipulate only
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

    /// Returns a new view with the given shape (zero-copy, strict mode).
    ///
    /// Mirrors PyTorch's `Tensor::view` semantics: the tensor **must** be
    /// contiguous. The returned view shares the same buffer and simply
    /// recomputes row-major strides for `new_shape`.
    ///
    /// One dimension may be passed as `usize::MAX` (analogous to `-1` in
    /// PyTorch) to be inferred from the others and `numel`.
    ///
    /// # Arguments
    ///
    /// - `new_shape`: The desired shape. Its product must equal `self.numel()`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `self` is not contiguous (use [`reshape`](Self::reshape) instead).
    /// - Returns an error if the element count doesn't match.
    /// - Returns an error if more than one dimension is `usize::MAX`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    /// let v = t.view(&[6, 4])?;
    /// assert_eq!(v.shape(), &[6, 4]);
    /// ```
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

    /// Reshapes the tensor, falling back to a copy if not contiguous.
    ///
    /// Attempts a zero-copy [`view`](Self::view) first. If the tensor is not
    /// contiguous (e.g. after a transpose), it first calls
    /// [`contiguous()`](Self::contiguous) to produce a dense copy, then
    /// applies the view.
    ///
    /// This is the "always works" reshape — at the cost of a potential allocation.
    ///
    /// # Arguments
    ///
    /// - `new_shape`: The desired shape. Same rules as [`view`](Self::view).
    ///
    /// # Errors
    ///
    /// Returns an error if the element count doesn't match or allocation fails.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        if self.is_contiguous() {
            return self.view(new_shape);
        }
        self.contiguous()?.view(new_shape)
    }

    // ──────────────────────── dim manipulation ────────────────────────

    /// Inserts a size-1 dimension at position `dim` (zero-copy).
    ///
    /// The stride for the new axis is set to the stride of the axis that
    /// follows it (or `1` if appending at the end). This preserves the
    /// existing memory layout.
    ///
    /// # Arguments
    ///
    /// - `dim`: Position to insert the new axis. Must be in `[0, ndim]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim > self.ndim()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[3, 4], DataType::F32, DeviceType::Cpu)?;
    /// let u = t.unsqueeze(0)?;
    /// assert_eq!(u.shape(), &[1, 3, 4]);
    /// ```
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

    /// Removes the size-1 dimension at position `dim` (zero-copy).
    ///
    /// The target dimension must have size exactly 1; otherwise an error
    /// is returned.
    ///
    /// # Arguments
    ///
    /// - `dim`: The axis to remove. Must satisfy `shape[dim] == 1`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `dim >= self.ndim()`.
    /// - Returns an error if `shape[dim] != 1`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[1, 3, 4], DataType::F32, DeviceType::Cpu)?;
    /// let u = t.squeeze(0)?;
    /// assert_eq!(u.shape(), &[3, 4]);
    /// ```
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

    /// Removes **all** size-1 dimensions from the tensor (zero-copy).
    ///
    /// If the tensor has shape `[1, 3, 1, 4]`, the result has shape `[3, 4]`.
    /// If no dimensions are size-1, the tensor is returned unchanged.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[1, 3, 1, 4, 1], DataType::F32, DeviceType::Cpu)?;
    /// let u = t.squeeze_all()?;
    /// assert_eq!(u.shape(), &[3, 4]);
    /// ```
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

    /// Reorders the dimensions according to `perm` (zero-copy).
    ///
    /// Only the metadata (shape and strides) is rearranged — the underlying
    /// buffer is shared. The result is typically non-contiguous; call
    /// [`contiguous()`](Self::contiguous) to produce a dense copy if needed.
    ///
    /// # Arguments
    ///
    /// - `perm`: A permutation of `[0, 1, ..., ndim-1]`. Must be the same
    ///   length as `ndim` and contain each index exactly once.
    ///
    /// # Errors
    ///
    /// - Returns an error if `perm.len() != self.ndim()`.
    /// - Returns an error if `perm` is not a valid permutation (duplicates or out-of-range).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    /// let p = t.permute(&[2, 0, 1])?;
    /// assert_eq!(p.shape(), &[4, 2, 3]);
    /// assert!(!p.is_contiguous());
    /// ```
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

    /// Swaps two dimensions in the tensor (zero-copy).
    ///
    /// Equivalent to a [`permute`](Self::permute) with a 2-element swap.
    /// If `d0 == d1`, returns a clone with no change.
    ///
    /// # Arguments
    ///
    /// - `d0`, `d1`: The two dimension indices to swap. Both must be `< ndim`.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension index is out of range.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    /// let tr = t.transpose(0, 2)?;
    /// assert_eq!(tr.shape(), &[4, 3, 2]);
    /// ```
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

    /// Narrows the tensor along a single dimension (zero-copy).
    ///
    /// Returns a view that keeps `length` elements along `dim`, starting at
    /// index `start`. The shape shrinks on the specified dimension; strides
    /// are unchanged; the storage offset advances by `start * strides[dim]`.
    ///
    /// This is the fundamental building block for all slicing operations
    /// ([`Self::select`], [`Self::slice`], [`Self::chunk`], [`Self::split`], etc.).
    ///
    /// # Arguments
    ///
    /// - `dim`: The dimension to narrow along.
    /// - `start`: The starting index (inclusive).
    /// - `length`: The number of elements to keep.
    ///
    /// # Errors
    ///
    /// - Returns an error if `dim >= ndim`.
    /// - Returns an error if `start + length > shape[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[10, 5], DataType::F32, DeviceType::Cpu)?;
    /// let n = t.narrow(0, 2, 3)?; // rows 2..5
    /// assert_eq!(n.shape(), &[3, 5]);
    /// ```
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

    /// Selects a single index along `dim`, reducing the rank by 1 (zero-copy).
    ///
    /// Equivalent to `self.narrow(dim, index, 1)?.squeeze(dim)?`. The selected
    /// dimension is removed from the output shape.
    ///
    /// # Arguments
    ///
    /// - `dim`: The dimension to index into.
    /// - `index`: The specific index to select.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` or `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[4, 3], DataType::F32, DeviceType::Cpu)?;
    /// let row = t.select(0, 2)?; // third row
    /// assert_eq!(row.shape(), &[3]);
    /// ```
    pub fn select(&self, dim: usize, index: usize) -> Result<Self> {
        self.narrow(dim, index, 1)?.squeeze(dim)
    }

    /// Multi-dimensional range slice (zero-copy).
    ///
    /// Applies one `Range<usize>` per dimension, equivalent to a chain of
    /// [`narrow`](Self::narrow) calls. Each range specifies `[start, end)`
    /// for its corresponding dimension.
    ///
    /// # Arguments
    ///
    /// - `ranges`: One range per dimension. Must satisfy `ranges.len() == ndim`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `ranges.len() != self.ndim()`.
    /// - Returns an error if any range has `end < start` or exceeds the dimension size.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[10, 20], DataType::F32, DeviceType::Cpu)?;
    /// let s = t.slice_ranges(&[2..5, 0..10])?;
    /// assert_eq!(s.shape(), &[3, 10]);
    /// ```
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

    /// Legacy-style offset/shape slice (zero-copy).
    ///
    /// Selects a sub-region by specifying an offset and a new shape for
    /// each dimension. Equivalent to applying
    /// [`narrow(dim, offsets[dim], new_shape[dim])`](Self::narrow) for every dimension.
    ///
    /// # Arguments
    ///
    /// - `offsets`: Starting offset per dimension. Must have length `ndim`.
    /// - `new_shape`: Desired size per dimension. Must have length `ndim`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `offsets.len()` or `new_shape.len()` differs from `ndim`.
    /// - Returns an error if any offset + size exceeds the dimension.
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

    /// Returns the first `count` elements along dimension 0 (zero-copy).
    ///
    /// Shorthand for `self.narrow(0, 0, count)`. Commonly used to take a
    /// prefix of a flat (rank-1) buffer.
    ///
    /// # Arguments
    ///
    /// - `count`: Number of elements to keep along dimension 0.
    ///
    /// # Errors
    ///
    /// Returns an error if `count > self.shape()[0]`.
    #[inline]
    pub fn view_prefix(&self, count: usize) -> Result<Self> {
        self.narrow(0, 0, count)
    }

    // ───────────────────────── split / chunk ──────────────────────────

    /// Splits `dim` into `n` equal-length chunks (zero-copy).
    ///
    /// Returns a `Vec` of `n` views, each covering a contiguous sub-range
    /// of `dim`. The dimension size must be evenly divisible by `n`.
    ///
    /// # Arguments
    ///
    /// - `n`: Number of chunks. Must be `> 0` and divide `shape[dim]` evenly.
    /// - `dim`: The dimension to split along.
    ///
    /// # Errors
    ///
    /// - Returns an error if `n == 0`.
    /// - Returns an error if `dim` is out of range.
    /// - Returns an error if `shape[dim] % n != 0`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[6, 4], DataType::F32, DeviceType::Cpu)?;
    /// let chunks = t.chunk(3, 0)?;
    /// assert_eq!(chunks.len(), 3);
    /// assert_eq!(chunks[0].shape(), &[2, 4]);
    /// ```
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

    /// Splits `dim` into chunks of given sizes (zero-copy).
    ///
    /// Returns a `Vec` of views whose sizes along `dim` correspond to
    /// the elements of `sizes`. The sum of `sizes` must equal `shape[dim]`.
    ///
    /// # Arguments
    ///
    /// - `sizes`: Slice specifying the size of each chunk. Sum must equal `shape[dim]`.
    /// - `dim`: The dimension to split along.
    ///
    /// # Errors
    ///
    /// - Returns an error if `dim` is out of range.
    /// - Returns an error if the sum of `sizes` doesn't equal `shape[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[10, 4], DataType::F32, DeviceType::Cpu)?;
    /// let parts = t.split(&[3, 7], 0)?;
    /// assert_eq!(parts[0].shape(), &[3, 4]);
    /// assert_eq!(parts[1].shape(), &[7, 4]);
    /// ```
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
        if new_ndim > super::dims::MAX_RANK {
            return Err(Error::InvalidArgument(format!(
                "expand: resulting rank {} exceeds MAX_RANK={}",
                new_ndim, super::dims::MAX_RANK
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

    /// Collapses consecutive dimensions `[start, end]` into a single axis.
    ///
    /// If the collapsed region is storage-contiguous (i.e. the strides form
    /// a proper row-major sub-sequence), this is zero-copy. Otherwise, the
    /// tensor is first densified via [`contiguous()`](Self::contiguous) before
    /// flattening.
    ///
    /// # Arguments
    ///
    /// - `start`: First dimension to merge (inclusive).
    /// - `end`: Last dimension to merge (inclusive). Must satisfy `start <= end < ndim`.
    ///
    /// # Errors
    ///
    /// - Returns an error if the range is invalid.
    /// - Returns an error if densification fails (allocation error).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    /// let f = t.flatten(1, 2)?; // merge dims 1 and 2
    /// assert_eq!(f.shape(), &[2, 12]);
    /// ```
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

    /// Splits a single dimension into multiple dimensions (inverse of flatten).
    ///
    /// Replaces `shape[dim]` with the entries of `sizes` (which must multiply
    /// to the original extent). Zero-copy when `self` is contiguous around `dim`.
    ///
    /// # Arguments
    ///
    /// - `dim`: The dimension to split.
    /// - `sizes`: The new dimensions to replace `dim` with.
    ///   Must satisfy `sizes.iter().product() == shape[dim]`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `dim >= ndim`.
    /// - Returns an error if the product of `sizes` doesn't equal `shape[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 12], DataType::F32, DeviceType::Cpu)?;
    /// let u = t.unflatten(1, &[3, 4])?;
    /// assert_eq!(u.shape(), &[2, 3, 4]);
    /// ```
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

/// Resolves a shape containing at most one `usize::MAX` (representing `-1`)
/// by inferring that dimension's size from `numel`.
///
/// If `shape` is fully specified (no `usize::MAX` entries), validates that
/// its product equals `numel`. If one entry is `usize::MAX`, computes the
/// inferred value as `numel / product_of_known_dims`.
///
/// # Arguments
///
/// - `shape`: The requested shape, possibly with one `usize::MAX` entry.
/// - `numel`: The total number of elements to match.
///
/// # Errors
///
/// - Returns an error if more than one dimension is `usize::MAX`.
/// - Returns an error if the known dimensions don't evenly divide `numel`.
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
