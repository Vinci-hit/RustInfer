//! Ordered output-row composition; compressed bytes are never concatenated.
use infer_core::device::MemoryPort;
use infer_core::error::{OpError, OpResult};
use infer_core::quantized::BlockQuantWeight;
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct BlockQuantProjection<D: MemoryPort> {
    parts: Vec<BlockQuantWeight<D>>,
    output_rows: Vec<Range<usize>>,
    shape: [usize; 2],
}

fn output_ranges(
    shapes: impl IntoIterator<Item = [usize; 2]>,
) -> OpResult<([usize; 2], Vec<Range<usize>>)> {
    let mut shapes = shapes.into_iter();
    let first = shapes
        .next()
        .ok_or_else(|| OpError::Shape("quantized projection requires at least one part".into()))?;
    let cols = first[1];
    let mut rows = 0usize;
    let mut ranges = Vec::new();
    for [n, k] in std::iter::once(first).chain(shapes) {
        if n == 0 || k == 0 || k != cols {
            return Err(OpError::Shape(
                "projection parts require nonzero shapes and identical input width".into(),
            ));
        }
        let end = rows
            .checked_add(n)
            .ok_or_else(|| OpError::Shape("projection rows overflow".into()))?;
        end.checked_mul(cols)
            .ok_or_else(|| OpError::Shape("projection logical size overflow".into()))?;
        ranges.push(rows..end);
        rows = end;
    }
    Ok(([rows, cols], ranges))
}

impl<D: MemoryPort> BlockQuantProjection<D> {
    pub fn try_new(parts: Vec<BlockQuantWeight<D>>) -> OpResult<Self> {
        let (shape, output_rows) = output_ranges(parts.iter().map(|p| p.layout().shape()))?;
        let device = parts[0].device().device_id();
        if parts.iter().any(|p| p.device().device_id() != device) {
            return Err(OpError::Shape(
                "quantized projection parts belong to different devices".into(),
            ));
        }
        Ok(Self {
            parts,
            output_rows,
            shape,
        })
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn device(&self) -> &D {
        self.parts[0].device()
    }
    pub fn parts(&self) -> impl ExactSizeIterator<Item = (Range<usize>, &BlockQuantWeight<D>)> {
        self.output_rows.iter().cloned().zip(self.parts.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_composition_without_allocating_large_weights() {
        let (shape, ranges) = output_ranges([[12288, 5120], [1024, 5120], [1024, 5120]]).unwrap();
        assert_eq!(shape, [14336, 5120]);
        assert_eq!(ranges, [0..12288, 12288..13312, 13312..14336]);
        assert!(output_ranges([]).is_err());
        assert!(output_ranges([[1, 32], [1, 64]]).is_err());
        assert!(output_ranges([[usize::MAX, 1], [1, 1]]).is_err());
        assert!(output_ranges([[usize::MAX / 32, 32], [1, 32]]).is_err());
    }
}
