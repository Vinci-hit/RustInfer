//! Validated block-quantized matrices. No file I/O or numerical decoding.

use std::ops::Range;
use std::ptr::NonNull;

use crate::device::MemoryPort;
use crate::dtype::quant::BlockQuantFormat;
use crate::error::{OpError, OpResult};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQuantLayout {
    format: BlockQuantFormat,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    byte_len: usize,
}

fn overflow() -> OpError {
    OpError::Shape("block-quantized matrix size overflows".into())
}

impl BlockQuantLayout {
    pub fn new(format: BlockQuantFormat, rows: usize, cols: usize) -> OpResult<Self> {
        let block = format.layout();
        if rows == 0 || cols == 0 || !cols.is_multiple_of(block.elements) {
            return Err(OpError::Shape(format!(
                "{format:?} requires nonzero [N,K] and K divisible by {}: [{rows},{cols}]",
                block.elements
            )));
        }
        rows.checked_mul(cols).ok_or_else(overflow)?;
        let row_bytes = (cols / block.elements)
            .checked_mul(block.bytes)
            .ok_or_else(overflow)?;
        let byte_len = rows.checked_mul(row_bytes).ok_or_else(overflow)?;
        // Rust slices/allocations cannot span more than isize::MAX bytes.
        if byte_len > isize::MAX as usize {
            return Err(overflow());
        }
        Ok(Self {
            format,
            rows,
            cols,
            row_bytes,
            byte_len,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
    pub fn format(&self) -> BlockQuantFormat {
        self.format
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
    pub fn blocks_per_row(&self) -> usize {
        self.cols / self.format.layout().elements
    }

    /// Byte range relative to this matrix, not to its underlying allocation.
    pub fn row_byte_range(&self, rows: Range<usize>) -> OpResult<Range<usize>> {
        if rows.start > rows.end || rows.end > self.rows {
            return Err(OpError::Shape(format!(
                "row range {rows:?} outside {} rows",
                self.rows
            )));
        }
        Ok(rows
            .start
            .checked_mul(self.row_bytes)
            .ok_or_else(overflow)?
            ..rows.end.checked_mul(self.row_bytes).ok_or_else(overflow)?)
    }

    pub fn block_byte_range(&self, row: usize, blocks: Range<usize>) -> OpResult<Range<usize>> {
        if row >= self.rows || blocks.start > blocks.end || blocks.end > self.blocks_per_row() {
            return Err(OpError::Shape(
                "block range outside quantized matrix".into(),
            ));
        }
        let base = row.checked_mul(self.row_bytes).ok_or_else(overflow)?;
        let block_bytes = self.format.layout().bytes;
        let offset = |b: usize| {
            b.checked_mul(block_bytes)
                .and_then(|b| base.checked_add(b))
                .ok_or_else(overflow)
        };
        Ok(offset(blocks.start)?..offset(blocks.end)?)
    }

    fn sliced(&self, rows: Range<usize>) -> OpResult<(Self, Range<usize>)> {
        let bytes = self.row_byte_range(rows.clone())?;
        Ok((
            Self::new(self.format, rows.end - rows.start, self.cols)?,
            bytes,
        ))
    }
}

/// A lightweight layout + borrowed byte slice. Cloning allocates nothing.
#[derive(Clone, Copy, Debug)]
pub struct BlockQuantView<'a> {
    layout: BlockQuantLayout,
    bytes: &'a [u8],
}

impl<'a> BlockQuantView<'a> {
    pub fn new(layout: BlockQuantLayout, bytes: &'a [u8]) -> OpResult<Self> {
        if bytes.len() != layout.byte_len() {
            return Err(OpError::Shape(format!(
                "quantized bytes: expected {}, got {}",
                layout.byte_len(),
                bytes.len()
            )));
        }
        Ok(Self { layout, bytes })
    }
    pub fn layout(&self) -> &BlockQuantLayout {
        &self.layout
    }
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
    pub fn slice_rows(&self, rows: Range<usize>) -> OpResult<Self> {
        let (layout, range) = self.layout.sliced(rows)?;
        Self::new(layout, &self.bytes[range])
    }
}

/// Owns encoded bytes on a device. Logical shape is separate from byte storage.
#[derive(Debug)]
pub struct BlockQuantWeight<D: MemoryPort> {
    layout: BlockQuantLayout,
    bytes: Tensor<u8, D>,
}

impl<D: MemoryPort> Clone for BlockQuantWeight<D> {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout,
            bytes: self.bytes.clone(),
        }
    }
}

impl<D: MemoryPort> BlockQuantWeight<D> {
    pub fn try_new(layout: BlockQuantLayout, bytes: Tensor<u8, D>) -> OpResult<Self> {
        if bytes.shape().as_slice() != [layout.byte_len()] || !bytes.is_contiguous() {
            return Err(OpError::Shape(
                "quantized storage must be a contiguous [byte_len] byte tensor".into(),
            ));
        }
        Ok(Self { layout, bytes })
    }

    /// Synchronous copy; source borrowing ends safely even on backend failure.
    /// Uses owned staging because MemoryPort permits pending work on errors.
    /// If synchronization fails, retain both allocations rather than freeing
    /// memory that the device might still access. No async API is exposed.
    pub fn from_host(view: BlockQuantView<'_>, device: &D) -> OpResult<Self> {
        let len = view.layout.byte_len();
        let bytes = Tensor::<u8, D>::zeros([len], device)?;
        let mut staging = device.alloc_host_buffer(len)?;
        if staging.bytes_mut().len() != len || staging.bytes().len() != len {
            return Err(OpError::Shape("host staging buffer length mismatch".into()));
        }
        staging.bytes_mut().copy_from_slice(view.bytes);
        let dst = NonNull::new(bytes.data_ptr_mut())
            .ok_or_else(|| OpError::Kernel("null weight allocation".into()))?;
        // SAFETY: both buffers hold len bytes and remain owned through sync.
        let uploaded = unsafe { device.upload(dst, staging.bytes().as_ptr(), len) };
        let synced = device.synchronize();
        if let Err(err) = synced {
            // A failing backend gives no generic guarantee that DMA has stopped.
            // Retaining on this exceptional path prevents use-after-free.
            std::mem::forget((staging, bytes));
            return Err(OpError::Fatal(format!(
                "block quant upload synchronization failed; buffers retained: {err}"
            )));
        }
        uploaded?;
        Self::try_new(view.layout, bytes)
    }

    pub fn layout(&self) -> &BlockQuantLayout {
        &self.layout
    }
    pub fn bytes(&self) -> &Tensor<u8, D> {
        &self.bytes
    }
    pub fn device(&self) -> &D {
        self.bytes.device()
    }
    pub fn slice_rows(&self, rows: Range<usize>) -> OpResult<Self> {
        let (layout, range) = self.layout.sliced(rows)?;
        let bytes = self.bytes.narrow(0, range.start, range.len())?;
        Self::try_new(layout, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_ranges_and_borrowed_subviews() {
        for &format in BlockQuantFormat::ALL {
            let b = format.layout();
            let layout = BlockQuantLayout::new(format, 5, b.elements * 3).unwrap();
            let bytes: Vec<_> = (0..layout.byte_len()).map(|i| i as u8).collect();
            let view = BlockQuantView::new(layout, &bytes).unwrap();
            assert_eq!(
                layout.row_byte_range(1..4).unwrap(),
                3 * b.bytes..12 * b.bytes
            );
            assert_eq!(
                layout.block_byte_range(2, 1..3).unwrap(),
                7 * b.bytes..9 * b.bytes
            );
            let sub = view.slice_rows(1..5).unwrap().slice_rows(1..3).unwrap();
            assert_eq!(sub.bytes(), &bytes[6 * b.bytes..12 * b.bytes]);
            assert_eq!(sub.bytes().as_ptr(), bytes[6 * b.bytes..].as_ptr());
            assert_eq!(sub.layout().shape(), [2, b.elements * 3]);
            assert_eq!(
                layout.row_byte_range(5..5).unwrap(),
                layout.byte_len()..layout.byte_len()
            );
            assert!(view.slice_rows(2..2).is_err());
            assert!(layout.row_byte_range(4..6).is_err());
            assert!(layout.row_byte_range(usize::MAX..usize::MAX).is_err());
            assert!(layout.block_byte_range(0, 0..4).is_err());
            assert!(layout.block_byte_range(5, 0..0).is_err());
            assert!(BlockQuantView::new(layout, &bytes[..bytes.len() - 1]).is_err());
            assert!(BlockQuantLayout::new(format, 2, b.elements - 1).is_err());
        }
    }

    #[test]
    fn dimensions_and_overflow_rejected() {
        for (n, k) in [(0, 256), (1, 0), (usize::MAX, 256), (2, usize::MAX - 255)] {
            assert!(BlockQuantLayout::new(BlockQuantFormat::Q3_K, n, k).is_err());
        }
    }
}
