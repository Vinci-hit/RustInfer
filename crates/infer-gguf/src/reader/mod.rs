//! GGUF v2/v3 little-endian file reader. No model interpretation or dequantization.
//!
//! Tensor bytes borrow the read-only mapping; metadata and the index are owned.
//! Like other mmap weight readers, this requires the file to remain unchanged
//! (in particular, not truncated) while the reader or a borrowed view exists.

mod error;
mod metadata;
mod parser;
mod types;

pub use error::{ErrorKind, GgufError};
pub use metadata::{GgufArray, GgufMetadata, GgufValue};
pub use types::{BlockLayout, GgmlType};

use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ParseLimits {
    pub max_metadata_entries: u64,
    pub max_tensors: u64,
    pub max_header_bytes: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    /// Defaults to 8; values above 64 are capped to protect the parser stack.
    pub max_array_depth: usize,
    /// Conservative cumulative allocation charge (including index overhead).
    /// This bounds parser-owned allocations, not mmap virtual size or OS RSS.
    pub max_index_bytes: u64,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_metadata_entries: 100_000,
            max_tensors: 1_000_000,
            max_header_bytes: 256 << 20,
            max_string_bytes: 64 << 20,
            max_array_elements: 16_000_000,
            max_array_depth: 8,
            max_index_bytes: 256 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub byte_order: ByteOrder,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u32,
    pub data_offset: u64,
}

#[derive(Debug)]
pub struct GgufTensorInfo {
    name: String,
    ggml_type: GgmlType,
    dimensions: Vec<u64>,
    relative_offset: u64,
    file_offset: u64,
    byte_len: u64,
}

impl GgufTensorInfo {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn ggml_type(&self) -> GgmlType {
        self.ggml_type
    }
    /// On-disk order: dimension 0 is contiguous / fastest varying.
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub fn relative_offset(&self) -> u64 {
        self.relative_offset
    }
    pub fn file_offset(&self) -> u64 {
        self.file_offset
    }
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

#[derive(Debug)]
pub struct GgufTensorView<'a> {
    pub info: &'a GgufTensorInfo,
    pub bytes: &'a [u8],
}

pub struct GgufReader {
    mmap: Mmap,
    index: parser::Index,
    _file: File,
    path: PathBuf,
}

impl GgufReader {
    /// Open one file. External writers must not modify/truncate it during use.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Self::open_with_limits(path, ParseLimits::default())
    }

    pub fn open_with_limits(
        path: impl AsRef<Path>,
        limits: ParseLimits,
    ) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let io = |source| GgufError::Io {
            path: path.to_owned(),
            source,
        };
        let file = File::open(path).map_err(io)?;
        let meta = file.metadata().map_err(io)?;
        if !meta.is_file() {
            return Err(GgufError::at(
                0,
                "file",
                ErrorKind::InvalidField("expected a regular file"),
            )
            .in_file(path));
        }
        if meta.len() < 24 {
            return Err(GgufError::at(
                0,
                "header",
                ErrorKind::Truncated {
                    needed: 24,
                    remaining: meta.len(),
                },
            )
            .in_file(path));
        }
        // SAFETY: read-only map; callers must keep the backing file stable.
        // Views borrow `self`, so no reference outlives the mapping. The parser
        // only uses bounds-checked byte slices and does not cast typed pointers.
        let mmap = unsafe { Mmap::map(&file) }.map_err(io)?;
        let index = parser::parse(&mmap, &limits).map_err(|e| e.in_file(path))?;
        Ok(Self {
            mmap,
            index,
            _file: file,
            path: path.to_owned(),
        })
    }

    pub fn header(&self) -> &GgufHeader {
        &self.index.header
    }
    pub fn metadata(&self) -> &GgufMetadata {
        &self.index.metadata
    }
    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.index.tensors
    }
    pub fn contains(&self, name: &str) -> bool {
        self.index.by_name.contains_key(name)
    }
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.index
            .by_name
            .get(name)
            .map(|&i| &self.index.tensors[i])
    }
    /// Borrow encoded bytes without copying or dequantizing them.
    ///
    /// A view cannot outlive its reader:
    /// ```compile_fail
    /// use infer_gguf::GgufReader;
    /// let reader = GgufReader::open("model.gguf").unwrap();
    /// let view = reader.read_view("output.weight").unwrap();
    /// drop(reader);
    /// println!("{}", view.bytes.len());
    /// ```
    pub fn read_view(&self, name: &str) -> Result<GgufTensorView<'_>, GgufError> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| GgufError::at(0, name, ErrorKind::TensorNotFound).in_file(&self.path))?;
        // Validated and usize-checked by parse; index fields cannot be mutated.
        let start = info.file_offset as usize;
        let end = start + info.byte_len as usize;
        Ok(GgufTensorView {
            info,
            bytes: &self.mmap[start..end],
        })
    }
}

#[cfg(test)]
mod tests;
