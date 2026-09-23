use std::path::PathBuf;

/// Structural errors, separate from model architecture or numerical validation.
#[derive(Debug, thiserror::Error)]
pub enum ErrorKind {
    #[error("invalid magic (expected GGUF)")]
    InvalidMagic,
    #[error("unsupported version {0}")]
    UnsupportedVersion(u32),
    #[error("big-endian GGUF is not supported")]
    UnsupportedEndian,
    #[error("truncated input: need {needed} bytes, have {remaining}")]
    Truncated { needed: u64, remaining: u64 },
    #[error("unsupported metadata type {0}")]
    UnsupportedMetadataType(u32),
    #[error("unsupported tensor type {0}")]
    UnsupportedTensorType(u32),
    #[error("unsupported tensor shape (require 1..=4 nonzero dimensions and whole blocks per row)")]
    UnsupportedTensorShape,
    #[error("invalid field: {0}")]
    InvalidField(&'static str),
    #[error("duplicate {0}")]
    Duplicate(&'static str),
    #[error("integer overflow")]
    Overflow,
    #[error("tensor data outside file")]
    OutOfBounds,
    #[error("tensor data overlaps another tensor")]
    Overlap,
    #[error("limit exceeded: {0}")]
    LimitExceeded(&'static str),
    #[error("allocation failed")]
    Allocation,
    #[error("tensor not found")]
    TensorNotFound,
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    File {
        path: PathBuf,
        source: Box<GgufError>,
    },
    #[error("GGUF at byte {offset} ({context}): {kind}")]
    Parse {
        offset: u64,
        context: String,
        kind: ErrorKind,
    },
}

impl GgufError {
    pub(super) fn at(offset: usize, context: impl Into<String>, kind: ErrorKind) -> Self {
        Self::Parse {
            offset: offset as u64,
            context: context.into(),
            kind,
        }
    }

    pub(super) fn in_file(self, path: &std::path::Path) -> Self {
        Self::File {
            path: path.to_owned(),
            source: Box::new(self),
        }
    }

    pub(super) fn context(mut self, context: &str) -> Self {
        if let Self::Parse {
            context: current, ..
        } = &mut self
        {
            *current = format!("{context}: {current}");
        }
        self
    }

    pub fn kind(&self) -> Option<&ErrorKind> {
        match self {
            Self::Parse { kind, .. } => Some(kind),
            Self::File { source, .. } => source.kind(),
            Self::Io { .. } => None,
        }
    }
}
