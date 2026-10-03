//! Shared GGUF file access. No CUDA or worker dependency.
mod reader;
pub use reader::*;
#[cfg(feature = "text")]
pub mod text;

/// Config paths ending in .gguf select the single-file loader. Actual contents
/// are validated by GgufReader, never inferred from the filename/model name.
pub fn is_gguf_path(path: impl AsRef<std::path::Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
}
