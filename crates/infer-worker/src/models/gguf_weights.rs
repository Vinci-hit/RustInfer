//! GGUF-to-matrix boundary. No architecture transforms or device allocation.
use crate::infrastructure::io::gguf::GgufTensorView;
use infer_core::error::{OpError, OpResult};
use infer_core::quantized::{BlockQuantLayout, BlockQuantView};

/// Interpret GGUF [K,N] as a logical [N,K] matrix without moving bytes.
///
/// ```compile_fail
/// use infer_worker::{infrastructure::io::GgufReader, models::gguf_weights::block_quant_view};
/// let reader = GgufReader::open("model.gguf").unwrap();
/// let view = block_quant_view(reader.read_view("output.weight").unwrap()).unwrap();
/// drop(reader);
/// println!("{}", view.bytes().len());
/// ```
pub fn block_quant_view(view: GgufTensorView<'_>) -> OpResult<BlockQuantView<'_>> {
    let info = view.info;
    let context = || format!("GGUF tensor {} at byte {}", info.name(), info.file_offset());
    let format = info
        .ggml_type()
        .block_quant_format()
        .ok_or_else(|| OpError::unsupported("GGUF block matrix", info.ggml_type().name()))?;
    let [k, n] = info.dimensions() else {
        return Err(OpError::Shape(format!(
            "{}: block matrix requires exactly two dimensions",
            context()
        )));
    };
    let size = |v| {
        usize::try_from(v)
            .map_err(|_| OpError::Shape(format!("{}: dimension exceeds usize", context())))
    };
    let layout = BlockQuantLayout::new(format, size(*n)?, size(*k)?)
        .map_err(|e| OpError::Shape(format!("{}: {e}", context())))?;
    BlockQuantView::new(layout, view.bytes)
        .map_err(|e| OpError::Shape(format!("{}: {e}", context())))
}
