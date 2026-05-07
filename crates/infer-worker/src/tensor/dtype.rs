//! Compile-time `Dtype` trait bridging Rust scalar types and the runtime
//! [`DataType`] enum.
//!
//! All element types that can back a `TypedTensor<T>` must implement this
//! trait. Keeping it `Send + Sync + Copy + 'static` means tensor storage can
//! cross thread boundaries and be memcpy'd freely.

use crate::base::DataType;
use half::{bf16, f16};

/// Marker trait for tensor element types.
///
/// `Copy` gives us byte-level copy semantics, which matches how we move
/// memory through `Buffer`. The associated `DTYPE` constant lets generic
/// code round-trip to the runtime [`DataType`].
///
/// # Supported Types
///
/// The tensor framework supports exactly five element types:
///
/// | Type | Size | Use Case |
/// |------|------|----------|
/// | `f32` | 4B | Primary float type; GPU and CPU kernels |
/// | `i32` | 4B | Integer computations; indices |
/// | `i8` | 1B | Quantized models; low-precision inference |
/// | `f16` | 2B | Reduced precision; memory-efficient models |
/// | `bf16` | 2B | Google Brain Float; hardware-accelerated on recent GPUs |
///
/// # Why These Types?
///
/// - **f32**: Near-universal support across all hardware. Default for most code.
/// - **i32**: Standard for integer-based computations and shape/index storage.
/// - **i8**: Memory-efficient quantized inference (reduced by 4×); essential for mobile/edge.
/// - **f16**: IEEE 754 half-precision; widely supported on GPUs; good precision/memory tradeoff.
/// - **bf16**: Truncated f32 (keeps exponent, loses mantissa); better numerical stability
///   than f16 for many ML workloads; native support on TPUs and modern GPUs (A100, H100).
///
/// # Why Not Other Types?
///
/// - **f64**: Generally too large for model weights; rarely justified outside scientific computing.
/// - **u8**: Overlaps with i8 use case; less canonical for signed-ness of quantized values.
/// - **u32 / u64**: Rarely needed in inference workloads.
///
/// # Implementation
///
/// Each implementation simply bridges to the corresponding [`DataType`] runtime tag.
/// This allows `Tensor::empty(shape, DataType::F32, device)` to dispatch to the
/// correct `TypedTensor::<f32>::new()` at runtime.
///
/// # Generic Code Pattern
///
/// The trait enables generic functions like:
/// ```ignore
/// fn process<T: Dtype>(tensor: &TypedTensor<T>) -> Result<()> {
///     println!("dtype: {:?}", T::DTYPE);  // Print the runtime type
///     // ... T-specific code ...
/// }
/// ```
pub trait Dtype: Send + Sync + Copy + 'static {
    const DTYPE: DataType;
}

/// Standard single-precision float (IEEE 754).
///
/// - **Size:** 4 bytes
/// - **Range:** ±1.2e-38 to ±3.4e38
/// - **Precision:** ~7 significant decimal digits
/// - **Use:** Default choice; supported everywhere (CPU, CUDA, other accelerators)
/// - **Cost:** Largest supported float; memory-heavy for large models
impl Dtype for f32 {
    const DTYPE: DataType = DataType::F32;
}

/// Standard 32-bit signed integer.
///
/// - **Size:** 4 bytes
/// - **Range:** ±2.1B (-2^31 to 2^31 - 1)
/// - **Use:** Integer tensors, indexing operations, quantized weight storage
/// - **Note:** Casting floats to i32 truncates (no rounding); be careful with large values
impl Dtype for i32 {
    const DTYPE: DataType = DataType::I32;
}

/// 8-bit signed integer.
///
/// - **Size:** 1 byte
/// - **Range:** -128 to 127
/// - **Use:** Quantized inference; post-training quantization (PTQ) of weights
/// - **Advantage:** 4× memory reduction vs. f32; enables larger batch sizes on memory-constrained devices
/// - **Tradeoff:** Requires careful calibration; loss of precision for outlier values
/// - **Best for:** Mobile inference, edge devices, on-device ML
impl Dtype for i8 {
    const DTYPE: DataType = DataType::I8;
}

/// IEEE 754 half-precision float (16-bit).
///
/// - **Size:** 2 bytes
/// - **Range:** ±6.1e-5 to ±6.5e4
/// - **Precision:** ~3-4 significant decimal digits
/// - **Use:** Mixed-precision training; reduced-precision inference
/// - **Advantage:** 2× memory savings; hardware support on most GPUs
/// - **Tradeoff:** Lower precision; prone to underflow/overflow without careful scaling
/// - **Best for:** Graphics, efficient inference on modern GPUs
impl Dtype for f16 {
    const DTYPE: DataType = DataType::F16;
}

/// bfloat16 (Brain Float, Google 16-bit).
///
/// - **Size:** 2 bytes
/// - **Format:** Truncated f32 (1 sign + 8 exponent + 7 mantissa; f32 is 1+8+23)
/// - **Range:** Same as f32 (±1.2e-38 to ±3.4e38) but reduced mantissa precision
/// - **Use:** Efficient inference and training; preferred for modern ML workflows
/// - **Advantage:** Exponent bits match f32, reducing numerical issues; native TPU/GPU support (A100, H100)
/// - **Tradeoff:** Lower mantissa precision than f16; less hardware support on older GPUs
/// - **Numerical property:** Casting f32→bf16 is a simple bit-shift (no rounding logic needed)
/// - **Best for:** Large-scale models, TPU/modern GPU training, cloud inference
impl Dtype for bf16 {
    const DTYPE: DataType = DataType::BF16;
}
