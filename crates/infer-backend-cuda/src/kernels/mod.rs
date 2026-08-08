//! CUDA kernel wrappers — paged KV path only.
//! Each kernel co-located with its `.cu`/`.h` source in its own dir.

// Kernel launch wrappers intentionally mirror their fixed C ABI signatures.
// Bundling those arguments into Rust-only structs would obscure the FFI
// boundary without making the launches safer.
#![allow(clippy::too_many_arguments)]

use crate::config::CudaDeviceInfo;
use infer_core::ports::{OpError, OpResult};

// Zero-cost dtype dispatch: per-kernel binding traits (`CudaFloat` supertrait)
// plus the single `narrow_float!` narrowing point used at the `MathOps for
// Cuda` boundary. See `dtype_kernel.rs`.
pub mod dtype_kernel;

// --- kernels with a co-located mod.rs in their dir ---
pub mod add;
pub mod broadcast_mul;
pub mod embedding;
pub mod ewise_mul;
pub mod flash_attn_gqa; // was attention_paged
pub mod fused_add_rmsnorm;
pub mod gather_merge;
pub mod groupnorm;
pub mod kv_cache; // was scatter_kv_paged
pub mod layernorm;
pub mod matmul;
pub mod moe_combine;
pub mod moe_grouped_gemm;
pub mod moe_permute;
pub mod moe_router;
pub mod qkv_norm_rope_scatter;
pub mod rmsnorm;
pub mod rope;
pub mod rope_interleaved;
pub mod sampler;
pub mod scalar;
pub mod softmax;
pub mod split_cols;
pub mod swiglu; // was activation
pub mod upsample;

// --- extra wrappers sharing a dir (distinct module names) ---
#[path = "cast_fill/cast_dtype.rs"]
pub mod cast_dtype;
#[path = "cast_fill/pad.rs"]
pub mod pad;
#[path = "matmul/sdpa.rs"]
pub mod sdpa;

// --- no-cu (pure Rust / cudnn) modules, kept flat ---
pub mod concat_seq;
pub mod conv2d;

/// One CUDA kernel family's fixed, per-device initialization hook.
///
/// Keep shape-dependent plan caches and request workspaces out of this table:
/// entries here may depend only on immutable [`CudaDeviceInfo`] and must finish
/// before NCCL setup, warmup, graph capture, or the first forward.
struct DeviceInitializer {
    name: &'static str,
    run: fn(CudaDeviceInfo) -> OpResult<()>,
}

const DEVICE_INITIALIZERS: &[DeviceInitializer] = &[
    DeviceInitializer {
        name: "matmul",
        run: matmul::init_device,
    },
    DeviceInitializer {
        name: "flash_attn_gqa",
        run: flash_attn_gqa::init_device,
    },
];

/// Initialize every kernel family for the active CUDA device.
///
/// The caller owns device selection. Initializers are deliberately idempotent
/// and lock-free so constructing two backend handles for one device is safe and
/// a failed attempt can be retried without process-global state.
pub(crate) fn initialize_device(info: CudaDeviceInfo) -> OpResult<()> {
    for initializer in DEVICE_INITIALIZERS {
        (initializer.run)(info).map_err(|error| {
            OpError::Kernel(format!(
                "CUDA device {} initializer '{}': {error}",
                info.device_id, initializer.name
            ))
        })?;
    }
    Ok(())
}
