//! Z-Image / Z-Image-Turbo text-to-image model.
//!
//! # Module layout
//!
//! ```text
//! z_image/
//! ├── mod.rs              ← you are here
//! ├── pipeline.rs         ← ZImagePipeline: full encode → denoise → decode flow
//! ├── transformer.rs      ← ZImageTransformer2DModel (S3-DiT): denoising backbone
//! ├── dit_block.rs        ← Single DiT transformer block
//! ├── state.rs            ← Pre-allocated DitState / PipelineState
//! ├── text_encoder.rs     ← Qwen3-based text encoder wrapper
//! ├── timestep_embedder.rs← Sinusoidal timestep → MLP embedding
//! ├── rope_embedder_3d.rs ← 3D Rotary Position Embedding
//! └── patchify.rs         ← Patchify / Unpatchify (zero-alloc _into variants)
//! ```

pub mod pipeline;
pub mod transformer;
pub mod timestep_embedder;
pub mod rope_embedder_3d;
pub mod patchify;
pub mod dit_block;
pub mod text_encoder;
pub mod state;

pub use pipeline::ZImagePipeline;
pub use text_encoder::Qwen3TextEncoder;
pub use state::{DitShapeSpec, DitState, PipelineState, ZImageCapacity};
