//! Z-Image / Z-Image-Turbo text-to-image model.
//!
//! # Module layout
//!
//! ```text
//! z_image/
//! ├── mod.rs           ← you are here
//! ├── pipeline.rs      ← ZImagePipeline: full encode → denoise → decode flow
//! └── transformer.rs   ← ZImageTransformer2DModel (S3-DiT): the denoising backbone
//! ```
//!
//! Mirrors the vllm-omni structure:
//! `diffusion/models/z_image/{pipeline_z_image.py, z_image_transformer.py}`

pub mod pipeline;
pub mod transformer;

pub use pipeline::ZImagePipeline;
