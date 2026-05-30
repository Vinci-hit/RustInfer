//! Diffusion model family — Z-Image text-to-image pipeline.
//!
//! Architecture:
//! ```text
//! ZImagePipeline<T, D>
//!   ├── ZImageTransformer<T, D>  (S3-DiT denoising backbone)
//!   │     ├── 30× DiTBlock<T, D>
//!   │     ├── 2× noise_refiner DiTBlock<T, D>
//!   │     └── 2× context_refiner DiTBlock<T, D>
//!   ├── VaeDecoder<T, D>         (conv + groupnorm + upsample)
//!   └── FlowMatchEulerScheduler  (pure math, no device ops)
//! ```

pub mod dit_block;
pub mod transformer;
pub mod vae_decoder;
pub mod pipeline;
pub mod scheduler;
pub mod state;
pub mod rope_3d;
pub mod timestep_embedder;
pub mod patchify;
pub mod text_encoder;
