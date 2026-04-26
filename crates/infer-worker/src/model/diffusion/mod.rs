//! Diffusion model family: text-to-image, TTS, video generation, etc.
//!
//! # Architecture
//!
//! ```text
//! DiffusionPipeline (trait)
//!   ├── TextToImagePipeline   ← 当前实现
//!   ├── TextToSpeechPipeline  ← 未来
//!   └── TextToVideoPipeline   ← 未来
//!
//! Scheduler (trait)
//!   ├── FlowMatchEulerScheduler  ← Z-Image Turbo
//!   ├── DDPMScheduler            ← 未来
//!   └── DDIMScheduler            ← 未来
//! ```

pub mod scheduler;
pub mod pipeline;
pub mod z_image;
pub mod diffusers_loader;
pub mod vae;
pub mod buffer;
