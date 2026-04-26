//! VAE Decoder for Z-Image (Flux AutoencoderKL).
//!
//! Architecture:
//! ```text
//! latents [B, 16, H', W']
//!   ↓ conv_in: Conv2d(16 → 512, 3x3, pad=1)
//! mid_block:
//!   ResnetBlock(512)
//!   AttnBlock(512) — group_norm → Linear q/k/v → sdpa → Linear proj_out
//!   ResnetBlock(512)
//!   ↓
//! up_blocks[0..4]: each 3 resnets (+ optional conv_shortcut on resnets[0] if ch change) + optional Upsample2x
//!   [0]: 512→512, upsample
//!   [1]: 512→512, upsample
//!   [2]: 512→256 (shortcut on r0), upsample
//!   [3]: 256→128 (shortcut on r0), NO upsample
//!   ↓
//! conv_norm_out: GroupNorm(32, 128)
//! SiLU
//! conv_out: Conv2d(128 → 3, 3x3, pad=1)
//! ```

pub mod decoder;
