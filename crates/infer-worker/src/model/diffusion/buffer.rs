//! Diffusion-specific pre-allocated buffer registry.
//!
//! Motivation
//! ----------
//! The diffusion forward pass touches `Tensor::new` hundreds of times per
//! `generate()` (confirmed by Nsight Systems: ~1200 `cudaMalloc` calls / run).
//! Every allocation is both a CPU bottleneck (`cudaMalloc` ≈ 200 µs each) and
//! a hard blocker for CUDA Graph capture, because capture requires a fixed,
//! allocation-free kernel stream.
//!
//! This module mirrors the LLM-side `model::BufferType` / `model::Workspace`
//! pattern but keeps an entirely **independent** enum space so the two
//! families can evolve without cross-talk.
//!
//! Usage
//! -----
//! - [`DiffBufferType`] enumerates every intermediate tensor the Z-Image /
//!   VAE path needs, grouped by owner scope.
//! - [`DiffWorkspace`] is just a `HashMap` of those buffers; the concrete
//!   state structs (`PipelineState`, `DitState`, `VaeState`) own one each
//!   and expose safe slicing helpers.
//!
//! ```ignore
//! // Pre-allocated in `DitState::new`:
//! let buf = state.buffers.get(&DiffBufferType::BlkQ).unwrap(); // [S_total_max, dim]
//! let mut q = buf.slice(&[0, 0], &[seq, dim])?;                // zero-copy view
//! self.to_q.forward(&norm1x, &mut q, cuda_config)?;
//! ```

use std::collections::HashMap;

use crate::tensor::Tensor;

/// A tag identifying a reusable tensor slot shared across the diffusion forward pass.
///
/// Every variant corresponds to exactly **one** pre-allocated `Tensor`, sized
/// to the capacity configured at pipeline construction time (max latent
/// resolution, max caption tokens, etc.). Callers borrow a right-sized
/// sub-view via `Tensor::slice`; nobody reallocates after `DitState::new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffBufferType {
    // ─────────────────────── Pipeline scope ───────────────────────
    /// `[1, 16, latent_h, latent_w]` weight-dtype — working latent between
    /// denoising steps. Written by `prepare_latents` and `scheduler.step`.
    Latents,
    /// `[1, 16, latent_h, latent_w]` weight-dtype — secondary latent buffer
    /// used to ping-pong with `Latents` across `scheduler.step` calls so
    /// each step can write to a fresh destination without aliasing its
    /// read source.
    LatentsTmp,
    /// `[1, 16, latent_h, latent_w]` — output of DiT per step, after the
    /// `× -1` sign flip. Consumed by `scheduler.step`.
    NoisePred,
    /// `[16, 1, latent_h, latent_w]` — reshape of `Latents` into the 5D
    /// input the DiT expects. Populated via a DtoD copy each step.
    Latent5D,

    // ─────────────────────── Transformer scope ─────────────────────
    /// `[1]` f32 on **device** — holds `norm_t × t_scale`. Written via
    /// `cudaMemcpyAsync` **outside** any CUDA Graph capture region so
    /// replays read a fresh scalar.
    TValueDev,
    /// `[1, 256]` f32 — sinusoidal frequency encoding (device memory).
    TFreq,
    /// `[1, mid_size]` weight-dtype — timestep MLP layer-1 output.
    TEmbHidden,
    /// `[1, ADALN_EMBED_DIM]` weight-dtype — timestep MLP layer-2 output.
    TEmbOut,
    /// `[ADALN_EMBED_DIM]` weight-dtype — `TEmbOut` flattened, used as
    /// conditioning `c` by modulation blocks.
    AdalnInput,
    /// `[ADALN_EMBED_DIM]` weight-dtype — `silu(c)` scratch for the final
    /// layer's adaLN modulation path.
    AdalnSilu,

    /// `[n_patches_max, patch_in_dim]` weight-dtype — flattened image
    /// patches after `patchify`.
    Patches,
    /// `[n_patches_max, dim]` weight-dtype — image token embeddings out of
    /// `x_embedder`.
    XEmb,
    /// `[S_img_max, dim]` weight-dtype — `XEmb` padded up to `SEQ_MULTI_OF`.
    XPadded,
    /// `[S_img_max, dim]` weight-dtype — ping-pong partner of `XPadded`
    /// so the noise_refiner block chain can write each layer's output to
    /// a distinct buffer from its input (block forward is dst-write).
    XPaddedTmp,

    /// `[S_cap_max, cap_feat_dim]` weight-dtype — raw caption features
    /// zero-padded (via `pad_last_row`) up to `SEQ_MULTI_OF`. Feeds
    /// `cap_embedder_norm`.
    CapFeatsPadded,
    /// `[S_cap_max, cap_feat_dim]` weight-dtype — caption embeddings after
    /// RMSNorm.
    CapNormed,
    /// `[S_cap_max, dim]` weight-dtype — `cap_embedder_linear` output.
    CapEmb,
    /// `[S_cap_max, dim]` weight-dtype — `CapEmb` with pad tokens stamped
    /// in for the original padding region.
    CapPadded,
    /// `[S_cap_max, dim]` weight-dtype — ping-pong partner of `CapPadded`
    /// for the context_refiner block chain.
    CapPaddedTmp,

    /// `[S_img_max, 3]` i32 on CPU — image 3D position ids.
    XPosIds,
    /// `[S_cap_max, 3]` i32 on CPU — caption 3D position ids.
    CapPosIds,

    /// `[S_img_max, head_dim/2]` f32 on device — cos cache for image tokens.
    XCos,
    /// `[S_img_max, head_dim/2]` f32 on device — sin cache for image tokens.
    XSin,
    /// `[S_cap_max, head_dim/2]` f32 — cos cache for caption tokens.
    CapCos,
    /// `[S_cap_max, head_dim/2]` f32 — sin cache for caption tokens.
    CapSin,
    /// `[S_total_max, head_dim/2]` f32 — concatenated cos cache for the
    /// unified (image|caption) stream.
    UnifiedCos,
    /// `[S_total_max, head_dim/2]` f32 — concatenated sin cache.
    UnifiedSin,

    /// `[S_total_max, dim]` weight-dtype — concatenated unified stream
    /// consumed by the 30 main DiT layers.
    Unified,
    /// `[S_total_max, dim]` weight-dtype — ping-pong partner of `Unified`
    /// for the main layer block chain.
    UnifiedTmp,

    /// `[S_img_max, dim]` weight-dtype — final-layer pre-projection tensor.
    FinalNormed,
    /// `[1, dim]` weight-dtype — final layer's adaLN scale vector.
    FinalScale,
    /// `[S_img_max, final_out_dim]` weight-dtype — final linear projection
    /// output, before unpatchify.
    FinalOut,
    /// `[16, 1, latent_h_max, latent_w_max]` weight-dtype — DiT predicted
    /// velocity, written by `unpatchify`.
    ImageOut,

    // ─────────────────────── DiTBlock scope (shared across all blocks) ─────
    //
    // These buffers are the **block-internal scratch pool**: every call to
    // `DiTBlock::forward` overwrites them from scratch, and the next block
    // starts fresh. Only their owning block is allowed to read/write them
    // during the block's forward. Cross-block communication flows through
    // `XPadded` / `CapPadded` / `Unified` (pipeline-level buffers), never
    // through these.
    /// `[1, 4*dim]` — modulation projection output.
    BlkModOut,
    /// `[S_total_max, dim]` — `attention_norm1(x)`.
    BlkNorm1X,
    /// `[S_total_max, dim]` — Q projection.
    BlkQ,
    /// `[S_total_max, dim]` — K projection.
    BlkK,
    /// `[S_total_max, dim]` — V projection.
    BlkV,
    /// `[S_total_max * n_heads, head_dim]` — Q flattened for per-head RMSNorm input.
    BlkQNormIn,
    /// `[S_total_max * n_heads, head_dim]` — Q per-head RMSNorm output.
    BlkQNormOut,
    /// `[S_total_max * n_heads, head_dim]` — K flattened for per-head RMSNorm input.
    BlkKNormIn,
    /// `[S_total_max * n_heads, head_dim]` — K per-head RMSNorm output.
    BlkKNormOut,
    /// `[1, n_heads, S_total_max, head_dim]` — Q in SDPA layout (B,H,S,D).
    BlkQHsd,
    /// `[1, n_heads, S_total_max, head_dim]` — K in SDPA layout.
    BlkKHsd,
    /// `[1, n_heads, S_total_max, head_dim]` — V in SDPA layout.
    BlkVHsd,
    /// `[1, n_heads, S_total_max, head_dim]` — SDPA output.
    BlkAttnSdpa,
    /// `[S_total_max, dim]` — attention output permuted back to `[S, H*D]`.
    BlkAttnFlat,
    /// `[S_total_max, dim]` — `to_out` projection output.
    BlkToOut,
    /// `[S_total_max, dim]` — `attention_norm2(to_out)` post-modulation.
    BlkNorm2Attn,
    /// `[S_total_max, dim]` — `ffn_norm1(x)` post-modulation.
    BlkNorm1Ffn,
    /// `[S_total_max, hidden_dim]` — `w1(x)` (SwiGLU gate branch).
    BlkW1Out,
    /// `[S_total_max, hidden_dim]` — `w3(x)` (SwiGLU value branch).
    BlkW3Out,
    /// `[S_total_max, dim]` — `w2(silu(w1)*w3)` output.
    BlkFfnOut,
    /// `[S_total_max, dim]` — `ffn_norm2(ffn_out)` post-modulation.
    BlkNorm2Ffn,

    // ─────────────────────── VAE decoder scope ──────────────────────
    //
    // VAE upsamples resolution between stages, so we dedicate one buffer
    // per stage (no single max-sized buffer could be reused without waste).
    /// `[B, mid_ch, lh,   lw  ]` — output of `conv_in` / mid-block.
    VaeStage0,
    /// `[B, ch1,    lh*2, lw*2]` — after `up_blocks[0]`.
    VaeStage1,
    /// `[B, ch2,    lh*4, lw*4]` — after `up_blocks[1]`.
    VaeStage2,
    /// `[B, ch3,    lh*8, lw*8]` — after `up_blocks[2..]` / `conv_norm_out`.
    VaeStage3,
    /// `[B, 3,      lh*8, lw*8]` — final `conv_out` output.
    VaeStageOut,

    /// `[B, ch_max, h_stage_max, w_stage_max]` — resnet GroupNorm scratch.
    VaeResnetTmp1,
    /// `[B, ch_max, h_stage_max, w_stage_max]` — resnet conv1/intermediate scratch.
    VaeResnetTmp2,

    /// `[B, mid_ch, lh, lw]` — mid-block self-attention Q.
    VaeAttnQ,
    /// `[B, mid_ch, lh, lw]` — mid-block self-attention K.
    VaeAttnK,
    /// `[B, mid_ch, lh, lw]` — mid-block self-attention V.
    VaeAttnV,
    /// `[B, mid_ch, lh, lw]` — mid-block self-attention output.
    VaeAttnOut,
}

/// A registry of pre-allocated diffusion buffers keyed by [`DiffBufferType`].
///
/// Concrete state structs own one of these and expose safe helpers; the
/// raw map is kept `pub` so call sites can use `HashMap::get_disjoint_mut`
/// to borrow several entries simultaneously without fighting the borrow
/// checker.
pub type DiffWorkspace = HashMap<DiffBufferType, Tensor>;

/// Convenience helper: borrow a buffer by key, expecting it to be present.
///
/// The state constructors install every key they advertise, so a missing
/// entry is a programmer error — we panic with a descriptive message to
/// surface the mismatch loudly in tests.
#[inline]
pub fn must_get(ws: &DiffWorkspace, ty: DiffBufferType) -> &Tensor {
    ws.get(&ty).unwrap_or_else(|| {
        panic!("DiffWorkspace: buffer {:?} not installed — state constructor bug", ty)
    })
}

/// Mutable counterpart of [`must_get`].
#[inline]
pub fn must_get_mut(ws: &mut DiffWorkspace, ty: DiffBufferType) -> &mut Tensor {
    ws.get_mut(&ty).unwrap_or_else(|| {
        panic!("DiffWorkspace: buffer {:?} not installed — state constructor bug", ty)
    })
}
