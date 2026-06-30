# DESIGN — Qwen3.5 (qwen3_5) support

Target weights: `/mnt/md2/liuwenqi/vllm_bench/Qwen3.5-4B`
Branch base: `feat/worker-batch-forward`

## 0. TL;DR — it is NOT "just linear attention"

The dominant new piece IS the linear-attention layer (Gated DeltaNet), but it
drags in a **new cache subsystem** (recurrent state, not paged KV) plus a
heterogeneous layer stack, and the *full*-attention layers also differ from our
current Qwen3. Concrete deltas from what we ship today:

| # | Delta | Size |
|---|-------|------|
| 1 | Gated DeltaNet "linear_attention" layer (24/32 layers) + new CUDA kernels | **huge** |
| 2 | Recurrent **state cache** (conv state + SSM state), separate from paged KV | **large** |
| 3 | Heterogeneous layer stack (3×linear, 1×full, repeat — `full_attention_interval:4`) | medium |
| 4 | Full-attn changes: gated output, partial RoPE (0.25), head_dim 256, GQA 16/4, **separate** q/k/v (no fused-qkv) | medium |
| 5 | Loader: `model.language_model.` prefix, per-layer-type dispatch, new tensors | medium |
| 6 | Config parse: nested `text_config`, `layer_types`, `linear_*`, model_type `qwen3_5` | small |
| 7 | Vision tower (`Qwen3_5ForConditionalGeneration`) | **SKIP v1** |
| 8 | MTP head (`mtp.*`, 1 draft layer) | **SKIP v1** |

**Scope v1 = text-only LLM decode** (matches the vLLM bench use). Skip vision +
MTP. Get correctness first (sequential SSM kernel), then perf (chunked scan).

---

## 1. Architecture facts (from config.json + safetensors headers)

```
hidden=2560  layers=32  vocab=248320  tie_word_embeddings=true
rms_norm_eps=1e-6  rope_theta=1e7  intermediate=9216 (SwiGLU)
layer_types = [L,L,L,F] × 8        (full_attention_interval=4)  → 24 linear, 8 full
```

### Full attention (8 layers, e.g. layer 3)
```
head_dim            = 256          (NOT hidden/heads)
num_attention_heads = 16  → q_dim = 4096
num_key_value_heads = 4   → kv_dim = 1024     (GQA 16/4)
attn_output_gate    = true
partial_rotary_factor = 0.25  → rotary_dim = 64  (RoPE on first 64 of 256)
q_norm/k_norm       = RMSNorm over head_dim=256
```
Weight shapes:
- `q_proj.weight  [8192, 2560]`  = `[gate(4096) | query(4096)]`  ← gate is the doubling
- `k_proj.weight  [1024, 2560]`, `v_proj.weight [1024, 2560]`    ← separate, NOT fused
- `o_proj.weight  [2560, 4096]`
- `q_norm.weight [256]`, `k_norm.weight [256]`

Forward: `gate,query = q_proj(x).chunk(2)`; per-head RMSNorm(query)/RMSNorm(k);
partial RoPE; paged GQA attention; `attn = attn * sigmoid(gate)`; `o_proj(attn)`.

### Linear attention = Gated DeltaNet (24 layers, e.g. layer 0)
```
linear_num_key_heads   = 16   key_head_dim = 128  → key_dim   = 2048
linear_num_value_heads = 32   value_head_dim = 128 → value_dim = 4096
linear_conv_kernel_dim = 4    (causal depthwise conv)
conv_dim = key_dim + key_dim + value_dim = 8192
mamba_ssm_dtype = float32     (recurrent state in fp32)
```
Weight shapes:
- `in_proj_qkv.weight [8192, 2560]`  → q(2048) | k(2048) | v(4096)
- `in_proj_a.weight   [32, 2560]`    → `a` (one per value head; dt input)
- `in_proj_b.weight   [32, 2560]`    → `b` (one per value head; beta input)
- `in_proj_z.weight   [4096, 2560]`  → `z` output gate (value_dim)
- `conv1d.weight [8192, 1, 4]`       → depthwise causal conv over the 8192 qkv channels
- `A_log [32] (f32)`, `dt_bias [32] (bf16)` → per-value-head decay params
- `norm.weight [128] (f32)`          → gated RMSNorm over value_head_dim
- `out_proj.weight [2560, 4096]`

Forward (matches HF `Qwen3NextGatedDeltaNet`):
```
qkv = silu(causal_conv1d(in_proj_qkv(x)))          # conv over 8192 ch, kernel 4
q,k,v = split(qkv, [2048,2048,4096]) → heads
q = l2norm(q) ; k = l2norm(k)                       # L2 over head_dim, no weight
beta = sigmoid(in_proj_b(x))                        # [T, 32]
g    = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)   # log-decay [T, 32]
S    = gated_delta_rule(q,k,v,g,beta)               # GQA: k/q head shared by 2 v heads
o    = gated_rmsnorm(S, norm.weight, silu(in_proj_z(x)))   # over value_head_dim
y    = out_proj(o)                                  # [T,4096]→[T,2560]
```
Gated delta recurrence (per value head, per step t):
```
S_t = diag(α_t)·S_{t-1} + β_t · k_tᵀ (v_t − k_t·S_{t-1})     α_t = exp(g_t)
out_t = q_t · S_t          # S is [k_head_dim=128, v_head_dim=128]
```

**Per-sequence recurrent state** (this is the new cache):
- `ssm_state`  : `[num_v_heads=32, k_head_dim=128, v_head_dim=128]` fp32 = 2 MB/seq/layer
- `conv_state` : `[conv_dim=8192, kernel-1=3]` bf16 = 48 KB/seq/layer
- ×24 linear layers → ~49 MB/seq. **Fixed size per seq (no paging).**

---

## 2. Code map — what changes, where

### 2a. Token-mixer becomes an enum (heterogeneous stack)
`crates/infer-worker/src/components/`

Today: `DecoderBlock { attention: Attention, ffn }`, homogeneous `Vec`.
Change: block holds a **mixer** that is one of two kinds.

```rust
// components/mixer.rs (new)
pub enum Mixer<T, D> {
    Full(Attention<T, D>),         // existing Attention, extended (gate + partial rope)
    Linear(GatedDeltaNet<T, D>),   // new
}
```
`DecoderBlock` runs `mixer.run(hidden, cache, ctx)?; ffn.run(...)`. Keep the
deferred-residual / `fused_add_rmsnorm` fusion exactly as-is — both mixers leave
their projection in `hidden.pending`.

`Decoder.blocks: Vec<DecoderBlock<...>>` stays, but `decode_layers` must hand
each block the **right** cache slice (paged-KV view for Full, state view for
Linear). See 2c.

### 2b. New component: `GatedDeltaNet`
`crates/infer-worker/src/components/gated_delta_net.rs` (new)

Holds: `input_layernorm` (RmsNorm), `in_proj_qkv/a/b/z` (Linear), `conv1d`
(depthwise weight + dims), `a_log`, `dt_bias`, `gated_norm` (weight+eps),
`out_proj`, dims. `run()` orchestrates the kernels in 2d, dispatching prefill
(chunked/sequential scan, seeds+writes state) vs decode (single recurrent step).

### 2c. New cache subsystem: `LinearStatePool`
`crates/infer-worker/src/domain/` (+ `infer-core` plan fields)

- `LinearStatePool { conv: Tensor[max_slots, n_lin_layers, conv_dim, k-1],
   ssm: Tensor[max_slots, n_lin_layers, n_v_heads, k_hd, v_hd] (f32) }`.
- **Slot allocator**: 1 fixed slot per *active* sequence (unlike paged KV's
  block table). Scheduler assigns a slot on admit, frees on finish. Far simpler
  than `GlobalKvAllocator` (no per-token growth).
- Paged-KV pool now sized to **8 full-attn layers only** (not 32). Need a
  layer→{full_idx | linear_idx} map in `ModelDims`/`Decoder`.
- Plan (`infer-core/plan.rs` `BatchPlan` + `SeqStep`): add `state_slot: i32`
  per seq, and a flag for "first prefill chunk" (state init = zeros).

This is the highest-integration item: scheduler (`infer-scheduler`), the worker
runtime KV build path (`runtime.rs`), and CUDA-graph capture all touch it.

### 2d. New CUDA kernels
`crates/infer-backend-cuda/src/kernels/`  (+ trait methods in
`infer-core/src/ports/fused_ops.rs`, impl in cuda backend)

1. **`causal_conv1d`** (new dir): depthwise causal conv, kernel 4, + SiLU.
   - prefill: over full ragged sequence, seed from zero, write last 3 → conv_state.
   - decode: 1 step from conv_state ring. (Mamba-style; well-known pattern.)
2. **`gated_delta_rule`** (new dir): the recurrence.
   - **v1 = sequential**: 1 block per (seq, value-head), loop over time. Correct,
     simple, slow. *De-risk here first.*
   - **v2 = chunked scan** (FLA `chunk_gated_delta_rule` style) for prefill perf.
   - decode = single-step recurrent update of `ssm_state` (cheap; cudagraph-able).
   - handles GQA repeat (k/q head i → value heads 2i, 2i+1), L2-norm of q/k,
     β=sigmoid, g=−exp(A_log)·softplus(a+dt_bias).
3. **`gated_rmsnorm`** (new, or compose): `rmsnorm(o, w, eps) * silu(z)` over
   value_head_dim.
4. **Full-attn extras**:
   - **partial RoPE**: rotary on first 64 of 256 dims. Extend
     `qkv_norm_rope_scatter` / `rope` with a `rotary_dim` param (today assumes
     full head_dim). Interleaved layout (`mrope_interleaved`); for **text-only**
     MRoPE collapses to standard 1-D RoPE (all 3 mrope sections share the text
     position) → reuse existing rope, just partial.
   - **output gate**: `attn *= sigmoid(gate)` — tiny elementwise (can fuse into
     o_proj input).
   - head_dim 256 + GQA 16/4 already supported by `flash_attn_gqa` (decode &
     prefill dispatch 64/128/192/256). ✓ no kernel work.

### 2e. Loader
`crates/infer-worker/src/models/loader.rs`

- Add weight-name **prefix** param (`model.language_model.`).
- Per-layer dispatch on `layer_types[i]`:
  - Full → load q/k/v **separately** (q is `[2*q_dim, dim]`, split gate|query);
    q_norm/k_norm over head_dim. NOT `load_fused_qkv` (assumes q+2kv, no gate).
  - Linear → load in_proj_qkv/a/b/z, conv1d, A_log(f32), dt_bias, norm(f32),
    out_proj. (Keep A_log/norm in fp32; ssm runs fp32.)
- `lm_head`: tie → `embed_tokens` (existing fallback already does this).
- RoPE cache over `rotary_dim=64`, theta=1e7.

### 2f. Config + dispatch
- `worker_main.rs HfConfig`: read nested `text_config.*` (Qwen3.5 nests
  everything), `layer_types`, `linear_*`, `attn_output_gate`,
  `partial_rotary_factor`, `head_dim`. New `LoadConfig` fields.
- `infer-protocol resolve_model_type`: map `qwen3_5` / `Qwen3_5*` → `"qwen3_5"`
  (currently any "qwen" → "qwen3"). Add `"qwen3_5" => load_qwen3_5` to
  `dispatch_worker_model!`.
- EOS ids: `eos_token_id=248044`, `im_end` from tokenizer_config.

---

## 3. Risks / unknowns

- **Chunked delta-rule kernel** is the hard part (numerics, fp32 state, GQA,
  chunk boundaries). Mitigation: ship sequential v1, validate against HF logits,
  then optimize. Prefill of long prompts will be slow until v2.
- **CUDA-graph decode** must capture the recurrent state update with stable
  pointers. State slots pre-assigned per seq, addresses stable across replay →
  should work, but the in-place `ssm_state` RMW under graph replay needs a test
  (same pattern as paged-KV scatter, which already graph-captures).
- **Interleaving with paged-KV scheduler**: two caches with different lifetimes;
  state slot must be freed exactly when seq finishes (and on preemption). Audit
  `infer-scheduler` free paths.
- **head_dim 256 KV footprint**: full-attn kv_dim=1024 × 8 layers; fine.
- **mamba_ssm_dtype f32** doubles state memory; the 49 MB/seq budget caps batch.
- Numerics: q/k **L2-norm** (not RMSNorm), softplus, exp decay — easy to get
  subtly wrong; golden-test each sub-op vs a Python reference on the real weights.

---

## 4. Suggested phasing

1. **Config + loader + dispatch** plumbing; load weights, assert all shapes
   (no forward yet). Cheap, flushes out prefix/nesting bugs.
2. **Full-attn layer** working in isolation (gate + partial rope + head_dim 256);
   verify a full-attn-only sanity path.
3. **GatedDeltaNet sequential** (conv + sequential delta-rule + gated norm) +
   **LinearStatePool** + plan `state_slot`. Correctness vs HF on short prompts.
4. **Hetero stack** end-to-end: full + linear interleaved, prefill decode loop;
   match HF logits on a real prompt.
5. **CUDA-graph decode** for the recurrent step; perf pass (chunked scan v2).
6. (later) MTP for spec-decode; vision.

Effort: items 1–2 small; 3–5 are the bulk (new kernels + new cache + scheduler).
This is a multi-week feature, not a one-file add.
