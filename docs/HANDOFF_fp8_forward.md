# HANDOFF — FP8 block-wise forward (Qwen3-4B-FP8 on H200)

Goal: run `/mnt/md2/liuwenqi/vllm_bench/qwen3-4b-fp8` (DeepSeek-style block FP8) and
match vLLM. Branch: `feat/worker-batch-forward`.

## Model scheme (verified from config.json + safetensors)
- `quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128,128]`, `activation_scheme: dynamic`.
- Quantized linears (`q/k/v/o_proj`, `gate/up/down_proj`): weight `F8_E4M3 [N,K]` (1 byte) +
  `weight_scale_inv` `BF16 [ceil(N/128), ceil(K/128)]` (128×128 weight blocks).
- NOT quantized (stay bf16): `embed_tokens`, `lm_head` (tied), all layernorms, `q_norm`/`k_norm`.
- Dynamic activation quant → at runtime, per-token × 128-K tile (1×128).

Math per linear: `out = (Xq⊙Sx) @ (Wq⊙Sw)ᵀ`, fp32 accum, bf16 out.
- `X bf16[M,K] → Xq e4m3[M,K] + Sx fp32[M,K/128]`
- `W e4m3[N,K]`, `Sw = weight_scale_inv [N/128,K/128]` (dequant multiplier: `w ≈ q * scale_inv`).

## HW / toolchain (gates)
- H200 sm_90, native e4m3 tensor cores.
- CUTLASS **4.3.3** vendored → `sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`
  is the chosen GEMM backend.
- Linked cuBLASLt = **12.8** (miniconda `rust_env`) → lacks 128-block FP8 scaling (cuBLAS 12.9+),
  so cuBLASLt blockwise is OUT. CUTLASS it is.
- `build.rs` emits `sm_90` from compute_cap "9.0"; CUTLASS Hopper FP8 needs **`sm_90a`** → build fix
  required in Phase 1.

## Decisions (locked)
- Phase 0 dequant-on-load correctness baseline FIRST.
- FP8 gemm for decode too (M=1) — halve weight mem; measure M=1 efficiency.
- GEMM backend = vendored CUTLASS 4.3.3.

## Existing scaffold to mirror (minimize new surface)
- `Packing::Fp8` already in `infer-core/src/dtype/quant.rs` `QuantScheme`.
- `QuantLinear<A,W,O,D>` component + `matmul_quant<A,W,O>` op-port = AWQ int4 precedent.
- `Fp8E4m3` type in `DTypeId` (decodes bytes→f64); legacy `DataType` enum lacks it (add in Phase 1).
- qkv & gate_up are fused **on host** into one bf16 tensor in `loader.rs`
  (`load_fused_qkv` :131, `load_fused_gate_up` :215) → Phase-0 dequant slots in before fusion.

---

## Phase 0 — correctness baseline (dequant-on-load), NO kernels
Contained to worker loader + config parse. Device sees bf16; existing path unchanged.

1. `crates/infer-worker/src/bin/worker_main.rs` `HfConfig`: add
   `quantization_config: Option<HfQuantConfig>` (`quant_method`, `weight_block_size:[usize;2]`,
   `fmt`, `activation_scheme`). Thread `fp8_block: Option<[usize;2]>` into `LoadConfig`.
2. `crates/infer-worker/src/models/loader.rs`:
   - host helper `dequant_fp8_block(w_view, scale_inv_view, [bm,bn]) -> Vec<u8>` (bf16 bytes),
     decoding e4m3 via core `Fp8E4m3::to_f64`, multiply by block scale_inv, write bf16.
   - `load_linear` / `load_fused_qkv` / `load_fused_gate_up`: if `<name>.weight_scale_inv` present,
     dequant→bf16 host buffer instead of `cast_bytes`; then existing fusion/concat path runs on bf16.
   - returns `Linear<bf16>` unchanged.
3. GATE: greedy logits / output match vLLM on a few prompts. Weights are bf16 → no mem/perf win yet.

## Phase 1 — real FP8 GEMM (prefill / eager)
1. dtype: add `DataType::F8E4M3` + `Dtype` impl for `Fp8E4m3` (SIZE_BYTES=1, DATA_TYPE=F8E4M3);
   map `safetensors::Dtype::F8_E4M3` in `tensor_from_safetensor_view`.
2. kernel `act_quant_1x128_e4m3`: bf16[M,K] → e4m3[M,K] + fp32 scales[M,K/128] (amax/448).
3. kernel `gemm_fp8_blockwise_sm90`: wrap CUTLASS collective (A-scale gran M=1,K=128; B 128×128) →
   C bf16. Preallocated workspace. Add TU to `cc` build with `-arch=sm_90a`.
4. op-port `matmul_fp8_block` (or route `matmul_quant` when `packing==Fp8`).
5. `Fp8BlockLinear<D>{weight:Tensor<Fp8E4m3>, scale_inv}` `forward(bf16→bf16)` = act-quant + gemm.
6. block field swap: `Proj<T,D>{Dense(Linear), Fp8(Fp8BlockLinear)}` for
   `Attention.qkv_proj/o_proj` + `DenseFfn.gate_up/down`. Fused qkv/gate_up = stack fp8 tiles +
   concat scale rows (legal: scale blocks are per-128-output-row). Validate vs Phase 0.

## Phase 2 — decode + CUDA graph capture
Decode runs under graph capture. CUTLASS launch + act-quant capturable (no host sync / split-K probe).
Reuse pad-to-captured-slot decode infra. Validate decode logits vs Phase 0.

## Phase 3 — perf tune vs vLLM
Fuse act-quant into prologue; M=1 path; threshold tuning; nsys vs vLLM qps sweep.

## Open items to verify during impl
- `weight_scale_inv` multiply convention (assume `w = q * scale_inv`; confirm vs vLLM Fp8 block).
- CUTLASS sm90 fp8 blockwise collective compiles under nvcc 12.8 + `sm_90a`.
- M=1 decode gemm efficiency on Hopper (else revisit decode dtype decision).
