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
- The generic CUDA translation units are built for `sm_90`; the Hopper FP8 CUTLASS translation
  unit is compiled separately for **`sm_90a`**.

## Implemented design (native only)
- There is no FP8-to-BF16 weight expansion. Safetensors E4M3 bytes remain one byte per weight on
  host and device; only the checkpoint's BF16 scale grid is normalized to FP32.
- `LinearWeight::Fp8Block` carries `{weight, weight_scale_inv, block}`. `Linear::forward`
  automatically dispatches it through `matmul_fp8_block`, matching the existing INT4 routing
  model.
- Q/K/V and gate/up fusion concatenates raw FP8 rows and scale-grid rows independently. Fusion
  boundaries must be aligned to the 128-row scale block.
- Runtime activation quantization is dynamic per token x 128 K values:
  BF16 `[M,K]` -> E4M3 `[M,K]` plus FP32 scale `[M,K/128]`.
- Decode (`M=1`) uses a W8A8 warp GEMV.
- Prefill (`M>1`) uses a separate SM90a CUTLASS blockwise Tensor Core GEMM. A device-side W8A8
  CUDA kernel is retained as a portable fallback; it still consumes raw FP8 weights.
- The activation and CUTLASS scratch regions come from the address-stable CUDA scope workspace;
  no allocation occurs inside forward or CUDA graph capture.

## Dispatch and storage
- `DataType::F8E4M3` and `Fp8E4m3: Dtype` make the one-byte storage dtype explicit.
- The backend port is deliberately separate from AWQ: `matmul_fp8_block<T>` accepts BF16
  activation/output, raw `Tensor<Fp8E4m3>`, FP32 scales, and `[128,128]` block metadata.
- Dense and AWQ loading paths are unchanged. FP8 and AWQ metadata are rejected if enabled
  simultaneously.
- The target checkpoint has 252 quantized projections. All have complete finite scales and aligned
  N/K dimensions; fused QKV is `[6144,2560]`, fused gate/up is `[19456,2560]`.

## Verification completed
- FP8 codec unit tests: known values, saturation, and distinct one-byte storage dtype.
- Loader tests: raw-byte preservation, fusion layout, invalid boundaries, and non-finite metadata.
- Component test: an FP8 weight automatically reaches the dedicated backend operation.
- GPU numerical tests against a CPU reference that performs the same dynamic activation
  quantization:
  - `M=1`: explicit dynamic W8A8 GEMV path.
  - `M=3, N=K=256`: explicit Hopper CUTLASS path (not fallback).
- The CUTLASS path passes CUDA graph capture, instantiate, replay, and post-replay numerical
  comparison.
- Native model parameters are approximately 4.834 GiB, about 3.383 GiB below expanding the
  projection weights to BF16.

## Remaining performance work
- Benchmark M=1 GEMV and prefill throughput against vLLM on representative batch/prompt lengths.
- Profile whether activation quantization should be fused into the GEMM prologue.
- Tune any shape threshold only from measured H200 results; correctness does not depend on it.
