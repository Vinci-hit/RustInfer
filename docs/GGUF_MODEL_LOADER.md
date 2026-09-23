# Qwen35 GGUF model loader

`models::qwen3_5::gguf::Qwen35GgufLoader` assembles the main decoder directly
from a single GGUF file into the existing Qwen3.5 components. This supports the
local `Qwen3.8-27B-UD-Q3_K_XL.gguf`, whose `general.architecture` is `qwen35`.
The filename is not used to infer model dimensions or quantization.

```rust,ignore
use infer_worker::infrastructure::io::gguf::GgufReader;
use infer_worker::models::qwen3_5::gguf::{LoadOptions, Qwen35GgufLoader};

let reader = GgufReader::open(path)?;
let loader = Qwen35GgufLoader::new(&reader, LoadOptions { context_length: 4096 })?;
println!("{:?}", loader.report());
let model = loader.load::<half::bf16, _>(&device)?;
// model owns its device allocations; loader and reader can now be dropped.
```

## Preflight and ownership

`new` performs no device allocation. It validates architecture, dimensions,
head geometry, RoPE, recurrent-layer selection, main/MTP boundaries, every
required tensor's exact disk shape and encoding, and the small transformed
weights. Missing and unexpected main tensors are errors. Unsupported fused
combinations of dense and encoded matrices are rejected before upload.

The requested context defaults to 4096 and must not exceed the checkpoint's
context. It controls only the generated RoPE table, not a preallocated KV cache.
An explicit recurrent-layer array takes priority over full-attention interval.
The main layer count is `block_count - nextn_predict_layers`.

Weights are uploaded one tensor at a time, with source and destination owned
until upload synchronization finishes. Quantized matrices remain encoded; no
whole-matrix dense copy is created. Fusion holds separate encoded segments for
Q/K/V and gate/up, preserving each tensor's quantization type. Dense F32, F16
and BF16 checkpoints are supported as well. Activations may be F32/F16/BF16.
If `output.weight` is absent, the output projection shares the embedding's
allocation, following the GGUF tied-output convention.

`report()` distinguishes loaded main tensors from skipped MTP tensors and
reports source payload bytes and per-format counts. `device_weight_bytes<T>()`
counts persistent weights and generated RoPE after dtype conversion; it excludes
allocator overhead, backend workspace, activations, KV and GDN recurrent state.

## Architecture conversions

The storage convention was checked against the pinned upstream
[Qwen converter](https://github.com/ggml-org/llama.cpp/blob/c550d2f60bde72df19fcef1fef627895095b8ba8/conversion/qwen.py).
This is a file-format reference only: no llama.cpp library, executable or
numerical runtime is used.

| GGUF | Rust component / conversion |
| --- | --- |
| `token_embd.weight` | Encoded or dense embedding |
| `output.weight` | LM head; absent means shared embedding |
| `output_norm.weight` | Final zero-centered RMSNorm |
| `blk.i.attn_norm.weight` | Input zero-centered RMSNorm |
| `blk.i.post_attention_norm.weight` | FFN zero-centered RMSNorm |
| `blk.i.attn_q/k/v.weight` | Separate encoded Q/K/V segments; Q retains per-head query/gate packing |
| `blk.i.attn_output.weight` | Attention output projection |
| `blk.i.attn_q/k_norm.weight` | Zero-centered per-head RMSNorm |
| `blk.i.ffn_gate/up/down.weight` | Dense FFN structure with encoded weights |
| `blk.i.attn_qkv.weight` | GDN Q/K/V projection |
| `blk.i.attn_gate.weight` | GDN Z gate |
| `blk.i.ssm_alpha/beta.weight` | GDN A/B projections |
| `blk.i.ssm_conv1d.weight` | Disk `[kernel, channels]` → logical `[channels, 1, kernel]` |
| `blk.i.ssm_a` | Convert finite negative `-exp(A_log)` to `ln(-x)` in F32 |
| `blk.i.ssm_dt.bias` | GDN time-step bias |
| `blk.i.ssm_norm.weight` | Ordinary F32 gated RMSNorm scale |
| `blk.i.ssm_out.weight` | GDN output projection |

The converter already adds one to zero-centered norm weights. Loading subtracts
one in F32 **before** converting to the activation dtype; the existing norm then
adds one in F32 at execution. This avoids erasing small learned scales by rounding
`1 + w` directly to BF16. GDN's ordinary output norm is not shifted.

GGUF's GDN value heads are tiled (`k0,k1,...,k0,k1,...`), while the existing delta
primitive expects grouped repetition. The loader selects a component mode that
repeats Q/K activations using existing concatenation operations, so the delta
primitive receives equal Q/K and V head counts. Quantized V/Z/A/B/out weights,
convolution channels and recurrent state stay in GGUF order. In particular,
128-column V heads are not permuted inside 256-element quantization blocks.
This correctness implementation allocates temporary Q/K expansion buffers;
fusing that mapping into the recurrent operator is a later optimization.

## Offline tool

Inspect metadata, tensor coverage and memory estimates without allocating GPU
weights:

```bash
cargo +stable run -p infer-worker --no-default-features --bin rustinfer-gguf -- \
  --model ~/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf
```

Validate a complete GPU load (the process releases the model after reporting):

```bash
export PATH="$PWD/.venv/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
cargo +stable run -p infer-worker --features cute-dsl --bin rustinfer-gguf -- \
  --model ~/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --backend cuda --dtype bf16 --context 4096
```

`--backend cpu` assembles the same model in host RAM. CPU loading of the real
27B model requires space for all encoded weights in addition to staging and
other process memory. The default `inspect` mode only maps the file.

## Verification

The small synthetic checkpoint has both GDN and full attention, multiple K and
V heads with a nontrivial 3:1 ratio, nonzero projections, different quantization
formats, query gates, partial RoPE and an excluded MTP tensor. A separate test
path decodes weights on CPU, undoes tiled V layout in rows/columns, writes HF
Safetensors and loads it through the existing model builder. Comparisons cover
ragged two-sequence prefill followed by decode with persistent KV and GDN state,
checking all logits. Deliberately omitting head-order restoration fails parity.
The GPU parity case uses BF16, matching the paged-attention execution path.

Additional cases cover F32/F16/BF16 dense storage, norm rounding, tied encoded
storage, ownership after closing the reader, malformed metadata, invalid A
values, wrong shapes, missing/extra tensors and explicit recurrent-layer maps.
GPU tests compare the tiny loaded model against CPU and load all real main
weights before checking real embeddings after dropping the mmap.

```bash
cargo +stable test -p infer-worker --no-default-features --lib models::qwen3_5::gguf
# CUDA environment as above:
cargo +stable test -p infer-worker --features cute-dsl --lib \
  models::qwen3_5::gguf::tests::cuda_tiny -- --ignored --test-threads=1
RUSTINFER_GGUF_MODEL="$HOME/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf" \
  cargo +stable test -p infer-worker --features cute-dsl --lib \
  models::qwen3_5::gguf::tests::cuda_real -- --ignored --nocapture --test-threads=1
```

Local model preflight: 64 main layers (16 full attention + 48 GDN), 851 main
weight tensors, 12,784,388,096 source payload bytes. With BF16 auxiliaries and
4096-position RoPE: 12,779,638,272 device bytes (11.902 GiB). The 15 MTP tensors
occupy 351,008,768 bytes and are excluded.

On the RTX 4070 Ti Super, the real BF16 load with 128-position RoPE completed
in 27.37 seconds including a three-token embedding comparison. All 64 main
layers were resident together. After closing the mapping, token IDs 0, 42 and
248319 produced embeddings exactly equal to independently CPU-decoded rows.
The 4096-position figure above is the reported capacity estimate, not a claim
that KV/state/vision buffers were allocated during this test.

Validation: 9 Qwen model tests (6 new loader cases + 3 existing cases), 11 block
weight tests, 23 hybrid decoder regressions, 2 GPU tiny/operator tests and 1 real
GPU load test: **46 passed**. CPU Clippy completed with only the 3 existing
`mixed_tuning` dead-code warnings. Both CPU and CuTe-enabled CLI inspection
completed. The FP32 fused-add RMSNorm wrapper was also repaired: its previous
FFI declaration referenced a nonexistent export; it now composes existing
addition and RMSNorm operators, with an explicit numerical GPU test.

## Remaining integration

This loader provides a model object and an offline inspection/load command.
The command also supports [raw-token prefill and persistent decode](GGUF_FORWARD.md),
validated through all 64 layers of the real 27B checkpoint on a 16 GB GPU.
The offline tool also supports [GGUF tokenization and chat templates](GGUF_TEXT.md).
It does not yet connect server model selection, image preprocessing or the
separate `mmproj` vision GGUF. MTP is
reported and skipped; split GGUF, MoE, scaled RoPE and dense/quantized segments
inside one fused projection are explicitly unsupported. CUDA block-quantized
execution requires CuTe DSL support for the device target. Existing CUDA paged
attention prefill supports BF16/F16, so FP32 loading does not imply an FP32
end-to-end GPU forward. Short greedy generation and chunked-input consistency
have been checked; full-model reference parity and language quality remain
unvalidated.
