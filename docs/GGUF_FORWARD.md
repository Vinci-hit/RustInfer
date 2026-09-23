# GGUF single-sequence inference

`rustinfer-gguf` can now execute the loaded Qwen35 main decoder: quantized
embedding → all decoder layers → final norm → quantized LM head. KV and GDN
convolution/recurrent state persist across calls. The implementation uses this
repository's CPU/CUDA operators and CuTe DSL block-quantized operators, without
llama.cpp. See [the loader](GGUF_MODEL_LOADER.md) for supported GGUF tensors.

This is an eager diagnostic runner. It accepts raw token IDs or
[text with GGUF tokenization and chat templates](GGUF_TEXT.md); a serving
endpoint, images and MTP remain unconnected. Per-step vocabulary pieces in
the console are raw GGUF BPE labels; text input also prints decoded generation.

## Run

From the repository root, with the configured CUDA/CuTe environment:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
cargo +stable run -p infer-worker --features cute-dsl --bin rustinfer-gguf -- \
  --model ~/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --backend cuda --dtype bf16 --context 32 \
  --token-ids 248045,846,198,9419,0,248046,198,248045,74455,198 \
  --steps 4 --trace --verify-chunks --dump /tmp/gguf-forward.json
```

These IDs belong to this particular checkpoint and encode:

```text
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

| Option | Meaning |
| --- | --- |
| `--token-ids` | Comma-separated IDs; no special tokens are added |
| `--steps` | Maximum greedy predictions, including the prefill prediction; stop early on GGUF EOS |
| `--context` | Bound on consumed input tokens, and allocated KV/RoPE capacity |
| `--trace` | Read every layer's stream and pending residual; reject non-finite activations |
| `--verify-chunks` | Compare full prefill against tokenwise input, then compare one more continuation with the same token; reset state before actual generation |
| `--dump` | JSON containing all vocabulary logits per prediction, top candidates, optional layer traces, and comparison metrics |

`--verify-chunks` needs one context position beyond the prompt. It exits with
an error if either comparison exceeds 2% relative L2, writing the requested
report first. The report also records maximum absolute error and top-1 agreement;
with tracing, it includes per-layer last-token residual comparisons.
The final predicted token is not fed back unless another prediction is needed,
so the main run consumes at most `prompt_length + steps - 1` positions.

CPU execution is available with `--no-default-features --backend cpu`; the
real 27B model needs enough host RAM for encoded weights, states and workspace.
CUDA forward supports BF16/F16; FP32 is rejected by the CLI because existing
paged prefill attention does not support it.

## Persistent state and attention dispatch

`models::qwen3_5::gguf::probe::GgufProbe<T, D>` owns the model, execution scope,
shared forward/GDN scratch, one paged KV sequence and one state slot per GDN
layer. The loader and GGUF reader can be dropped after construction.
`step(ids, trace)` appends tokens and returns last-row logits. Capacity limits
each call's input size, while context limits total consumed tokens. `reset()`
clears recurrent state and returns to position zero; stale KV rows are excluded
by lengths and overwritten before reuse.

Invalid IDs, empty input and capacity/context overflow leave the session usable.
An execution failure may have changed caches, so further execution requires
reset. A fatal device error makes the session unusable. Tracing observes the
deferred residual without modifying model execution.

For correctness, **all calls use causal ragged attention**, including a single
new token with a cached prefix. The legacy CUDA `DecodeOnly` kernel rounds QK
products and partial output reductions in the activation dtype. On this model,
mixing that path with prefill produced a 5.75% relative L2 discrepancy in a
continuation comparison. Using ragged attention for both paths removed the
observed discrepancy. This runner bypasses that legacy dispatch; it does not
repair or change serving's decode kernel.

KV uses one token per block for simple single-sequence indexing. Allocation
scales with context, while recurrent state is fixed per sequence. Neither KV,
GDN state nor scratch is included in the loader's printed weight estimate.
This path favors correctness and diagnostics; it does not use CUDA graphs,
continuous batching or a tuned decode kernel.

## Verification on RTX 4070 Ti Super 16 GB

The local `Qwen3.8-27B-UD-Q3_K_XL.gguf` completed all 64 main layers (16 full
attention and 48 GDN), with about 11.90 GiB of weights plus caches and scratch.
BF16 activations, context 32, and the ten-token prompt above produced:

| Prediction | Token ID | Piece | Logit |
| --- | --- | --- | --- |
| 1 | 248068 | `<think>` | 43.75 |
| 2 | 198 | newline (`Ċ` in the vocabulary) | 27.375 |
| 3 | 760 | `The` | 27.5 |
| 4 | 1156 | ` user` (`Ġuser` in the vocabulary) | 36.25 |

All 248,320 logits per step and all 64 layer traces were finite. Full-prompt
versus tokenwise execution matched exactly for both prompt logits and the
next continuation, including the last residual at every layer. These are
internal consistency checks, not full-model numerical parity against an
independent engine or an assessment of language quality.

One warm-cache run loaded weights/state in 18.75 s, executed ten-token prefill
in 0.878 s and subsequent steps in 0.239–0.242 s. Step times include uploads,
GPU launches, readbacks and layer tracing; they are diagnostic wall times,
not optimized serving throughput measurements.

Small-model regressions additionally cover CPU/GPU agreement, shared scratch,
chunked suffixes, continuation, reset, reader ownership, invalid inputs,
context exhaustion and recovery requirements after non-finite activations:

```bash
cargo +stable test -p infer-worker --no-default-features --lib \
  models::qwen3_5::gguf::tests::probe
# CUDA environment as above:
cargo +stable test -p infer-worker --features cute-dsl --lib \
  models::qwen3_5::gguf::tests::cuda_probe -- --ignored --test-threads=1
```

GGUF tokenizer/chat-template support and text decoding are now available via
`--prompt` and `--raw-prompt`; see [text inference](GGUF_TEXT.md). The next
integration step is connection to the worker/server model-selection path.
Vision additionally requires the separate `mmproj` loader and image pipeline.
