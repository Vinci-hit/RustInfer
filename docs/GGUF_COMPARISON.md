# GGUF comparison against llama.cpp

This is an independent numerical reference for the Qwen35 GGUF text decoder.
llama.cpp is built outside RustInfer, and is never a RustInfer runtime dependency.
Both executables load the same GGUF file and run sequentially on one GPU. They
exit after testing. Images, MTP, HTTP serving and concurrency are outside this test.

## Build the reference

The initial reference is pinned to llama.cpp commit
`b1ff4ca23630ac7c5a275405353ddb5c486332c9`. The harness uses its public C API and
the JSON header included in that checkout. Upstream instructions are in
[llama.cpp's build documentation](https://github.com/ggml-org/llama.cpp/blob/b1ff4ca23630ac7c5a275405353ddb5c486332c9/docs/build.md).

From the RustInfer root, with CMake, Ninja, a C++17 compiler and CUDA installed:

```bash
LLAMA_REF=/tmp/rustinfer-llama-reference
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_REF"
git -C "$LLAMA_REF" checkout b1ff4ca23630ac7c5a275405353ddb5c486332c9
cmake -S "$LLAMA_REF" -B "$LLAMA_REF/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$LLAMA_REF/build" --target llama -j 2
g++ -O3 -std=c++17 scripts/gguf_llama_reference.cpp \
  -I "$LLAMA_REF/include" -I "$LLAMA_REF/ggml/include" \
  -I "$LLAMA_REF/vendor/nlohmann" -L "$LLAMA_REF/build/bin" \
  -Wl,-rpath,"$LLAMA_REF/build/bin" -lllama -lggml -lggml-base \
  -o "$LLAMA_REF/build/bin/gguf-llama-reference"
```

Architecture `89` is for the local RTX 4070 Ti Super. The reference enables
Flash Attention, CUDA graphs by the upstream build default, all decoder layers
on GPU, one sequence, 2048 context positions, and BF16 K/V caches. llama.cpp
still uses its normal mixed/F32 arithmetic internally; this is **not** identical
to RustInfer's BF16 activations and BF16 final logits.

## Build and run RustInfer

```bash
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
CARGO_BUILD_JOBS=1 cargo +stable build --release \
  -p infer-worker --features cute-dsl --bin rustinfer-gguf
.venv/bin/python scripts/compare_gguf_llama.py \
  --model ~/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --llama-source /tmp/rustinfer-llama-reference \
  --llama-probe /tmp/rustinfer-llama-reference/build/bin/gguf-llama-reference
```

The Python harness requires NumPy. Stop other inference processes first so the
two engines can each load all decoder weights into the 16 GB card.
No model download, repository config edit, service restart or kernel change is
performed by the harness.

## What is compared

Three prompts cover Chinese chat with thinking disabled, arithmetic with
thinking disabled, and English arithmetic with low thinking effort. The model's
embedded template is rendered by RustInfer. llama.cpp independently tokenizes
the rendered text; its own chat-template renderer is not tested here.

For each prompt:

1. RustInfer generates greedily and saves every vocabulary logit.
2. llama.cpp consumes the **same prompt token IDs**, then replays RustInfer's
   generated IDs. Logits at each step therefore have the same prefix, even when
   the engines prefer different next tokens. The report includes relative L2,
   maximum absolute error, cosine, KL divergence, top-1 agreement and top-5 overlap.
3. llama.cpp clears KV and recurrent state and generates its own greedy sequence.
   The report records the common prefix and decoded outputs separately.

There is no arbitrary full-model pass/fail tolerance. Divergence in free
generation alone does not establish a bug: a small change between nearly tied
logits can change all subsequent tokens. The same-prefix statistics and the
reference's margin over RustInfer's chosen token help assess those cases.

`target/gguf-comparison/summary.json` contains compact results and provenance.
The manifest records the model SHA-256, llama.cpp commit, RustInfer commit and
dirty-tree status, commands, and sampled total GPU memory. Raw per-step reports
and process logs are kept beside it. Reports contain the entire vocabulary and
can occupy hundreds of MB; they are not committed.

## Timing limits

Both executables use release builds. Step times cover forward execution and a
full-vocabulary CPU readback; they exclude model loading, top-token selection,
JSON serialization and HTTP. Decode throughput is `(number of steps - 1) /
sum(decode step seconds)`. llama.cpp's teacher-forced run uses exactly the same
number of steps and tokens as RustInfer.

This is a single-run diagnostic measurement, without a standardized warmup.
RustInfer starts a new process per prompt; llama.cpp retains its model across
prompts. Prefill includes first-use initialization, so it is **not HTTP TTFT** or
a reliable steady-state prefill benchmark. Total GPU memory includes the desktop
and other processes, and is sampled every approximately 0.5 seconds; it is not a
precise allocator peak. Follow-up performance work should add warmup, repetitions,
fixed-length generation, longer prompts and an equivalent HTTP benchmark.

## Initial local result, 2026-09-23

Hardware: RTX 4070 Ti Super, 16 GB, WSL, driver 616.92. Model SHA-256:
`8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
RustInfer was built from `3aed782` plus the uncommitted GGUF service integration.
Both builds used their normal CUDA execution policies: eager ragged attention
for RustInfer, Flash Attention and CUDA graphs for llama.cpp. This compares the
current implementations, not isolated kernels with equal dispatch policies.

| Case | Input tokens | Compared predictions | Exact greedy IDs | RustInfer decode tok/s | llama.cpp decode tok/s | Ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Chinese introduction, thinking off | 19 | 26 | yes | 4.41 | 33.82 | 7.67× |
| `17 + 25`, thinking off | 27 | 3 | yes | 4.42 | 33.19 | 7.50× |
| `17 + 25`, low thinking | 50 | 16 | yes, truncated at 16 | 4.43 | 38.23 | 8.63× |

All three rendered prompts tokenized identically. All **45/45 same-prefix top-1
predictions** matched, and all three independent greedy token sequences matched.
The first two cases reached EOS; the thinking case stopped at the 16-token limit
while still reasoning, so this does not test completion of its final answer.

The introduction was identical in both engines:

> 你好，我是通义千问，一个由阿里巴巴通义实验室独立开发的大语言模型，很高兴能为你提供帮助。

The direct arithmetic answer was `42`. The thinking prefix in both engines was:

> The user is asking a simple arithmetic question: 17 + 25

The logits are **not numerically identical**:

| Case | Maximum relative L2 | Minimum cosine | Mean KL, llama.cpp → RustInfer |
| --- | ---: | ---: | ---: |
| Chinese | 5.31% | 0.99859 | 0.0006184 |
| Arithmetic | 6.60% | 0.99783 | 0.000002803 |
| Thinking | 14.22% | 0.99010 | 0.0012410 |

Matching top-1 on these short prompts is encouraging, but does not prove full
model parity, long-context correctness or equal model quality. Arithmetic and
precision differ between engines; this experiment does not isolate which
operations account for the logit errors.

Measured first-prefill times were 1.205/1.338/2.275 seconds in RustInfer and
0.220/0.058/0.065 seconds in llama.cpp. These have different first-use conditions
as described above and should not be treated as a steady-state prefill speedup.
The arithmetic decode sample has only two steps; the 25-step Chinese case is
a more useful starting point for profiling.

Sampled total GPU use reached 15,492 MiB for RustInfer and 14,623 MiB for llama.cpp.
The latter kept a 521 MiB model buffer CPU-mapped and reported a 11,671 MiB CUDA
model buffer; RustInfer places its embedding on CUDA too. These memory totals
include the desktop, so the difference is not an allocator-efficiency comparison.
Both executables exited after the experiment, and total GPU use returned to
2,326 MiB. The HTTP and web services remained stopped.

Raw results for this run are in
[`target/gguf-comparison/summary.json`](../target/gguf-comparison/summary.json),
with logits and process logs alongside it. The harness records binary hashes
in addition to the source commit IDs so the exact tested binaries can be identified.
