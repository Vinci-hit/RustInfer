# GGUF text inference

The offline `rustinfer-gguf` tool accepts text, constructs Qwen35 byte-level BPE
from the GGUF metadata, executes the file's Jinja chat template, runs the main
decoder, and decodes generated IDs to Unicode text. No separate tokenizer files
or llama.cpp runtime are needed. This extends the [stateful forward runner](GGUF_FORWARD.md).

For worker/server and browser chat, see [GGUF serving](GGUF_SERVING.md).
For same-prefix logits and greedy-output checks against an independent engine,
see [the llama.cpp comparison](GGUF_COMPARISON.md).

## Run a conversation turn

From the repository root:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
cargo +stable run -p infer-worker --features cute-dsl --bin rustinfer-gguf -- \
  --model ~/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --backend cuda --context 128 --steps 64 \
  --prompt '你好，请用一句话介绍自己。' --no-thinking
```

This command completed on the RTX 4070 Ti Super 16 GB with BF16 activations.
The rendered input had 19 tokens. Greedy generation produced 26 tokens
including EOS, stopping before the 64-token limit, and decoded to:

> 你好，我是通义千问，一个由阿里巴巴通义实验室独立开发的大语言模型，很高兴能为你提供帮助。

This verifies a complete text turn, not general language quality or numerical
parity of the full model against an independent engine. Quantized kernels and
the correctness attention dispatch are unchanged from the forward runner.

| Input / option | Behavior |
| --- | --- |
| `--prompt TEXT` | Render one user turn with the GGUF's own chat template |
| `--system TEXT` | Prepend a system message; requires `--prompt` |
| `--no-thinking` | Pass `enable_thinking=false` to the template |
| `--reasoning-effort low\|medium\|high\|xhigh` | Pass the effort to the template; requires chat and conflicts with `--no-thinking` |
| `--raw-prompt TEXT` | Completion text without a chat template; honor GGUF BOS/EOS insertion flags |
| `--token-ids LIST` | Existing raw-ID diagnostic input |
| `--backend inspect` | Default: inspect and tokenize without allocating model weights |
| `--steps N` | Maximum predictions, default 128; early stop on an end token |
| `--dump PATH` | Inspection: rendered prompt, IDs and decoded input. Inference: also generated text, finish reason, logits and diagnostics |

The three input forms are mutually exclusive. The CLI reports `eos` or `length`
as the finish reason. It prints the complete decoded result after generation;
the per-step diagnostics still show raw BPE vocabulary pieces. The configured
context must accommodate `input_tokens + steps - 1` for a forward run.

Thinking is enabled by default. The checkpoint's template decides its exact
prefix and instructions; the local checkpoint inserts an `xhigh` instruction
when no effort is specified. Use a sufficient token budget for thinking, or
`--no-thinking` for short direct responses. Rendering does not synthesize a
second BOS/EOS around the template's own boundaries.

To inspect text encoding without CUDA, including the exact rendered prompt:

```bash
cargo +stable run -p infer-worker --no-default-features --bin rustinfer-gguf -- \
  --model ~/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --prompt '你好' --no-thinking --dump /tmp/gguf-tokenization.json
```

## Tokenizer and template contract

`models::qwen3_5::gguf::text::GgufText` owns its vocabulary and template; the
reader may be dropped after construction. Its API accepts text messages with
system/developer/user/assistant roles and optional assistant reasoning content.
The CLI currently exposes a single user turn plus an optional system message.

Supported metadata is deliberately explicit: `tokenizer.ggml.model=gpt2` and
`tokenizer.ggml.pre=qwen35`. The implementation reads tokens, token types,
ranked merges, BOS/EOS flags/IDs and the chat template. It validates duplicate
tokens/merges, byte alphabet coverage, merge references and special-token IDs.
Unsupported profiles or malformed metadata fail instead of falling back to a
different tokenizer. Missing chat templates only prohibit chat rendering; raw
completion remains available.

The Qwen35 profile uses NFC normalization, a Unicode-aware split regex with
combining marks and single-digit splitting, and byte-level BPE without an
automatic leading space. This follows the
[Transformers Qwen3.5 tokenizer definition](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/tokenization_qwen3_5.py).
CONTROL tokens are skipped in displayed output; USER_DEFINED tags such as
`<think>` remain visible. Complete ID sequences are decoded together so bytes
split across token boundaries form valid UTF-8 characters.

Generation stops on GGUF EOS, optional EOT/EOM IDs, or Qwen's `<|endoftext|>`.
The underlying generated-ID report includes the stopping token. The text
display removes control tokens without stripping ordinary whitespace or
reasoning tags.

MiniJinja executes `tokenizer.chat_template` with Python string-method
compatibility, `raise_exception`, generation/thinking flags and optional
reasoning effort. Namespace assignment, reverse slicing, macros and history
formatting are covered by the real-checkpoint reference test. Template errors
are propagated before any GPU weights are loaded.

## Verification

Four small tokenizer/template unit tests and a CLI input-contract test cover
merges, digit boundaries, Unicode/NFC, control versus user-defined tokens,
BOS/EOS policy, invalid IDs/metadata, Jinja execution and error propagation.
The real-checkpoint test adds 12 exact encoding/decoding cases and 7 exact
template/prompt-ID cases, including default/low/medium/high thinking, merged
system/developer messages, and assistant history with reasoning.

Reference results are checked in at
`crates/infer-gguf/tests/fixtures/gguf/qwen35_text_reference.json`. They were
generated separately using Python Tokenizers 0.22.2 and Jinja2 3.1.6; Rust uses
Tokenizers 0.23.1 and MiniJinja. The fixture records source hashes and the exact
regex. The reference script uses the canonical Qwen35 normalization/splitting
profile with the local HF vocabulary/merges/added tokens, which it checks
against GGUF IDs. It does **not** use the local vendor `tokenizer.json` regex
as-is: that export contains a doubly escaped/older split pattern. Runtime
construction depends only on GGUF metadata and the explicit Qwen35 profile.

```bash
cargo +stable test -p infer-gguf --features text --lib text
cargo +stable test -p infer-worker --no-default-features --bin rustinfer-gguf
RUSTINFER_GGUF_MODEL="$HOME/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf" \
  cargo +stable test -p infer-gguf --features text --lib text::tests::real_tokenizer -- --ignored
```

To regenerate the reference fixture from the existing inspection manifest:

```bash
.venv/bin/python scripts/gguf_text_reference.py \
  --manifest target/gguf-inspection/model.reference.json \
  --hf-tokenizer ~/models/Qwen3.8-27B-AWQ-INT4/tokenizer.json \
  --output crates/infer-gguf/tests/fixtures/gguf/qwen35_text_reference.json
```

The diagnostic CLI remains eager, single-sequence and greedy.
[Worker/server integration](GGUF_SERVING.md) adds browser chat, streaming and
sampling. Tool calls, GGUF vision/mmproj and GGUF MTP execution remain separate work.
