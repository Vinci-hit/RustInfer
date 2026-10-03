# GGUF worker, HTTP API and web chat

A single `qwen35` GGUF file now works through the ordinary serving chain:

```text
Browser → HTTP/SSE server → ZMQ scheduler → worker Runtime → CuTe quantized kernels
```

The worker previously exposed GGUF only through the `rustinfer-gguf` diagnostic
binary. Serving now loads the same model into `Runtime` and uses its existing
KV allocator, recurrent state slots, sampling, chunked prefill, cancellation,
and EOS handling. The scheduler discovers architecture from GGUF metadata.
The server reads the embedded vocabulary/merges and Jinja chat template; no
`config.json`, `tokenizer.json` or llama.cpp runtime is required.

`infer-gguf` owns the shared reader and optional `text` feature. It has no worker
or CUDA dependency. Existing worker reader/text paths re-export its API.

## Local launch

All three processes use the root `rustinfer.toml`. Its local configuration points
to the downloaded Qwen3.8 27B UD-Q3_K_XL checkpoint, with BF16 activations,
2048 context tokens, one active sequence, 64-token prefill chunks, a 32 MiB
kernel workspace, and no graph arena. Edit `model` for another installation.
HTTP uses port **8080** (8000 was occupied on the development machine).

Build and start in separate terminals, from the repository root:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
CARGO_BUILD_JOBS=2 cargo +stable build \
  -p infer-worker --features cute-dsl --bin rustinfer-worker \
  -p infer-scheduler --bin rustinfer-scheduler \
  -p infer-server --bin rustinfer-server

target/debug/rustinfer-scheduler --config rustinfer.toml
# Next terminal (with the same CUDA library environment):
target/debug/rustinfer-worker --config rustinfer.toml
# Next terminal:
target/debug/rustinfer-server --config rustinfer.toml \
  --cors-allowed-origin http://localhost:3000 \
  --cors-allowed-origin http://127.0.0.1:3000
```

Wait for `/ready` to return 200. `scripts/start_worker.sh` also selects the
`cute-dsl` feature automatically for `.gguf` paths (its launcher builds release).
Starting a GGUF worker without that feature fails with an actionable message.

For the browser:

```bash
cd crates/infer-frontend
RUSTUP_TOOLCHAIN=stable ./dev.sh
```

Open <http://localhost:3000>. The default API address uses port 8080; previously
saved browser settings take precedence, so update **连接与设置** if necessary.
The model list and capabilities come from the running server. **深度思考** is
shown only when the service supports the GGUF template controls, and defaults
to off in the web UI. Replies stream through the existing SSE decoder.

## HTTP usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],
       "enable_thinking":false,"temperature":0,"max_tokens":64,
       "stream":true,"stream_options":{"include_usage":true}}'
```

Chat supports text system/developer/user/assistant messages, including history;
text content parts are accepted. `enable_thinking` and `reasoning_effort`
(`low`, `medium`, `high`, `xhigh`) are passed to the checkpoint template. Omitting
`enable_thinking` retains the offline runner's default (true). Unsupported roles,
images, invalid effort values, and overlong contexts return HTTP 400.

`/v1/completions` accepts raw text or IDs. Raw text honors the GGUF's BOS/EOS
insertion flags; chat does not add duplicate boundaries. Temperature/top-p/top-k,
stop strings, early EOS, usage, and cancellation reuse existing service behavior.

## Execution limits and validation

- TP1, paged block size 1; prefix caching and speculative decoding are rejected
  for GGUF. Vision/mmproj remains unconnected for this format.
- GGUF models request eager ragged attention for both prefill and decode. This
  preserves the offline runner's validated arithmetic and disables graph
  capture/warmup. Safetensors models retain their existing execution policy.
- The shipped local limits target this 16 GiB card. Larger contexts/concurrency
  need additional KV, recurrent state and scratch memory. Queued HTTP requests
  are still served in turn with one active GPU sequence.

Validated on RTX 4070 Ti Super with the real 27B checkpoint:

- HTTP greedy introduction exactly matched the offline result: 19 prompt tokens,
  26 generated tokens including EOS, normal `stop` finish.
- SSE and non-streaming text/usage agree; Unicode is intact.
- History spanning multiple prefill chunks, thinking mode, stochastic sampling,
  raw completion, stop strings, cancellation followed by slot reuse, and queued
  requests passed.
- Browser opened the actual WASM app, discovered the model, submitted a message,
  displayed a streamed answer and returned to the ready state without JS errors.

Repeat service checks against a running stack:

```bash
python3 scripts/e2e_gguf_chat.py --url http://127.0.0.1:8080
```

The script writes `/tmp/gguf-service-results.json`, including on failure. Unit
coverage also compares the serving Runtime against the offline probe across
chunked prefill, one-token continuation and reused recurrent/KV slots.
