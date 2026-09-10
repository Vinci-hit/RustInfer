# Sampling and beam search

## top-k / top-p

The existing `/v1/completions` and `/v1/chat/completions` parameters now use a
GPU sampling tail for BF16/F32 CUDA logits:

```json
{"model":"Qwen3.5-4B", "prompt":"The capital of France is", "max_tokens":32,
 "temperature":0.8, "top_k":40, "top_p":0.9}
```

Positive temperature scales logits. top-k keeps the highest k tokens; 0 or -1
means unrestricted at the HTTP boundary. top-p keeps the smallest prefix whose
probability mass reaches p, **including the boundary token**, after top-k
renormalization. The retained distribution is normalized again before drawing.
Temperature 0, top-k 1 or top-p 0 selects greedy; top-p 1 disables nucleus
filtering. Ties prefer the lowest token ID.

The CUDA backend uses CUB radix sorting and prefix sums. A single vocabulary-sized
workspace is allocated at runtime startup and reused across batch rows. Logits,
filtering and the CDF stay on device; only selected IDs and scalar log-probabilities
are downloaded. Ordinary greedy still uses its existing argmax/ABC/graph path.
Stochastic decoding remains eager, not an ABC self-loop or a captured sampler.
Full vocabulary sorting is still performed, including when k is small; this is
not a specialized fast top-k kernel. CPU reference sampling uses partial selection
for top-k and converts weights in place to avoid a second vocabulary-sized vector.

MTP still requires greedy. Seed and penalty support at the HTTP boundary has not
changed. Custom samplers retain their own behavior. Unsupported device dtypes or
requests for top log-probabilities use the host reference sampler.

## Beam search

First entry points are the public Rust `BeamSession` and a standalone worker CLI:

```sh
# Uses the model/device from the shared TOML, without starting scheduler/server.
target/release/rustinfer-worker --config rustinfer.toml \
  --beam-prompt 'The capital of France is' --beam-width 4 --beam-max-tokens 32
```

The prompt is raw text completion input, without automatic chat templating.
Output JSON contains ranked beams with `text`, `token_ids`, cumulative `logprob`,
normalized `score`, and `ended_with_eos`, plus search elapsed time (excluding load).
The HTTP server does **not yet expose beam search**. It needs scheduler-owned
candidate groups, cancellation and admission accounting before that integration.
The CLI requires text input, TP=1 and MTP disabled; it runs eagerly.

Rust callers construct `BeamSession::new(model, scope, BeamSearchConfig { ... })`
and call `generate(&prompt_ids)`. The session may be reused across prompts.
`max_context` must cover prompt + `max_new_tokens`; `max_step_tokens` must be at
least the beam width. Prompt prefill is chunked to this limit.

Each candidate uses the full model log-softmax probability, without temperature,
top-k or top-p filtering. The score is cumulative log-probability divided by
`generated_length.powf(length_penalty)`. Generated length includes EOS and excludes
the prompt. The CLI uses length penalty 1; Rust callers can specify any finite,
non-negative value. EOS candidates are finalized and not expanded. Live beams are
pruned by cumulative log-probability; finished results by normalized score. Early
termination uses an optimistic bound at the maximum generated length. Width one
follows greedy, including immediate termination on EOS. Ties use token order.

Full-attention KV uses reference-counted token slots with block size 1. Read-only
prefixes are shared across children; each next token gets a new slot, and pruning
reclaims slots no longer referenced. Forking does not copy the full KV prefix.
The dedicated pool reserves `width * max_context` slots once.

Qwen3.5's GDN convolution and FP32 recurrent states must be independent per child.
A startup-allocated snapshot captures parents before scattering states into the
new row order. Duplicate parents and cyclic reorderings therefore cannot clobber
source histories. Identity mappings skip these copies, and width one does not
allocate a fork snapshot. These device copies are required by the mutable recurrent state;
state storage is not allocated per decoding step. The CPU still maintains beam
histories, block-table metadata and scores. BF16/F32 CUDA candidate selection only
downloads a bounded list of token IDs and log-probabilities, not full logits.

Beam search typically costs more compute/memory than greedy; it is a search
feature, not an inference speedup.

## Validation

- CPU hybrid decoder: width one matches greedy; width three with shared KV and
  forked GDN matches an independent oracle that recomputes each full prefix.
  Reusing a session produces the same results.
- Unit coverage: top-k/top-p composition, stable ties, full-vocabulary beam
  normalization, EOS termination, parent duplication and output-length limits.
- CUDA: Qwen-sized 248,320-token logits, top-k/top-p composition, infinite-logit
  ties, bounded candidate readout and undersized-workspace rejection.
- RTX 4070 Ti SUPER / Qwen3.5-4B: width-four, 12-token standalone search completed
  and returned four ranked continuations starting with ` Paris.`. This is a model
  smoke test, not a performance comparison or a claim of better answer quality.

The final HTTP smoke check also exercised greedy and three stochastic settings
with two concurrent request slots. The standalone width-one Qwen run returned
the same 12 tokens as the HTTP greedy check. Local reports are under
`target/sampling-smoke/`; numerical correctness does not depend on this smoke test.
