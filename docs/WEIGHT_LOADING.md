# Layer read/upload pipeline

Qwen3.5 and the shared dense decoder builder use two owned host buffers. A
background reader reads layer N+1 while the model-building thread uploads layer
N. The filesystem reads directly into the buffers used as upload sources:

```text
reader thread:   file -> buffer A (layer 0) -> buffer B (layer 1) -> buffer A (layer 2)
model thread:              buffer A -> GPU          buffer B -> GPU
```

There is no application-level mmap-to-staging copy. `SafetensorsReader` uses its
existing parsed headers only to find file offsets, lengths, dtypes and shapes.
It retains file handles and issues positional reads directly into each buffer.
Normal buffered filesystem I/O may still use the OS page cache; this is not GDS
or a claim of physical zero-copy storage I/O.

`MemoryPort::alloc_host_buffer` supplies owned host memory. CUDA allocates pinned
memory with `cudaMallocHost`; CPU backends use ordinary initialized byte vectors.
Both buffers are allocated once at the beginning of the layer loop, sized to the
largest checkpoint layer selected by that builder. This is determined from model
metadata, not a hardware benchmark or a fixed chunk size. There is no production
configuration switch. Qwen3.5-4B uses 225,848,000 bytes per buffer, about 431 MiB
of host memory for both. Buffers are released when loading finishes.

## Ownership and failure handling

A bounded ready queue and a free-buffer queue transfer exclusive ownership.
The reader cannot overwrite a buffer until the consumer returns it. The current
synchronous tensor upload contract is retained: each weight upload finishes
before its source can be reused. This lets the next layer's filesystem reads
run concurrently without changing inference stream ordering or model construction.
Views borrow the current layer buffer; device tensors own their GPU allocations.

The reader completes all reads for a layer before publishing it. An I/O error is
reported to the consumer rather than publishing partial weights. On a model
construction error, both queues are disconnected before joining the reader,
including when it is waiting for a free buffer or space in the ready queue.

Contiguous same-dtype weights and row shards borrow their host bytes. Existing
CPU dtype conversion, TP column gathering, and QKV/gate-up fusion still perform
their required transformations. Those transformations can allocate owned
buffers, so total loading memory is larger than the two-buffer capacity.

Embedding, final output weights, separate vision/MTP builders and other paths
outside the integrated decoder layer loops keep their existing loading behavior.
Each TP worker currently prefetches whole checkpoint layers before selecting its
local shards; this is correct but does not reduce storage reads or pinned-host
capacity by TP size. MoE-specific builders are not integrated here.

## Validation and measurement

CPU tests verify that the next layer becomes available while the current buffer
is held, only recycled buffers are reused, tensor ranges across checkpoint shards
are correct, and the pipeline does not depend on the original mmap remaining
alive. Other tests exercise early consumer exit and file-read error propagation.
A GPU test reads a file into pinned memory on another thread, uploads those same
bytes, then overwrites/frees the source and verifies the device result.

```bash
python3 scripts/bench_weight_loading.py \
  --model /root/models/Qwen3.5-4B --gpu 0 \
  --baseline-bin-dir target/weight-upload-before-bin \
  --output target/layer-prefetch-comparison --repeats 3
```

The benchmark compares saved baseline binaries with the current default path,
excludes one warm-up round per variant, alternates order and checks three short
greedy responses. Weight-loading time and full readiness time are reported
separately. OS caches are retained; no global cache eviction or termination of
unrelated GPU jobs occurs. These results do not measure controlled cold-disk
performance or steady-state decode throughput.

### Local result (2026-09-10)

RTX 4070 Ti SUPER, WSL2, Qwen3.5-4B BF16. Artifacts:
`target/layer-prefetch-comparison/results.json` and per-run worker logs.

| Version | Median weight-loading interval |
| --- | ---: |
| Saved original worker, before loader changes | 4.78 s |
| Current layer pipeline and borrowed-byte loader | 2.34 s |

All three short responses matched across eight worker starts (one excluded
warm-up plus three measured starts per variant). The combined loading change
reduced this warm-cache interval by about 51%. This comparison includes removal
of temporary full-size CPU copies; it does not isolate the speedup of overlapping
layer reads and uploads. It is not evidence of a cold-storage speedup.

Validation: 173 worker unit/integration tests, 43 core/CPU tests and doctests,
CUDA direct-read/upload lifecycle test, CPU-side Clippy with warnings denied,
and CUDA worker/scheduler/server Release build passed. The build retains its
existing flash-attention dead-code warning.
