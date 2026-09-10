# Weight loading with bounded double buffering

`Tensor::from_host_bytes` uses `MemoryPort::upload_bulk`. CPU and other backends
retain a synchronous fallback; CUDA preallocates two pinned host buffers and two
completion events when constructing `CudaConfig`. Default staging capacity is
2 × 8 MiB per CUDA context, independent of checkpoint size. The buffers are
reused across tensors and released with the context. They occupy host RAM, not
VRAM, and are separate from the GPU kernel/graph workspace.

For an input larger than one chunk:

1. CPU copies the first source range into slot A and enqueues its H2D copy.
2. CPU fills slot B while A's transfer can run on the GPU copy engine.
3. Before overwriting A, wait for A's completion event; enqueue alternating chunks.
4. Drain the compute stream before returning, including after a submission error.

No reader thread is required: CUDA performs the transfer asynchronously while
the CPU prepares the next chunk. For borrowed mmap ranges, accessing the next
chunk also drives filesystem page faults/readahead while the preceding H2D is
in flight. This permits overlap; it does not guarantee physical disk reads on a
warm page cache. There is still one completion boundary per tensor. Model
construction, weight ownership and the ordinary decode upload paths stay intact.
Bulk uploads are rejected during graph capture before staging or enqueueing.

Same-dtype tensors, contiguous output-row shards, and TP=1 input-column matrices
now borrow the checkpoint's bytes instead of allocating/copying a full host
vector. Dtype conversion, non-contiguous TP column gathering and fused projection
packing still use owned CPU buffers before upload; their CPU preparation is not
pipelined in this version. Total loader memory is therefore not limited to the
16 MiB staging pool. This is ordinary host-to-device loading, not GDS.

## Configuration and comparison

`RUSTINFER_BULK_UPLOAD_CHUNK_MIB` is read when creating the CUDA context:

- Unset: 8 MiB per slot.
- `0`: disable staging and use the existing synchronous upload.
- `1..=64`: chunk size in MiB; total pinned allocation is twice this size.
- Invalid values fail initialization. Pinned allocation failures are reported.

Transfers of one chunk or less use the original upload to avoid an extra copy.
No buffers, events or worker threads are allocated per chunk. A failed stream
drain makes staging unusable and retains its pinned buffers rather than freeing
a potentially active DMA source.

```bash
python3 scripts/bench_weight_loading.py \
  --model /root/models/Qwen3.5-4B --gpu 0 \
  --output target/weight-loading-comparison --repeats 3
```

The script alternates serial and double-buffered loading in isolated workers,
excludes one warm-up round by default, records the worker's weight-loading
duration and full readiness time separately,
and compares three short greedy responses. `--baseline-bin-dir` optionally adds
saved pre-change binaries to measure the combined loader change. It retains OS
caches and never drops global caches or stops unrelated GPU jobs. Therefore its
results are not controlled cold-storage measurements. The serial variant retains
the new borrowed-byte paths, isolating the additional effect of pinned staging.

The intended gain is startup/load latency and less full-size host copying, not
decode throughput or MTP acceptance. End-to-end startup also includes CUDA
initialization, model setup and graph priming outside weight upload.

## Validation

- Loader tests cover borrowing contiguous same-dtype data, TP row offsets,
  gathering non-contiguous columns, and retaining dtype conversion behavior.
- CUDA `bulk_upload` integration test covers small/empty inputs, exact chunk
  boundaries, odd tails, repeated slot reuse, immediate source overwrite/drop,
  and graph-capture rejection before destination modification.
- Existing worker/hybrid decoder and CPU/core tests cover model construction
  and runtime behavior.

## Local measurement (2026-09-10)

Qwen3.5-4B BF16, RTX 4070 Ti SUPER, WSL2. One excluded warm-up round
per variant, then three measured starts per variant with alternating order.
No concurrent builds in this final run; OS caches retained. Artifacts:
`target/weight-loading-final/results.json` and its per-run worker logs.

| Variant | Median weights loaded |
| --- | ---: |
| Saved pre-change worker | 4.91 s |
| Borrowed-byte loader, staging disabled | 2.40 s |
| Borrowed-byte loader, 2 × 8 MiB staging | 2.44 s |

The combined default change reduced this warm-cache loading interval by about
50.3%. The isolated staging comparison was about 1.7% slower (40 ms), so this
run provides **no evidence of an additional warm-cache speedup from double
buffering**. The demonstrated improvement is from avoiding full-size temporary
CPU copies; cold-storage overlap still needs a separate controlled measurement.
This does not imply faster steady-state decoding.

All three short greedy responses matched across all twelve worker starts. This
is a loading regression check, not a comprehensive model-accuracy benchmark.

Validation completed: 170 worker unit/integration tests, 43 core/CPU tests and
doctests, GPU bulk-copy test with staging both enabled and disabled, CPU-side
Clippy with warnings denied, and CUDA release worker/scheduler/server builds.
CUDA Clippy completed with existing warnings in flash-attention/scalar kernels;
no new warnings originated in the upload implementation. Python compilation and
existing script tests passed.
