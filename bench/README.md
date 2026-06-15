# Bench Environment & Config

Last updated: 2026-06-15

## Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA H20 |
| GPU Driver | 535.161.08 |
| GPU Memory | 97871 MiB |

## Model

| Item | Value |
|---|---|
| Model name | Qwen3-4B-Instruct |

## RustInfer Config (`rustinfer.toml`)

| Parameter | Value |
|---|---|
| `device` | `cuda:0` |
| `host` | `0.0.0.0` |
| `port` | `8000` |
| `max_batch_tokens` | `2048` |
| `max_batch_seqs` | `32` |
| `max_model_len` | `512` |
| `paged_block_size` | `1` |
| `chunked_prefill_size` | `0` (disabled) |
| `enable_prefix_caching` | `false` |
| `mem_fraction_static` | `0.9` |
| `num_blocks` | `0` (auto-size) |
| `ignore_eos` | `true` |
| `mode` | `llm` |
| `worker_id` | `worker-0` |
| `log_level` | `info` |

**Build**: `cargo build --release` (from `/root/RustInfer`)

**Start**: Each binary (`rustinfer-server`, `rustinfer-scheduler`, `rustinfer-worker`) takes `--config rustinfer.toml`.

## vLLM Config (`start_vllm.sh`)

| Parameter | Value |
|---|---|
| `--port` | `8000` |
| `--tensor-parallel-size` | `1` |
| `--max-num-seqs` | `32` |
| `--max-model-len` | `512` |
| `--max-num-batched-tokens` | `2048` |
| `--gpu-memory-utilization` | `0.87` |
| `--no-enable-prefix-caching` | set (disabled) |
| `--block-size` | `16` |

**Start command**: `bash bench/start_vllm.sh`

## Request Distribution

### Prompt File (`bench_prompts.json`)

| Stat | Value |
|---|---|
| Total prompts | 51,906 |
| Length (chars) min | 21 |
| Length (chars) max | 798 |
| Length (chars) mean | 83 |
| Estimated tokens min | 5 |
| Estimated tokens max | 199 |
| Estimated tokens mean | ~20 |

Prompts are short typical chat queries. No shuffle — sent in file order, round-robin.

### Actual Bench Token Distribution (2026-06-15)

| Stat | RustInfer | vLLM |
|---|---|---|
| prompt_tokens min | 15 | ~15 |
| prompt_tokens max | 79 | ~79 |
| prompt_tokens mean | 28 | ~28 |
| prompt_tokens median | 25 | ~25 |
| completion_tokens min | 433 | ~433 |
| completion_tokens max | 497 | ~497 |
| completion_tokens mean | 484 | 486 |
| completion_tokens median | 488 | ~486 |

Both targets hit `max_model_len=512` limit: `prompt_tokens + completion_tokens ≈ 512`. Distribution is identical → fair comparison.

## Bench Script (`bench_online_compare.py`)

| Parameter | Value |
|---|---|
| Script | `bench/bench_online_compare.py` |
| `--tag` | `rustinfer` or `vllm` |
| `--url` | `http://127.0.0.1:8000` |
| `--duration` | `60` |
| `--concurrency` | `32` |
| Prompts file | `bench/bench_prompts.json` |
| `ignore_eos` in request | `true` (both targets) |

**Run**: `python3 bench/bench_online_compare.py --tag rustinfer` (or `--tag vllm`)

**Output**: `/tmp/bench_online_rustinfer.json` / `/tmp/bench_online_vllm.json`

## Latest Results (2026-06-15)

| Metric | RustInfer | vLLM | Ratio (RI/vLLM) |
|---|---|---|---|
| Throughput (tok/s) | 6,220 | 5,702 | 1.09x |
| Requests/s | 12.8 | 11.73 | 1.09x |
| Avg completion tokens/req | 486 | 486 | 1.00x |
| Latency P50 (s) | 2.57 | 2.78 | 0.93x |
| Latency P90 (s) | 2.62 | 2.84 | 0.92x |
| Latency P99 (s) | 2.67 | 2.92 | 0.91x |
| Per-req tok/s | 190.3 | 175.6 | 1.08x |

**Conclusion**: RustInfer outperforms vLLM by ~9% in throughput and ~8% in per-request tok/s, with ~7-9% lower latency.
