# RustInfer Scheduler Configuration Guide

The Scheduler configuration system provides a unified way to configure the scheduler through command-line arguments or YAML/JSON configuration files.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Usage](#usage)
- [Configuration Sections](#configuration-sections)
- [Examples](#examples)
- [Validation](#validation)

## Architecture Overview

### Important: Scheduler Does NOT Handle Tokenization

The RustInfer architecture follows a clean separation of concerns:

```
┌─────────────────────┐
│   Server/Frontend   │  ← Holds Tokenizer
│  (HTTP/gRPC Layer)  │  ← String ↔ Token IDs conversion
└──────────┬──────────┘
           │ Token IDs only
           ▼
┌─────────────────────┐
│     Scheduler       │  ← NO Tokenizer
│  (Coordination)     │  ← Only Token IDs
└──────────┬──────────┘
           │ Token IDs + metadata
           ▼
┌─────────────────────┐
│      Worker         │  ← GPU Inference
│   (GPU Compute)     │  ← Forward pass
└─────────────────────┘
```

**Why `model_path` in Scheduler?**
1. Read `config.json` metadata (eos_token_id, max_position_embeddings, etc.)
2. Pass model path to Worker for loading weights
3. **NOT** for loading tokenizer (handled by Server)

## Usage

### Method 1: Command-line Arguments Only

```bash
rustinfer-scheduler \
  --model-path /models/llama3-8b \
  --worker-endpoint "ipc:///tmp/scheduler.ipc" \
  --block-size 16 \
  --total-blocks 1000 \
  --max-batch-size 256
```

### Method 2: Config File + Command-line Overrides

```bash
# Use config file with CLI overrides
rustinfer-scheduler \
  --config-file scheduler-config.yaml \
  --log-level debug \
  --max-batch-size 128
```

**Priority**: Command-line arguments > Config file > Defaults

### Method 3: YAML Configuration File

Create `scheduler-config.yaml`:

```yaml
# Network Configuration
worker_endpoint: "tcp://*:5555"
num_workers: 4

# Model Configuration
model_path: "/models/llama3-70b"
dtype: "bf16"
enable_flash_attn: true

# Memory Configuration
block_size: 16
total_blocks: 5000
gpu_memory_utilization: 0.9
enable_prefix_cache: true

# Scheduling Configuration
max_batch_size: 256
enable_preemption: true

# Parallelism (for multi-GPU)
tp_size: 4
pp_size: 1

# Logging
log_level: "info"
```

Then run:
```bash
rustinfer-scheduler --config-file scheduler-config.yaml
```

### Method 4: JSON Configuration File

Create `scheduler-config.json`:

```json
{
  "worker_endpoint": "tcp://*:5555",
  "model_path": "/models/llama3-8b",
  "block_size": 16,
  "total_blocks": 1000,
  "max_batch_size": 256,
  "log_level": "info"
}
```

## Configuration Sections

### 1. Network Configuration

Controls Worker communication settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `worker_endpoint` | string | `"ipc:///tmp/rustinfer-scheduler.ipc"` | ZeroMQ endpoint for Worker communication |
| `worker_timeout_ms` | u64 | `30000` | Worker RPC timeout in milliseconds |
| `num_workers` | usize | `1` | Number of workers to wait for |

**Endpoint Formats:**
- IPC: `"ipc:///tmp/scheduler.ipc"` (local machine, fastest)
- TCP: `"tcp://*:5555"` (network, multi-machine)
- TCP: `"tcp://192.168.1.10:5555"` (specific interface)

### 2. Model Configuration

Model loading and inference settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | **Required** | Path to model directory (contains config.json) |
| `dtype` | string | `"bf16"` | Data type: `bf16`, `fp16`, or `fp32` |
| `enable_flash_attn` | bool | `true` | Enable Flash Attention optimization |
| `custom_config` | string | `None` | Optional JSON config overrides |

**Note**: `tokenizer_path` is intentionally NOT included. Tokenization happens at the Server layer.

### 3. Memory Configuration

GPU memory management and KV Cache settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | usize | `16` | Number of tokens per block (PagedAttention) |
| `total_blocks` | usize | `1000` | Total number of GPU blocks |
| `gpu_memory_utilization` | f32 | `0.9` | GPU memory ratio for KV Cache (0.0-1.0) |
| `enable_prefix_cache` | bool | `true` | Enable RadixTree prefix caching |
| `prefix_cache_capacity` | usize | `100000` | Max tokens in prefix cache |
| `enable_cow` | bool | `false` | Enable Copy-on-Write for Beam Search |

**Memory Calculation**:
- If `total_blocks = 0`, auto-calculated from `gpu_memory_utilization`
- Memory per block: `block_size * 2 * num_layers * num_kv_heads * head_dim * dtype_size`
- Total KV Cache memory: `total_blocks * memory_per_block`

### 4. Scheduling Configuration

Request scheduling policy parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy` | string | `"continuous"` | Scheduling policy name |
| `max_batch_size` | usize | `256` | Max concurrent requests |
| `max_tokens_per_step` | usize | `4096` | Max tokens per forward pass |
| `enable_preemption` | bool | `true` | Allow preempting low-priority requests |
| `preemption_threshold` | usize | `0` | Free blocks threshold for preemption |
| `enable_swap` | bool | `false` | Enable swapping to CPU memory |
| `default_priority` | i32 | `0` | Default request priority |
| `idle_sleep_ms` | u64 | `1` | Sleep duration when idle (ms) |

**Policies**:
- `"continuous"`: vLLM-style continuous batching (default)
- `"priority"`: Priority-based scheduling (future)
- `"fair"`: Fair scheduling (future)

### 5. Parallelism Configuration

Multi-GPU parallelism settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp_size` | usize | `1` | Tensor Parallel size (model sharding) |
| `pp_size` | usize | `1` | Pipeline Parallel size (layer sharding) |

**Requirements**:
- `num_workers >= tp_size * pp_size`
- Tensor Parallel: Split model weights across GPUs
- Pipeline Parallel: Split model layers across GPUs

**Example**: Llama3-70B on 4xA100
```yaml
tp_size: 4
pp_size: 1
num_workers: 4
```

### 6. Logging Configuration

Logging and monitoring settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | string | `"info"` | Log level: trace/debug/info/warn/error |
| `log_format` | string | `"text"` | Log format: text/json |
| `log_to_file` | bool | `false` | Enable file logging |
| `log_file` | string | `"/tmp/rustinfer-scheduler.log"` | Log file path |

## Examples

### Example 1: Single GPU Development

```bash
rustinfer-scheduler \
  --model-path /models/llama3-8b \
  --total-blocks 1000 \
  --max-batch-size 64 \
  --log-level debug
```

### Example 2: Multi-GPU Production (4x A100)

```yaml
# production-4gpu.yaml
worker_endpoint: "tcp://*:5555"
num_workers: 4

model_path: "/models/llama3-70b"
dtype: "bf16"

total_blocks: 8000
block_size: 16
enable_prefix_cache: true

max_batch_size: 512
enable_preemption: true

tp_size: 4
pp_size: 1

log_level: "info"
log_to_file: true
log_file: "/var/log/rustinfer/scheduler.log"
```

```bash
rustinfer-scheduler --config-file production-4gpu.yaml
```

### Example 3: High Throughput Configuration

Focus on maximizing throughput:

```yaml
# high-throughput.yaml
model_path: "/models/llama3-8b"

# Large batch size for throughput
max_batch_size: 512
max_tokens_per_step: 8192

# Aggressive memory usage
total_blocks: 10000
gpu_memory_utilization: 0.95

# Enable prefix caching for shared prompts
enable_prefix_cache: true
prefix_cache_capacity: 200000

# Enable preemption for oversubscription
enable_preemption: true
```

### Example 4: Low Latency Configuration

Focus on minimizing latency:

```yaml
# low-latency.yaml
model_path: "/models/llama3-8b"

# Smaller batch size for faster scheduling
max_batch_size: 32
max_tokens_per_step: 1024

# Conservative memory to avoid preemption
total_blocks: 5000
gpu_memory_utilization: 0.8

# Disable preemption for predictable latency
enable_preemption: false
```

## Validation

The config system performs validation on load:

```rust
// In your code
let config = SchedulerConfig::load()?;  // Returns Error if invalid
```

**Validation Checks**:
- `block_size > 0`
- `0.0 < gpu_memory_utilization <= 1.0`
- `max_batch_size > 0`
- `tp_size > 0` and `pp_size > 0`
- `num_workers >= tp_size * pp_size`
- `model_path` is not empty
- `log_level` is valid (trace/debug/info/warn/error)

**Example Error**:
```
Error: Configuration validation failed
  Caused by: gpu_memory_utilization must be in range (0.0, 1.0]
```

## Programmatic Usage

### In Rust Code

```rust
use infer_scheduler::config::SchedulerConfig;

// Load from command-line
let config = SchedulerConfig::load()?;

// Read model metadata
let metadata = config.read_model_metadata()?;
println!("Model has {} layers", metadata.num_layers);

// Convert to internal configs
let coordinator_config = config.to_coordinator_config();
let policy_config = config.to_policy_config();
let memory_config = config.to_memory_config();
```

### From YAML File Directly

```rust
let config = SchedulerConfig::from_file("config.yaml")?;
config.validate()?;
```

## Best Practices

1. **Start with defaults**: Only override what you need
2. **Use config files for production**: Easier to version control
3. **Use CLI args for debugging**: Quick iteration
4. **Validate before deployment**: Test with `--help` and `--validate`
5. **Monitor GPU memory**: Adjust `gpu_memory_utilization` based on profiling
6. **Tune batch size**: Balance throughput vs latency
7. **Enable prefix cache**: For workloads with shared prompts
8. **Set proper timeouts**: `worker_timeout_ms` based on network latency

## Troubleshooting

### "Configuration validation failed"
Check all required fields are set, especially `model_path`.

### "Out of memory during allocation"
- Reduce `total_blocks` or `gpu_memory_utilization`
- Reduce `max_batch_size`
- Enable `enable_preemption`

### "Worker timeout"
- Increase `worker_timeout_ms`
- Check network connectivity (for TCP endpoints)
- Verify Worker is running and healthy

### "Model metadata read failed"
- Verify `model_path` points to directory with `config.json`
- Check file permissions
- Ensure model is in correct format (not GGUF)

## See Also

- [Architecture Documentation](./ARCHITECTURE.md)
- [Worker Configuration](./WORKER_CONFIG.md)
- [Server Configuration](./SERVER_CONFIG.md)
- [Performance Tuning Guide](./PERFORMANCE.md)
