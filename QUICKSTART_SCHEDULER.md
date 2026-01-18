# RustInfer Scheduler Quick Start

## 1. Basic Usage (Single GPU)

```bash
cargo run --bin rustinfer-scheduler -- \
  --model-path /path/to/llama3-8b \
  --worker-endpoint "ipc:///tmp/scheduler.ipc" \
  --log-level info
```

## 2. Using Configuration File

Create `scheduler.yaml`:
```yaml
model_path: "/path/to/llama3-8b"
worker_endpoint: "ipc:///tmp/scheduler.ipc"
block_size: 16
total_blocks: 1000
max_batch_size: 256
log_level: "info"
```

Run:
```bash
cargo run --bin rustinfer-scheduler -- --config-file scheduler.yaml
```

## 3. Multi-GPU Setup (4x GPU)

Create `scheduler-4gpu.yaml`:
```yaml
model_path: "/path/to/llama3-70b"
worker_endpoint: "tcp://*:5555"
num_workers: 4
tp_size: 4
pp_size: 1
total_blocks: 8000
max_batch_size: 512
```

Run:
```bash
cargo run --bin rustinfer-scheduler -- --config-file scheduler-4gpu.yaml
```

## 4. Development Mode (Verbose Logging)

```bash
cargo run --bin rustinfer-scheduler -- \
  --model-path /path/to/llama3-8b \
  --log-level debug \
  --max-batch-size 32
```

## Configuration Files

See:
- `scheduler-config.example.yaml` - Complete example config
- `docs/SCHEDULER_CONFIG.md` - Full configuration documentation

## Important Notes

**Scheduler does NOT handle tokenization!**
- Tokenizer belongs in Server/Frontend layer
- Scheduler only processes Token IDs
- `model_path` is for metadata reading and passing to Worker

## Next Steps

1. Start Worker: `cargo run --bin rustinfer-worker`
2. Start Scheduler: `cargo run --bin rustinfer-scheduler --config-file scheduler.yaml`
3. Start Server: `cargo run --bin rustinfer-server --config-file server.yaml`
