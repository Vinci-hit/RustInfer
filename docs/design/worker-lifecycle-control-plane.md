# Worker Lifecycle Control Plane Design

## Goal

Modernize RustInfer Worker startup and lifecycle management while preserving the current Worker-local continuous batching hot path.

## Non-Goals

- Do not implement scheduler-driven decode.
- Do not make the Scheduler send per-step decode commands.
- Do not replace the Data Plane in the first four phases.
- Do not require multi-card execution in the first implementation.

## Architecture

RustInfer keeps two communication paths:

1. Control Plane: ZMQ + MessagePack lifecycle protocol.
2. Data Plane: existing ZMQ + MessagePack inference protocol.

The Scheduler owns Worker readiness, model assignment, request lifecycle, cancel, drain, and Worker Group state.

The Worker owns model loading, profiling, runtime allocation, warmup, graph capture, active decode state, and the autonomous decode loop.

## Phase 1: Handshake and Ready

Status: implemented as the first control-plane slice.

Worker still accepts model-related CLI arguments and loads the model locally, but startup becomes observable and gated.

### New messages

- `WorkerHello`
- `SchedulerHello`
- `WorkerProgress`
- `WorkerReady`
- `WorkerError`
- `Heartbeat`

### Worker Lifecycle states

- `Spawned`
- `Connecting`
- `Registered`
- `LoadingModel`
- `ProfilingMemory`
- `AllocatingRuntime`
- `Warmup`
- `Ready`
- `Running`
- `Draining`
- `Error`
- `Stopped`

### Scheduler behavior

The Scheduler must not send Data Plane batches until the Worker has sent `WorkerReady`.

## Phase 2: Scheduler-issued LoadModel

Worker becomes a Worker agent. Model loading is initiated by the Scheduler.

### New command

`LoadModel` includes:

- `model_instance_id`
- `model_path`
- `model_type`
- `device`
- `max_batch_tokens`
- `max_batch_seqs`
- `max_model_len`
- `mem_fraction_static`
- `tp_rank`
- `tp_size`
- `pp_rank`
- `pp_size`

Compatibility rule: if `rustinfer-worker --model ...` is still provided, the Worker uses that CLI model assignment and reports `model_instance_id = "default"`. If no `--model` is provided, the Worker waits for Scheduler `LoadModel`.

### Ready report

`WorkerReady` includes capacity:

- `max_batch_tokens`
- `max_batch_seqs`
- `max_running_requests`
- `max_total_kv_tokens`
- `free_mem_before_load`
- `free_mem_after_load`
- `weight_mem_usage`
- `workspace_mem_usage`
- `graph_mem_usage`

## Phase 3: Cancel, Drain, and Active Request Table

The Scheduler maintains an Active Request Table as a lifecycle record only. It does not schedule decode steps.

### Active Request Table fields

- `request_id`
- `model_instance_id`
- `worker_id`
- `kv_slot`
- `status`
- `prompt_len`
- `generated_tokens`
- `max_tokens`

### New worker commands

These are carried over the existing Data Plane command channel so the Worker can process them at safe step boundaries without adding Scheduler-driven decode.

- `CancelRequest`
- `CancelAck`
- `DrainWorker`
- `DrainAck`
- `UnloadModel`

### Cancel behavior

The Scheduler sends `CancelRequest { request_id }`.

The Worker removes the request from pending prefill or active decode at a safe step boundary, invalidates runtime metadata if required, releases the slot, and returns `CancelAck`.

## Phase 4: Worker Group

Introduce Worker Group as a logical abstraction for future multi-card support.

Single-card deployments use a Worker Group with one rank.

A Worker Group becomes ready only when every rank is ready. Effective capacity is the minimum capacity reported by its ranks.

Current behavior:

- `rustinfer-scheduler` wraps the single `WorkerReady` response into `WorkerGroup::from_single_ready`.
- `SchedulerEngine` owns a `WorkerGroup` instead of treating a Worker as the direct scheduling unit.
- `rustinfer-serve` starts only rank 0 for now. If multiple devices are passed, extra devices are logged and ignored until multi-rank Worker Group support lands.

## Explicitly Excluded Phase

Scheduler-driven decode is excluded. Autonomous Decode remains the execution model.
