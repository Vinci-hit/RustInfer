# InferEngine Documentation

A high-performance, production-ready LLM inference engine built in Rust, featuring **RadixAttention** for efficient KV cache utilization, **high-concurrency request processing**, and **O(1) LRU eviction** for optimal memory management.

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Data Flow](#data-flow)
- [API Reference](#api-reference)
- [Quick Start](#quick-start)
- [Advanced Topics](#advanced-topics)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

The `infer-engine` crate provides a complete inference engine for large language models (LLMs) with the following key features:

- **RadixAttention**: Prefix-sharing KV cache that maximizes cache hit rates by reusing computed KV values across requests with common prefixes
- **High Concurrency**: Fine-grained locking allows multiple requests to be processed in parallel
- **O(1) LRU Eviction**: Fast eviction algorithm with constant time complexity per operation
- **GPU Acceleration**: CUDA support for efficient model execution
- **Namespace Isolation**: Support for multi-tenant deployments and LoRA adapters
- **Production Ready**: Comprehensive metrics, error handling, and monitoring

### Use Cases

- **Chatbots and Assistants**: Handle many concurrent conversations efficiently
- **Code Generation**: Cache common code patterns and boilerplate
- **Document Analysis**: Process multiple documents with overlapping content
- **Multi-Tenant Services**: Isolate different users or organizations

---

## Design Philosophy

### 1. Maximum Throughput Through Fine-Grained Locking

The engine uses carefully designed locking strategies to minimize contention:

```
Resource             | Lock Type      | Concurrency
---------------------|----------------|------------------
RadixCache           | RwLock         | High (N readers)
KVCachePool          | RwLock         | High (N readers)
Model (GPU)          | Mutex          | Serialized (1 writer)
Request Queue        | Mutex          | Short-duration
Concurrency Limit    | Semaphore       | Resource-limited
```

- **Read-Heavy Resources**: Use `RwLock` to allow concurrent reads
- **Serialized Resources**: Use `Mutex` when GPU execution must be serialized
- **Short-Lived Locks**: Release locks as quickly as possible

### 2. Cache Efficiency Through Prefix Sharing

RadixAttention allows multiple requests to share KV cache for common prefixes:

```
Request 1: "The quick brown fox jumps over the lazy dog"
Request 2: "The quick brown fox jumps over the moon"
Request 3: "The quick brown fox runs fast"

Shared prefix: "The quick brown fox" → Cached once, used 3 times
```

This dramatically reduces computation for:
- System prompts reused across conversations
- Code completion with common headers
- Document analysis with overlapping sections

### 3. Production Readiness

The engine is designed for production deployment:

- **Metrics Collection**: Comprehensive timing and cache statistics
- **Error Handling**: Graceful failure recovery with detailed error messages
- **Monitoring Support**: Built-in metrics endpoints for observability
- **Configurable Limits**: Bounded queues, memory limits, and concurrency controls

### 4. Extensibility

The architecture supports pluggable components:

- **Eviction Policies**: LRU, LFU, FIFO, custom strategies
- **Memory Pools**: CPU, CUDA, and custom backends
- **Model Implementations**: Any model implementing `CacheAwareModel`

---

## Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZMQ Server (ROUTER)                            │
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │  Receiver   │     │   Sender    │     │  Monitor    │                │
│  │  (Thread)   │ --> │  (Thread)   │     │  (Async)    │                │
│  └──────┬──────┘     └──────▲──────┘     └─────────────┘                │
│         │                   │                                            │
│         ▼                   │                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Async Request Handlers                        │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │ Task 1  │  │ Task 2  │  │ Task 3  │  │ Task N  │   (tokio)   │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘             │    │
│  │       └─────────────┴─────────────┴─────────────┘                │    │
│  │                          │                                       │    │
│  └──────────────────────────┼───────────────────────────────────────┘    │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      InferenceEngine                             │    │
│  │                                                                  │    │
│  │  ┌────────────────────┐    ┌────────────────────┐               │    │
│  │  │  SharedRadixCache  │    │  SharedKVCachePool │               │    │
│  │  │   (Arc<RwLock>)    │    │   (Arc<RwLock>)    │               │    │
│  │  └─────────┬──────────┘    └─────────┬──────────┘               │    │
│  │            │                          │                          │    │
│  │            │     CacheInstruction     │                          │    │
│  │            └──────────┬───────────────┘                          │    │
│  │                       ▼                                          │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │  Model (Arc<Mutex<Llama3>>)                              │  │    │
│  │  │  - prefill_with_cache()                                  │  │    │
│  │  │  - decode_with_cache()                                   │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Radix Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RadixCache                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                       Radix Tree                                │ │
│  │                                                                  │ │
│  │  [root] ─┬─ "The quick brown fox" (lock_ref:3) ─┬─ " jumps"  │ │
│  │          │     [0,1,2,3,4,5,6,7,8,9,10,11,12]   │  → [13,14] │ │
│  │          │                                      └─ " runs"    │ │
│  │          │                                         → [13]     │ │
│  │          │                                                      │ │
│  │          └─ "Hello world" → [20,21,22,23,24,25,26]           │ │
│  │                                                                  │ │
│  │  Operations: match_prefix, insert, split_node, evict            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              │ token indices                         │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    TokenToKVPool                                │ │
│  │                                                                  │ │
│  │  K cache: [num_layers, max_tokens, kv_dim]                      │ │
│  │  V cache: [num_layers, max_tokens, kv_dim]                      │ │
│  │                                                                  │ │
│  │  free_slots: [100, 101, 102, 103, ...]                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. InferenceEngine (`engine.rs`)

The main orchestrator for inference requests.

#### Key Features

- **Token Semaphore**: Limits concurrent requests to prevent resource exhaustion
- **Multi-Phase Processing**: Separates tokenization, cache preparation, inference, and completion
- **Parallel Batch Processing**: Processes multiple requests simultaneously
- **Thread-Safe Statistics**: Tracks request metrics in real-time

#### Structure

```rust
pub struct InferenceEngine {
    model: Arc<Mutex<Llama3>>,              // Model instance
    radix_cache: SharedRadixCache,           // Prefix-sharing cache
    kv_pool: SharedKVPool,                   // KV memory pool
    request_queue: Arc<Mutex<VecDeque<...>>>, // Pending requests
    concurrent_limit: Arc<Semaphore>,        // Concurrency control
    config: EngineConfig,                    // Configuration
    stats: Arc<RwLock<EngineStats>>,         // Statistics
}
```

### 2. RadixCache (`kv_cache/`)

A prefix-sharing cache based on radix trees.

#### Key Features

- **Prefix Sharing**: Multiple requests share cached KV values for common prefixes
- **Node Splitting**: Dynamic tree restructuring for partial matches
- **Reference Counting**: Lock propagation protects active prefixes from eviction
- **O(1) Eviction**: LRU list enables constant-time eviction

#### Core Types

```rust
// Tree node storing token segments and KV indices
pub struct TreeNode {
    children: HashMap<i32, Box<TreeNode>>,
    parent: Option<*mut TreeNode>,
    key: Vec<i32>,              // Token segment
    value: Option<Vec<usize>>,  // KV pool indices
    lock_ref: i32,              // Reference count
    last_access_time: Instant,  // LRU tracking
    hit_count: u64,            // LFU tracking
}

// Radix key with optional namespace
pub struct RadixKey {
    tokens: Vec<i32>,
    extra_key: Option<String>,  // Namespace for isolation
}
```

### 3. ZMQ Server (`zmq_server.rs`)

High-concurrency server using ZMQ ROUTER pattern.

#### Architecture

- **Dedicated ZMQ Thread**: Handles I/O operations
- **Async Task per Request**: True parallelism with tokio
- **Identity Mapping**: Routes responses to correct clients
- **Metrics Task**: Periodic statistics reporting

### 4. TokenToKVPool (`kv_cache/token_pool.rs`)

Memory pool for KV cache storage.

#### Features

- **Token-Based Allocation**: Flexible memory management
- **CPU and CUDA Backends**: Support for different storage types
- **Free List Management**: Efficient reuse of released slots

---

## Data Flow

### Complete Request Lifecycle

```
1. Request Arrival (ZMQ)
   ↓
2. Create InferenceRequest (protocol parsing)
   ↓
3. Enqueue in Request Queue (Mutex<VecDeque>)
   ↓
4. Process Request (parallel execution)

   ┌─────────────────────────────────────┐
   │ Phase 1: Tokenization              │
   │ - Lock Model                        │
   │ - Encode prompt → tokens           │
   │ - Release Model Lock               │
   └─────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────┐
   │ Phase 2: Cache Preparation          │
   │ - Match prefix in RadixCache        │
   │ - Allocate new token slots         │
   │ - Lock the matched prefix           │
   │ - Create CacheInstruction           │
   └─────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────┐
   │ Phase 3: Prefill Inference         │
   │ - Lock Model & KV Pool             │
   │ - prefill_with_cache()             │
   │ - Get first generated token        │
   └─────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────┐
   │ Phase 4: Decode Loop               │
   │ For each token to generate:        │
   │   - Check EOS condition            │
   │   - Allocate new slot in cache     │
   │   - Create decode instruction      │
   │   - decode_with_cache()            │
   │   - Track all indices              │
   └─────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────┐
   │ Phase 5: Cache Completion          │
   │ - Insert new tokens to cache       │
   │ - Unlock the prefix                │
   │ - Release locks                    │
   └─────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────┐
   │ Phase 6: Response                  │
   │ - Decode generated tokens to text  │
   │ - Build InferenceResponse         │
   │ - Send via ZMQ                    │
   └─────────────────────────────────────┘
   ↓
5. Client receives response
```

### Cache Operations Detail

#### Prefix Matching

```
Request tokens: [A, B, C, D, E, F]
Cache tree:
  [root]
    └── [A, B, C] → indices [0, 1, 2]
          └── [D, E] → indices [3, 4]

Match result:
  - Matched: [A, B, C, D, E] → indices [0, 1, 2, 3, 4]
  - New: [F] → needs new slot
  - Cached length: 5
```

#### Node Splitting

```
Before: node.key = [A, B, C, D], value = [0, 1, 2, 3]
Insert key [A, B, X, Y] (matches first 2 tokens)

After split at position 2:
  node.key = [A, B], value = [0, 1]
    ├── child1.key = [C, D], value = [2, 3]  (original suffix)
    └── child2.key = [X, Y], value = [4, 5]  (new suffix)
```

#### Lock Propagation

```
Request uses prefix "The quick fox"
Tree:
  [root]
    └── "The quick fox" (lock_ref: 1)
          └── " jumps" (lock_ref: 1)  ← Propagated!

When request completes, both nodes are decremented.
Node won't be evicted while lock_ref > 0.
```

---

## API Reference

### Engine APIs

#### Creating an Engine

```rust
use infer_engine::{InferenceEngine, EngineConfig};

let config = EngineConfig {
    max_batch_size: 32,
    max_queue_size: 256,
    schedule_interval_ms: 1,
    max_concurrent_requests: 64,
    max_cache_tokens: 65536,
    num_layers: 28,
    kv_dim: 256,
};

let engine = InferenceEngine::new(model, config)?;
```

#### Processing a Request

```rust
use infer_protocol::InferenceRequest;

let request = InferenceRequest {
    request_id: "req-001".to_string(),
    prompt: "Once upon a time".to_string(),
    max_tokens: 100,
    temperature: 0.8,
    top_p: 0.9,
    top_k: 50,
    stop_sequences: vec![],
    stream: false,
    priority: 0,
};

let response = engine.process_request(request).await?;
```

#### Getting Statistics

```rust
let stats = engine.engine_stats().await;
println!(
    "Total requests: {}, Completed: {}, Hit rate: {:.1}%",
    stats.total_requests,
    stats.completed_requests,
    stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0
);

let cache_stats = engine.cache_stats()?;
println!("Cache usage: {} / {} tokens",
    cache_stats.total_cached,
    cache_stats.total_capacity
);
```

### Cache APIs

#### Creating a Cache

```rust
use infer_engine::kv_cache::{new_shared_cache, RadixCacheConfig};

let config = RadixCacheConfig {
    max_tokens: 65536,
    num_layers: 28,
    kv_dim: 256,
    dtype_size: 2,  // BF16 = 2 bytes
    disable: false,
};

let cache = new_shared_cache(config)?;
```

#### Using Cache Directly

```rust
use infer_engine::kv_cache::RadixKey;

// Prepare a request
let key = RadixKey::new(vec![1, 2, 3, 4, 5]);
let result = cache.prepare_request(&key, 10)?;

println!("Cached {} tokens, allocated {} new slots",
    result.cached_len,
    result.new_indices.len()
);

// ... process request with result.matched_indices and result.new_indices ...

// Complete the request
cache.complete_request(
    result.handle,
    Some(&key),
    Some(result.all_indices())
)?;
```

#### Working with Namespaces

```rust
// Create namespace-isolated cache entries
let key1 = RadixKey::with_namespace(
    vec![1, 2, 3],
    "tenant-1".to_string()
);

let key2 = RadixKey::with_namespace(
    vec![1, 2, 3],
    "tenant-2".to_string()
);

// These won't share cache even though tokens are identical
```

### Server APIs

#### Starting a ZMQ Server

```rust
use infer_engine::{run_zmq_server, ZmqServerConfig};

let config = ZmqServerConfig {
    max_concurrent_handlers: 128,
    zmq_recv_timeout_ms: 5,
    incoming_buffer_size: 1024,
};

run_zmq_server(engine, "tcp://0.0.0.0:5555", config).await?;
```

#### Client Communication

```rust
use zmq::{Context, ROUTER};
use infer_protocol::{EngineRequest, InferenceRequest};
use rmp_serde;

// Create ZMQ dealer socket
let context = Context::new();
let socket = context.socket(DEALER)?;
socket.connect("tcp://localhost:5555")?;

// Send request
let request = EngineRequest::Inference(InferenceRequest { /* ... */ });
let data = rmp_serde::to_vec(&request)?;
socket.send(&data, 0)?;

// Receive response
let response_data = socket.recv_bytes(0)?;
let response = rmp_serde::from_slice::<EngineResponse>(&response_data)?;
```

---

## Quick Start

### Basic Example

```rust
use infer_engine::{InferenceEngine, EngineConfig};
use infer_core::model::llama3::Llama3;
use infer_protocol::InferenceRequest;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let model = Llama3::from_file("path/to/model")?;

    // Create engine
    let config = EngineConfig::default();
    let engine = InferenceEngine::new(model, config)?;

    // Create request
    let request = InferenceRequest {
        request_id: "example".to_string(),
        prompt: "Explain quantum computing".to_string(),
        max_tokens: 200,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        stop_sequences: vec![],
        stream: false,
        priority: 0,
    };

    // Process request
    let response = engine.process_request(request).await?;

    println!("Generated: {}", response.text.unwrap());
    println!("Tokens: {}, Time: {:.2}ms",
        response.num_tokens,
        response.metrics.prefill_ms + response.metrics.decode_ms
    );

    Ok(())
}
```

### Running the Server

```rust
use infer_engine::{run_zmq_server, InferenceEngine, EngineConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = Llama3::from_file("path/to/model")?;
    let engine = InferenceEngine::new(model, EngineConfig::default())?;

    let config = infer_engine::ZmqServerConfig::default();

    run_zmq_server(engine, "tcp://0.0.0.0:5555", config).await?;

    Ok(())
}
```

### Example: Multiple Concurrent Requests

```rust
use std::sync::Arc;
use infer_engine::InferenceEngine;
use infer_protocol::InferenceRequest;

async fn process_concurrent(
    engine: Arc<InferenceEngine>,
    prompts: Vec<String>,
) -> Vec<String> {
    let mut handles = vec![];

    for (i, prompt) in prompts.into_iter().enumerate() {
        let engine = engine.clone();
        let handle = tokio::spawn(async move {
            let request = InferenceRequest {
                request_id: format!("req-{}", i),
                prompt,
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                top_k: 50,
                stop_sequences: vec![],
                stream: false,
                priority: 0,
            };

            engine.process_request(request).await
        });
        handles.push(handle);
    }

    let mut results = vec![];
    for handle in handles {
        if let Ok(Ok(resp)) = handle.await {
            results.push(resp.text.unwrap_or_default());
        }
    }
    results
}
```

---

## Advanced Topics

### Custom Eviction Policies

Create custom eviction strategies by implementing the `EvictionPolicy` trait:

```rust
use infer_engine::kv_cache::eviction::EvictionPolicy;

struct CustomEvictionPolicy {
    // Your policy state
}

impl EvictionPolicy for CustomEvictionPolicy {
    fn select_eviction_candidates(
        &mut self,
        tokens_to_evict: usize,
        nodes: &mut Vec<*mut TreeNode>,
    ) -> Vec<*mut TreeNode> {
        // Implement your custom logic
        vec![]
    }

    fn on_access(&mut self, node: &mut TreeNode) {
        // Update access statistics
    }

    fn on_insert(&mut self, node: &mut TreeNode) {
        // Handle new node insertion
    }
}
```

### Custom Memory Pools

Implement the `TokenToKVPool` trait for custom backends:

```rust
use infer_engine::kv_cache::TokenToKVPool;

struct CustomTokenPool {
    // Your pool implementation
}

impl TokenToKVPool for CustomTokenPool {
    fn alloc(&mut self, num_tokens: usize) -> Option<Vec<usize>> {
        // Allocate slots
    }

    fn free(&mut self, indices: &[usize]) {
        // Free slots
    }

    fn get_k_cache(&self) -> &[u8] {
        // Return K cache data
    }

    fn get_v_cache(&self) -> &[u8] {
        // Return V cache data
    }
}
```

### Streaming Responses

For streaming inference, modify the decode loop:

```rust
async fn stream_request(
    engine: Arc<InferenceEngine>,
    request: InferenceRequest,
    mut tx: mpsc::Sender<String>,
) -> anyhow::Result<()> {
    let mut generated_tokens = vec![];
    let mut model = engine.model.lock().await;

    // Prefill phase (same as before)
    let current_token = /* prefill logic */;
    generated_tokens.push(current_token);

    // Streaming decode loop
    for _ in 0..request.max_tokens {
        let next_token = /* decode logic */;

        generated_tokens.push(next_token);

        // Stream the new token
        if let Some(text) = model.tokenizer().decode(&[next_token])? {
            tx.send(text).await?;
        }
    }

    Ok(())
}
```

---

## Performance Tuning

### Configuration Parameters

| Parameter | Description | Default | Tuning Guide |
|-----------|-------------|---------|--------------|
| `max_concurrent_requests` | Max concurrent requests | 64 | Increase if GPU underutilized, decrease if OOM |
| `max_batch_size` | Batch size for parallel processing | 32 | Match GPU memory constraints |
| `max_cache_tokens` | KV cache capacity | 65536 | Higher = better hit rate, more memory |
| `schedule_interval_ms` | Queue processing interval | 1 | Lower = lower latency, higher CPU usage |
| `max_queue_size` | Request queue capacity | 256 | Increase for bursty traffic |

### Memory Calculation

Estimated memory usage:

```
KV Cache Size = max_cache_tokens × num_layers × kv_dim × 2 × dtype_size

For default config:
= 65536 × 28 × 256 × 2 × 2 bytes
= ~1.8 GB

Plus model weights (typically 14 GB for 7B models)
```

### Bottleneck Analysis

Check metrics to identify bottlenecks:

```rust
let stats = engine.engine_stats().await;
let cache_stats = engine.cache_stats()?;

println!("Avg queue time: {:.2}ms", stats.avg_queue_time_ms);
println!("Avg prefill time: {:.2}ms", stats.avg_prefill_time_ms);
println!("Avg decode time: {:.2}ms", stats.avg_decode_time_ms);
println!("Cache hit rate: {:.1}%", cache_stats.hit_rate() * 100.0);
println!("Concurrent requests: {}", engine.concurrent_request_count());
```

- **High queue time**: Increase `max_concurrent_requests`
- **High prefill time**: Check cache hit rate, increase `max_cache_tokens`
- **High decode time**: May be GPU-bound, consider batching

### Optimizing Cache Hit Rate

1. **Use consistent system prompts**: Share prefixes across conversations
2. **Enable caching for code**: Code has high repetition patterns
3. **Tune cache size**: Larger cache = higher hit rate (diminishing returns)
4. **Group similar requests**: Process requests with similar prompts together

---

## Troubleshooting

### Common Issues

#### Out of Memory

**Symptoms**: Process crashes with OOM

**Solutions**:
```rust
// Reduce cache size
config.max_cache_tokens = 32768;  // Half default

// Reduce concurrent requests
config.max_concurrent_requests = 32;

// Reduce batch size
config.max_batch_size = 16;
```

#### High Latency

**Symptoms**: Requests taking too long to complete

**Diagnose**:
```rust
let stats = engine.engine_stats().await;
println!("Queue time: {:.2}ms", stats.avg_queue_time_ms);
println!("Concurrent: {}", engine.concurrent_request_count());
```

**Solutions**:
- If queue time high: Increase `max_concurrent_requests`
- If queue time low, prefill/decode high: GPU bottleneck
- Consider multiple model replicas with load balancing

#### Low Cache Hit Rate

**Symptoms**: Cache hits < 30%

**Diagnose**:
```rust
let stats = engine.cache_stats()?;
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
println!("Nodes: {}", stats.node_count);
```

**Solutions**:
- Check if prompts have common prefixes
- Increase `max_cache_tokens`
- Group similar requests
- Consider namespace usage (different namespaces don't share)

#### Lock Contention

**Symptoms**: Low throughput despite available resources

**Diagnose**:
```rust
let concurrent = engine.concurrent_request_count();
println!("Active requests: {}", concurrent);
```

**Solutions**:
- Monitor lock acquisition time
- Reduce lock hold duration
- Consider request batching

### Debug Logging

Enable debug logging for detailed tracing:

```rust
use tracing_subscriber::{EnvFilter, fmt};

fn init_logging() {
    let filter = EnvFilter::from_default_env()
        .add_directive("infer_engine=debug".parse().unwrap())
        .add_directive("infer_core=debug".parse().unwrap());

    fmt().with_env_filter(filter).init();
}
```

### Monitoring

Use the built-in metrics endpoint:

```rust
// Query engine metrics
let request = EngineRequest::MetricsQuery;

// Response contains:
// - Total/completed/failed requests
// - Total tokens generated
// - Cache hit rate
// - Queue size
// - Concurrent requests
```

---

## Glossary

- **KV Cache**: Key-Value cache storing attention keys and values for each token position
- **RadixAttention**: Prefix-sharing KV cache using radix tree data structure
- **Prefill**: Processing the prompt tokens (computes KV for each)
- **Decode**: Generating tokens one at a time (appends to cache)
- **Prefix Sharing**: Multiple requests reusing KV cache for common prefixes
- **Lock Propagation**: Reference counting that protects ancestor nodes
- **Namespace**: Isolation key for multi-tenant or LoRA scenarios
- **LRU**: Least Recently Used eviction policy
- **LFU**: Least Frequently Used eviction policy

---

## Contributing

When extending the engine, keep these principles in mind:

1. **Maintain thread safety**: Use appropriate locks and Arc wrappers
2. **Minimize lock contention**: Release locks as early as possible
3. **Add metrics**: Track new operations in EngineStats
4. **Write tests**: Cover concurrent access patterns
5. **Update documentation**: Keep this README in sync

---

## License

This crate is part of the RustInfer project. See LICENSE for details.
