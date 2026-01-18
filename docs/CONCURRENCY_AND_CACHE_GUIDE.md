# RustInfer High-Concurrency KV Cache System Developer's Guide

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Key Components](#key-components)
5. [Usage Patterns](#usage-patterns)
6. [Concurrency Model](#concurrency-model)
7. [Performance Characteristics](#performance-characteristics)
8. [Troubleshooting](#troubleshooting)

---

## Overview

RustInfer implements a high-performance, high-concurrency inference engine for large language models (LLMs). The system is optimized to handle multiple simultaneous user requests while efficiently reusing cached computation.

### What Problem Does This Solve?

When serving an LLM, the same prompt prefixes often appear across different requests:
- "Translate from English to Spanish:" (appears in many requests)
- "Q: How are you feeling? A:" (conversation templates)
- Context from the same document (summarization tasks)

**Without caching:** Each request recalculates KV cache for the entire prompt
**With RadixAttention:** Requests share cached computation for common prefixes

### Why High Concurrency?

Modern GPUs can handle multiple inference tasks, but synchronizing them is challenging. RustInfer uses fine-grained locking to allow:
- Multiple requests to check the cache simultaneously
- GPU forward passes to happen one at a time (necessary for correctness)
- Minimal lock contention, maximum throughput

---

## Core Concepts

### What is KV Cache?

In transformer models, the Key (K) and Value (V) tensors for past tokens are stored in memory rather than recomputed. This dramatically speeds up inference.

```
Example: Generating "Hello, World!"

Token 0: "Hello"
  → Compute K, V for "Hello" → Store in cache

Token 1: ","
  → Reuse K,V for "Hello" (from cache)
  → Compute new K,V for "," → Add to cache

Token 2: "World"
  → Reuse K,V for "Hello" and "," (from cache)
  → Compute new K,V for "World" → Add to cache
```

### What is Prefix Sharing?

When multiple requests have overlapping prompts, they can share the cached KV values for the common prefix.

```
Request A: "Translate English to Spanish: Hello"
Request B: "Translate English to Spanish: Goodbye"
                    ↑ common prefix ↑

Both requests can use the same cached KV for the prefix.
Saves computation for the common part!
```

### What is RadixAttention?

RadixAttention is a prefix-sharing technique that stores cached KV values in a **radix tree** (a compressed tree structure). The tree enables:

1. **Efficient prefix matching** - Find common prefixes quickly
2. **Memory reuse** - Multiple requests point to the same cached values
3. **Automatic cleanup** - Least-recently-used entries are evicted when memory is full

---

## Architecture

### System-Level Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Client Requests (via ZMQ)                          │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │ (msgpack-encoded InferenceRequest)
                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                   High-Concurrency ZMQ Server                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Each request spawned as async task for true parallelism            │  │
│  │ Non-blocking request/response handling                             │  │
│  └──────────────────────────┬──────────────────────────────────────────┘  │
│                             │
└─────────────────────────────┼─────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              ConcurrentInferenceEngine (Brain/Coordinator)                 │
│                                                                            │
│  Role: Schedule work, manage cache decisions                              │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  SharedRadixCache (Arc<RwLock<>>)                                    │ │
│  │  ─────────────────────────────────────────────────────────────────   │ │
│  │  What: Prefix-sharing KV cache with O(1) LRU eviction              │ │
│  │  Lock: RwLock = many readers (cache queries) OR one writer        │ │
│  │  Stores: Indices to physical token storage in KVPool              │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  SharedKVPool (Arc<RwLock<>>)                                        │ │
│  │  ─────────────────────────────────────────────────────────────────   │ │
│  │  What: Physical storage for K and V tensors                         │ │
│  │  Lock: RwLock for concurrent access                                 │ │
│  │  Stores: Actual tensor data on GPU/CPU                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Semaphore: Limits to N concurrent request handlers (default: 64)        │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                    CacheInstruction
                 (cached_indices, new_indices)
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    Llama3 Model (Worker/Executor)                          │
│                                                                            │
│  Role: Execute forward passes using cache instructions                    │
│  Lock: Mutex = one request at a time (GPU is serial)                      │
│                                                                            │
│  Receives CacheInstruction telling it:                                    │
│    • Which KV indices are already cached (READ)                           │
│    • Which indices need new KV computation (WRITE)                        │
│    • Where to write results (token positions)                             │
│                                                                            │
│  This is the actual LLM model with weights loaded                         │
└────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow Diagram

```
Client submits "Translate: Hello"
     │
     ▼
[1] Tokenize: [1234, 5678, 9, ...] ◄── Ask tokenizer
     │
     ▼
[2] Cache Query: "Have we seen [1234, 5678] before?" ◄── Query RadixCache
     ├─ YES: Found 15 cached tokens ──► Skip KV compute for those
     └─ NO:  Cache miss              ──► Compute all KV
     │
     ▼
[3] Lock Cache: Increment reference count (prevent eviction)
     │
     ▼
[4] Allocate Storage: Reserve space in KVPool for new tokens
     │
     ▼
[5] Create CacheInstruction: Package cache decisions
     │
     ▼
[6] Forward Pass (Prefill):
     ├─ Tokenize & embed all tokens
     ├─ For each layer:
     │  ├─ Skip KV compute for cached tokens (reuse from pool)
     │  └─ Compute KV only for new tokens → write to allocated slots
     ├─ Compute logits for last token
     └─ Sample next token
     │
     ▼
[7] Generation Loop (Decode):
     For each new token:
     ├─ Create CacheInstruction: "Append this new token"
     ├─ Forward Pass (1 token):
     │  ├─ Reuse ALL previous KV (now fully cached)
     │  ├─ Compute KV for 1 new token
     │  └─ Get logits, sample
     └─ Append to output
     │
     ▼
[8] Complete Request:
     ├─ Insert full sequence into RadixCache (for future prefix matching)
     ├─ Unlock (decrement reference count)
     └─ Return text to client
```

---

## Key Components

### 1. RadixCache (infer-scheduler/src/kv_cache/radix_cache.rs)

**Purpose:** Maintain a prefix-sharing KV cache with O(1) LRU eviction

**Core Operations:**

```rust
// Query cache for matching prefix
let (matched_indices, node_handle) = cache.match_prefix(&key);
// Returns: How many tokens are already cached

// Lock node to prevent eviction during request processing
cache.inc_lock_ref(node_handle);

// Insert completed sequence into cache
cache.insert(&key, value_indices);

// Unlock node (may become evictable)
cache.dec_lock_ref(node_handle);

// Allocate storage for new tokens
let indices = cache.alloc_tokens(num_tokens)?;

// Evict least-recently-used entries when memory is full
let freed = cache.evict(num_tokens_needed);
```

**Internal Structure:**

```
Radix Tree (prefix matching):
├─ [root]
│  ├─ [1, 2, 3] → Node(lock_ref: 1, value: [0, 1, 2])
│  │  └─ [4, 5] → Node(lock_ref: 0, value: [3, 4])
│  └─ [1, 2, 6] → Node(lock_ref: 1, value: [5, 6])
│
LRU List (eviction order):
  [Head] → Node([4,5]) → Node([1,2,6]) → [Tail]
  ↑ oldest (evict first)      newest ↑
```

### 2. ConcurrentRadixCache (infer-scheduler/src/kv_cache/concurrent_cache.rs)

**Purpose:** Thread-safe wrapper around RadixCache using RwLock

**Key Design Decisions:**

| Operation | Lock Type | Why |
|-----------|-----------|-----|
| `match_prefix` | Write | Needs to touch LRU (update access time) |
| `get_stats` | Read | Just query, no modifications |
| `insert` | Write | Modifies tree structure |
| `inc_lock_ref` | Write | Updates node state |
| `prepare_request` | Write (atomic) | Match + lock + allocate in one shot |

**Thread Safety:** Generation counters detect stale handles

```rust
// Handle contains generation ID
// If cache is cleared, generation increments
// Old handles become invalid, preventing use-after-free
let handle = cache.match_prefix(&key)?;
// ... cache gets cleared ...
cache.inc_lock_ref(handle)?;  // ❌ Error: Stale handle!
```

### 3. ConcurrentInferenceEngine (infer-scheduler/src/concurrent_engine.rs)

**Purpose:** Coordinate multiple concurrent requests, manage cache decisions

**Key Features:**

```rust
// Spawn request as async task (concurrent with other requests)
let response = engine.process_request(request).await?;

// Process batch with true parallelism (tasks spawn concurrently)
let responses = engine.process_batch_parallel().await;

// Access statistics
let stats = engine.engine_stats().await;
println!("Completed: {}, Cache hit rate: {:.1}%",
    stats.completed_requests,
    hit_rate);
```

**Internal Coordination:**

```
Phase 1: Tokenize (quick)
    ├─ Release model lock
    └─ Allow other requests to progress

Phase 2: Query Cache (RwLock - concurrent reads)
    ├─ Multiple requests can query simultaneously
    └─ Cache decides what's already computed

Phase 3: Allocate Slots (RwLock write)
    ├─ Reserve memory for new tokens
    └─ Serialize with other allocations

Phase 4: Forward Pass (Model Mutex - serial)
    ├─ GPU can only run one inference at a time
    ├─ But cache queries happened in parallel!
    └─ Maximize useful work per GPU cycle

Phase 5: Generate Tokens (Model Mutex)
    └─ Same as Phase 4

Phase 6: Complete (RwLock write)
    ├─ Insert into cache for future requests
    └─ Release resources
```

### 4. Llama3 Model with CacheAwareModel (infer-worker/src/model/llama3.rs)

**Purpose:** Execute forward passes using cache instructions from Engine

**Cache-Aware Methods:**

```rust
// Full forward pass with cache instructions
fn forward_with_cache(
    &mut self,
    tokens: &Tensor,
    instruction: &CacheInstruction,  // What to reuse/compute
    kv_pool: &mut KVCachePool,
) -> Result<i32> { ... }

// Prefill phase: process entire prompt
fn prefill_with_cache(...) -> Result<i32> { ... }

// Decode phase: process one token at a time
fn decode_with_cache(...) -> Result<i32> { ... }
```

**What CacheInstruction Contains:**

```rust
pub struct CacheInstruction {
    pub request_id: String,
    pub cached_indices: Vec<usize>,    // ← READ these from KVPool
    pub new_indices: Vec<usize>,       // ← WRITE to these in KVPool
    pub seq_start_pos: usize,          // ← For RoPE position encoding
    pub total_seq_len: usize,
}
```

**How Worker Uses Instruction:**

```
Prefill Phase:
  For each layer:
    ├─ Compute embeddings for ALL tokens
    ├─ For CACHED tokens (skip KV computation):
    │  └─ Just use stored K, V from pool
    └─ For NEW tokens (compute KV):
       ├─ Run W_k, W_v projections
       └─ Write results to pool at allocated indices

Decode Phase:
  Single new token
    ├─ Embed new token
    └─ For each layer:
       ├─ Reuse ALL previous K, V (fully cached by now)
       ├─ Compute K, V for new token
       └─ Write to pool
```

---

## Usage Patterns

### Pattern 1: Basic Server Setup

```rust
use infer_engine::{ConcurrentInferenceEngine, ConcurrentEngineConfig};
use infer_engine::concurrent_zmq_server::{run_concurrent, ConcurrentServerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Load model
    let model = Llama3::new(
        "/path/to/model",
        DeviceType::Cuda(0),
        false
    )?;

    // Configure engine
    let engine_config = ConcurrentEngineConfig {
        max_batch_size: 32,
        max_queue_size: 256,
        max_concurrent_requests: 64,  // Key parameter!
        ..Default::default()
    };

    let engine = ConcurrentInferenceEngine::new(model, engine_config)?;

    // Start high-concurrency server
    let server_config = ConcurrentServerConfig::default();
    run_concurrent(engine, "tcp://0.0.0.0:5555", server_config).await?;

    Ok(())
}
```

### Pattern 2: Processing Single Request

```rust
// In async context
let request = InferenceRequest {
    request_id: uuid::Uuid::new_v4().to_string(),
    prompt: "Translate: Hello".to_string(),
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
    stop_sequences: vec![],
    stream: false,
    priority: 0,
};

let response = engine.process_request(request).await?;
println!("Generated: {}", response.text.unwrap());
println!("Tokens: {}", response.num_tokens);
println!("Time: {}ms", response.metrics.prefill_ms + response.metrics.decode_ms);
```

### Pattern 3: Batch Processing

```rust
// Enqueue multiple requests
for i in 0..100 {
    let request = create_request(i);
    engine.enqueue_request(request).await?;
}

// Process as batch with parallelism
let responses = engine.process_batch_parallel().await;
for resp in responses {
    println!("✅ {}: {} tokens", resp.request_id, resp.num_tokens);
}
```

### Pattern 4: Monitoring Performance

```rust
// Get engine statistics
let stats = engine.engine_stats().await;
println!("Total requests: {}", stats.total_requests);
println!("Cache hit rate: {:.1}%", stats.cache_hits as f64 / stats.total_requests as f64 * 100.0);
println!("Avg throughput: {:.1} tokens/sec", stats.total_tokens_generated as f64 / stats.total_requests as f64);

// Get cache details
let cache_stats = engine.cache_stats()?;
println!("Cached tokens: {}", cache_stats.total_cached);
println!("Evictable: {}, Protected: {}", cache_stats.evictable_size, cache_stats.protected_size);
println!("Evictions: {}", cache_stats.evictions);
```

---

## Concurrency Model

### How Multiple Requests Run Concurrently

```
Time →

Request A: [Token] [Cache] [GPU  ] [GPU  ] [Cache]
Request B:            [Token] [Cache] [GPU  ] [GPU  ] [Cache]
Request C:                     [Token] [Cache] [GPU  ] [GPU  ]

Notice:
- Token processing (async, no GPU) → happens in parallel
- Cache queries (RwLock, read) → concurrent!
- GPU computation (Model Mutex) → serialized (one at a time)
- Cache completion (RwLock, write) → queued
```

### Lock Hierarchy

```
To prevent deadlocks, always acquire locks in this order:

1. RadixCache (RwLock read/write) ← Highest priority
   └─ Quick operations, released soon

2. KVPool (RwLock write)
   └─ Allocate/deallocate storage

3. Model (Mutex)
   └─ Long operations (forward pass)
      Acquired LAST, held LONGEST
```

### How RwLock Maximizes Concurrency

```
Multiple Readers (cache queries):
┌─────────────────────────────────────┐
│  Request A: match_prefix()          │
│  Request B: match_prefix()  ← concurrent!
│  Request C: match_prefix()  ← all reading simultaneously
│  Request D: get_stats()     ← also concurrent!
└─────────────────────────────────────┘

Exclusive Writer (cache modifications):
┌─────────────────────────────────────┐
│  Request E: inc_lock_ref() ← only E holds lock
│  (Other requests wait)
└─────────────────────────────────────┘

Key Insight:
- Most operations are reads (check cache)
- Few operations are writes (lock/unlock/insert)
- → Readers never block readers
- → High throughput for read-heavy cache queries!
```

### Semaphore Limits Concurrent Requests

```
Configuration: max_concurrent_requests = 64

Request 1-64:  ✓ Acquire permit, proceed
Request 65:    ⏳ Wait (queue at semaphore)
Request 66:    ⏳ Wait (queue at semaphore)

When Request 1 completes:
  Release permit
  Request 65: ✓ Acquire permit, proceed
```

**Why This Matters:**

Without semaphore:
- 1000 requests could all try to tokenize simultaneously
- Token processing is CPU-bound, creates contention
- GPU idles waiting for CPU

With semaphore:
- Only 64 requests active at once
- GPUs stay busy, CPU isn't overloaded
- Better resource utilization

---

## Performance Characteristics

### Operation Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| `match_prefix` | O(L) | L = key length (tokens), but fast radix tree |
| `insert` | O(L) | May split nodes, but amortized |
| `inc_lock_ref` | O(D) | D = tree depth, usually 5-10 |
| `dec_lock_ref` | O(D) | Same as above |
| `evict` | O(1) | Pop from LRU list - constant! |
| `alloc_tokens` | O(1) | Simple free list management |

### Memory Usage

```
For typical config (28 layers, 256 kv_dim, 65536 max_tokens):

K Cache:  28 * 65536 * 256 * 2 bytes = 233 MB
V Cache:  28 * 65536 * 256 * 2 bytes = 233 MB
Metadata: ~1 MB (tree nodes, pointers)
─────────────────────────────────────
Total:    ~467 MB (adjustable via config)
```

### Throughput Expectations

With optimal conditions:

```
Prefill Phase:
  - 200-500 tokens/second (depends on GPU)

Decode Phase:
  - 50-150 tokens/second (slower, single token)

With Caching:
  - 500-1000 tokens/second (many requests with shared prefixes)
```

### Scaling Behavior

```
Number of concurrent requests → Throughput

No caching:
  1 request:   100 tokens/sec
  10 requests: 100 tokens/sec (each slower)

With caching (shared prefixes):
  1 request:   100 tokens/sec
  10 requests: 500 tokens/sec (shared prefix computed once!)

Optimal:
  - 10-20 requests with 80% prefix overlap
  - Cache hit rate: 80-90%
  - Throughput: 2-3x improvement
```

---

## Troubleshooting

### Problem: Low Cache Hit Rate

**Symptoms:**
```
Cache stats show < 20% hit rate even though requests seem similar
```

**Possible Causes:**

1. **Diverse prompts** - Each request uses different prompt
   - Solution: Look for real duplicates/templates

2. **Requests arrive too quickly**
   - First request hasn't finished (uncached) before second arrives
   - Solution: Increase time between requests in load test

3. **Cache eviction** - Older prompts removed before reuse
   - Check: `cache_stats.evictions`
   - Solution: Increase `max_cache_tokens` in config

**Debug Tip:**
```rust
let stats = engine.cache_stats()?;
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
println!("Node count: {}", stats.node_count);
println!("Evictions: {}", stats.evictions);
```

### Problem: Out of Memory

**Symptoms:**
```
Error: Failed to allocate cache slots
GPU memory exhausted
```

**Possible Causes:**

1. **Max cache tokens too high**
   - Current: `65536 tokens * 28 layers * 256 dim * 2 bytes = 467 MB`
   - Reduce `max_cache_tokens` in config

2. **Concurrent requests grow unbounded**
   - Check: `engine.concurrent_request_count()`
   - Solution: Lower `max_concurrent_requests`

3. **Eviction not working**
   - Check: Are locked nodes preventing eviction?
   - Verify: `dec_lock_ref` being called on all requests

**Debug Tip:**
```rust
let cache_stats = engine.cache_stats()?;
println!("Utilization: {:.1}%", cache_stats.utilization() * 100.0);
println!("Protected (locked): {} tokens", cache_stats.protected_size);
println!("Evictable: {} tokens", cache_stats.evictable_size);
```

### Problem: High Lock Contention

**Symptoms:**
```
CPU usage high but GPU utilization low
Throughput doesn't improve with more concurrent requests
```

**Possible Causes:**

1. **All requests hit cache simultaneously** (good problem!)
   - They all try to acquire write lock for cache operations
   - Solution: This is actually optimal; GPU becomes bottleneck

2. **Model lock held too long**
   - Forward pass should take ~50-500ms for typical model
   - If longer, check GPU kernels

3. **Too many concurrent requests**
   - Reduce `max_concurrent_requests`
   - Allow more requests in flight, but serialize better

**Debug Tip:**
```rust
let stats = engine.engine_stats().await;
println!("Concurrent requests active: {}", engine.concurrent_request_count());
println!("Queue size: {}", queue_status.0);
println!("Avg prefill: {:.1}ms", stats.avg_prefill_time_ms);
println!("Avg decode: {:.1}ms", stats.avg_decode_time_ms);
```

### Problem: Stale Handle Error

**Symptoms:**
```
Error: "Stale node handle (gen 5 vs current 6)"
```

**Cause:** Cache was cleared while request was processing

**Solution:**

1. **Don't call `clear_cache()` during serving**
   ```rust
   // ❌ WRONG: Clears cache, invalidates all handles
   engine.clear_cache();
   ```

2. **If you must clear:**
   - Drain request queue first
   - Wait for all in-flight requests to complete
   - Then clear
   ```rust
   // ✓ RIGHT: Wait for completion
   while engine.queue_status().0 > 0 {
       tokio::time::sleep(Duration::from_millis(10)).await;
   }
   engine.clear_cache();
   ```

### Problem: Uneven Request Latency

**Symptoms:**
```
Request A: 100ms
Request B: 200ms
Request C: 500ms
Varies widely even with same model
```

**Cause:** Some requests hit cache, others don't

**Expected Behavior:**
- Uncached request: slow (compute all KV)
- Cached request: fast (reuse KV)
- → Variance is expected and desired!

**If Variance is Too High:**

1. **Check queue length**
   - If request waits in queue, latency increases
   - Solution: Increase `max_queue_size`

2. **Check semaphore limits**
   - If request waits on semaphore, latency increases
   - Solution: Tune `max_concurrent_requests`

---

## Best Practices

### 1. Batch Similar Requests

**Good:**
```rust
// These will share prefixes
"Translate to Spanish: Hello"
"Translate to Spanish: Goodbye"
"Translate to Spanish: Thank you"
```

**Bad:**
```rust
// All different, no caching benefit
"What is 2+2?"
"Describe quantum mechanics"
"Generate a poem about cats"
```

### 2. Tune Parameters for Your Hardware

```rust
let config = ConcurrentEngineConfig {
    max_concurrent_requests: 64,  // Tune based on GPU memory
    max_batch_size: 32,           // Balance throughput vs latency
    max_queue_size: 256,          // Handle traffic spikes
    max_cache_tokens: 65536,      // GPU memory available
    ..Default::default()
};
```

### 3. Monitor Key Metrics

```rust
// Production monitoring
let stats = engine.engine_stats().await;
let cache = engine.cache_stats()?;

tracing::info!(
    "Throughput: {:.1} tok/s, Hit: {:.1}%, Util: {:.1}%",
    throughput,
    cache.hit_rate() * 100.0,
    cache.utilization() * 100.0
);
```

### 4. Handle Errors Gracefully

```rust
match engine.process_request(request).await {
    Ok(response) => send_to_client(response),
    Err(e) => {
        tracing::error!("Request failed: {}", e);
        send_error_response(&request.request_id, &e.to_string())
    }
}
```

### 5. Release Resources Properly

```rust
// Always unlock after use
cache.inc_lock_ref(handle);
// ... do work ...
cache.dec_lock_ref(handle);  // ✓ Essential!

// Without unlock, memory leaks (locked tokens can't be evicted)
```

---

## Testing

### Unit Tests

Run existing tests:
```bash
cargo test -p infer-scheduler --lib
cargo test -p infer-worker --lib
```

### Concurrency Tests

```rust
#[tokio::test]
async fn test_concurrent_requests() {
    let engine = setup_engine();

    // Spawn 100 requests concurrently
    let mut handles = Vec::new();
    for i in 0..100 {
        let req = make_request(i);
        handles.push(tokio::spawn({
            let e = engine.clone();
            async move { e.process_request(req).await }
        }));
    }

    // Wait for all
    for h in handles { h.await.unwrap(); }

    // Verify all succeeded
    let stats = engine.engine_stats().await;
    assert_eq!(stats.completed_requests, 100);
}
```

### Load Testing

```bash
# Simple load test with multiple concurrent clients
for i in {1..10}; do
    curl -X POST http://localhost:5555 \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"Hello\", \"max_tokens\": 50}" &
done
wait
```

---

## References

### Related Files

- **Radix Cache:** `/home/vinci/RustInfer/crates/infer-scheduler/src/kv_cache/radix_cache.rs`
- **Concurrent Cache:** `/home/vinci/RustInfer/crates/infer-scheduler/src/kv_cache/concurrent_cache.rs`
- **Concurrent Engine:** `/home/vinci/RustInfer/crates/infer-scheduler/src/concurrent_engine.rs`
- **ZMQ Server:** `/home/vinci/RustInfer/crates/infer-scheduler/src/concurrent_zmq_server.rs`
- **Model Integration:** `/home/vinci/RustInfer/crates/infer-worker/src/model/llama3.rs`

### Papers & References

- **Attention is All You Need** - Transformer architecture basics
- **RadixAttention** - Prefix sharing for KV cache
- **SGLang** - Inspiration for Engine-Worker architecture
- **Flash Attention** - Efficient attention computation

---

## Summary

RustInfer's high-concurrency system achieves 2-3x throughput improvement through:

1. **Prefix Sharing** via RadixAttention - Cache computation for common prompt prefixes
2. **Fine-Grained Locking** via RwLock - Allow cache queries to happen in parallel
3. **Semaphore Limits** - Prevent resource exhaustion, keep GPU busy
4. **Engine-Worker Pattern** - Separate cache management (CPU) from model execution (GPU)

The result: **Multiple requests processed faster than individually**, with intelligent cache reuse and minimal lock contention.
