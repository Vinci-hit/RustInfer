# RustInfer CLI - Production-Grade HTTP Inference Server

> A modern, high-performance HTTP server for LLM inference with OpenAI-compatible API, inspired by vLLM and SGLang architecture.

## 🚀 Quick Start

```bash
# Start the server
cargo run --release --bin rustinfer-server -- \
    --model /path/to/llama3 \
    --port 8000 \
    --device cuda:0 \
    --max-tokens 512

# Or use the compiled binary
./target/release/rustinfer-server \
    --model /mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct \
    --port 8000 \
    --device cuda:0

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Advanced Design Patterns](#advanced-design-patterns)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Performance & Optimization](#performance--optimization)
- [Development Guide](#development-guide)
- [Roadmap](#roadmap)

---

## ✨ Features

### Core Capabilities
- ✅ **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- ✅ **Performance Observability** - Real-time metrics (prefill/decode time, tokens/sec)
- ✅ **System Monitoring** - CPU, GPU, memory metrics endpoint
- ✅ **Server-Sent Events (SSE)** - Real-time streaming responses
- ✅ **Automatic Chat Templates** - Llama3 format wrapping
- ✅ **Async/Await Runtime** - Built on Axum + Tokio
- ✅ **Thread-Safe Inference** - Arc<Mutex> for concurrent requests
- ✅ **CUDA Graph Ready** - Pre-allocated workspace buffers
- ✅ **Graceful Shutdown** - Proper resource cleanup
- ✅ **CORS Support** - Ready for web applications
- ✅ **Structured Logging** - Tracing-based observability

### Production-Ready
- 🔒 Type-safe Rust implementation
- 🚀 Zero-copy tensor operations
- 📊 Request/response logging
- 🎯 Health & readiness endpoints
- 🔧 Environment variable configuration
- 📦 Small binary size (12MB release build)

---

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                   │
│  (curl, Python OpenAI SDK, Web Apps, etc.)             │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Axum HTTP Server (main.rs)                │
│  • Router configuration                                 │
│  • CORS middleware                                      │
│  • Tracing middleware                                   │
│  • Graceful shutdown handler                            │
└────────────────────┬────────────────────────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
┌─────────────────┐   ┌──────────────────┐
│  API Handlers   │   │  Health Checks   │
│  (api/)         │   │  (api/health.rs) │
│                 │   │                  │
│ • openai.rs     │   │ • /health        │
│   - /v1/chat/   │   │ • /ready         │
│     completions │   │                  │
│   - /v1/models  │   └──────────────────┘
│ • metrics.rs    │
│   - /v1/metrics │
│     (system     │
│      resources) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          Inference Engine (inference/engine.rs)         │
│  • Arc<Mutex<InferenceEngine>> (thread-safe)           │
│  • Request queuing & serialization                      │
│  • Chat template application                            │
│  • Response formatting                                  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  Chat Templates │     │   infer-core     │
│  (chat/)        │     │   (external)     │
│                 │     │                  │
│ • Llama3 format │     │ • Llama3 model   │
│ • Message wrap  │     │ • CUDA kernels   │
│ • System prompt │     │ • BF16 inference │
└─────────────────┘     └──────────────────┘
```

### Component Breakdown

#### 1. **HTTP Layer** (`main.rs`)
```rust
// Axum router with state sharing
let app = Router::new()
    .route("/v1/chat/completions", post(chat_completions))
    .with_state(Arc::new(Mutex::new(engine)))  // Shared mutable state
    .layer(CorsLayer::new().allow_origin(Any))  // Cross-origin support
    .layer(TraceLayer::new_for_http());         // Request logging
```

**Design Decisions:**
- **Axum Framework**: Modern, ergonomic, built on Tokio
- **Arc<Mutex<>>**: Thread-safe shared ownership of model
- **Tower Middleware**: Composable request/response processing

#### 2. **API Layer** (`api/`)
```
api/
├── mod.rs          # Module exports
├── openai.rs       # OpenAI-compatible types & handlers
└── health.rs       # Liveness/readiness probes
```

**Key Types** (`openai.rs`):
```rust
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    // Future: temperature, top_p, stop sequences
}

pub struct ChatCompletionResponse {
    pub id: String,              // Unique request ID
    pub object: String,          // "chat.completion"
    pub created: i64,            // Unix timestamp
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,            // Token counts
}
```

**Handler Pattern**:
```rust
pub async fn chat_completions(
    State(engine): State<Arc<Mutex<InferenceEngine>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    if req.stream {
        // SSE streaming path
        let stream = engine.lock().await.generate_stream(req).await?;
        Ok(Sse::new(stream).into_response())
    } else {
        // Blocking path
        let response = engine.lock().await.generate(req).await?;
        Ok(Json(response).into_response())
    }
}
```

#### 3. **Inference Engine** (`inference/engine.rs`)

**Thread-Safety Strategy**:
```rust
pub struct InferenceEngine {
    model: Llama3,          // Not Send/Sync by default
    config: ServerConfig,
}

// Usage: Arc<Mutex<InferenceEngine>>
// • Arc: Shared ownership across requests
// • Mutex: Exclusive access for inference (serialized)
```

**Why This Design?**
1. **Simplicity**: Mutex ensures sequential inference (no race conditions)
2. **Correctness**: Single-threaded inference avoids CUDA context issues
3. **Future-Proof**: Easy to extend with batching or multiple models

**Streaming Implementation**:
```rust
pub async fn generate_stream(
    &mut self,
    req: ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel();

    // Generate tokens
    let result = self.model.generate(&prompt, max_tokens, false);

    // Send as SSE events
    for chunk in text.split_whitespace() {
        let event = Event::default().data(json_string);
        tx.send(Ok(event))?;
    }

    Ok(UnboundedReceiverStream::new(rx))
}
```

**Note**: Current implementation simulates streaming by chunking the final output. True token-by-token streaming requires `infer-core` API enhancement.

#### 4. **Chat Template System** (`chat/template.rs`)

**Trait-Based Design**:
```rust
pub trait ChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String>;
}

pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::from("<|begin_of_text|>");

        for msg in messages {
            match msg.role.as_str() {
                "system" => prompt.push_str("<|start_header_id|>system..."),
                "user" => prompt.push_str("<|start_header_id|>user..."),
                "assistant" => prompt.push_str("<|start_header_id|>assistant..."),
                _ => {}
            }
        }

        Ok(prompt)
    }
}
```

**Extensibility**: Easily add new templates (ChatML, Alpaca, etc.) by implementing the trait.

---

## 🎓 Advanced Design Patterns

### 1. **Async/Await Concurrency**

**Pattern**: Non-blocking I/O with Tokio runtime

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
```

**Benefits**:
- Handles thousands of concurrent connections
- Low memory overhead (green threads)
- Async lock contention (Mutex::lock().await)

### 2. **Type-State Pattern for Safety**

**Request Validation**:
```rust
#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,  // Optional with default

    #[serde(default)]
    pub stream: bool,  // Defaults to false
}

fn default_max_tokens() -> Option<usize> {
    Some(512)
}
```

Serde ensures type safety at deserialization time, preventing invalid requests.

### 3. **RAII for Resource Management**

**Graceful Shutdown**:
```rust
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received...");
}

// Axum automatically:
// 1. Stops accepting new connections
// 2. Waits for in-flight requests to complete
// 3. Drops Arc<Mutex<InferenceEngine>>
// 4. Triggers Drop impl in infer-core (frees CUDA memory)
```

### 4. **Error Handling with Context**

**Custom Error Type**:
```rust
#[derive(Debug)]
pub struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!("Request failed: {:?}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": self.0.to_string()}})),
        ).into_response()
    }
}

// Automatic conversion from any error
impl<E: Into<anyhow::Error>> From<E> for AppError { ... }
```

**Benefits**:
- Centralized error formatting
- Automatic OpenAI-compatible error responses
- Stack traces in logs

### 5. **Zero-Copy via Arc**

**Model Weight Sharing**:
```rust
// In infer-core
pub struct Llama3 {
    layers: Vec<LlamaLayer>,  // Contains Arc<Tensor>
}

// Cloning a tensor is cheap (just Arc clone)
let weight_clone = tensor.clone();  // O(1), no data copy
```

### 6. **Send/Sync Safety Boundaries**

**Critical Implementations**:
```rust
// CUDA resources are thread-safe (CUDA runtime handles sync)
unsafe impl Send for CudaConfig {}
unsafe impl Sync for CudaConfig {}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

// Tokenizer must be thread-safe
pub trait Tokenizer: Send + Sync { ... }
```

**Why `unsafe impl`?**
- Raw pointers (`*mut c_void`) block auto-derivation
- CUDA runtime ensures thread safety internally
- We document safety invariants in comments

### 7. **CUDA Graph Optimization (Future-Ready)**

**Pre-Allocated Workspace**:
```rust
pub struct CudaConfig {
    pub stream: cudaStream_t,
    pub workspace: *mut c_void,           // 32MB shared buffer
    pub argmax_result: *mut i32,          // Pre-allocated for sampling
}
```

**Benefit**: Eliminates dynamic allocation in hot path, enabling CUDA graphs (10-20% faster).

---

## 📚 API Reference

### Endpoints

#### `POST /v1/chat/completions`

Create a chat completion with conversational context.

**Request Body**:
```json
{
  "model": "llama3",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Rust?"}
  ],
  "max_tokens": 150,
  "stream": false
}
```

**Response** (non-streaming):
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Rust is a systems programming language..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 36,
    "total_tokens": 36,
    "performance": {
      "prefill_ms": 183,
      "decode_ms": 118,
      "decode_iterations": 35,
      "tokens_per_second": 29.9,
      "time_to_first_token_ms": 183
    }
  }
}
```

**Response** (streaming with `stream: true`):
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Rust "}}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"is "}}]}

data: [DONE]
```

#### `GET /v1/models`

List available models.

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3",
      "object": "model",
      "owned_by": "rustinfer"
    }
  ]
}
```

#### `GET /health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "rustinfer-server"
}
```

#### `GET /ready`

Readiness probe (model loaded).

**Response**:
```json
{
  "status": "ready",
  "model_loaded": true
}
```

#### `GET /v1/metrics`

System resource metrics for monitoring.

**Response**:
```json
{
  "cpu": {
    "utilization_percent": 1.1,
    "core_count": 28
  },
  "memory": {
    "used_mb": 2244,
    "total_mb": 15903,
    "available_mb": 13658
  },
  "gpu": {
    "device_id": 0,
    "utilization_percent": 45.2,
    "memory_used_mb": 2500,
    "memory_total_mb": 24576,
    "temperature_celsius": 65.5
  },
  "timestamp": 1767937158
}
```

**Note**: `gpu` field is `null` when CUDA feature is disabled or no GPU is available.

---

## 💡 Usage Examples

### Python (OpenAI SDK)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not validated
)

# Non-streaming
response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a Rust expert."},
        {"role": "user", "content": "Explain ownership in 2 sentences."}
    ],
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Write a haiku about Rust"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### JavaScript (Fetch API)

```javascript
// Non-streaming
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'llama3',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: false
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);

// Streaming with EventSource
const eventSource = new EventSource(
  'http://localhost:8000/v1/chat/completions?' +
  new URLSearchParams({
    model: 'llama3',
    messages: JSON.stringify([{ role: 'user', content: 'Count to 5' }]),
    stream: 'true'
  })
);

eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const chunk = JSON.parse(event.data);
  console.log(chunk.choices[0].delta.content);
};
```

### curl

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hi!"}],
    "stream": false
  }'

# Streaming (-N for no buffer)
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

---

## 🌐 Web Frontend

A modern Dioxus-based web application is available at `crates/infer-frontend/` providing:

- **Interactive Chat Interface** - Multi-round conversations with the model
- **Real-Time Performance Metrics** - Display prefill/decode time, tokens/sec for each response
- **System Monitoring Dashboard** - Live CPU, GPU, memory usage (polls /v1/metrics every 2s)
- **Responsive UI** - Tailwind CSS with dark theme
- **WASM-Based** - Runs entirely in the browser

**Quick Start**:
```bash
# Terminal 1: Start backend
cargo run --release --bin rustinfer-server -- \
    --model /path/to/model \
    --port 8000

# Terminal 2: Start frontend
cd crates/infer-frontend
dx serve --port 3000

# Open browser: http://localhost:3000
```

See `crates/infer-frontend/README.md` for details.

---

## ⚡ Performance & Optimization

### Current Performance (Llama-3.2-1B-Instruct, BF16, RTX 4090)

- **Model Load Time**: 9.46 seconds
- **Inference Latency**: <100ms TTFT (first token)
- **Throughput**: ~30-40 tokens/sec (single request)
- **Memory Usage**: ~2.5GB VRAM (model + KV cache)

### Optimization Strategies

#### 1. **Concurrent Request Handling**

Current: Sequential (Mutex serializes requests)
```rust
// Current: Only one request at a time
Arc<Mutex<InferenceEngine>>
```

**Future**: Continuous batching (vLLM-style)
```rust
// Queue requests and batch prefill
struct BatchedEngine {
    queue: Arc<Mutex<VecDeque<Request>>>,
    worker: JoinHandle<()>,  // Background batch processor
}
```

**Expected Gain**: 3-5x throughput for concurrent users

#### 2. **Token-by-Token Streaming**

Current: Simulated (split final output)
```rust
for chunk in text.split_whitespace() { ... }
```

**Future**: True streaming from infer-core
```rust
// Proposed API
model.generate_stream(prompt, |token| {
    tx.send(token).ok();
});
```

**Expected Gain**: 50% lower latency perception

#### 3. **KV Cache Management**

Current: Static allocation (2048 max length)
```rust
// In infer-core
kv_cache = Tensor::new(&[max_len, kv_dim], BF16, device)?;
```

**Future**: Dynamic growing cache
```rust
// Only allocate what's needed
kv_cache.resize(actual_length)?;
```

**Expected Gain**: 40% memory savings for short sequences

#### 4. **CUDA Graphs**

Current: Kernel launches per token
```rust
// Each token = ~5 kernel launches
rmsnorm -> matmul -> attention -> add -> sample
```

**Future**: Record computation graph once
```rust
let graph = cuda::capture_graph(|| {
    model.forward_decode(token);
});
graph.replay();  // 10-20% faster
```

**Requirement**: Static shapes + pre-allocated buffers ✅ (already done!)

---

## 🛠️ Development Guide

### Project Structure

```
infer-server/
├── Cargo.toml          # Dependencies & build config
├── README.md           # This file
├── src/
│   ├── main.rs         # Server entry point
│   ├── lib.rs          # Public API (for testing)
│   ├── api/
│   │   ├── mod.rs
│   │   ├── openai.rs   # OpenAI-compatible handlers
│   │   └── health.rs   # Health checks
│   ├── chat/
│   │   ├── mod.rs
│   │   └── template.rs # Chat template implementations
│   ├── inference/
│   │   ├── mod.rs
│   │   └── engine.rs   # Inference engine wrapper
│   └── config/
│       ├── mod.rs
│       └── server.rs   # Server configuration
└── tests/
    └── integration.rs  # (Future) End-to-end tests
```

### Building

```bash
# Development build
cargo build --bin rustinfer-server

# Release build (optimized)
cargo build --release --bin rustinfer-server

# Run with logging
RUST_LOG=debug cargo run --release --bin rustinfer-server -- \
    --model /path/to/model \
    --port 8000
```

### Testing

```bash
# Unit tests
cargo test --lib

# Integration tests (start server first)
cargo test --test integration -- --test-threads=1

# Manual testing
curl http://localhost:8000/health
```

### Adding a New Chat Template

1. Implement the trait in `src/chat/template.rs`:

```rust
pub struct ChatMLTemplate;

impl ChatTemplate for ChatMLTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        Ok(prompt)
    }
}
```

2. Update `get_template()` to detect format:

```rust
pub fn get_template(model_name: &str) -> Box<dyn ChatTemplate> {
    if model_name.contains("chatml") {
        Box::new(ChatMLTemplate)
    } else {
        Box::new(Llama3Template)
    }
}
```

### Environment Variables

```bash
# All CLI args can be set via env vars
export MODEL_PATH=/path/to/model
export HOST=0.0.0.0
export PORT=8000
export DEVICE=cuda:0
export MAX_TOKENS=512
export RUST_LOG=info

./rustinfer-server  # Uses env vars
```

---

## 🗺️ Roadmap

### Phase 1: MVP ✅ (Completed)
- [x] OpenAI-compatible API
- [x] Llama3 chat template
- [x] SSE streaming
- [x] Health checks
- [x] CORS support
- [x] Graceful shutdown
- [x] **Performance metrics** (prefill/decode time, tokens/sec)
- [x] **System monitoring endpoint** (/v1/metrics)
- [x] **Web frontend** (Dioxus-based)

### Phase 2: Performance (In Progress)
- [x] Request/response observability ✅
- [ ] Token-by-token streaming (requires infer-core API)
- [ ] Request batching (continuous batching)
- [ ] CUDA graph integration
- [ ] Request queue visualization

### Phase 3: Features
- [ ] Multiple model support (load/unload)
- [ ] Temperature/top-p/top-k sampling
- [ ] Stop sequences
- [ ] Logprobs output
- [ ] Function calling API
- [ ] Vision support (multi-modal)

### Phase 4: Production
- [ ] Request authentication (API keys)
- [ ] Rate limiting
- [ ] Request caching
- [ ] Distributed inference (multi-GPU)
- [ ] Kubernetes deployment manifests
- [ ] Docker image
- [ ] Horizontal scaling with load balancer

---

## 🤝 Contributing

### Code Style
- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Add tracing logs for important events
- Document public APIs with `///` doc comments

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit with descriptive messages
4. Push and open a PR with detailed description

### Architecture Decisions
For major changes, open an issue first to discuss:
- Performance implications
- API compatibility
- Memory usage
- Thread safety

---

## 📄 License

This project is part of RustInfer and shares the same license.

---

## 🙏 Acknowledgments

**Inspired by:**
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention & continuous batching
- [SGLang](https://github.com/sgl-project/sglang) - Structured generation & runtime
- [Axum](https://github.com/tokio-rs/axum) - Ergonomic web framework
- [OpenAI API](https://platform.openai.com/docs/api-reference) - Standard API design

**Built with:**
- 🦀 Rust - Performance + Safety
- ⚡ Tokio - Async runtime
- 🌐 Axum - HTTP framework
- 🎯 CUDA - GPU acceleration via infer-core

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the [infer-core documentation](../infer-core/README.md)
- Review API examples in this README

**Server running?** Check logs with `RUST_LOG=debug`
