# infer-server

The HTTP API server for RustInfer - a high-performance LLM inference platform. Provides OpenAI-compatible REST API endpoints with comprehensive monitoring and metrics.

## Overview

`infer-server` is the API gateway component of the RustInfer platform. It:
- Exposes HTTP REST APIs compatible with OpenAI's chat completion format
- Communicates with the inference engine via ZeroMQ for low-latency IPC
- Provides real-time system metrics and monitoring endpoints
- Supports GPU acceleration through CUDA
- Handles request routing, template application, and response formatting

## Table of Contents

- [Architecture](#architecture)
- [Design Philosophy](#design-philosophy)
- [Data Flow](#data-flow)
- [API Documentation](#api-documentation)
- [Component Structure](#component-structure)
- [Configuration](#configuration)
- [Development Guide](#development-guide)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Browser  │  │ cURL     │  │ Python   │  │ Any HTTP     │  │
│  │ (Frontend)│  │ CLI      │  │ Script   │  │ Client       │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
└───────┼────────────┼────────────┼─────────────┼────────────┘
        │            │            │             │
        └────────────────────────┴─────────────┘
                      │ HTTP/JSON
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      infer-server                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Layer (Axum)                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ /chat/       │  │ /metrics     │  │ /health      │  │  │
│  │  │ completions  │  │              │  │ /ready       │  │  │
│  │  └──────┬───────┘  └──────────────┘  └──────────────┘  │  │
│  │         │                                              │  │
│  │  ┌──────▼────────────────────────────────────────┐     │  │
│  │  │  Chat Template Engine                         │     │  │
│  │  │  (Llama3, and more...)                       │     │  │
│  │  └──────┬────────────────────────────────────────┘     │  │
│  └─────────┼──────────────────────────────────────────────┘  │
│            │ ZMQ Client                                      │
└────────────┼────────────────────────────────────────────────┘
             │ ZeroMQ (MessagePack)
             │ ipc:///tmp/rustinfer.ipc
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      infer-scheduler                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Inference Engine                                        │  │
│  │  • Model Loading & Execution                             │  │
│  │  • KV Cache Management (RadixTree)                        │  │
│  │  • Request Scheduling & Batching                         │  │
│  │  • CUDA GPU Acceleration                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **infer-server** | HTTP API, request routing, chat templates, metrics collection |
| **infer-scheduler** | Model inference, GPU execution, cache management, batch scheduling |
| **infer-protocol** | Shared communication protocol (Message types, serialization) |
| **infer-worker** | Shared utilities (tensor ops, kernels, model abstractions) |
| **infer-frontend** | Web UI for chat interface and monitoring dashboard |

---

## Design Philosophy

### Core Principles

1. **Process Separation**
   - API server and inference engine run in separate processes
   - Benefits: Isolation, independent scaling, fault tolerance
   - Enables graceful degradation (API can continue if engine restarts)

2. **Async-First Architecture**
   - Built on Tokio async runtime for high concurrency
   - Non-blocking I/O for all external operations
   - Efficient resource utilization with thousands of concurrent requests

3. **Type Safety & Memory Safety**
   - Leverages Rust's type system for compile-time guarantees
   - Zero-cost abstractions for high performance
   - No runtime overhead from garbage collection

4. **Low-Latency Communication**
   - ZeroMQ with IPC transport for inter-process communication
   - MessagePack serialization for efficient binary encoding
   - Sub-millisecond latency between server and engine

5. **OpenAI Compatibility**
   - Drop-in replacement for OpenAI's chat completion API
   - Existing OpenAI clients work without modification
   - Lowers migration barrier for users

6. **Observable by Design**
   - Comprehensive metrics collection from day one
   - Real-time monitoring endpoints
   - Performance metrics embedded in responses

### Design Patterns Used

#### 1. Service Layer Pattern
```rust
// ZMQ client as a shared service
struct ZmqClient {
    sender: mpsc::Sender<ClientRequest>,
    pending: Arc<RwLock<HashMap<String, PendingRequest>>>,
}

// Passed to all API handlers via Axum state
let state = Arc::new(ZmqClient::new(engine_endpoint)?);
```

#### 2. Request-Response Correlation
```rust
// Each request has unique ID for tracking
struct InferenceRequest {
    request_id: String,  // UUID v4
    // ... other fields
}

// Response matched by request_id
struct InferenceResponse {
    request_id: String,
    // ... other fields
}
```

#### 3. Factory Pattern for Templates
```rust
// Template selection based on model
enum ChatTemplate {
    Llama3,
    // Future: GPT, Claude, etc.
}

impl ChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Llama3 => apply_llama3_template(messages),
        }
    }
}
```

#### 4. Observer Pattern for Metrics
```rust
// Metrics collected throughout request lifecycle
let mut metrics = InferenceMetrics::default();

// Update at various stages
metrics.queue_ms = elapsed.as_millis() as u64;
metrics.prefill_ms = prefill_time;
metrics.decode_ms = decode_time;
```

---

## Data Flow

### Request Lifecycle

```
1. HTTP Request
   Client ──► POST /v1/chat/completions
              {
                "model": "llama3",
                "messages": [
                  {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 512
              }

2. API Handler (openai.rs)
   ├─ Parse request body
   ├─ Extract model name
   ├─ Apply chat template
   │   └─ Convert chat messages → single prompt string
   │
   └─ Build InferenceRequest
       {
         "request_id": "uuid-123",
         "prompt": "<|begin_of_text|><|start_header_id|>user...",
         "max_tokens": 512,
         "temperature": 0.7,
         // ... other params
       }

3. ZMQ Client (zmq_client.rs)
   ├─ Serialize to MessagePack
   ├─ Send to engine via DEALER socket
   │   └─ Engine queue management
   │
   └─ Wait for response (with timeout)

4. Inference Engine
   ├─ Receive request
   ├─ Schedule execution
   ├─ Prefill phase
   │   ├─ Tokenize input
   │   ├─ Load into KV cache
   │   └─ Compute initial attention
   │
   ├─ Decode phase (iterative)
   │   ├─ Generate next token
   │   ├─ Update cache
   │   └─ Check stop conditions
   │
   └─ Build InferenceResponse
       {
         "request_id": "uuid-123",
         "status": "Success",
         "text": "Hello! How can I help you?",
         "tokens": [15496, 1115, ...],
         "num_tokens": 7,
         "metrics": {
           "prefill_ms": 45,
           "decode_ms": 123,
           "queue_ms": 0,
           "batch_size": 1,
           "tokens_per_second": 14.5
         }
       }

5. Response Processing
   ├─ Receive via ZMQ
   ├─ Deserialize from MessagePack
   ├─ Transform to OpenAI format
   │
   └─ Build OpenAIResponse
       {
         "id": "chatcmpl-123",
         "object": "chat.completion",
         "created": 1234567890,
         "model": "llama3",
         "choices": [{
           "index": 0,
           "message": {
             "role": "assistant",
             "content": "Hello! How can I help you?"
           },
           "finish_reason": "stop"
         }],
         "usage": {
           "prompt_tokens": 10,
           "completion_tokens": 7,
           "total_tokens": 17
         },
         "performance": {
           "prefill_time_ms": 45.2,
           "decode_time_ms": 123.8,
           "tokens_per_second": 14.5,
           "queue_time_ms": 0
         }
       }

6. HTTP Response
   Server ──► 200 OK
              {
                "id": "chatcmpl-123",
                "choices": [...],
                "usage": {...},
                "performance": {...}
              }

   Client receives and displays response
```

### Metrics Collection Flow

```
1. Client Request
   GET /v1/metrics

2. Server Processing
   ├─ Collect CPU stats (sysinfo)
   │   ├─ Utilization % per core
   │   └─ Total core count
   │
   ├─ Collect memory stats
   │   ├─ Used / Total GB
   │   └─ Utilization %
   │
   ├─ Collect GPU stats (if CUDA enabled)
   │   ├─ Utilization %
   │   ├─ Memory used/total
   │   └─ Temperature
   │
   ├─ Query engine metrics (ZMQ)
   │   ├─ Request counts (total, completed, failed)
   │   ├─ Token statistics
   │   ├─ Average timings (queue, prefill, decode)
   │   ├─ Queue status
   │   └─ Cache statistics
   │       ├─ Hit rate
   │       ├─ Size distribution
   │       └─ Eviction count
   │
   └─ Aggregate into SystemMetrics

3. HTTP Response
   {
     "cpu": {
       "utilization_percent": 45.2,
       "core_count": 8
     },
     "memory": {
       "used_gb": 8.5,
       "total_gb": 16.0,
       "utilization_percent": 53.1
     },
     "gpu": {
       "utilization_percent": 67.3,
       "memory_used_gb": 12.2,
       "memory_total_gb": 24.0,
       "temperature_c": 72
     },
     "engine": {
       "total_requests": 1234,
       "completed_requests": 1200,
       "failed_requests": 34,
       "total_tokens_generated": 1234567,
       "avg_queue_time_ms": 12.5,
       "avg_prefill_time_ms": 45.2,
       "avg_decode_time_ms": 123.8,
       "queue_size": 3,
       "queue_capacity": 100,
       "concurrent_requests": 2
     },
     "cache": {
       "hit_rate": 0.85,
       "hits": 850,
       "misses": 150,
       "evictable_size": 5000,
       "protected_size": 3000,
       "total_cached": 8000,
       "total_capacity": 10000,
       "evictions": 25,
       "node_count": 42
     }
   }
```

### Error Handling Flow

```
1. Error Occurs at various stages:
   ├─ Request validation errors
   ├─ ZMQ communication errors
   ├─ Engine timeout errors
   ├─ Serialization errors
   └─ Template application errors

2. Error Propagation
   ├─ Low-level errors wrapped in anyhow::Error
   ├─ API-level errors converted to AppError
   ├─ Status codes mapped appropriately
   │   ├─ 400: Bad Request (validation)
   │   ├─ 404: Not Found (resource)
   │   ├─ 502: Bad Gateway (engine unavailable)
   │   ├─ 504: Gateway Timeout (engine timeout)
   │   └─ 500: Internal Server Error (unexpected)
   └─ Error responses formatted consistently

3. Error Response Format
   {
     "error": {
       "message": "Engine request timed out after 30s",
       "type": "timeout_error",
       "code": "engine_timeout"
     }
   }
```

---

## API Documentation

### Base URL

```
http://localhost:8000
```

### Authentication

Currently no authentication (development mode). Production deployments should add:
- API key authentication
- Rate limiting
- Request signing

### Endpoints

#### 1. Chat Completion

Creates a model response for the given chat conversation.

```http
POST /v1/chat/completions
Content-Type: application/json
```

**Request Body:**
```json
{
  "model": "llama3",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stream": false
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model ID to use (e.g., "llama3") |
| `messages` | array | Yes | - | Array of message objects |
| `max_tokens` | integer | No | 512 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0-2) |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | No | 50 | Top-k sampling value |
| `stream` | boolean | No | false | Enable streaming response |

**Message Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | "system", "user", or "assistant" |
| `content` | string | Yes | Message content |

**Response (200 OK):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699017642,
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 10,
    "total_tokens": 32
  },
  "performance": {
    "prefill_time_ms": 45.2,
    "decode_time_ms": 123.8,
    "queue_time_ms": 0,
    "tokens_per_second": 14.5,
    "batch_size": 1
  }
}
```

**Performance Fields (RustInfer extension):**

| Field | Type | Description |
|-------|------|-------------|
| `prefill_time_ms` | float | Time to process prompt tokens |
| `decode_time_ms` | float | Time to generate completion tokens |
| `queue_time_ms` | float | Time spent in request queue |
| `tokens_per_second` | float | Overall generation throughput |
| `batch_size` | integer | Batch size used for inference |

**Error Response (400 Bad Request):**
```json
{
  "error": {
    "message": "Invalid model parameter",
    "type": "invalid_request_error"
  }
}
```

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

#### 2. List Models

Lists the currently available models.

```http
GET /v1/models
```

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3",
      "object": "model",
      "created": 1699017642,
      "owned_by": "rustinfer"
    }
  ]
}
```

#### 3. Get System Metrics

Retrieves comprehensive system metrics including CPU, memory, GPU, engine, and cache statistics.

```http
GET /v1/metrics
```

**Response (200 OK):**
```json
{
  "cpu": {
    "utilization_percent": 45.2,
    "core_count": 8
  },
  "memory": {
    "used_gb": 8.5,
    "total_gb": 16.0,
    "utilization_percent": 53.1
  },
  "gpu": {
    "utilization_percent": 67.3,
    "memory_used_gb": 12.2,
    "memory_total_gb": 24.0,
    "temperature_c": 72
  },
  "engine": {
    "total_requests": 1234,
    "completed_requests": 1200,
    "failed_requests": 34,
    "total_tokens_generated": 1234567,
    "avg_queue_time_ms": 12.5,
    "avg_prefill_time_ms": 45.2,
    "avg_decode_time_ms": 123.8,
    "queue_size": 3,
    "queue_capacity": 100,
    "concurrent_requests": 2
  },
  "cache": {
    "hit_rate": 0.85,
    "hits": 850,
    "misses": 150,
    "evictable_size": 5000,
    "protected_size": 3000,
    "total_cached": 8000,
    "total_capacity": 10000,
    "evictions": 25,
    "node_count": 42
  }
}
```

#### 4. Health Check

Basic health check endpoint.

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "mode": "api_server",
  "version": "0.1.0"
}
```

#### 5. Readiness Check

Checks if the server is ready to accept requests and connected to the engine.

```http
GET /ready
```

**Response (200 OK):**
```json
{
  "ready": true,
  "engine_connected": true
}
```

**Response (503 Service Unavailable):**
```json
{
  "ready": false,
  "engine_connected": false
}
```

---

## Component Structure

### Directory Layout

```
crates/infer-server/
├── src/
│   ├── main.rs              # Entry point, server initialization
│   ├── lib.rs               # Public API exports
│   ├── api/
│   │   ├── mod.rs           # API module exports
│   │   ├── openai.rs        # OpenAI-compatible endpoints
│   │   ├── health.rs        # Health check endpoints
│   │   └── metrics.rs       # Metrics collection endpoint
│   ├── chat/
│   │   ├── mod.rs           # Chat module exports
│   │   └── template.rs      # Chat template system
│   └── zmq_client.rs        # ZMQ client implementation
├── Cargo.toml               # Dependencies
└── README.md                # This file
```

### Key Components

#### 1. `src/main.rs` - Server Entry Point

**Responsibilities:**
- Parse command-line arguments and environment variables
- Initialize logging system
- Create and configure ZMQ client
- Build Axum router with all endpoints
- Apply middleware (CORS, tracing)
- Start HTTP server
- Handle graceful shutdown

**Key Code:**
```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse arguments
    let args = Args::parse();

    // Setup logging
    init_tracing(&args.log_level);

    // Create ZMQ client
    let zmq_client = Arc::new(ZmqClient::new(&args.engine_endpoint).await?);

    // Build router
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/metrics", get(get_metrics))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .layer(CorsLayer::permissive())
        .with_state(zmq_client);

    // Start server
    let listener = TcpListener::bind((args.host, args.port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

#### 2. `src/zmq_client.rs` - ZMQ Communication

**Responsibilities:**
- Manage ZMQ DEALER socket connection
- Handle async request/response cycle
- Serialize/deserialize messages with MessagePack
- Track pending requests with timeouts
- Thread-safe via Arc<RwLock>

**Key Structures:**
```rust
pub struct ZmqClient {
    sender: mpsc::Sender<ClientRequest>,
    pending: Arc<RwLock<HashMap<String, PendingRequest>>>,
}

struct PendingRequest {
    response_tx: oneshot::Sender<Result<EngineResponse>>,
    created_at: Instant,
}

impl ZmqClient {
    // Send inference request
    pub async fn send_inference(
        &self,
        request: InferenceRequest
    ) -> Result<InferenceResponse>;

    // Query engine metrics
    pub async fn get_metrics(&self) -> Result<EngineMetrics>;
}
```

#### 3. `src/api/openai.rs` - OpenAI API

**Responsibilities:**
- Parse OpenAI-compatible requests
- Apply chat templates
- Transform to internal protocol
- Handle ZMQ communication
- Transform responses back to OpenAI format
- Extract and embed performance metrics

**Chat Template Application:**
```rust
// Llama3 template example
fn apply_llama3_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    prompt.push_str("<|begin_of_text|>");

    for message in messages {
        match message.role.as_str() {
            "system" => {
                prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
            }
            "user" => {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
            }
            "assistant" => {
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            _ => continue,
        }
        prompt.push_str(&message.content);
        prompt.push_str("<|eot_id|>");
    }

    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}
```

#### 4. `src/api/metrics.rs` - Metrics Collection

**Responsibilities:**
- Collect CPU statistics via `sysinfo`
- Collect memory statistics via `sysinfo`
- Collect GPU statistics via `nvml-wrapper` (optional)
- Query engine metrics via ZMQ
- Aggregate into unified response

**Example Implementation:**
```rust
pub async fn get_metrics(
    State(client): State<Arc<ZmqClient>>,
) -> Json<SystemMetrics> {
    // Get system metrics
    let sys = sysinfo::System::new_all();
    sys.refresh_cpu();
    sys.refresh_memory();

    // Get GPU metrics (if available)
    let gpu_metrics = get_gpu_metrics().await;

    // Get engine metrics via ZMQ
    let engine_metrics = client.get_metrics().await.unwrap_or_default();

    // Aggregate
    SystemMetrics {
        cpu: CpuMetrics::from_system(&sys),
        memory: MemoryMetrics::from_system(&sys),
        gpu: gpu_metrics,
        engine: engine_metrics.engine,
        cache: engine_metrics.cache,
    }
}
```

#### 5. `src/api/health.rs` - Health Checks

**Responsibilities:**
- Simple health status endpoint
- Readiness check with engine connectivity
- Service mode and version reporting

---

## Configuration

### Command-Line Arguments

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--host` | `HOST` | `0.0.0.0` | Server bind address |
| `--port` | `PORT` | `8000` | Server port |
| `--engine-endpoint` | `ENGINE_ENDPOINT` | `ipc:///tmp/rustinfer.ipc` | ZMQ engine endpoint |
| `--log-level` | `RUST_LOG` | `info` | Logging level (trace, debug, info, warn, error) |

### Example Usage

```bash
# Using CLI arguments
cargo run -p infer-server -- \
  --host 0.0.0.0 \
  --port 8000 \
  --engine-endpoint ipc:///tmp/rustinfer.ipc \
  --log-level debug

# Using environment variables
export HOST=0.0.0.0
export PORT=8000
export ENGINE_ENDPOINT=ipc:///tmp/rustinfer.ipc
export RUST_LOG=debug
cargo run -p infer-server

# Mixed (CLI overrides env vars)
export ENGINE_ENDPOINT=tcp://127.0.0.1:5555
cargo run -p infer-server -- --port 9000
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `cuda` | Enable CUDA GPU support and GPU metrics | Yes |

**Disable CUDA (CPU-only mode):**
```bash
cargo run -p infer-server --no-default-features
```

### Logging

Structured logging using `tracing` crate:

```bash
# Log level hierarchy: trace > debug > info > warn > error
export RUST_LOG=info                    # Info and above
export RUST_LOG=debug                   # Debug and above
export RUST_LOG=rustinfer_server=debug  # Module-specific
export RUST_LOG=trace                   # All messages
```

---

## Development Guide

### Prerequisites

```bash
# Rust toolchain
rustup update stable
rustup target add x86_64-unknown-linux-gnu

# ZeroMQ
sudo apt-get install libzmq3-dev  # Ubuntu/Debian

# CUDA (optional, for GPU support)
# Install CUDA Toolkit from NVIDIA

# Development tools
cargo install cargo-watch
cargo install cargo-expand
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd RustInfer

# Build all components
cargo build --release

# Build only server
cargo build -p infer-server --release
```

### Running Locally

```bash
# Terminal 1: Start inference engine
cargo run -p infer-scheduler

# Terminal 2: Start API server
cargo run -p infer-server

# Terminal 3: Test with cURL
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Development Workflow

```bash
# Auto-reload on file changes
cargo watch -x run -p infer-server

# Run tests
cargo test -p infer-server

# Run with debug logging
RUST_LOG=debug cargo run -p infer-server

# Build without CUDA (CPU mode)
cargo build -p infer-server --no-default-features
```

### Adding a New Endpoint

1. **Create handler function** in appropriate module:
```rust
// src/api/custom.rs
pub async fn custom_endpoint(
    State(client): State<Arc<ZmqClient>>,
    Json(request): Json<CustomRequest>,
) -> Result<Json<CustomResponse>, AppError> {
    // Your logic here
    Ok(Json(response))
}
```

2. **Register in router** (`main.rs`):
```rust
let app = Router::new()
    .route("/api/custom", post(custom_endpoint))
    // ... other routes
```

3. **Add types** (if needed):
```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct CustomRequest {
    pub field: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CustomResponse {
    pub result: String,
}
```

### Adding a New Chat Template

1. **Define template variant** (`src/chat/template.rs`):
```rust
pub enum ChatTemplate {
    Llama3,
    GPT4,  // New template
}
```

2. **Implement template application**:
```rust
impl ChatTemplate {
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Llama3 => apply_llama3_template(messages),
            ChatTemplate::GPT4 => apply_gpt4_template(messages),
        }
    }
}

fn apply_gpt4_template(messages: &[ChatMessage]) -> String {
    // Implement GPT-4 template logic
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant:");
    prompt
}
```

3. **Update factory** to select based on model:
```rust
pub fn get_template(model: &str) -> Option<ChatTemplate> {
    match model {
        "llama3" => Some(ChatTemplate::Llama3),
        "gpt4" => Some(ChatTemplate::GPT4),
        _ => None,
    }
}
```

### Testing

```bash
# Unit tests
cargo test -p infer-server

# Integration tests
cargo test --test '*'

# Specific test
cargo test test_chat_completion

# Run tests with output
cargo test -- --nocapture

# Run tests with debug logging
RUST_LOG=debug cargo test
```

### Debugging

```bash
# Enable detailed logging
RUST_LOG=debug,tokio=trace cargo run -p infer-server

# Debug specific module
RUST_LOG=rustinfer_server::zmq_client=trace cargo run

# Use rust-lldb (Linux) or lldb (macOS)
rust-lldb target/debug/infer-server
```

---

## Deployment

### Production Build

```bash
# Optimized release build
cargo build -p infer-server --release

# Output: target/release/infer-server
```

### Systemd Service

Create `/etc/systemd/system/rustinfer-server.service`:

```ini
[Unit]
Description=RustInfer API Server
After=network.target

[Service]
Type=simple
User=rustinfer
WorkingDirectory=/opt/rustinfer
Environment="HOST=0.0.0.0"
Environment="PORT=8000"
Environment="ENGINE_ENDPOINT=ipc:///tmp/rustinfer.ipc"
Environment="RUST_LOG=info"
ExecStart=/opt/rustinfer/infer-server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rustinfer-server
sudo systemctl start rustinfer-server

# Check status
sudo systemctl status rustinfer-server

# View logs
sudo journalctl -u rustinfer-server -f
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY . .
RUN cargo build --release -p infer-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libzmq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/infer-server .
COPY --from=builder /app/crates/infer-server/ .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV ENGINE_ENDPOINT=ipc:///tmp/rustinfer.ipc

EXPOSE 8000

CMD ["./infer-server"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  server:
    build:
      context: .
      dockerfile: Dockerfile.server
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - ENGINE_ENDPOINT=tcp://engine:5555
      - RUST_LOG=info
    depends_on:
      - engine

  engine:
    build:
      context: .
      dockerfile: Dockerfile.engine
    environment:
      - ZMQ_ENDPOINT=tcp://*:5555
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

**Run:**
```bash
docker-compose up -d
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.rustinfer.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Monitoring

**Prometheus Metrics** (planned):
- HTTP request metrics
- Response times
- Error rates
- Engine metrics

**Current Monitoring Options**:
- `/v1/metrics` endpoint for custom monitoring
- Structured logs
- Health check endpoints

---

## Troubleshooting

### Common Issues

#### 1. "Engine request timeout"

**Symptoms:**
- 504 Gateway Timeout response
- Logs show "Engine request timed out after 30s"

**Solutions:**
- Verify engine is running: `ps aux | grep infer-scheduler`
- Check ZMQ endpoint configuration matches
- Increase timeout in `zmq_client.rs` if needed
- Monitor engine CPU/GPU utilization

#### 2. "Connection refused" to ZMQ endpoint

**Symptoms:**
- Server fails to start
- Error: "Address already in use" or "Connection refused"

**Solutions:**
```bash
# Check if endpoint exists
ls -la /tmp/rustinfer.ipc

# Remove stale socket (if exists)
rm /tmp/rustinfer.ipc

# Check engine is running
ps aux | grep infer-scheduler

# Use tcp:// instead of ipc:// for network communication
ENGINE_ENDPOINT=tcp://127.0.0.1:5555 cargo run
```

#### 3. CORS errors in browser

**Symptoms:**
- Browser console shows CORS errors
- Frontend cannot connect to API

**Solutions:**
- CORS is enabled with `CorsLayer::permissive()` in development
- For production, configure specific origins:
```rust
CorsLayer::new()
    .allow_origin("https://yourdomain.com".parse::<HeaderValue>().unwrap())
    .allow_methods([Method::GET, Method::POST])
    .allow_headers(Any)
```

#### 4. GPU metrics not showing

**Symptoms:**
- `/v1/metrics` returns `gpu: null`

**Solutions:**
- Verify CUDA is installed: `nvidia-smi`
- Check feature flag: `cargo build -p infer-server --features cuda`
- Check GPU is accessible to process

#### 5. High memory usage

**Symptoms:**
- Server process consuming lots of RAM
- OOM killer kills process

**Solutions:**
- Monitor request queue size
- Reduce max concurrent requests
- Check for memory leaks with `valgrind` or `heaptrack`
- Profile with `cargo flamegraph`

### Debug Mode

```bash
# Enable all debug output
RUST_LOG=debug,tokio=trace cargo run -p infer-server

# Monitor ZMQ communication
# Add logging to zmq_client.rs

# Check engine connection
curl http://localhost:8000/ready

# View metrics
curl http://localhost:8000/v1/metrics | jq
```

### Performance Tuning

```rust
// Increase ZMQ timeout (zmq_client.rs)
const ZMQ_RECV_TIMEOUT_MS: i32 = 50;  // Increase from 10

// Adjust request timeout
const REQUEST_TIMEOUT_SECS: u64 = 60;  // Increase from 30

// Tune thread pool (main.rs)
let runtime = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(num_cpus::get() * 2)
    .build()?;
```

---

## Architecture Decisions FAQ

### Why separate server and engine processes?

1. **Isolation**: Engine crashes don't take down the API
2. **Scaling**: Can run multiple engines behind one server
3. **Flexibility**: Can swap engines without restarting API
4. **Language**: Engine could be in different language (e.g., Python)
5. **Deployment**: Can deploy to different machines

### Why ZeroMQ instead of gRPC?

1. **Simplicity**: Easier to set up and debug
2. **Performance**: Lower overhead for our use case
3. **Flexibility**: Multiple patterns (request-response, pub-sub, etc.)
4. **Language Support**: Available in most languages

### Why MessagePack instead of JSON?

1. **Performance**: Faster serialization/deserialization
2. **Size**: Smaller message payloads
3. **Schema**: Built-in schema validation

### Why OpenAI-compatible API?

1. **Ecosystem**: Existing clients work out-of-the-box
2. **Migration**: Low barrier for users
3. **Tooling**: Compatible with OpenAI SDKs and tools

---

## Contributing

When contributing to `infer-server`:

1. **Code Style**: Follow Rust naming conventions
2. **Tests**: Add unit tests for new features
3. **Docs**: Update this README with changes
4. **API Stability**: Maintain OpenAI compatibility
5. **Error Handling**: Use `anyhow` for errors, provide helpful messages

---

## License

Part of the RustInfer project. See main LICENSE file.

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/RustInfer/issues
- Documentation: See main `docs/` directory
- Protocol: See `infer-protocol/` crate
