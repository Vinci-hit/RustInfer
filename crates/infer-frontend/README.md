# RustInfer Frontend

A modern, reactive web interface for the RustInfer LLM inference engine, built with Rust and Dioxus.

## Overview

The RustInfer Frontend is a WebAssembly-based web application that provides:
- **Interactive Chat Interface**: Real-time conversation with LLM models
- **Admin Console**: Comprehensive system monitoring dashboard
- **Real-time Metrics**: Live CPU, memory, GPU, and cache statistics
- **Rich Content Support**: Markdown rendering, syntax highlighting, math equations, and diagrams

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Dioxus 0.7.1 | Reactive web framework for Rust |
| **Build Target** | WebAssembly | High-performance browser execution |
| **Styling** | Tailwind CSS | Utility-first CSS framework |
| **HTTP Client** | `reqwest`, `gloo-net` | REST API and Server-Sent Events |
| **State Management** | Dioxus Signals | Reactive state management |
| **Markdown** | `comrak` | GitHub Flavored Markdown rendering |
| **Syntax Highlighting** | `syntect` | Code block highlighting |
| **Math Rendering** | KaTeX | Client-side LaTeX rendering |
| **Diagrams** | Mermaid.js | Diagram rendering |

## Architecture

### Directory Structure

```
crates/infer-frontend/
├── src/
│   ├── main.rs                 # Application entry point and router
│   ├── components/             # UI components
│   │   ├── chat_interface.rs   # Main chat UI
│   │   ├── message_bubble.rs   # Individual message display
│   │   ├── metrics_panel.rs    # Real-time metrics display
│   │   ├── admin_console.rs    # Admin dashboard
│   │   ├── streaming_message.rs # Streaming response handling
│   │   ├── code_block.rs       # Code blocks with syntax highlighting
│   │   ├── mermaid_diagram.rs  # Mermaid diagram rendering
│   │   └── streaming_indicator.rs # Loading indicators
│   ├── state/                  # State management
│   │   ├── conversation.rs     # Message data structures
│   │   └── metrics.rs          # System metrics structures
│   ├── api/                    # API integration
│   │   └── client.rs           # HTTP client and API types
│   └── utils/                  # Utilities
│       └── markdown.rs         # Markdown rendering
├── assets/                     # Static assets
│   ├── main.css               # Tailwind input
│   └── output.css             # Compiled CSS
├── public/                     # Public files
│   ├── index.html             # Entry HTML
│   ├── katex-init.js          # KaTeX initialization
│   └── mermaid-init.js        # Mermaid initialization
├── Cargo.toml                 # Rust dependencies
├── Dioxus.toml                # Dioxus configuration
├── tailwind.config.js         # Tailwind configuration
├── dev.sh                     # Development script
└── build.sh                   # Build script
```

### Component Hierarchy

```
App (main.rs)
├── Router
│   ├── Chat Interface
│   │   ├── Message List
│   │   │   └── Message Bubble
│   │   │       ├── Markdown Content
│   │   │       ├── Code Block
│   │   │       └── Mermaid Diagram
│   │   ├── Input Area
│   │   └── Streaming Indicator
│   └── Metrics Panel (side panel)
│
└── Admin Console (separate route)
    ├── CPU/Memory/GPU Gauges
    ├── RadixCache Metrics
    └── Engine Performance Stats
```

## Dataflow

### Chat Request Flow

```
User Input
    ↓
ChatInterface Component
    ↓
[State Update] Add user message to messages signal
    ↓
ApiClient.send_message()
    ↓
HTTP POST /v1/chat/completions
    {
      "model": "llama3",
      "messages": [...],
      "max_tokens": 512,
      "stream": false
    }
    ↓
Backend (infer-server)
    ↓
Response received
    ↓
Parse ChatResponse → Extract content & metrics
    ↓
[State Update] Add assistant message with metrics
    ↓
UI re-render (reactive)
```

### Metrics Polling Flow

```
use_future hook (polling)
    ↓
Every 2 seconds (Metrics Panel)
Every 1 second (Admin Console)
    ↓
GET /v1/metrics
    ↓
SystemMetrics response
    {
      "cpu": {...},
      "memory": {...},
      "gpu": {...},
      "cache": {...},
      "engine": {...}
    }
    ↓
[State Update] Update metrics signal
    ↓
UI re-render with new values
```

### State Management Pattern

```rust
// Create reactive signals
let mut messages = use_signal(Vec::<Message>::new);
let mut input_text = use_signal(String::new);
let mut is_generating = use_signal(|| false);

// Reading from signals (automatically tracked)
let message_list = messages.read();

// Writing to signals (triggers re-render)
messages.write().push(new_message);

// Async operations with use_future
let _ = use_future(|| async move {
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        // Fetch and update metrics
    }
});
```

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Chat Completion
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama3",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 512,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    }
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  },
  "performance": {
    "prefill_time_ms": 45.2,
    "decode_time_ms": 123.8,
    "tokens_per_second": 14.5
  }
}
```

#### 2. System Metrics
```http
GET /v1/metrics
```

**Response:**
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
  "cache": {
    "hit_rate": 0.85,
    "total_entries": 1000,
    "evicted_count": 50
  },
  "engine": {
    "active_requests": 3,
    "total_requests": 1234,
    "avg_latency_ms": 245.6
  }
}
```

### Client Types

```rust
// API Request
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub stream: bool,
}

// API Response
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub performance: Option<PerformanceMetrics>,
}

// Message
pub struct Message {
    pub id: Uuid,
    pub role: MessageRole,  // User | Assistant
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metrics: Option<MessageMetrics>,
}

// System Metrics
pub struct SystemMetrics {
    pub cpu: CpuMetrics,
    pub memory: MemoryMetrics,
    pub gpu: Option<GpuMetrics>,
    pub cache: Option<CacheMetrics>,
    pub engine: EngineMetrics,
}
```

## Development Guide

### Prerequisites

```bash
# Rust toolchain
rustup install stable
rustup target add wasm32-wasi

# Node.js (for Tailwind CSS)
node --version  # Should be >= 18

# Dioxus CLI
cargo install dioxus-cli

# Additional tools
npm install -g tailwindcss
```

### Setup

```bash
# Install dependencies
cd ~/RustInfer/crates/infer-frontend
npm install

# Build Tailwind CSS (once)
npx tailwindcss -i ./assets/main.css -o ./assets/output.css
```

### Running Development Server

**From workspace root:**
```bash
cd ~/RustInfer
dx serve --package infer-frontend
```

**From frontend directory (using script):**
```bash
cd ~/RustInfer/crates/infer-frontend
./dev.sh
```

**Access the app:**
- Chat Interface: http://localhost:3000
- Admin Console: Click "Admin Console" tab

### Development Workflow

1. **Start backend server** (port 8000):
   ```bash
   cd ~/RustInfer
   cargo run -p infer-server
   ```

2. **Start frontend** (port 3000):
   ```bash
   cd ~/RustInfer/crates/infer-frontend
   ./dev.sh
   ```

3. **Make changes** to Rust files → auto-reload via Dioxus
4. **Make changes** to CSS → auto-recompile via Tailwind watcher

### File Watching

| File Type | Watcher | Reload |
|-----------|---------|--------|
| `.rs` files | Dioxus dev server | Full hot reload |
| `.css` files | Tailwind watcher | CSS only |
| `.html` files | Dioxus dev server | Full hot reload |

## Building for Production

### Build WASM

```bash
cd ~/RustInfer/crates/infer-frontend

# Optimize CSS
NODE_ENV=production npx tailwindcss \
  -i ./assets/main.css \
  -o ./assets/output.css \
  --minify

# Build with Dioxus (optimized)
dx build --release

# Output: dist/ directory
```

### Deploy

The `dist/` directory contains:
- `index.html` - Entry point
- `assets/infer-frontend.js` - Compiled WASM bindings
- `assets/infer-frontend_bg.wasm` - WASM binary
- `assets/output.css` - Optimized CSS

Deploy to any static hosting service:
- GitHub Pages
- Netlify
- Vercel
- nginx/Apache

## Component Reference

### ChatInterface

**Location:** `src/components/chat_interface.rs`

**Responsibilities:**
- Manage conversation state
- Handle user input
- Communicate with API
- Display loading indicators

**Key Signals:**
- `messages: Vec<Message>` - Conversation history
- `input_text: String` - Current input
- `is_generating: bool` - Loading state

**Example:**
```rust
let mut messages = use_signal(Vec::<Message>::new);
let mut input_text = use_signal(String::new);

let send_message = move |_| {
    // Add user message
    messages.write().push(user_message);

    // Call API
    spawn(async move {
        let response = api_client.send_message(&messages).await;
        messages.write().push(assistant_message);
    });
};
```

### MessageBubble

**Location:** `src/components/message_bubble.rs`

**Responsibilities:**
- Render individual messages
- Role-based styling (user/assistant)
- Display performance metrics
- Render markdown content

**Props:**
- `message: &Message` - Message to display

### MetricsPanel

**Location:** `src/components/metrics_panel.rs`

**Responsibilities:**
- Display real-time system metrics
- Poll metrics every 2 seconds
- Show CPU, memory, GPU usage

**Example:**
```rust
let metrics = use_signal(SystemMetrics::default);

let _ = use_future(|| async move {
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let new_metrics = api_client.get_metrics().await?;
        metrics.set(new_metrics);
    }
});
```

### AdminConsole

**Location:** `src/components/admin_console.rs`

**Responsibilities:**
- Advanced monitoring dashboard
- High-frequency metrics (1 second polling)
- Historical data visualization
- Circular SVG gauges
- RadixCache metrics
- Engine performance stats

**Features:**
- 60-second history buffer
- Auto-refresh toggle
- Color-coded status indicators
- Smooth animations

## Customization

### Styling

**Tailwind Configuration** (`tailwind.config.js`):
```javascript
module.exports = {
  content: [
    "./src/**/*.rs",
    "./index.html",
    "./assets/**/*.{css,js}"
  ],
  theme: {
    extend: {
      colors: {
        'brand': '#3b82f6',
        // Add custom colors
      }
    }
  }
}
```

### API Configuration

**Change backend URL** (`src/api/client.rs`):
```rust
const API_BASE: &str = "http://localhost:8000";

// Or use environment variable
const API_BASE: &str = std::env!("API_URL");
```

### Model Configuration

**Change default model** (`src/components/chat_interface.rs`):
```rust
let request = ChatRequest {
    model: "llama3".to_string(),  // Change this
    messages: history,
    max_tokens: 512,
    stream: false,
};
```

## Troubleshooting

### `dx serve` fails with workspace error

**Problem:** Dioxus CLI can't find the package in workspace.

**Solution:** Run from workspace root with `--package` flag:
```bash
cd ~/RustInfer
dx serve --package infer-frontend
```

Or use the dev script:
```bash
cd ~/RustInfer/crates/infer-frontend
./dev.sh
```

### WASM build fails with katex error

**Problem:** `katex` crate requires WASM target.

**Solution:** Ensure you're building for the correct target:
```bash
rustup target add wasm32-wasi
cargo build --target wasm32-wasi
```

### Tailwind CSS not updating

**Problem:** CSS changes not reflected.

**Solution:** Restart Tailwind watcher:
```bash
pkill -f tailwindcss
npx tailwindcss -i ./assets/main.css -o ./assets/output.css --watch
```

### API connection refused

**Problem:** Frontend can't connect to backend.

**Solution:** Ensure backend is running on port 8000:
```bash
cd ~/RustInfer
cargo run -p infer-server
```

### CORS errors in browser

**Problem:** Browser blocks cross-origin requests.

**Solution:** Configure backend CORS settings in `infer-server`:
```rust
// Allow requests from frontend
HttpServer::new(move || {
    App::new()
        .wrap(Cors::permissive())
        // ...
})
```

## Performance Optimization

### WASM Binary Size

Current size: ~2-3 MB (uncompressed)

**Optimization techniques:**
- Use `lto = true` in release profile
- Enable `opt-level = "z"` for size optimization
- Use `wasm-opt` for additional optimization:
  ```bash
  wasm-opt -Oz -o output.wasm input.wasm
  ```

### Metrics Polling

**Current intervals:**
- Metrics Panel: 2 seconds
- Admin Console: 1 second

**Adjust polling frequency** in respective components:
```rust
// Faster polling (1 second)
tokio::time::sleep(Duration::from_secs(1)).await;

// Slower polling (5 seconds)
tokio::time::sleep(Duration::from_secs(5)).await;
```

### Bundle Size

**Current breakdown:**
- Dioxus framework: ~500 KB
- Application code: ~100 KB
- Dependencies: ~1.5 MB
- Total: ~2.1 MB

## Future Enhancements

### Planned Features

1. **Streaming Responses**
   - SSE support for real-time token streaming
   - `use_stream_response` hook ready for implementation

2. **Authentication**
   - User login/logout
   - API key management
   - Session persistence

3. **Conversation History**
   - Local storage persistence
   - Export/import conversations
   - Search functionality

4. **Multi-model Support**
   - Model selection dropdown
   - Model-specific parameters
   - Custom endpoint configuration

5. **Advanced Admin Features**
   - Request logs viewer
   - Configuration editor
   - System health alerts
   - Performance charts

### Contributing

When adding new features:

1. **Components:** Place in `src/components/`
2. **State:** Define types in `src/state/`
3. **API:** Add to `src/api/client.rs`
4. **Styles:** Use Tailwind utility classes
5. **Tests:** Add unit tests in `src/` directory
6. **Docs:** Update this README

## License

Part of the RustInfer project. See main LICENSE file.
