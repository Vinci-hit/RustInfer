# RustInfer Frontend - Web UI with Real-Time Observability

> Modern Dioxus-based web application for interactive LLM inference with comprehensive performance monitoring.

## 🎯 Features

- **Multi-Round Conversations** - Interactive chat with full conversation history
- **Real-Time Performance Metrics** - Display per-message metrics:
  - ⚡ Prefill time (prompt processing)
  - 🔄 Decode time (token generation)
  - 🚀 Tokens per second
  - 📝 Total tokens generated
- **System Monitoring Dashboard** - Live metrics updated every 2 seconds:
  - CPU usage and core count
  - Memory usage (used/total/available)
  - GPU utilization and VRAM (when available)
  - GPU temperature
- **Modern UI** - Tailwind CSS with dark theme
- **Responsive Layout** - 2/3 chat interface, 1/3 metrics panel
- **WASM-Based** - Runs entirely in the browser, no backend rendering needed

## 🚀 Quick Start

```bash
# 1. Start the backend server (in project root)
cargo run --release --bin rustinfer-server -- \
    --model /path/to/llama3/model \
    --port 8000

# 2. Start the frontend dev server
cd crates/infer-frontend
dx serve --port 3000

# 3. Open your browser
# Navigate to: http://localhost:3000
```

## 📦 Dependencies

- **Dioxus 0.6** - Reactive web framework
- **reqwest** - HTTP client (WASM-compatible)
- **gloo-net** - Web APIs for Rust (EventSource for future streaming)
- **Tailwind CSS** - Utility-first CSS framework

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           Browser (WASM Application)                │
│                                                       │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │  Chat Interface  │      │  Metrics Panel   │    │
│  │  (2/3 width)     │      │  (1/3 width)     │    │
│  │                  │      │                  │    │
│  │  • Message list  │      │  • CPU usage     │    │
│  │  • Input box     │      │  • Memory stats  │    │
│  │  • Metrics per   │      │  • GPU info      │    │
│  │    message       │      │  • Polls /v1/    │    │
│  │                  │      │    metrics       │    │
│  └────────┬─────────┘      └────────┬─────────┘    │
│           │                         │               │
└───────────┼─────────────────────────┼───────────────┘
            │ POST                    │ GET (every 2s)
            │ /v1/chat/completions    │ /v1/metrics
            ▼                         ▼
┌─────────────────────────────────────────────────────┐
│         Backend Server (localhost:8000)             │
│                                                      │
│  • Inference engine with Llama3 model               │
│  • OpenAI-compatible API                            │
│  • Performance metrics collection                   │
│  • System resource monitoring                       │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
crates/infer-frontend/
├── Cargo.toml              # Dependencies
├── Dioxus.toml             # Dioxus build configuration
├── README.md               # This file
├── src/
│   ├── main.rs             # App entry point & layout
│   ├── api/
│   │   └── client.rs       # HTTP client for backend
│   ├── state/
│   │   ├── conversation.rs # Message types & state
│   │   └── metrics.rs      # System metrics types
│   └── components/
│       ├── chat_interface.rs       # Main chat UI
│       ├── message_bubble.rs       # Individual message with metrics
│       ├── metrics_panel.rs        # System monitoring dashboard
│       └── streaming_indicator.rs  # Loading animation
└── assets/
    └── main.css            # Tailwind styles
```

## 🔧 Development

### Building

```bash
# Development build
dx build

# Release build (optimized WASM)
dx build --release

# Output located at:
# target/dx/rustinfer-frontend/release/web/public/
```

### Running

```bash
# Development server with hot reload
dx serve --port 3000

# The server will automatically rebuild on file changes
```

### Testing

```bash
# Ensure backend is running first
cargo run --release --bin rustinfer-server -- --model /path/to/model

# Then access the frontend at http://localhost:3000
# Test the following:
# 1. Send a message and verify response appears
# 2. Check that metrics display for assistant messages
# 3. Verify system metrics panel updates every 2 seconds
# 4. Test multiple conversation rounds
```

## 💡 Usage

### Sending Messages

1. Type your message in the input box
2. Click "Send" or press Enter
3. Watch the streaming indicator while the model generates
4. See the response with detailed performance metrics

### Reading Performance Metrics

Each assistant message displays:
- **Prefill Time**: Time to process your prompt (ms)
- **Decode Time**: Time to generate the response (ms)
- **Speed**: Tokens generated per second
- **Tokens**: Total number of tokens in the response

### Monitoring System Resources

The right panel shows live system metrics:
- **CPU**: Current utilization % and core count
- **Memory**: Used vs total RAM
- **GPU**: Utilization %, VRAM usage, temperature (if available)

## 🎨 Customization

### Styling

Edit `assets/main.css` to customize:
```css
/* Change color scheme */
.bg-gray-900 { background-color: #your-color; }

/* Modify animations */
@keyframes bounce { ... }
```

### API Configuration

The frontend expects the backend at `http://localhost:8000`. To change:

Edit `src/api/client.rs`:
```rust
impl ApiClient {
    pub fn new() -> Self {
        Self {
            base_url: "http://your-backend:port".to_string(),
            // ...
        }
    }
}
```

## 🚀 Deployment

### Option 1: Static Hosting

```bash
# Build release WASM
dx build --release

# Serve the output directory
cd target/dx/rustinfer-frontend/release/web/public
python3 -m http.server 8080

# Or use any static file server (nginx, caddy, etc.)
```

### Option 2: Integrated with Backend

Configure the backend to serve the frontend:

```rust
// In backend main.rs
let app = Router::new()
    // ... API routes ...
    .nest_service("/", ServeDir::new("path/to/frontend/dist"));
```

## 🐛 Troubleshooting

### Frontend won't connect to backend

- Check that backend is running on port 8000
- Verify CORS is enabled in backend
- Check browser console for errors

### Metrics not updating

- Ensure `/v1/metrics` endpoint is accessible
- Check browser network tab for 404 errors
- Verify backend was built with metrics support

### Build errors

```bash
# Clean and rebuild
cargo clean
dx build

# Check Dioxus CLI version
dx --version  # Should be 0.6.x
```

## 📚 API Integration

The frontend uses two main endpoints:

### Chat Completion

```rust
// POST /v1/chat/completions
ChatRequest {
    model: "llama3",
    messages: vec![/* conversation history */],
    max_tokens: Some(150),
    stream: false,
}
```

### System Metrics

```rust
// GET /v1/metrics (polled every 2 seconds)
// Returns: SystemMetrics { cpu, memory, gpu, timestamp }
```

## 🤝 Contributing

See the main project README for contribution guidelines.

## 📄 License

Part of the RustInfer project. Shares the same license as the main project.

---

**Built with:**
- 🦀 Rust + WASM
- ⚡ Dioxus 0.6
- 🎨 Tailwind CSS
- 🌐 Modern web standards
