# Quick Start Guide

## Installation

```bash
cd /home/vinci/RustInfer
cargo build --release --bin rustinfer-server
```

Binary location: `./target/release/rustinfer-server`

## Basic Usage

### 1. Start the Server

```bash
./target/release/rustinfer-server \
    --model /mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct \
    --port 8000 \
    --device cuda:0 \
    --max-tokens 150
```

Expected output:
```
INFO Loading model from /mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct...
Model successfully initialized. Loading time: 9.46 seconds
INFO Model loaded successfully!
INFO 🚀 RustInfer server listening on http://0.0.0.0:8000
```

### 2. Test the Server

#### Health Check
```bash
curl http://localhost:8000/health
# Response: {"status":"healthy","service":"rustinfer-server"}
```

#### Simple Chat
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }'
```

#### Streaming Chat
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "max_tokens": 30,
    "stream": true
  }'
```

## Command-Line Options

```
OPTIONS:
    -m, --model <PATH>           Path to model directory (required)
    -p, --port <PORT>            Server port [default: 8000]
    -d, --device <DEVICE>        Device: cpu or cuda:0 [default: cuda:0]
        --host <HOST>            Server host [default: 0.0.0.0]
        --max-tokens <N>         Max tokens to generate [default: 512]
        --log-level <LEVEL>      Log level: trace|debug|info|warn|error [default: info]
```

## Environment Variables

Alternative to command-line arguments:

```bash
export MODEL_PATH=/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct
export PORT=8000
export DEVICE=cuda:0
export MAX_TOKENS=512
export RUST_LOG=info

./rustinfer-server
```

## Common Use Cases

### 1. Python Integration

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### 2. Web Application

```javascript
async function chat(message) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'llama3',
      messages: [{ role: 'user', content: message }],
      stream: false
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}
```

### 3. Kubernetes Deployment

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: rustinfer-server
spec:
  containers:
  - name: server
    image: rustinfer:latest
    command: ["./rustinfer-server"]
    args:
      - --model=/models/llama3
      - --port=8000
      - --device=cuda:0
    ports:
    - containerPort: 8000
    resources:
      limits:
        nvidia.com/gpu: 1
```

## Troubleshooting

### Server won't start

**Problem**: "Failed to load model"
```
Solution: Check that model path exists and contains:
  - config.json
  - tokenizer.json
  - model.safetensors (or sharded files)
```

**Problem**: "CUDA out of memory"
```
Solution: Use a smaller model or reduce max sequence length
  - Try --max-tokens 128 (instead of 512)
  - Use CPU: --device cpu
```

### Slow responses

**Problem**: First request takes long
```
This is normal - model needs to warm up
  - First token: ~300ms (CUDA kernel compilation)
  - Subsequent: ~50ms
```

**Problem**: All requests are slow
```
Check:
  - GPU utilization: nvidia-smi
  - BF16 support: Model should use BF16, not F32
  - Logs: RUST_LOG=debug ./rustinfer-server
```

### Connection refused

**Problem**: Can't connect to server
```bash
# Check if server is running
ps aux | grep rustinfer

# Check if port is open
netstat -tulpn | grep 8000

# Check firewall
sudo ufw allow 8000
```

## Performance Tips

1. **Use Release Build**
   ```bash
   cargo build --release  # 10x faster than debug
   ```

2. **Enable Logging Only When Needed**
   ```bash
   # Production
   RUST_LOG=warn ./rustinfer-server

   # Debug
   RUST_LOG=debug ./rustinfer-server
   ```

3. **Adjust Max Tokens**
   ```bash
   # Short responses = lower latency
   --max-tokens 50

   # Long responses = higher throughput
   --max-tokens 512
   ```

## Next Steps

- Read [README.md](./README.md) for architecture details
- Check [API Reference](./README.md#api-reference) for full endpoint documentation
- See [Development Guide](./README.md#development-guide) to contribute

## Getting Help

- GitHub Issues: Report bugs or request features
- Logs: Always include logs when reporting issues (`RUST_LOG=debug`)
- System Info: GPU model, CUDA version, Rust version
