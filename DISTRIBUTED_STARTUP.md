# RustInfer 分布式启动指南

## 概述

RustInfer 已从 all-in-one 架构拆分为三个独立进程，可以手动启动：

1. **Scheduler + Worker** (`rustinfer-server`) - 推理引擎核心
2. **API Server** (`rustinfer-api`) - HTTP API 服务
3. 三者通过 ZMQ IPC 通信

## 快速开始

### 方式 1: 自动启动（推荐）

```bash
./scripts/start_distributed.sh --model /path/to/model --device cuda:0 --port 8000
```

这会自动：
- 启动 Scheduler 和 Worker
- 启动 API Server
- 监控所有进程
- Ctrl+C 时优雅关闭所有进程

### 方式 2: 手动启动（用于调试）

查看启动命令：
```bash
bash ./scripts/manual_startup.sh /path/to/model cuda:0 info 8000
```

然后在 **3 个不同的终端** 中逐个运行命令。

**终端 1 - 启动 Scheduler 和 Worker：**
```bash
export RUST_LOG=info
rustinfer-server --model /path/to/model --device cuda:0
```

**终端 2 - 启动 API Server（需要等待 Scheduler 初始化）：**
```bash
export RUST_LOG=info
rustinfer-api \
    --model /path/to/model \
    --frontend-endpoint ipc:///tmp/rustinfer-12345-frontend.ipc \
    --port 8000
```

**终端 3 - 测试 API：**
```bash
curl http://localhost:8000/v1/models
```

## 关键配置参数

### Scheduler & Worker (rustinfer-server)

```bash
rustinfer-server --help

Options:
  --model <PATH>                    模型路径 (必需)
  --device <DEVICE>                GPU 设备，默认: cuda:0
  --max-batch-tokens <N>           最大批处理 token 数，默认: 4096
  --max-batch-seqs <N>             最大并发序列数，默认: 32
  --max-model-len <N>              模型最大长度，默认: 8192
  --chunked-prefill-size <N>       分块预填大小
  --paged-block-size <N>           分页块大小，默认: 1
  --enable-prefix-caching          启用前缀缓存
  --log-level <LEVEL>              日志级别，默认: info
```

### API Server (rustinfer-api)

```bash
rustinfer-api --help

Options:
  --model <PATH>                    模型路径 (必需)
  --frontend-endpoint <ENDPOINT>    Scheduler ZMQ 地址 (必需)
  --port <PORT>                     监听端口，默认: 8000
  --host <HOST>                     监听地址，默认: 0.0.0.0
  --request-timeout-secs <N>       请求超时，默认: 120
  --log-level <LEVEL>              日志级别，默认: info
```

## 构建

### 构建 Debug 版本
```bash
cargo build --bin rustinfer-server --bin rustinfer-api
```

### 构建 Release 版本（推荐用于生产）
```bash
cargo build --release --bin rustinfer-server --bin rustinfer-api
```

### 使用 Makefile
```bash
make -f Makefile.distributed build-release
make -f Makefile.distributed install-release
```

## 日志和调试

### 查看实时日志

**Scheduler & Worker：**
```bash
tail -f /tmp/rustinfer-<PID>/scheduler-worker.log
```

**API Server：**
```bash
tail -f /tmp/rustinfer-<PID>/api.log
```

### 调整日志级别

```bash
export RUST_LOG=debug
rustinfer-server --model /path/to/model
```

日志级别：`trace`, `debug`, `info`, `warn`, `error`

## IPC 通信

进程间通过 Unix IPC sockets 通信：

```
/tmp/rustinfer-<PID>-frontend.ipc      API ↔ Scheduler
/tmp/rustinfer-<PID>-worker-in.ipc     Scheduler → Worker
/tmp/rustinfer-<PID>-worker-out.ipc    Worker → Scheduler  
/tmp/rustinfer-<PID>-worker-control.ipc Scheduler ↔ Worker (控制)
```

### 清理孤立的 IPC 文件

```bash
rm -f /tmp/rustinfer-*.ipc
```

## 常见问题

### 问题：API Server 无法连接到 Scheduler

**症状：** `Failed to connect to scheduler`

**解决：**
1. 确保 Scheduler 已启动：`pgrep -l rustinfer`
2. 检查 IPC 文件是否存在：`ls -la /tmp/rustinfer-*.ipc`
3. 检查使用的 endpoint 是否匹配
4. 查看 Scheduler 的日志

### 问题：进程卡住或无法关闭

**解决：**
```bash
# 查看进程
pgrep -l rustinfer

# 杀死进程
pkill -f rustinfer-

# 清理 IPC 文件
rm -f /tmp/rustinfer-*.ipc
```

### 问题：模型加载失败

**症状：** `Failed to load tokenizer` 或 `resolve model type`

**解决：**
1. 确保模型目录存在
2. 确保模型目录包含 `tokenizer.json` 和 `config.json`
3. 检查文件权限：`ls -la /path/to/model/`

## 多实例运行

可以在同一台机器上运行多个 RustInfer 实例（使用不同的 GPU 和端口）：

```bash
# 实例 1 - GPU 0, 端口 8000
./scripts/start_distributed.sh --model model1 --device cuda:0 --port 8000

# 实例 2 - GPU 1, 端口 8001（另一个终端）
./scripts/start_distributed.sh --model model2 --device cuda:1 --port 8001
```

## 架构图

```
┌─────────────────────────────────────────────────┐
│           RustInfer Distributed                 │
└─────────────────────────────────────────────────┘

    Terminal 1          Terminal 2          Terminal 3
  ┌────────────┐      ┌────────────┐      ┌──────────┐
  │ Scheduler  │◄────►│   Worker   │      │ HTTP     │
  │  + Worker  │      │            │      │  Server  │
  │(main.rs)   │      │ (spawned)  │      │(api.rs)  │
  └────────────┘      └────────────┘      └──────────┘
         ▲                                      ▲
         └──────────────────────────────────────┘
              IPC (ZMQ Unix Sockets)

Scheduler: Receives requests from API, dispatches to Workers
Worker: Executes model inference on GPU
API: OpenAI-compatible HTTP endpoints
```

## 下一步

- 修改 Scheduler 参数以调整推理性能
- 实现 Worker 的多GPU 支持
- 添加 Scheduler 集群模式
