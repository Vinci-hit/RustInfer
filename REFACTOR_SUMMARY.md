# RustInfer All-in-One 拆分总结

## 完成的工作

成功将 RustInfer 的 all-in-one 架构拆分为三个独立进程，可手动启动。

## 新的架构

### 三个独立进程

| 进程 | 二进制文件 | 职责 | 启动方式 |
|------|-----------|------|---------|
| **Scheduler & Worker** | `rustinfer-server` | 推理引擎核心，模型推理 | 主进程 main.rs |
| **API Server** | `rustinfer-api` | HTTP API 服务，请求处理 | 独立二进制 bin/api.rs |
| **（可选）** | - | 通过 ZMQ IPC 通信 | - |

### 通信方式

三个进程通过 **ZMQ Unix IPC Sockets** 通信：

```
/tmp/rustinfer-<PID>-frontend.ipc          API ↔ Scheduler (请求/响应)
/tmp/rustinfer-<PID>-worker-in.ipc         Scheduler → Worker (输入队列)
/tmp/rustinfer-<PID>-worker-out.ipc        Worker → Scheduler (输出队列)
/tmp/rustinfer-<PID>-worker-control.ipc    Scheduler ↔ Worker (控制信号)
```

## 代码修改

### 1. Cargo.toml 更新
- 添加了第二个二进制目标 `rustinfer-api`
- 指向新文件 `src/bin/api.rs`

### 2. 修改 main.rs
- 移除了 API Server 的所有代码
- 现在只启动 Scheduler 和 Worker
- 在启动完成后打印连接信息和启动 API 的命令

### 3. 创建新文件 src/bin/api.rs
- 独立的 API Server 实现
- 从 Scheduler 读取连接端点
- 初始化 Tokenizer 和 ZMQ 客户端
- 启动 HTTP 服务

## 启动脚本

### 1. 自动启动脚本 (scripts/start_distributed.sh)
一键启动所有三个进程，带完整的进程管理和清理：

```bash
./scripts/start_distributed.sh --model /path/to/model --device cuda:0 --port 8000
```

**特点：**
- ✅ 自动检查二进制文件，如果不存在则编译
- ✅ 独立的日志文件管理
- ✅ 等待 Scheduler 初始化
- ✅ Ctrl+C 优雅关闭所有进程
- ✅ 自动清理 IPC 文件
- ✅ 支持 `--skip-api` 标志单独启动 Scheduler/Worker

### 2. 手动启动指南 (scripts/manual_startup.sh)
用于调试，显示如何在三个不同终端中手动启动每个进程：

```bash
bash ./scripts/manual_startup.sh /path/to/model cuda:0 info 8000
```

## 文件结构

```
RustInfer/
├── crates/infer-server/
│   ├── src/
│   │   ├── main.rs                    (修改: Scheduler & Worker 启动)
│   │   ├── bin/
│   │   │   └── api.rs                 (新增: API Server)
│   │   ├── lib.rs
│   │   ├── config.rs
│   │   ├── router.rs
│   │   ├── state.rs
│   │   └── ...
│   └── Cargo.toml                      (修改: 添加 api bin 目标)
├── scripts/
│   ├── start_distributed.sh            (新增: 自动启动脚本)
│   └── manual_startup.sh               (新增: 手动启动指南)
├── Makefile.distributed                (新增: 便利的 Make 命令)
└── DISTRIBUTED_STARTUP.md              (新增: 详细使用文档)
```

## 编译

### Debug 版本
```bash
cargo build --bin rustinfer-server --bin rustinfer-api
```

### Release 版本（推荐）
```bash
cargo build --release --bin rustinfer-server --bin rustinfer-api
```

### 使用 Makefile
```bash
make -f Makefile.distributed build-release
make -f Makefile.distributed install-release
```

## 使用示例

### 快速启动
```bash
./scripts/start_distributed.sh \
    --model ~/models/mistral-7b \
    --device cuda:0 \
    --port 8000
```

### 手动启动（三个终端）

**终端 1:**
```bash
rustinfer-server --model ~/models/mistral-7b --device cuda:0
```

**终端 2:**
```bash
rustinfer-api \
    --model ~/models/mistral-7b \
    --frontend-endpoint ipc:///tmp/rustinfer-12345-frontend.ipc \
    --port 8000
```

**终端 3:**
```bash
curl http://localhost:8000/v1/models
```

### 多实例运行（不同 GPU）
```bash
# 实例 1
./scripts/start_distributed.sh --model model1 --device cuda:0 --port 8000 &

# 实例 2
./scripts/start_distributed.sh --model model2 --device cuda:1 --port 8001 &
```

## 关键特性

✅ **完全分离** - 三个进程可独立运行、调试、重启  
✅ **自动化启动** - 一条命令启动所有服务  
✅ **优雅关闭** - SIGTERM 支持，资源正确清理  
✅ **日志隔离** - 每个进程独立的日志文件  
✅ **灵活配置** - 支持所有现有的 CLI 参数  
✅ **多实例** - 同一机器上可运行多个实例  
✅ **调试友好** - 手动启动脚本便于问题诊断  

## 向后兼容性

旧的 all-in-one 模式已完全移除。用户需要使用新的启动脚本或手动启动三个进程。

## 下一步（可选）

1. 实现 Scheduler 的 HA（高可用）模式
2. 添加 Worker 的多 GPU 支持
3. 实现 Scheduler 集群模式
4. 添加负载均衡器支持
5. 创建 Docker Compose 配置
6. 实现 systemd 服务单元文件
