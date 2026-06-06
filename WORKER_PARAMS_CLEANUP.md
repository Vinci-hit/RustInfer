# Worker 启动参数清理

## 修改概述

已从 Worker 启动参数中移除所有冗余参数，现在所有配置统一从 Scheduler 的 `LoadModel` 消息中获取。

## 删除的参数

| 参数 | 原因 | 来源 |
|------|------|------|
| `--model-path` | 已在 LoadModel.model_path 中 | Scheduler |
| `--device` | 已在 LoadModel.device 中 | Scheduler |
| `--model-type` | 已在 LoadModel.model_type 中 | Scheduler |
| `--max-batch-tokens` | 已在 LoadModel.max_batch_tokens 中 | Scheduler |
| `--max-batch-seqs` | 已在 LoadModel.max_batch_seqs 中 | Scheduler |
| `--max-seq-len` | 使用 LoadModel.max_model_len | Scheduler |
| `--heartbeat-interval-ms` | 从 SchedulerHello 消息中获取 | Scheduler |

## 保留的参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--control-endpoint` | Scheduler 控制端点 | `ipc:///tmp/rustinfer-worker-control.ipc` |
| `--data-recv-endpoint` | 接收数据端点 | `ipc:///tmp/rustinfer-worker-in.ipc` |
| `--data-send-endpoint` | 发送数据端点 | `ipc:///tmp/rustinfer-worker-out.ipc` |
| `--worker-id` | Worker 标识符 | `worker-0` |
| `--block-size` | KV 块大小（必须为 1） | `1` |
| `--num-blocks-override` | KV 块数覆盖（诊断用） | `0` |
| `--profile-cuda-steps` | CUDA profiler 步数 | 无 |
| `--log-level` | 日志级别 | `info` |

## 代码修改

### 变量重命名
- `num_blocks` → `num_blocks_override` (更清晰的含义)
- `Bootstrap.heartbeat_interval_ms_arg` → 移除 (从 SchedulerHello 获取)

### 启动流程

```rust
// Before: 需要手动传递多个参数
rustinfer-worker \
    --model-path /path/to/model \
    --device cuda:0 \
    --model-type llama3 \
    --max-batch-tokens 4096 \
    --max-batch-seqs 32 \
    --max-seq-len 8192 \
    --heartbeat-interval-ms 1000

// After: 所有配置从 Scheduler 获取
rustinfer-worker \
    --worker-id worker-0 \
    --control-endpoint ipc:///tmp/rustinfer-worker-control.ipc \
    --data-recv-endpoint ipc:///tmp/rustinfer-worker-in.ipc \
    --data-send-endpoint ipc:///tmp/rustinfer-worker-out.ipc
```

## Heartbeat 优先级

```rust
// Heartbeat 优先级：
let hb_ms = bs.server_heartbeat_ms
    .unwrap_or(1000)  // 若 SchedulerHello 未设置，默认 1000ms
    .max(200);        // 最小间隔 200ms
```

## 启动脚本更新

启动脚本需要更新 Worker 启动命令（如适用）：

```bash
# 旧方式（已弃用）
rustinfer-worker --model-path /model --device cuda:0 --max-batch-seqs 32

# 新方式（现在使用）
rustinfer-worker \
    --control-endpoint ipc:///tmp/rustinfer-12345-worker-control.ipc \
    --data-recv-endpoint ipc:///tmp/rustinfer-12345-worker-in.ipc \
    --data-send-endpoint ipc:///tmp/rustinfer-12345-worker-out.ipc
```

## 优势

✅ **减少参数冗余** - 避免主进程和 Worker 之间的配置不一致  
✅ **简化启动** - Worker 只需知道通信端点  
✅ **配置集中** - 所有模型/性能参数在 Scheduler 中管理  
✅ **灵活扩展** - Scheduler 可以动态调整 Worker 配置而无需重启  
✅ **更少出错** - 参数不匹配的问题消除  

## 编译验证

✅ 代码编译无错误  
✅ 保留诊断参数 (`--num-blocks-override`, `--profile-cuda-steps`)
