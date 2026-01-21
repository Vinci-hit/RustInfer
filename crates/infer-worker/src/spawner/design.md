# Worker Auto-Spawner 设计文档

## 1. 概述

Auto-Spawner 是一个自动化的 Worker 进程管理系统，负责根据配置自动启动和监控多个 Worker 进程，每个进程绑定到不同的 GPU 设备。

## 2. 设计目标

### 2.1 核心功能

1. **设备自动发现**: 自动检测可用的 GPU/CPU 资源
2. **智能进程调度**: 根据资源分配策略启动 Worker 进程
3. **进程生命周期管理**: 监控 Worker 进程健康状态，自动重启失败进程
4. **多机扩展支持**: 为未来多机部署预留接口

### 2.2 非功能性需求

- **配置驱动**: 通过启动参数和环境变量控制行为
- **故障自愈**: Worker 进程异常退出时自动重启
- **资源隔离**: 每个 Worker 独占一个 GPU，避免资源竞争
- **日志聚合**: 统一收集所有 Worker 的日志输出

## 3. 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      RustInfer Spawner                            │
│                         (Main Process)                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Device Discovery Module                      │  │
│  │    - Query CUDA devices                                   │  │
│  │    - Get memory info                                      │  │
│  │    - Filter by constraints                                │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                         │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │              Worker Spawner Module                        │  │
│  │    - Generate per-worker configs                         │  │
│  │    - Launch subprocesses                                 │  │
│  │    - Monitor health status                               │  │
│  │    - Handle restart policy                               │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                         │
└─────────────────────────┼─────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │Worker 0 │    │Worker 1 │    │Worker N │
    │ GPU: 0  │    │ GPU: 1  │    │ GPU: N  │
    └─────────┘    └─────────┘    └─────────┘
```

### 3.2 模块结构

```
crates/infer-worker/src/
├── spawner/                     # Spawner 模块
│   ├── mod.rs                  # 模块导出
│   ├── design.md               # 本设计文档
│   ├── device.rs               # 设备发现和筛选
│   ├── spawner.rs              # Worker 进程生成器
│   ├── monitor.rs              # 进程监控
│   └── config.rs               # Spawner 配置
│
└── bin/
    ├── worker_main.rs          # 单 Worker 入口 (现有)
    └── spawner_main.rs         # Spawner 入口 (新增)
```

### 3.3 未来多机扩展

```
当前架构（单机多卡）:
┌──────────────────┐
│   Spawner Host   │
│                  │
│  ┌────────────┐  │  ┌─────────────┐
│  │   Worker   │  │  │             │
│  │  GPU 0     │──┼──►  Scheduler  │
│  └────────────┘  │  │             │
│  ┌────────────┐  │  └─────────────┘
│  │   Worker   │  │
│  │  GPU 1     │──┼──►
│  └────────────┘  │
└──────────────────┘

未来架构（多机多卡）:
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Host A       │      │ Host B       │      │ Host C       │
│              │      │              │      │              │
│ Spawner A    │      │ Spawner B    │      │ Spawner C    │
│ ┌──────────┐ │      │ ┌──────────┐ │      │ ┌──────────┐ │
│ │Worker A0 │ │      │ │Worker B0 │ │      │ │Worker C0 │ │
│ │  GPU:0   │─┼──────┼─►  GPU:0  │─┼──────┼─►  GPU:0  │─┼──►
│ └──────────┘ │      │ └──────────┘ │      │ └──────────┘ │
│ ┌──────────┐ │      │              │      │              │
│ │Worker A1 │ │      │              │      │              │
│ │  GPU:1   │─┼──────┼──────────────┼──────┼──────────────┼──►
│ └──────────┘ │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Scheduler   │
                    │   (Master)   │
                    └──────────────┘

关键设计点:
1. Spawner 支持通过 `--host-id` 和 `--node-list` 配置多机信息
2. 每个 Spawner 管理本地设备，但通过统一的 Scheduler 协调
3. 未来可扩展为支持配置中心 (etcd/consul) 进行节点发现
```

## 4. 详细设计

### 4.1 设备发现模块 (device.rs)

#### 功能
- 检测可用的 CUDA 设备
- 获取设备内存信息
- 根据约束条件筛选设备

#### API 设计

```rust
/// 设备发现器
pub struct DeviceDiscovery;

impl DeviceDiscovery {
    /// 查询所有可用的 CUDA 设备
    pub fn discover_cuda_devices() -> Result<Vec<CudaDeviceInfo>>;

    /// 查询所有可用的设备 (包括 CPU)
    pub fn discover_all() -> Result<Vec<AvailableDevice>>;

    /// 根据约束筛选设备
    pub fn filter_devices(
        devices: Vec<AvailableDevice>,
        constraints: &DeviceConstraints,
    ) -> Vec<AvailableDevice>;
}

/// CUDA 设备信息
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub total_memory: u64,
    pub free_memory: u64,
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
}

/// 可用设备（支持 CPU 和 CUDA）
#[derive(Debug, Clone)]
pub enum AvailableDevice {
    Cpu,
    Cuda(CudaDeviceInfo),
}

/// 设备筛选约束
#[derive(Debug, Clone)]
pub struct DeviceConstraints {
    /// 最小可用内存 (MB)
    pub min_memory_mb: Option<u64>,
    /// 排除的设备 ID
    pub exclude_device_ids: Vec<i32>,
    /// 仅使用指定的设备 ID
    pub include_device_ids: Option<Vec<i32>>,
    /// 最大设备数量
    pub max_devices: Option<usize>,
    /// 是否包含 CPU
    pub include_cpu: bool,
}

impl Default for DeviceConstraints {
    fn default() -> Self {
        Self {
            min_memory_mb: Some(8192),  // 默认至少 8GB
            exclude_device_ids: vec![],
            include_device_ids: None,
            max_devices: None,
            include_cpu: false,
        }
    }
}
```

### 4.2 Worker Spawner 模块 (spawner.rs)

#### 功能
- 为每个设备生成 Worker 配置
- 启动 Worker 子进程
- 管理 Worker 进程生命周期

#### API 设计

```rust
/// Worker 生成器
pub struct WorkerSpawner {
    config: SpawnerConfig,
    workers: HashMap<u32, WorkerProcess>,
}

/// 单个 Worker 进程
pub struct WorkerProcess {
    pub worker_id: String,
    pub rank: usize,
    pub device_id: i32,
    pub device_info: AvailableDevice,
    pub pid: u32,
    pub status: WorkerStatus,
    pub started_at: Instant,
    pub restart_count: u32,
}

/// Worker 状态
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed(String),
}

/// Spawner 配置
#[derive(Debug, Clone)]
pub struct SpawnerConfig {
    /// Scheduler 地址
    pub scheduler_url: String,

    /// 主机 ID（多机场景）
    pub host_id: Option<String>,

    /// 节点列表（多机场景）: HashMap<host_id, devices>
    pub node_list: Option<HashMap<String, Vec<AvailableDevice>>>,

    /// 设备约束
    pub device_constraints: DeviceConstraints,

    /// 重启策略
    pub restart_policy: RestartPolicy,

    /// 是否等待所有 Worker 启动完成
    pub wait_for_all: bool,

    /// 日志级别
    pub log_level: String,
}

/// 重启策略
#[derive(Debug, Clone)]
pub enum RestartPolicy {
    /// 从不重启
    Never,
    /// 总是重启
    Always,
    /// 失败 N 次后停止
    OnFailure { max_retries: u32 },
}

impl WorkerSpawner {
    /// 创建新的 Spawner
    pub fn new(config: SpawnerConfig) -> Self;

    /// 发现设备并生成 Worker 配置
    pub fn discover_and_plan(&mut self) -> Result<Vec<WorkerPlan>>;

    /// 生成 Worker 启动计划
    pub fn generate_plans(
        &self,
        devices: Vec<AvailableDevice>,
    ) -> Vec<WorkerPlan>;

    /// 启动所有 Worker 进程
    pub async fn spawn_all(&mut self) -> Result<()>;

    /// 启动单个 Worker 进程
    pub fn spawn_worker(&mut self, plan: &WorkerPlan) -> Result<u32>;

    /// 监控所有 Worker 进程
    pub async fn monitor_workers(&mut self) -> Result<()>;

    /// 处理 Worker 进程退出
    pub async fn handle_worker_exit(
        &mut self,
        worker_id: &str,
        exit_code: Option<i32>,
    ) -> Result<()>;

    /// 停止所有 Worker 进程
    pub async fn shutdown(&mut self) -> Result<()>;
}

/// Worker 启动计划
#[derive(Debug, Clone)]
pub struct WorkerPlan {
    pub worker_id: String,
    pub rank: usize,
    pub world_size: usize,
    pub device: AvailableDevice,
    pub device_id: i32,
    pub tp_rank: u32,
    pub tp_world_size: u32,
    pub scheduler_url: String,
    pub env_vars: HashMap<String, String>,
}

impl WorkerPlan {
    /// 构建命令行参数
    pub fn build_command_args(&self) -> Vec<String>;

    /// 构建环境变量
    pub fn build_env(&self) -> HashMap<String, String>;
}
```

### 4.3 进程监控模块 (monitor.rs)

#### 功能
- 轮询检查 Worker 进程状态
- 收集 Worker 输出日志
- 触发重启逻辑

#### API 设计

```rust
/// Worker 监控器
pub struct WorkerMonitor {
    spawner: Arc<Mutex<WorkerSpawner>>,
    check_interval: Duration,
}

impl WorkerMonitor {
    /// 创建新的监控器
    pub fn new(spawner: Arc<Mutex<WorkerSpawner>>, check_interval: Duration) -> Self;

    /// 启动监控循环
    pub async fn run(&self) -> Result<()>;

    /// 检查所有 Worker 状态
    async fn check_workers(&self) -> Result<()>;

    /// 处理异常退出的 Worker
    async fn handle_dead_worker(&self, worker: &WorkerProcess) -> Result<()>;
}

/// 监控事件
#[derive(Debug)]
pub enum MonitorEvent {
    WorkerStarted { worker_id: String, rank: usize },
    WorkerStopped { worker_id: String, rank: usize },
    WorkerFailed { worker_id: String, rank: usize, error: String },
    WorkerRestarted { worker_id: String, rank: usize, attempt: u32 },
}
```

### 4.4 配置模块 (config.rs)

#### 功能
- 解析命令行参数
- 生成 SpawnerConfig
- 验证配置有效性

#### CLI 设计

```bash
# 单机多卡自动启动 (检测所有 GPU)
cargo run --bin infer-spawner -- \
    --scheduler-url "tcp://localhost:5555" \
    --auto

# 单机多卡指定 GPU 范围
cargo run --bin infer-spawner -- \
    --scheduler-url "tcp://localhost:5555" \
    --gpu-ids 0,1,2,3

# 单机多卡带内存约束
cargo run --bin infer-spawner -- \
    --scheduler-url "tcp://localhost:5555" \
    --auto \
    --min-gpu-memory 16GB

# 未来: 多机部署 (预留接口)
cargo run --bin infer-spawner -- \
    --scheduler-url "tcp://scheduler:5555" \
    --host-id "node-0" \
    --node-list "node-0=0,1,2,3;node-1=0,1,2,3;node-2=0,1,2"

# 重启策略
cargo run --bin infer-spawner -- \
    --scheduler-url "tcp://localhost:5555" \
    --auto \
    --restart-policy on-failure \
    --max-retries 3
```

#### CLI 参数设计

```rust
#[derive(Parser, Debug)]
#[command(name = "infer-spawner")]
#[command(about = "RustInfer Worker Spawner - Auto-launch multiple workers")]
struct SpawnerArgs {
    /// Scheduler ZeroMQ endpoint
    #[arg(long, default_value = "ipc:///tmp/rustinfer-scheduler.ipc")]
    scheduler_url: String,

    /// 自动发现并使用所有可用设备
    #[arg(long)]
    auto: bool,

    /// 指定 GPU ID 列表 (逗号分隔，如: 0,1,2,3)
    #[arg(long, value_delimiter = ',')]
    gpu_ids: Option<Vec<i32>>,

    /// 排除的 GPU ID 列表 (逗号分隔)
    #[arg(long, value_delimiter = ',')]
    exclude_gpu_ids: Option<Vec<i32>>,

    /// 最小 GPU 内存 (如: 8GB, 16GB)
    #[arg(long)]
    min_gpu_memory: Option<String>,

    /// 最大 Worker 数量
    #[arg(long)]
    max_workers: Option<usize>,

    /// 是否包含 CPU 作为计算设备
    #[arg(long)]
    include_cpu: bool,

    /// 重启策略: never, always, on-failure
    #[arg(long, default_value = "never")]
    restart_policy: String,

    /// 最大重启次数 (仅对 on-failure 策略有效)
    #[arg(long, default_value_t = 3)]
    max_retries: u32,

    /// 等待所有 Worker 启动完成后再返回
    #[arg(long)]
    wait_for_all: bool,

    /// 主机 ID (多机场景)
    #[arg(long)]
    host_id: Option<String>,

    /// 节点列表 (格式: "host1=0,1,2;host2=0,1,2")
    #[arg(long)]
    node_list: Option<String>,

    /// 日志级别: error, warn, info, debug, trace
    #[arg(long, default_value = "info")]
    log_level: String,

    /// 启用详细输出
    #[arg(long, short)]
    verbose: bool,
}
```

## 5. 工作流程

### 5.1 启动流程

```
1. Spawner 主进程启动
   │
   ├─> 解析命令行参数
   │
   ├─> 设备发现
   │   ├─> 如果 --auto: 查询所有 CUDA 设备
   │   ├─> 如果 --gpu-ids: 只使用指定设备
   │   └─> 应用约束 (内存、排除列表等)
   │
   ├─> 生成 Worker 计划
   │   ├─> 为每个设备分配 rank
   │   ├─> 设置 tp_rank / tp_world_size
   │   └─> 生成环境变量
   │
   ├─> 启动 Worker 进程
   │   ├─> 对于每个计划:
   │   │   ├─> 构建命令行: worker_main --rank X --world-size Y --scheduler-url ...
   │   │   ├─> 设置 CUDA_VISIBLE_DEVICES
   │   │   └─> 启动子进程
   │   │
   │   └─> 如果 --wait-for-all: 等待所有 Worker 完成握手
   │
   └─> 启动监控循环
       ├─> 轮询 Worker 进程状态
       ├─> 收集日志输出
       └─> 处理异常退出 (根据重启策略)
```

### 5.2 多机扩展流程

```
当前设计 (预留接口):

1. 启动时指定 --host-id 和 --node-list
2. Spawner 根据本地 hostname 或配置确定自己是哪个节点
3. 每个节点只启动本地 GPU 对应的 Worker
4. 所有 Worker 连接到同一个 Scheduler
5. Scheduler 通过 worker_id 中的 host 前缀区分节点

未来增强:
- 使用 etcd/consul 进行节点自动发现
- 支持 GPU 资源自动均衡分配
- 支持节点间通信优化 (RDMA)
```

## 6. 环境变量设计

### 6.1 Spawner 控制变量

```bash
# 设备相关
CUDA_VISIBLE_DEVICES      # 控制 Worker 可见的 GPU
RUSTINFER_DEVICE_ID       # 当前 Worker 使用的设备 ID
RUSTINFER_DEVICE_COUNT    # 总设备数量

# 网络相关
RUSTINFER_SCHEDULER_URL   # Scheduler 地址
RUSTINFER_HOST_ID         # 当前主机 ID

# Worker 标识
RUSTINFER_WORKER_ID       # Worker ID (如: worker-0)
RUSTINFER_WORKER_RANK     # Worker rank
RUSTINFER_WORLD_SIZE      # 总 Worker 数量

# 调试相关
RUSTINFER_LOG_LEVEL       # 日志级别
RUSTINFER_VERBOSE         # 是否启用详细输出
```

## 7. 错误处理

### 7.1 错误分类

```rust
#[derive(Debug, thiserror::Error)]
pub enum SpawnerError {
    #[error("Device discovery failed: {0}")]
    DeviceDiscoveryFailed(String),

    #[error("No available devices found")]
    NoAvailableDevices,

    #[error("Worker spawn failed: {0}")]
    WorkerSpawnFailed(String),

    #[error("Worker process exited unexpectedly: {worker_id}, code: {code:?}")]
    WorkerUnexpectedExit {
        worker_id: String,
        code: Option<i32>,
    },

    #[error("Max restart attempts exceeded for worker: {0}")]
    MaxRestartExceeded(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
```

### 7.2 容错策略

1. **设备不可用**: 跳过该设备，记录警告日志
2. **Worker 启动失败**: 记录错误，根据重启策略决定是否重试
3. **Worker 运行时崩溃**: 收集崩溃日志，根据重启策略决定
4. **所有 Worker 失败**: Spawner 优雅退出，返回错误码

## 8. 实现计划

### 阶段 1: 基础功能 (当前)
- [x] 设计文档
- [ ] 设备发现模块
- [ ] Worker Spawner 基础框架
- [ ] 单机多卡自动启动
- [ ] 进程监控和日志

### 阶段 2: 增强功能
- [ ] 灵活的设备筛选
- [ ] 重启策略实现
- [ ] 健康检查
- [ ] 性能统计

### 阶段 3: 多机扩展 (未来)
- [ ] 主机 ID 和节点列表支持
- [ ] 节点间通信优化
- [ ] 分布式状态管理
- [ ] 配置中心集成

## 9. 示例用法

### 单机多卡自动启动

```bash
# 检测所有可用 GPU
./infer-spawner \
    --scheduler-url "tcp://localhost:5555" \
    --auto \
    --log-level info

# 输出示例:
# [Spawner] Discovering devices...
# [Spawner] Found 4 CUDA devices:
#   - GPU 0: NVIDIA A100 (80GB)
#   - GPU 1: NVIDIA A100 (80GB)
#   - GPU 2: NVIDIA A100 (80GB)
#   - GPU 3: NVIDIA A100 (80GB)
# [Spawner] Generating worker plans...
# [Spawner] Starting workers...
# [Worker-0] Initializing...
# [Worker-1] Initializing...
# [Worker-2] Initializing...
# [Worker-3] Initializing...
# [Spawner] All 4 workers started successfully
```

### 指定 GPU

```bash
# 只使用 GPU 0 和 2
./infer-spawner \
    --scheduler-url "tcp://localhost:5555" \
    --gpu-ids 0,2 \
    --auto
```

### 带重启策略

```bash
./infer-spawner \
    --scheduler-url "tcp://localhost:5555" \
    --auto \
    --restart-policy on-failure \
    --max-retries 3
```

## 10. 参考资料

- [vLLM Multi-GPU](https://docs.vllm.ai/en/latest/serving/mult gpu/)
- [torch.distributed](https://pytorch.org/docs/stable/distributed.html)
- [ZeroMQ: The Guide](https://zguide.zeromq.org/)
