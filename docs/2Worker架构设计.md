# 2 worker工作逻辑
## 2.1 spawner启动器

目前只支持单机单卡，执行以下命令，启动自动检测卡的模式，spawn是多worker启动器。

`cargo run --bin infer-spawner --release --features server -- --scheduler-url "tcp://localhost:5555"     --auto    `

接着SpawnerArgs 解析命令行参数并转为 SpawnerConfig，DeviceDiscovery::discover_all() 发现CUDA设备，其中首先会调用ffi::cudaGetDeviceCount获取cuda设备数量，接着遍历每一个设备，获取总内存大小和可用内存大小，暂未实现获取SM数量和计算能力。

为每个cuda设备分配Plan，设置它们的worker_id, rank, url等，然后根据Plan spawn_worker() 用 std::process::Command 启动 infer-worker 子进程，子进程通过 CUDA_VISIBLE_DEVICES 环境变量绑定到指定GPU，进入 monitor_workers() 循环监控进程状态。

## 2.2 worker进程启动
### 2.2.1 初始化与两次握手

spawner配置好启动参数后，会启动worker进程，worker进程从启动参数中初始化自己，创建 WorkerServer：WorkerServer::new(rank, world_size, scheduler_url)，初始化 ZeroMQ DEALER socket，尝试连接到 scheduler，如果ZMQ连接到Scheduler，会尝试握手，准备注册信息，告诉自己的设备类型是Cuda，设备id是多少，worker id是多少，rank、world size(并行的卡数), 然后向Scheduler发送注册信息，握手成功了就代表通信成功。接下来进入run_loop无限循环，根据接收到的指令反序列化后启动不同的功能。

```rust
/// Handle incoming command and return response
async fn handle_command(&mut self, cmd: WorkerCommand) -> WorkerResponse {
    match cmd {
        WorkerCommand::Register(registration) => self.handle_register(registration),
        WorkerCommand::LoadModel(params) => self.handle_load_model(params),
        WorkerCommand::Profile(params) => self.handle_profile(params),
        WorkerCommand::InitKVCache(params) => self.handle_init_kv_cache(params),
        WorkerCommand::Forward(params) => self.handle_forward(params),
        WorkerCommand::GetStatus => self.handle_get_status(),
        WorkerCommand::UnloadModel => self.handle_unload_model(),
        WorkerCommand::HealthCheck => WorkerResponse::Healthy,
    }
}
```

### 2.2.2 模型工厂设计模式

**ModelRegistry：**作为注册中心，承担调度中心和标准制定者的角色，目标是将模型架构名称（字符串）映射到模型构建逻辑（函数），实现解耦。本体是一个全局的线程安全的Map，在程序启动时，各个模型将自己的名字和构建函数挂号在这里。在使用的时候，接受config.json里面的架构名，找到对应的构建函数，传入Loader，返回构建好的模型对象。

**ModelTrait：**作为Registry的输出产品，定义了一个模型应该有的方法，如forward、reset_kv_cache等。

**ModelBuilderFn：**模型构建函数，作为Registry的value，也就是通过输入名称，会自动调用到不同的构建函数。输入是ModelLoader和运行时参数（包含设备信息，TP Rank等），输出是ModelTrait。

**Weight Mapping Adapter：**模型权重映射器，模型自己构建函数知道自己要载入哪些权重，每个模型管理自己的命名规则。

### 2.2.3 Profile探测最大可用显存

在载入模型后和KVCache初始化之前，会先进行一次profile操作，执行一次dummy_run，构造最大化的输入进行一次前向传播，目的是探测当模型在完全激活后最大占用的显存，使得能够最大化KVCache空间的申请而不会在未来爆显存。每张卡独立的worker进程在profile后会进行上报Scheduler，然后由Scheduler决定统一分配的KVCache的大小，为了避免存在某些卡由于连接了显示器而导致额外的显存占用，TP时导致显存不足，避免木桶效应。

```rust
WorkerResponse::ProfileCompleted(ProfileResult {
            peak_memory_forward: memory_stats.workspace_memory,
            memory_model: memory_stats.model_memory,
            total_memory: memory_stats.total,
            available_kv_cache_memory: available_kv_cache,
            avg_prefill_time_ms: 0.0, // TODO: Actually run profiling
            avg_decode_time_ms: 0.0,
            profiled_batch_size: params.batch_size,
            profiled_seq_len: params.seq_len,
        })
```
### 2.2.4 初始化KVCache（BlockPool）

探测完最大KVCache可用显存后，由Scheduler统一让Worker申请一块很大的显存，并在逻辑上划分为一个个Block，由Scheduler通过RadixTree统一管理。
请求参数：
```rust
pub struct InitKVCacheParams {
    /// 总的KVCache block数量
    pub num_blocks: usize,

    /// 每个block的大小 (token数量)
    pub block_size: usize,

    /// 层数 (number of layers)
    pub num_layers: u32,

    /// 注意力头数量
    pub num_heads: u32,

    /// 每个头的维度
    pub head_dim: u32,

    /// KVCache的数据类型: "bf16", "fp16", "fp32", "int8"
    pub dtype: String,
}
```

总大小 = block_num x block size x layer_nums x head num x head dim x 2 x sizeof(type)

### 2.2.5 forward with paged
#### 2.2.5.1 前向传播的参数定义

待定……