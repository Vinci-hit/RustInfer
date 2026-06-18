# RustInfer 异构后端与便利部署设计

## 背景

当前 `infer-worker` 默认启用 CUDA 构建，并在 `build.rs` 中把 `cuDNN frontend` 头文件作为硬依赖检测：

```text
cudnn_frontend.h
cudnn_frontend/graph_interface.h
```

这会导致一个实际部署问题：机器已经安装 CUDA 和 cuDNN runtime/dev 包，但缺少 NVIDIA 单独开源的 header-only `cudnn-frontend` 时，整个 worker 构建直接失败。对于后续支持 CPU、CUDA、ROCm、Metal、NPU 等异构后端的目标，这种“某个加速库缺失就阻断整个 worker”的模式不适合生产部署。

目标是把 RustInfer 设计成：

- 基础能力永远可构建、可启动。
- 加速路径按能力可插拔。
- 缺少可选依赖时自动降级。
- 用户通过镜像 tag 或 Cargo feature 选择部署档位，而不是手动拼环境。
- Scheduler 基于 worker 上报的能力做路由，而不是只判断“有没有 GPU”。

## 设计原则

1. **核心可运行**

   CPU 后端和基础推理流程应作为最小可运行闭环。没有 CUDA、cuDNN、ROCm 等依赖时，项目仍应能构建基础 worker。

2. **加速可插拔**

   cuDNN frontend、CUTLASS 特化 kernel、CUDA Graph、ROCm kernel、Metal kernel 都应是能力模块，而不是全局硬依赖。

3. **显式要求才 fail fast**

   如果用户显式启用 `cuda-cudnn-frontend`，但环境中没有对应头文件或库，应构建失败并给出明确错误。普通 `cuda` 构建不应因为缺少 cuDNN frontend 失败。

4. **运行时按能力路由**

   Scheduler 不应假设所有 CUDA worker 能力相同。Worker 启动后应上报 device、dtype、attention、quant、KV 容量、graph 支持等能力。

5. **部署产物分层**

   构建和镜像按能力档位发布。部署方选择 `cpu`、`cuda`、`cuda-cudnn` 等镜像，而不是手动安装复杂依赖。

## Cargo Feature 分层

建议将 `infer-worker` 的 feature 拆成以下层级：

```toml
[features]
default = ["cpu"]

cpu = []

cuda = [
    "dep:cc",
    "dep:bindgen",
    "dep:walkdir",
]

cuda-cutlass = ["cuda"]
cuda-graph = ["cuda"]
cuda-cudnn = ["cuda"]
cuda-cudnn-frontend = ["cuda-cudnn"]

rocm = []
metal = []
npu = []
```

推荐语义：

- `cpu`：最小后端，默认启用，保证项目可构建。
- `cuda`：CUDA runtime、cuBLAS、手写基础 CUDA kernel。
- `cuda-cutlass`：依赖 vendored CUTLASS/CUTE 的优化 kernel。
- `cuda-graph`：CUDA Graph 捕获和回放。
- `cuda-cudnn`：cuDNN C API，例如 conv2d 等基础 cuDNN 调用。
- `cuda-cudnn-frontend`：cuDNN frontend C++ header-only graph API，例如 paged decode SDPA。
- `rocm`、`metal`、`npu`：后续异构后端扩展点。

关键点：`cuda` 不应隐式等价于 `cuda-cudnn-frontend`。

## cuDNN Frontend 依赖策略

`cudnn-frontend` 是 header-only C++ API，不一定随系统 cuDNN 包安装。为了降低部署复杂度，建议采用“vendored 优先，环境覆盖”的方式。

推荐目录：

```text
crates/infer-worker/src/infrastructure/cuda/kernels/third_party/
  cutlass/
  cute/
  cudnn-frontend/
    include/
      cudnn_frontend.h
      cudnn_frontend/
        graph_interface.h
```

`build.rs` 搜索顺序：

1. `CUDNN_FRONTEND_INCLUDE_DIR`
2. `CUDNN_FRONTEND_ROOT/include`
3. `CUDNN_FRONTEND_PATH/include`
4. repo 内 `third_party/cudnn-frontend/include`
5. conda/pip site-packages include
6. `/usr/local/cuda/include`
7. `/usr/local/include`
8. `/usr/include`
9. `/usr/include/x86_64-linux-gnu`

构建行为：

- 启用 `cuda-cudnn-frontend`：找不到 frontend 头文件时直接失败。
- 只启用 `cuda`：找不到 frontend 头文件时打印 warning，跳过 `cudnn_paged_attention.cu`，继续构建其他 CUDA kernel。

这样默认 CUDA worker 可以部署，cuDNN frontend 只是额外优化路径。

## CUDA Kernel 编译拆分

当前 CUDA kernel 统一由 `build.rs` 搜索 `.cu` 并全部编译。建议改为按能力模块筛选：

```text
always cuda:
  add.cu
  emb.cu
  ewise_mul.cu
  flash_attn_batched_decode.cu
  ...

cuda-cudnn-frontend:
  flash_attn_gqa/cudnn_paged_attention.cu

cuda-cutlass:
  cutlass/cute dependent kernels
```

`cudnn_paged_attention.cu` 内的 `#include <cudnn_frontend.h>` 只应在 `cuda-cudnn-frontend` feature 开启时进入编译单元。

建议在 Rust 侧也用 capability guard，而不是到处散落条件编译：

```rust
#[cfg(feature = "cuda-cudnn-frontend")]
const HAS_CUDNN_FRONTEND: bool = true;

#[cfg(not(feature = "cuda-cudnn-frontend"))]
const HAS_CUDNN_FRONTEND: bool = false;
```

运行时 attention 选择策略：

```text
paged_attention
  -> cuDNN frontend paged decode, if compiled and runtime supported
  -> custom flash attention kernel, if supported
  -> naive CUDA fallback, if available
  -> CPU fallback, if allowed by request policy
```

## Worker 能力模型

Worker 启动时应探测并上报能力，而不是只上报 rank 或 KV 容量。

示例能力描述：

```json
{
  "worker_id": "worker-0",
  "backend": "cuda",
  "device": {
    "name": "NVIDIA H20",
    "compute_capability": "9.0",
    "memory_bytes": 100000000000
  },
  "features": {
    "cuda_graph": true,
    "cudnn": true,
    "cudnn_frontend": true,
    "cutlass": true
  },
  "dtypes": ["fp16", "bf16"],
  "attention": ["custom_flash", "cudnn_paged_decode"],
  "quantization": ["int4_awq", "bf16"],
  "limits": {
    "max_batch": 8,
    "max_total_kv_tokens": 262144,
    "block_size": 1
  }
}
```

Scheduler 根据能力做调度：

- BF16 请求只路由到支持 BF16 的 worker。
- 需要高吞吐 paged decode 时优先选择有 `cudnn_paged_decode` 的 worker。
- 没有 cuDNN frontend 的 CUDA worker 仍可接请求，但走 fallback attention。
- CPU worker 作为低优先级兜底，或用于小模型、测试、健康检查。
- 后续 ROCm、Metal、NPU worker 用同一能力协议接入。

## 后端接口抽象

建议把模型上层逻辑与具体设备实现隔离，定义稳定的后端接口：

```rust
pub trait InferBackend {
    fn name(&self) -> BackendName;
    fn capabilities(&self) -> BackendCapabilities;

    fn allocate(&self, shape: Shape, dtype: DType) -> Result<Tensor>;
    fn upload(&self, host: &[u8], shape: Shape, dtype: DType) -> Result<Tensor>;
    fn download(&self, tensor: &Tensor) -> Result<Vec<u8>>;

    fn matmul(&self, args: MatmulArgs) -> Result<Tensor>;
    fn rms_norm(&self, args: RmsNormArgs) -> Result<Tensor>;
    fn paged_attention(&self, args: PagedAttentionArgs) -> Result<Tensor>;
}
```

CUDA 后端内部再做策略选择：

```text
CudaBackend::paged_attention
  -> CudnnFrontendPagedDecode
  -> CustomFlashPagedDecode
  -> NaiveCudaAttention
```

这样新增 ROCm 或 Metal 后端时，只需要实现 `InferBackend` 和能力描述，不需要改动 scheduler 或模型编排的大部分逻辑。

## 构建产物分层

推荐发布以下二进制或镜像档位：

```text
rustinfer-worker-cpu
rustinfer-worker-cuda
rustinfer-worker-cuda-cudnn
rustinfer-worker-cuda-full
```

对应 Cargo 构建：

```bash
cargo build -p infer-worker --no-default-features --features cpu --release
cargo build -p infer-worker --no-default-features --features cuda --release
cargo build -p infer-worker --no-default-features --features cuda,cuda-cudnn,cuda-cudnn-frontend --release
cargo build -p infer-worker --no-default-features --features cuda,cuda-cutlass,cuda-graph,cuda-cudnn,cuda-cudnn-frontend --release
```

对应镜像：

```text
rustinfer:cpu
rustinfer:cuda-13-runtime
rustinfer:cuda-13-cudnn
rustinfer:cuda-13-full
```

镜像职责：

- `cpu`：无 GPU 依赖，最小镜像，可用于开发、CI、fallback。
- `cuda-13-runtime`：CUDA runtime、cuBLAS、基础 CUDA worker。
- `cuda-13-cudnn`：额外包含 cuDNN runtime，启用 cuDNN C API 和可选 frontend。
- `cuda-13-full`：包含 profiling/debug 工具、Nsight 支持、完整优化路径，适合性能调优。

## 配置建议

Worker 配置中显式声明期望后端和降级策略：

```toml
[worker]
backend = "auto" # auto | cpu | cuda | rocm | metal
allow_fallback = true

[worker.cuda]
device = 0
prefer_cudnn_frontend = true
prefer_cuda_graph = true

[worker.routing]
priority = 100
labels = ["h20", "bf16", "paged-decode"]
```

语义：

- `backend = "auto"`：启动时按可用硬件和编译能力选择最佳后端。
- `allow_fallback = true`：缺少某个优化 path 时允许降级。
- `prefer_cudnn_frontend = true`：有能力时优先使用，不存在时不阻断启动。
- 如果用户设置 `backend = "cuda"` 但二进制未编译 CUDA，则启动失败并给出明确错误。

## 错误与日志规范

构建期：

- 必需依赖缺失：`error`，构建失败。
- 可选优化依赖缺失：`warning`，继续构建并标记 capability 为 false。

启动期：

- 编译支持但运行时库缺失：按配置决定失败或降级。
- 设备不支持某 dtype 或 kernel：记录 warning，上报 capabilities 时剔除该能力。

请求期：

- 请求明确要求某能力但没有匹配 worker：Scheduler 返回可解释错误。
- 请求允许 fallback：Scheduler 选择次优 worker，并可在 tracing 中记录降级原因。

## 落地步骤

### 第一阶段：解除 cuDNN frontend 硬依赖

- 新增 `cuda-cudnn-frontend` feature。
- `build.rs` 只在该 feature 开启时强制检测 frontend 头文件。
- 普通 `cuda` 构建跳过 `cudnn_paged_attention.cu`。
- Rust 侧 attention 调用在无 frontend 时走已有 custom flash attention fallback。

### 第二阶段：vendor header-only 依赖

- 将 NVIDIA `cudnn-frontend/include` 放入 `third_party/cudnn-frontend/include`。
- 保留 `CUDNN_FRONTEND_INCLUDE_DIR` 覆盖能力，方便用户使用系统或自定义版本。
- 在 `docs/` 和 README 中说明 vendored 版本和升级方法。

### 第三阶段：能力上报

- 定义 `BackendCapabilities`。
- Worker 启动时探测 device、dtype、attention、quant、KV limits。
- Scheduler 注册 worker 时保存能力信息。

### 第四阶段：能力路由

- Scheduler 按 request requirements 过滤 worker。
- 同等能力下再按负载、KV 余量、优先级选择 worker。
- 增加无匹配 worker 的错误返回。

### 第五阶段：多后端扩展

- 固化 `InferBackend` trait。
- 将 CUDA 专属实现收敛到 `infrastructure/cuda`。
- 增加 `infrastructure/cpu`、`infrastructure/rocm`、`infrastructure/metal` 等实现目录。
- 模型编排层只依赖 backend trait 和 capability，不直接依赖 CUDA FFI。

## 验收标准

1. 未安装 CUDA 的机器可以构建 CPU worker。
2. 安装 CUDA 但未安装 `cudnn-frontend` 的机器可以构建 CUDA worker。
3. 显式启用 `cuda-cudnn-frontend` 且缺少 frontend 头文件时构建失败，错误信息包含安装或环境变量指引。
4. CUDA worker 启动日志打印实际启用的 capabilities。
5. Scheduler 能区分 `custom_flash` 和 `cudnn_paged_decode` worker。
6. 请求在缺少最优能力时可以按配置降级，而不是直接崩溃。
7. 新增后端不需要改动主调度协议的大部分结构。

## 结论

RustInfer 的部署便利性不应依赖用户手工补齐所有加速库。更合理的方向是把 worker 拆成“基础后端 + 可选加速能力”，让构建期和运行期都支持能力降级。`cuDNN frontend` 当前应从 CUDA 硬依赖中拆出，作为 `cuda-cudnn-frontend` 可选 feature，并通过 vendored headers 降低环境安装成本。Scheduler 则从 worker 上报的能力出发做路由，为后续 ROCm、Metal、NPU 等异构后端提供统一接入方式。
