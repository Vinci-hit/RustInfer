# RustInfer Worker 重构进展文档

> 最后更新：2026-05-26  
> Commit: `271e126` on branch `feat/worker-batch-forward`

---

## 一、架构总览

```
crates/infer-worker/src/
├── domain/        ← 领域层（纯 trait + 类型，零依赖）
├── infra/         ← 基础设施层（CPU / CUDA 实现）
├── models/        ← 模型层（LLM + Diffusion）
├── app/           ← 应用层（ModelRunner + CudaGraphRunner）
└── process/       ← 进程层（ZMQ 通信 + 二级调度 + 两线程运行时）
```

### 核心设计原则

| 原则 | 实现 |
|------|------|
| DDD | domain 定义 port (trait), infra 提供 adapter (impl) |
| TypeState | `Tensor<T: Dtype, D: Device>` — dtype + device 编译期确定 |
| NewType | `Shape` / `Strides` 独立类型，防止参数交换 |
| 无 Unsupported | OpBackend 每个方法都有实际实现，不允许 `Unsupported` 错误 |
| 模型不持有资源 | Model 纯计算，KvCache/CudaGraph 归 Worker 管理 |

---

## 二、文件清单（61 .rs）

### domain/ (6 files)
| 文件 | 内容 |
|------|------|
| `types.rs` | Dtype trait, DataType enum, Shape, Strides, Float marker |
| `tensor.rs` | `Tensor<T, D>` 核心结构体 |
| `ports.rs` | Device trait, OpBackend trait (20 方法), OpError |
| `model.rs` | LlmModel trait, ForwardContext |
| `runtime.rs` | KvCache 结构 |
| `ops.rs` | 算子组合辅助 |

### infra/cpu/ (1 file)
| 文件 | 内容 |
|------|------|
| `mod.rs` | Cpu device + OpBackend 全部 20 方法纯 Rust 实现 + 16 单元测试 |

### infra/cuda/ (20 files)
| 文件 | 内容 |
|------|------|
| `mod.rs` | Cuda device + OpBackend dispatch + `Tensor<T,Cuda>::zeros_cuda` |
| `config.rs` | CudaConfig (stream/cublas/cublasLt/cudnn/workspace/graph) |
| `error.rs` | CudaError + `cuda_check!` 宏 |
| `ffi.rs` | bindgen 生成的 FFI |
| `device_utils.rs` | set_device / current_device |
| `thread_stream.rs` | 线程局部 stream |
| `kernels/` | 19 个 .rs wrapper: rmsnorm, add, matmul, softmax, activation, scalar, embedding, rope, attention, fused_add_rmsnorm, sampler, split_cols, scatter_kv, groupnorm, upsample, broadcast_mul, ewise_mul, layernorm, conv2d |

### models/ (12 files)
| 文件 | 内容 |
|------|------|
| `layers.rs` | Linear, RMSNorm, Embedding, Conv2D, GroupNorm, LayerNorm |
| `llama3.rs` | Llama3Model forward (fused_add_rmsnorm) |
| `qwen3.rs` | Qwen3Model forward (+ QK-norm) |
| `loader.rs` | safetensors→模型 + RoPE cache 计算 |
| `diffusion/dit_block.rs` | DiTBlock<T,D> forward |
| `diffusion/transformer.rs` | ZImageTransformer<T,D> 完整 forward |
| `diffusion/vae_decoder.rs` | VaeDecoder<T,D> (conv+groupnorm+upsample) |
| `diffusion/pipeline.rs` | ZImagePipeline encode→denoise→decode |
| `diffusion/scheduler.rs` | FlowMatchEulerScheduler (纯数学) |
| `diffusion/state.rs` | DitState<T,D>, PipelineState<T,D> |
| `diffusion/rope_3d.rs` | 3D RoPE embedder + interleaved RoPE |
| `diffusion/patchify.rs` | patchify/unpatchify |
| `diffusion/timestep_embedder.rs` | 正弦嵌入 + MLP |

### app/ (3 files)
| 文件 | 内容 |
|------|------|
| `model_runner.rs` | ModelRunner::step/generate + D::argmax |
| `cuda_graph_runner.rs` | warmup→capture→binary search→replay |

### process/ (6 files)
| 文件 | 内容 |
|------|------|
| `sync_flags.rs` | AtomicBool lock-free Runner↔SubScheduler 握手 |
| `sub_scheduler.rs` | decode 自循环 + prefill interleave + batch 组装 |
| `control_pump.rs` | ZMQ DEALER 控制面 (real implementation) |
| `data_pump.rs` | ZMQ PULL/PUSH 数据面 (real implementation) |
| `serve_loop.rs` | 两线程运行时 (Runner + SubScheduler) |
| `main.rs` | clap CLI + bootstrap + serve |

---

## 三、OpBackend trait 方法一览（20 个）

| 类别 | 方法 |
|------|------|
| 分配 | `alloc_tensor` |
| 加法 | `add`, `add_inplace` |
| 归一化 | `rmsnorm`, `rmsnorm_inplace`, `fused_add_rmsnorm`, `layernorm`, `groupnorm`, `groupnorm_silu` |
| 线性 | `matmul`, `matmul_quant` |
| 激活 | `silu_inplace`, `swiglu_inplace` |
| Softmax | `softmax` |
| 标量 | `scalar_mul_inplace` |
| Embedding | `embedding` |
| RoPE | `rope_inplace` |
| Attention | `attention`, `sdpa` |
| KV Cache | `split_qkv`, `scatter_kv` |
| 采样 | `argmax` |
| 空间 | `conv2d`, `upsample_nearest_2x` |
| 逐元素 | `broadcast_mul_inplace`, `ewise_mul` |

---

## 四、测试现状

```
27 tests pass, 0 warnings
├── infra::cpu::tests (16): tensor ops, add, matmul, rmsnorm, silu, softmax,
│   embedding, conv2d, groupnorm, layernorm, upsample, ewise_mul
├── app::cuda_graph_runner::tests (2): binary_search, needs_padding
├── app::model_runner::tests (1): e2e_cpu_forward_and_argmax (Llama3 1层 prefill+decode)
├── models::diffusion::scheduler::tests (2): monotonic_sigmas, dt_negative
├── models::diffusion::rope_3d::tests (3): precompute_shapes, embed_gathers, interleaved_math
├── models::diffusion::patchify::tests (1): roundtrip
├── process::sync_flags::tests (2): handshake, shutdown_unblocks
└── process::sub_scheduler::tests (3): build_batch_decode/mixed, process_output
```

---

## 五、依赖

```toml
[dependencies]
infer-protocol, half, thiserror, num-traits, dashmap, once_cell,
rayon, safetensors, serde, serde_json, tracing, rmp-serde, clap, zmq

[build-dependencies] (cuda feature)
cc, bindgen, walkdir
```

系统依赖：`zeromq-devel` (`dnf install -y zeromq-devel`)

---

## 六、构建与运行

```bash
# 编译检查（跳过 CUDA kernel 编译，不需要 GPU）
SKIP_BUILD_KERNELS=1 cargo check -p infer-worker

# 运行测试
SKIP_BUILD_KERNELS=1 cargo test -p infer-worker

# 完整编译（需要 CUDA toolkit + GPU）
cargo build -p infer-worker

# 运行 worker binary
cargo run -p infer-worker --bin rustinfer-worker -- \
  --model-path /path/to/qwen3-0.6B \
  --device cuda:0 \
  --control-endpoint tcp://scheduler:5500 \
  --data-recv-endpoint tcp://scheduler:5501 \
  --data-send-endpoint tcp://scheduler:5502
```

---

## 七、剩余工作（GPU 可用后）

| 项目 | 说明 | 复杂度 |
|------|------|--------|
| GPU 端到端验证 | `cargo test --features cuda` 跑通 Qwen3 forward | 低 |
| serve_loop output 读取 | 把 `vec![0i32; num_items]` 替换为真实 D2H 读 output tensor | 低 |
| main.rs 模型加载 | `Cuda::new() + WeightLoader::load_qwen3()` + `ModelRunner::new()` | 中 |
| BatchWorkspace | device-resident 输入/输出 tensor 池（地址稳定，CUDA Graph friendly） | 中 |
| CUDA Graph capture | decode-only path graph capture + replay | 中 |
| Paged KV Cache | block_table based KV management (对接 scheduler InitPagedKv) | 高 |
| Diffusion 权重加载 | `load_zimage_transformer` / `load_vae_decoder` | 中 |
| 投机解码 | DecodePolicy trait + draft_runner（接口已预留） | 高 |

---

## 八、通信架构

```
Scheduler                              Worker
   │                                     │
   │  [ZMQ ROUTER/DEALER]               │
   │◄─── WorkerHello ───────────────────│  control plane
   │──── SchedulerHello + LoadModel ───►│
   │◄─── MemoryProfile ────────────────│
   │──── InitPagedKv ─────────────────►│
   │◄─── WorkerReady ──────────────────│
   │                                     │
   │  [ZMQ PUSH→PULL]                   │
   │──── PrefillBatchCmd ─────────────►│  data plane (scheduler→worker)
   │                                     │  Worker 内部 SubScheduler
   │                                     │  自回归 decode 循环
   │  [ZMQ PUSH→PULL]                   │  （不需要 scheduler 每步推送 decode）
   │◄──── StepOutput(tokens) ──────────│  data plane (worker→scheduler)
   │                                     │
   │──── Cancel / Drain ──────────────►│  control plane (runtime)
   │◄──── Heartbeat ───────────────────│
```

Worker 内部两线程：
- **Runner 线程**：spin-wait → forward → signal (CUDA Graph captured decode path)
- **SubScheduler 线程**：ZMQ IO + 状态管理 + batch 组装 + output 处理
