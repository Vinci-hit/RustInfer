# Model Registry 设计文档

## 1. 设计概述

### 1.1 核心设计理念
基于架构名称的分发 (Architecture-Based Dispatch)，将"模型架构名称"（字符串）映射到"模型构建逻辑"（函数），实现完全解耦。

### 1.2 设计目标
- **开闭原则**：新增模型无需修改核心代码
- **解耦加载与构建**：Loader 只管 IO，Builder 只管组装
- **支持多态配置**：每个模型可自定义配置转换
- **线程安全**：支持并发访问
- **高性能**：低开销的模型分发机制

### 1.3 适用场景
- LLM 推理框架中的模型管理
- 支持多种模型架构的统一接口
- 动态加载和卸载模型
- 多模型并发推理

## 2. 系统架构

### 2.1 组件图
```
┌─────────────────────────────────────────────────────────────────┐
│                         ModelRegistry                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    Architectures Map                    │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │  "LlamaForCausalLM" → LlamaBuilderFn           │   │     │
│  │  │  "Qwen2ForCausalLM" → Qwen2BuilderFn           │   │     │
│  │  │  "MistralForCausalLM" → MistralBuilderFn       │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    Models Map                        │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │  ModelId("llama3-8b") → Llama3 Model           │   │     │
│  │  │  ModelId("qwen2-1.5b") → Qwen2 Model           │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
          │                          │
          ▼                          ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   ModelBuilderFn        │  │      Model Trait        │
└─────────────────────────┘  └─────────────────────────┘
          │                          │
          ▼                          ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   Concrete Builder      │  │   Concrete Model        │
│  (LlamaBuilder, etc.)   │  │  (Llama3, Qwen2, etc.)  │
└─────────────────────────┘  └─────────────────────────┘
```

### 2.2 模块依赖

```
┌───────────────────┐     ┌───────────────────┐
│   model/mod.rs    │────▶│   model/config.rs │
└──────────┬────────┘     └───────────────────┘
           │
           ▼
┌───────────────────┐     ┌───────────────────┐
│ model/loader/     │────▶│ model/architectures/ │
│   registry.rs     │     │   llama3.rs       │
│   model_loader.rs │     │   qwen2.rs        │
│   safetensor_loader.rs │  │   ...             │
└───────────────────┘     └───────────────────┘
```

## 3. 核心组件设计

### 3.1 统一模型接口 (ModelTrait)

**设计目标**：定义所有模型必须具备的行为，是 Registry 的"输出产品"。

**核心方法**：
```rust
pub trait Model: Send + Sync {
    // 获取模型配置
    fn config(&self) -> &RuntimeModelConfig;
    
    // 执行推理
    fn forward(&mut self, input: &Tensor, pos: &Tensor) -> Result<Tensor>;
    
    // 带缓存管理的推理（用于连续批处理）
    fn forward_with_cache(
        &mut self,
        input: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor>;
    
    // 带分页注意力的推理
    fn forward_paged(
        &mut self,
        input_tokens: &Tensor,
        positions: &Tensor,
        block_tables: &[Vec<u32>],
        slot_mapping: &Tensor,
        context_lens: &[usize],
        is_prefill: bool,
    ) -> Result<Tensor>;
    
    // 重置 KV 缓存
    fn reset_kv_cache(&mut self) -> Result<()>;
    
    // 切片 KV 缓存
    fn slice_kv_cache(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        len: usize,
    ) -> Result<(Tensor, Tensor)>;
    
    // 获取设备类型
    fn device_type(&self) -> DeviceType;
}
```

### 3.2 模型构建器接口 (ModelBuilderFn)

**设计目标**：定义模型构建函数的签名，是 Registry 存储的"Value"。

**函数签名**：
```rust
pub type ModelBuilderFn = fn(
    model_dir: &Path,
    device_type: DeviceType,
    is_quant_model: bool,
) -> Result<Box<dyn Model>>;
```

**设计特点**：
- 函数指针类型，轻量级且高效
- 支持不同模型的自定义构建逻辑
- 隐藏具体模型实现细节

### 3.3 注册表单例 (ModelRegistry)

**设计目标**：全局的、线程安全的 Map，是 Registry 的"本体"。

**核心属性**：
```rust
pub struct ModelRegistry {
    // 架构到构建器的映射
    architectures: Arc<RwLock<HashMap<String, ModelBuilderFn>>>,
    // 已加载模型的映射
    models: Arc<RwLock<HashMap<ModelId, ModelInstance>>>,
    // 默认设备类型
    device_type: DeviceType,
}
```

**核心方法**：

| 方法名 | 功能 | 参数 | 返回值 |
|--------|------|------|--------|
| `register_architecture` | 注册模型架构 | `architecture: &str`, `builder: ModelBuilderFn` | `Result<()>` |
| `get_builder` | 获取模型构建器 | `architecture: &str` | `Result<Option<ModelBuilderFn>>` |
| `detect_architecture` | 从配置文件检测架构 | `model_dir: &Path` | `Result<String>` |
| `load_model_by_architecture` | 按架构加载模型 | `model_id: ModelId`, `model_dir: &Path`, `architecture: &str`, `is_quant_model: bool` | `Result<()>` |
| `load_model_auto` | 自动检测架构并加载模型 | `model_id: ModelId`, `model_dir: &Path`, `is_quant_model: bool` | `Result<String>` |
| `get_model` | 获取模型只读引用 | `model_id: &ModelId` | `Result<Option<impl Deref<Target = dyn Model> + '_>>` |
| `get_model_mut` | 获取模型可变引用 | `model_id: &ModelId` | `Result<Option<impl DerefMut<Target = dyn Model> + '_>>` |
| `unload_model` | 卸载模型 | `model_id: &ModelId` | `Result<()>` |

### 3.4 权重映射适配器 (Weight Mapping Adapter)

**设计目标**：处理不同模型权重名称的差异，是 sglang/vllm 的精髓。

**设计思路**：
- Registry 不负责改名，而是具体模型实现自己负责
- 每个模型实现都有自己的 `WeightMapping` 实现
- 在模型构建过程中，将通用权重名称映射到模型内部的权重名称

**示例实现**：
```rust
pub trait WeightMapping {
    fn map(&self, weight_name: &str) -> Option<String>;
}

impl WeightMapping for LlamaWeightMapping {
    fn map(&self, weight_name: &str) -> Option<String> {
        // 将 HuggingFace 权重名称映射到 Llama 模型内部名称
        match weight_name {
            "model.embed_tokens.weight" => Some("embedding.weight".to_string()),
            "model.layers.0.self_attn.q_proj.weight" => Some("layers.0.attn.q.weight".to_string()),
            // ... 更多映射规则
            _ => None,
        }
    }
}
```

## 4. 工作流程设计

### 4.1 启动与注册阶段

1. **初始化 Registry**：创建 `ModelRegistry` 实例，设置默认设备类型
2. **注册架构**：调用 `register_architecture` 方法，将架构名称与构建器函数关联
   ```rust
   registry.register_architecture("LlamaForCausalLM", llama3::builder)?;
   registry.register_architecture("Qwen2ForCausalLM", qwen2::builder)?;
   ```
3. **Registry 就绪**：等待加载模型请求

### 4.2 模型加载阶段

1. **接收加载请求**：Worker 接收到加载模型的请求
2. **检测架构**：调用 `detect_architecture` 从模型目录的 config.json 中读取架构名称
3. **获取构建器**：根据架构名称从 Registry 中获取对应的构建器函数
4. **构建模型**：调用构建器函数，创建模型实例
5. **注册模型**：将模型实例存储到 Registry 的模型映射中
6. **返回结果**：返回加载成功的信息

### 4.3 推理阶段

1. **接收推理请求**：Worker 接收到推理请求，包含模型 ID
2. **获取模型**：从 Registry 中获取模型实例
3. **执行推理**：调用模型的 `forward` 或 `forward_paged` 方法
4. **返回结果**：将推理结果返回给调用方

### 4.4 卸载阶段

1. **接收卸载请求**：Worker 接收到卸载模型的请求
2. **卸载模型**：从 Registry 中移除模型实例，释放资源
3. **返回结果**：返回卸载成功的信息

## 5. 实现细节

### 5.1 线程安全设计

- 使用 `Arc<RwLock<>>` 实现线程安全的共享访问
- 读操作使用 `read()` 锁，支持并发访问
- 写操作使用 `write()` 锁，确保独占访问
- 锁粒度优化：将架构映射和模型映射分离为两个独立的锁

### 5.2 错误处理

- 使用 Rust 的 `Result` 类型进行错误处理
- 定义清晰的错误类型层次结构
- 详细的错误信息，便于调试和监控

### 5.3 性能优化

- 函数指针调用，避免虚函数开销
- 延迟加载：仅在需要时才加载模型
- 高效的权重映射算法
- 支持批量操作

### 5.4 扩展性设计

- 支持动态注册新架构
- 支持热插拔模型
- 可扩展的配置系统
- 支持自定义构建器

## 6. API 设计

### 6.1 注册 API

```rust
// 创建 Registry
let registry = ModelRegistry::new(DeviceType::Cuda(0));

// 注册架构
registry.register_architecture("LlamaForCausalLM", llama3::builder)?;
registry.register_architecture("Qwen2ForCausalLM", qwen2::builder)?;
```

### 6.2 模型加载 API

```rust
// 自动检测架构并加载模型
let architecture = registry.load_model_auto(
    ModelId::new("llama3-8b"),
    Path::new("/models/llama3-8b"),
    false,
)?;

// 按指定架构加载模型
registry.load_model_by_architecture(
    ModelId::new("qwen2-1.5b"),
    Path::new("/models/qwen2-1.5b"),
    "Qwen2ForCausalLM",
    false,
)?;
```

### 6.3 模型使用 API

```rust
// 获取模型并执行推理
if let Some(mut model) = registry.get_model_mut(&ModelId::new("llama3-8b"))? {
    let logits = model.forward(&input_tokens, &positions)?;
    // 处理推理结果
}
```

### 6.4 模型管理 API

```rust
// 列出所有加载的模型
let models = registry.list_models()?;

// 列出所有注册的架构
let architectures = registry.list_architectures()?;

// 卸载模型
registry.unload_model(&ModelId::new("llama3-8b"))?;
```

## 7. 测试策略

### 7.1 单元测试

- **ModelTrait**：测试模型接口的正确性
- **ModelRegistry**：测试注册、加载、卸载等核心功能
- **WeightMapping**：测试权重映射的正确性
- **ArchitectureDetection**：测试架构检测功能

### 7.2 集成测试

- **完整工作流测试**：测试从注册到推理的完整流程
- **多模型测试**：测试同时加载多个模型的情况
- **并发测试**：测试多线程并发访问的正确性
- **错误处理测试**：测试各种错误场景的处理

### 7.3 性能测试

- **模型加载时间**：测试不同模型的加载时间
- **推理延迟**：测试模型推理的延迟
- **内存占用**：测试模型加载和运行时的内存占用
- **并发吞吐量**：测试多线程并发推理的吞吐量

## 8. 部署方案

### 8.1 配置管理

- 支持通过配置文件设置默认设备类型
- 支持动态调整模型加载参数
- 支持配置模型缓存策略

### 8.2 监控与日志

- 详细的日志记录，包括模型加载、卸载、推理等关键事件
- 性能指标监控，如加载时间、推理延迟、内存占用等
- 支持集成 Prometheus 等监控系统

### 8.3 高可用性设计

- 支持模型的热备份
- 支持故障自动恢复
- 支持负载均衡

## 9. 与现有系统的集成

### 9.1 与 Worker 的集成

- Worker 作为 Registry 的使用者，通过 API 加载和使用模型
- Worker 负责处理推理请求，Registry 负责模型管理

### 9.2 与 Scheduler 的集成

- Scheduler 负责请求调度和批处理
- Registry 提供模型信息，如显存需求、支持的特性等，供 Scheduler 决策

### 9.3 与 Loader 的集成

- Loader 负责从磁盘加载权重到内存
- Registry 负责将内存中的权重组装成模型实例

## 10. 未来扩展方向

### 10.1 支持更多模型类型

- 支持 MoE 模型（如 Mixtral）
- 支持多模态模型（如 Qwen3-VL）
- 支持扩散模型（如 Stable Diffusion 3）

### 10.2 优化性能

- 支持模型量化（INT8、INT4 等）
- 支持模型并行和流水线并行
- 支持自动混合精度推理

### 10.3 增强功能

- 支持模型微调
- 支持模型蒸馏
- 支持模型剪枝

### 10.4 改进开发体验

- 提供更丰富的 API 文档
- 提供示例代码和教程
- 支持自动生成模型构建器

## 11. 设计决策与权衡

### 11.1 为什么使用函数指针而不是 trait 对象？

- **性能考虑**：函数指针调用比虚函数调用更高效
- **简洁性**：函数指针类型更简洁，易于使用
- **灵活性**：支持不同的构建逻辑，包括闭包和普通函数

### 11.2 为什么使用 Arc<RwLock<>> 而不是 Mutex？

- **并发性能**：RwLock 支持多读者单写者，适合读多写少的场景
- **死锁风险**：RwLock 的死锁风险较低
- **内存开销**：Arc<RwLock<>> 的内存开销较小

### 11.3 为什么将 Registry 设计为可变的？

- **动态性**：支持运行时注册和卸载模型
- **灵活性**：适应不同的部署场景
- **可扩展性**：便于未来添加新功能

### 11.4 为什么不使用全局单例？

- **测试便利性**：非全局设计便于单元测试
- **灵活性**：支持多个 Registry 实例共存
- **清晰的依赖关系**：显式传递 Registry 实例，便于跟踪依赖

## 12. 总结

本文档详细设计了一个基于架构名称分发的 Model Registry 系统，该系统具有以下特点：

1. **解耦设计**：将模型架构名称与构建逻辑分离，实现完全解耦
2. **高性能**：采用函数指针和高效的锁机制，确保低开销
3. **线程安全**：支持多线程并发访问
4. **易于扩展**：新增模型无需修改核心代码
5. **灵活的 API**：提供简洁易用的 API，便于集成到现有系统
6. **完善的错误处理**：清晰的错误类型和详细的错误信息

该设计充分利用了 Rust 的静态类型特性和并发安全特性，结合了 sglang 和 vllm 的设计模式，为 LLM 推理框架提供了一个优雅、高效、灵活的模型管理解决方案。