# RustInfer crates/infer-worker 目录结构和关键代码探索报告

## 一、模块组织方式

### 目录结构
```
crates/infer-worker/src/
├── lib.rs                 # 模块入口
├── base/                  # 基础设施层
│   ├── mod.rs            # DeviceType, DataType 定义
│   ├── allocator.rs      # 内存分配器（CPU和CUDA）
│   ├── buffer.rs         # Buffer 封装
│   └── error.rs          # 错误处理
├── cuda/                 # CUDA 相关支持
│   ├── mod.rs           
│   ├── device.rs        # 设备管理
│   ├── config.rs        # CUDA 配置（Stream, cuBLAS等）
│   ├── ffi.rs           # CUDA FFI 绑定
│   └── error.rs         # CUDA 错误处理
├── tensor/              # Tensor 类型系统
│   └── mod.rs          # Tensor 和 TypedTensor 定义
├── model/              # 模型层
│   ├── mod.rs          # 模型接口和 ModelLoader
│   ├── config.rs       # 模型配置结构
│   ├── llama3.rs       # Llama3 具体实现（~954行）
│   ├── kvcache.rs      # KV Cache 实现
│   ├── safetensor_loader.rs  # 权重加载
│   ├── tokenizer.rs    # Tokenizer
│   ├── factory.rs      # 模型工厂
│   ├── registry.rs     # 模型注册表
│   └── layers/         # 层抽象
│       ├── decoder_layers.rs    # 通用 Decoder 层
│       ├── weight_mapping.rs    # 权重映射（Llama/Qwen等）
│       └── mod.rs
├── op/                 # 操作层（矩阵乘、Attention等）
└── worker/            # Worker 实现
```

---

## 二、数据类型定义

### 2.1 DeviceType 和 DataType

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/base/mod.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(i32),  // i32 为设备 ID
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    UnKnown,
    F32,      // 4 字节
    F16,      // 2 字节
    I8,       // 1 字节
    I16,      // 2 字节
    I32,      // 4 字节
    BF16,     // 2 字节
}
```

**关键方法**:
- `DeviceType::is_cpu()` / `is_cuda()`: 检查设备类型
- `DataType::size_in_bytes()`: 获取数据类型的字节大小

### 2.2 Tensor 类型系统

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/tensor/mod.rs`

#### Dtype Trait
```rust
pub trait Dtype: Send + Sync + Copy + 'static {
    const DTYPE: DataType;
}

// 实现
impl Dtype for f32 { const DTYPE: DataType = DataType::F32; }
impl Dtype for i32 { const DTYPE: DataType = DataType::I32; }
impl Dtype for i8 { const DTYPE: DataType = DataType::I8; }
impl Dtype for bf16 { const DTYPE: DataType = DataType::BF16; }
```

#### TypedTensor 结构体
```rust
pub struct TypedTensor<T: Dtype> {
    dims: Arc<[usize]>,           // 形状
    num_elements: usize,           // 元素总数
    buffer: Buffer,                // 底层存储
    _phantom: std::marker::PhantomData<T>,
}
```

#### 动态类型 Tensor 枚举
```rust
pub enum Tensor {
    F32(TypedTensor<f32>),
    I32(TypedTensor<i32>),
    I8(TypedTensor<i8>),
    BF16(TypedTensor<bf16>),
}
```

**关键方法**:
- `Tensor::new(shape, dtype, device)`: 创建新张量
- `Tensor::from_buffer()`: 从 Buffer 创建
- `Tensor::reshape()`: 零拷贝重塑形状
- `Tensor::slice()`: 零拷贝切片
- `Tensor::to_cpu()` / `to_cuda()`: 设备转移
- `Tensor::to_dtype()`: 数据类型转换
- `Tensor::copy_from()`: 数据拷贝

---

## 三、模型配置结构

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/config.rs`

### ModelFileConfig (从config.json解析)
```rust
pub struct ModelFileConfig {
    pub hidden_size: usize,           // 模型维度
    pub intermediate_size: usize,     // FFN中间维度
    pub num_attention_heads: usize,   // 注意力头数
    pub num_hidden_layers: usize,     // 模型层数
    pub num_key_value_heads: usize,   // KV头数
    pub max_position_embeddings: usize,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub torch_dtype: String,          // "float32" 或 "bfloat16"
    #[serde(default)]
    pub immediate_dim: Option<usize>, // 模型特定参数
}
```

### RuntimeModelConfig (运行时使用)
```rust
pub struct RuntimeModelConfig {
    // === 直接参数 ===
    pub dim: usize,                  // hidden_size
    pub intermediate_size: usize,    // FFN中间维度
    pub layer_num: usize,            // num_hidden_layers
    pub head_num: usize,             // num_attention_heads
    pub kv_head_num: usize,          // num_key_value_heads
    pub seq_len: usize,              // 最大序列长度（默认2048）
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    
    // === 派生参数 ===
    pub kv_dim: usize,               // (dim * kv_head_num) / head_num
    pub kv_mul: usize,               // head_num / kv_head_num
    pub head_size: usize,            // dim / head_num
    
    pub is_shared_weight: bool,
    pub torch_dtype: String,
    pub immediate_dim: Option<usize>,
}
```

**获取参数的方式**:
- 模型层数: `config.layer_num`
- 注意力头数: `config.head_num` 和 `config.kv_head_num`
- Head大小: `config.head_size = config.dim / config.head_num`
- KV维度: `config.kv_dim = (config.dim * config.kv_head_num) / config.head_num`

---

## 四、KVCache 结构和实现

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/llama3.rs` (第39-133行)

### KvCache 结构体
```rust
struct KvCache {
    cache: Vec<(Tensor, Tensor)>,  // 每层一个(K, V)张量对
}
```

### 初始化方法
```rust
pub fn init_kv_cache(
    config: &RuntimeModelConfig,
    device: &DeviceType
) -> Result<Self> {
    let cache_shape = vec![
        config.seq_len,                        // 序列长度维度
        config.kv_head_num * config.head_size // KV维度
    ];
    
    // 数据类型选择：CPU用F32，GPU根据torch_dtype选择
    let float_type = if device.is_cpu() {
        DataType::F32
    } else {
        match config.torch_dtype.as_str() {
            "float32" => DataType::F32,
            "bfloat16" => DataType::BF16,
            _ => return Err(...),
        }
    };
    
    let mut kv_cache = Vec::with_capacity(config.layer_num);
    for _ in 0..config.layer_num {
        let k_cache = Tensor::new(&cache_shape, float_type, *device)?;
        let v_cache = Tensor::new(&cache_shape, float_type, *device)?;
        kv_cache.push((k_cache, v_cache));
    }
    
    Ok(KvCache { cache: kv_cache })
}
```

### KVCache 切片方法
```rust
pub fn slice_kv_cache(
    &mut self, 
    layer_idx: usize, 
    start_pos: i32, 
    len: usize,
    kv_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let (k_cache_full, v_cache_full) = self.get_mut(layer_idx)?;
    
    // 在第一维（序列维）进行切片
    let k_slice = k_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;
    let v_slice = v_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;
    
    Ok((k_slice, v_slice))
}
```

### 关键特点
- **形状**: `[max_seq_len, kv_dim]`，其中 `kv_dim = kv_head_num * head_size`
- **每层独立**: 为模型的每一层都创建独立的K、V缓存
- **零拷贝切片**: `slice()` 方法返回同一底层Buffer的不同视图
- **设备适配**: 自动根据设备类型选择数据精度

---

## 五、内存管理系统

### 5.1 Buffer 结构（内存管理的核心）

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/base/buffer.rs`

```rust
#[derive(Clone, Debug)]
pub struct Buffer {
    inner: Option<Arc<BufferInner>>,    // 所有权管理
    ptr: NonNull<u8>,                  // 当前视图的指针
    len_bytes: usize,                  // 当前视图的大小
    device: DeviceType,                // 设备类型
}

#[derive(Debug)]
struct BufferInner {
    ptr: NonNull<u8>,                  // 实际分配的指针
    len_bytes: usize,                  // 实际分配的大小
    allocator: Arc<dyn DeviceAllocator + Send + Sync>,  // 分配器
}
```

**关键特点**:
- Arc所有权计数：最后一个Arc销毁时自动释放内存
- Buffer视图：多个Buffer可以共享同一个内存（Arc）但有不同的指针和大小
- 设备适配：自动处理CPU和CUDA内存

**核心操作**:
```rust
// 创建
pub fn new(len_bytes: usize, allocator: Arc<..>) -> Result<Self>

// 零拷贝切片
pub fn slice(&self, offset_bytes: usize, len_bytes: usize) -> Result<Self>

// 数据拷贝（自动推断方向：H2H, H2D, D2H, D2D）
pub fn copy_from(&mut self, src: &Buffer) -> Result<()>
pub fn async_copy_from(&mut self, src: &Buffer, stream: Option<cudaStream_t>) -> Result<()>

// 从Host数据填充
pub fn copy_from_host<T: Copy>(&mut self, host_slice: &[T]) -> Result<()>
```

### 5.2 分配器系统

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/base/allocator.rs`

#### DeviceAllocator Trait
```rust
pub trait DeviceAllocator: Debug {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
    fn device(&self) -> DeviceType;
}
```

#### CPU 分配器
```rust
pub struct CpuAllocator;

impl DeviceAllocator for CpuAllocator {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let ptr = std::alloc::alloc(layout);
        NonNull::new(ptr).ok_or(...)?
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        std::alloc::dealloc(ptr.as_ptr(), layout);
    }
}
```

#### CUDA 缓存分配器 (关键！)
```rust
pub struct CachingCudaAllocator {
    state: AllocatorState,
}

struct AllocatorState {
    small_pool: DashMap<i32, Vec<CudaMemoryChunk>>,    // <1MB
    large_pool: DashMap<i32, Vec<CudaMemoryChunk>>,    // >=1MB
    idle_bytes: DashMap<i32, usize>,  // 用于GC跟踪
}
```

**显存管理策略**:

| 参数 | 值 | 用途 |
|------|-----|------|
| BIG_BUFFER_THRESHOLD | 1MB | 大小块分界 |
| GC_THRESHOLD | 1GB | 垃圾回收阈值 |
| 小块分配 | First-fit | 快速分配 |
| 大块分配 | Best-fit (±1MB) | 节省内存 |

**分配流程**:
1. 获取当前CUDA设备ID
2. 根据大小选择小池或大池
3. 查找合适的空闲块（first-fit/best-fit）
4. 如果找到则标记为busy，更新GC计数
5. 如果未找到则调用`cudaMalloc`新分配
6. 当idle_bytes超过1GB时触发GC，释放所有空闲块

---

## 六、设备管理

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/cuda/device.rs`

```rust
pub fn current_device() -> Result<i32>  // 获取当前活跃设备

pub fn set_current_device(device_id: i32) -> Result<()>  // 设置当前设备
```

### CUDA 配置

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/cuda/config.rs`

```rust
pub struct CudaConfig {
    pub stream: cudaStream_t,              // CUDA stream
    pub cublaslt_handle: cublasLtHandle_t, // cuBLASLt handle
    pub cublas_handle_v2: cublasHandle_t,  // cuBLAS v2 handle
    pub workspace: *mut c_void,            // 32MB workspace
    pub workspace_size: usize,
    pub cuda_graph: Option<CudaGraph>,     // CUDA graph for optimization
}

pub struct CudaGraph {
    graph: cudaGraph_t,
    exec: cudaGraphExec_t,
}
```

**关键特点**:
- 实现了Drop trait自动清理资源
- 支持CUDA Graph捕获和重放（用于decoding优化）
- Stream同步接口：`sync_stream()`
- 线程安全：实现了Send + Sync

---

## 七、Workspace 管理

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/mod.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferType {
    // Input and Embedding
    InputTokens,          // (CPU)
    InputEmbeddings,      // [batch, dim]
    InputPos,             // (CPU)
    
    // Caches for RoPE
    SinCache,             // [seq_len, head_size]
    CosCache,             // [seq_len, head_size]
    
    // Attention
    Query,                // [batch, dim]
    AttnScores,           // [heads, seq_len, seq_len]
    AttnOutput,
    
    // FFN
    W1Output,             // [batch, intermediate_size]
    W3Output,             // [batch, intermediate_size]
    
    // General purpose
    RmsOutput,            // [batch, dim]
    
    // Output
    ForwardOutput,        // [vocab_size]
    ForwardOutputCpu,     // (CPU only)
    
    // KV Cache (temporary)
    KeyCache,             // [batch, kv_dim]
    ValueCache,           // [batch, kv_dim]
    
    IntermediateBuffer1,  // [batch, dim]
}

pub type Workspace = HashMap<BufferType, Tensor>;
```

**初始化代码** (llama3.rs第232-348行):

```rust
fn init_workspace(
    config: &RuntimeModelConfig,
    device: &DeviceType
) -> Result<Workspace> {
    let mut buffers = HashMap::new();
    
    let float_dtype = if device.is_cpu() {
        DataType::F32
    } else {
        match config.torch_dtype.as_str() {
            "float32" => DataType::F32,
            "bfloat16" => DataType::BF16,
            _ => return Err(...),
        }
    };
    
    let max_seq_len = config.seq_len;
    
    // 为所有中间计算预分配张量
    buffers.insert(BufferType::InputTokens, 
        Tensor::new(&[max_seq_len], DataType::I32, *device)?);
    
    buffers.insert(BufferType::InputEmbeddings,
        Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?);
    
    // ... 更多缓冲区
    
    Ok(buffers)
}
```

---

## 八、权重加载与映射

### 权重映射系统

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/layers/weight_mapping.rs`

```rust
pub struct WeightMapping {
    pub embedding: &'static str,
    pub rmsnorm_final: &'static str,
    pub cls: &'static str,
    pub layer_prefix: &'static str,
    
    // 按层权重名称
    pub attn_q: &'static str,
    pub attn_k: &'static str,
    pub attn_v: &'static str,
    pub attn_o: &'static str,
    pub ffn_gate: &'static str,
    pub ffn_up: &'static str,
    pub ffn_down: &'static str,
    pub rmsnorm_attn: &'static str,
    pub rmsnorm_ffn: &'static str,
}

impl WeightMapping {
    pub const LLAMA: WeightMapping = WeightMapping {
        embedding: "model.embed_tokens.weight",
        rmsnorm_final: "model.norm.weight",
        cls: "lm_head.weight",
        layer_prefix: "model.layers",
        attn_q: "self_attn.q_proj.weight",
        attn_k: "self_attn.k_proj.weight",
        attn_v: "self_attn.v_proj.weight",
        attn_o: "self_attn.o_proj.weight",
        ffn_gate: "mlp.gate_proj.weight",
        ffn_up: "mlp.up_proj.weight",
        ffn_down: "mlp.down_proj.weight",
        rmsnorm_attn: "input_layernorm.weight",
        rmsnorm_ffn: "post_attention_layernorm.weight",
    };
    
    pub fn format_layer_weight(&self, layer_idx: usize, weight_name: &str) -> String {
        format!("{}.{}.{}", self.layer_prefix, layer_idx, weight_name)
    }
}
```

### 模型加载过程

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/llama3.rs` (第138-183行)

```rust
pub fn new<P: AsRef<Path>>(
    model_dir: P,
    device_type: DeviceType,
    is_quant_model: bool
) -> Result<Self> {
    // 1. 加载config.json
    let mut loader = ModelLoader::load(model_dir.as_ref())?;
    let config = loader.config.clone();
    
    // 2. 创建Decoder层（自动加载所有权重）
    let layers = DecoderLayers::from_loader(
        &loader,
        &config,
        &WeightMapping::LLAMA,
        device_type,
        is_quant_model,
    )?;
    
    // 3. 初始化KV Cache
    let kv_cache = KvCache::init_kv_cache(&config, &device_type)?;
    
    // 4. 初始化Workspace
    let workspace = Self::init_workspace(&config, &device_type)?;
    
    // 5. 计算RoPE缓存
    Self::calculate_rope_cache(&config, &mut workspace)?;
    
    Ok(Self { config, device_type, ... })
}
```

---

## 九、权重异步加载优化

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/layers/decoder_layers.rs` (第85-267行)

**关键特点**:
1. **CUDA Stream创建**: 为权重异步转移创建独立Stream
2. **异步转移**: 使用`to_cuda_async()`在后台转移权重
3. **Stream同步**: 完成所有权重转移后才继续执行
4. **CPU回退**: 非CUDA环境自动使用同步加载

```rust
#[cfg(feature = "cuda")]
let stream = if device_type.is_cuda() {
    let mut stream_ptr: cudaStream_t = std::ptr::null_mut();
    unsafe {
        cudaStreamCreate(&mut stream_ptr)?;
    }
    Some(stream_ptr)
} else {
    None
};

// 加载第i层的WQ权重
wq_layers.push(Self::load_matmul_async(
    &weight_mapping.format_layer_weight(i, weight_mapping.attn_q),
    loader,
    device_type,
    stream
)?);

// 完成所有权重加载后同步
if let Some(stream_ptr) = stream {
    unsafe {
        cudaStreamSynchronize(stream_ptr)?;
        cudaStreamDestroy(stream_ptr)?;
    }
}
```

---

## 十、Tensor 操作示例

```rust
// 创建张量
let tensor = Tensor::new(&[batch, seq_len, hidden], DataType::BF16, DeviceType::Cuda(0))?;

// 零拷贝重塑
let reshaped = tensor.reshape(&[batch * seq_len, hidden])?;

// 零拷贝切片
let slice = tensor.slice(&[0, 0], &[1, hidden])?;  // 第一个token的hidden

// 设备转移（CPU←→GPU）
let cpu_tensor = gpu_tensor.to_cpu()?;
let gpu_tensor = cpu_tensor.to_cuda(0)?;

// 异步转移
let gpu_tensor = cpu_tensor.to_cuda_async(0, Some(cuda_stream))?;

// 数据类型转换
let f32_tensor = bf16_tensor.to_dtype(DataType::F32)?;

// 数据拷贝
dst_tensor.copy_from(&src_tensor)?;

// 获取数据（CPU only）
let slice: &[f32] = tensor.as_f32()?.as_slice()?;
let slice_mut: &mut [f32] = tensor.as_f32_mut()?.as_slice_mut()?;
```

---

## 十一、Forward 推理流程概览

### Prefill 阶段（单token或batch）

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/llama3.rs` (第630-707行)

```rust
fn forward_prefill(&mut self, tokens: &Tensor, pos_cpu: &Tensor, seq_len: usize) -> Result<i32> {
    // 1. Token Embedding: [seq_len] → [seq_len, dim]
    let x = embedding_layer.forward(input_tokens)?;
    
    // 2. 对每一层处理
    for i in 0..layer_num {
        // 2.1 Attention Block
        let attn_norm_out = rmsnorm_attn_layers[i].forward(&x)?;
        let q = wq_layers[i].forward(&attn_norm_out)?;
        let k = wk_layers[i].forward(&attn_norm_out)?;  // 新K
        let v = wv_layers[i].forward(&attn_norm_out)?;  // 新V
        
        // 2.2 RoPE
        rope_layers[i].forward(&q, &k)?;  // 原地修改
        
        // 2.3 写入KV Cache
        kv_cache.slice_kv_cache(i, pos, seq_len)?;  // 获取切片
        scatter(&k, &v, pos);  // 写入KV Cache
        
        // 2.4 Multi-Head Attention
        attn_out = mha_layers[i].forward(&q, k_history, v_history)?;
        wo_out = wo_layers[i].forward(&attn_out)?;
        x = add(&x, &wo_out)?;  // 残差连接
        
        // 2.5 FFN Block
        ffn_norm_out = rmsnorm_ffn_layers[i].forward(&x)?;
        w1_out = w1_layers[i].forward(&ffn_norm_out)?;
        w3_out = w3_layers[i].forward(&ffn_norm_out)?;
        w1_out = swiglu(&w3_out, &w1_out)?;  // SwiGLU激活
        w2_out = w2_layers[i].forward(&w1_out)?;
        x = add(&x, &w2_out)?;  // 残差连接
    }
    
    // 3. Final Norm & Classifier
    final_norm_out = rmsnorm_final_layer.forward(&x)?;
    logits = cls_layer.forward(&final_norm_out)?;
    
    // 4. Sampling
    next_token = sampler.sample(&logits)?;
    
    Ok(next_token)
}
```

### Decoding 阶段（单token）

**文件**: `/home/vinci/RustInfer/crates/infer-worker/src/model/llama3.rs` (第420-512行)

```rust
fn forward_decoding(&mut self, tokens: &Tensor, pos_cpu: &Tensor) -> Result<i32> {
    // 1. Token Embedding: [1] → [1, dim]
    let x = embedding_layer.forward(input_token)?;
    
    // 2-3. 同prefill，但seq_len=1
    // （省略细节，流程相同）
    
    // 4. 关键优化：CUDA Graph
    if cuda_graph_enabled {
        launch_graph()?;  // 重放已捕获的计算图
        sync_stream()?;
    }
    
    Ok(next_token)
}
```

---

## 十二、总结与关键概念

### 关键数据流向

```
Config.json
    ↓
RuntimeModelConfig (dim, layer_num, head_num等)
    ↓
[权重加载] → Tensor (模型参数)
    ↓
Workspace (中间计算缓冲区预分配)
    ↓
KvCache (每层一对K、V张量)
    ↓
Forward推理: Input → Embedding → 32层Transformer → Logits → Sampling
    ↑        ↑         ↑          ↑                  ↑
    └─────────────────────────────────────────────────┘
         (所有中间结果复用Workspace的张量)
```

### 显存优化策略

1. **预分配Workspace**: 避免推理时频繁分配/释放
2. **张量切片复用**: 通过 `slice()` 实现零拷贝
3. **缓存分配器**: 小块(<1MB)用first-fit，大块(≥1MB)用best-fit
4. **GC机制**: idle显存>1GB时自动释放空闲块
5. **异步权重加载**: 使用CUDA Stream减少加载时间
6. **CUDA Graph捕获**: Decoding第一次完整计算后捕获图，之后只需重放

### 多设备支持

- **CPU**: 使用std::alloc分配堆内存
- **CUDA**: 使用cudaMalloc分配显存，自动根据设备ID管理多卡

---

## 十三、文件路径参考

完整的文件路径（绝对路径）:

1. `/home/vinci/RustInfer/crates/infer-worker/src/lib.rs` - 模块入口
2. `/home/vinci/RustInfer/crates/infer-worker/src/base/mod.rs` - DeviceType, DataType
3. `/home/vinci/RustInfer/crates/infer-worker/src/base/buffer.rs` - Buffer 实现 (576行)
4. `/home/vinci/RustInfer/crates/infer-worker/src/base/allocator.rs` - 分配器 (217行)
5. `/home/vinci/RustInfer/crates/infer-worker/src/tensor/mod.rs` - Tensor类型系统
6. `/home/vinci/RustInfer/crates/infer-worker/src/model/llama3.rs` - Llama3实现 (954行)
7. `/home/vinci/RustInfer/crates/infer-worker/src/model/kvcache.rs` - KvCache
8. `/home/vinci/RustInfer/crates/infer-worker/src/model/config.rs` - 模型配置
9. `/home/vinci/RustInfer/crates/infer-worker/src/model/mod.rs` - 模型接口、Workspace定义
10. `/home/vinci/RustInfer/crates/infer-worker/src/model/layers/decoder_layers.rs` - 通用Decoder层
11. `/home/vinci/RustInfer/crates/infer-worker/src/model/layers/weight_mapping.rs` - 权重映射
12. `/home/vinci/RustInfer/crates/infer-worker/src/cuda/device.rs` - CUDA设备管理
13. `/home/vinci/RustInfer/crates/infer-worker/src/cuda/config.rs` - CUDA配置、Graph

