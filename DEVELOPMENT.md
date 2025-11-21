# RustInfer 开发者指南

本文档为RustInfer项目的开发者提供基础指南和技术参考。

## 核心架构

### 主要组件

1. **Model Trait**: 所有模型实现的基础接口
   ```rust
   pub trait Model {
       fn init(&mut self, device_type: DeviceType) -> Result<()>;
       fn forward(&mut self, input: &Tensor, pos: &Tensor) -> Result<Tensor>;
       fn tokenizer(&self) -> &dyn Tokenizer;
       fn encode(&self, text: &str) -> Result<Vec<i32>>;
       fn decode(&self, ids: &[i32]) -> Result<String>;
       fn is_eos_token(&self, token_id: u32) -> bool;
       fn slice_kv_cache(&self, layer_idx: usize, start_pos: usize, end_pos: usize) -> Result<(Tensor, Tensor)>;
   }
   ```

2. **Op Trait**: 算子实现的标准接口
   ```rust
   pub trait Op {
       fn name(&self) -> &'static str;
       fn forward(&self, ctx: &mut OpContext) -> Result<()>;
   }
   ```

3. **Llama3实现**: 核心模型实现
   - 支持CPU和CUDA路径
   - 实现KV缓存管理
   - 包含RoPE编码和RMSNorm等核心组件

## 开发指南

### 添加新模型

1. 在`model/`目录下创建新模型文件
2. 实现`Model` trait接口
3. 在`model/mod.rs`中导出新模型

### 添加新算子

1. 在`op/`目录下创建新算子文件
2. 实现`Op` trait接口
3. 同时提供CPU和CUDA实现(可选)

### 开发流程

1. **设置环境**
   ```bash
   # CPU开发
   cargo build
   
   # CUDA开发
   cargo build --features cuda
   ```

2. **运行测试**
   ```bash
   cargo test
   cargo test --features cuda
   ```

3. **代码规范**
   - 遵循Rust官方风格
   - 使用`rustfmt`和`clippy`
   - 使用`anyhow`进行错误处理

## 调试建议

1. **常见错误排查**
   - CUDA错误: 检查CUDA版本兼容性
   - 内存错误: 关注张量形状匹配
   - 性能问题: 使用性能分析工具定位瓶颈

2. **测试策略**
   - 单元测试: 测试单个组件功能
   - 集成测试: 测试组件间交互
   - 性能测试: 确保性能达标

## 部署提示

1. **CUDA依赖**: 确保部署环境CUDA版本兼容
2. **内存需求**: 大型模型需要足够内存
3. **模型路径**: 确保可访问模型权重文件

## 路线图

- 支持更多模型架构
- 实现量化推理
- 分布式推理支持
- 进一步优化性能

---

感谢您的贡献！