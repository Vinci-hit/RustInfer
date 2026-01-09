# RustInfer Frontend - 带实时可观测性的Web界面

> 基于Dioxus的现代化Web应用，提供交互式LLM推理和全面的性能监控。

## 🎯 功能特性

- **多轮对话** - 带完整对话历史的交互式聊天
- **实时性能指标** - 显示每条消息的详细指标：
  - ⚡ 预填充时间（提示词处理）
  - 🔄 解码时间（token生成）
  - 🚀 每秒token数
  - 📝 生成的总token数
- **系统监控仪表板** - 每2秒更新的实时指标：
  - CPU使用率和核心数
  - 内存使用情况（已用/总计/可用）
  - GPU利用率和显存（如果可用）
  - GPU温度
- **现代化UI** - Tailwind CSS深色主题
- **响应式布局** - 2/3聊天界面，1/3指标面板
- **基于WASM** - 完全在浏览器中运行，无需后端渲染

## 🚀 快速开始

```bash
# 1. 启动后端服务器（在项目根目录）
cargo run --release --bin rustinfer-server -- \
    --model /path/to/llama3/model \
    --port 8000

# 2. 启动前端开发服务器
cd crates/infer-frontend
dx serve --port 3000

# 3. 打开浏览器
# 访问: http://localhost:3000
```

## 📦 依赖

- **Dioxus 0.6** - 响应式Web框架
- **reqwest** - HTTP客户端（WASM兼容）
- **gloo-net** - Rust的Web API（未来用于EventSource流式传输）
- **Tailwind CSS** - 实用优先的CSS框架

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────┐
│           浏览器（WASM应用程序）                     │
│                                                       │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │  聊天界面        │      │  指标面板        │    │
│  │  (2/3宽度)       │      │  (1/3宽度)       │    │
│  │                  │      │                  │    │
│  │  • 消息列表      │      │  • CPU使用率     │    │
│  │  • 输入框        │      │  • 内存统计      │    │
│  │  • 每条消息的    │      │  • GPU信息       │    │
│  │    指标          │      │  • 轮询/v1/      │    │
│  │                  │      │    metrics       │    │
│  └────────┬─────────┘      └────────┬─────────┘    │
│           │                         │               │
└───────────┼─────────────────────────┼───────────────┘
            │ POST                    │ GET (每2秒)
            │ /v1/chat/completions    │ /v1/metrics
            ▼                         ▼
┌─────────────────────────────────────────────────────┐
│         后端服务器 (localhost:8000)                  │
│                                                      │
│  • 基于Llama3模型的推理引擎                         │
│  • OpenAI兼容API                                    │
│  • 性能指标收集                                      │
│  • 系统资源监控                                      │
└─────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
crates/infer-frontend/
├── Cargo.toml              # 依赖配置
├── Dioxus.toml             # Dioxus构建配置
├── README.md               # 英文文档
├── README_CN.md            # 中文文档（本文件）
├── src/
│   ├── main.rs             # 应用入口点与布局
│   ├── api/
│   │   └── client.rs       # 后端HTTP客户端
│   ├── state/
│   │   ├── conversation.rs # 消息类型与状态
│   │   └── metrics.rs      # 系统指标类型
│   └── components/
│       ├── chat_interface.rs       # 主聊天UI
│       ├── message_bubble.rs       # 带指标的单条消息
│       ├── metrics_panel.rs        # 系统监控仪表板
│       └── streaming_indicator.rs  # 加载动画
└── assets/
    └── main.css            # Tailwind样式
```

## 🔧 开发

### 构建

```bash
# 开发构建
dx build

# 发布构建（优化的WASM）
dx build --release

# 输出位置：
# target/dx/rustinfer-frontend/release/web/public/
```

### 运行

```bash
# 带热重载的开发服务器
dx serve --port 3000

# 服务器会在文件更改时自动重新构建
```

### 测试

```bash
# 首先确保后端正在运行
cargo run --release --bin rustinfer-server -- --model /path/to/model

# 然后访问前端 http://localhost:3000
# 测试以下内容：
# 1. 发送消息并验证响应出现
# 2. 检查助手消息的指标显示
# 3. 验证系统指标面板每2秒更新
# 4. 测试多轮对话
```

## 💡 使用方法

### 发送消息

1. 在输入框中输入消息
2. 点击"Send"或按Enter键
3. 在模型生成时观察流式指示器
4. 查看带有详细性能指标的响应

### 阅读性能指标

每条助手消息显示：
- **预填充时间**：处理提示词的时间（毫秒）
- **解码时间**：生成响应的时间（毫秒）
- **速度**：每秒生成的token数
- **Token数**：响应中的总token数

### 监控系统资源

右侧面板显示实时系统指标：
- **CPU**：当前利用率百分比和核心数
- **内存**：已用 vs 总内存
- **GPU**：利用率百分比、显存使用情况、温度（如果可用）

## 🎨 自定义

### 样式定制

编辑 `assets/main.css` 进行自定义：
```css
/* 更改配色方案 */
.bg-gray-900 { background-color: #your-color; }

/* 修改动画 */
@keyframes bounce { ... }
```

### API配置

前端默认期望后端在 `http://localhost:8000`。要更改：

编辑 `src/api/client.rs`：
```rust
impl ApiClient {
    pub fn new() -> Self {
        Self {
            base_url: "http://your-backend:port".to_string(),
            // ...
        }
    }
}
```

## 🚀 部署

### 选项1：静态托管

```bash
# 构建发布版WASM
dx build --release

# 提供输出目录服务
cd target/dx/rustinfer-frontend/release/web/public
python3 -m http.server 8080

# 或使用任何静态文件服务器（nginx、caddy等）
```

### 选项2：与后端集成

配置后端以提供前端服务：

```rust
// 在后端 main.rs 中
let app = Router::new()
    // ... API路由 ...
    .nest_service("/", ServeDir::new("path/to/frontend/dist"));
```

## 🐛 故障排除

### 前端无法连接到后端

- 检查后端是否在端口8000上运行
- 验证后端已启用CORS
- 检查浏览器控制台的错误信息

### 指标未更新

- 确保 `/v1/metrics` 端点可访问
- 检查浏览器网络选项卡的404错误
- 验证后端构建时包含了指标支持

### 构建错误

```bash
# 清理并重新构建
cargo clean
dx build

# 检查Dioxus CLI版本
dx --version  # 应该是 0.6.x
```

## 📚 API集成

前端使用两个主要端点：

### 聊天补全

```rust
// POST /v1/chat/completions
ChatRequest {
    model: "llama3",
    messages: vec![/* 对话历史 */],
    max_tokens: Some(150),
    stream: false,
}
```

### 系统指标

```rust
// GET /v1/metrics (每2秒轮询一次)
// 返回: SystemMetrics { cpu, memory, gpu, timestamp }
```

## 🎯 技术亮点

### 响应式架构

- **状态管理**：使用Dioxus的 `use_state` 和 `use_ref` 实现响应式状态更新
- **并发请求**：通过 `cx.spawn` 异步处理API调用，不阻塞UI
- **轮询机制**：使用 `use_future` 实现系统指标的定时轮询

### 性能优化

- **零拷贝**：WASM二进制直接在浏览器中运行
- **懒加载**：组件按需渲染
- **批量更新**：Dioxus的虚拟DOM确保最小化重绘

### 用户体验

- **实时反馈**：动画加载指示器提供视觉反馈
- **指标可视化**：每条消息都有清晰的性能数据展示
- **响应式设计**：适配不同屏幕尺寸

## 🔄 与后端的数据流

### 对话流程

```
1. 用户输入消息 → 添加到本地状态
                ↓
2. 构建API请求 → 包含完整对话历史
                ↓
3. POST到后端   → /v1/chat/completions
                ↓
4. 后端推理     → Llama3生成响应
                ↓
5. 返回响应     → 包含文本 + 性能指标
                ↓
6. 更新UI       → 显示助手消息和指标
```

### 指标流程

```
1. 组件挂载 → 启动轮询定时器（2秒间隔）
            ↓
2. GET请求  → /v1/metrics
            ↓
3. 后端采集 → sysinfo（CPU/内存）+ NVML（GPU）
            ↓
4. 返回JSON → SystemMetrics结构
            ↓
5. 更新UI   → 仪表板显示最新数据
            ↓
6. 等待2秒  → 重复步骤2
```

## 🛠️ 扩展建议

### 添加流式响应

当前使用非流式模式，未来可以添加SSE支持：

```rust
// 使用 gloo-net 的 EventSource
use gloo_net::eventsource::EventSource;

let es = EventSource::new("/v1/chat/completions?stream=true")?;
es.on_message(|msg| {
    // 逐token更新UI
});
```

### 添加对话历史持久化

```rust
// 使用 localStorage 或 IndexedDB
use web_sys::window;

fn save_conversation(messages: &[Message]) {
    let storage = window()?.local_storage()?;
    let json = serde_json::to_string(messages)?;
    storage.set_item("conversation", &json)?;
}
```

### 添加主题切换

```rust
// 实现明暗主题切换
let theme = use_state(cx, || "dark");

rsx! {
    div { class: "bg-{theme}-900 text-{theme}-100",
        button { onclick: |_| theme.set(if *theme.get() == "dark" { "light" } else { "dark" }),
            "切换主题"
        }
    }
}
```

## 🤝 贡献

详见主项目README的贡献指南。

### 代码风格

- 遵循Rust标准格式 (`cargo fmt`)
- 使用有意义的组件和变量名
- 为复杂组件添加文档注释
- 保持组件职责单一

## 📄 许可证

本项目是RustInfer的一部分，使用与主项目相同的许可证。

---

## 🙏 致谢

**构建工具：**
- 🦀 Rust + WASM - 性能与安全性
- ⚡ Dioxus 0.6 - 现代化响应式框架
- 🎨 Tailwind CSS - 快速样式开发
- 🌐 现代Web标准 - 浏览器原生支持

**设计理念：**
- 用户体验优先 - 清晰的视觉反馈和指标展示
- 性能可观测性 - 让推理过程透明化
- 模块化架构 - 易于维护和扩展

---

## 📞 支持

如有问题或疑问：
- 在GitHub上开issue
- 查看[后端服务器文档](../infer-server/README_CN.md)
- 查看[核心库文档](../infer-core/README.md)

**前端无法正常工作？**
1. 检查后端服务器是否启动：`curl http://localhost:8000/health`
2. 检查浏览器控制台的错误信息
3. 验证CORS配置：后端应允许 `http://localhost:3000` 访问
4. 使用 `RUST_LOG=debug dx serve` 查看详细日志
