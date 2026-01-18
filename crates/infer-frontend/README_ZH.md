# RustInfer 前端 (RustInfer Frontend)

一个为 RustInfer LLM 推理引擎打造的现代化响应式 Web 界面，使用 Rust 和 Dioxus 构建。

## 项目概述

RustInfer 前端是一个基于 WebAssembly 的 Web 应用程序，提供以下功能：
- **交互式聊天界面**：与 LLM 模型进行实时对话。
- **管理控制台**：全面的系统监控仪表板。
- **实时指标**：实时显示 CPU、内存、GPU 和缓存统计数据。
- **丰富的内容支持**：支持 Markdown 渲染、语法高亮、数学公式和图表展示。

## 技术栈

| 组件 | 技术 | 用途 |
|-----------|-----------|---------|
| **UI 框架** | Dioxus 0.7.1 | 针对 Rust 的响应式 Web 框架 |
| **构建目标** | WebAssembly | 高性能浏览器端执行 |
| **样式处理** | Tailwind CSS | 原子化 (Utility-first) CSS 框架 |
| **HTTP 客户端** | `reqwest`, `gloo-net` | REST API 和 服务器发送事件 (SSE) |
| **状态管理** | Dioxus Signals | 响应式状态管理 |
| **Markdown** | `comrak` | GitHub 风格 Markdown 渲染 |
| **语法高亮** | `syntect` | 代码块高亮 |
| **数学渲染** | KaTeX | 客户端 LaTeX 渲染 |
| **图表渲染** | Mermaid.js | 流程图及图解渲染 |

## 系统架构

### 目录结构

```
crates/infer-frontend/
├── src/
│   ├── main.rs                 # 应用程序入口点与路由配置
│   ├── components/             # UI 组件
│   │   ├── chat_interface.rs   # 主聊天界面
│   │   ├── message_bubble.rs   # 单条消息显示组件
│   │   ├── metrics_panel.rs    # 实时指标显示面板
│   │   ├── admin_console.rs    # 管理员仪表板
│   │   ├── streaming_message.rs # 流式响应处理
│   │   ├── code_block.rs       # 带语法高亮的代码块
│   │   ├── mermaid_diagram.rs  # Mermaid 图表渲染
│   │   └── streaming_indicator.rs # 加载状态指示器
│   ├── state/                  # 状态管理
│   │   ├── conversation.rs     # 消息数据结构
│   │   └── metrics.rs          # 系统指标结构
│   ├── api/                    # API 集成
│   │   └── client.rs           # HTTP 客户端与 API 类型定义
│   └── utils/                  # 工具类
│       └── markdown.rs         # Markdown 渲染引擎
├── assets/                     # 静态资源
│   ├── main.css               # Tailwind 输入文件
│   └── output.css             # 编译后的 CSS
├── public/                     # 公共文件
│   ├── katex-init.js          # KaTeX 初始化脚本
│   └── mermaid-init.js        # Mermaid 初始化脚本
├── Cargo.toml                 # Rust 依赖管理
├── Dioxus.toml                # Dioxus 配置文件
└── tailwind.config.js         # Tailwind 配置文件
```

### 组件层级

```
App (main.rs)
├── Router (路由)
│   ├── Chat Interface (聊天界面)
│   │   ├── Message List (消息列表)
│   │   │   └── Message Bubble (消息气泡)
│   │   │       ├── Markdown Content (Markdown 内容)
│   │   │       ├── Code Block (代码块)
│   │   │       └── Mermaid Diagram (Mermaid 图表)
│   │   ├── Input Area (输入区)
│   │   └── Streaming Indicator (流式加载指示器)
│   └── Metrics Panel (指标面板 - 侧边栏)
│
└── Admin Console (管理控制台 - 独立路由)
    ├── CPU/Memory/GPU Gauges (仪表盘)
    ├── RadixCache Metrics (缓存指标)
    └── Engine Performance Stats (引擎性能统计)
```

## 数据流

### 聊天请求流程

```
用户输入
    ↓
ChatInterface 组件
    ↓
[状态更新] 将用户消息添加至 messages 信号 (signal)
    ↓
ApiClient.send_message()
    ↓
HTTP POST /v1/chat/completions
    {
      "model": "llama3",
      "messages": [...],
      "max_tokens": 512,
      "stream": false
    }
    ↓
后端 (infer-server)
    ↓
收到响应
    ↓
解析 ChatResponse → 提取内容与指标
    ↓
[状态更新] 添加带有指标的助手消息
    ↓
UI 重新渲染 (响应式)
```

### 指标轮询流程

```
use_future 钩子 (轮询)
    ↓
每 2 秒 (指标面板)
每 1 秒 (管理控制台)
    ↓
GET /v1/metrics
    ↓
SystemMetrics 响应
    {
      "cpu": {...},
      "memory": {...},
      "gpu": {...},
      "cache": {...},
      "engine": {...}
    }
    ↓
[状态更新] 更新 metrics 信号
    ↓
UI 使用新数值重新渲染
```

### 状态管理模式

```rust
// 创建响应式信号
let mut messages = use_signal(Vec::<Message>::new);
let mut input_text = use_signal(String::new);
let mut is_generating = use_signal(|| false);

// 从信号中读取（自动追踪依赖）
let message_list = messages.read();

// 写入信号（触发重新渲染）
messages.write().push(new_message);

// 使用 use_future 进行异步操作
let _ = use_future(|| async move {
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        // 获取并更新指标
    }
});
```

## API 文档

### 基地址
```
http://localhost:8000
```

### 接口端点

#### 1. 聊天补全 (Chat Completion)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama3",
  "messages": [
    {"role": "user", "content": "你好！"}
  ],
  "max_tokens": 512,
  "stream": false
}
```

**响应示例:**
```json
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "你好！今天有什么我可以帮你的吗？"
    }
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  },
  "performance": {
    "prefill_time_ms": 45.2,
    "decode_time_ms": 123.8,
    "tokens_per_second": 14.5
  }
}
```

#### 2. 系统指标 (System Metrics)
```http
GET /v1/metrics
```

## 开发指南

### 前置条件

```bash
# Rust 工具链
rustup install stable
rustup target add wasm32-wasi

# Node.js (用于 Tailwind CSS)
node --version  # 版本需 >= 18

# Dioxus CLI
cargo install dioxus-cli

# 额外工具
npm install -g tailwindcss
```

### 设置

```bash
# 安装依赖
cd ~/RustInfer/crates/infer-frontend
npm install

# 编译 Tailwind CSS (执行一次)
npx tailwindcss -i ./assets/main.css -o ./assets/output.css
```

### 运行开发服务器

**在工作区根目录运行:**
```bash
cd ~/RustInfer
dx serve --package infer-frontend
```

**在前端目录运行 (使用脚本):**
```bash
cd ~/RustInfer/crates/infer-frontend
./dev.sh
```

**访问应用:**
- 聊天界面: http://localhost:3000
- 管理控制台: 点击 "Admin Console" 选项卡

### 开发工作流

1. **启动后端服务** (端口 8000):
   ```bash
   cd ~/RustInfer
   cargo run -p infer-server
   ```
2. **启动前端** (端口 3000):
   ```bash
   cd ~/RustInfer/crates/infer-frontend
   ./dev.sh
   ```
3. **修改 Rust 文件** → 通过 Dioxus 自动重载。
4. **修改 CSS** → 通过 Tailwind 监听器自动重新编译。

## 生产环境构建

### 构建 WASM

```bash
cd ~/RustInfer/crates/infer-frontend

# 优化 CSS
NODE_ENV=production npx tailwindcss \
  -i ./assets/main.css \
  -o ./assets/output.css \
  --minify

# 使用 Dioxus 构建 (优化模式)
dx build --release
```

### 部署

`dist/` 目录包含：
- `index.html` - 入口文件
- `assets/infer-frontend.js` - 编译后的 WASM 绑定脚本
- `assets/infer-frontend_bg.wasm` - WASM 二进制文件
- `assets/output.css` - 优化后的 CSS

可部署至任何静态托管服务，如 GitHub Pages、Netlify、Vercel 或 nginx。

## 组件参考

### ChatInterface (聊天界面)
- **位置**: `src/components/chat_interface.rs`
- **职责**: 管理对话状态、处理用户输入、与 API 通信、显示加载指示器。

### MetricsPanel (指标面板)
- **位置**: `src/components/metrics_panel.rs`
- **职责**: 显示实时系统指标，每 2 秒轮询一次。

### AdminConsole (管理控制台)
- **位置**: `src/components/admin_console.rs`
- **职责**: 高级监控仪表板，支持高频（1 秒）指标轮询、历史数据可视化、SVG 仪表盘展示。

## 常见问题排查

### `dx serve` 因工作区错误失败
**问题**: Dioxus CLI 在工作区中找不到包。
**解决**: 在工作区根目录使用 `--package` 参数运行，或使用 `dev.sh` 脚本。

### WASM 构建由于 katex 错误失败
**问题**: `katex` crate 需要 WASM 编译目标。
**解决**: 确保已添加目标 `rustup target add wasm32-wasi`。

### 浏览器出现 CORS 错误
**问题**: 浏览器拦截了跨域请求。
**解决**: 在后端 `infer-server` 中配置 CORS 允许前端地址访问。

## 未来规划

1. **流式响应**：支持 SSE 以实现实时 Token 流式输出。
2. **身份验证**：用户登录/登出、API 密钥管理。
3. **对话历史**：本地存储持久化、对话导入导出。
4. **多模型支持**：模型选择下拉列表及自定义参数配置。

## 许可证

属于 RustInfer 项目的一部分。详见主项目的 LICENSE 文件。