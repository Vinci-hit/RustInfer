# RustInfer Frontend 设计文档

## 1. 设计哲学

### 1.1 核心原则

#### 1.1.1 响应式优先 (Reactive-First)
- 采用 Dioxus 框架的信号（Signals）机制实现细粒度的响应式状态管理
- 状态变化自动触发 UI 更新，无需手动 DOM 操作
- 单向数据流：State → Props → UI

#### 1.1.2 组件化架构 (Component-Based)
- 每个UI元素都封装为独立、可复用的组件
- 组件通过 Props 通信，保持清晰的依赖关系
- 遵循"单一职责原则"，每个组件只做一件事

#### 1.1.3 WebAssembly 原生性能
- 将 Rust 编译为 WebAssembly，在浏览器中实现接近原生的性能
- 零运行时开销，相比 JavaScript 框架有显著性能优势
- 利用 Rust 的类型系统和内存安全保证

#### 1.1.4 渐进增强 (Progressive Enhancement)
- 核心功能（聊天）不依赖外部库
- 富媒体功能（LaTeX、Mermaid 图表）按需加载
- 优雅降级：当外部资源不可用时，核心功能依然可用

### 1.2 架构理念

```
┌─────────────────────────────────────────────────────┐
│                   用户浏览器                         │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         RustInfer Frontend (WASM)             │  │
│  │                                              │  │
│  │  ┌────────────┐  ┌────────────┐             │  │
│  │  │   State    │→ │ Components │→ │   DOM     │  │
│  │  │ Management │  │   Render   │  │   Tree   │  │
│  │  └────────────┘  └────────────┘             │  │
│  │       ↑               ↓                      │  │
│  │  ┌──────────────────────────┐                │  │
│  │  │      API Client         │                │  │
│  │  │  (HTTP/JSON/SSE)         │                │  │
│  │  └──────────────────────────┘                │  │
│  └──────────────────────────────────────────────┘  │
└───────────────────────┬───────────────────────────┘
                        │ HTTP/SSE
                        ↓
              ┌─────────────────┐
              │  RustInfer API  │
              │    (Server)     │
              └─────────────────┘
```

---

## 2. 服务边界

### 2.1 Frontend 的职责

#### ✅ 职责范围内

1. **用户交互 (User Interaction)**
   - 接收用户输入（文本、按钮点击等）
   - 展示系统响应
   - 提供直观的操作界面

2. **状态管理 (State Management)**
   - 维护会话历史（消息列表）
   - 跟踪UI状态（加载中、错误等）
   - 缓存性能指标

3. **UI 渲染 (UI Rendering)**
   - Markdown 到 HTML 的转换
   - 代码高亮显示
   - 响应式布局适配

4. **API 集成 (API Integration)**
   - 调用后端 API
   - 处理 HTTP 响应
   - SSE 流式数据接收

5. **性能监控 (Performance Monitoring)**
   - 采集响应时间
   - 计算吞吐量
   - 展示系统指标

#### ❌ 职责范围外

1. **业务逻辑 (Business Logic)**
   - 推理决策（完全由后端处理）
   - 缓存策略
   - 批处理优化

2. **数据持久化 (Data Persistence)**
   - 数据库操作
   - 文件系统访问
   - 长期存储管理

3. **模型推理 (Model Inference)**
   - Token 化/反 Token 化
   - KV Cache 管理
   - GPU 计算调度

4. **认证授权 (Authentication/Authorization)**
   - 用户认证
   - 权限检查
   - 会话管理（目前前端不处理）

### 2.2 与其他组件的接口

| 组件 | 通信方式 | 数据格式 | 职责划分 |
|------|----------|----------|----------|
| **API Server** | HTTP/SSE | JSON | Frontend 请求推理，Server 返回结果 |
| **浏览器** | WASM/DOM | 原生 API | Frontend 控制页面渲染 |
| **外部库 (CDN)** | HTTP | JS/CSS | Frontend 按需加载（KaTeX, Mermaid） |

### 2.3 数据流向

```
用户输入
   ↓
[输入处理]
   ↓
[API Client] ────HTTP Request───→ [API Server]
   ↓                                          ↓
[更新状态] ←───HTTP Response──────── [返回推理结果]
   ↓
[组件重渲染]
   ↓
[DOM 更新]
   ↓
用户看到响应
```

---

## 3. 架构设计

### 3.1 目录结构

```
crates/infer-frontend/
├── src/
│   ├── main.rs                    # 应用入口和路由
│   │
│   ├── api/                       # API 集成层
│   │   └── client.rs              # HTTP 客户端和数据类型
│   │
│   ├── components/                # UI 组件库
│   │   ├── chat_interface.rs      # 主聊天界面
│   │   ├── admin_console.rs       # 管理控制台
│   │   ├── message_bubble.rs      # 消息气泡
│   │   ├── metrics_panel.rs       # 性能指标面板
│   │   ├── streaming_indicator.rs # 流式加载指示器
│   │   ├── code_block.rs           # 代码块组件
│   │   ├── mermaid_diagram.rs     # Mermaid 图表
│   │   └── streaming_message.rs    # 流式消息
│   │
│   ├── state/                     # 状态管理
│   │   ├── conversation.rs        # 会话数据结构
│   │   └── metrics.rs             # 指标数据结构
│   │
│   └── utils/                     # 工具函数
│       └── markdown.rs            # Markdown 渲染管道
│
├── assets/                        # 样式资源
│   └── tailwind/                  # Tailwind CSS
│
├── public/                        # 公共资源
│   ├── katex-init.js              # KaTeX 初始化
│   └── mermaid-init.js            # Mermaid 初始化
│
├── Cargo.toml                     # Rust 依赖配置
├── Dioxus.toml                    # Dioxus 框架配置
└── package.json                   # Node.js 脚本
```

### 3.2 核心模块详解

#### 3.2.1 主应用 (main.rs)

```rust
// 路由定义
#[derive(Clone, Copy, PartialEq)]
enum Page {
    Chat,    // 聊天界面
    Admin,   // 管理控制台
}

// 主组件
#[component]
fn App() -> Element {
    let mut page = use_signal(|| Page::Chat);

    rsx! {
        div { class: "min-h-screen bg-gray-900 text-white",
            // 导航栏
            nav { class: "bg-gray-800 border-b border-gray-700",
                // 页面切换按钮
            }

            // 路由匹配
            match page() {
                Page::Chat => rsx! { ChatInterface {} },
                Page::Admin => rsx! { AdminConsole {} },
            }
        }
    }
}
```

**设计要点**：
- 单页应用（SPA）架构，无需页面刷新
- 客户端路由，切换页面速度快
- Dioxus 的 `use_signal` 提供响应式状态

#### 3.2.2 API 客户端 (api/client.rs)

```rust
pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    // 发送聊天请求
    pub async fn send_chat_request(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, ApiError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            Err(ApiError::ServerError(response.status().as_u16()))
        }
    }

    // 获取系统指标
    pub async fn get_metrics(&self) -> Result<SystemMetrics, ApiError> {
        let url = format!("{}/v1/metrics", self.base_url);
        let response = self.client
            .get(&url)
            .send()
            .await?;

        Ok(response.json().await?)
    }
}

// 数据类型定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
```

**设计要点**：
- 封装 HTTP 客户端，提供类型安全的 API
- 异步请求（`async/await`）支持非阻塞 UI
- 统一的错误处理机制

#### 3.2.3 聊天界面组件 (components/chat_interface.rs)

```rust
#[component]
pub fn ChatInterface() -> Element {
    // 响应式状态
    let mut messages = use_signal(Vec::<Message>::new);
    let mut input_text = use_signal(String::new);
    let mut is_generating = use_signal(|| false);
    let api_client = use_signal(|| ApiClient::new("http://localhost:8000".into()));

    let send_message = move |_| {
        let text = input_text();
        if text.trim().is_empty() {
            return;
        }

        // 添加用户消息
        messages.mut().push(Message {
            id: Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content: text.clone(),
            timestamp: chrono::Utc::now().timestamp(),
            metrics: None,
        });

        // 清空输入框
        input_text.set(String::new());
        is_generating.set(true);

        // 调用 API
        spawn({
            let api = api_client();
            let messages_signal = messages.clone();
            let generating = is_generating.clone();

            async move {
                match api.send_chat_request(/* ... */).await {
                    Ok(response) => {
                        // 添加助手回复
                        messages_signal.mut().push(/* ... */);
                    }
                    Err(e) => {
                        // 处理错误
                        eprintln!("API error: {:?}", e);
                    }
                }
                generating.set(false);
            }
        });
    };

    rsx! {
        div { class: "flex flex-col h-screen",
            // 消息列表
            div { class: "flex-1 overflow-y-auto p-4",
                for message in messages().iter() {
                    MessageBubble { message: message.clone() }
                }
            }

            // 输入区域
            div { class: "border-t border-gray-700 p-4",
                textarea {
                    class: "w-full bg-gray-800 text-white rounded-lg p-4",
                    placeholder: "输入消息...",
                    value: "{input_text()}",
                    oninput: move |e| input_text.set(e.value())
                }
                button {
                    class: "mt-2 px-4 py-2 bg-blue-600 rounded-lg",
                    disabled: is_generating(),
                    onclick: send_message,
                    "发送"
                }
            }
        }
    }
}
```

**设计要点**：
- 使用 Dioxus 信号管理状态
- `spawn` 执行异步操作，不阻塞 UI
- 条件渲染（如禁用发送按钮）

#### 3.2.4 消息气泡组件 (components/message_bubble.rs)

```rust
#[component]
pub fn MessageBubble(message: Message) -> Element {
    let is_user = message.role == "user";

    rsx! {
        div { class: if is_user {
            "flex justify-end"
        } else {
            "flex justify-start"
        },
            div { class: format!(
                "max-w-[70%] rounded-lg p-4 {}",
                if is_user {
                    "bg-blue-600"
                } else {
                    "bg-gray-700"
                }
            ),
                // 消息内容
                if is_user {
                    div { "{message.content}" }
                } else {
                    // Markdown 渲染
                    div { class: "prose prose-invert max-w-none",
                        dangerous_inner_html: "{render_markdown(&message.content)}"
                    }
                }

                // 时间戳
                div { class: "text-sm text-gray-400 mt-2",
                    "{format_timestamp(message.timestamp)}"
                }

                // 性能指标（仅助手消息）
                if let Some(metrics) = &message.metrics {
                    MessageMetrics { metrics: metrics.clone() }
                }
            }
        }
    }
}
```

**设计要点**：
- Props 驱动的组件设计
- 条件样式（用户/助手）
- Markdown 渲染集成

#### 3.2.5 管理控制台 (components/admin_console.rs)

```rust
#[component]
pub fn AdminConsole() -> Element {
    let mut metrics = use_signal(|| Option::<SystemMetrics>::None);
    let api_client = use_signal(|| ApiClient::new("http://localhost:8000".into()));

    // 定时刷新指标
    use_effect(move || {
        let api = api_client();
        let metrics_signal = metrics.clone();

        spawn(async move {
            loop {
                match api.get_metrics().await {
                    Ok(m) => metrics_signal.set(Some(m)),
                    Err(e) => eprintln!("Failed to fetch metrics: {:?}", e),
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        })
    });

    rsx! {
        div { class: "p-8",
            h1 { "系统监控面板" }

            if let Some(m) = metrics() {
                // CPU 使用率
                div { class: "bg-gray-800 rounded-lg p-6 mb-4",
                    h3 { "CPU 使用率" }
                    MetricsGauge { value: m.cpu.usage_percent, max: 100.0 }
                }

                // 内存使用
                div { class: "bg-gray-800 rounded-lg p-6 mb-4",
                    h3 { "内存使用" }
                    MetricsBar { used: m.memory.used_mb, total: m.memory.total_mb }
                }

                // 缓存指标
                if let Some(cache) = m.cache {
                    CacheMetricsPanel { metrics: cache }
                }

                // 引擎指标
                if let Some(engine) = m.engine {
                    EngineMetricsPanel { metrics: engine }
                }
            }
        }
    }
}
```

**设计要点**：
- 定时数据刷新（`use_effect`）
- 模块化的指标展示组件
- 优雅的数据加载状态处理

#### 3.2.6 状态管理 (state/)

**会话状态 (state/conversation.rs)**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,                      // 唯一标识
    pub role: String,                    // "user" 或 "assistant"
    pub content: String,                 // 消息内容
    pub timestamp: i64,                   // Unix 时间戳
    pub metrics: Option<MessageMetrics>, // 性能指标
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetrics {
    pub prefill_ms: u64,           // 首次解码时间
    pub decode_ms: u64,            // 解码总时间
    pub tokens_per_second: f64,    // 令牌/秒
    pub total_tokens: u32,         // 总令牌数
}
```

**系统指标 (state/metrics.rs)**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu: CpuMetrics,
    pub memory: MemoryMetrics,
    pub gpu: Option<GpuMetrics>,
    pub cache: Option<CacheMetrics>,
    pub engine: Option<EngineMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hit_rate: f64,
    pub total_requests: u64,
    pub cached_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub active_requests: u32,
    pub total_requests: u64,
    pub avg_latency_ms: f64,
}
```

### 3.3 渲染管道 (utils/markdown.rs)

```rust
// Markdown 到 HTML 的渲染流程
pub fn render_markdown(input: &str) -> String {
    let mut options = comrak::ComrakOptions::default();

    // 启用 GitHub 风格 Markdown
    options.extension.github_pre_lang = true;
    options.extension.strikethrough = true;
    options.extension.table = true;
    options.extension.autolink = true;
    options.extension.tasklist = true;

    options.render.unsafe_ = true;  // 允许 HTML

    let html = comrak::markdown_to_html(input, &options);

    // 后处理：添加代码高亮
    let highlighted = apply_syntax_highlighting(&html);

    // 后处理：处理 LaTeX 公式
    let latex_processed = process_latex(&highlighted);

    latex_processed
}
```

---

## 4. API 设计

### 4.1 内部 API

#### 4.1.1 ApiClient 接口

```rust
pub trait ApiClientTrait {
    // 发送聊天请求
    async fn send_chat_request(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, ApiError>;

    // 获取系统指标
    async fn get_metrics(&self) -> Result<SystemMetrics, ApiError>;
}
```

#### 4.1.2 数据类型

**请求类型**:
```rust
pub struct ChatRequest {
    pub model: String,                     // 模型名称
    pub messages: Vec<ChatMessage>,        // 对话历史
    pub max_tokens: Option<usize>,        // 最大生成长度
    pub stream: bool,                      // 是否流式输出
}

pub struct ChatMessage {
    pub role: String,      // "user" | "assistant" | "system"
    pub content: String,   // 消息内容
}
```

**响应类型**:
```rust
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

**错误类型**:
```rust
pub enum ApiError {
    NetworkError(String),
    ServerError(u16),
    ParseError(String),
    Timeout,
}
```

### 4.2 外部 API 调用

#### 4.2.1 端点列表

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/v1/chat/completions` | 发送聊天请求 |
| GET | `/v1/metrics` | 获取系统指标 |

#### 4.2.2 请求示例

**POST /v1/chat/completions**:
```json
{
  "model": "llama3-8b",
  "messages": [
    { "role": "user", "content": "你好，请介绍一下 Rust" }
  ],
  "max_tokens": 512,
  "stream": true
}
```

**GET /v1/metrics**:
```json
{
  "cpu": {
    "usage_percent": 45.2
  },
  "memory": {
    "used_mb": 8192,
    "total_mb": 16384
  },
  "cache": {
    "hit_rate": 0.78,
    "total_requests": 1024,
    "cached_blocks": 256
  }
}
```

### 4.3 组件 API

#### 4.3.1 Props 定义模式

所有组件都遵循统一的 Props 定义模式：

```rust
#[derive(Props, Clone, PartialEq)]
pub struct MessageBubbleProps {
    pub message: Message,
    pub on_copy: Option<EventHandler<String>>,
}

#[component]
pub fn MessageBubble(props: MessageBubbleProps) -> Element {
    // 组件实现
}
```

#### 4.3.2 事件处理

```rust
// 点击事件
button { onclick: move |_| { /* 处理逻辑 */ } }

// 输入事件
input { oninput: move |e| { /* e.value() 获取输入 */ } }

// 自定义事件回调
component { on_click: move |data| { /* 处理 data */ } }
```

---

## 5. 通信协议

### 5.1 HTTP/JSON 协议

#### 5.1.1 请求格式

```
POST /v1/chat/completions HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Accept: application/json

{
  "model": "llama3-8b",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

#### 5.1.2 响应格式

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "llama3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 12,
    "total_tokens": 20
  }
}
```

### 5.2 SSE (Server-Sent Events) 协议

用于流式输出响应：

```
GET /v1/chat/completions HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Accept: text/event-stream

{
  "model": "llama3-8b",
  "messages": [...],
  "stream": true
}
```

**流式响应**:
```
data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"Hello"}}],"finish_reason":null}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"! How"}}],"finish_reason":null}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":" can I"}}],"finish_reason":null}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":" help?"}}],"finish_reason":"stop"}

data: [DONE]
```

### 5.3 时序图

#### 5.3.1 聊天流程

```
用户                     Frontend (WASM)                  API Server
 │                           │                              │
 ├─── 输入文本 ─────────────→│                              │
 │                           ├─── 构建请求 ───────────────→│
 │                           │                              │
 │                           │                              ├─── 处理请求
 │                           │                              │
 │                           │←──── HTTP Response ──────────┤
 │                           │                              │
 │                           ├─── 更新状态 (生成中)          │
 │                           ├─── 显示加载动画              │
 │                           │                              │
 │                           │←─── SSE 事件 (token流) ──────┤
 │                           │                              │
 │                           ├─── 实时更新 UI               │
 │                           │                              │
 │                           │←───── SSE: [DONE] ──────────┤
 │                           │                              │
 │                           ├─── 更新状态 (完成)           │
 │                           ├─── 显示完整响应             │
 │                           ├─── 计算性能指标             │
 │                           │                              │
 │←─── 显示最终回复 ──────────┤                              │
 │                           │                              │
```

#### 5.3.2 指标监控流程

```
Frontend (WASM)              API Server
     │                           │
     │                           │
     ├─── GET /v1/metrics ────→│
     │                           │
     │←─── SystemMetrics ───────┤
     │                           │
     ├─── 解析 JSON             │
     ├─── 更新 Signal 状态       │
     ├─── 触发组件重渲染         │
     │                           │
     │                           │
     │←─── 显示更新后的指标 ─────┤
     │                           │
     │                           │
     │   (2秒后重复)              │
     │                           │
```

---

## 6. 扩展性设计

### 6.1 添加新功能模块

#### 6.1.1 添加新的页面

```rust
// 在 main.rs 中添加新的 Page 枚举值
#[derive(Clone, Copy, PartialEq)]
enum Page {
    Chat,
    Admin,
    History,    // 新增：历史记录页面
}

// 添加新路由
match page() {
    Page::Chat => rsx! { ChatInterface {} },
    Page::Admin => rsx! { AdminConsole {} },
    Page::History => rsx! { HistoryPanel {} },  // 新组件
}
```

#### 6.1.2 添加新组件

```rust
// 在 components/ 目录下创建新文件
// components/history_panel.rs

#[component]
pub fn HistoryPanel() -> Element {
    let mut conversations = use_signal(Vec::<Conversation>::new);

    rsx! {
        div { class: "p-8",
            h1 { "历史记录" }

            for conv in conversations().iter() {
                ConversationCard { conversation: conv.clone() }
            }
        }
    }
}
```

### 6.2 添加新的 API 端点

```rust
// 在 api/client.rs 中添加新方法
impl ApiClient {
    // 新增：获取模型列表
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, ApiError> {
        let url = format!("{}/v1/models", self.base_url);
        let response = self.client
            .get(&url)
            .send()
            .await?;

        Ok(response.json().await?)
    }

    // 新增：取消请求
    pub async fn cancel_request(&self, request_id: String) -> Result<(), ApiError> {
        let url = format!("{}/v1/cancel/{}", self.base_url, request_id);
        self.client.delete(&url).send().await?;

        Ok(())
    }
}
```

### 6.3 添加新的数据可视化

```rust
// 创建新的可视化组件
// components/performance_chart.rs

#[component]
pub fn PerformanceChart(data: Vec<DataPoint>) -> Element {
    rsx! {
        div { class: "chart-container",
            // 使用 Chart.js 或其他可视化库
            canvas { id: "perf-chart" }
        }

        script {
            r#"
            const ctx = document.getElementById('perf-chart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: { /* data */ },
                options: { /* options */ }
            });
            "#
        }
    }
}
```

### 6.4 主题系统扩展

```rust
// 创建主题上下文
#[derive(Clone, PartialEq)]
pub struct Theme {
    pub name: String,
    pub colors: ThemeColors,
}

#[derive(Clone, PartialEq)]
pub struct ThemeColors {
    pub background: String,
    pub text: String,
    pub primary: String,
    pub secondary: String,
}

// 在 main.rs 中提供主题切换
let mut current_theme = use_signal(|| Theme::dark());

rsx! {
    ThemeProvider { theme: current_theme(),
        AppContent {}
    }

    ThemeSwitcher {
        on_change: move |theme| current_theme.set(theme)
    }
}
```

---

## 7. 性能优化

### 7.1 WASM 优化

#### 7.1.1 编译优化

在 `Cargo.toml` 中配置 Profile：

```toml
[profile.release]
opt-level = 'z'        # 最小化体积
lto = true             # 链接时优化
codegen-units = 1       # 单个编译单元
strip = true           # 移除符号表

[profile.release.package."*"]
opt-level = 2          # 依赖包适度优化
```

#### 7.1.2 减小 WASM 体积

```toml
# 使用 wee_alloc（可选）
[dependencies]
wee_alloc = { version = "0.4", optional = true }

[features]
default = []
wee_alloc = ["dep:wee_alloc"]
```

### 7.2 渲染性能优化

#### 7.2.1 虚拟滚动

对于长消息列表，使用虚拟滚动：

```rust
#[component]
pub fn VirtualList(messages: Vec<Message>) -> Element {
    let visible_range = use_signal(|| (0..10).collect::<Vec<_>>());
    let container_height = 600;
    const ITEM_HEIGHT: u32 = 100;

    rsx! {
        div { class: "h-[600px] overflow-y-auto",
            style: "height: {container_height}px",
            for i in visible_range() {
                if let Some(msg) = messages.get(i) {
                    MessageBubble { message: msg.clone() }
                }
            }
        }
    }
}
```

#### 7.2.2 防抖和节流

```rust
// 防抖：延迟搜索输入处理
let debounced_search = use_debounce(input_text(), 300);

// 节流：限制指标刷新频率
let throttled_update = use_throttle(|| update_metrics(), 2000);
```

#### 7.2.3 组件记忆化

```rust
// 使用 memo 避免不必要的重渲染
#[component]
pub fn ExpensiveComponent(data: ComplexData) -> Element {
    let cached_result = use_memo(move |_| {
        compute_expensive(&data())
    });

    rsx! { div { "{cached_result()}" } }
}
```

### 7.3 网络优化

#### 7.3.1 请求缓存

```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct CachedApiClient {
    client: ApiClient,
    cache: Mutex<HashMap<String, (Instant, CachedData)>>,
}

impl CachedApiClient {
    pub async fn get_metrics_cached(&self) -> Result<SystemMetrics, ApiError> {
        let cache_key = "metrics".to_string();

        // 检查缓存
        {
            let cache = self.cache.lock().unwrap();
            if let Some((timestamp, data)) = cache.get(&cache_key) {
                if timestamp.elapsed() < Duration::from_secs(5) {
                    return Ok(data.clone());
                }
            }
        }

        // 获取新数据
        let metrics = self.client.get_metrics().await?;

        // 更新缓存
        self.cache.lock().unwrap().insert(
            cache_key,
            (Instant::now(), metrics.clone())
        );

        Ok(metrics)
    }
}
```

#### 7.3.2 批量请求

```rust
// 合并多个 API 请求
pub async fn fetch_all_metrics(&self) -> Result<AllMetrics, ApiError> {
    let (cpu, memory, cache, engine) = tokio::try_join!(
        self.get_cpu_metrics(),
        self.get_memory_metrics(),
        self.get_cache_metrics(),
        self.get_engine_metrics(),
    )?;

    Ok(AllMetrics { cpu, memory, cache, engine })
}
```

### 7.4 资源加载优化

#### 7.4.1 按需加载

```rust
// 动态加载重型组件
#[component]
pub fn CodeEditor() -> Element {
    let mut editor_loaded = use_signal(|| false);

    rsx! {
        button {
            onclick: move |_| editor_loaded.set(true),
            "加载代码编辑器"
        }

        if editor_loaded() {
            CodeEditorHeavy {}
        }
    }
}
```

#### 7.4.2 CDN 资源预加载

```html
<head>
  <link rel="preconnect" href="https://cdn.jsdelivr.net">
  <link rel="preload" href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css" as="style">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js" async></script>
</head>
```

---

## 8. 监控调试

### 8.1 日志系统

#### 8.1.1 客户端日志

```rust
// 使用 web-sys 的 console API
use web_sys::console;

fn log_info(message: &str) {
    console::info_1(&message.into());
}

fn log_error(error: &str) {
    console::error_1(&error.into());
}

fn log_debug(data: &JsValue) {
    console::log_1(data);
}
```

#### 8.1.2 结构化日志

```rust
#[derive(Debug, Serialize)]
struct LogEvent {
    timestamp: i64,
    level: String,
    message: String,
    context: serde_json::Value,
}

fn log_event(level: &str, message: &str, context: serde_json::Value) {
    let event = LogEvent {
        timestamp: chrono::Utc::now().timestamp_millis(),
        level: level.to_string(),
        message: message.to_string(),
        context,
    };

    if let Ok(json) = serde_json::to_string(&event) {
        console::log_1(&json.into());
    }
}

// 使用示例
log_event("INFO", "API request sent", json!({
    "endpoint": "/v1/chat/completions",
    "request_id": request.id,
}));
```

### 8.2 错误处理

#### 8.2.1 统一错误显示

```rust
#[component]
pub fn ErrorBanner(error: Option<String>) -> Element {
    rsx! {
        if let Some(msg) = error {
            div { class: "bg-red-600 text-white p-4 rounded-lg mb-4",
                div { class: "flex items-center",
                    span { class: "text-xl mr-2", "⚠" }
                    span { "错误: {msg}" }
                }
            }
        }
    }
}
```

#### 8.2.2 重试机制

```rust
async fn with_retry<F, Fut, T, E>(
    mut f: F,
    max_retries: u32,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut retries = 0;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if retries < max_retries => {
                retries += 1;
                log_warn(&format!("Retry {}/{}", retries, max_retries));
                tokio::time::sleep(Duration::from_millis(1000 * retries as u64)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

// 使用示例
let response = with_retry(
    || api_client.send_chat_request(request.clone()),
    3,
).await?;
```

### 8.3 性能监控

#### 8.3.1 请求耗时追踪

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub request_id: String,
    pub start_time: i64,
    pub end_time: i64,
    pub duration_ms: u64,
    pub tokens_per_second: Option<f64>,
}

impl PerformanceMetrics {
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            start_time: chrono::Utc::now().timestamp_millis(),
            end_time: 0,
            duration_ms: 0,
            tokens_per_second: None,
        }
    }

    pub fn finish(&mut self, total_tokens: u32) {
        self.end_time = chrono::Utc::now().timestamp_millis();
        self.duration_ms = (self.end_time - self.start_time) as u64;
        self.tokens_per_second = Some((total_tokens as f64) / (self.duration_ms as f64 / 1000.0));
    }
}
```

#### 8.3.2 性能报告

```rust
#[component]
pub fn PerformanceReport(metrics: PerformanceMetrics) -> Element {
    rsx! {
        div { class: "bg-gray-800 rounded-lg p-4 mt-2",
            h4 { class: "font-bold", "性能指标" }
            div { "总耗时: {metrics.duration_ms}ms" }
            if let Some(tps) = metrics.tokens_per_second {
                div { "吞吐量: {tps:.2} tokens/s" }
            }
        }
    }
}
```

### 8.4 调试工具

#### 8.4.1 开发者模式

```rust
#[derive(Clone, Copy, PartialEq)]
enum BuildMode {
    Debug,
    Release,
}

fn is_debug_mode() -> bool {
    cfg!(debug_assertions)
}

#[component]
pub fn App() -> Element {
    let debug_mode = is_debug_mode();

    rsx! {
        div {
            if debug_mode {
                DevToolsPanel {}
            }

            AppContent {}
        }
    }
}
```

#### 8.4.2 状态检查器

```rust
#[component]
pub fn StateInspector(state: UseSignal<StateType>) -> Element {
    rsx! {
        div { class: "fixed bottom-0 right-0 bg-black text-white p-4 text-xs",
            pre { "{format!("{:#?}", state())}" }
        }
    }
}
```

---

## 9. 部署考虑

### 9.1 构建流程

#### 9.1.1 开发构建

```bash
# 启动开发服务器
trunk serve

# 或使用 Dioxus CLI
dx serve
```

#### 9.1.2 生产构建

```bash
# 使用 Trunk 构建
trunk build --release

# 或使用 wasm-pack
wasm-pack build --release --target web
```

#### 9.1.3 构建优化脚本

```bash
#!/bin/bash
# build.sh

echo "Building RustInfer Frontend..."

# 清理旧构建
rm -rf dist/

# 编译 Tailwind CSS
npx tailwindcss -i assets/styles.css -o dist/output.css --minify

# 构建 WASM
trunk build --release --public-url /infer-frontend

# 优化输出
wasm-opt dist/*.wasm -Oz -o dist/*.wasm

echo "Build complete!"
```

### 9.2 资源配置

#### 9.2.1 Nginx 配置

```nginx
server {
    listen 80;
    server_name rustinfer.example.com;

    location /infer-frontend/ {
        alias /var/www/rustinfer/frontend/dist/;
        index index.html;

        # WASM MIME 类型
        types {
            application/wasm wasm;
        }

        # Gzip 压缩
        gzip on;
        gzip_types text/plain application/json application/javascript text/css application/wasm;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 9.2.2 Docker 配置

```dockerfile
# Dockerfile
FROM node:18-alpine AS builder

# 安装 Trunk
RUN cargo install trunk

# 复制源代码
WORKDIR /app
COPY . .

# 构建前端
RUN trunk build --release --public-url /infer-frontend

# 生产镜像
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html/infer-frontend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 9.3 性能考虑

#### 9.3.1 CDN 部署

```toml
# Dioxus.toml 配置
[web.app]
title = "RustInfer Frontend"

# 静态资源 CDN
[web.resource]
script = [
  "https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js",
  "https://cdn.jsdelivr.net/npm/mermaid@10.0.0/dist/mermaid.min.js"
]
```

#### 9.3.2 PWA 支持

```html
<!-- public/manifest.json -->
{
  "name": "RustInfer",
  "short_name": "RustInfer",
  "start_url": "/infer-frontend/",
  "display": "standalone",
  "background_color": "#1f2937",
  "theme_color": "#2563eb"
}
```

### 9.4 兼容性

#### 9.4.1 浏览器支持

| 浏览器 | 版本 | WASM 支持 | 状态 |
|--------|------|-----------|------|
| Chrome | 57+ | ✅ | 完全支持 |
| Firefox | 52+ | ✅ | 完全支持 |
| Safari | 11+ | ✅ | 完全支持 |
| Edge | 16+ | ✅ | 完全支持 |
| IE | - | ❌ | 不支持 |

#### 9.4.2 Polyfill 策略

```html
<!-- public/index.html -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=default"></script>
```

---

## 10. 未来规划

### 10.1 短期目标 (1-3个月)

#### 10.1.1 功能增强
- ✅ 实现完整的流式输出（目前是模拟）
- ✅ 添加用户认证和会话管理
- ✅ 支持文件上传（图片、PDF）
- ✅ 添加代码执行功能（沙箱环境）

#### 10.1.2 用户体验优化
- ✅ 键盘快捷键支持
- ✅ 自定义主题切换
- ✅ 响应式移动端优化
- ✅ 离线模式支持（PWA）

#### 10.1.3 性能优化
- ✅ 实现虚拟滚动
- ✅ 添加请求缓存机制
- ✅ 优化 WASM 体积（目标：< 500KB）
- ✅ 减少 API 调用频率

### 10.2 中期目标 (3-6个月)

#### 10.2.1 高级功能
- 🔄 多模型切换
- 🔄 参数调节界面（temperature, top_p, etc.）
- 🔄 对话模板管理
- 🔄 导出对话历史（Markdown, PDF）

#### 10.2.2 协作功能
- 🔄 共享对话链接
- 🔄 多用户会话
- 🔄 评论和标注功能

#### 10.2.3 开发者工具
- 🔄 API Playground
- 🔄 请求日志查看器
- 🔄 性能分析工具

### 10.3 长期目标 (6-12个月)

#### 10.3.1 平台扩展
- 🎯 桌面应用（Tauri）
- 🎯 移动应用（Flutter/React Native 集成）
- 🎯 CLI 工具

#### 10.3.2 生态集成
- 🎯 插件系统
- 🎯 与其他 AI 工具集成
- 🎯 企业级功能（SSO, 权限管理）

#### 10.3.3 创新功能
- 🎯 多模态输入（语音、视频）
- 🎯 协作编辑（类似 Google Docs）
- 🎯 AI 辅助界面设计

### 10.4 技术债务清理

#### 10.4.1 测试覆盖
- 添加单元测试（目标覆盖率 > 80%）
- 添加组件测试
- 添加 E2E 测试（Playwright）

#### 10.4.2 代码质量
- 统一代码风格（rustfmt）
- 添加 linter（clippy）
- 完善文档注释

#### 10.4.3 可维护性
- 重构大型组件
- 提取公共工具函数
- 改进类型系统

---

## 附录

### A. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| WASM | WebAssembly | 可在浏览器中运行的低级字节码 |
| SSE | Server-Sent Events | 服务器推送事件 |
| SPA | Single Page Application | 单页应用 |
| PWA | Progressive Web App | 渐进式 Web 应用 |
| Props | Properties | 组件属性 |
| Signal | Signal | Dioxus 的响应式状态机制 |

### B. 相关资源

- **Dioxus 文档**: https://dioxuslabs.com/docs/
- **MDN Web API**: https://developer.mozilla.org/
- **Wasm-bindgen**: https://rustwasm.github.io/wasm-bindgen/
- **Tailwind CSS**: https://tailwindcss.com/

### C. 常见问题

**Q: 为什么选择 Rust + Dioxus 而不是 React/Vue？**
A: Rust + WASM 提供更高的性能和更强的类型安全，特别适合对性能要求高的应用。

**Q: 如何调试 WASM 应用？**
A: 使用浏览器的开发者工具，配合 `console::log` 和 `console::error` API。

**Q: 如何处理跨域请求？**
A: 确保后端 API 配置了 CORS 头，或通过 Nginx 反向代理统一域名。

---

*文档版本: 1.0*
*最后更新: 2025-01-18*
*作者: GLM*
