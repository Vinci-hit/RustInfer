# RustInfer 工作台

基于 Rust / Dioxus 的本地推理前端。浅色画布与石墨色导航，铜橙色强调输入和执行；深色主题、窄屏导航和会话配置抽屉共用一套语义化样式。使用系统字体，不依赖外部字体服务。

## 启动

需要仓库指定的 Rust 工具链、`wasm32-unknown-unknown` target、Node.js / npm 和 Dioxus CLI **0.7.10**：

```bash
rustup target add wasm32-unknown-unknown
cargo install dioxus-cli --version 0.7.10 --locked
cd crates/infer-frontend
npm ci
./dev.sh
```

打开 `http://localhost:3000`，在「连接与设置」填写实际推理服务地址。默认使用当前网页主机的 `8080` 端口；支持反向代理路径和末尾 `/v1`。后端须允许该前端来源的 CORS，HTTPS 页面也应配置 HTTPS 服务。模型就绪、模型列表和能力每 15 秒重新检查；资源监控每 5 秒刷新。

构建：

```bash
npm run build:css
dx build --platform web --release --locked --debug-symbols false
```

Tailwind 和 CLI 已锁定 **4.3.3**，Dioxus **0.7.10**，Markdown 解析使用 pulldown-cmark **0.13**。Dioxus 0.8 的预发布版本未用于此项目。

## 当前可用

- 中文界面；响应式布局；真正的明暗主题切换。
- 多会话、搜索、删除确认、JSON 导出、按会话保留文字草稿。
- 浏览器本地保存历史和偏好。存储被禁用或超额时显示提示；图片较多时建议导出备份。生成中的用户消息会先保存，刷新后标记未完成回复。
- 从服务获取模型和就绪状态，模型选择、系统提示词、温度、Top P 和输出长度直接进入请求。
- 中文流式回复、安全 Markdown、复制、停止、重新生成；请求失败或流提前结束有明确提示。
- 支持视觉的模型可以选择、粘贴、拖放 PNG/JPEG，预览和移除图片，发送纯图片消息。会话全历史最多 4 张、单张 10 MiB、宽高不超过 8192，最终视觉 token 预算由服务校验。
- `Enter` 发送、`Shift+Enter` 换行，中文输入法确认不会误发送。`Ctrl/⌘ + N` 新建对话、`Ctrl/⌘ + K` 搜索（浏览器保留的快捷键可能优先于网页）。

## 多模态扩展

实际能力来自 `GET /v1/capabilities` 和 `/v1/models` 的模型能力，而非通过模型名称推断。当前后端支持文字与条件启用的 Qwen3.5 图片理解。

语音转写、语音合成、文件上传、实时语音会话有类型化客户端和请求/响应模型；音频、文件、视频内容分片及实时会话事件也已预留。相应 UI 入口标注未启用。**这些并不意味着音频推理、麦克风采集、WebRTC 或文件解析已经实现**；接入时需要后端、传输/采集逻辑和能力声明一起就绪。

完整请求示例、限制和接入顺序见 [多模态接口契约](docs/multimodal-api.md)。

## 验证

```bash
cargo check -p infer-frontend --target wasm32-unknown-unknown
cargo test -p infer-frontend
cargo test -p infer-server --lib api::capabilities
```

测试覆盖分块 UTF-8 / SSE 错误、Markdown 注入防护、文本格式兼容、能力保守降级、图片历史校验、重新生成与采样参数。浏览器检查需要启动前端，并连接实际服务或实现相同契约的测试服务。

设计参考：按用户要求检索并应用 [Anthropic frontend-design skill](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md)；框架版本与用法参考 [Dioxus 文档](https://docs.rs/dioxus/0.7.10/dioxus/)，样式构建参考 [Tailwind 官方文档](https://tailwindcss.com/docs/upgrade-guide)。
