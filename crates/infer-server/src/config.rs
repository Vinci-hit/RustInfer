//! 服务器配置
//!
//! 启动配置统一由 `infer_protocol::RustInferConfig`（共享 TOML）提供；
//! 模型类型从模型目录的 `config.json` 解析（见
//! `infer_protocol::resolve_model_type`）。此处不再保留独立的 CLI 配置结构。
