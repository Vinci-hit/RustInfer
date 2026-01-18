//! RustInfer Server - Frontend API Server
//!
//! 架构：
//! ```text
//! HTTP Client -> Axum Server -> ZMQ Client -> Scheduler
//!                      |             ^
//!                      v             |
//!                  Tokenizer    Request Map
//! ```

mod config;
mod state;
mod processor;
mod backend;
mod http;

use anyhow::Result;
use axum::{Router, routing::post};
use config::ServerConfig;
use processor::TokenizerWrapper;
use state::AppState;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. 加载配置
    let config = ServerConfig::from_env();
    
    // 2. 初始化日志
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log.level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    tracing::info!("🚀 RustInfer Server starting...");
    tracing::info!("  Model: {}", config.model.model_name);
    tracing::info!("  Tokenizer: {}", config.model.tokenizer_path.display());
    tracing::info!("  Scheduler: {}", config.scheduler.address);
    
    // 3. 加载 Tokenizer
    tracing::info!("Loading tokenizer...");
    let tokenizer = TokenizerWrapper::from_file(&config.model.tokenizer_path)?;
    tracing::info!("✅ Tokenizer loaded, vocab_size={}", tokenizer.vocab_size());
    
    // 4. 创建通信 Channel
    let (scheduler_tx, scheduler_rx) = mpsc::unbounded_channel();
    
    // 5. 构建全局状态
    let state = Arc::new(AppState::new(
        config.clone(),
        tokenizer,
        scheduler_tx,
    ));
    
    // 6. 启动 ZMQ 后台循环
    tracing::info!("Starting ZMQ client...");
    let zmq_task = tokio::spawn({
        let scheduler_address = config.scheduler.address.clone();
        let request_map = state.request_map.clone();
        async move {
            backend::start_zmq_loop(scheduler_address, scheduler_rx, request_map).await;
        }
    });
    
    tracing::info!("✅ ZMQ client started");
    
    // 7. 构建 HTTP Router
    let app = Router::new()
        .route("/v1/chat/completions", post(http::chat_completions))
        .with_state(state.clone())
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(tower_http::trace::TraceLayer::new_for_http());
    
    // 8. 启动 HTTP Server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.http.port));
    
    tracing::info!("🚀 Server listening on http://{}:{}", config.http.host, config.http.port);
    tracing::info!("📚 Endpoints:");
    tracing::info!("   - POST /v1/chat/completions (OpenAI-compatible)");
    tracing::info!("");
    tracing::info!("Ready to accept requests!");
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    
    // 9. 运行 Server
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    
    // 10. 等待 ZMQ 任务结束
    zmq_task.await?;
    
    Ok(())
}

/// Graceful Shutdown Signal
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
    
    tracing::info!("Shutdown signal received, gracefully shutting down...");
}
