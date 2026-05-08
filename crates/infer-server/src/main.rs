use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_server::{
    AppState, ServerConfig, ZmqClient,
    router::build_router,
    state::ModelInfo,
};

#[tokio::main]
async fn main() -> Result<()> {
    let config = ServerConfig::parse();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("RustInfer API Server starting...");
    tracing::info!("  Scheduler endpoint: {}", config.engine_endpoint);
    tracing::info!("  Tokenizer: {}", config.tokenizer);
    tracing::info!("  Model: {}", config.model_name);

    // 加载 tokenizer
    let tokenizer_path = std::path::Path::new(&config.tokenizer).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tracing::info!("Tokenizer loaded (vocab_size={})", tokenizer.get_vocab_size(true));

    // 连接 Scheduler
    let client = ZmqClient::new(&config.engine_endpoint, config.request_timeout_secs).await?;
    tracing::info!("Connected to scheduler");

    // 构建应用状态
    let model_info = ModelInfo {
        model_id: config.model_name.clone(),
        owned_by: "rustinfer".to_string(),
    };

    let state = Arc::new(AppState {
        client,
        tokenizer,
        config: config.clone(),
        model_info,
    });

    // 构建 Router
    let app = build_router(state);

    // 启动服务器
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("API Server listening on http://{}:{}", config.host, config.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
    tracing::info!("Shutdown signal received");
}
