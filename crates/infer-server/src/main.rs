use anyhow::Result;
use axum::{Router, routing::{get, post}};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod chat;
mod zmq_client;

use zmq_client::ZmqClient;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-server")]
#[command(about = "RustInfer API Server - Connects to rustinfer-engine via ZMQ", long_about = None)]
struct Args {
    /// Server host
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    host: String,

    /// Server port
    #[arg(short, long, default_value = "8000", env = "PORT")]
    port: u16,

    /// Engine endpoint (ZMQ地址)
    #[arg(short, long, default_value = "ipc:///tmp/rustinfer.ipc", env = "ENGINE_ENDPOINT")]
    engine_endpoint: String,

    /// Log level: trace, debug, info, warn, errorf
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🚀 RustInfer API Server starting...");
    tracing::info!("  Connecting to engine: {}", args.engine_endpoint);

    // 连接到Engine
    let zmq_client = Arc::new(ZmqClient::new(&args.engine_endpoint).await?);

    tracing::info!("✅ Connected to engine successfully!");

    // 构建Router (先创建需要state的部分)
    let stateful_router = Router::new()
        .route("/v1/chat/completions", post(api::openai::chat_completions))
        .route("/v1/models", get(api::openai::list_models))
        .route("/health", get(api::health::health_check))
        .route("/ready", get(api::health::ready_check))
        .with_state(zmq_client);

    // 合并不需要state的路由
    let app = stateful_router
        .merge(
            Router::new()
                .route("/v1/metrics", get(api::metrics::get_system_metrics))
        )
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("🚀 API Server listening on http://{}:{}", args.host, args.port);
    tracing::info!("📚 Endpoints:");
    tracing::info!("   - POST /v1/chat/completions");
    tracing::info!("   - GET  /v1/models");
    tracing::info!("   - GET  /health");
    tracing::info!("   - GET  /ready");

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
    tracing::info!("Shutdown signal received, gracefully shutting down...");
}
