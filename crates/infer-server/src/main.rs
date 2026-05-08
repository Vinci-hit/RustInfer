use anyhow::Result;
use axum::{Router, routing::{get, post}};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_server::{AppState, ZmqClient, api};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-server")]
#[command(about = "RustInfer API Server")]
struct Args {
    /// Server host
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    host: String,

    /// Server port
    #[arg(short, long, default_value = "8000", env = "PORT")]
    port: u16,

    /// Scheduler endpoint (ZMQ地址)
    #[arg(short, long, default_value = "ipc:///tmp/rustinfer.ipc", env = "SCHEDULER_ENDPOINT")]
    engine_endpoint: String,

    /// Tokenizer 路径（tokenizer.json 所在目录）
    #[arg(short, long, env = "TOKENIZER_PATH")]
    tokenizer: String,

    /// Log level
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("RustInfer API Server starting...");
    tracing::info!("  Scheduler endpoint: {}", args.engine_endpoint);
    tracing::info!("  Tokenizer: {}", args.tokenizer);

    // 加载 tokenizer
    let tokenizer_path = std::path::Path::new(&args.tokenizer).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tracing::info!("Tokenizer loaded (vocab_size={})", tokenizer.get_vocab_size(true));

    // 连接 Scheduler
    let zmq_client = ZmqClient::new(&args.engine_endpoint).await?;
    tracing::info!("Connected to scheduler");

    let state = Arc::new(AppState { zmq_client, tokenizer });

    let app = Router::new()
        .route("/v1/chat/completions", post(api::openai::chat_completions))
        .route("/v1/models", get(api::openai::list_models))
        .route("/health", get(api::health::health_check))
        .route("/ready", get(api::health::ready_check))
        .with_state(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("API Server listening on http://{}:{}", args.host, args.port);

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
