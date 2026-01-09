use anyhow::Result;
use axum::{Router, routing::{get, post}};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod chat;
mod config;
mod inference;

use config::ServerConfig;
use inference::InferenceEngine;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-server")]
#[command(about = "RustInfer HTTP inference server with OpenAI-compatible API", long_about = None)]
struct Args {
    /// Path to model directory
    #[arg(short, long, env = "MODEL_PATH")]
    model: String,

    /// Server host
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    host: String,

    /// Server port
    #[arg(short, long, default_value = "8000", env = "PORT")]
    port: u16,

    /// Device: cpu or cuda:0, cuda:1, etc.
    #[arg(short, long, default_value = "cuda:0", env = "DEVICE")]
    device: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512", env = "MAX_TOKENS")]
    max_tokens: usize,

    /// Log level: trace, debug, info, warn, error
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

    // Parse device
    let device = parse_device(&args.device)?;

    // Load model
    tracing::info!("Loading model from {}...", args.model);
    let config = ServerConfig {
        model_path: args.model.clone(),
        device,
        max_tokens: args.max_tokens,
    };

    let engine = InferenceEngine::new(config).await?;
    let engine = Arc::new(Mutex::new(engine));

    tracing::info!("Model loaded successfully!");

    // Build router
    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(api::openai::chat_completions))
        .route("/v1/models", get(api::openai::list_models))

        // System metrics
        .route("/v1/metrics", get(api::metrics::get_system_metrics))

        // Health & status
        .route("/health", get(api::health::health_check))
        .route("/ready", get(api::health::ready_check))

        // Share state
        .with_state(engine)

        // Middleware
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("🚀 RustInfer server listening on http://{}:{}", args.host, args.port);
    tracing::info!("📚 API docs: http://{}:{}/health", args.host, args.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn parse_device(s: &str) -> Result<infer_core::base::DeviceType> {
    use infer_core::base::DeviceType;

    match s.to_lowercase().as_str() {
        "cpu" => Ok(DeviceType::Cpu),
        s if s.starts_with("cuda:") => {
            let device_id: i32 = s[5..].parse()?;
            Ok(DeviceType::Cuda(device_id))
        }
        _ => Err(anyhow::anyhow!("Invalid device: {}. Use 'cpu' or 'cuda:0'", s)),
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
    tracing::info!("Shutdown signal received, gracefully shutting down...");
}
