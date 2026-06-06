//! Standalone API Server
//! 
//! 这个进程只负责运行 HTTP API 服务器。
//! 它需要连接到已启动的 Scheduler 进程。
//!
//! Usage:
//!   rustinfer-api --model <model_path> --frontend-endpoint ipc:///tmp/rustinfer-XXX-frontend.ipc

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

#[derive(Parser, Debug)]
#[command(name = "rustinfer-api")]
#[command(about = "RustInfer Standalone API Server")]
struct ApiConfig {
    /// Server host
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    pub host: String,

    /// Server port
    #[arg(short, long, default_value = "8000", env = "PORT")]
    pub port: u16,

    /// Model path (directory containing model weights + tokenizer.json).
    #[arg(short, long, env = "MODEL_PATH")]
    pub model: String,

    /// Model name (用于 /v1/models 返回)
    #[arg(long, env = "MODEL_NAME")]
    pub model_name: Option<String>,

    /// Frontend ZMQ endpoint (connects to Scheduler)
    #[arg(long, env = "FRONTEND_ENDPOINT")]
    pub frontend_endpoint: String,

    /// Request timeout (seconds)
    #[arg(long, default_value = "120", env = "REQUEST_TIMEOUT_SECS")]
    pub request_timeout_secs: u64,

    /// Log level
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    pub log_level: String,

    /// Ignore EOS tokens during generation
    #[arg(long, default_value_t = false, env = "IGNORE_EOS")]
    pub ignore_eos: bool,
}

impl ApiConfig {
    fn effective_model_name(&self) -> String {
        self.model_name.clone().unwrap_or_else(|| {
            std::path::Path::new(&self.model)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("default")
                .to_string()
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_config = ApiConfig::parse();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| api_config.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let model_name = api_config.effective_model_name();

    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║       RustInfer API Server v0.1.0 (Standalone)   ║");
    tracing::info!("╚══════════════════════════════════════════════════╝");
    tracing::info!("  Model: {}", api_config.model);
    tracing::info!("  API Server Port: {}", api_config.port);
    tracing::info!("  Frontend Endpoint: {}", api_config.frontend_endpoint);

    // Load tokenizer
    tracing::info!("[server] Initializing Tokenizer and ZMQ Client...");
    let tokenizer_path = std::path::Path::new(&api_config.model).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tracing::info!("Tokenizer loaded (vocab_size={})", tokenizer.get_vocab_size(true));

    // Connect to Scheduler
    let client = ZmqClient::new(&api_config.frontend_endpoint, api_config.request_timeout_secs).await?;
    tracing::info!("Connected to scheduler via {}", api_config.frontend_endpoint);

    let model_info = ModelInfo {
        model_id: model_name,
        owned_by: "rustinfer".to_string(),
    };

    // Create a dummy ServerConfig for AppState
    // We only need the fields that AppState uses
    let server_config = ServerConfig::parse();

    let state = Arc::new(AppState {
        client,
        tokenizer,
        config: server_config,
        model_info,
    });

    let app = build_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], api_config.port));
    tracing::info!("API Server listening on http://{}:{}", api_config.host, api_config.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Create shutdown channel
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);
    
    // Spawn axum server
    let mut axum_rx = shutdown_tx.subscribe();
    let server_task = tokio::spawn(async move {
        let result = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = axum_rx.recv().await;
                tracing::info!("Axum server shutting down gracefully.");
            })
            .await;
        if let Err(e) = result {
            tracing::error!("Axum server error: {}", e);
        }
    });

    // Wait for Ctrl+C
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received Ctrl+C, initiating shutdown...");
        }
    }

    let _ = shutdown_tx.send(()); // Signal Axum to shut down
    tracing::info!("Waiting for Axum server to exit...");
    let _ = server_task.await;

    tracing::info!("API Server stopped. Goodbye!");
    Ok(())
}
