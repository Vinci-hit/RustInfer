//! Standalone HTTP API Server
//!
//! 这个进程只负责运行 HTTP API 服务器。
//! 它需要连接到已启动的 Scheduler 进程。
//!
//! Usage:
//!   rustinfer-server --config rustinfer.toml

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_protocol::{RustInferConfig, resolve_model_type};
use infer_server::{AppState, ZmqClient, router::build_router, state::ModelInfo};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-server")]
#[command(about = "RustInfer Standalone HTTP API Server")]
struct ServerArgs {
    /// Path to the shared TOML launch config.
    #[arg(long, default_value = "rustinfer.toml")]
    config: String,

    /// Explicit browser origins allowed to call the API cross-origin.
    /// Repeat the flag or provide a comma-separated environment value.
    /// Omitted by default, which leaves browser access same-origin only.
    #[arg(
        long = "cors-allowed-origin",
        env = "RUSTINFER_CORS_ALLOWED_ORIGINS",
        value_delimiter = ','
    )]
    cors_allowed_origins: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = ServerArgs::parse();
    let config = RustInferConfig::load(&args.config).map_err(|e| anyhow::anyhow!(e))?;

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let model_name = config.effective_model_name();
    let frontend_endpoint = config.frontend_endpoint();

    // model_type drives the chat template; resolve from the model's config.json.
    let model_type = resolve_model_type(&config.model).map_err(|e| anyhow::anyhow!(e))?;

    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║       RustInfer API Server v0.1.0 (Standalone)   ║");
    tracing::info!("╚══════════════════════════════════════════════════╝");
    tracing::info!("  Model: {}", config.model);
    tracing::info!("  Model type: {}", model_type);
    tracing::info!("  API Server Port: {}", config.port);
    tracing::info!("  API Server Host: {}", config.host);
    if args.cors_allowed_origins.is_empty() {
        tracing::info!("  CORS: same-origin only");
    } else {
        tracing::info!("  CORS allowed origins: {:?}", args.cors_allowed_origins);
    }
    tracing::info!("  Frontend Endpoint: {}", frontend_endpoint);

    // Load tokenizer
    tracing::info!("[server] Initializing Tokenizer and ZMQ Client...");
    let tokenizer_path = std::path::Path::new(&config.model).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tracing::info!(
        "Tokenizer loaded (vocab_size={})",
        tokenizer.get_vocab_size(true)
    );

    // Connect to Scheduler
    let client = ZmqClient::new(&frontend_endpoint, config.request_timeout_secs).await?;
    tracing::info!("Connected to scheduler via {}", frontend_endpoint);

    let model_info = ModelInfo {
        model_id: model_name,
        owned_by: "rustinfer".to_string(),
    };

    let port = config.port;
    let host = config.host.clone();

    // Admission gate: bound concurrent in-flight requests so overload sheds at
    // ingress (429) instead of growing internal queues. Read before `config` is
    // moved into AppState. `0` => unlimited.
    let permits = if config.max_inflight_requests == 0 {
        tokio::sync::Semaphore::MAX_PERMITS
    } else {
        config.max_inflight_requests
    };
    let admission = Arc::new(tokio::sync::Semaphore::new(permits));
    tracing::info!("  Max in-flight requests: {}", config.max_inflight_requests);

    let state = Arc::new(AppState {
        client,
        tokenizer: Arc::new(tokenizer),
        config,
        model_type,
        model_info,
        admission,
    });

    let app = build_router(state, &args.cors_allowed_origins)?;
    tracing::info!("API Server listening on http://{}:{}", host, port);

    let listener = bind_listener(&host, port).await?;

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

    // Wait for SIGINT (Ctrl+C) or SIGTERM (systemctl/docker stop). Handling
    // SIGTERM here lets Axum drain gracefully under a supervisor stop instead of
    // being killed by the default disposition.
    wait_for_shutdown_signal().await;
    tracing::info!("Shutdown signal received, initiating shutdown...");

    let _ = shutdown_tx.send(()); // Signal Axum to shut down
    tracing::info!("Waiting for Axum server to exit...");
    let _ = server_task.await;

    tracing::info!("API Server stopped. Goodbye!");
    Ok(())
}

async fn bind_listener(host: &str, port: u16) -> std::io::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind((host, port)).await
}

/// Resolve when the process receives SIGINT (Ctrl-C) or SIGTERM. On non-unix
/// targets only Ctrl-C is observed.
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler ({}); Ctrl-C only", e);
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::bind_listener;

    #[tokio::test]
    async fn listener_honors_configured_host() {
        let listener = bind_listener("127.0.0.1", 0).await.unwrap();
        assert_eq!(listener.local_addr().unwrap().ip().to_string(), "127.0.0.1");
    }
}
