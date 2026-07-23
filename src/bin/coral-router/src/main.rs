use std::sync::Arc;

use clap::Parser;
use common_core::config::load_json_or_default;
use fluent_router::config::RouterConfig;
use fluent_router::logging::init_router_logging;
use fluent_router::server::RouterServer;

#[derive(Parser)]
#[command(name = "coral-router", about = "LLM Router & Agent Orchestration Server")]
struct Args {
    /// Path to the router configuration JSON file.
    #[arg(short, long, default_value = "coral-router.json")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load configuration
    let config: RouterConfig = load_json_or_default(args.config.as_ref());

    // Initialize structured logging
    init_router_logging(&config.logging)?;

    // Build the pipeline from config
    let pipeline = Arc::new(config.build_pipeline());

    tracing::info!(
        bind_addr = %config.server.bind_addr,
        "starting coral-router server"
    );

    // Start the HTTP server
    let server = RouterServer::new(pipeline, &config.server);
    server.serve().await?;

    Ok(())
}
