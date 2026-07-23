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

    let config: RouterConfig = load_json_or_default(args.config.as_ref());

    init_router_logging(&config.logging)?;

    let pipeline = Arc::new(config.build_pipeline());

    let frontier_url = config
        .models
        .frontier
        .first()
        .map(|f| {
            f.api_base
                .clone()
                .unwrap_or_else(|| f.provider.clone())
        });

    tracing::info!(
        bind_addr = %config.server.bind_addr,
        frontier_url = ?frontier_url,
        "starting coral-router server"
    );

    let server = RouterServer::new(pipeline, &config.server, frontier_url);
    server.serve().await?;

    Ok(())
}
