use std::collections::HashMap;
use std::sync::Arc;

use clap::Parser;
use common_core::config::load_json_or_default;
use fluent_router::config::RouterConfig;
use fluent_router::logging::init_router_logging;
use fluent_router::pipeline::PipelineOrchestrator;
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

    let pipelines: HashMap<String, Arc<PipelineOrchestrator>> = config.build_all_pipelines();
    let routes = config.routes.clone();

    let classifier_url = config
        .model_groups
        .get("fast")
        .and_then(|names| names.first())
        .and_then(|name| config.models.get(name))
        .map(|m| m.endpoint.clone());

    tracing::info!(
        bind_addr = %config.server.bind_addr,
        classifier_url = ?classifier_url,
        "starting coral-router server"
    );

    let server = RouterServer::new(pipelines, routes, &config.server, classifier_url);
    server.serve().await?;

    Ok(())
}
