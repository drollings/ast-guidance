use std::sync::Arc;

use clap::Parser;
use common_core::config::load_json_or_default;
use fluent_router::config::RouterConfig;
use fluent_router::logging::init_router_logging;
use fluent_router::server::RouterServer;
use fluent_router::testing::{
    load_transcript_file, transcript_provider_from_entries, MockDispatchContext,
};
use guidance_llm::client::ChatBackend;

#[derive(Parser)]
#[command(name = "coral-router", about = "LLM Router & Agent Orchestration Server")]
struct Args {
    /// Path to the router configuration JSON file.
    #[arg(short, long, default_value = "coral-router.json")]
    config: String,

    /// Run in mock mode with a transcript file (bypasses the mock.transcript_path
    /// in config if set).
    #[arg(long)]
    mock: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let config: RouterConfig = load_json_or_default(args.config.as_ref());

    init_router_logging(&config.logging)?;

    let transcript_path = args
        .mock
        .or_else(|| config.mock.as_ref().map(|m| m.transcript_path.clone()));

    let (pipelines, mock_dispatch) = if let Some(ref path) = transcript_path {
        tracing::info!(target: "coral-router", transcript = %path, "mock mode enabled");

        let entries = load_transcript_file(path)?;
        let provider = transcript_provider_from_entries(&entries);
        let provider: Arc<dyn ChatBackend> = Arc::new(provider);
        let dispatch_ctx = MockDispatchContext::new(entries);

        let pipelines = config.build_all_pipelines_with_backend(Some(&provider));
        (pipelines, Some(dispatch_ctx))
    } else {
        let pipelines = config.build_all_pipelines();
        (pipelines, None)
    };

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

    let mut server = RouterServer::new(
        pipelines,
        routes,
        &config.server,
        classifier_url,
    );

    if let Some(ctx) = mock_dispatch {
        server = server.with_mock(ctx);
    }

    server.serve().await?;

    Ok(())
}
