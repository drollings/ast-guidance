use std::collections::HashSet;
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

    /// Comma-separated list of model names that should NOT be mocked.
    /// Only meaningful when --mock is also set. These models make real
    /// LLM calls instead of returning canned dispatch responses.
    #[arg(long, value_delimiter = ',')]
    mock_except: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let config: RouterConfig = load_json_or_default(args.config.as_ref());

    init_router_logging(&config.logging)?;

    let transcript_path = args
        .mock
        .or_else(|| config.mock.as_ref().map(|m| m.transcript_path.clone()));

    let mock_except_models: HashSet<String> = args.mock_except.iter().cloned().collect();

    let (pipelines, mock_dispatch) = if let Some(ref path) = transcript_path {
        tracing::info!(target: "coral-router", transcript = %path, "mock mode enabled");

        let entries = load_transcript_file(path)?;
        let dispatch_ctx = MockDispatchContext::new(
            entries,
            args.mock_except.clone(),
        );

        let classifier_model_name = config.classifier_model.as_deref().unwrap_or("fast");
        let classifier_is_excepted = mock_except_models.contains(classifier_model_name);

        let pipelines = if classifier_is_excepted {
            // Classifier model is in the except list — use real LLM backend
            config.build_all_pipelines_with_backend(None::<&Arc<dyn ChatBackend>>)
        } else {
            let provider =
                transcript_provider_from_entries(dispatch_ctx.transcripts());
            let provider: Arc<dyn ChatBackend> = Arc::new(provider);
            config.build_all_pipelines_with_backend(Some(&provider))
        };

        (pipelines, Some(dispatch_ctx))
    } else {
        if !args.mock_except.is_empty() {
            tracing::warn!(target: "coral-router", "--mock-except has no effect without --mock");
        }
        let pipelines = config.build_all_pipelines();
        (pipelines, None)
    };

    let routes = config.routes.clone();

    let classifier_model_name = config.classifier_model.as_deref().unwrap_or("fast");
    let classifier_url = config
        .models
        .get(classifier_model_name)
        .map(|m| m.endpoint.clone());

    tracing::info!(
        bind_addr = %config.server.bind_addr,
        classifier_url = ?classifier_url,
        classifier_model = %classifier_model_name,
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
