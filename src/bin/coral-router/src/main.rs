use std::collections::HashSet;
use std::sync::Arc;

use clap::Parser;
use common_core::config::load_json_or_default;
use fluent_router::config::{
    validate_no_self_routing, RouterConfig,
};
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

    /// Override the server bind host (takes priority over config file).
    #[arg(long)]
    host: Option<String>,

    /// Override the server bind port (takes priority over config file).
    #[arg(long)]
    port: Option<u16>,

    /// Override the mock dispatch base URL (takes priority over config file).
    /// Only relevant when --mock is also set.
    #[arg(long)]
    mock_base_url: Option<String>,

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

    let mut config: RouterConfig = load_json_or_default(args.config.as_ref());

    // CLI overrides take priority over config file
    let bind_addr = match (args.host.as_deref(), args.port) {
        (Some(host), Some(port)) => format!("{host}:{port}"),
        (Some(host), None) => {
            // Preserve port from config or use default-implied
            let existing_port = config
                .server
                .bind_addr
                .rsplit(':')
                .next()
                .and_then(|p| p.parse::<u16>().ok());
            match existing_port {
                Some(p) => format!("{host}:{p}"),
                None => return Err("--host requires --port or a port in config server.bind_addr".into()),
            }
        }
        (None, Some(port)) => {
            let existing_host = config
                .server
                .bind_addr
                .rsplit(':')
                .next()
                .map(|p| {
                    let host_part = &config.server.bind_addr[..config.server.bind_addr.len() - p.len() - 1];
                    if host_part.is_empty() { "0.0.0.0" } else { host_part }
                })
                .unwrap_or("0.0.0.0");
            format!("{existing_host}:{port}")
        }
        (None, None) => config.server.bind_addr.clone(),
    };
    config.server.bind_addr = bind_addr;

    // Apply mock base URL override
    if let Some(ref url) = args.mock_base_url {
        config.mock.get_or_insert_with(|| fluent_router::config::MockConfig {
            transcript_path: String::new(),
            fail_on_unexpected: true,
            base_url: url.clone(),
        }).base_url = url.clone();
    }

    // Validate no model endpoint points to the router's own address
    if let Err(e) = validate_no_self_routing(&config.server.bind_addr, &config.models) {
        tracing::error!(target: "coral-router", error = %e, "self-routing validation failed");
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }

    init_router_logging(&config.logging)?;

    let transcript_path = args
        .mock
        .or_else(|| config.mock.as_ref().map(|m| m.transcript_path.clone()));

    let mock_except_models: HashSet<String> = args.mock_except.iter().cloned().collect();

    if !args.mock_except.is_empty() {
        tracing::info!(target: "coral-router", except_models = ?args.mock_except, "mock-except models configured");
    }

    let (pipelines, mock_dispatch) = if let Some(ref path) = transcript_path {
        tracing::info!(target: "coral-router", transcript = %path, mock_except = ?args.mock_except, "mock mode enabled");

        let entries = load_transcript_file(path)?;
        let dispatch_ctx = MockDispatchContext::new(
            entries,
            args.mock_except.clone(),
        );

        let classifier_model_name = config.classifier_model.as_deref().unwrap_or("fast");
        let classifier_is_excepted = mock_except_models.contains(classifier_model_name);
        tracing::info!(target: "coral-router", classifier_model = %classifier_model_name, classifier_excepted = classifier_is_excepted, "classifier mock decision");

        let pipelines = if classifier_is_excepted {
            tracing::info!(target: "coral-router", "classifier model is excepted — building with real LLM backend");
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
        config.models,
        &config.server,
        classifier_url,
    );

    if let Some(ctx) = mock_dispatch {
        server = server.with_mock(ctx);
    }

    server.serve().await?;

    Ok(())
}

#[cfg(test)]
mod config_tests {
    use std::collections::HashMap;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct TestModelEntry {
        pub endpoint: String,
        #[serde(default)]
        pub name: Option<String>,
        pub intelligence: u8,
        pub cost_input: f64,
        pub cost_output: f64,
        pub cost_cached_read: f64,
        pub speed: u8,
        #[serde(default)]
        pub total_timeout_ms: u64,
        #[serde(default)]
        pub idle_timeout_ms: u64,
        #[serde(default)]
        pub stream: bool,
        #[serde(default)]
        pub filter_thinking: bool,
        #[serde(default)]
        pub retry_count: u32,
        #[serde(default)]
        pub retry_base_interval_s: u64,
        #[serde(default)]
        pub params: Option<serde_json::Value>,
        #[serde(default)]
        pub sessions: Option<HashMap<String, serde_json::Value>>,
    }

    #[test]
    fn test_parse_config() {
        let content = std::fs::read_to_string("env/coral-router.json").unwrap();
        let c: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(c.get("server").and_then(|v| v.get("bind_addr")).is_some());
        assert!(c.get("models").is_some());
    }
}
