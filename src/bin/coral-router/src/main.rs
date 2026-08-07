use std::collections::HashSet;
use std::sync::Arc;

use clap::Parser;
use common_core::config::load_json_or_default;
use fluent_llm::client::ChatBackend;
use fluent_llm::{create_embedding_provider, EmbeddingProvider};
use fluent_router::charts::store::ChartStore;
use fluent_router::config::{validate_no_self_routing, RouterConfig};
use fluent_router::hnsw::HnswIndexHandle;
use fluent_router::ledger::ContentNodeLedger;
use fluent_router::logging::init_router_logging;
use fluent_router::routes::plan::PlanRoute;
use fluent_router::routes::rigor::RigorRoute;
use fluent_router::server::RouterServer;
use fluent_router::testing::{
    load_transcript_file, transcript_provider_from_entries, MockDispatchContext,
};

#[derive(Parser)]
#[command(
    name = "coral-router",
    about = "LLM Router & Agent Orchestration Server"
)]
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
                None => {
                    return Err(
                        "--host requires --port or a port in config server.bind_addr".into(),
                    )
                }
            }
        }
        (None, Some(port)) => {
            let existing_host = config
                .server
                .bind_addr
                .rsplit(':')
                .next()
                .map(|p| {
                    let host_part =
                        &config.server.bind_addr[..config.server.bind_addr.len() - p.len() - 1];
                    if host_part.is_empty() {
                        "0.0.0.0"
                    } else {
                        host_part
                    }
                })
                .unwrap_or("0.0.0.0");
            format!("{existing_host}:{port}")
        }
        (None, None) => config.server.bind_addr.clone(),
    };
    config.server.bind_addr = bind_addr;

    // Apply mock base URL override
    if let Some(ref url) = args.mock_base_url {
        config
            .mock
            .get_or_insert_with(|| fluent_router::config::MockConfig {
                transcript_path: String::new(),
                fail_on_unexpected: true,
                base_url: url.clone(),
            })
            .base_url = url.clone();
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
        let dispatch_ctx = MockDispatchContext::new(entries, args.mock_except.clone());

        let classifier_model_name = config.classifier_model.as_deref().unwrap_or("fast");
        let classifier_is_excepted = mock_except_models.contains(classifier_model_name);
        tracing::info!(target: "coral-router", classifier_model = %classifier_model_name, classifier_excepted = classifier_is_excepted, "classifier mock decision");

        let pipelines = if classifier_is_excepted {
            tracing::info!(target: "coral-router", "classifier model is excepted — building with real LLM backend");
            config.build_all_pipelines_with_backend(None::<&Arc<dyn ChatBackend>>)
        } else {
            let provider = transcript_provider_from_entries(dispatch_ctx.transcripts());
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

    // M4.4: with a classification tree the `routes` view is derived from the
    // tree's terminal nodes (plus the explicit flat map) so the server's
    // model→pipeline resolution needs no structural change.
    let routes = config.routes_view();

    let classifier_model_name = config.classifier_model.as_deref().unwrap_or("fast");
    let classifier = config
        .models
        .get(classifier_model_name)
        .map(|m| (classifier_model_name.to_string(), m.clone()));

    tracing::info!(
        bind_addr = %config.server.bind_addr,
        classifier_url = ?classifier.as_ref().map(|(_, m)| m.endpoint.clone()),
        classifier_model = %classifier_model_name,
        "starting coral-router server"
    );

    // Chart store boot: load `config.charts.dir` (fail fast on a corrupt
    // file — a half-loaded library must not serve), attach the shared store
    // to the plan route. A missing directory is tolerated (empty store).
    let plan_route = Arc::new(build_plan_route(&config));
    let rigor_route = Arc::new(build_rigor_route(&config));

    // M3 escalation ladders: one per `model_groups[g].escalation` config.
    let http_client = reqwest::Client::new();
    let ladders = config.build_escalation_ladders(&http_client);

    // M4 sidecar: build one instance manager per endpoint that declares an
    // instance pool. The server owns their reconcile + residency tasks. A
    // malformed instance grammar (duplicate name / group-name collision)
    // fails fast so boot aborts loudly.
    let instance_managers = match fluent_router::instances::build_instance_managers(&config) {
        Ok(managers) => managers,
        Err(e) => {
            tracing::error!(
                target: "coral-router",
                error = %e,
                "fatal: instance pool grammar validation failed",
            );
            eprintln!("FATAL: {e}");
            std::process::exit(1);
        }
    };

    // M2 composition: when the operator opts in via the `ledger`/`session`
    // sections, open a `ContentNodeLedger` (with a real `Summarizer` backend
    // targeting `<base>:ledger`) and/or a `SessionRegistry`, and attach both
    // to the server so rigor rewind and ledger LOD derivation exist at runtime.
    // Both are default-absent, so existing deployments are untouched.
    let ledger = if let Some(ledger_cfg) = &config.ledger {
        let opened = match &ledger_cfg.path {
            Some(path) => ContentNodeLedger::open(path),
            None => {
                tracing::warn!(
                    target: "coral-router",
                    "ledger section has no path - using an in-memory ledger (ephemeral)",
                );
                ContentNodeLedger::open_in_memory()
            }
        };
        let ledger = match opened {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(
                    target: "coral-router",
                    error = %e,
                    "fatal: ledger open failed",
                );
                eprintln!("FATAL: ledger open failed: {e}");
                std::process::exit(1);
            }
        };
        match config.summarizer_for_ledger() {
            Some(summarizer) => {
                let model_key = ledger_cfg
                    .model
                    .clone()
                    .or_else(|| config.classifier_model.clone());
                tracing::info!(
                    target: "coral-router",
                    ledger_model = ?model_key,
                    summarizer = true,
                    "ledger summarizer attached",
                );
                Some(Arc::new(ledger.with_summarizer(summarizer)))
            }
            None => {
                tracing::warn!(
                    target: "coral-router",
                    "ledger section present but no summarizer derivable - ledger attached without LOD derivation",
                );
                Some(Arc::new(ledger))
            }
        }
    } else {
        None
    };

    let sessions = if let Some(session_cfg) = &config.session {
        let kv_root = session_cfg.root.as_ref().map(std::path::PathBuf::from);
        let sessions =
            Arc::new(fluent_router::dag_session::SessionRegistry::new(kv_root));
        tracing::info!(
            target: "coral-router",
            session_root = ?session_cfg.root,
            "session registry attached",
        );
        Some(sessions)
    } else {
        None
    };

    let mut server =
        RouterServer::new(pipelines, routes, config.models, &config.server, classifier)
            .with_plan_route(plan_route)
            .with_rigor_route(rigor_route)
            .with_ladders(ladders);

    if let Some(ledger) = ledger {
        server = server.with_ledger(ledger);
    }
    if let Some(sessions) = sessions {
        server = server.with_sessions(sessions);
    }

    if !instance_managers.is_empty() {
        server = server.with_instance_managers(instance_managers);
    }

    if let Some(ctx) = mock_dispatch {
        server = server.with_mock(ctx);
    }

    server.serve().await?;

    Ok(())
}

/// Construct the boot-loaded chart store and attach it to the plan route.
///
/// Semantics (decision D3 — fail fast): a missing chart directory yields an
/// empty store (`ChartStore::load_dir` logs a `warn!`); a present-but-invalid
/// chart file aborts boot so a corrupted library never half-loads.
///
/// M7 retrieval: when `charts.index_path` is configured the `workflow_library`
/// index is built at boot (lazy + failure-tolerant — a down embedding endpoint
/// disables HNSW retrieval but never aborts boot; deterministic match and LLM
/// adjudication still work). The adjudicator backend is wired from
/// `charts.selector_model` when set.
fn build_plan_route(config: &RouterConfig) -> PlanRoute {
    let index_handle = config
        .charts
        .index_path
        .as_deref()
        .map(|path| HnswIndexHandle {
            name: "workflow_library".into(),
            path: path.into(),
        });
    let store = ChartStore::new(index_handle);

    if let Some(ref dir) = config.charts.dir {
        match store.load_dir(std::path::Path::new(dir)) {
            Ok(()) => {}
            Err(e) => {
                tracing::error!(
                    target: "coral-router",
                    chart_dir = %dir,
                    error = %e,
                    "fatal: chart store failed to load",
                );
                eprintln!("FATAL: chart store failed to load: {e}");
                std::process::exit(1);
            }
        }
    }

    let mut names = store.list();
    names.sort_unstable();
    tracing::info!(
        target: "coral-router",
        chart_dir = ?config.charts.dir,
        chart_count = store.len(),
        chart_names = ?names,
        "chart store loaded",
    );

    // Build the workflow_library HNSW index at boot (M7 step 2). Lazy: only
    // when index_path is configured. Failure-tolerant: a missing/unreachable
    // embedding endpoint skips the build with a warning, never aborts boot.
    if config.charts.index_path.is_some() {
        match default_chart_embedder(config) {
            Some(embedder) => match store.build_index(embedder) {
                Ok(()) => tracing::info!(
                    target: "coral-router",
                    index_path = ?config.charts.index_path,
                    "workflow_library index built at boot",
                ),
                Err(e) => tracing::warn!(
                    target: "coral-router",
                    error = %e,
                    "workflow_library index build skipped — HNSW retrieval disabled (degraded)",
                ),
            },
            None => tracing::warn!(
                target: "coral-router",
                "no embedder derivable from model config — HNSW retrieval disabled (degraded)",
            ),
        }
    }

    let store = Arc::new(store);
    let mut route = PlanRoute::new()
        .with_chart_store(store.clone())
        .with_charts_config(config.charts.clone());
    if let Some(backend) = default_adjudicator_backend(config) {
        route = route.with_selector_backend(backend);
    }
    if let Some(backend) = default_reranker_backend(config) {
        route = route.with_reranker_backend(backend);
    }
    // M4 server-side execution: the same charts model runs a selected chart's
    // targets (and doubles as the rubric judge). A shared limiter bounds
    // concurrent chart-target LLM calls. When no charts model is configured
    // the exact fit degrades to a fresh draft (see `PlanRoute::execute_chart`).
    if let Some(backend) = default_adjudicator_backend(config) {
        route = route.with_execution_backend(backend);
    }
    route = route.with_limiter(Arc::new(fluent_concurrency::pool::Limiter::new(
        CHART_EXECUTION_CONCURRENCY,
    )));
    // M10 learning loop: attach the dispatch post-processing hook when the
    // operator opts in (`post_process.workflow_extraction`). Off by default.
    // The two `Arc`s are NOT redundant: `plan_route` is Arc-shared into the
    // HTTP server, while the extractor is separately Arc-wrapped because the
    // same `WorkflowExtractor` instance is handed to the `PlanRoute` *and*
    // cloned out of it by the dispatch post-process path (handler.rs).
    if config.post_process.workflow_extraction {
        let extractor = fluent_router::charts::extract::WorkflowExtractor::new(store)
            .enabled(true)
            .with_extraction_mode(config.post_process.workflow_extraction_mode);
        route = route.with_workflow_extractor(Arc::new(extractor));
        tracing::info!(
            target: "coral-router",
            "workflow extraction enabled — successful dispatches become draft charts",
        );
    }
    route
}

/// Number of embedding dimensions to declare for the chart embedder. The
/// actual vector length is whatever the endpoint returns (the embeddings HTTP
/// client parses the response); this only sets the declared capacity.
const CHART_EMBEDDING_DIMS: u32 = 768;

/// Max concurrent chart-target LLM calls during M4 server-side execution.
const CHART_EXECUTION_CONCURRENCY: usize = 4;

/// Derive an OpenAI-compatible embeddings base URL from a chat-completions
/// endpoint: `http://host:port/v1/chat/completions` → `http://host:port/v1`
/// (the embeddings client appends `/embeddings`).
fn embeddings_base_url(endpoint: &str) -> String {
    fluent_llm::url::derive_embeddings_url(endpoint)
}

/// Build the default chart embedder from the model config, if derivable.
///
/// Uses the root-level `embedding_model` (falling back to the selector model,
/// then the classifier model's) to reach an OpenAI-compatible `/v1/embeddings`.
/// An empty API key is sent — local llama.cpp servers ignore the header.
/// Returns `None` when no model is configured or the URL is not embeddable,
/// leaving HNSW retrieval disabled.
fn default_chart_embedder(config: &RouterConfig) -> Option<Arc<dyn EmbeddingProvider>> {
    let key = config
        .embedding_model
        .as_deref()
        .or(config.charts.selector_model.as_deref())
        .or(config.classifier_model.as_deref())?;
    let entry = config.models.get(key)?;
    let base = embeddings_base_url(&entry.endpoint);
    let boxed = create_embedding_provider(
        "openai",
        entry.name.as_deref(),
        Some(&base),
        Some(""),
        CHART_EMBEDDING_DIMS,
        None,
        entry.params.as_ref(),
    )
    .ok()?;
    Some(Arc::from(boxed))
}

/// Build the chart-selection adjudicator backend from the selector model, if
/// configured. Mirrors `build_classifier_client` (the DIP factory: exactly one
/// place constructs a concrete `LlmClient` for the selector).
fn default_adjudicator_backend(config: &RouterConfig) -> Option<Arc<dyn ChatBackend>> {
    let key = config.charts.selector_model.as_deref()?;
    config.local_backend(key)
}

/// Build the chart-candidate reranker backend (M7 step 2.5) from the
/// root-level `reranker_model`, if configured. Mirrors
/// `default_adjudicator_backend`: exactly one place constructs a concrete
/// `LlmClient` for the reranker. The rerank is a cross-encoder-style LLM call
/// over the HNSW candidates before adjudication (`None` skips the stage).
fn default_reranker_backend(config: &RouterConfig) -> Option<Arc<dyn ChatBackend>> {
    let key = config.reranker_model.as_deref()?;
    config.local_backend(key)
}

/// Build the rigor route (M3) from `config.rigor`, mirroring `build_plan_route`.
///
/// Each role backend is DIP-constructed exactly once from its model key via
/// `default_rigor_backend`. With no `rigor` section (or missing keys), the
/// route is present but unconfigured — requests return an explicit
/// `Unconfigured` error, never a crash (`env/coral-router.json` ships without
/// a `rigor` section).
fn build_rigor_route(config: &RouterConfig) -> RigorRoute {
    let Some(cfg) = &config.rigor else {
        return RigorRoute::new();
    };
    let mut route = RigorRoute::new().with_config(cfg.clone());
    if cfg.kv_cache_enabled {
        route = route.with_kv_cache();
    }
    if let Some(backend) = default_rigor_backend(config, cfg.blue_model.as_deref()) {
        route = route.with_blue_backend(backend);
    }
    if let Some(backend) = default_rigor_backend(config, cfg.red_model.as_deref()) {
        route = route.with_red_backend(backend);
    }
    if let Some(backend) = default_rigor_backend(config, cfg.judge_model.as_deref()) {
        route = route.with_judge_backend(backend);
    }
    tracing::info!(
        target: "coral-router",
        blue_model = ?cfg.blue_model,
        red_model = ?cfg.red_model,
        judge_model = ?cfg.judge_model,
        kv_cache_enabled = cfg.kv_cache_enabled,
        max_passes = cfg.max_passes,
        "rigor route configured",
    );
    route
}

/// Build one rigor role backend from a model key, if derivable. Mirrors
/// `default_adjudicator_backend`: exactly one `LlmClient` construction site
/// for rigor's role backends (DIP).
fn default_rigor_backend(config: &RouterConfig, key: Option<&str>) -> Option<Arc<dyn ChatBackend>> {
    let key = key?;
    config.local_backend(key)
}

/// Load `env/coral-router.json` relative to the crate root (test helper).
#[cfg(test)]
fn load_router_config() -> RouterConfig {
    let config_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../env/coral-router.json"
    );
    let content = std::fs::read_to_string(config_path).unwrap();
    serde_json::from_str(&content).unwrap()
}

#[cfg(test)]
mod config_tests {
    use serde::Deserialize;
    use std::collections::HashMap;

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
        let config_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../env/coral-router.json"
        );
        let content = std::fs::read_to_string(config_path).unwrap();
        let c: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(c.get("server").and_then(|v| v.get("bind_addr")).is_some());
        assert!(c.get("models").is_some());
    }

    #[test]
    fn test_embedding_model_key_derives_embedder() {
        // The env config points `embedding_model` at the `embed` model; the
        // embedder must derive from that key (and build against its endpoint).
        let config = super::load_router_config();
        let embedder = super::default_chart_embedder(&config);
        assert!(
            embedder.is_some(),
            "embedding_model: \"embed\" must yield a working chart embedder"
        );
    }

    #[test]
    fn test_reranker_model_key_derives_backend() {
        let config = super::load_router_config();
        // No reranker_model in the env config today → no backend (stage off).
        assert!(
            super::default_reranker_backend(&config).is_none(),
            "no reranker_model configured → rerank stage disabled"
        );
    }
}
