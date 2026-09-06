//! coral-router — LLM Router & Agent Orchestration Server and CLI administration
//! tool.
//!
//! `coral-router start` runs the router server (the process owner of the local
//! `llama-server` fleet). The remaining subcommands (`list`, `ps`, `pull`,
//! `scan`, `rm`, `show`, `stop`, `speedtest`) are the CLI administration
//! surface, ported from `gguf_tool.py`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

mod boot;

use clap::{Args, Parser, Subcommand};
use common_core::config::load_json_or_default;
use fluent_llm::client::ChatBackend;
use fluent_llm::protocol::ChatMessage;
use fluent_llm::{create_embedding_provider, EmbeddingProvider};
use fluent_router::charts::store::ChartStore;
use fluent_router::cli::{commands, CliContext};
use fluent_router::config::builder::NlpDeps;
use fluent_router::config::{validate_no_self_routing, RouterConfig};
use fluent_router::concept_store_sqlite::SqliteConceptStore;
use fluent_db::hnsw::HnswIndexHandle;
use fluent_router::ledger::correction_index::SqliteCorrectionIndex;
use fluent_router::ledger::ContentNodeLedger;
use fluent_router::logging::init_router_logging;
use fluent_router::routes::plan::PlanRoute;
use fluent_router::routes::rigor::RigorRoute;
use fluent_router::server::RouterServer;
use fluent_router::server::review::{ReviewFetch, ReviewWorker};
use fluent_router::server::entity_link::{EntityLinkWorker, EntityLinkScorer};
use fluent_router::testing::{
    load_transcript_file, transcript_provider_from_entries, MockDispatchContext,
};
use fluent_concept::ConceptStore;

#[derive(Parser)]
#[command(
    name = "coral-router",
    about = "LLM Router & Agent Orchestration Server + CLI administration tool",
    version,
    subcommand_required = true
)]
struct Cli {
    /// Path to the router configuration JSON file.
    #[arg(short, long, global = true, default_value = "coral-router.json")]
    config: String,

    /// GGUF directory scanned by the admin subcommands (default:
    /// /app/ai/models/gguf).
    #[arg(long, global = true)]
    gguf_dir: Option<PathBuf>,

    /// Show what would be done without making changes.
    #[arg(short = 'n', long, global = true)]
    dry_run: bool,

    /// Enable verbose logging.
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable debug mode.
    #[arg(long, global = true)]
    debug: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the router server (spawns and supervises the managed llama-servers).
    Start(StartArgs),
    /// List models in the GGUF directory.
    #[command(alias = "ls")]
    List,
    /// List running models via the router's /v1/models and /instances API.
    Ps(ServerArgs),
    /// Pull a model from a registry (HuggingFace) or a local GGUF file.
    Pull(PullArgs),
    /// Scan the GGUF directory, generate configs and the llama.cpp preset.
    Scan(ScanArgs),
    /// Remove a model.
    Rm(RmArgs),
    /// Show information for a model.
    Show(ShowArgs),
    /// Stop a running model.
    Stop(StopArgs),
    /// Measure generation throughput via /metrics.
    Speedtest(SpeedtestArgs),
}

#[derive(Args)]
struct StartArgs {
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

    /// Synthesize a default ledger config when none is configured (M4).
    /// Opt-in via env `CORAL_LEDGER_DEFAULT=1` or this flag; keeps wire default `None`.
    #[arg(long)]
    ledger_default: bool,
}

/// Shared server-address args for the router-API commands.
#[derive(Args)]
struct ServerArgs {
    /// Router base URL (default: derived from config server.bind_addr).
    #[arg(short = 'u', long)]
    api_url: Option<String>,
}

#[derive(Args)]
struct PullArgs {
    /// Model name (e.g. hf.co/author/model:tag).
    model: String,
    /// Local GGUF file to use instead of downloading.
    #[arg(short, long)]
    input: Option<PathBuf>,
    /// Overwrite existing destination.
    #[arg(short, long)]
    force: bool,
}

#[derive(Args)]
struct ScanArgs {
    /// Write LiteLLM YAML config to path.
    #[arg(short = 'L', long)]
    write_litellm: Option<PathBuf>,
    /// Write aichat config to path.
    #[arg(short = 'A', long)]
    write_aichat: Option<PathBuf>,
    /// Prefix for model paths in the preset (e.g. /app/ai/models/gguf;
    /// default: absolute host paths).
    #[arg(long)]
    path_prefix: Option<String>,
    /// Output models as JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args)]
struct RmArgs {
    /// Model name.
    model: String,
}

#[derive(Args)]
struct ShowArgs {
    /// Model name.
    model: String,
    /// Show the Modelfile.
    #[arg(long)]
    modelfile: bool,
    /// Show the license.
    #[arg(long)]
    license: bool,
    /// Show parameters.
    #[arg(long)]
    parameters: bool,
    /// Show the system message.
    #[arg(long)]
    system: bool,
    /// Show the chat template.
    #[arg(long)]
    template: bool,
}

#[derive(Args)]
struct StopArgs {
    /// Model name (router model key, or a GGUF-layout name).
    model: String,
    /// Force the child process to exit (reserved; the router owns the server).
    #[arg(short, long)]
    force: bool,
    /// Router base URL (default: derived from config server.bind_addr).
    #[arg(short = 'u', long)]
    api_url: Option<String>,
}

#[derive(Args)]
struct SpeedtestArgs {
    /// Model to benchmark (default: first configured model key).
    #[arg(short, long)]
    model: Option<String>,
    /// Number of tokens to generate; 0 reports previous performance only.
    #[arg(short, long, default_value_t = 0)]
    tokens: u32,
    /// Prompt to send.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Router base URL (default: derived from config server.bind_addr).
    #[arg(short = 'u', long)]
    api_url: Option<String>,
    /// Sampling temperature.
    #[arg(short = 'T', long, default_value_t = 0.9)]
    temperature: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config_path = resolve_config_path(&cli.config);
    // Best-effort config load so CLI defaults (e.g. the GGUF dir) come from
    // the config file instead of hardcoded paths. An explicit `--gguf-dir`
    // still wins.
    let cli_config: RouterConfig = load_json_or_default(std::path::Path::new(&config_path));
    let gguf_dir = cli
        .gguf_dir
        .clone()
        .or_else(|| cli_config.gguf_dir.as_ref().map(PathBuf::from));
    let ctx = CliContext::new(gguf_dir, cli.dry_run, cli.verbose, cli.debug);

    match cli.command {
        Command::Start(args) => run_start(&config_path, args).await?,
        Command::List => commands::list(&ctx)?,
        Command::Ps(args) => {
            commands::ps(
                &ctx,
                args.api_url.as_deref(),
                Some(std::path::Path::new(&config_path)),
            )
            .await?
        }
        Command::Pull(args) => commands::pull(&ctx, &args.model, args.input, args.force).await?,
        Command::Scan(args) => commands::scan(
            &ctx,
            args.write_litellm.as_ref(),
            args.write_aichat.as_ref(),
            args.path_prefix.as_deref(),
            args.json,
        )?,
        Command::Rm(args) => commands::rm(&ctx, &args.model)?,
        Command::Show(args) => {
            let flags = fluent_router::cli::commands::ShowFlags {
                modelfile: args.modelfile,
                license: args.license,
                parameters: args.parameters,
                system: args.system,
                template: args.template,
            };
            commands::show(&ctx, &args.model, &flags)?
        }
        Command::Stop(args) => {
            commands::stop(
                &ctx,
                args.api_url.as_deref(),
                Some(std::path::Path::new(&config_path)),
                &args.model,
                args.force,
            )
            .await?
        }
        Command::Speedtest(args) => {
            let st = fluent_router::cli::commands::SpeedtestArgs {
                model: args.model.unwrap_or_default(),
                tokens: args.tokens,
                prompt: args.prompt,
                temperature: args.temperature,
            };
            commands::speedtest(
                &ctx,
                args.api_url.as_deref(),
                Some(std::path::Path::new(&config_path)),
                &st,
            )
            .await?
        }
    }
    Ok(())
}

/// Resolve the config path: the explicit value, or the repository default
/// (`env/coral-router.json`) when the default path does not exist.
fn resolve_config_path(explicit: &str) -> String {
    if std::path::Path::new(explicit).exists() || explicit != "coral-router.json" {
        return explicit.to_string();
    }
    if std::path::Path::new("env/coral-router.json").exists() {
        "env/coral-router.json".to_string()
    } else {
        explicit.to_string()
    }
}

/// Resolve the classifier model key for logging/attribution, entirely from
/// config (never a hardcoded name): the root `classifier_model`, else the
/// first pipeline's `classifier_model`, else the default route's first model,
/// else the first configured model key. Empty when nothing resolves.
fn resolve_classifier_model_name(config: &RouterConfig) -> String {
    if let Some(m) = &config.classifier_model {
        return m.clone();
    }
    for params in config.pipelines.values() {
        if let Some(m) = &params.classifier_model {
            return m.clone();
        }
    }
    if let Some(route) = config.routes_view().get(&config.default_route) {
        if let Some(group) = config.model_groups.get(&route.group) {
            if let Some(first) = group.models().first() {
                return first.clone();
            }
        }
    }
    let mut keys: Vec<&String> = config.models.keys().collect();
    keys.sort_unstable();
    keys.first().map_or_else(String::new, |k| k.to_string())
}

/// Start the router server: build config, boot the llama-server supervisor,
/// attach the pipeline/server, and serve until a shutdown signal.
async fn run_start(config_path: &str, args: StartArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut config: RouterConfig = load_json_or_default(config_path.as_ref());
    config.apply_defaults();
    // M7: flat vs tree coherence — fail fast on drift
    if let Err(e) = config.validate_flat_tree_coherence() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }

    // M4: wire default stays None, but CI/operator opt-in via flag/env synthesizes a ledger at the composition root.
    let ledger_default_env = std::env::var("CORAL_LEDGER_DEFAULT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let is_mock = args.mock.is_some() || config.mock.is_some();
    if config.ledger.is_none() && !is_mock && (ledger_default_env || args.ledger_default) {
        tracing::info!(target: "coral-router", path = "/tmp/coral-ledger.db", "CORAL_LEDGER_DEFAULT/--ledger-default synthesized ledger config");
        config.ledger = Some(fluent_router::config::LedgerConfig {
            path: Some("/tmp/coral-ledger.db".into()),
            background_tiering: false,
            ..Default::default()
        });
    }

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

    // Managed models (weights/hf_repo/instances declared) get their own
    // spawned `llama-server`: the supervisor assigns each a free localhost
    // port, spawns the process, and waits for /health BEFORE any backend or
    // pipeline is built, so classifiers, dispatch, and the sidecar all talk to
    // a live server. Each managed model's `endpoint` is then rewritten to its
    // server's address.
    //
    // In mock mode the whole point is canned dispatch - no real model is
    // needed - so the supervisor is skipped (the config endpoints stay as-is).
    // The `_supervisor` binding is deliberately kept alive for the life of the
    // process so the spawned servers are not dropped on shutdown.
    //
    // `build` resolves the slot-save dir with a capability-gated
    // `create_dir_all`, so it runs with the `FsCapability` grant installed in
    // the current task-local. This is the serving-path grant: boot is the
    // boundary that establishes the router's filesystem authority.
    let _supervisor: Option<Arc<fluent_router::supervisor::LlamaServerSupervisor>> =
        if transcript_path.is_some() {
            tracing::info!(
                target: "coral-router",
                "mock mode - skipping managed llama-server supervision",
            );
            None
        } else {
            let supervisor = fluent_concurrency::scope::CURRENT_CAPS.sync_scope(
                fluent_concurrency::capability::default_capability_set(),
                || fluent_router::supervisor::LlamaServerSupervisor::build(&config),
            )?;
            let supervisor = Arc::new(supervisor);
            if let Err(e) = supervisor.start_all().await {
                tracing::error!(target: "coral-router", error = %e, "fatal: managed llama-server failed to start");
                eprintln!("FATAL: {e}");
                std::process::exit(1);
            }
            for key in supervisor.model_keys() {
                if let Some(server) = supervisor.server_for(&key) {
                    if let Some(entry) = config.models.get_mut(&key) {
                        entry.endpoint = format!("{}/v1/chat/completions", server.base_url());
                        tracing::info!(
                            target: "coral-router",
                            model = %key,
                            endpoint = %entry.endpoint,
                            "model endpoint rewritten to managed llama-server",
                        );
                    }
                }
            }
            Some(supervisor)
        };

    let mock_except_models: HashSet<String> = args.mock_except.iter().cloned().collect();

    if !args.mock_except.is_empty() {
        tracing::info!(target: "coral-router", except_models = ?args.mock_except, "mock-except models configured");
    }

    // ONNX registry (ROADMAP_20260827_ORT §0.5): one session per onnx-declared
    // model. `Always`-resident models load here (a missing model file is a
    // loud, actionable boot error); `Unloadable` models load on first use.
    // Absent onnx config yields `None` (fully fail-open). The llama.cpp
    // supervisor, sidecar, and `/instances` never touch onnx models.
    let onnx_registry = match fluent_router::ort::build_onnx_registry(&config) {
        Ok(registry) => registry,
        Err(e) => {
            tracing::error!(target: "coral-router", error = %e, "fatal: onnx registry build failed");
            eprintln!("FATAL: onnx registry build failed: {e}");
            std::process::exit(1);
        }
    };
    if let Some(registry) = onnx_registry.as_ref() {
        tracing::info!(
            target: "coral-router",
            onnx_models = ?registry.model_keys(),
            "onnx registry built",
        );
    }
    // Install the onnx `ChatBackend` resolver (ROADMAP M2.3/M6): the generative
    // LLM role's backend, served behind the single `RouterConfig::local_backend`
    // factory. The `(key, instance)` shape lets the resolver build a
    // **context-bound** backend for a named context (created on demand — the
    // onnx lazy-residency load point) on first use. Absent / unregistered /
    // non-CausalLm → `None` (fail-open to the HTTP/deterministic path). The
    // role's default (`instance = None`) is the single-shot backend unless it
    // declares an `instances` block, in which case it binds to the pool context
    // (the `pool_qualifier` rule) — byte-identical to M0 for the shipped config.
    let onnx_llm_backend = {
        let onnx_llm_key = fluent_onnx::OnnxRole::Llm.registry_key();
        let role = config.onnx.as_ref().and_then(|f| f.llm.as_ref());
        let onnx_llm_backend = onnx_registry
            .as_ref()
            .and_then(|reg| {
                fluent_router::ort::onnx_chat_backend(reg, onnx_llm_key).ok().flatten()
            });
        // Context-bound backends need the onnx-gated `OnnxWeights` type, so
        // they only exist in an `onnx` build. Ort-free builds resolve the bare
        // generative key (always `None` there — the fallback above — fail-open)
        // and never serve a named onnx context.
        #[cfg(feature = "onnx")]
        let llm_weights = match (onnx_registry.as_ref(), role) {
            (Some(reg), Some(role)) => Some(Arc::new(
                fluent_router::ort::OnnxWeights::new(
                    onnx_llm_key.to_string(),
                    reg.clone(),
                    role.clone(),
                ),
            )),
            _ => None,
        };
        let has_instances = role
            .is_some_and(|r| r.instances.as_ref().is_some_and(|m| !m.is_empty()));
        let pool_context = role.and_then(fluent_router::ort::onnx_pool_context);
        if onnx_llm_backend.is_some() {
            tracing::info!(
                target: "coral-router",
                model = onnx_llm_key,
                has_instances = has_instances,
                "onnx generative backend wired as the default local backend",
            );
        }
        let resolver_backend = onnx_llm_backend.clone();
        config.install_onnx_resolver(move |key, instance| {
            if key != onnx_llm_key {
                return None;
            }
            match instance {
                #[cfg(feature = "onnx")]
                Some(name) => llm_weights.as_ref().and_then(|w| {
                    fluent_router::ort::onnx_context_backend(w, name).ok().flatten()
                }),
                #[cfg(not(feature = "onnx"))]
                Some(_) => None,
                None if has_instances => {
                    let ctx = pool_context.clone()?;
                    #[cfg(feature = "onnx")]
                    {
                        llm_weights.as_ref().and_then(|w| {
                            fluent_router::ort::onnx_context_backend(w, &ctx).ok().flatten()
                        })
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        let _ = ctx;
                        None
                    }
                }
                None => resolver_backend.clone(),
            }
        });
        onnx_llm_backend
    };
    // The shared residency engine (M5): ONE loop over the fleet's weights
    // (llama adapters + onnx implementors), replacing both the llama sidecar
    // task and the onnx residency sibling. Built here so the `sidecar` knobs
    // are read before `config` is partially moved into the server. The VRAM
    // budget is detected inside an `FsCapability` scope (ROCm sysfs), exactly
    // as the llama loop's boot did.
    let residency_engine = fluent_concurrency::scope::CURRENT_CAPS.sync_scope(
            fluent_concurrency::capability::default_capability_set(),
            || {
                fluent_llm::runtime::LlmResidencyEngine::new(
                    std::time::Duration::from_secs(config.sidecar.poll_interval_s.max(1)),
                    config.sidecar.allocation_limit(),
                    config.sidecar.onnx_working_set_budget_bytes,
                    fluent_onnx::residency::DEFAULT_SLEEP_IDLE_SECONDS,
                    config.sidecar.evict_batch,
                )
            },
    );

    // ROADMAP_20260828_ORT M1.1: when any pipeline needs interlingua ids
    // (`nlp: true`) or a durable concept store (`review` configured,
    // `overlay.entity_link_enabled`), open the ledger and build the shared
    // `SqliteConceptStore` + `SqliteCorrectionIndex` + YaGO reconcile BEFORE
    // the pipeline build, so a resolver can be threaded into the NLP pipeline
    // (G3 — pipelines were previously built before the store existed). When
    // nothing needs the store the boot order is byte-identical to today
    // (fail-open; `NlpDeps` stays default).
    let nlp_enabled = config.pipelines.values().any(|p| p.nlp);
    let needs_concept_store = nlp_enabled
        || config.review.is_some()
        || config.overlay.as_ref().is_some_and(|o| o.entity_link_enabled)
        || config
            .overlay
            .as_ref()
            .is_some_and(|o| o.arc_ready.as_ref().is_some_and(|a| a.enabled && a.nlp));
    let mut ledger: Option<Arc<ContentNodeLedger>> = None;
    let mut concept_store: Option<Arc<SqliteConceptStore>> = None;
    let mut correction_index: Option<Arc<SqliteCorrectionIndex>> = None;
    let mut nlp_deps = NlpDeps::default();
    if needs_concept_store {
        let opened = match boot::open_ledger(&config) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(target: "coral-router", error = %e, "fatal: ledger open failed");
                eprintln!("FATAL: {e}");
                std::process::exit(1);
            }
        };
        let store_boot = match boot::build_concept_store_boot(&opened) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(target: "coral-router", error = %e, "fatal: boot reconciliation failed");
                eprintln!("FATAL: boot reconciliation failed: {e}");
                std::process::exit(1);
            }
        };
        ledger = Some(Arc::clone(&opened));
        concept_store = Some(Arc::clone(&store_boot.concept_store));
        correction_index = Some(Arc::clone(&store_boot.correction_index));
        nlp_deps = NlpDeps {
            concept_store: Some(Arc::clone(&store_boot.concept_store) as Arc<dyn ConceptStore>),
            strings_path: None,
        };
        tracing::info!(
            target: "coral-router",
            classes = store_boot.stats.classes,
            coral_nodes = store_boot.stats.coral_nodes,
            router_concepts = store_boot.stats.router_concepts,
            "boot concept store + reconcile complete (before pipeline build)",
        );
    }

    let (pipelines, mock_dispatch) = if let Some(ref path) = transcript_path {
        tracing::info!(target: "coral-router", transcript = %path, mock_except = ?args.mock_except, "mock mode enabled");

        let entries = load_transcript_file(path)?;
        let dispatch_ctx = MockDispatchContext::new(entries, args.mock_except.clone());

        let classifier_model_name = resolve_classifier_model_name(&config);
        let classifier_is_excepted = mock_except_models.contains(&classifier_model_name);
        tracing::info!(target: "coral-router", classifier_model = %classifier_model_name, classifier_excepted = classifier_is_excepted, "classifier mock decision");

        let pipelines = if classifier_is_excepted {
            tracing::info!(target: "coral-router", "classifier model is excepted — building with real LLM backend");
            config.build_all_pipelines_with_backend_onnx_and_nlp(
                None::<&Arc<dyn ChatBackend>>,
                onnx_registry.as_ref(),
                &nlp_deps,
            )
        } else {
            let provider = transcript_provider_from_entries(dispatch_ctx.transcripts());
            let provider: Arc<dyn ChatBackend> = Arc::new(provider);
            config.build_all_pipelines_with_backend_onnx_and_nlp(
                Some(&provider),
                onnx_registry.as_ref(),
                &nlp_deps,
            )
        };

        (pipelines, Some(dispatch_ctx))
    } else {
        if !args.mock_except.is_empty() {
            tracing::warn!(target: "coral-router", "--mock-except has no effect without --mock");
        }
        let pipelines = config.build_all_pipelines_with_backend_onnx_and_nlp(
            None::<&Arc<dyn ChatBackend>>,
            onnx_registry.as_ref(),
            &nlp_deps,
        );
        (pipelines, None)
    };

    // With a classification tree the `routes` view is derived from the
    // tree's terminal nodes (plus the explicit flat map) so the server's
    // model→pipeline resolution needs no structural change.
    let routes = config.routes_view();

    let classifier_model_name = resolve_classifier_model_name(&config);
    let classifier = config
        .models
        .get(&classifier_model_name)
        .map(|m| (classifier_model_name.clone(), m.clone()));

    tracing::info!(
        bind_addr = %config.server.bind_addr,
        classifier_url = ?classifier.as_ref().map(|(_, m)| m.endpoint.clone()),
        classifier_model = %classifier_model_name,
        "starting coral-router server"
    );

    // When the operator opts in via the `ledger`/`session` sections, open a
    // `ContentNodeLedger` (with a real `Summarizer` backend targeting
    // `<base>:ledger`) and/or a `SessionRegistry`, and attach both to the
    // server so rigor rewind and ledger LOD derivation exist at runtime. 
    // Both are default-absent, so existing deployments are untouched.
    //
    // The ledger may already have been opened by the M1.1 early step (when
    // nlp/review/overlay need a concept store); reuse it rather than open a
    // second connection (DRY — one shared ledger/store per boot).
    if ledger.is_none() && config.ledger.is_some() {
        match boot::open_ledger(&config) {
            Ok(l) => ledger = Some(l),
            Err(e) => {
                tracing::error!(target: "coral-router", error = %e, "fatal: ledger open failed");
                eprintln!("FATAL: {e}");
                std::process::exit(1);
            }
        }
    }

    let sessions = if let Some(session_cfg) = &config.session {
        let kv_root = session_cfg.root.as_ref().map(std::path::PathBuf::from);
        let sessions = Arc::new(fluent_router::dag_session::SessionRegistry::new(kv_root));
        tracing::info!(
            target: "coral-router",
            session_root = ?session_cfg.root,
            "session registry attached",
        );
        Some(sessions)
    } else {
        None
    };

    // The shared ledger store (when a ledger is attached) is threaded
    // into the plan/rigor route builders so their selector/judge models render
    // the session ledger through the assembler's budget/relevance rules.
    let ledger_store = ledger.as_ref().map(|l| l.node_store().clone());

    // Chart store boot: load `config.charts.dir` (fail fast on a corrupt
    // file — a half-loaded library must not serve), attach the shared store
    // to the plan route. A missing directory is tolerated (empty store).
    let plan_route = Arc::new(build_plan_route(
        &config,
        ledger_store.as_ref(),
        onnx_registry.as_ref(),
    ));
    let rigor_route = Arc::new(build_rigor_route(&config, ledger_store.as_ref()));

    // Escalation ladders: one per `model_groups[g].escalation` config.
    let http_client = reqwest::Client::new();
    let ladders = config.build_escalation_ladders(&http_client);

    // Sidecar: build one instance manager per endpoint that declares an
    // instance pool. The server owns their reconcile + residency tasks. A
    // malformed instance grammar (duplicate name / group-name collision)
    // fails fast so boot aborts loudly.
    //
    // manager build stats each managed model's weights file with a
    // capability-gated `metadata`, so it runs with `FsCapability` installed.
    let instance_pool = match fluent_concurrency::scope::CURRENT_CAPS.sync_scope(
        fluent_concurrency::capability::default_capability_set(),
        || fluent_router::instances::build_instance_managers(&config, _supervisor.clone()),
    ) {
        Ok(pool) => pool,
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

    // The unified weights facade (ROADMAP M4): llama adapters + onnx
    // implementors behind the shared `LlmWeights` surface. Backs the
    // `/instances` + `/v1/models` aggregation (onnx rows), `ps`, and the
    // onnx branch of `POST /models/unload`. The llama rows stay byte-identical
    // (delegated to the llama-only `InstancePool`); onnx rows appear only
    // when the onnx fleet is configured.
    let fleet = Arc::new(fluent_router::instances::traits::LlmFleet::build(
        instance_pool.clone(),
        _supervisor.as_deref(),
        onnx_registry.clone(),
        &config,
    ));

    // Background tiering: when the operator opts in via
    // `ledger.background_tiering`, attach a `LedgerTierWorker` to the shared
    // store so LOD4 (short summary) and LOD5 (LLM description) are derived
    // continuously in the background. Reuses the single `LlmClient` factory
    // (`ledger_tier_backend`) — no second HTTP client. Held on the server for
    // the process lifetime.
    let mut tier_worker_handle: Option<tokio::task::JoinHandle<()>> = None;
    let mut tier_worker: Option<Arc<fluent_router::ledger::tiering::LedgerTierWorker>> = None;
    if let (Some(ledger_arc), Some(ledger_cfg)) = (&ledger, &config.ledger) {
        if ledger_cfg.background_tiering {
            match config.ledger_tier_backend(ledger_cfg.tier_model.as_deref()) {
                Some(backend) => {
                    let tier_cfg = config
                        .ledger_tier_config()
                        .expect("ledger section present -> tier config");
                    let store = Arc::clone(ledger_arc.node_store());
                    let worker = fluent_router::ledger::tiering::LedgerTierWorker::new(
                        Arc::clone(&store),
                        backend,
                        vec![4, 5],
                        tier_cfg,
                        fluent_concurrency::tokio_runtime(),
                    );
                    store.set_tier_events(worker.sender());
                    let handle = worker.start();
                    tracing::info!(
                        target: "coral-router",
                        lod4_max_chars = ledger_cfg.lod4_max_chars,
                        lod5_max_chars = ledger_cfg.lod5_max_chars,
                        "background ledger tiering enabled",
                    );
                    tier_worker_handle = Some(handle);
                    tier_worker = Some(Arc::clone(&worker));
                }
                None => {
                    tracing::warn!(
                        target: "coral-router",
                        tier_model = ?ledger_cfg.tier_model,
                        "ledger.background_tiering set but no tier backend derivable - tiering skipped",
                    );
                }
            }
        }
    }

    // The `LedgerAgentCoordinator` (the ledger-as-synchronization point). 
    // Opt-in via `ledger.orchestrator.enabled`; requires a ledger, a
    // session registry, and a tier worker.  Reuses the single `LlmClient`
    // factory (`ledger_tier_backend`) — no second HTTP client.
    let mut coordinator: Option<Arc<fluent_router::ledger::orchestrator::LedgerAgentCoordinator>> =
        None;
    if let (Some(ledger_arc), Some(ledger_cfg), Some(sessions_arc)) =
        (&ledger, &config.ledger, &sessions)
    {
        if ledger_cfg.orchestrator.enabled {
            let backend = config.ledger_tier_backend(ledger_cfg.tier_model.as_deref());
            if let (Some(tier_worker), Some(backend)) = (&tier_worker, backend) {
                let kv = sessions_arc.kv_cache().clone();
                if let Some(coord) = config.build_ledger_coordinator(
                    Arc::clone(ledger_arc.node_store()),
                    Arc::clone(sessions_arc),
                    kv,
                    Arc::clone(tier_worker),
                    backend,
                ) {
                    tracing::info!(
                        target: "coral-router",
                        kv_policy = ?ledger_cfg.orchestrator.kv_policy,
                        prompt_budget_chars = ledger_cfg.orchestrator.prompt_budget_chars,
                        role = %ledger_cfg.orchestrator.role,
                        "ledger-agent coordinator enabled",
                    );
                    coordinator = Some(Arc::new(coord));
                }
            } else {
                tracing::warn!(
                    target: "coral-router",
                    "ledger.orchestrator.enabled set but no ledger/tier backend derivable - coordinator skipped",
                );
            }
        }
    }

    // Boot reconciliation: load YaGO taxonomy into both coral's content-addressed
    // graph and the router's SqliteConceptStore, then assert they match (ROADMAP
    // §11.10/§13.10 — C3). Requires both a ledger (for the shared SQLite
    // connection) and a review config (which implies the concept store is needed).
    let mut review_worker: Option<Arc<ReviewWorker>> = None;
    let mut review_fetch: Option<ReviewFetch> = None;
    if let (Some(ledger_arc), Some(review_cfg)) = (&ledger, &config.review) {
        // Build the review model backend (ChatBackend) from the review_model
        // key, else the classifier key, else the onnx LLM key (ROADMAP M2.6 —
        // the generative onnx model is the default review backend).
        let review_model_key = review_cfg
            .review_model
            .clone()
            .or_else(|| config.classifier_model.clone())
            .or_else(|| config.onnx_llm_key());
        let review_backend: Option<Arc<dyn ChatBackend>> = review_model_key
            .as_deref()
            .and_then(|key| config.local_backend(key));

        // Reuse the M1.1 early-built shared store (ROADMAP_20260828_ORT M1.1):
        // review configured ⇒ `needs_concept_store` ⇒ the `SqliteConceptStore`
        // + `SqliteCorrectionIndex` were built and YaGO reconciled **before**
        // the pipeline build. One shared store, not one per consumer (DRY).
        let sqlite_store = concept_store
            .clone()
            .expect("review configured => concept store built before pipeline build");
        let correction_index = correction_index
            .clone()
            .expect("review configured => correction index built before pipeline build");
        let concept_store_arc = sqlite_store;
        let fetch: ReviewFetch = if let Some(backend) = review_backend {
            Arc::new(move |prompt: String| {
                let messages = vec![ChatMessage {
                    role: "user".into(),
                    content: prompt,
                }];
                // ChatBackend::chat_complete_with_extras is synchronous. The
                // ParseReview schema travels as `response_format.schema`
                // (ROADMAP M2.5) so a grammar-constrained backend (the onnx
                // LLM) prevents structurally-invalid review output; a backend
                // without the extras seam ignores it (post-hoc serde parsing
                // of `ParseReview` remains the backstop).
                let extras = serde_json::json!({
                    "response_format": {
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "corrections": {"type": "array", "items": {"type": "object"}},
                                "linked_entities": {"type": "array", "items": {"type": "object"}},
                                "note": {"type": "string"}
                            },
                            "required": ["corrections"]
                        }
                    }
                });
                backend
                    .chat_complete_with_extras(&messages, &extras)
                    .map_err(|e| format!("review model call failed: {e}"))
            })
        } else {
            // No review backend derivable — return an error closure so the
            // worker logs and continues without the LLM call.
            Arc::new(|_prompt| Err("no review backend configured".into()))
        };
        review_fetch = Some(fetch.clone());

        // Build the PII pre-filter seam: the ort PII-Detector when the `pii`
        // role is configured and registered, else the deterministic regex
        // baseline when `auto_enqueue` is on. Fail-open — never a boot error.
        let pii_model = if onnx_registry.as_ref().is_some_and(|r| {
            r.config(fluent_onnx::OnnxRole::Pii.registry_key())
                .is_some()
        }) {
            Some(fluent_onnx::OnnxRole::Pii.registry_key())
        } else {
            None
        };
        let pii_prefilter: Option<Arc<dyn fluent_onnx::PiiSpanDetector>> =
            match fluent_router::ort::pii_prefilter(
                onnx_registry.as_ref(),
                pii_model,
                review_cfg.auto_enqueue,
            ) {
            Ok(prefilter) => prefilter,
            Err(e) => {
                tracing::warn!(
                    target: "coral-router",
                    error = %e,
                    "PII pre-filter build failed — falling back to the regex baseline",
                );
                review_cfg.auto_enqueue.then(|| {
                    Arc::new(fluent_onnx::RegexPiiDetector) as Arc<dyn fluent_onnx::PiiSpanDetector>
                })
            }
        };

        let worker = Arc::new(ReviewWorker::new(
            ledger_arc,
            &(correction_index as Arc<dyn spacy_rs::CorrectionIndex>),
            &(concept_store_arc as Arc<dyn ConceptStore>),
            &fetch,
            pii_prefilter.clone(),
            review_cfg.auto_enqueue,
            review_cfg.review_model.clone().unwrap_or_else(|| "review-model".into()),
            review_cfg.queue_capacity,
            review_cfg.credit_limit,
            fluent_concurrency::tokio_runtime(),
        ));
        review_worker = Some(worker);
        tracing::info!(
            target: "coral-router",
            review_model = %review_cfg.review_model.as_deref().unwrap_or("default"),
            queue_capacity = review_cfg.queue_capacity,
            credit_limit = review_cfg.credit_limit,
            auto_enqueue = review_cfg.auto_enqueue,
            prefilter = pii_prefilter.is_some(),
            "review worker enabled"
        );
    }

    // Async entity-link overlay worker (ROADMAP_20260827_ORT §6.2): opt-in
    // (`overlay.entity_link_enabled`), fail-open. It scores unresolved PROPN
    // spans against boot-cached concept-label embeddings and writes `EntityLink`
    // candidates to `overlay_candidates` — never a doc-id write. Requires a
    // ledger (the candidate plane) and the YaGO concept store.
    let mut entity_link_worker: Option<Arc<EntityLinkWorker>> = None;
    if let Some(overlay_cfg) = &config.overlay {
        if overlay_cfg.entity_link_enabled {
            let Some(ledger_arc) = ledger.as_ref() else {
                tracing::warn!(
                    target: "coral-router",
                    "entity-link overlay enabled but no ledger — skipping",
                );
                return Err("overlay.entity_link_enabled requires a ledger".into());
            };
            // Reuse the M1.1 early-built shared store (overlay enabled ⇒
            // `needs_concept_store` ⇒ built before the pipeline build); fall
            // back to a lazy build only if absent (defensive).
            let concepts: Arc<dyn ConceptStore> = match concept_store.clone() {
                Some(cs) => cs as Arc<dyn ConceptStore>,
                None => Arc::new(SqliteConceptStore::new(
                    ledger_arc
                        .node_store()
                        .shared_sqlite()
                        .expect("ledger must have shared sqlite for overlay"),
                )),
            };
            // The YaGO `Entity` reference class gates which candidates are
            // genuine entities (`is_subclass_of`), resolved through the store.
            let entity_root = concepts
                .resolve_yago_iri(fluent_router::overlay::canonical::ENTITY_ROOT_IRI)
                .unwrap_or_else(|_| {
                    fluent_types::InterlinguaId::from_u64(0x0100_0000_0000_0000)
                });
            // ColBERT entity-link scorer (the `colbert` role): baked over the
            // concept store's labels at boot. Fail-open — the colbert role
            // unconfigured/unregistered → empty scorer, and the worker yields
            // no candidates (identical to a pre-colbert stub).
            let colbert_key = fluent_onnx::OnnxRole::Colbert.registry_key();
            let scorer_model = onnx_registry
                .as_ref()
                .and_then(|r| r.config(colbert_key).map(|_| colbert_key));
            let scorer_wired = scorer_model.is_some();
            let scorer: EntityLinkScorer = match (onnx_registry.as_ref(), scorer_model) {
                (Some(registry), Some(model_key)) => {
                    match fluent_router::ort::colbert_entity_scorer(
                        registry,
                        model_key,
                        &concepts,
                        overlay_cfg.entity_link_threshold,
                    ) {
                        Ok(scorer) => {
                            tracing::info!(
                                target: "coral-router",
                                model = %model_key,
                                "entity-link scorer backed by ColBERT",
                            );
                            scorer
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "coral-router",
                                model = %model_key,
                                error = %e,
                                "entity-link ColBERT scorer failed to bake — empty scorer \
                                 (fail-open)",
                            );
                            Arc::new(|_text| Vec::new())
                        }
                    }
                }
                (_, _) => {
                    tracing::warn!(
                        target: "coral-router",
                        "entity-link overlay enabled but no registered ColBERT (LateInteraction) \
                         model — empty scorer (fail-open)",
                    );
                    Arc::new(|_text| Vec::new())
                }
            };
            let candidates = fluent_router::ledger::overlay::OverlayCandidateStore::new(
                ledger_arc.node_store().shared_sqlite().expect("shared sqlite"),
            );
            let worker = Arc::new(EntityLinkWorker::new(
                &candidates,
                &concepts,
                &scorer,
                overlay_cfg.entity_link_threshold,
                entity_root,
                overlay_cfg.queue_capacity,
                overlay_cfg.credit_limit,
                fluent_concurrency::tokio_runtime(),
            ));
            entity_link_worker = Some(worker);
            tracing::info!(
                target: "coral-router",
                threshold = overlay_cfg.entity_link_threshold,
                queue_capacity = overlay_cfg.queue_capacity,
                credit_limit = overlay_cfg.credit_limit,
                scorer_wired = scorer_wired,
                "entity-link overlay worker enabled",
            );
        }
    }

    // ArcReady annotation-overlay worker (OVERLAYS §8): opt-in
    // (`overlay.arc_ready.enabled`), fail-open. It derives three lazy,
    // at-most-once overlays per recorded node (spacy parse, LLM enrichment,
    // embedding) in parallel from LOD0. Requires a ledger (the shared
    // `ContentNodeStore`); each seam is attached only when its model/pipeline
    // resolves — a missing seam leaves that overlay off (fail-open) without
    // affecting the others. Attached to the server and drained on shutdown so
    // in-flight derivations complete before the router exits.
    let mut overlay_worker: Option<
        Arc<fluent_router::ledger::overlay_worker::OverlayWorker>,
    > = None;
    if let Some(overlay_cfg) = &config.overlay {
        if let Some(arc_cfg) = overlay_cfg.arc_ready.as_ref() {
            if arc_cfg.enabled {
                let Some(ledger_arc) = ledger.as_ref() else {
                    tracing::warn!(
                        target: "coral-router",
                        "overlay.arc_ready.enabled but no ledger — skipping",
                    );
                    return Err("overlay.arc_ready.enabled requires a ledger".into());
                };
                let store = Arc::clone(ledger_arc.node_store());

                // Spacy overlay seam: the standalone NLP pipeline (threads the
                // same resolver as the request-time pipelines). `nlp: false` →
                // not wired (fail-open).
                if arc_cfg.nlp {
                    match config.overlay_nlp_pipeline(&nlp_deps) {
                        Some(pipeline) => store.set_overlay_pipeline(pipeline),
                        None => tracing::warn!(
                            target: "coral-router",
                            "overlay.arc_ready.nlp set but the spacy pipeline failed to build \
                             — spacy overlay skipped (fail-open)",
                        ),
                    }
                }

                // LLM enrichment overlay seam: the named model's `ChatBackend`.
                if let Some(model_key) = arc_cfg.llm_model.as_deref() {
                    match config.local_backend(model_key) {
                        Some(backend) => store.set_overlay_llm(backend),
                        None => tracing::warn!(
                            target: "coral-router",
                            llm_model = %model_key,
                            "overlay.arc_ready.llm_model unresolved — LLM overlay skipped \
                             (fail-open)",
                        ),
                    }
                }

                // Embedding overlay seam: the named model's `EmbeddingProvider`.
                if let Some(model_key) = arc_cfg.embedding_model.as_deref() {
                    match overlay_embedding_provider(&config, model_key) {
                        Some(embedder) => store.set_overlay_embedder(embedder),
                        None => tracing::warn!(
                            target: "coral-router",
                            embedding_model = %model_key,
                            "overlay.arc_ready.embedding_model unresolved — embedding overlay \
                             skipped (fail-open)",
                        ),
                    }
                }

                let worker_cfg =
                    fluent_router::ledger::overlay_worker::OverlayWorkerConfig::from_arc_ready(
                        arc_cfg,
                    );
                let worker = fluent_router::ledger::overlay_worker::OverlayWorker::new(
                    Arc::clone(&store),
                    worker_cfg,
                    fluent_concurrency::tokio_runtime(),
                );
                store.set_overlay_events(worker.sender());
                worker.start();
                tracing::info!(
                    target: "coral-router",
                    nlp = arc_cfg.nlp,
                    llm_model = ?arc_cfg.llm_model,
                    embedding_model = ?arc_cfg.embedding_model,
                    queue_capacity = arc_cfg.queue_capacity,
                    max_concurrent = arc_cfg.max_concurrent,
                    backfill = arc_cfg.backfill,
                    "arc_ready overlay worker enabled",
                );
                overlay_worker = Some(worker);
            }
        }
    }

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
    if let Some(handle) = tier_worker_handle {
        server = server.with_tier_worker(handle);
    }
    if let Some(coordinator) = coordinator {
        server = server.with_coordinator(coordinator);
    }
    if let Some(worker) = review_worker {
        server = server.with_review_worker(worker);
    }
    if let Some(fetch) = review_fetch {
        server = server.with_review_fetch(fetch);
    }
    if let Some(worker) = entity_link_worker {
        server = server.with_entity_link_worker(worker);
    }
    if let Some(worker) = overlay_worker {
        server = server.with_overlay_worker(worker);
    }

    if !instance_pool.is_empty() {
        server = server.with_instance_pool(instance_pool);
    }
    server = server.with_management_api_key(config.sidecar.api_key_env.clone());
    server = server.with_supervisor(_supervisor.clone());
    server = server.with_onnx_registry(onnx_registry);
    server = server.with_onnx_llm_backend(onnx_llm_backend);
    server = server.with_residency_engine(Some(residency_engine));
    // Only attach the unified facade when either fleet exists — an empty fleet
    // would otherwise turn `/instances` into a 200 empty envelope where the
    // pre-M4 path returned 404 "no managed instances".
    if !fleet.is_empty() {
        server = server.with_fleet(fleet);
    }

    if let Some(ctx) = mock_dispatch {
        server = server.with_mock(ctx);
    }

    // Serve until a shutdown signal. Coral Router is the process owner of the
    // spawned llama-servers, so a signal must stop the supervisor (killing its
    // children) before the process exits - a plain SIGTERM/SIGINT default
    // would orphan every managed llama-server, leaking ports and VRAM.
    //
    // The shutdown watch is the single graceful-stop signal: a background
    // task fires it on SIGTERM/SIGINT, `serve` drains its owned background and
    // connection tasks (abort + await, within a timeout) and returns, and the
    // supervisor is stopped afterwards - so no server task and no llama-server
    // is left detached on shutdown.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    let serve_result = server.serve(shutdown_rx).await;

    // Always stop the managed llama-servers before exiting, whether serving
    // ended normally (graceful shutdown) or failed (e.g. bind error) - a
    // failed serve still leaves the already-spawned llama-servers running.
    if let Some(supervisor) = _supervisor.as_ref() {
        supervisor.shutdown().await;
    }

    serve_result?;
    Ok(())
}

/// Resolve on SIGINT or SIGTERM (whichever comes first), so restart loops
/// (`make router-start`) can stop the process tree cleanly.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Construct the boot-loaded chart store and attach it to the plan route.
///
/// Semantics (decision — fail fast): a missing chart directory yields an
/// empty store (`ChartStore::load_dir` logs a `warn!`); a present-but-invalid
/// chart file aborts boot so a corrupted library never half-loads.
///
/// When `charts.index_path` is configured the `workflow_library` index is
/// built at boot (lazy + failure-tolerant — a down embedding endpoint
/// disables HNSW retrieval but never aborts boot; deterministic match and
/// LLM adjudication still work).  The adjudicator backend is wired from
/// `charts.selector_model` when set.
fn build_plan_route(
    config: &RouterConfig,
    ledger_store: Option<&Arc<fluent_router::node_store::ContentNodeStore>>,
    onnx: Option<&fluent_router::ort::OrtRegistry>,
) -> PlanRoute {
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

    // Build the workflow_library HNSW index at boot. Lazy: only
    // when index_path is configured. Failure-tolerant: a missing/unreachable
    // embedding endpoint skips the build with a warning, never aborts boot.
    if config.charts.index_path.is_some() {
        match default_chart_embedder(config, onnx) {
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
    // ColBERT two-stage retrieval: when the `colbert` role is configured and
    // resolves to a LateInteraction session, build the MaxSim reranker.
    #[cfg(feature = "onnx")]
    {
        let colbert_key = fluent_onnx::OnnxRole::Colbert.registry_key();
        if let Some(registry) = &onnx {
            if config
                .onnx
                .as_ref()
                .is_some_and(|f| f.has(fluent_onnx::OnnxRole::Colbert))
            {
                match fluent_router::ort::onnx_colbert_reranker(registry, colbert_key) {
                    Ok(Some(retriever)) => {
                        let reranker = Arc::new(
                            fluent_router::ort::ColbertChartReranker::new(retriever),
                        );
                        route = route.with_colbert_reranker(reranker);
                        tracing::info!(
                            target: "router.main",
                            role = %colbert_key,
                            "ColBERT MaxSim reranker built",
                        );
                    }
                    Ok(None) => {
                        tracing::warn!(
                            target: "router.main",
                            role = %colbert_key,
                            "colbert role configured but not a registered LateInteraction \
                             session — ColBERT reranking disabled",
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "router.main",
                            role = %colbert_key,
                            error = %e,
                            "ColBERT reranker build failed — falling back to HNSW-only retrieval",
                        );
                    }
                }
            }
        }
    }
    // Server-side execution: the same charts model runs a selected chart's
    // targets (and doubles as the rubric judge). A shared limiter bounds
    // concurrent chart-target LLM calls. When no charts model is configured
    // the exact fit degrades to a fresh draft (see `PlanRoute::execute_chart`).
    if let Some(backend) = default_adjudicator_backend(config) {
        route = route.with_execution_backend(backend);
    }
    route = route.with_limiter(Arc::new(fluent_concurrency::pool::Limiter::new(
        CHART_EXECUTION_CONCURRENCY,
    )));
    // Learning loop: attach the dispatch post-processing hook when the
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
    // When a shared ledger store exists, attach the prompt assembler so
    // the selector/adjudicator render the session ledger through the same
    // budget/relevance rules (a request that carries a `session_id` folds it in).
    if let Some(store) = ledger_store {
        let ctx = fluent_router::routes::plan::PromptAssemblerCtx::new(
            Arc::clone(store),
            fluent_router::ledger::prompt::LedgerPromptAssembler,
            fluent_router::ledger::prompt::PromptBudget::new(
                config
                    .ledger
                    .as_ref()
                    .map(|l| l.orchestrator.prompt_budget_chars)
                    .unwrap_or(32768),
            ),
            fluent_router::ledger::prompt::LodSpec::full(),
        );
        route = route.with_prompt_assembler(ctx);
        tracing::info!(target: "coral-router", "plan route prompt assembler attached");
    }
    route
}

/// Number of embedding dimensions to declare for the OpenAI chart embedder.
/// The actual vector length is whatever the endpoint returns (the embeddings
/// HTTP client parses the response); this only sets the declared capacity.
/// Onnx-declared embedding models derive their dims from config/metadata
/// instead (ROADMAP_20260827_ORT §0.6).
const CHART_EMBEDDING_DIMS: u32 = 768;

/// Max concurrent chart-target LLM calls during server-side execution.
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
/// then the classifier model's). Two branches:
///
/// - **Onnx** (ROADMAP_20260827_ORT §0.6): when the resolved model declares an
///   `onnx` block with `task: FillMask`, build the ort encoder from the boot
///   registry's session (dims model-derived), wrapped in
///   `CachedEmbeddingProvider`. A declared-but-broken onnx encoder returns
///   `None` (HNSW disabled, degraded) — it never silently falls back to the
///   OpenAI path for a model the operator declared as onnx.
/// - **OpenAI**: unchanged — an empty API key is sent (local llama.cpp servers
///   ignore the header).
///
/// Returns `None` when no model is configured or no embedder is derivable,
/// leaving HNSW retrieval disabled.
fn default_chart_embedder(
    config: &RouterConfig,
    onnx: Option<&fluent_router::ort::OrtRegistry>,
) -> Option<Arc<dyn EmbeddingProvider>> {
    // In-process ONNX encoder role: when configured, it is the preferred chart
    // embedder (dims model-derived), independent of any HTTP model key.
    #[cfg(feature = "onnx")]
    {
        if let Some(registry) = onnx {
            if config
                .onnx
                .as_ref()
                .is_some_and(|f| f.has(fluent_onnx::OnnxRole::Encoder))
            {
                let key = fluent_onnx::OnnxRole::Encoder.registry_key();
                return match fluent_router::ort::onnx_chart_embedder(registry, key) {
                    Ok(Some(provider)) => Some(provider),
                    Ok(None) => {
                        tracing::warn!(
                            target: "coral-router",
                            "onnx encoder role present but not a FillMask session — HNSW retrieval disabled (degraded)",
                        );
                        None
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "coral-router",
                            error = %e,
                            "onnx encoder chart embedder failed to build — HNSW retrieval disabled (degraded)",
                        );
                        None
                    }
                };
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = onnx;
        if config
            .onnx
            .as_ref()
            .is_some_and(|f| f.has(fluent_onnx::OnnxRole::Encoder))
        {
            tracing::warn!(
                target: "coral-router",
                "onnx encoder role declared but this build has the `onnx` feature off — HNSW retrieval disabled (degraded)",
            );
            return None;
        }
    }

    // OpenAI-compatible path: an HTTP embedding model key (no onnx encoder
    // role configured). An empty API key is sent (local llama.cpp servers
    // ignore the header).
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
/// Build an `EmbeddingProvider` from a named `models` key (the arc_ready
/// embedding overlay seam). Mirrors `default_chart_embedder`'s OpenAI-compatible
/// path: one factory, no new transport. `None` when the key is absent → the
/// embedding overlay is off (fail-open).
fn overlay_embedding_provider(
    config: &RouterConfig,
    key: &str,
) -> Option<Arc<dyn EmbeddingProvider>> {
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

fn default_adjudicator_backend(config: &RouterConfig) -> Option<Arc<dyn ChatBackend>> {
    let key = config.charts.selector_model.as_deref()?;
    config.local_backend(key)
}

/// Build the chart-candidate reranker backend from the root-level
/// `reranker_model`, if configured.  Mirrors `default_adjudicator_backend`:
/// exactly one place constructs a concrete `LlmClient` for the reranker. 
/// The rerank is a cross-encoder-style LLM call over the HNSW candidates
/// before adjudication (`None` skips the stage).
fn default_reranker_backend(config: &RouterConfig) -> Option<Arc<dyn ChatBackend>> {
    let key = config.reranker_model.as_deref()?;
    config.local_backend(key)
}

/// Build the rigor route from `config.rigor`, mirroring `build_plan_route`.
///
/// Each role backend is DIP-constructed exactly once from its model key via
/// `default_rigor_backend`. With no `rigor` section (or missing keys), the
/// route is present but unconfigured — requests return an explicit
/// `Unconfigured` error, never a crash (`env/coral-router.json` ships without
/// a `rigor` section).
fn build_rigor_route(
    config: &RouterConfig,
    ledger_store: Option<&Arc<fluent_router::node_store::ContentNodeStore>>,
) -> RigorRoute {
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
    // When a shared ledger store exists, the judge renders its review
    // prompt over the session ledger through the assembler's budget/relevance
    // rules (the red team keeps its LOD0 `FilteredLedger` view unchanged). The
    // store presence is the opt-in gate; the route reads the ledger by session.
    if ledger_store.is_some() {
        route = route.with_prompt_assembler(
            fluent_router::ledger::prompt::LedgerPromptAssembler,
            fluent_router::ledger::prompt::PromptBudget::new(
                config
                    .ledger
                    .as_ref()
                    .map(|l| l.orchestrator.prompt_budget_chars)
                    .unwrap_or(32768),
            ),
            fluent_router::ledger::prompt::LodSpec::full(),
        );
        tracing::info!(target: "coral-router", "rigor judge prompt assembler attached");
    }
    tracing::info!(
        target: "coral-router",
        blue_model = ?cfg.blue_model,
        red_model = ?cfg.red_model,
        judge_model = ?cfg.judge_model,
        kv_cache_enabled = cfg.kv_cache_enabled,
        max_passes = cfg.max_passes.rounds(),
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
        let embedder = super::default_chart_embedder(&config, None);
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

    /// A config with a valid OpenAI embedding model AND a declared `encoder`
    /// role. The endpoint is left VALID so the OpenAI path WOULD succeed — the
    /// encoder role must be preferred and, when its encoder cannot be built
    /// (stub registry), return `None` rather than silently switching models.
    fn onnx_embedder_config() -> super::RouterConfig {
        let mut config = super::load_router_config();
        let model: serde_json::Value =
            serde_json::from_str(r#"{
                "endpoint": "http://127.0.0.1:9999/v1/chat/completions",
                "name": "embed",
                "intelligence": 2,
                "cost_input": 0.0,
                "cost_output": 0.0,
                "cost_cached_read": 0.0,
                "speed": 10
            }"#)
            .unwrap();
        config
            .models
            .insert("embed".into(), serde_json::from_value(model).unwrap());
        config.embedding_model = Some("embed".into());
        config.onnx = Some(fluent_onnx::OnnxFleetConfig {
            encoder: Some(fluent_onnx::OnnxRoleConfig {
                pinned: false,
                no_sleep: false,
                sleep_idle_seconds: None,
                total_timeout_ms: 0,
                idle_timeout_ms: 0,
                params: None,
                instances: None,
                model: fluent_onnx::OnnxConfig::new()
                    .model_path("/models/encoder/onnx/model_q8.onnx")
                    .tokenizer_path("/models/encoder/tokenizer.json")
                    .build(),
            }),
            ..Default::default()
        });
        config
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn onnx_embedder_is_preferred_over_openai() {
        use std::sync::Arc;

        let config = onnx_embedder_config();
        // A stub registry: the encoder role registered as an Always FillMask
        // encoder, but its stub handle holds no real session — the encoder
        // build must fail.
        let registry = fluent_onnx::OrtSessionRegistry::new(Arc::new(StubLoader));
        let encoder_cfg = config
            .onnx
            .as_ref()
            .and_then(|f| f.encoder.as_ref())
            .map(|rc| rc.clone().to_onnx_config(fluent_onnx::OnnxRole::Encoder))
            .unwrap();
        registry
            .register(fluent_onnx::OnnxRole::Encoder.registry_key().to_string(), encoder_cfg)
            .expect("register");

        // The onnx encoder role is preferred: even though the endpoint is valid
        // enough for OpenAI, a broken onnx encoder yields None — never a
        // wrong-model OpenAI fallback.
        let embedder = super::default_chart_embedder(&config, Some(&Arc::new(registry)));
        assert!(
            embedder.is_none(),
            "onnx-declared encoder role must not fall back to the OpenAI path"
        );
    }

    #[test]
    #[cfg(not(feature = "onnx"))]
    fn onnx_embedder_without_feature_is_fail_open() {
        let config = onnx_embedder_config();
        let embedder = super::default_chart_embedder(&config, None);
        assert!(
            embedder.is_none(),
            "no-ort build with an onnx encoder role: fail-open (degraded, no wrong-model fallback)"
        );
    }

    #[derive(Default)]
    struct StubLoader;

    impl fluent_onnx::SessionLoader for StubLoader {
        fn load(
            &self,
            _config: &fluent_onnx::OnnxConfig,
            _model_key: &str,
        ) -> Result<fluent_onnx::SessionHandle, fluent_onnx::OrtError> {
            Ok(fluent_onnx::SessionHandle::new("stub"))
        }
    }
}
