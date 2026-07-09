pub mod audit;
pub mod auth;
pub mod handler;
pub mod http;
pub mod stdio;

pub use handler::DaemonHandler;

use std::sync::{Arc, RwLock};

use crate::components::AnalyzeFormComponent;
use crate::config::DaemonConfig;
use crate::dispatcher::{FieldValueDispatcher, LlmDispatcher, LocalDispatcher, TieredDispatcher};
use crate::error::CopilotError;
use crate::memory::{FormFillConfig, FormFillStore};
use crate::profile::Profile;
use crate::server::audit::AuditLog;
use crate::similarity::FieldSimilarityStore;
use common_core::metrics::LatencyHistogram;
use dag::middleware::{MiddlewareChain, RetryMiddleware, TimingMiddleware};
use fluent_wvr::Component;

/// Start the Native Messaging STDIO transport.
///
/// Constructs a tiered dispatcher (Local → LLM), an optional audit log,
/// wraps the `AnalyzeFormComponent` in a middleware chain (retry + timing),
/// and runs the synchronous Native Messaging framing loop in a blocking task.
pub async fn serve_native_messaging(config: DaemonConfig) -> Result<(), CopilotError> {
    let profile = Profile::load_from_path(&config.profile_path)?;
    profile.validate()?;
    let profile = crate::profile::shared(profile);

    let local = Arc::new(LocalDispatcher::new(profile.clone()));
    let client = Arc::new(guidance_llm::client::LlmClient::new(
        &config.llm_url,
        &config.llm_model,
    ));

    // Load or create the similarity store.
    let store_path = config.profile_path.parent().map_or_else(
        || std::path::PathBuf::from("similarity-store.jsonl"),
        |p| p.join("similarity-store.jsonl"),
    );
    let store = Arc::new(RwLock::new(
        FieldSimilarityStore::load(&store_path).unwrap_or_default(),
    ));

    let llm = Arc::new(LlmDispatcher::with_store(
        client,
        profile.clone(),
        store.clone(),
    ));

    let dispatcher: Arc<dyn FieldValueDispatcher> =
        Arc::new(TieredDispatcher::new().with(local).with(llm));

    // Open or create the form fill memory store.
    let memory_db_path = config.profile_path.parent().map_or_else(
        || std::path::PathBuf::from("form-fills.db"),
        |p| p.join("form-fills.db"),
    );
    let memory_store = Arc::new(
        FormFillStore::open(FormFillConfig {
            db_path: memory_db_path,
            ..FormFillConfig::default()
        })
        .unwrap_or_else(|e| {
            tracing::warn!("failed to open form fill store: {e}; memory disabled");
            // Create a fallback in-memory store.
            FormFillStore::open_memory().expect("in-memory store must succeed")
        }),
    );

    let histogram = Arc::new(LatencyHistogram::new());

    // Build the AnalyzeFormComponent and wrap with middleware chain.
    let base: Arc<dyn Component> = Arc::new(
        AnalyzeFormComponent::builder()
            .dispatcher(dispatcher)
            .profile(profile.clone())
            .memory(memory_store.clone())
            .build(),
    );
    let chain = MiddlewareChain::new()
        .push(Box::new(TimingMiddleware))
        .push(Box::new(RetryMiddleware::new(2, 50)));
    let unit = chain.apply(base);

    let mut handler = DaemonHandler::new(profile, unit)
        .with_histogram(histogram.clone())
        .with_memory(memory_store);

    // Attach audit log if configured.
    if let Some(ref path) = config.audit_log_path {
        let audit = Arc::new(AuditLog::open(path)?);
        handler = handler.with_audit(audit);
    }

    let handler = Arc::new(handler);

    if config.enable_rest {
        let http_config = config.clone();
        let http_handler = handler.clone();
        tokio::spawn(async move {
            if let Err(e) = http::run_http(&http_config, http_handler).await {
                tracing::error!("HTTP server error: {e}");
            }
        });
    }

    let result =
        tokio::task::spawn_blocking(move || stdio::run_native_messaging(handler.as_ref(), &config))
            .await
            .map_err(|e| CopilotError::Internal(format!("blocking task panicked: {e}")))?;

    // Save similarity store on shutdown.
    if let Ok(store) = store.read() {
        if let Err(e) = store.save() {
            tracing::warn!("failed to save similarity store: {e}");
        }
    }

    result
}
