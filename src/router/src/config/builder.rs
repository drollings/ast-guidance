//! Pipeline builder — constructs pipeline stages from `RouterConfig`.
//! Separated from `config.rs` to keep the configuration types focused
//! on data definition rather than orchestration.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use common_core::config::load_json_or_default;
use fluent_llm::client::ChatBackend;
use fluent_llm::{LlmClient, LlmConfig};
use fluent_wvr::prelude::Component;

use super::{default_true, RejectPatterns, RouterConfig};
use crate::pipeline::PipelineOrchestrator;
use crate::score_matrix::ScoreMatrix;

/// Named pipeline parameters. Pipelines are stored as a map keyed by name.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct PipelineParams {
    #[serde(default = "default_true")]
    pub deterministic_prefilter: bool,
    #[serde(default = "default_true")]
    pub classifier: bool,
    #[serde(default = "default_true")]
    pub router: bool,
    #[serde(default = "default_coherence_threshold")]
    pub coherence_threshold: f64,
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Bounds the number of concurrently executing classifier LLM calls for
    /// this pipeline. `None` defaults to `available_parallelism()`.
    #[serde(default)]
    pub classifier_max_concurrency: Option<usize>,
    #[serde(default)]
    pub blacklist: Option<String>,
    #[serde(default)]
    pub score_matrix: Option<ScoreMatrix>,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            deterministic_prefilter: true,
            classifier: true,
            router: true,
            coherence_threshold: default_coherence_threshold(),
            classifier_model: None,
            classifier_max_concurrency: None,
            blacklist: None,
            score_matrix: None,
        }
    }
}

fn default_coherence_threshold() -> f64 {
    0.70
}

/// Default classifier concurrency cap: the machine's available parallelism,
/// never fewer than 1 worker.
fn default_classifier_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().max(1))
}

impl RouterConfig {
    pub fn load_reject_patterns(path: &str) -> RejectPatterns {
        load_json_or_default::<RejectPatterns>(Path::new(path))
    }

    pub fn routing_config(&self) -> super::RoutingConfig {
        super::RoutingConfig {
            routes: self.routes.clone(),
            models: self.models.clone(),
            model_groups: self.model_groups.clone(),
            system_prompt: self.system_prompt.clone(),
            safety_threshold: self.safety_threshold,
            default_route: self.default_route.clone(),
            score_matrix: self.score_matrix.clone(),
        }
    }

    pub fn build_named_pipeline(&self, name: &str) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend(name, None)
    }

    pub fn build_named_pipeline_with_backend(
        &self,
        name: &str,
        classifier_backend: Option<Arc<dyn ChatBackend>>,
    ) -> Option<PipelineOrchestrator> {
        let params = self.pipelines.get(name)?;
        let mut stages: Vec<Arc<dyn Component>> = Vec::new();

        if params.deterministic_prefilter {
            if let Some(ref blacklist_path) = params.blacklist {
                let reject_patterns = Self::load_reject_patterns(blacklist_path);
                stages.push(Arc::new(
                    crate::stages::deterministic::DeterministicPreFilter::from_config(
                        &reject_patterns,
                    ),
                ));
            } else {
                stages.push(Arc::new(
                    crate::stages::deterministic::DeterministicPreFilter::new(),
                ));
            }
        }

        if params.classifier {
            let routing_config = self.routing_config();
            let classifier_intel = classifier_intelligence(self, params);
            let classifier_model = resolve_classifier_model_key(self, params)
                .map_or_else(|| "unknown".into(), str::to_string);
            let client: Arc<dyn ChatBackend> = if let Some(backend) = classifier_backend {
                tracing::info!(target: "router.config", pipeline = %name, backend = "mock/transcript", "classifier using injected backend");
                backend
            } else {
                let client = build_classifier_client(self, name, params)?;
                tracing::info!(target: "router.config", pipeline = %name, "classifier using real LLM client");
                client
            };
            let max_concurrency = params
                .classifier_max_concurrency
                .unwrap_or_else(default_classifier_concurrency);
            let limiter = Arc::new(fluent_concurrency::pool::Limiter::new(max_concurrency));
            tracing::debug!(target: "router.config", pipeline = %name, classifier_max_concurrency = max_concurrency, "classifier concurrency limiter constructed");
            stages.push(Arc::new(crate::stages::classifier::ClassifierStage::new(
                client,
                routing_config,
                params.coherence_threshold,
                params.score_matrix.clone(),
                classifier_intel,
                classifier_model,
                limiter,
            )));
        } else if classifier_backend.is_some() {
            tracing::warn!(
                target: "router.config",
                pipeline = %name,
                "classifier backend was provided but classifier is disabled for this pipeline"
            );
        }

        Some(PipelineOrchestrator::new(stages))
    }

    pub fn build_all_pipelines(&self) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend(None)
    }

    pub fn build_all_pipelines_with_backend(
        &self,
        classifier_backend: Option<&Arc<dyn ChatBackend>>,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        let mut map = HashMap::new();
        let mut dropped = Vec::new();
        let pipeline_count = self.pipelines.len();
        let has_mock = classifier_backend.is_some();
        tracing::info!(target: "router.config", pipeline_count = pipeline_count, mock_backend = has_mock, classifier_model = ?self.classifier_model, default_route = %self.default_route, "building pipelines");
        for name in self.pipelines.keys() {
            let backend_for_pipeline = classifier_backend.cloned();
            if let Some(pipeline) =
                self.build_named_pipeline_with_backend(name, backend_for_pipeline)
            {
                map.insert(name.clone(), Arc::new(pipeline));
            } else {
                dropped.push(name.clone());
                let params = &self.pipelines[name];
                tracing::warn!(
                    target: "router.config",
                    pipeline = %name,
                    configured_classifier = ?params.classifier_model.as_deref(),
                    resolved_classifier = ?resolve_classifier_model_key(self, params),
                    "pipeline not built — classifier model unresolved or invalid",
                );
            }
        }
        if !dropped.is_empty() {
            tracing::error!(
                target: "router.config",
                built = map.len(),
                configured = pipeline_count,
                dropped = ?dropped,
                "some configured pipelines were not built",
            );
        }
        tracing::info!(target: "router.config", built = map.len(), "pipelines built");
        map
    }

    pub fn route_pipeline_names(&self, model_name: &str) -> Vec<String> {
        self.routes
            .get(model_name)
            .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone())
    }
}

/// Resolve the classifier model key from config, following the priority:
/// 1. Pipeline-level `classifier_model`
/// 2. Root-level `classifier_model`
/// 3. First model in the `fast` model group
fn resolve_classifier_model_key<'a>(
    config: &'a RouterConfig,
    params: &'a PipelineParams,
) -> Option<&'a str> {
    params
        .classifier_model
        .as_deref()
        .or(config.classifier_model.as_deref())
        .or_else(|| {
            config
                .model_groups
                .get("fast")
                .and_then(|names| names.first())
                .map(String::as_str)
        })
}

/// Return the classifier model's intelligence rating, or 0 if not found.
fn classifier_intelligence(config: &RouterConfig, params: &PipelineParams) -> u8 {
    resolve_classifier_model_key(config, params)
        .and_then(|k| config.models.get(k))
        .map_or(0, |m| m.intelligence)
}

/// Build a classifier LLM client from the model config.
///
/// # DIP note
/// This factory is the **only** place in the crate that constructs a concrete
/// `LlmClient`.  The rest of the pipeline receives `Arc<dyn ChatBackend>` and
/// is oblivious to the concrete implementation.  There is exactly one
/// `ChatBackend` implementation today (`LlmClient`); if a second appears,
/// the factory can inject it without touching pipeline code.
fn build_classifier_client(
    config: &RouterConfig,
    _name: &str,
    params: &PipelineParams,
) -> Option<Arc<dyn ChatBackend>> {
    let model_key = resolve_classifier_model_key(config, params)?;
    let entry = config.models.get(model_key)?;
    let model_name_for_llm = entry.name.as_deref().unwrap_or(model_key);
    let classifier_config = LlmConfig::new()
        .api_url(entry.endpoint.clone())
        .model(model_name_for_llm.to_string())
        .timeout_ms(entry.total_timeout_ms)
        .maybe_extra_body_params(entry.params.clone())
        .build();
    Some(Arc::new(LlmClient::with_config(classifier_config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;
    use tracing_subscriber::layer::SubscriberExt;

    use common_core::sync::lock;

    use crate::charts::binding::Entity;
    use crate::charts::{ChartDef, ChartError};
    use crate::test_stubs::StubChatBackend;
    use fluent_concurrency::pool::Limiter;

    /// A `MakeWriter` that captures formatted log lines for assertions.
    #[derive(Clone, Default)]
    struct LogCapture(Arc<Mutex<Vec<String>>>);

    impl Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            lock(&self.0).push(String::from_utf8_lossy(buf).into_owned());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl tracing_subscriber::fmt::MakeWriter<'_> for LogCapture {
        type Writer = Self;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(capture.clone())
                .with_ansi(false)
                .with_target(true),
        );
        let result = tracing::subscriber::with_default(subscriber, f);
        let logs = lock(&capture.0).clone();
        (result, logs)
    }

    fn config_with_unresolvable_classifier() -> RouterConfig {
        // `classifier` is enabled but no `classifier_model`, no root
        // `classifier_model`, and no `fast` model group resolves a key.
        serde_json::from_str(
            r#"{
                "pipelines": {
                    "default": {"deterministic_prefilter": true, "classifier": true}
                },
                "models": {},
                "model_groups": {},
                "routes": {}
            }"#,
        )
        .expect("valid config")
    }

    #[test]
    fn unresolvable_classifier_drops_pipeline_with_warning() {
        let config = config_with_unresolvable_classifier();
        let (map, logs) = capture_logs(|| config.build_all_pipelines());
        let joined = logs.join("\n");

        assert!(map.is_empty(), "no pipeline should build");
        assert!(
            joined.contains("pipeline not built"),
            "missing per-pipeline warning, logs:\n{joined}"
        );
        assert!(
            joined.contains("\"default\""),
            "warning must name the dropped pipeline, logs:\n{joined}"
        );
        assert!(
            joined.contains("configured_classifier") && joined.contains("resolved_classifier"),
            "warning must log resolved-vs-configured classifier keys, logs:\n{joined}"
        );
        assert!(
            joined.contains("some configured pipelines were not built"),
            "missing aggregate error, logs:\n{joined}"
        );
    }

    #[test]
    fn resolvable_classifier_builds_pipeline_without_warnings() {
        let config: RouterConfig = serde_json::from_str(
            r#"{
                "pipelines": {"default": {"classifier": true}},
                "models": {"fast": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 10}},
                "model_groups": {"fast": ["fast"]}
            }"#,
        )
        .expect("valid config");
        let (map, logs) = capture_logs(|| config.build_all_pipelines());
        let joined = logs.join("\n");

        assert_eq!(map.len(), 1, "pipeline should build");
        assert!(
            !joined.contains("pipeline not built"),
            "no drop warning expected, logs:\n{joined}"
        );
        assert!(
            !joined.contains("some configured pipelines were not built"),
            "no aggregate error expected, logs:\n{joined}"
        );
    }

    /// Records every system prompt it receives, and returns a canned response.
    struct RecordingBackend {
        prompts: Arc<Mutex<Vec<String>>>,
    }

    impl ChatBackend for RecordingBackend {
        fn chat_complete(
            &self,
            messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            lock(&self.prompts).extend(
                messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.clone()),
            );
            Ok(r#"{"ok": true}"#.to_string())
        }
    }

    fn triage_chart() -> ChartDef {
        serde_json::from_str(
            r#"{
                "name": "bug_triage",
                "description": "triage",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    {
                        "name": "reproduce",
                        "provides": ["repro_plan"],
                        "depends": [],
                        "template": "Plan repro for: {{ request }}",
                        "essential": true
                    },
                    {
                        "name": "root_cause",
                        "provides": ["root_cause"],
                        "depends": [
                            { "kind": "capability", "name": "repro_plan" },
                            { "kind": "entity_match", "name": "report",
                              "description": "the report",
                              "predicate": {
                                "fields": [
                                    { "path": "title", "ty": "string", "required": true }
                                ]
                              },
                              "required": true }
                        ],
                        "template": "Prior plan: {{ upstream.reproduce.output }}\nReport: {% for e in deps.report %}{{ e.value.title }}{% endfor %}\nCause of: {{ request }}",
                        "essential": true
                    },
                    {
                        "name": "fix_plan",
                        "provides": ["fix_plan"],
                        "depends": [
                            { "kind": "capability", "name": "root_cause" }
                        ],
                        "template": "Fix for: {{ request }}",
                        "essential": true
                    }
                ]
            }"#,
        )
        .expect("triage chart JSON")
    }

    fn request_ctx(text: &str, entities: &[Entity]) -> fluent_wvr::WorkContext {
        let ctx_json = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": text}]
        });
        let mut ctx = fluent_wvr::WorkContext::default();
        ctx.set_structured("request", &ctx_json);
        if !entities.is_empty() {
            ctx.set_structured(crate::charts::binding::ENTITIES_META_KEY, &entities);
        }
        ctx
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chart_executes_in_topo_order_with_preamble_and_prior_output() {
        let entity = Entity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on startup"}),
        };

        let prompts = Arc::new(Mutex::new(Vec::<String>::new()));
        let backend: Arc<dyn ChatBackend> = Arc::new(RecordingBackend {
            prompts: prompts.clone(),
        });
        let limiter = Arc::new(Limiter::new(4));
        let plan = crate::charts::execute::ChartExecutionPlan::compile(
            &triage_chart(),
            std::slice::from_ref(&entity),
            &backend,
            &limiter,
        )
        .expect("chart compiles into an executable plan");

        let ctx = request_ctx("app crashes on startup", std::slice::from_ref(&entity));
        let opts = crate::charts::execute::ChartExecOptions {
            runtime: fluent_concurrency::tokio_runtime(),
            ..Default::default()
        };
        let summary = plan
            .execute(&ctx, &opts)
            .await
            .expect("chart executes under Zone supervision");

        // Topo order: reproduce → root_cause → fix_plan (3 completed targets).
        assert_eq!(summary.completed.len(), 3);
        assert!(summary.failed.is_empty());
        assert!(summary.accepted);
        let reasons: Vec<&str> = summary
            .completed
            .iter()
            .map(|d| d.reason.as_str())
            .collect();
        assert_eq!(
            reasons,
            vec![
                "chart target 'reproduce' completed",
                "chart target 'root_cause' completed",
                "chart target 'fix_plan' completed",
            ]
        );

        // Every stage made one LLM call (3 system prompts recorded).
        let recorded = prompts.lock().unwrap().clone();
        assert_eq!(recorded.len(), 3, "one LLM call per chart target");

        // reproduce's prompt carries the request.
        assert!(recorded[0].contains("app crashes on startup"));
        // root_cause's prompt carries the entity preamble AND the prior output.
        assert!(
            recorded[1].contains("Segfault on startup"),
            "root_cause prompt must include the bound entity preamble: {}",
            recorded[1]
        );
        assert!(
            recorded[1].contains(r#"{"ok": true}"#),
            "root_cause prompt must include the prior target output: {}",
            recorded[1]
        );
        // fix_plan's prompt carries the request.
        assert!(recorded[2].contains("app crashes on startup"));
    }

    #[test]
    fn chart_compile_rejects_unbound_chart_at_build_time() {
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always("{}"));
        let limiter = Arc::new(Limiter::new(4));
        // No entities → root_cause's required `report` dep is unmatched.
        let Err(err) =
            crate::charts::compile::compile_chart_stages(&triage_chart(), &[], &backend, &limiter)
        else {
            panic!("expected compile error for unbound chart")
        };
        assert!(
            matches!(&err, ChartError::Compile { reason } if reason.contains("not fully bound")),
            "expected compile error, got: {err}"
        );
    }
}
