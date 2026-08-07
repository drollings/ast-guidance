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

use super::{default_true, strip_declaration_params, RejectPatterns, RouterConfig};
use crate::pipeline::PipelineOrchestrator;
use crate::score_matrix::ScoreMatrix;
use crate::target_match::{TargetBackends, TargetMatcher};

/// In-group target-matching policy for a pipeline (§4.6 of the routing
/// roadmap). `SelfAssess` (default) runs the VISION ladder: each candidate
/// target self-assesses the prompt and defers to the next, more-intelligent
/// group member when the assessed complexity exceeds its `intelligence`.
/// `Static` restores today's behavior — the cheapest qualifying model is
/// picked at route-resolution time with no self-assessment calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetMatchMode {
    /// Run the per-candidate self-assessment ladder for 2+ member groups
    /// (single-member groups resolve statically, byte-identical to today).
    #[default]
    #[serde(rename = "self_assess")]
    SelfAssess,
    /// Pick the cheapest qualifying model at resolution time (no LLM calls).
    #[serde(rename = "static")]
    Static,
}

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
    /// When `true` and a `score_matrix` is configured, the matrix's
    /// top-scoring route **decides** the dispatch target (weighted selection
    /// over the four score axes) instead of the LLM's `action`/`target` being
    /// metadata-only. Coherence/safety thresholds and the `reject` action stay
    /// as hard gates that run first (M5). Default `false` so existing behavior
    /// and goldens are untouched until a deployment opts in.
    #[serde(default)]
    pub score_matrix_authoritative: bool,
    /// Maximum retry attempts for the classifier when its LLM response fails
    /// JSON parsing (`0` = disabled, the default — existing behavior is
    /// unchanged). When `> 0`, the classifier stage is wrapped in a
    /// `RetryClassifier` that re-executes it with escalating corrective
    /// prompts on `metadata.fallback = true` (M6).
    #[serde(default)]
    pub classifier_retry_max: u32,
    /// Escalating corrective system prompts used on each retry attempt (the
    /// last prompt is reused when retries exceed the list length). Defaults to
    /// two stock prompts that demand strict JSON.
    #[serde(default = "default_classifier_retry_prompts")]
    pub classifier_retry_prompts: Vec<String>,
    /// In-group target-matching policy (§4.6). `SelfAssess` (default) runs the
    /// target-matching ladder for 2+ member groups; `Static` restores today's
    /// cheapest-qualifying pick.
    #[serde(default)]
    pub target_match: TargetMatchMode,
    /// Per-self-assessment wall-clock budget for the target-matching ladder.
    /// Defaults to `DEFAULT_TOTAL_TIMEOUT_MS` (the shared timeout constant).
    #[serde(default = "default_target_match_timeout_ms")]
    pub target_match_timeout_ms: u64,
}

fn default_target_match_timeout_ms() -> u64 {
    common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS
}

fn default_classifier_retry_prompts() -> Vec<String> {
    vec![
        "Your previous output failed JSON parsing. Respond with ONLY a single valid JSON \
         object matching the requested schema — no prose, no markdown fences, no trailing text."
            .into(),
        "Your previous output was still not valid JSON. Output exactly one JSON object with \
         the required fields and nothing else."
            .into(),
    ]
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
            score_matrix_authoritative: false,
            classifier_retry_max: 0,
            classifier_retry_prompts: default_classifier_retry_prompts(),
            target_match: TargetMatchMode::SelfAssess,
            target_match_timeout_ms: default_target_match_timeout_ms(),
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
        // M4.4: when a classification tree is configured and no explicit
        // `system_prompt` is set, derive one from the root classifier node's
        // children so flat consumers still observe the auto-generated prompt.
        let system_prompt = if self.system_prompt.is_empty() {
            self.classification
                .as_ref()
                .and_then(super::ClassificationTree::derive_system_prompt)
                .unwrap_or_default()
        } else {
            self.system_prompt.clone()
        };
        super::RoutingConfig {
            routes: self.routes_view(),
            models: self.models.clone(),
            model_groups: self.model_groups.clone(),
            system_prompt,
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
            let injected_backend = classifier_backend.is_some();
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

            // M3 target-matching ladder: built only when the pipeline opts in
            // (`target_match: "self_assess"`). The injected mock/transcript
            // backend is the matcher's `default` covering every key absent from
            // the per-key map (test mode: the map is empty, so every candidate
            // routes through the injected backend); real mode builds one
            // dedicated `LlmClient` per group member via the single `local_backend`
            // factory (DIP) and uses the classifier client as defense-in-depth
            // default for keys outside all groups.
            let target_matcher = if params.target_match == TargetMatchMode::SelfAssess {
                let backends = if injected_backend {
                    TargetBackends::new(HashMap::new(), Arc::clone(&client))
                } else {
                    TargetBackends::new(self.target_backends(), Arc::clone(&client))
                };
                tracing::debug!(
                    target: "router.config",
                    pipeline = %name,
                    target_backends = backends.len(),
                    target_match_timeout_ms = params.target_match_timeout_ms,
                    "target-matching ladder enabled (self-assess)",
                );
                Some(TargetMatcher::new(
                    backends,
                    Arc::clone(&limiter),
                    params.target_match_timeout_ms,
                ))
            } else {
                tracing::debug!(
                    target: "router.config",
                    pipeline = %name,
                    "target-matching ladder disabled (static)",
                );
                None
            };

            let stage = if let Some(tree) = &self.classification {
                // M4: classification tree drives the classifier stage. The
                // injected backend (mock/transcript) is always the default
                // client; per-node model backends are only built in real mode.
                // The target-matching ladder is shared with the flat path —
                // the engine resolves 2+ member group terminals through it.
                let engine = build_classification_engine(
                    self,
                    tree,
                    routing_config.clone(),
                    Arc::clone(&client),
                    Arc::clone(&limiter),
                    params.coherence_threshold,
                    !injected_backend,
                    target_matcher.clone(),
                );
                tracing::info!(
                    target: "router.config",
                    pipeline = %name,
                    tree_models = ?tree.classifier_model_keys(),
                    "classifier stage driven by classification tree",
                );
                crate::stages::classifier::ClassifierStage::with_tree(
                    client,
                    routing_config,
                    params.coherence_threshold,
                    params.score_matrix.clone(),
                    params.score_matrix_authoritative,
                    classifier_intel,
                    classifier_model,
                    limiter,
                    Arc::new(engine),
                    target_matcher,
                )
            } else {
                crate::stages::classifier::ClassifierStage::new(
                    client,
                    routing_config,
                    params.coherence_threshold,
                    params.score_matrix.clone(),
                    params.score_matrix_authoritative,
                    classifier_intel,
                    classifier_model,
                    limiter,
                    target_matcher,
                )
            };
            // M6: when configured, wrap the classifier in the retry decorator
            // so parse/LLM failures re-run with escalating corrective prompts.
            // `RetryClassifier` is a `Component`, so it pushes as
            // `Arc<dyn Component>`; it is deliberately NOT a
            // `StageDecisionProducer`, so the orchestrator consumes it through
            // the `WorkOutput` serialization boundary (one serialize/deserialize
            // per request) rather than the by-reference typed path.
            if params.classifier_retry_max > 0 {
                let retry_max = params.classifier_retry_max as usize;
                let retry_prompts = params.classifier_retry_prompts.clone();
                tracing::info!(
                    target: "router.config",
                    pipeline = %name,
                    classifier_retry_max = params.classifier_retry_max,
                    retry_prompt_count = retry_prompts.len(),
                    "classifier wrapped in RetryClassifier",
                );
                stages.push(Arc::new(
                    crate::stages::retry_classifier::RetryClassifier::new(
                        Arc::new(stage),
                        retry_max,
                        retry_prompts,
                    ),
                ));
            } else {
                stages.push(Arc::new(stage));
            }
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
/// 3. Root `classification` classifier node's `model` (M4.1 — tree configs
///    boot without a flat classifier key)
/// 4. First model in the `fast` model group
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
                .classification
                .as_ref()
                .and_then(super::ClassificationTree::root_classifier_model)
        })
        .or_else(|| {
            config
                .model_groups
                .get("fast")
                .and_then(|group| group.models().first())
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
    config.local_backend(model_key)
}

/// Build the M4 classification-tree engine for a pipeline.
///
/// `default_client` (the injected mock/transcript backend or the real
/// classifier client) serves every classifier node whose `model` key has no
/// dedicated backend. When `use_per_node_backends` is true (real mode only —
/// never when a backend was injected for mock/transcript runs), a dedicated
/// `LlmClient` is built for each distinct classifier-node model key that
/// differs from the resolved classifier model.
fn build_classification_engine(
    config: &RouterConfig,
    tree: &super::ClassificationTree,
    routing: super::RoutingConfig,
    default_client: Arc<dyn ChatBackend>,
    limiter: Arc<fluent_concurrency::pool::Limiter>,
    coherence_threshold: f64,
    use_per_node_backends: bool,
    target_matcher: Option<TargetMatcher>,
) -> crate::stages::tree::ClassificationEngine {
    let default_params = PipelineParams::default();
    let default_model_key = resolve_classifier_model_key(config, &default_params);
    let mut clients = HashMap::new();
    if use_per_node_backends {
        for key in tree.classifier_model_keys() {
            if default_model_key == Some(key.as_str()) {
                continue;
            }
            if let Some(backend) = config.local_backend(&key) {
                clients.insert(key, backend);
            }
        }
    }
    crate::stages::tree::ClassificationEngine::new(
        tree.clone(),
        routing,
        default_client,
        clients,
        limiter,
        coherence_threshold,
        target_matcher,
    )
}

impl RouterConfig {
    /// Build the escalation ladder for every model group that configures one
    /// (`model_groups[g].escalation`). Groups without a ladder (or without a
    /// frontier endpoint) are absent — dispatch falls back to
    /// `fallback_completion` as before.
    ///
    /// The ladders are keyed by group name; `RoutingTarget.group` resolves
    /// which one a failed local chain escalates through
    pub fn build_escalation_ladders(
        &self,
        http_client: &reqwest::Client,
    ) -> HashMap<String, Arc<crate::dispatch::escalation::EscalationLadder>> {
        use crate::dispatch::backend::OpenAiChatBackend;
        use crate::dispatch::escalation::{EscalationBackends, EscalationLadder};

        let mut ladders = HashMap::new();
        for (group, group_cfg) in &self.model_groups {
            let Some(ladder_cfg) = group_cfg.escalation() else {
                continue;
            };
            let Some(frontier) = &ladder_cfg.frontier else {
                continue;
            };
            let frontier_client = frontier_api_client(http_client, frontier.api_key_env.as_deref());
            let backends = EscalationBackends {
                frontier: Arc::new(OpenAiChatBackend::new(
                    frontier_client,
                    frontier.endpoint.clone(),
                )),
                decomposer: ladder_cfg
                    .decomposer_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                assembler: ladder_cfg
                    .assembler_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                classifier: ladder_cfg
                    .classifier_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                draft: ladder_cfg
                    .draft_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                judge: ladder_cfg
                    .judge_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
            };
            tracing::info!(
                target: "router.config",
                group = %group,
                modes = ?ladder_cfg.modes,
                frontier_model = %frontier.model,
                "escalation ladder built",
            );
            ladders.insert(
                group.clone(),
                Arc::new(EscalationLadder::new(ladder_cfg.clone(), backends)),
            );
        }
        ladders
    }

    /// Build a sync local-model `ChatBackend` from a `models` key — the single
    /// `LlmClient` construction site shared by the classifier and the
    /// escalation ladder's local roles (DIP: exactly one concrete
    /// `ChatBackend` factory in the crate). The model id is qualified to the
    /// entry's default dispatch point and declaration-only params are stripped.
    pub fn local_backend(&self, key: &str) -> Option<Arc<dyn ChatBackend>> {
        let entry = self.models.get(key)?;
        let base = entry.name.as_deref().unwrap_or(key);
        let model = match entry.default_dispatch_qualifier() {
            Some(qualifier) => format!("{base}:{qualifier}"),
            None => base.to_string(),
        };
        let params = entry.params.clone().map(strip_declaration_params);
        let llm_config = LlmConfig::new()
            .api_url(entry.endpoint.clone())
            .model(model)
            .timeout_ms(entry.total_timeout_ms)
            .maybe_extra_body_params(params)
            .build();
        Some(Arc::new(LlmClient::with_config(llm_config)))
    }

    /// Build a `ChatBackend` for a specific named inference point
    /// (`<base>:<instance_or_group>`) of a `models` key, reusing the single
    /// `LlmClient` factory (DIP — same construction site as `local_backend`).
    /// Used by the ledger summarizer (`<base>:ledger`) and any on-demand
    /// scratch route (`<base>:scratch`), which must target a named instance
    /// rather than the entry's default dispatch point.
    pub fn local_backend_for_instance(
        &self,
        key: &str,
        instance_or_group: &str,
    ) -> Option<Arc<dyn ChatBackend>> {
        let entry = self.models.get(key)?;
        let base = entry.name.as_deref().unwrap_or(key);
        let model = format!("{base}:{instance_or_group}");
        let params = entry.params.clone().map(strip_declaration_params);
        let llm_config = LlmConfig::new()
            .api_url(entry.endpoint.clone())
            .model(model)
            .timeout_ms(entry.total_timeout_ms)
            .maybe_extra_body_params(params)
            .build();
        Some(Arc::new(LlmClient::with_config(llm_config)))
    }

    /// Build the target-matching ladder's per-candidate backend set (DIP —
    /// reuses the private `local_backend` helper, the single `LlmClient`
    /// factory; no second construction site).
    ///
    /// Iterates every model key referenced by any `model_groups` member and
    /// maps it to its dedicated `ChatBackend`. The matcher's `default` (for
    /// keys absent from the map) is supplied by the caller: the injected
    /// mock/transcript backend when one is provided, otherwise a real client
    /// (defense in depth — every real group member has a dedicated backend,
    /// so the default is only reached for a key outside all groups).
    pub fn target_backends(&self) -> HashMap<String, Arc<dyn ChatBackend>> {
        let mut backends = HashMap::new();
        for group in self.model_groups.values() {
            for key in group.models() {
                if let Some(backend) = self.local_backend(key) {
                    backends.insert(key.clone(), backend);
                }
            }
        }
        backends
    }
}

/// A reqwest client for the frontier backend: the shared client by default,
/// or a per-ladder client carrying the `Bearer` token from `api_key_env`
/// (when the variable is set and resolvable).
fn frontier_api_client(shared: &reqwest::Client, api_key_env: Option<&str>) -> reqwest::Client {
    let Some(env) = api_key_env else {
        return shared.clone();
    };
    let Ok(key) = std::env::var(env) else {
        tracing::warn!(
            target: "router.config",
            env = %env,
            "frontier api_key_env set but unreadable — falling back to shared client (no auth header)",
        );
        return shared.clone();
    };
    let Ok(auth) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")) else {
        return shared.clone();
    };
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(reqwest::header::AUTHORIZATION, auth);
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .unwrap_or_else(|_| shared.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use common_core::sync::lock;

    use crate::charts::binding::Entity;
    use crate::charts::{ChartDef, ChartError};
    use crate::test_stubs::StubChatBackend;
    use crate::test_support::capture_logs;
    use fluent_concurrency::pool::Limiter;

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

    #[test]
    fn local_backend_for_instance_builds_ledger_and_scratch_backends() {
        // M5: the ledger summarizer and on-demand scratch route must dispatch
        // to their named instances. `local_backend_for_instance` builds an
        // `LlmClient` for the `models` key qualified to `<base>:<instance>`,
        // and `RoutingTarget::from_model_entry_instance` mirrors the model id.
        let config: RouterConfig = serde_json::from_value(serde_json::json!({
            "models": {
                "swarm": {
                    "endpoint": "http://x/v1/chat/completions",
                    "name": "abiray/lfm2.5-2.6b-heretic-abliterated",
                    "intelligence": 2,
                    "cost_input": 1.0, "cost_output": 6.0, "cost_cached_read": 0.4,
                    "speed": 8,
                    "instances": {
                        "ledger": { "num_ctx": 131072, "pinned": true, "default": true },
                        "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30 }
                    }
                }
            }
        })).expect("valid config");

        // The named-instance backends build (single LlmClient factory).
        assert!(config.local_backend_for_instance("swarm", "ledger").is_some());
        assert!(config.local_backend_for_instance("swarm", "scratch").is_some());

        // The canonical target builder confirms the exact model id each point
        // resolves to on the wire.
        let entry = config.models.get("swarm").expect("swarm");
        let ledger_rt =
            crate::pipeline::RoutingTarget::from_model_entry_instance("swarm", entry, "ledger");
        assert_eq!(
            ledger_rt.model,
            "abiray/lfm2.5-2.6b-heretic-abliterated:ledger"
        );
        assert_eq!(ledger_rt.instance.as_deref(), Some("ledger"));
        let scratch_rt =
            crate::pipeline::RoutingTarget::from_model_entry_instance("swarm", entry, "scratch");
        assert_eq!(
            scratch_rt.model,
            "abiray/lfm2.5-2.6b-heretic-abliterated:scratch"
        );
        assert_eq!(scratch_rt.instance.as_deref(), Some("scratch"));
    }

    #[test]
    fn target_backends_builds_every_group_member_key() {
        let config: RouterConfig = serde_json::from_str(
            r#"{
                "models": {
                    "swarm": {"endpoint": "http://a/v1/chat/completions", "name": "swarm", "intelligence": 2, "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.4, "speed": 8},
                    "qwen3.6-27b": {"endpoint": "http://b/v1/chat/completions", "name": "qwen3.6-27b", "intelligence": 6, "cost_input": 3.0, "cost_output": 3.0, "cost_cached_read": 1.0, "speed": 4},
                    "unused": {"endpoint": "http://c/v1/chat/completions", "name": "unused", "intelligence": 9, "cost_input": 9.0, "cost_output": 9.0, "cost_cached_read": 3.0, "speed": 2}
                },
                "model_groups": {
                    "default": ["swarm", "qwen3.6-27b"],
                    "translation": {"models": ["qwen3.6-27b"]}
                }
            }"#,
        )
        .expect("valid config");

        let backends = config.target_backends();
        // Exactly the model keys referenced by any model_groups member are
        // built (deduplicated across groups) — `unused` is not a group member.
        let mut keys: Vec<&str> = backends.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["qwen3.6-27b", "swarm"]);
    }

    #[test]
    fn builder_threads_target_match_timeout_ms_into_matcher() {
        // `target_match_timeout_ms` must flow from PipelineParams into the
        // TargetMatcher's per-assessment budget (M6). The builder logs the
        // value it passes on the self-assess path; assert it is the configured
        // knob, not the hardcoded constant.
        let config: RouterConfig = serde_json::from_str(
            r#"{
                "pipelines": {
                    "default": {
                        "classifier": true,
                        "classifier_model": "fast",
                        "target_match": "self_assess",
                        "target_match_timeout_ms": 4321
                    }
                },
                "models": {
                    "fast": {"endpoint": "http://a/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.4, "speed": 10},
                    "swarm": {"endpoint": "http://b/v1/chat/completions", "name": "swarm", "intelligence": 2, "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.4, "speed": 9},
                    "qwen3.6-27b": {"endpoint": "http://c/v1/chat/completions", "name": "qwen3.6-27b", "intelligence": 6, "cost_input": 5.0, "cost_output": 5.0, "cost_cached_read": 2.0, "speed": 4}
                },
                "model_groups": {
                    "default": ["swarm", "qwen3.6-27b"]
                },
                "routes": {
                    "code": {"group": "default", "pipelines": ["default"]}
                },
                "default_route": "fast"
            }"#,
        )
        .expect("valid config");
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always("{}"));

        let (pipeline, logs) = capture_logs(|| {
            config
                .build_named_pipeline_with_backend("default", Some(Arc::clone(&backend)))
                .expect("pipeline builds")
        });
        let _ = pipeline;
        let joined = logs.join("\n");
        assert!(
            joined.contains("target_match_timeout_ms=4321"),
            "builder must thread the configured per-assessment timeout, got:\n{joined}"
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
