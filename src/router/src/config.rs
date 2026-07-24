//! Router configuration types — deserialized from JSON via `common_core::config`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use common_core::config::load_json_or_default;
use fluent_wvr::prelude::Component;
use guidance_llm::client::ChatBackend;
use guidance_llm::{LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};

use crate::logging::LoggingConfig;
use crate::pipeline::PipelineOrchestrator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Pipeline definitions keyed by name.
    #[serde(default)]
    pub pipelines: HashMap<String, PipelineParams>,
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
    #[serde(default)]
    pub model_groups: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub routes: HashMap<String, RouteRef>,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub safety_threshold: f64,
    #[serde(default = "default_fast_route")]
    pub default_route: String,
    #[serde(default = "ServerConfig::default")]
    pub server: ServerConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Root-level classifier model name.  Used by all pipelines that do not
    /// set their own `classifier_model`.  Falls back to the first model in the
    /// `"fast"` model group when unset.
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Mock mode configuration. When set, the server runs with a transcript
    /// provider instead of real LLM calls, and validates routing decisions.
    #[serde(default)]
    pub mock: Option<MockConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockConfig {
    /// Path to a JSON transcript file containing mock test cases.
    pub transcript_path: String,
    /// Whether to fail (exit non-zero) on unexpected routing decisions.
    #[serde(default = "default_true")]
    pub fail_on_unexpected: bool,
    /// Base URL for mock LLM dispatch responses (default: http://127.0.0.1:8081).
    #[serde(default = "default_mock_base_url")]
    pub base_url: String,
}

fn default_mock_base_url() -> String {
    "http://127.0.0.1:8081".into()
}

impl Default for RouterConfig {
    fn default() -> Self {
        let mut pipelines = HashMap::new();
        pipelines.insert("default".into(), PipelineParams::default());
        Self {
            pipelines,
            models: HashMap::new(),
            model_groups: HashMap::new(),
            routes: HashMap::new(),
            system_prompt: String::new(),
            safety_threshold: 0.5,
            default_route: "fast".into(),
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
            classifier_model: None,
            mock: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,
    #[serde(default = "default_max_payload")]
    pub max_payload: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_bind_addr(),
            max_payload: default_max_payload(),
        }
    }
}

fn default_bind_addr() -> String {
    "127.0.0.1:8080".into()
}
fn default_max_payload() -> usize {
    1048576
}

/// Named pipeline parameters. Pipelines are stored as a map keyed by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Pipeline-level classifier model name.  When set, overrides the
    /// root-level `classifier_model`.  Falls back to root `classifier_model`
    /// then to the `"fast"` model group when unset.
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Path to a reject-patterns JSON file. When set, the patterns from that
    /// file are used as a blacklist (matches are rejected).
    #[serde(default)]
    pub blacklist: Option<String>,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            deterministic_prefilter: true,
            classifier: true,
            router: true,
            coherence_threshold: default_coherence_threshold(),
            classifier_model: None,
            blacklist: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    #[serde(default)]
    pub name: Option<String>,
    pub endpoint: String,
    /// Model capability level 0-10. Used by the router to match against
    /// the request complexity score emitted by the classifier.
    pub intelligence: u8,
    pub cost_input: f64,
    pub cost_output: f64,
    pub cost_cached_read: f64,
    pub speed: u8,
    #[serde(default = "default_total_timeout_ms")]
    pub total_timeout_ms: u64,
    #[serde(default = "default_idle_timeout_ms")]
    pub idle_timeout_ms: u64,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub filter_thinking: bool,
    #[serde(default)]
    pub retry_count: u32,
    #[serde(default = "default_retry_interval")]
    pub retry_base_interval_s: u64,
    /// Arbitrary inference parameters merged into the request body
    /// (e.g. stop, num_ctx, repeat_penalty, top_k, top_p, n_gpu_layers, etc.).
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRef {
    pub group: String,
    /// Pipeline names to execute in sequence for this route.
    #[serde(default = "default_pipelines")]
    pub pipelines: Vec<String>,
}

fn default_pipelines() -> Vec<String> {
    vec!["default".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EvictionPolicy {
    #[default]
    Lru,
    Ttl,
    Hybrid,
}

// ── Reject Patterns ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RejectPatterns {
    #[serde(default)]
    pub patterns: Vec<PatternEntry>,
    #[serde(default)]
    pub commands: Option<CommandConfig>,
}

/// A single pattern entry from a reject-patterns JSON file.
/// These are loaded from file and used as blacklist entries when
/// a pipeline's `blacklist` field references the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEntry {
    pub name: String,
    pub http_code: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default)]
    pub regexes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandConfig {
    pub pattern: String,
    #[serde(default)]
    pub handlers: HashMap<String, String>,
}

// ── Classifier output ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierOutput {
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub coherence_score: f64,
    pub safety_score: f64,
    /// Query complexity from 0 (trivial) to 10 (very complex).
    /// Used by the router to select a model with matching capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub complexity: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    pub reason: String,
}

// ── Resolved routing target ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub routes: HashMap<String, RouteRef>,
    pub models: HashMap<String, ModelEntry>,
    pub model_groups: HashMap<String, Vec<String>>,
    pub system_prompt: String,
    pub safety_threshold: f64,
    pub default_route: String,
}

impl RoutingConfig {
    pub fn resolve_route(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Option<(&ModelEntry, String)> {
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route));

        let route_ref = match route_ref {
            Some(r) => Some(r),
            None => {
                return self.models.get(route_name).map(|entry| {
                    let name = entry.name.clone().unwrap_or_else(|| route_name.to_string());
                    tracing::info!(target: "router.config", route = %route_name, model = %name,
                        "route resolved as direct model"
                    );
                    (entry, name)
                }).or_else(|| {
                    tracing::warn!(target: "router.config", route = %route_name,
                        default = %self.default_route,
                        "no route or model found for target"
                    );
                    None
                });
            }
        };

        let route_ref = route_ref?;

        let model_names = self.model_groups.get(route_ref.group.as_str());
        let Some(model_names) = model_names else {
            tracing::warn!(target: "router.config", route = %route_name, group = %route_ref.group, "model group not found for route");
            return None;
        };

        tracing::debug!(target: "router.config",
            route = %route_name,
            group = %route_ref.group,
            model_count = model_names.len(),
            min_complexity = ?min_complexity,
            "resolving route"
        );

        let candidates: Vec<(&String, &ModelEntry)> = model_names
            .iter()
            .filter_map(|n| self.models.get(n).map(|m| (n, m)))
            .filter(|(_, m)| {
                m.intelligence >= min_complexity.unwrap_or(0)
            })
            .collect();

        if candidates.is_empty() {
            // Fall back to any model in the group if complexity filter eliminates all.
            tracing::debug!(target: "router.config", route = %route_name, "no candidates passed complexity filter, falling back to cheapest in group");
            model_names
                .iter()
                .filter_map(|n| self.models.get(n).map(|m| (n, m)))
                .min_by(|(_, a), (_, b)| {
                    (a.cost_input + a.cost_output)
                        .partial_cmp(&(b.cost_input + b.cost_output))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(entry_key, entry)| {
                    let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
                    tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved (cheapest fallback)");
                    (entry, name)
                })
        } else {
            let (entry_key, entry) = candidates
                .into_iter()
                .min_by(|(_, a), (_, b)| {
                    (a.cost_input + a.cost_output)
                        .partial_cmp(&(b.cost_input + b.cost_output))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;
            let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
            tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved");
            Some((entry, name))
        }
    }
}

impl RouterConfig {
    pub fn load_reject_patterns(path: &str) -> RejectPatterns {
        load_json_or_default::<RejectPatterns>(Path::new(path))
    }

    pub fn routing_config(&self) -> RoutingConfig {
        RoutingConfig {
            routes: self.routes.clone(),
            models: self.models.clone(),
            model_groups: self.model_groups.clone(),
            system_prompt: self.system_prompt.clone(),
            safety_threshold: self.safety_threshold,
            default_route: self.default_route.clone(),
        }
    }

    /// Build stages for a single named pipeline.
    pub fn build_named_pipeline(&self, name: &str) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend(name, None)
    }

    /// Build stages for a single named pipeline, optionally injecting a mock
    /// backend for the classifier stage. When `classifier_backend` is `None`,
    /// the real `LlmClient` is constructed from the config.
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
            let client: Arc<dyn ChatBackend> = if let Some(backend) = classifier_backend {
                tracing::info!(target: "router.config", pipeline = %name, backend = "mock/transcript", "classifier using injected backend");
                backend
            } else {
                // Priority: pipeline classifier_model > root classifier_model > "fast" group
                let classifier_key = params
                    .classifier_model
                    .as_ref()
                    .or(self.classifier_model.as_ref())
                    .or_else(|| {
                        self.model_groups
                            .get("fast")
                            .and_then(|names| names.first())
                    });
                let classifier_entry = classifier_key.and_then(|k| self.models.get(k));
                let (entry, model_key) = if let Some(e) = classifier_entry {
                    let key = classifier_key.unwrap();
                    (e, key.as_str())
                } else {
                    tracing::error!(target: "router.config", pipeline = %name, pipeline_model = ?params.classifier_model, root_model = ?self.classifier_model, "no classifier model found in config");
                    return None;
                };
                let model_name_for_llm = entry.name.as_deref().unwrap_or(model_key);
                tracing::info!(target: "router.config", pipeline = %name, classifier_url = %entry.endpoint, model_name = %model_name_for_llm, "classifier using real LLM client");
                let classifier_config = LlmConfig::new()
                    .api_url(entry.endpoint.clone())
                    .model(model_name_for_llm.to_string())
                    .timeout_ms(5000)
                    .build();
                Arc::new(LlmClient::with_config(classifier_config))
            };
            stages.push(Arc::new(
                crate::stages::classifier::ClassifierStage::new(
                    client,
                    routing_config,
                    params.coherence_threshold,
                ),
            ));
        } else if classifier_backend.is_some() {
            tracing::warn!(
                target: "router.config",
                pipeline = %name,
                "classifier backend was provided but classifier is disabled for this pipeline"
            );
        }

        Some(crate::pipeline::PipelineOrchestrator::new(stages))
    }

    /// Build all named pipelines defined in the config.
    pub fn build_all_pipelines(&self) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend(None)
    }

    /// Build all named pipelines with an optional classifier backend injection.
    pub fn build_all_pipelines_with_backend(
        &self,
        classifier_backend: Option<&Arc<dyn ChatBackend>>,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        let mut map = HashMap::new();
        let pipeline_count = self.pipelines.len();
        let has_mock = classifier_backend.is_some();
        tracing::info!(target: "router.config", pipeline_count = pipeline_count, mock_backend = has_mock, classifier_model = ?self.classifier_model, default_route = %self.default_route, "building pipelines");
        for name in self.pipelines.keys() {
            let backend_for_pipeline = classifier_backend.cloned();
            if let Some(pipeline) =
                self.build_named_pipeline_with_backend(name, backend_for_pipeline)
            {
                map.insert(name.clone(), Arc::new(pipeline));
            }
        }
        tracing::info!(target: "router.config", built = map.len(), "pipelines built");
        map
    }

    /// Return the pipeline names to execute for a given route (model name).
    pub fn route_pipeline_names(&self, model_name: &str) -> Vec<String> {
        self.routes
            .get(model_name)
            .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone())
    }
}

use common_core::constants::default_true;

fn default_coherence_threshold() -> f64 {
    0.70
}
fn default_fast_route() -> String {
    "fast".into()
}
fn default_total_timeout_ms() -> u64 {
    300_000
}
fn default_idle_timeout_ms() -> u64 {
    30_000
}
fn default_retry_interval() -> u64 {
    1
}
