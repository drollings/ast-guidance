//! Router configuration types — deserialized from JSON via `common_core::config`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use common_core::config::load_json_or_default;
use fluent_wvr::prelude::Component;
use guidance_llm::LlmConfig;
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
    #[serde(default = "default_generic")]
    pub classifier_group: String,
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
            classifier_group: "fast".into(),
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
            .or_else(|| self.routes.get(&self.default_route))?;
        let model_names = self.model_groups.get(route_ref.group.as_str())?;
        let candidates: Vec<(&String, &ModelEntry)> = model_names
            .iter()
            .filter_map(|n| self.models.get(n).map(|m| (n, m)))
            .filter(|(_, m)| {
                m.intelligence >= min_complexity.unwrap_or(0)
            })
            .collect();
        if candidates.is_empty() {
            // Fall back to any model in the group if complexity filter eliminates all.
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

        let classifier_llm_config = self
            .model_groups
            .get(params.classifier_group.as_str())
            .and_then(|names| names.first())
            .and_then(|n| {
                self.models
                    .get(n)
                    .map(|m| (m.endpoint.clone(), m.total_timeout_ms))
            })
            .map(|(endpoint, timeout_ms)| {
                LlmConfig::new()
                    .api_url(endpoint)
                    .model("classifier".into())
                    .timeout_ms(timeout_ms.max(5000))
                    .build()
            });

        if params.classifier {
            if let Some(ref cfg) = classifier_llm_config {
                let routing_config = self.routing_config();
                let classifier_config = LlmConfig::new()
                    .api_url(cfg.api_url.clone())
                    .model(cfg.model.clone())
                    .timeout_ms(5000)
                    .build();
                stages.push(Arc::new(
                    crate::stages::classifier::ClassifierStage::new(
                        classifier_config,
                        routing_config,
                        params.coherence_threshold,
                    ),
                ));
            }
        }

        if params.router {
            let policy = crate::stages::router::RoutingPolicy::LocalFirst;
            stages.push(Arc::new(crate::stages::router::RouterStage::new(policy)));
        }

        Some(crate::pipeline::PipelineOrchestrator::new(stages))
    }

    /// Build all named pipelines defined in the config.
    pub fn build_all_pipelines(&self) -> HashMap<String, Arc<PipelineOrchestrator>> {
        let mut map = HashMap::new();
        for name in self.pipelines.keys() {
            if let Some(pipeline) = self.build_named_pipeline(name) {
                map.insert(name.clone(), Arc::new(pipeline));
            }
        }
        map
    }

    /// Return the pipeline names to execute for a given route (model name).
    pub fn route_pipeline_names(&self, model_name: &str) -> Vec<String> {
        self.routes
            .get(model_name)
            .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone())
    }
}

fn default_true() -> bool {
    true
}
fn default_coherence_threshold() -> f64 {
    0.70
}
fn default_fast_route() -> String {
    "fast".into()
}
fn default_generic() -> String {
    String::new()
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
