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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouterConfig {
    #[serde(default = "PipelineConfig::default")]
    pub pipeline: PipelineConfig,
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
    #[serde(default)]
    pub model_groups: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub adapters: Vec<AdapterEntry>,
    #[serde(default)]
    pub routes: HashMap<String, RouteRef>,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub safety_threshold: f64,
    #[serde(default = "default_fast_route")]
    pub default_route: String,
    #[serde(default = "SessionConfig::default")]
    pub sessions: SessionConfig,
    #[serde(default = "KvCacheConfig::default")]
    pub kv_cache: KvCacheConfig,
    #[serde(default = "WatchdogConfig::default")]
    pub watchdogs: WatchdogConfig,
    #[serde(default = "GuardrailConfig::default")]
    pub guardrails: GuardrailConfig,
    #[serde(default = "ServerConfig::default")]
    pub server: ServerConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub reject_patterns_path: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct PipelineConfig {
    #[serde(default = "default_true")]
    pub deterministic_prefilter: bool,
    #[serde(default = "default_true")]
    pub classifier: bool,
    #[serde(default = "default_true")]
    pub router: bool,
    #[serde(default = "default_coherence_threshold")]
    pub coherence_threshold: f64,
    #[serde(default)]
    pub guardrail_mode: GuardrailMode,
    #[serde(default = "default_generic")]
    pub classifier_group: String,
    #[serde(default = "default_generic")]
    pub frontier_group: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            deterministic_prefilter: true,
            classifier: true,
            router: true,
            coherence_threshold: default_coherence_threshold(),
            guardrail_mode: GuardrailMode::default(),
            classifier_group: default_generic(),
            frontier_group: default_generic(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GuardrailMode {
    #[default]
    FrontierOnly,
    Always,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    #[serde(default)]
    pub name: Option<String>,
    pub endpoint: String,
    pub intelligence: u8,
    pub cost_input: f64,
    pub cost_output: f64,
    pub cost_cached_read: f64,
    pub speed: u8,
    #[serde(default = "default_total_timeout_ms")]
    pub total_timeout_ms: u64,
    #[serde(default = "default_idle_timeout_ms")]
    pub idle_timeout_ms: u64,
    #[serde(default)]
    pub context_size: Option<usize>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterEntry {
    pub name: String,
    pub path: String,
    pub base_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRef {
    pub group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionConfig {
    #[serde(default)]
    pub compaction_policy: CompactionPolicy,
    #[serde(default = "default_max_nodes")]
    pub max_nodes_before_compaction: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CompactionPolicy {
    #[default]
    Recency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    #[serde(default = "default_hot_mb")]
    pub hot_tier_mb: usize,
    #[serde(default = "default_cold_mount")]
    pub cold_tier_mount: String,
    #[serde(default = "default_cold_mb")]
    pub cold_tier_max_mb: usize,
    #[serde(default = "default_ttl")]
    pub cold_tier_ttl_secs: u64,
    #[serde(default)]
    pub cold_tier_eviction: EvictionPolicy,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            hot_tier_mb: default_hot_mb(),
            cold_tier_mount: default_cold_mount(),
            cold_tier_max_mb: default_cold_mb(),
            cold_tier_ttl_secs: default_ttl(),
            cold_tier_eviction: EvictionPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EvictionPolicy {
    #[default]
    Lru,
    Ttl,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_wall_clock")]
    pub wall_clock_secs: u64,
    #[serde(default = "default_repeat_threshold")]
    pub repeat_threshold: usize,
    #[serde(default = "default_repeat_window")]
    pub repeat_window: usize,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            wall_clock_secs: default_wall_clock(),
            repeat_threshold: default_repeat_threshold(),
            repeat_window: default_repeat_window(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuardrailConfig {
    #[serde(default = "default_pii_classes")]
    pub pii_classes: Vec<String>,
    #[serde(default)]
    pub blocked_topics: Vec<String>,
    #[serde(default)]
    pub check_local_agents: bool,
}

// ── Reject Patterns ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RejectPatterns {
    #[serde(default)]
    pub blacklist: Vec<BlacklistEntry>,
    #[serde(default)]
    pub commands: Option<CommandConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlacklistEntry {
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
    pub fn resolve_route(&self, route_name: &str) -> Option<(&ModelEntry, String)> {
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route))?;
        let model_names = self.model_groups.get(route_ref.group.as_str())?;
        let (entry_key, entry) = model_names
            .iter()
            .filter_map(|n| self.models.get(n).map(|m| (n, m)))
            .min_by(|(_, a), (_, b)| {
                (a.cost_input + a.cost_output)
                    .partial_cmp(&(b.cost_input + b.cost_output))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
        let name = entry
            .name
            .clone()
            .unwrap_or_else(|| entry_key.clone());
        Some((entry, name))
    }
}

impl RouterConfig {
    pub fn reject_patterns(&self) -> RejectPatterns {
        self.reject_patterns_path
            .as_deref()
            .map(|p| load_json_or_default::<RejectPatterns>(Path::new(p)))
            .unwrap_or_default()
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

    pub fn build_pipeline(&self) -> PipelineOrchestrator {
        let mut stages: Vec<Arc<dyn Component>> = Vec::new();

        if self.pipeline.deterministic_prefilter {
            let reject_patterns = self.reject_patterns();
            stages.push(Arc::new(
                crate::stages::deterministic::DeterministicPreFilter::from_config(
                    &reject_patterns,
                ),
            ));
        }

        let classifier_llm_config = self
            .model_groups
            .get(self.pipeline.classifier_group.as_str())
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

        if self.pipeline.classifier {
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
                        self.pipeline.coherence_threshold,
                    ),
                ));
            }
        }

        if self.pipeline.router {
            let policy = if self
                .model_groups
                .get(self.pipeline.frontier_group.as_str())
                .is_none_or(std::vec::Vec::is_empty)
            {
                crate::stages::router::RoutingPolicy::LocalFirst
            } else {
                crate::stages::router::RoutingPolicy::FrontierOnly
            };
            stages.push(Arc::new(crate::stages::router::RouterStage::new(policy)));
        }

        crate::pipeline::PipelineOrchestrator::new(stages)
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
fn default_max_nodes() -> usize {
    100
}
fn default_hot_mb() -> usize {
    4096
}
fn default_cold_mount() -> String {
    "/tmp/coral-kv-cache".into()
}
fn default_cold_mb() -> usize {
    32_768
}
fn default_ttl() -> u64 {
    86_400
}
fn default_max_tokens() -> u32 {
    4096
}
fn default_wall_clock() -> u64 {
    300
}
fn default_repeat_threshold() -> usize {
    50
}
fn default_repeat_window() -> usize {
    100
}
fn default_pii_classes() -> Vec<String> {
    vec![
        "ssn".into(),
        "card_number".into(),
        "email".into(),
        "phone".into(),
    ]
}
