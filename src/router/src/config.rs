//! Router configuration types — deserialized from JSON via `common_core::config`.

use std::sync::Arc;

use fluent_wvr::prelude::Component;
use guidance_llm::LlmConfig;
use serde::{Deserialize, Serialize};

use crate::logging::LoggingConfig;
use crate::pipeline::PipelineOrchestrator;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouterConfig {
    #[serde(default = "PipelineConfig::default")]
    pub pipeline: PipelineConfig,
    #[serde(default = "ModelCatalog::default")]
    pub models: ModelCatalog,
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
    pub quality_gate: bool,
    #[serde(default = "default_true")]
    pub planning_refinement: bool,
    #[serde(default = "default_true")]
    pub guardrail_check: bool,
    #[serde(default = "default_true")]
    pub router: bool,
    #[serde(default = "default_quality_threshold")]
    pub quality_threshold: f64,
    #[serde(default)]
    pub guardrail_mode: GuardrailMode,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            deterministic_prefilter: true,
            quality_gate: true,
            planning_refinement: true,
            guardrail_check: true,
            router: true,
            quality_threshold: default_quality_threshold(),
            guardrail_mode: GuardrailMode::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GuardrailMode {
    #[default]
    FrontierOnly,
    Always,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCatalog {
    #[serde(default)]
    pub orchestrators: Vec<ModelEntry>,
    #[serde(default)]
    pub agents: Vec<ModelEntry>,
    #[serde(default)]
    pub frontier: Vec<FrontierEntry>,
    #[serde(default)]
    pub adapters: Vec<AdapterEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierEntry {
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterEntry {
    pub name: String,
    pub path: String,
    pub base_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default = "default_orchestrator_ctx")]
    pub orchestrator_context_size: usize,
    #[serde(default = "default_agent_ctx")]
    pub agent_context_size: usize,
    #[serde(default)]
    pub compaction_policy: CompactionPolicy,
    #[serde(default = "default_max_nodes")]
    pub max_nodes_before_compaction: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            orchestrator_context_size: default_orchestrator_ctx(),
            agent_context_size: default_agent_ctx(),
            compaction_policy: CompactionPolicy::default(),
            max_nodes_before_compaction: default_max_nodes(),
        }
    }
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

// ── Pipeline construction ──────────────────────────────────────────────

impl RouterConfig {
    /// Build a `PipelineOrchestrator` from the config.
    ///
    /// Stages that require an LLM (quality gate, planning, guardrail) are
    /// only included when `PipelineConfig` enables them AND a model entry
    /// exists in the catalog. The deterministic pre-filter and router
    /// stages are always included when enabled (they need no LLM).
    pub fn build_pipeline(&self) -> PipelineOrchestrator {
        let mut stages: Vec<Arc<dyn Component>> = Vec::new();

        // Stage 1: deterministic pre-filter (no LLM)
        if self.pipeline.deterministic_prefilter {
            stages.push(Arc::new(
                crate::stages::deterministic::DeterministicPreFilter::new(),
            ));
        }

        // Derive an LlmConfig from the first orchestrator or frontier entry
        let llm_config = self
            .models
            .orchestrators
            .first()
            .map(|m| {
                LlmConfig::new()
                    .api_url(m.path.clone())
                    .model(m.name.clone())
                    .build()
            })
            .or_else(|| {
                self.models.frontier.first().map(|f| {
                    LlmConfig::new()
                        .api_url(
                            f.api_base
                                .clone()
                                .unwrap_or_else(|| "http://localhost:11434/v1".into()),
                        )
                        .model(f.model.clone())
                        .build()
                })
            });

        if let Some(ref cfg) = llm_config {
            // Stage 2: quality gate
            if self.pipeline.quality_gate {
                stages.push(Arc::new(
                    crate::stages::quality_gate::QualityGate::new(
                        cfg.clone(),
                        self.pipeline.quality_threshold,
                    ),
                ));
            }
            // Stage 3: planning refinement
            if self.pipeline.planning_refinement {
                stages.push(Arc::new(
                    crate::stages::planning::PlanningRefinementAgent::new(
                        cfg.clone(),
                        true,
                    ),
                ));
            }
            // Stage 4: guardrail check
            if self.pipeline.guardrail_check {
                stages.push(Arc::new(
                    crate::stages::guardrail::GuardrailCheck::new(
                        cfg.clone(),
                        self.guardrails.pii_classes.clone(),
                        self.guardrails.check_local_agents,
                    ),
                ));
            }
        }

        // Stage 5: router stage (no LLM — policy-based decision)
        if self.pipeline.router {
            let policy = if self.models.frontier.is_empty() {
                crate::stages::router::RoutingPolicy::LocalFirst
            } else {
                crate::stages::router::RoutingPolicy::FrontierOnly
            };
            stages.push(Arc::new(
                crate::stages::router::RouterStage::new(policy),
            ));
        }

        crate::pipeline::PipelineOrchestrator::new(stages)
    }
}

fn default_true() -> bool {
    true
}
fn default_quality_threshold() -> f64 {
    0.7
}
fn default_orchestrator_ctx() -> usize {
    131_072
}
fn default_agent_ctx() -> usize {
    65_536
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