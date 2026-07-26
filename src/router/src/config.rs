//! Router configuration types — deserialized from JSON via `common_core::config`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common_core::config::load_json_or_default;
use fluent_wvr::prelude::Component;
use guidance_llm::client::ChatBackend;
use guidance_llm::{LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};

use crate::logging::LoggingConfig;
use crate::pipeline::PipelineOrchestrator;
use crate::score_matrix::ScoreMatrix;

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
    /// Score-matrix routing configuration. When set, routing decisions use
    /// weighted scoring across coherence/complexity/completeness/risk dimensions.
    /// Overridable per-pipeline via `PipelineParams::score_matrix`.
    #[serde(default)]
    pub score_matrix: Option<ScoreMatrix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockConfig {
    /// Path to a JSON transcript file containing mock test cases.
    pub transcript_path: String,
    /// Whether to fail (exit non-zero) on unexpected routing decisions.
    #[serde(default = "default_true")]
    pub fail_on_unexpected: bool,
    /// Base URL for mock LLM dispatch responses (must be set via config file or CLI).
    #[serde(default = "default_mock_base_url")]
    pub base_url: String,
}

fn default_mock_base_url() -> String {
    String::new()
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
            score_matrix: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub bind_addr: String,
    #[serde(default = "default_max_payload")]
    pub max_payload: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: String::new(),
            max_payload: default_max_payload(),
        }
    }
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
    /// Score-matrix routing configuration (MOA_ROUTER_SPEC §2.2).
    /// When set, the classifier uses the weighted score matrix to resolve
    /// routing decisions (plan/rigor/local) instead of single-threshold logic.
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
            blacklist: None,
            score_matrix: None,
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
    /// Named session profiles for this model (e.g. "orchestrator", "code", "compact").
    /// Each profile overrides base params with session-specific settings.
    #[serde(default)]
    pub sessions: Option<HashMap<String, SessionProfile>>,
}

/// Session-specific parameter overrides for a model entry.
/// When a routing decision resolves to a named session profile,
/// the profile's values override the model's base `params`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionProfile {
    /// Context window size for this session profile (tokens).
    pub num_ctx: u64,
    /// Seconds of inactivity before the model is unloaded.
    /// 0 = never sleep (always resident).
    #[serde(default)]
    pub sleep_idle_seconds: Option<u64>,
    /// Additional session-specific params merged over the model's base params.
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

// ── Audit log config ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogConfig {
    #[serde(default = "default_audit_log_dir")]
    pub log_dir: PathBuf,
    #[serde(default = "default_audit_file_size_mb")]
    pub max_file_size_mb: u64,
    #[serde(default = "default_audit_age_days")]
    pub max_age_days: u64,
    #[serde(default = "default_audit_max_files")]
    pub max_files: usize,
    #[serde(default)]
    pub json_format: bool,
    #[serde(default)]
    pub console_output: bool,
}

fn default_audit_log_dir() -> PathBuf {
    PathBuf::from("/tmp/coral-router-audit-logs")
}

const fn default_audit_file_size_mb() -> u64 {
    50
}

const fn default_audit_age_days() -> u64 {
    90
}

const fn default_audit_max_files() -> usize {
    20
}

impl Default for AuditLogConfig {
    fn default() -> Self {
        Self {
            log_dir: default_audit_log_dir(),
            max_file_size_mb: default_audit_file_size_mb(),
            max_age_days: default_audit_age_days(),
            max_files: default_audit_max_files(),
            json_format: true,
            console_output: false,
        }
    }
}

// ── Filter types (MOA_ROUTER_SPEC §2) ─────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterOutcome {
    #[default]
    HardReject,
    SoftRedirect,
    OutputFilter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterAction {
    Redact,
    Anonymize,
    Omit,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceGate {
    #[serde(rename = "luhn_valid")]
    LuhnValid,
    #[default]
    #[serde(rename = "none")]
    None,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterScope {
    #[default]
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "frontier_bound")]
    FrontierBound,
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
    #[serde(default)]
    pub outcome: FilterOutcome,
    #[serde(default)]
    pub filter_action: Option<FilterAction>,
    #[serde(default)]
    pub confidence_gate: ConfidenceGate,
    #[serde(default)]
    pub scope: Vec<FilterScope>,
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
    /// How complete/well-specified the request is (0.0–1.0). Low values
    /// trigger the `plan` route for clarification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completeness: Option<f64>,
    /// Perceived risk/sensitivity score (0.0–1.0). High values trigger
    /// the `rigor` route for adversarial validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk: Option<f64>,
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
    /// Score-matrix routing configuration. When set, routing decisions use
    /// weighted scoring across coherence/complexity/completeness/risk dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_matrix: Option<ScoreMatrix>,
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
            score_matrix: self.score_matrix.clone(),
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
                    .timeout_ms(entry.total_timeout_ms)
                    .maybe_extra_body_params(entry.params.clone())
                    .build();
                Arc::new(LlmClient::with_config(classifier_config))
            };
            stages.push(Arc::new(
                crate::stages::classifier::ClassifierStage::new(
                    client,
                    routing_config,
                    params.coherence_threshold,
                    params.score_matrix.clone(),
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

/// Resolve a hostname to its canonical comparison form.
/// Returns `true` if two hosts should be considered equivalent
/// (e.g. `"localhost"` and `"127.0.0.1"`).
fn normalize_host(h: &str) -> String {
    match h.trim() {
        "localhost" | "127.0.0.1" | "::1" => "127.0.0.1".into(),
        other => other.to_string(),
    }
}

pub fn hosts_equivalent(a: &str, b: &str) -> bool {
    normalize_host(a) == normalize_host(b)
}

/// Parse a `host:port` string into its components.
pub fn parse_bind_addr(addr: &str) -> Result<(&str, u16), String> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err("bind_addr is empty".into());
    }
    // Handle IPv6: [::1]:port
    if addr.starts_with('[') {
        let close_bracket = addr.rfind(']').ok_or_else(|| "unclosed '[' in bind_addr".to_string())?;
        let host = &addr[1..close_bracket];
        let rest = addr[close_bracket + 1..].trim_start_matches(':');
        let port: u16 = rest.parse().map_err(|e| format!("invalid port in bind_addr '{addr}': {e}"))?;
        return Ok((host, port));
    }
    if let Some(colon_pos) = addr.rfind(':') {
        let host = &addr[..colon_pos];
        let port: u16 = addr[colon_pos + 1..]
            .parse()
            .map_err(|e| format!("invalid port in bind_addr '{addr}': {e}"))?;
        Ok((host, port))
    } else {
        Err(format!("bind_addr '{addr}' missing port (expected host:port)"))
    }
}

/// Validate that none of the configured model endpoints point back at the
/// router's own bind address.  This prevents accidental self-routing loops.
#[allow(clippy::implicit_hasher)]
pub fn validate_no_self_routing(
    bind_addr: &str,
    models: &std::collections::HashMap<String, ModelEntry>,
) -> Result<(), String> {
    if bind_addr.is_empty() {
        return Err("server.bind_addr must be set".into());
    }
    let (my_host, my_port) = parse_bind_addr(bind_addr)?;

    for (name, entry) in models {
        let url = entry.endpoint.trim();
        // Parse host:port from the endpoint URL
        let url_no_scheme = url
            .strip_prefix("http://")
            .or_else(|| url.strip_prefix("https://"))
            .unwrap_or(url);
        let (host, port) = match url_no_scheme.find('/') {
            Some(pos) => {
                let hp = &url_no_scheme[..pos];
                parse_bind_addr(hp)?
            }
            None => parse_bind_addr(url_no_scheme)?,
        };

        if hosts_equivalent(host, my_host) && port == my_port {
            return Err(format!(
                "model '{name}' endpoint ({}) points to the router's own bind address ({}) — would create a routing loop",
                entry.endpoint, bind_addr
            ));
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn hosts_equivalent_same_host() {
        assert!(hosts_equivalent("localhost", "localhost"));
    }

    #[test]
    fn hosts_equivalent_localhost_and_ip() {
        assert!(hosts_equivalent("localhost", "127.0.0.1"));
    }

    #[test]
    fn hosts_equivalent_ipv6_local() {
        assert!(hosts_equivalent("::1", "127.0.0.1"));
    }

    #[test]
    fn hosts_equivalent_different_hosts() {
        assert!(!hosts_equivalent("upstream.test", "127.0.0.1"));
        assert!(!hosts_equivalent("0.0.0.0", "127.0.0.1"));
    }

    #[test]
    fn hosts_equivalent_works_with_whitespace() {
        assert!(hosts_equivalent("  localhost  ", "127.0.0.1"));
    }

    #[test]
    fn parse_bind_addr_simple() {
        let (host, port) = parse_bind_addr("0.0.0.0:8079").unwrap();
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 8079);
    }

    #[test]
    fn parse_bind_addr_empty_fails() {
        assert!(parse_bind_addr("").is_err());
    }

    #[test]
    fn parse_bind_addr_missing_port_fails() {
        assert!(parse_bind_addr("localhost").is_err());
    }

    #[test]
    fn parse_bind_addr_ipv6() {
        let (host, port) = parse_bind_addr("[::1]:8079").unwrap();
        assert_eq!(host, "::1");
        assert_eq!(port, 8079);
    }

    #[test]
    fn validate_ok_when_no_models() {
        let models = HashMap::new();
        assert!(validate_no_self_routing("0.0.0.0:8079", &models).is_ok());
    }

    #[test]
    fn validate_ok_when_models_point_upstream() {
        let mut models = HashMap::new();
        models.insert("fast".into(), ModelEntry {
            endpoint: "http://upstream.test:8080/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            sessions: None,
        });
        assert!(validate_no_self_routing("0.0.0.0:8079", &models).is_ok());
    }

    #[test]
    fn validate_rejects_self_loop_localhost() {
        let mut models = HashMap::new();
        models.insert("fast".into(), ModelEntry {
            endpoint: "http://localhost:8079/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            sessions: None,
        });
        let err = validate_no_self_routing("127.0.0.1:8079", &models)
            .expect_err("should reject self-routing model");
        assert!(err.contains("routing loop"), "error should mention routing loop: {err}");
    }

    #[test]
    fn validate_rejects_self_loop_exact_match() {
        let mut models = HashMap::new();
        models.insert("fast".into(), ModelEntry {
            endpoint: "http://127.0.0.1:8079/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            sessions: None,
        });
        let err = validate_no_self_routing("127.0.0.1:8079", &models)
            .expect_err("should reject self-routing model");
        assert!(err.contains("routing loop"));
    }

    #[test]
    fn validate_rejects_when_port_differs_but_host_is_same() {
        let mut models = HashMap::new();
        models.insert("fast".into(), ModelEntry {
            endpoint: "http://127.0.0.1:8080/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            sessions: None,
        });
        // Different port (8080 vs 8079) should be OK
        assert!(validate_no_self_routing("127.0.0.1:8079", &models).is_ok());
    }

    #[test]
    fn validate_empty_bind_addr_errors() {
        let models = HashMap::new();
        let err = validate_no_self_routing("", &models).expect_err("empty bind_addr should error");
        assert!(err.contains("must be set"));
    }
}
