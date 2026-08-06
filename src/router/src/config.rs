//! Router configuration types — deserialized from JSON via `common_core::config`.

pub mod addr;
pub mod builder;
pub mod classification;
pub mod escalation;
pub mod filters;
pub mod routing;

pub use self::addr::{hosts_equivalent, parse_bind_addr, validate_no_self_routing};
pub use self::builder::PipelineParams;
pub use self::classification::{ClassificationChild, ClassificationNode, ClassificationTree};
pub use self::escalation::{EscalationLadderConfig, FrontierConfig, ModelGroup};
pub use self::filters::{
    CommandConfig, ConfidenceGate, FilterAction, FilterOutcome, FilterScope, MockConfig,
    PatternEntry, RejectPatterns,
};
pub use self::routing::{RouteRef, RoutingConfig};

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::logging::LoggingConfig;
use crate::score_matrix::ScoreMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    #[serde(default)]
    pub pipelines: HashMap<String, PipelineParams>,
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
    #[serde(default)]
    pub model_groups: HashMap<String, ModelGroup>,
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
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Chart-embedding model key (M7 HNSW index). Selects an entry from
    /// `models`. `None` falls back to `charts.selector_model`, then
    /// `classifier_model`.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Chart-candidate reranker model key (M7 step 2.5). Selects an entry
    /// from `models`. `None` skips the rerank stage (Step 2 → Step 3
    /// directly).
    #[serde(default)]
    pub reranker_model: Option<String>,
    #[serde(default)]
    pub mock: Option<MockConfig>,
    #[serde(default)]
    pub score_matrix: Option<ScoreMatrix>,
    /// Chart store configuration (DAG workflow library). See M6–M10.
    #[serde(default)]
    pub charts: ChartsConfig,
    /// Post-processing configuration (M10 learning loop).
    #[serde(default)]
    pub post_process: PostProcessConfig,
    /// Nested classification tree.  `Some` switches the classifier stage
    /// into tree-driven mode; the flat pipeline sections remain for
    /// backward compatibility and are derived from the tree where the rest
    /// of the server needs flat views.
    #[serde(default)]
    pub classification: Option<ClassificationTree>,
    /// Rigor-route configuration (M3). `None` (the default) leaves the route
    /// present but unconfigured — requests return an explicit `Unconfigured`
    /// error, never a crash.
    #[serde(default)]
    pub rigor: Option<RigorConfig>,
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
            embedding_model: None,
            reranker_model: None,
            mock: None,
            score_matrix: None,
            charts: ChartsConfig::default(),
            post_process: PostProcessConfig::default(),
            classification: None,
            rigor: None,
        }
    }
}

impl RouterConfig {
    /// The flat `routes` view the server consumes (model → pipeline mapping).
    ///
    /// Flat configs return `routes` unchanged. When a classification tree is
    /// configured, every `terminal` node whose route has no explicit entry gets
    /// a synthesized `RouteRef` (routed through the terminal's own `group`, or
    /// the route name when no group is given) so `RoutingConfig::resolve_route`
    /// and `resolve_pipeline` work with no structural change to the server.
    pub fn routes_view(&self) -> HashMap<String, RouteRef> {
        let mut routes = self.routes.clone();
        if let Some(tree) = &self.classification {
            for (route, group, description) in tree.terminal_views() {
                routes.entry(route.clone()).or_insert(RouteRef {
                    group: group.unwrap_or_else(|| route.clone()),
                    pipelines: vec!["default".into()],
                    description,
                });
            }
        }
        routes
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
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub filter_thinking: bool,
    #[serde(default)]
    pub retry_count: u32,
    #[serde(default = "default_retry_interval")]
    pub retry_base_interval_s: u64,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    #[serde(default)]
    pub sessions: Option<HashMap<String, SessionProfile>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionProfile {
    pub num_ctx: u64,
    #[serde(default)]
    pub sleep_idle_seconds: Option<u64>,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EvictionPolicy {
    #[default]
    Lru,
    Ttl,
    Hybrid,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierOutput {
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub coherence_score: f64,
    pub safety_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub complexity: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completeness: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk: Option<f64>,
}

use common_core::constants::default_true;

fn default_fast_route() -> String {
    "fast".into()
}

// ── Charts (DAG workflow library) configuration ──────────────────────────

/// Chart store configuration — the `charts` section of `RouterConfig`.
///
/// The store is owned by `fluent-router` (see `coral-router`/`charts/`): a
/// directory of human-authored chart JSON files (`D3`), a router-side
/// `workflow_library` HNSW/SQLite path for retrieval (M7), and the model key
/// used by chart-selection LLM adjudication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartsConfig {
    /// Directory of `*.json` chart files loaded at boot. `None` → empty
    /// store (a missing directory is tolerated with a `warn!`).
    #[serde(default)]
    pub dir: Option<String>,
    /// `workflow_library` HNSW/SQLite file path. The index is built lazily
    /// at boot only when this is set (M7).
    #[serde(default)]
    pub index_path: Option<String>,
    /// Chart-selection classifier model key (M7 LLM adjudication step).
    #[serde(default)]
    pub selector_model: Option<String>,
    /// Max candidates surfaced to the selector's LLM adjudication.
    #[serde(default = "default_charts_max_candidates")]
    pub max_candidates: usize,
    /// Embedding-similarity threshold below which a chart is not a candidate.
    #[serde(default = "default_charts_min_score")]
    pub min_score: f64,
    /// Whether bound context entities are exposed to chart templates.
    #[serde(default = "default_charts_entity_context")]
    pub entity_context: bool,
}

impl Default for ChartsConfig {
    fn default() -> Self {
        Self {
            dir: None,
            index_path: None,
            selector_model: None,
            max_candidates: default_charts_max_candidates(),
            min_score: default_charts_min_score(),
            entity_context: default_charts_entity_context(),
        }
    }
}

const fn default_charts_max_candidates() -> usize {
    5
}

// ── Rigor (M3) configuration ──────────────────────────────────────────────

/// Rigor-route configuration — the `rigor` section of `RouterConfig`.
///
/// Model keys select entries from `config.models`; backends are built **only**
/// in `coral-router`'s `build_rigor_route` (DIP, mirroring
/// `build_plan_route`/`default_adjudicator_backend`). `None` at the
/// `RouterConfig` level leaves `/v1/rigor` present but unconfigured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigorConfig {
    /// Model key for the blue-team candidate-answer backend.
    #[serde(default)]
    pub blue_model: Option<String>,
    /// Model key for the red-team objections backend.
    #[serde(default)]
    pub red_model: Option<String>,
    /// Model key for the judge backend.
    #[serde(default)]
    pub judge_model: Option<String>,
    /// Whether the route expects KV-cache checkpoint/rewind to be load-bearing
    /// (a `DependencySession` with a `KvCacheManager`). Rewind always resets
    /// steps; this flag only gates the KV-restore expectation.
    #[serde(default)]
    pub kv_cache_enabled: bool,
    /// Max blue/red/judge passes. Fixed round count (VISION: terminate, don't
    /// loop); a material rejection triggers **at most one** re-run of
    /// blue+judge. Default 2.
    #[serde(default = "default_rigor_max_passes")]
    pub max_passes: usize,
    /// Objection severity at/above which a judge rejection is **material**
    /// (triggers rewind + the second blue pass). Default 0.7.
    #[serde(default = "default_rigor_severity_threshold")]
    pub severity_threshold: f64,
    /// Judge confidence below which a final rejection escalates to frontier.
    /// An explicit config value — never "red scored a point" (M3.5).
    /// Default 0.4.
    #[serde(default = "default_rigor_escalation_confidence")]
    pub escalation_confidence: f64,
}

impl Default for RigorConfig {
    fn default() -> Self {
        Self {
            blue_model: None,
            red_model: None,
            judge_model: None,
            kv_cache_enabled: false,
            max_passes: default_rigor_max_passes(),
            severity_threshold: default_rigor_severity_threshold(),
            escalation_confidence: default_rigor_escalation_confidence(),
        }
    }
}

const fn default_rigor_max_passes() -> usize {
    2
}

const fn default_rigor_severity_threshold() -> f64 {
    0.7
}

const fn default_rigor_escalation_confidence() -> f64 {
    0.4
}

const fn default_charts_min_score() -> f64 {
    0.6
}

const fn default_charts_entity_context() -> bool {
    true
}

// ── Post-processing (M10 learning loop) configuration ─────────────────────

/// Post-processing configuration — the `post_process` section of
/// `RouterConfig`.
///
/// Controls the VISION learning loop: whether a *successful* dispatch is
/// distilled into a reusable draft chart (M10). Per VISION §"Post-processing:
/// audit + workflow extraction", extraction is opt-in and the produced chart
/// is a draft that only becomes selectable after a rubric-validated run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PostProcessConfig {
    /// Whether successful dispatches are decomposed into draft charts
    /// automatically. Default `false` — the operator opts in.
    #[serde(default)]
    pub workflow_extraction: bool,
    /// Which successful dispatches are distilled into draft charts.
    /// Default `"frontier"` — the VISION learning loop learns from
    /// frontier-assisted (escalated/fallback) solutions, not the common
    /// local-primary path. `"all"` restores the blanket behavior by
    /// explicit opt-in.
    #[serde(default)]
    pub workflow_extraction_mode: WorkflowExtractionMode,
}

/// Extraction scope for the M10 learning loop (see `PostProcessConfig`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowExtractionMode {
    /// Only frontier-assisted dispatches (an index > 0 in the primary +
    /// fallback chain) are distilled into draft charts.
    #[default]
    #[serde(rename = "frontier")]
    Frontier,
    /// Every successful dispatch is distilled.
    #[serde(rename = "all")]
    All,
}

fn default_total_timeout_ms() -> u64 {
    common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS
}

fn default_idle_timeout_ms() -> u64 {
    common_core::constants::DEFAULT_IDLE_TIMEOUT_MS
}

fn default_retry_interval() -> u64 {
    common_core::constants::DEFAULT_RETRY_INTERVAL_S
}

#[cfg(test)]
mod tests {
    // Tests assert float config values against literal defaults — deliberate.
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn charts_config_defaults() {
        let cfg = ChartsConfig::default();
        assert_eq!(cfg.max_candidates, 5);
        assert_eq!(cfg.min_score, 0.6);
        assert!(cfg.entity_context);
        assert!(cfg.dir.is_none());
        assert!(cfg.index_path.is_none());
        assert!(cfg.selector_model.is_none());
    }

    // ── M3 rigor-route configuration ─────────────────────────────────────

    #[test]
    fn rigor_config_defaults() {
        let cfg = RigorConfig::default();
        assert_eq!(cfg.max_passes, 2);
        assert_eq!(cfg.severity_threshold, 0.7);
        assert_eq!(cfg.escalation_confidence, 0.4);
        assert!(!cfg.kv_cache_enabled);
        assert!(cfg.blue_model.is_none());
        assert!(cfg.red_model.is_none());
        assert!(cfg.judge_model.is_none());
    }

    #[test]
    fn router_config_absent_rigor_section_defaults_to_none() {
        // The shipped env/coral-router.json has no `rigor` section; the route
        // stays present-but-unconfigured (None), never a crash.
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
        assert!(cfg.rigor.is_none());
    }

    #[test]
    fn rigor_config_round_trip() {
        let json = serde_json::json!({
            "rigor": {
                "blue_model": "fast",
                "red_model": "code",
                "judge_model": "code",
                "kv_cache_enabled": true,
                "max_passes": 3,
                "severity_threshold": 0.8,
                "escalation_confidence": 0.3,
            }
        });
        let cfg: RouterConfig = serde_json::from_value(json).unwrap();
        let rigor = cfg.rigor.expect("rigor section parsed");
        assert_eq!(rigor.blue_model.as_deref(), Some("fast"));
        assert_eq!(rigor.red_model.as_deref(), Some("code"));
        assert_eq!(rigor.judge_model.as_deref(), Some("code"));
        assert!(rigor.kv_cache_enabled);
        assert_eq!(rigor.max_passes, 3);
        assert_eq!(rigor.severity_threshold, 0.8);
        assert_eq!(rigor.escalation_confidence, 0.3);

        // Partial section still round-trips with defaults for the rest.
        let partial: RouterConfig = serde_json::from_value(serde_json::json!({
            "rigor": {"blue_model": "fast"}
        }))
        .unwrap();
        let partial_cfg = partial.rigor.expect("rigor parsed");
        assert_eq!(partial_cfg.blue_model.as_deref(), Some("fast"));
        assert_eq!(partial_cfg.max_passes, 2, "absent fields default");
        assert_eq!(partial_cfg.severity_threshold, 0.7);
    }

    #[test]
    fn router_config_absent_charts_section_defaults_cleanly() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
        assert_eq!(cfg.charts.max_candidates, 5);
        assert_eq!(cfg.charts.min_score, 0.6);
        assert!(cfg.charts.entity_context);
        assert!(cfg.charts.dir.is_none());
    }

    #[test]
    fn router_config_embedding_and_reranker_models_parse() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"embedding_model": "embed", "reranker_model": "rerank"}"#)
                .unwrap();
        assert_eq!(cfg.embedding_model.as_deref(), Some("embed"));
        assert_eq!(cfg.reranker_model.as_deref(), Some("rerank"));

        let absent: RouterConfig = serde_json::from_str(r"{}").unwrap();
        assert!(absent.embedding_model.is_none());
        assert!(absent.reranker_model.is_none());
    }

    #[test]
    fn charts_section_round_trips() {
        let json = r#"{
            "dir": "env/workflows/charts",
            "index_path": "data/workflow_library.sqlite",
            "selector_model": "qwen3.5-4b",
            "max_candidates": 5,
            "min_score": 0.6,
            "entity_context": true
        }"#;
        let cfg: ChartsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.dir.as_deref(), Some("env/workflows/charts"));
        assert_eq!(
            cfg.index_path.as_deref(),
            Some("data/workflow_library.sqlite")
        );
        assert_eq!(cfg.selector_model.as_deref(), Some("qwen3.5-4b"));
        assert_eq!(cfg.max_candidates, 5);
        assert_eq!(cfg.min_score, 0.6);
        assert!(cfg.entity_context);

        let serialized = serde_json::to_string(&cfg).unwrap();
        let back: ChartsConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back.dir, cfg.dir);
        assert_eq!(back.max_candidates, cfg.max_candidates);
        assert_eq!(back.min_score, cfg.min_score);
    }

    #[test]
    fn partial_charts_section_defaults_missing_fields() {
        let cfg: ChartsConfig = serde_json::from_str(r#"{"dir": "env/workflows/charts"}"#).unwrap();
        assert_eq!(cfg.dir.as_deref(), Some("env/workflows/charts"));
        assert_eq!(cfg.max_candidates, 5);
        assert_eq!(cfg.min_score, 0.6);
        assert!(cfg.entity_context);
        assert!(cfg.index_path.is_none());
        assert!(cfg.selector_model.is_none());
    }

    #[test]
    fn router_config_parses_charts_section() {
        let json = r#"{
            "charts": { "dir": "env/workflows/charts", "max_candidates": 8 }
        }"#;
        let cfg: RouterConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.charts.dir.as_deref(), Some("env/workflows/charts"));
        assert_eq!(cfg.charts.max_candidates, 8);
        assert_eq!(cfg.charts.min_score, 0.6, "unset field keeps its default");
    }

    // ── Post-process (M10 learning loop) ────────────────────────────────

    #[test]
    fn post_process_defaults_to_disabled() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.workflow_extraction, "extraction is opt-in");
        assert_eq!(
            cfg.workflow_extraction_mode,
            WorkflowExtractionMode::Frontier,
            "default scope is frontier-assisted only"
        );
    }

    #[test]
    fn post_process_absent_section_defaults_cleanly() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
        assert!(
            !cfg.post_process.workflow_extraction,
            "absent post_process section defaults extraction off"
        );
        assert_eq!(
            cfg.post_process.workflow_extraction_mode,
            WorkflowExtractionMode::Frontier,
            "absent mode field defaults to frontier"
        );
    }

    #[test]
    fn post_process_round_trips() {
        let json = r#"{ "workflow_extraction": true }"#;
        let cfg: PostProcessConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.workflow_extraction);
        assert_eq!(
            cfg.workflow_extraction_mode,
            WorkflowExtractionMode::Frontier,
            "absent mode field keeps the frontier default"
        );

        let serialized = serde_json::to_string(&cfg).unwrap();
        let back: PostProcessConfig = serde_json::from_str(&serialized).unwrap();
        assert!(back.workflow_extraction);
        assert_eq!(back.workflow_extraction_mode, cfg.workflow_extraction_mode);
    }

    #[test]
    fn workflow_extraction_mode_parses_both_variants() {
        let all: WorkflowExtractionMode = serde_json::from_str(r#""all""#).expect("all parses");
        assert_eq!(all, WorkflowExtractionMode::All);

        let frontier: WorkflowExtractionMode =
            serde_json::from_str(r#""frontier""#).expect("frontier parses");
        assert_eq!(frontier, WorkflowExtractionMode::Frontier);

        assert!(serde_json::from_str::<WorkflowExtractionMode>(r#""bogus""#).is_err());
    }

    #[test]
    fn router_config_parses_post_process_section() {
        let json = r#"{
            "post_process": { "workflow_extraction": true },
            "charts": { "dir": "env/workflows/charts" }
        }"#;
        let cfg: RouterConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.post_process.workflow_extraction);
        assert_eq!(cfg.charts.dir.as_deref(), Some("env/workflows/charts"));
        assert_eq!(
            cfg.post_process.workflow_extraction_mode,
            WorkflowExtractionMode::Frontier,
            "existing configs without the new field still deserialize"
        );
    }

    #[test]
    fn router_config_parses_extraction_mode_all() {
        let json = r#"{
            "post_process": {
                "workflow_extraction": true,
                "workflow_extraction_mode": "all"
            }
        }"#;
        let cfg: RouterConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.post_process.workflow_extraction);
        assert_eq!(
            cfg.post_process.workflow_extraction_mode,
            WorkflowExtractionMode::All
        );
    }

    #[test]
    fn model_entry_serde_defaults_read_canonical_constants() {
        // The same constants `RoutingTarget` reads (M7.2 divergence guard).
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://localhost:8080/v1/chat/completions",
            "intelligence": 2,
            "cost_input": 1e-6,
            "cost_output": 6e-6,
            "cost_cached_read": 4e-7,
            "speed": 8,
        }))
        .unwrap();
        assert_eq!(
            entry.total_timeout_ms,
            common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS
        );
        assert_eq!(
            entry.idle_timeout_ms,
            common_core::constants::DEFAULT_IDLE_TIMEOUT_MS
        );
        assert_eq!(
            entry.retry_base_interval_s,
            common_core::constants::DEFAULT_RETRY_INTERVAL_S
        );
    }

    // ── M4 classification-tree derived flat views ────────────────────────

    fn tree_section() -> serde_json::Value {
        serde_json::json!({
            "classification": {
                "root": {
                    "type": "classifier",
                    "description": "router",
                    "model": "fast",
                    "children": [
                        {
                            "key": "code",
                            "description": "programming",
                            "node": { "type": "terminal", "route": "code", "group": "code" }
                        },
                        {
                            "key": "brand_new",
                            "description": "not in flat routes",
                            "node": { "type": "terminal", "route": "brand_new", "group": "question" }
                        }
                    ]
                }
            },
            "models": {
                "fast": {"endpoint": "http://upstream.test/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7, "speed": 8}
            },
            "model_groups": {
                "fast": ["fast"],
                "code": ["fast"],
                "question": ["fast"]
            },
            "routes": {
                "code": {"group": "code", "pipelines": ["default"], "description": "code"}
            }
        })
    }

    #[test]
    fn routes_view_synthesizes_terminal_routes() {
        let cfg: RouterConfig = serde_json::from_value(tree_section()).unwrap();
        let routes = cfg.routes_view();
        // Explicit flat route is preserved.
        assert_eq!(routes["code"].group, "code");
        assert_eq!(routes["code"].pipelines, vec!["default".to_string()]);
        // Terminal route without a flat entry is synthesized from its group.
        assert_eq!(routes["brand_new"].group, "question");
        assert_eq!(routes["brand_new"].pipelines, vec!["default".to_string()]);
    }

    #[test]
    fn routes_view_flat_config_is_unchanged() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"routes": {"a": {"group": "g"}}}"#).unwrap();
        assert_eq!(cfg.routes_view().len(), 1);
        assert!(cfg.routes_view().contains_key("a"));
    }

    #[test]
    fn routing_config_derives_system_prompt_from_tree() {
        let cfg: RouterConfig = serde_json::from_value(tree_section()).unwrap();
        let routing = cfg.routing_config();
        assert!(
            routing.system_prompt.contains("You are a router."),
            "tree-derived system prompt, got: {}",
            routing.system_prompt
        );
        assert!(
            routing.routes.contains_key("brand_new"),
            "derived routes reach the RoutingConfig so terminal resolution works"
        );
    }

    #[test]
    fn routing_config_keeps_explicit_system_prompt() {
        let mut cfg: RouterConfig = serde_json::from_value(tree_section()).unwrap();
        cfg.system_prompt = "custom preamble".into();
        let routing = cfg.routing_config();
        assert_eq!(routing.system_prompt, "custom preamble");
    }
}
