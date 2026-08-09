//! Router configuration types - deserialized from JSON via `common_core::config`.

pub mod addr;
pub mod builder;
pub mod classification;
pub mod escalation;
pub mod filters;
pub mod routing;

pub use self::addr::{hosts_equivalent, parse_bind_addr, validate_no_self_routing};
pub use self::builder::{PipelineParams, TargetMatchMode};
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
    /// from `models`. `None` skips the rerank stage (Step 2 - Step 3
    /// directly).
    #[serde(default)]
    pub reranker_model: Option<String>,
    #[serde(default)]
    pub mock: Option<MockConfig>,
    #[serde(default)]
    pub score_matrix: Option<ScoreMatrix>,
    /// Chart store configuration (DAG workflow library). See M6-M10.
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
    /// present but unconfigured - requests return an explicit `Unconfigured`
    /// error, never a crash.
    #[serde(default)]
    pub rigor: Option<RigorConfig>,
    /// Sidecar instance-management policy (M4). Governs the sidecar task that
    /// reconciles the fork's shared-weight instances against the configured
    /// profiles, polls `/memory`, and evicts/allocates KV + compute only (the
    /// weights stay loaded in `llama-server`).
    #[serde(default)]
    pub sidecar: SidecarConfig,
    /// Ledger composition section (M2). `Some` opts the boot path into opening
    /// a `ContentNodeLedger` (with a real `Summarizer` backend targeting
    /// `<base>:ledger`) so LOD derivation exists at runtime. `None` (the
    /// default) leaves today's behavior - no ledger at boot.
    #[serde(default)]
    pub ledger: Option<LedgerConfig>,
    /// Session composition section (M2). `Some` opts the boot path into a
    /// `SessionRegistry` (D6 canonical session home) so rigor rewind and
    /// checkpoint/rewind state exist at runtime. `None` (the default) leaves
    /// today's behavior - no session registry at boot.
    #[serde(default)]
    pub session: Option<SessionConfig>,
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
            sidecar: SidecarConfig::default(),
            ledger: None,
            session: None,
        }
    }
}

impl RouterConfig {
    /// The flat `routes` view the server consumes (model - pipeline mapping).
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
    /// Instance-pool declaration for the fork's shared-weight instances. The
    /// old `sessions` key is accepted as an alias during the transition.
    #[serde(default, alias = "sessions")]
    pub instances: Option<HashMap<String, InstanceProfile>>,
    /// Local GGUF weights file path. When set (or when `hf_repo` or `instances`
    /// is set), Coral Router is the process owner: it spawns and supervises a
    /// dedicated `llama-server` for this model on a free localhost port and
    /// rewrites `endpoint` to it at boot. Passed to the server as `--model`.
    #[serde(default)]
    pub weights: Option<String>,
    /// HuggingFace repo to load (`-hf <repo>[:quant]`), the on-demand
    /// alternative to `weights`. The repo name also becomes the server's
    /// primary model alias when `name` is unset.
    #[serde(default)]
    pub hf_repo: Option<String>,
    /// HuggingFace file within `hf_repo` (`-hff <file>`); optional, overrides
    /// the quant default.
    #[serde(default)]
    pub hf_file: Option<String>,
}

impl ModelEntry {
    /// Whether Coral Router manages a dedicated `llama-server` process for this
    /// model (the model declares a weights source or an instance pool). Managed
    /// models are spawned on a free localhost port at boot and their `endpoint`
    /// is rewritten to the spawned server.
    pub fn is_managed(&self) -> bool {
        self.weights.is_some() || self.hf_repo.is_some() || self.instances.is_some()
    }

    /// The model name handed to the spawned `llama-server` (`--alias`): the
    /// configured llama.cpp model name, else the HF repo, else the config key.
    pub fn llama_model_name(&self, model_key: &str) -> String {
        self.name
            .clone()
            .or_else(|| self.hf_repo.clone())
            .unwrap_or_else(|| model_key.to_string())
    }
}

/// One config-declared instance profile. The map key on `ModelEntry.instances`
/// provides the default instance name; `count > 1` expands into sibling
/// instances named `<key>-0` .. `<key>-{count-1}` sharing the profile's group.
/// Sampling `params` are merged into the request body for dispatches through
/// these instances; declaration-only keys (`num_ctx`/`parallel`/
/// `sleep_idle_seconds`) are stripped before dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceProfile {
    /// Instance name; default = the map key (expanded `<name><i>` for count > 1).
    #[serde(default)]
    pub name: Option<String>,
    /// Group; default = instance name. count > 1 instances share this group.
    #[serde(default)]
    pub group: Option<String>,
    /// Number of sibling instances this profile expands to (1 = single instance).
    #[serde(default = "default_instance_count")]
    pub count: u32,
    /// Context size in tokens.
    pub num_ctx: u64,
    /// Slots per instance; default = inherit server global.
    #[serde(default)]
    pub parallel: Option<u32>,
    /// Exempt from auto-sleep and in-process eviction; implies no_sleep.
    #[serde(default)]
    pub pinned: bool,
    /// Never auto-sleep (stays warm); the fork grammar's sleep=0. `warm` is a
    /// friendly serde alias for the same flag.
    #[serde(default, alias = "warm")]
    pub no_sleep: bool,
    /// >0 = per-instance idle timeout seconds; -1 = inherit global; None = inherit.
    #[serde(default)]
    pub sleep_idle_seconds: Option<i32>,
    /// Target of a bare `<base>` request.
    #[serde(default)]
    pub default: bool,
    /// Sampling params merged into the request body for dispatches through this
    /// instance.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

fn default_instance_count() -> u32 {
    1
}

impl ModelEntry {
    /// The expanded flat list of `InstanceProfile`s for this model: applies
    /// `count` expansion (naming each sibling `<key>-0` .. `<key>-{count-1}`)
    /// and resolves the name/group defaults (name = map key, group = name when
    /// absent). Empty when no instances are configured.
    pub fn instance_profiles(&self) -> Vec<InstanceProfile> {
        let Some(instances) = &self.instances else {
            return Vec::new();
        };
        let mut keys: Vec<&String> = instances.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            let profile = &instances[key];
            let base_name = profile.name.clone().unwrap_or_else(|| key.clone());
            let count = profile.count.max(1);
            // All siblings share the profile's group (default = base name).
            let group = profile.group.clone().unwrap_or_else(|| base_name.clone());
            for i in 0..count {
                let name = if count > 1 {
                    format!("{base_name}-{i}")
                } else {
                    base_name.clone()
                };
                let mut p = profile.clone();
                p.name = Some(name);
                p.group = Some(group.clone());
                out.push(p);
            }
        }
        out
    }

    /// The dispatch qualifier for the model's default inference point: the
    /// `default: true` profile's group, else the single shared group across all
    /// profiles, else `None` (bare `<base>`). `None` also when no instances are
    /// configured. Encoded as `model = "<base>:<qualifier>"`.
    pub fn default_dispatch_qualifier(&self) -> Option<String> {
        let profiles = self.instance_profiles();
        if profiles.is_empty() {
            return None;
        }
        if let Some(d) = profiles.iter().find(|p| p.default) {
            return d.group.clone();
        }
        let first = profiles[0].group.clone()?;
        if profiles.iter().all(|p| p.group.as_deref() == Some(first.as_str())) {
            Some(first)
        } else {
            None
        }
    }

    /// The dispatch qualifier for the router's *internal work group* (the
    /// "pool"): the classifier, chart selector/adjudicator/reranker,
    /// target-matching ladder, and rigor role backends spread across the
    /// instance pool rather than pinning to the client-facing default instance.
    /// This is a distinct intent from `default_dispatch_qualifier`, which
    /// resolves the fork's *default instance* for client-facing bare-`<base>`
    /// dispatch. Resolution order (D1), deterministic:
    ///
    /// 1. The group of the `default: false` profile with the largest `count`
    ///    (the "work pool"; for the reference config this is `swarm`).
    /// 2. Else the `default: true` profile's group.
    /// 3. Else the single group shared by all profiles.
    /// 4. Else `None` (bare `<base>`, upstream models unchanged).
    pub fn pool_qualifier(&self) -> Option<String> {
        let profiles = self.instance_profiles();
        if profiles.is_empty() {
            return None;
        }
        // 1. The non-default profile with the largest sibling count (ties
        //    resolve to the first encountered in deterministic map order).
        let mut best: Option<&InstanceProfile> = None;
        let mut best_count: u32 = 0;
        for p in profiles.iter().filter(|p| !p.default) {
            let c = p.count.max(1);
            if best.is_none() || c > best_count {
                best = Some(p);
                best_count = c;
            }
        }
        if let Some(b) = best {
            return b.group.clone();
        }
        // 2. The default profile's group.
        if let Some(d) = profiles.iter().find(|p| p.default) {
            return d.group.clone();
        }
        // 3. The single group shared by all profiles.
        let first = profiles[0].group.clone()?;
        if profiles.iter().all(|p| p.group.as_deref() == Some(first.as_str())) {
            Some(first)
        } else {
            None
        }
    }
}

/// Declaration-only request-body keys the fork ignores: the instance grammar
/// owns them (`ctx`/`parallel`/`sleep`), so they must not leak into the body.
pub const DECLARATION_PARAM_KEYS: [&str; 4] =
    ["num_ctx", "parallel", "sleep_idle_seconds", "rope_freq_base"];

/// Remove declaration-only keys from a params object, keeping sampling params
/// (`temperature`, `repeat_penalty`, `chat_template_kwargs`, ...). Non-object
/// params are returned unchanged.
pub fn strip_declaration_params(params: serde_json::Value) -> serde_json::Value {
    let Some(obj) = params.as_object() else {
        return params;
    };
    let mut out = obj.clone();
    for k in DECLARATION_PARAM_KEYS {
        out.remove(k);
    }
    serde_json::Value::Object(out)
}

/// Merge a model entry's top-level sampling `params` with a specific
/// instance profile's `params` (profile wins), returning the merged object.
/// Non-object params degrade to an empty object (nothing to merge). This is
/// the single canonical merge for per-instance sampling knobs; the profile is
/// looked up by name-or-group so both the exact-instance and group dispatch
/// paths reach the same value.
pub(crate) fn merge_sampling_params(
    entry: Option<&serde_json::Value>,
    profile: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut merged = serde_json::Map::new();
    if let Some(v) = entry.and_then(serde_json::Value::as_object) {
        merged.extend(v.clone());
    }
    if let Some(v) = profile.and_then(serde_json::Value::as_object) {
        merged.extend(v.clone());
    }
    serde_json::Value::Object(merged)
}

impl ModelEntry {
    /// Resolve the sampling params to send when dispatching to `qualifier`
    /// (an instance name or group of this model's pool): the matching
    /// profile's `params` overlaid onto the entry's top-level `params`
    /// (profile wins), declaration-only keys stripped. `None` when no profile
    /// matches `qualifier` — callers fall back to the entry's bare params.
    pub fn instance_params_for(&self, qualifier: &str) -> Option<serde_json::Value> {
        let profile = self.instance_profiles().into_iter().find(|p| {
            p.name.as_deref() == Some(qualifier) || p.group.as_deref() == Some(qualifier)
        })?;
        let merged =
            merge_sampling_params(self.params.as_ref(), profile.params.as_ref());
        Some(strip_declaration_params(merged))
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

// -- Charts (DAG workflow library) configuration --------------------------

/// Chart store configuration - the `charts` section of `RouterConfig`.
///
/// The store is owned by `fluent-router` (see `coral-router`/`charts/`): a
/// directory of human-authored chart JSON files (`D3`), a router-side
/// `workflow_library` HNSW/SQLite path for retrieval (M7), and the model key
/// used by chart-selection LLM adjudication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartsConfig {
    /// Directory of `*.json` chart files loaded at boot. `None` - empty
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

// -- Rigor (M3) configuration ----------------------------------------------

/// Rigor-route configuration - the `rigor` section of `RouterConfig`.
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
    /// An explicit config value - never "red scored a point" (M3.5).
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

/// Default cap on the ledger `Summarizer`'s summary length (tokens). Only a
/// named constant - `LedgerConfig.max_summary_tokens` defaults to it.
pub const DEFAULT_LEDGER_MAX_SUMMARY_TOKENS: u32 = 200;

/// Ledger composition section (M2) - the `ledger` block of `RouterConfig`.
///
/// `Some` opts the composition root (`main.rs`) into opening a
/// `ContentNodeLedger` and attaching a `Summarizer` backend targeting the
/// named `ledger` instance. `None` (absent) keeps today's behavior - no
/// ledger at boot - so existing deployments are untouched until they opt in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerConfig {
    /// Durable store path. `None` falls back to an in-memory ledger with a
    /// `warn!` (ephemeral, still functional for LOD derivation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Model key for the ledger `Summarizer`. `None` falls back to the
    /// classifier model key, then to no summarizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Max summary length (tokens) for LOD1-LOD4 derivation.
    #[serde(default = "default_ledger_max_summary_tokens")]
    pub max_summary_tokens: u32,
}

const fn default_ledger_max_summary_tokens() -> u32 {
    DEFAULT_LEDGER_MAX_SUMMARY_TOKENS
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            path: None,
            model: None,
            max_summary_tokens: DEFAULT_LEDGER_MAX_SUMMARY_TOKENS,
        }
    }
}

/// Session composition section (M2) - the `session` block of `RouterConfig`.
///
/// `Some` opts the composition root into a `SessionRegistry` (D6 canonical
/// session home) so checkpoint/rewind state and rigor rewind exist at runtime.
/// `None` (absent) keeps today's behavior - no session registry at boot.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionConfig {
    /// Cold-tier mountpoint for KV cache snapshots, mapped to
    /// `SessionRegistry::new`'s `kv_root`. `None` uses a process-local temp
    /// directory (durable across requests, ephemeral across restarts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
}

/// Sidecar instance-management policy (M4).
///
/// The sidecar task is the external VRAM-policy owner the fork's docs
/// describe: it boot-reconciles configured instance profiles against
/// `GET /instances`, polls `/memory`, and evicts least-recently-used unpinned
/// instances when free device VRAM drops below the watermark. It only ever
/// allocates or frees KV + compute buffers - the shared weights stay loaded in
/// `llama-server`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarConfig {
    /// How often the residency loop polls `/memory`, in seconds.
    #[serde(default = "default_sidecar_poll_interval_s")]
    pub poll_interval_s: u64,
    /// Free-VRAM threshold (bytes) below which the residency loop evicts.
    #[serde(default = "default_sidecar_watermark")]
    pub vram_low_watermark_bytes: u64,
    /// Max unpinned instances evicted per low-VRAM pass.
    #[serde(default = "default_sidecar_evict_batch")]
    pub evict_batch: usize,
    /// Device VRAM ceiling (bytes). `None` disables residency eviction (the
    /// loop still polls and logs) because free VRAM cannot be computed.
    #[serde(default)]
    pub vram_total_bytes: Option<u64>,
    /// Slot-save directory the fork writes KV snapshots under
    /// (`<slot_save_path>/<model_key>/`). Feeds M3 snapshot-path derivation.
    #[serde(default)]
    pub slot_save_path: Option<String>,
    /// Env var naming the management API key sent as `Authorization: Bearer`.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            poll_interval_s: default_sidecar_poll_interval_s(),
            vram_low_watermark_bytes: default_sidecar_watermark(),
            evict_batch: default_sidecar_evict_batch(),
            vram_total_bytes: None,
            slot_save_path: None,
            api_key_env: None,
        }
    }
}

const fn default_sidecar_poll_interval_s() -> u64 {
    5
}

const fn default_sidecar_watermark() -> u64 {
    1073741824
}

const fn default_sidecar_evict_batch() -> usize {
    1
}

const fn default_charts_min_score() -> f64 {
    0.6
}

const fn default_charts_entity_context() -> bool {
    true
}

// -- Post-processing (M10 learning loop) configuration ---------------------

/// Post-processing configuration - the `post_process` section of
/// `RouterConfig`.
///
/// Controls the VISION learning loop: whether a *successful* dispatch is
/// distilled into a reusable draft chart (M10). Per VISION -"Post-processing:
/// audit + workflow extraction", extraction is opt-in and the produced chart
/// is a draft that only becomes selectable after a rubric-validated run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PostProcessConfig {
    /// Whether successful dispatches are decomposed into draft charts
    /// automatically. Default `false` - the operator opts in.
    #[serde(default)]
    pub workflow_extraction: bool,
    /// Which successful dispatches are distilled into draft charts.
    /// Default `"frontier"` - the VISION learning loop learns from
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
    // Tests assert float config values against literal defaults - deliberate.
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

    // -- M3 rigor-route configuration -------------------------------------

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

    // -- Post-process (M10 learning loop) --------------------------------

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

    // -- M4 classification-tree derived flat views ------------------------

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

    // -- M6: in-group target-matching knob (PipelineParams) ----------------

    #[test]
    fn pipeline_params_target_match_defaults() {
        let defaults = builder::PipelineParams::default();
        assert_eq!(
            defaults.target_match,
            builder::TargetMatchMode::SelfAssess,
            "the self-assess ladder is the default policy (-4.6)"
        );
        assert_eq!(
            defaults.target_match_timeout_ms,
            common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS,
            "per-self-assessment budget defaults to the shared total-timeout constant"
        );
    }

    #[test]
    fn pipeline_params_target_match_absent_fields_deserialize_to_defaults() {
        // A pipeline that omits both knob fields must deserialize to the same
        // defaults (mirror the `classifier_retry_max` pattern) - existing
        // configs stay byte-identical.
        let cfg: RouterConfig = serde_json::from_str(
            r#"{
                "pipelines": {"default": {"classifier": true, "classifier_model": "fast"}}
            }"#,
        )
        .expect("valid config");
        let params = &cfg.pipelines["default"];
        assert_eq!(params.target_match, builder::TargetMatchMode::SelfAssess);
        assert_eq!(
            params.target_match_timeout_ms,
            common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS
        );
    }

    #[test]
    fn pipeline_params_target_match_parses_both_variants() {
        let self_assess: builder::TargetMatchMode =
            serde_json::from_str(r#""self_assess""#).expect("self_assess parses");
        assert_eq!(self_assess, builder::TargetMatchMode::SelfAssess);

        let static_mode: builder::TargetMatchMode =
            serde_json::from_str(r#""static""#).expect("static parses");
        assert_eq!(static_mode, builder::TargetMatchMode::Static);

        assert!(
            serde_json::from_str::<builder::TargetMatchMode>(r#""bogus""#).is_err(),
            "unknown policy must be rejected, not silently defaulted"
        );
    }

    #[test]
    fn pipeline_params_target_match_round_trips() {
        // Non-default values survive a serialize - deserialize cycle.
        let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
            "pipelines": {
                "default": {
                    "classifier": true,
                    "classifier_model": "fast",
                    "target_match": "static",
                    "target_match_timeout_ms": 12345
                }
            }
        }))
        .unwrap();
        assert_eq!(cfg.pipelines["default"].target_match, builder::TargetMatchMode::Static);
        assert_eq!(cfg.pipelines["default"].target_match_timeout_ms, 12345);

        let serialized = serde_json::to_string(&cfg).unwrap();
        let back: RouterConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back.pipelines["default"].target_match, builder::TargetMatchMode::Static);
        assert_eq!(back.pipelines["default"].target_match_timeout_ms, 12345);
    }

    // -- M1 instance-pool declaration -------------------------------------

    fn profile_json(name: &str, count: u32, group: &str, num_ctx: u64) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "count": count,
            "group": group,
            "num_ctx": num_ctx,
        })
    }

    #[test]
    fn instances_count_expansion_names_siblings_in_shared_group() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 2,
            "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
            "speed": 8,
            "instances": {
                "swarm": profile_json("swarm", 3, "swarm", 16384),
                "ledger": { "num_ctx": 131072, "pinned": true, "default": true }
            }
        }))
        .unwrap();

        let profiles = entry.instance_profiles();
        assert_eq!(profiles.len(), 4);
        // Profiles are emitted in sorted map-key order: ledger < swarm.
        assert_eq!(profiles[0].name.as_deref(), Some("ledger"));
        assert_eq!(profiles[0].group.as_deref(), Some("ledger"));
        assert!(profiles[0].pinned);
        assert!(profiles[0].default);
        // count: 3 -> `<key>-0` .. `<key>-2` in the shared group.
        assert_eq!(profiles[1].name.as_deref(), Some("swarm-0"));
        assert_eq!(profiles[1].group.as_deref(), Some("swarm"));
        assert_eq!(profiles[2].name.as_deref(), Some("swarm-1"));
        assert_eq!(profiles[3].name.as_deref(), Some("swarm-2"));
        assert_eq!(profiles[3].group.as_deref(), Some("swarm"));
        assert_eq!(profiles[3].num_ctx, 16384);
    }

    #[test]
    fn instances_single_profile_defaults_name_to_map_key() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
            "instances": { "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30 } }
        }))
        .unwrap();
        let profiles = entry.instance_profiles();
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].name.as_deref(), Some("scratch"));
        assert_eq!(profiles[0].group.as_deref(), Some("scratch"));
        assert_eq!(profiles[0].sleep_idle_seconds, Some(30));
        assert_eq!(profiles[0].count, 1);
    }

    #[test]
    fn old_sessions_key_still_parses_as_instances() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
            "sessions": { "ctx16384": { "num_ctx": 16384 } }
        }))
        .unwrap();
        let instances = entry.instances.expect("sessions alias maps into instances");
        assert_eq!(instances.len(), 1);
        assert!(instances.contains_key("ctx16384"));
    }

    #[test]
    fn no_instances_yields_empty_profile_list() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
        }))
        .unwrap();
        assert!(entry.instance_profiles().is_empty());
    }

    #[test]
    fn warm_alias_maps_to_no_sleep() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
            "instances": { "swarm": { "num_ctx": 16384, "warm": true } }
        }))
        .unwrap();
        let profiles = entry.instance_profiles();
        assert!(profiles[0].no_sleep);
    }

    // -- M1 pool vs default qualifier (D1) -------------------------------

    /// The reference swarm entry: a count=3 non-default `swarm` work pool, a
    /// pinned `default: true` ledger, and a non-default scratch profile.
    fn reference_swarm_entry() -> ModelEntry {
        serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "name": "abiray/lfm2.5-2.6b-heretic-abliterated",
            "intelligence": 2,
            "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
            "speed": 8,
            "instances": {
                "swarm": profile_json("swarm", 3, "swarm", 16384),
                "ledger": { "num_ctx": 131072, "pinned": true, "default": true },
                "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30 }
            }
        }))
        .expect("reference swarm entry parses")
    }

    #[test]
    fn pool_qualifier_reference_config_targets_swarm() {
        let entry = reference_swarm_entry();
        assert_eq!(
            entry.pool_qualifier().as_deref(),
            Some("swarm"),
            "the largest non-default profile (count=3) is the work pool"
        );
    }

    #[test]
    fn pool_qualifier_vs_default_qualifier_two_intents_two_answers() {
        // The two intents must diverge on the same entry: pool = swarm (the
        // work group), default = ledger (the client-facing default instance).
        let entry = reference_swarm_entry();
        assert_eq!(entry.pool_qualifier().as_deref(), Some("swarm"));
        assert_eq!(
            entry.default_dispatch_qualifier().as_deref(),
            Some("ledger")
        );
    }

    #[test]
    fn pool_qualifier_ledger_only_defaults_to_ledger() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
            "instances": { "ledger": { "num_ctx": 131072, "default": true } }
        }))
        .unwrap();
        assert_eq!(entry.pool_qualifier().as_deref(), Some("ledger"));
    }

    #[test]
    fn pool_qualifier_single_shared_group() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
            "instances": {
                "a": { "num_ctx": 8192, "group": "shared" },
                "b": { "num_ctx": 8192, "group": "shared" }
            }
        }))
        .unwrap();
        assert_eq!(entry.pool_qualifier().as_deref(), Some("shared"));
    }

    #[test]
    fn pool_qualifier_no_instances_is_none() {
        let entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "endpoint": "http://x/v1/chat/completions",
            "intelligence": 1,
            "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
            "speed": 1,
        }))
        .unwrap();
        assert!(entry.pool_qualifier().is_none());
    }

    // -- M4 sidecar policy -----------------------------------------------

    #[test]
    fn sidecar_absent_section_defaults_cleanly() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
        assert_eq!(cfg.sidecar.poll_interval_s, 5);
        assert_eq!(cfg.sidecar.vram_low_watermark_bytes, 1073741824);
        assert_eq!(cfg.sidecar.evict_batch, 1);
        assert!(cfg.sidecar.vram_total_bytes.is_none());
        assert!(cfg.sidecar.slot_save_path.is_none());
        assert!(cfg.sidecar.api_key_env.is_none());
    }

    #[test]
    fn sidecar_section_round_trips() {
        let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
            "sidecar": {
                "poll_interval_s": 10,
                "vram_low_watermark_bytes": 536870912,
                "evict_batch": 2,
                "vram_total_bytes": 1048576,
                "slot_save_path": "/srv/slots",
                "api_key_env": "LLAMA_API_KEY",
            }
        }))
        .unwrap();
        assert_eq!(cfg.sidecar.poll_interval_s, 10);
        assert_eq!(cfg.sidecar.vram_low_watermark_bytes, 536870912);
        assert_eq!(cfg.sidecar.evict_batch, 2);
        assert_eq!(cfg.sidecar.vram_total_bytes, Some(1048576));
        assert_eq!(cfg.sidecar.slot_save_path.as_deref(), Some("/srv/slots"));
        assert_eq!(cfg.sidecar.api_key_env.as_deref(), Some("LLAMA_API_KEY"));
    }

    // -- M2 ledger + session composition sections ------------------------

    #[test]
    fn router_config_absent_ledger_and_session_sections_default_to_none() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
        assert!(
            cfg.ledger.is_none(),
            "absent ledger section -> no ledger at boot (byte-identical behavior)"
        );
        assert!(
            cfg.session.is_none(),
            "absent session section -> no session registry at boot (byte-identical behavior)"
        );
    }

    #[test]
    fn ledger_and_session_sections_round_trip() {
        let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
            "ledger": {
                "path": "data/ledger.sqlite",
                "model": "swarm",
                "max_summary_tokens": 300,
            },
            "session": { "root": "data/sessions" },
        }))
        .unwrap();

        let ledger = cfg.ledger.as_ref().expect("ledger section parsed");
        assert_eq!(ledger.path.as_deref(), Some("data/ledger.sqlite"));
        assert_eq!(ledger.model.as_deref(), Some("swarm"));
        assert_eq!(ledger.max_summary_tokens, 300);

        let session = cfg.session.as_ref().expect("session section parsed");
        assert_eq!(session.root.as_deref(), Some("data/sessions"));

        let serialized = serde_json::to_string(&cfg).unwrap();
        let back: RouterConfig = serde_json::from_str(&serialized).unwrap();
        let back_ledger = back.ledger.expect("ledger round-trips");
        assert_eq!(back_ledger.path, ledger.path);
        assert_eq!(back_ledger.model, ledger.model);
        assert_eq!(back_ledger.max_summary_tokens, ledger.max_summary_tokens);
        assert_eq!(back.session.unwrap().root, session.root);
    }

    #[test]
    fn ledger_section_partial_defaults_max_summary_tokens() {
        // A ledger section that omits `max_summary_tokens` gets the named
        // constant default; the shipped config round-trips cleanly.
        let cfg: RouterConfig =
            serde_json::from_value(serde_json::json!({ "ledger": { "model": "swarm" } })).unwrap();
        let ledger = cfg.ledger.as_ref().expect("ledger parsed");
        assert_eq!(ledger.max_summary_tokens, DEFAULT_LEDGER_MAX_SUMMARY_TOKENS);
        assert_eq!(ledger.model.as_deref(), Some("swarm"));
        assert!(ledger.path.is_none());

        let serialized = serde_json::to_string(&cfg).unwrap();
        let back: RouterConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            back.ledger.unwrap().max_summary_tokens,
            DEFAULT_LEDGER_MAX_SUMMARY_TOKENS
        );
    }
}
