//! Router configuration root types - thin facade ownership (M3).
//! This module owns RouterConfig and its directly-associated helpers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use fluent_wvr::{Describable, FieldAccess};

use crate::logging::LoggingConfig;
use crate::config::builder::PipelineParams;
use crate::config::classification::ClassificationTree;
use crate::config::escalation::ModelGroup;
use crate::config::filters::MockConfig;
use crate::config::rounds::{BoundedRounds, EscalationConfidence, SeverityThreshold};
use crate::config::routing::{RoleEntry, RouteRef};

/// Decompose a possibly-qualified model key (`base:qualifier`) into the base
/// `models` config key and the optional qualifier. A bare key (or a malformed
/// `:` split) passes through as `(key, None)`. The qualifier selects a named
/// instance/group of the base model; `latest` is normalized to the entry's
/// default dispatch point by the callers that honor it.
///
/// The typed form is `crate::pipeline::QualifiedModelId` (router-local newtype,
/// `pipeline.rs:12`) — this function is the low-level `&str` parser used by
/// callers that need a zero-alloc split. Prefer `QualifiedModelId::parse` for
/// owned values.
pub(crate) fn split_model_key(key: &str) -> (&str, Option<&str>) {
    match key.split_once(':') {
        Some((base, qualifier)) if !base.is_empty() && !qualifier.is_empty() => {
            (base, Some(qualifier))
        }
        _ => (key, None),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FieldAccess, Describable)]
pub struct RouterConfig {
    /// Named pipeline stage tables (keyed by pipeline name). This is stage
    /// configuration — not flat route authoring — so it stays past M3c: the
    /// tree carries routes/groups/descriptions, never stage knobs
    /// (deterministic prefilter, blacklist, nlp flags, thresholds).
    #[field(skip)]
    #[serde(default)]
    pub pipelines: HashMap<String, PipelineParams>,
    #[field(skip)]
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
    #[field(skip)]
    #[serde(default)]
    pub model_groups: HashMap<String, ModelGroup>,
    /// Routing-vocabulary table: role name → candidate model keys + the
    /// inference point each candidate serves. Absent (the default) leaves
    /// today's key-based routing untouched; roles resolve to concrete
    /// `base:qualifier` targets per request once consumed.
    #[field(skip)]
    #[serde(default)]
    pub roles: HashMap<String, RoleEntry>,
    #[field(desc="safety threshold", min=0.0, max=1.0)]
    #[serde(default)]
    pub safety_threshold: f64,
    #[field(desc="default route")]
    #[serde(default = "default_route")]
    pub default_route: String,
    /// What the classifier stage does when its LLM call fails or its response
    /// cannot be parsed. Safe default: reject rather than route on fabricated
    /// scores.
    #[field(skip)]
    #[serde(default = "default_classifier_failure_policy")]
    pub classifier_failure_policy: ClassifierFailurePolicy,
    #[field(skip)]
    #[serde(default = "ServerConfig::default")]
    pub server: ServerConfig,
    #[field(skip)]
    #[serde(default)]
    pub logging: LoggingConfig,
    #[field(skip)]
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Chart-embedding model key (HNSW index). Selects an entry from
    /// `models`. `None` falls back to `charts.selector_model`, then
    /// `classifier_model`.
    #[field(skip)]
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Chart-candidate reranker model key. Selects an entry
    /// from `models`. `None` skips the rerank stage (Step 2 - Step 3
    /// directly).
    #[field(skip)]
    #[serde(default)]
    pub reranker_model: Option<String>,
    /// ColBERT late-interaction retrieval model key for two-stage retrieval.
    /// Selects an entry from `models` with `task: LateInteraction`. `None`
    /// skips ColBERT re-ranking; when set, HNSW coarse retrieval is
    /// followed by ColBERT MaxSim re-ranking of top-k candidates.
    #[field(skip)]
    #[serde(default)]
    pub retrieval_model: Option<String>,
    #[field(skip)]
    #[serde(default)]
    pub mock: Option<MockConfig>,
    /// Chart store configuration (DAG workflow library).
    #[field(skip)]
    #[serde(default)]
    pub charts: ChartsConfig,
    /// Post-processing configuration.
    #[field(skip)]
    #[serde(default)]
    pub post_process: PostProcessConfig,
    /// Nested classification tree.  `Some` switches the classifier stage
    /// into tree-driven mode; the flat pipeline sections remain for
    /// backward compatibility and are derived from the tree where the rest
    /// of the server needs flat views.
    #[field(skip)]
    #[serde(default)]
    pub classification: Option<ClassificationTree>,
    /// Rigor-route configuration. `None` (the default) leaves the route
    /// present but unconfigured - requests return an explicit `Unconfigured`
    /// error, never a crash.
    #[field(skip)]
    #[serde(default)]
    pub rigor: Option<RigorConfig>,
    /// Sidecar instance-management policy. Governs the sidecar task that
    /// reconciles the fork's shared-weight instances against the configured
    /// profiles, polls `/memory`, and evicts/allocates KV + compute only (the
    /// weights stay loaded in `llama-server`).
    #[field(skip)]
    #[serde(default)]
    pub sidecar: SidecarConfig,
    /// Ledger composition section. `Some` opts the boot path into opening
    /// a `ContentNodeLedger` (with a real `Summarizer` backend targeting
    /// `<base>:ledger`) so LOD derivation exists at runtime. `None` (the
    /// default) leaves today's behavior - no ledger at boot.
    #[field(skip)]
    #[serde(default)]
    pub ledger: Option<LedgerConfig>,
    /// Async review composition section. `Some` opts the boot path into the
    /// `ReviewWorker` (async parse review, correction reuse, parse_review
    /// ledger handoff). `None` (the default) leaves today's behavior — no
    /// review worker at boot.
    #[field(skip)]
    #[serde(default)]
    pub review: Option<ReviewConfig>,
    /// Async overlay composition section (ROADMAP_20260827_ORT §6). `Some`
    /// opts the boot path into the entity-link overlay worker — the async
    /// plane that writes `EntityLink` candidates to `overlay_candidates` (never
    /// a doc-id write). `None` (the default) leaves today's behavior — no
    /// overlay worker at boot.
    #[field(skip)]
    #[serde(default)]
    pub overlay: Option<OverlayConfig>,
    /// Session composition section. `Some` opts the boot path into a
    /// `SessionRegistry` (canonical session home) so rigor rewind and
    /// checkpoint/rewind state exist at runtime. `None` (the default) leaves
    /// today's behavior - no session registry at boot.
    #[field(skip)]
    #[serde(default)]
    pub session: Option<SessionConfig>,
    /// Default "how a model is run" parameters (the `default_params` block).
    /// Applied to every managed model that does not declare the key itself.
    #[field(skip)]
    #[serde(default)]
    pub default_params: DefaultModelParams,
    /// In-process ONNX fleet: one optional role-scoped model declaration per
    /// role (Encoder / PII / Router / Policy / ColBERT). Every role is optional
    /// and the pipeline is fully functional (pure-deterministic) with none of
    /// them loaded. The config vocabulary parallels the llama.cpp
    /// `ModelEntry`/`default_params` surface (resident/pinned residency, run
    /// and idle timeouts, sampling `params`), but the models run in-process via
    /// `ort` — never a spawned `llama-server`. Absent → fully fail-open.
    #[field(skip)]
    #[serde(default)]
    pub onnx: Option<fluent_llm::onnx_config::OnnxFleetConfig>,
    /// Top-level ONNX role keys: an alternative to the nested `onnx` section.
    /// When `onnx` is absent but any of these are present, they are merged
    /// into `onnx` during `apply_defaults()`. This supports the simplified
    /// config format where roles are declared at the root level.
    #[field(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colbert: Option<fluent_llm::onnx_config::OnnxRoleConfig>,
    #[field(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<fluent_llm::onnx_config::OnnxRoleConfig>,
    #[field(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pii: Option<fluent_llm::onnx_config::OnnxRoleConfig>,
    #[field(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<fluent_llm::onnx_config::OnnxRoleConfig>,
    /// ONNX CPU decode concurrency cap (M10): the single `Limiter` budget that
    /// bounds concurrent ONNX decodes (CPU-bound, never blocks the async executor).
    /// Defaults to `DEFAULT_ONNX_LIMITER_CAP` (2).
    #[field(desc = "ONNX CPU decode concurrency cap", min = 1.0, max = 16.0)]
    #[serde(default = "default_onnx_limiter_cap")]
    pub onnx_limiter_cap: usize,
    /// ONNX intra-op threads (M10): per-decode parallelism, forwarded to `ort`
    /// `intra_threads`. Defaults to `DEFAULT_ONNX_THREADS` (1, deterministic decode).
    #[field(desc = "ONNX intra-op threads", min = 1.0, max = 8.0)]
    #[serde(default = "default_onnx_threads")]
    pub onnx_threads: usize,
    /// GGUF model directory for the admin CLI commands (`list`, `scan`, `rm`,
    /// `show`, `pull`, and `ps` weights resolution). Overridable per-invocation
    /// with `--gguf-dir`; `None` falls back to the built-in default.
    #[field(skip)]
    #[serde(default)]
    pub gguf_dir: Option<String>,
    /// The shared inference-backend registry (llama + onnx adapters),
    /// installed by the composition root behind a lock so backends whose
    /// inputs boot later (the llama pool) can register after the first
    /// resolvers run. Not serialized. When present, `local_backend` /
    /// `local_backend_for_instance` resolve through it; when absent (unit
    /// tests, registry-less boots) the legacy construction path serves
    /// instead, byte-identically.
    #[field(skip)]
    #[serde(skip)]
    pub(crate) inference_registry:
        Option<Arc<std::sync::RwLock<fluent_llm::backend::InferenceRegistry>>>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        let mut pipelines = HashMap::new();
        pipelines.insert("default".into(), PipelineParams::default());
        Self {
            pipelines,
            models: HashMap::new(),
            model_groups: HashMap::new(),
            roles: HashMap::new(),
            safety_threshold: 0.5,
            default_route: "local".into(),
            classifier_failure_policy: ClassifierFailurePolicy::Reject,
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
            classifier_model: None,
            embedding_model: None,
            reranker_model: None,
            retrieval_model: None,
            mock: None,
            charts: ChartsConfig::default(),
            post_process: PostProcessConfig::default(),
            classification: None,
            rigor: None,
            sidecar: SidecarConfig::default(),
            ledger: None,
            review: None,
            overlay: None,
            session: None,
            default_params: DefaultModelParams::default(),
            onnx_limiter_cap: common_core::constants::DEFAULT_ONNX_LIMITER_CAP,
            onnx_threads: common_core::constants::DEFAULT_ONNX_THREADS,
            gguf_dir: None,
            onnx: None,
            colbert: None,
            encoder: None,
            pii: None,
            router: None,
            inference_registry: None,
        }
    }
}

impl RouterConfig {
    /// Merge the `default_params` sampling defaults into every model entry that
    /// does not declare its own values (per-model values win). Call once after
    /// config load so the rest of the crate sees fully-materialized params.
    ///
    /// Only the sampling `params` object is merged here — the server-launch
    /// defaults (`batch_size`, KV cache types, GPU offload, context size) are
    /// consumed directly by the supervisor (`build_server_args`).
    pub fn apply_defaults(&mut self) {
        // Merge top-level ONNX role keys into the `onnx` fleet when `onnx` is
        // absent. This supports the simplified config format where roles like
        // `encoder`, `pii`, `router`, `colbert` are declared at the root level
        // instead of nested under `onnx`.
        self.normalize_onnx();

        let Some(default_params) = self.default_params.params.clone() else {
            return;
        };
        let serde_json::Value::Object(defaults) = default_params else {
            return;
        };
        for entry in self.models.values_mut() {
            let Some(serde_json::Value::Object(existing)) = entry.params.as_mut() else {
                entry.params = Some(serde_json::Value::Object(defaults.clone()));
                continue;
            };
            for (key, value) in &defaults {
                existing.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
    }

    /// Merge top-level ONNX role keys into the `onnx` fleet. When `onnx` is
    /// `None` but top-level `encoder`/`pii`/`router`/`colbert` fields are
    /// present, they are collected into an `OnnxFleetConfig`. This supports the
    /// simplified config format where roles are declared at the root level
    /// instead of nested under `onnx`.
    fn normalize_onnx(&mut self) {
        if self.onnx.is_some() {
            // Already has a nested `onnx` section — top-level role keys are
            // redundant and ignored (nested wins).
            if self.encoder.is_some() || self.pii.is_some() || self.router.is_some() || self.colbert.is_some() {
                tracing::warn!(
                    target: "router.config",
                    "both `onnx` section and top-level role keys present; \
                     `onnx` section takes precedence, top-level roles ignored",
                );
            }
            return;
        }
        // Check for any top-level role key.
        let has_roles = self.encoder.is_some()
            || self.pii.is_some()
            || self.router.is_some()
            || self.colbert.is_some();
        if !has_roles {
            return;
        }
        let mut fleet = fluent_llm::onnx_config::OnnxFleetConfig {
            encoder: self.encoder.take(),
            pii: self.pii.take(),
            router: self.router.take(),
            colbert: self.colbert.take(),
            policy: None,
            llm: None,
        };
        // Roles like `router` and `pii` often share the encoder's tokenizer.
        // When a role is missing its tokenizer_path, inherit it from the
        // encoder role. This keeps the simplified config format concise.
        if let Some(ref tok) = fleet.encoder.as_ref().and_then(|e| e.model.tokenizer_path.clone()) {
            let tok = tok.clone();
            for role_cfg in [&mut fleet.pii, &mut fleet.router, &mut fleet.colbert, &mut fleet.policy] {
                if let Some(cfg) = role_cfg.as_mut() {
                    if cfg.model.tokenizer_path.is_none() {
                        cfg.model.tokenizer_path = Some(tok.clone());
                    }
                }
            }
        }
        tracing::info!(
            target: "router.config",
            roles = ?fleet.iter().map(|(r, _)| r.registry_key()).collect::<Vec<_>>(),
            "top-level ONNX role keys merged into fleet config",
        );
        self.onnx = Some(fleet);
    }

    /// Validate tree presence after `apply_defaults` (M3c): configs without a
    /// `classification` tree are rejected fail-fast — flat-only JSON no
    /// longer loads. (Name retained for call-site stability; the flat arms
    /// are gone with the flat fields.)
    pub fn validate_flat_tree_coherence(&self) -> Result<(), String> {
        if self.classification.is_none() {
            return Err("flat config removed, set classification.tree".into());
        }
        Ok(())
    }

    /// The `routes` view the server consumes (model - pipeline mapping).
    ///
    /// Derived solely from the classification tree's `terminal` nodes: each
    /// terminal synthesizes a `RouteRef` routed through its own `group` (or
    /// the route name when no group is given) carrying the terminal's
    /// `always_route`, so `RoutingConfig::resolve_route` and
    /// `resolve_pipeline` work with no structural change to the server.
    pub fn routes_view(&self) -> HashMap<String, RouteRef> {
        let mut routes = HashMap::new();
        if let Some(tree) = &self.classification {
            for (route, group, description) in tree.terminal_views() {
                routes.insert(
                    route.clone(),
                    RouteRef {
                        group: group.unwrap_or_else(|| route.clone()),
                        pipelines: vec!["default".into()],
                        description,
                        always_route: tree.terminal_always_route(&route),
                    },
                );
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
    /// OpenAI-compatible chat-completions endpoint. Optional for managed
    /// (weights/hf_repo/instances) models — Coral Router rewrites it to the
    /// spawned `llama-server`'s address at boot. Required for external OpenAI
    /// endpoints (a model with no `weights`/`hf_repo`/`instances`).
    #[serde(default)]
    pub endpoint: String,
    /// Name of an environment variable holding the `Authorization: Bearer`
    /// token for an external OpenAI endpoint. When set and the variable
    /// resolves, dispatch sends it as the Bearer token; a managed model (one
    /// Coral Router spawns) ignores this. `None` sends no auth header.
    #[serde(default)]
    pub api_key: Option<String>,
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
    /// Whether Coral Router manages this model's lifecycle itself: a dedicated
    /// `llama-server` process (weights/hf_repo/instances). In-process ONNX
    /// models are NOT `ModelEntry`s — they are declared in the top-level
    /// `onnx` role section and managed by the ort registry (see `RouterConfig::onnx`).
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
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
    /// Preserve this context across eviction: when the router must free VRAM,
    /// the context's KV cache is snapshotted (and its session transcript is
    /// already durable in the ledger) before it is dropped, so a later request
    /// can resume it with `snapshot=<name>-resume`. `pinned` contexts are never
    /// evicted, so `resume` is moot on them. Cleared at runtime (explicitly via
    /// `POST /instances/:name/no-resume`, or automatically after
    /// `sidecar.resume_ttl_s` of idle) when Coral Router concludes the work is
    /// done - the snapshot is then deleted.
    #[serde(default)]
    pub resume: bool,
    /// Sampling params merged into the request body for dispatches through this
    /// instance.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Per-profile context-size cap (tokens). `None` = inherit (the global
    /// `default_params.max_ctx`, else no cap). A cap below this profile's
    /// `num_ctx` clamps the context window at materialization
    /// (`ModelEntry::instance_profiles`).
    #[serde(default)]
    pub max_ctx: Option<u64>,
    /// Multi-step instance: holds snapshottable state across requests (the
    /// resume/snapshot path applies). Absent (`false`, the default) means the
    /// profile is one-shot — never snapshotted, and snapshot-scoped request
    /// fields do not apply to its contexts. Declared intent, not a
    /// measurement: the author states the instance carries multi-step state.
    #[serde(default)]
    pub session: bool,
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
        self.instance_profiles_with(None)
    }

    /// Expand over the fleet-wide default profiles merged under this entry's
    /// own `instances`. A model that declares none inherits the whole default
    /// map; a model that declares some keeps them, and any map key present in
    /// both resolves to the entry's profile **whole** — profiles are replaced
    /// per key, never field-merged, so two `default: true` flags can never
    /// fuse silently. `None` (no fleet defaults) is byte-identical to
    /// [`Self::instance_profiles`]: the merge, the `count` expansion, and the
    /// `max_ctx` clamp below all run on one code path, never a fork.
    pub fn instance_profiles_with(
        &self,
        defaults: Option<&HashMap<String, InstanceProfile>>,
    ) -> Vec<InstanceProfile> {
        let mut merged: HashMap<String, InstanceProfile> =
            defaults.cloned().unwrap_or_default();
        if let Some(own) = &self.instances {
            for (key, profile) in own {
                merged.insert(key.clone(), profile.clone());
            }
        }
        // Exactly one profile may carry `default: true`: a second flag is a
        // declaration collision, warned loudly and resolved first-wins in
        // deterministic map order (the same order the expansion below and
        // every `find(|p| p.default)` consumer observe).
        if merged.values().filter(|p| p.default).count() > 1 {
            let mut keys: Vec<&str> = merged
                .iter()
                .filter(|(_, p)| p.default)
                .map(|(k, _)| k.as_str())
                .collect();
            keys.sort_unstable();
            tracing::warn!(
                target: "router.config",
                keys = ?keys,
                "multiple `default: true` instance profiles merged; \
                 the first in map order wins",
            );
        }
        let mut keys: Vec<&String> = merged.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            let profile = &merged[key];
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
                // A profile whose `max_ctx` cap sits below its `num_ctx` is
                // clamped at materialization; absent cap (the default) is a
                // no-op — byte-identical to today's profiles.
                if let Some(cap) = p.max_ctx {
                    if cap < p.num_ctx {
                        p.num_ctx = cap;
                    }
                }
                // One-shot profiles (the default) never carry multi-step
                // state: a `resume: true` on them is inapplicable, forced
                // false fail-open with a loud warn. Session profiles keep
                // their declared `resume` value unchanged.
                if !p.session && p.resume {
                    tracing::warn!(
                        target: "router.config",
                        profile = %key,
                        "resume:true on a one-shot instance profile is inapplicable \
                         (multi-step state needs session:true); forcing resume:false",
                    );
                    p.resume = false;
                }
                p.name = Some(name);
                p.group = Some(group.clone());
                out.push(p);
            }
        }
        out
    }
}

/// The entry-default step of the inference-point precedence: the `default:
/// true` profile's group, else the single shared group across all profiles,
/// else `None` (bare `<base>`). `None` also when no instances are configured.
/// Runs over the fleet-default-merged map, so entries declaring none inherit
/// the fleet default through the same code path. Shared by the single
/// precedence function below and `RoutingTarget` construction (whose entries
/// arrive with fleet defaults materialized, hence `None` there) so backend
/// model ids and dispatch wire ids agree; not a second path — the rule lives
/// here once.
pub(crate) fn default_inference_point(
    entry: &ModelEntry,
    defaults: Option<&HashMap<String, InstanceProfile>>,
) -> Option<String> {
    let profiles = entry.instance_profiles_with(defaults);
    if profiles.is_empty() {
        return None;
    }
    if let Some(d) = profiles.iter().find(|p| p.default) {
        return d.group.clone();
    }
    let first = profiles[0].group.clone()?;
    if profiles
        .iter()
        .all(|p| p.group.as_deref() == Some(first.as_str()))
    {
        Some(first)
    } else {
        None
    }
}

/// Resolve the inference point a role or model key serves, as the qualifier
/// of the dispatch `base:qualifier` id (`None` = bare `<base>`). The single
/// qualifier resolver — every construction path and adapter composes it, so
/// the precedence is documented once, here:
///
/// 1. Explicit qualifier — embedded (`base:point`) or parametric — wins,
///    except `latest`, which normalizes away and falls through.
/// 2. A role's named instance point (`roles[name].instance`).
/// 3. The entry default over the fleet-default-merged map (the `default: true`
///    profile's group, else the single shared group — [`default_inference_point`]).
/// 4. Bare key (`None`): no instances, or no rule matched.
///
/// A bare role name resolves its qualifier through step 2; the role's model
/// key itself (head candidate, config order) is resolved by the caller via
/// [`role_head_key`]. Unknown roles and keys fail closed (`None`); a role
/// with no instance point on a model without a pool stays bare.
#[allow(clippy::implicit_hasher)]
pub fn resolve_inference_point(
    models: &HashMap<String, ModelEntry>,
    roles: &HashMap<String, RoleEntry>,
    role_or_key: &str,
    qualifier: Option<&str>,
    default_instances: Option<&HashMap<String, InstanceProfile>>,
) -> Option<String> {
    let (base, embedded) = split_model_key(role_or_key);
    if let Some(point) = embedded {
        if point != "latest" {
            return Some(point.to_string());
        }
    }
    if let Some(point) = qualifier {
        if point != "latest" {
            return Some(point.to_string());
        }
    }
    if let Some(role) = roles.get(base) {
        if let Some(point) = role.instance.as_deref() {
            return Some(point.to_string());
        }
    }
    models
        .get(base)
        .and_then(|entry| default_inference_point(entry, default_instances))
}

/// Resolve a role or model key to its serving model key: a role name fans out
/// to its head candidate (config order), everything else passes through
/// unchanged. `None` for unknown roles, roles with no candidates, and (when
/// `require_entry` is set) keys with no `models` entry. The companion to
/// [`resolve_inference_point`]: the key identifies *what* serves, the point
/// identifies *where* on it.
#[allow(clippy::implicit_hasher)]
pub fn role_head_key(
    models: &HashMap<String, ModelEntry>,
    roles: &HashMap<String, RoleEntry>,
    role_or_key: &str,
    require_entry: bool,
) -> Option<String> {
    let (base, _) = split_model_key(role_or_key);
    if let Some(role) = roles.get(base) {
        let head = role.models.first()?;
        if require_entry {
            let (head_base, _) = split_model_key(head);
            models.get(head_base)?;
        }
        return Some(head.clone());
    }
    if require_entry {
        models.get(base)?;
    }
    Some(role_or_key.to_string())
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
        self.instance_params_for_with(qualifier, None)
    }

    /// [`Self::instance_params_for`] over the fleet-default-merged profile
    /// map, so a profile inherited from `default_params.instances` contributes
    /// the same sampling knobs as a per-model one. `None` defaults are
    /// byte-identical to [`Self::instance_params_for`].
    pub fn instance_params_for_with(
        &self,
        qualifier: &str,
        defaults: Option<&HashMap<String, InstanceProfile>>,
    ) -> Option<serde_json::Value> {
        let profile = self.instance_profiles_with(defaults).into_iter().find(|p| {
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

/// The classifier's parsed LLM output. `FieldAccess` + the `#[field(...)]`
/// coercions make the struct the single source of truth for the boundary
/// decode path (`fluent_wvr::boundary::decode_boundary`): the `coerce`/`parse`
/// modes shape the raw model value strings exactly as the repair walker does,
/// so both decode paths share one vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize, Default, FieldAccess, Describable)]
pub struct ClassifierOutput {
    #[field(desc = "classifier action", coerce = "strip_quotes,trim")]
    pub action: String,
    #[field(desc = "direct response text", coerce = "strip_quotes,trim")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[field(desc = "routing target", coerce = "strip_quotes,trim")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[field(desc = "coherence score", min = 0.0, max = 1.0, coerce = "strip_quotes,trim", parse = "number")]
    pub coherence_score: f64,
    #[field(desc = "safety score", min = 0.0, max = 1.0, coerce = "strip_quotes,trim", parse = "number")]
    pub safety_score: f64,
    #[field(desc = "complexity", min = 0.0, max = 10.0, coerce = "strip_quotes,trim", parse = "number")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub complexity: Option<u8>,
    #[field(desc = "intent", coerce = "strip_quotes,trim,normalize_literal")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[field(desc = "routing reason", coerce = "strip_quotes,trim")]
    pub reason: String,
    #[field(desc = "completeness", min = 0.0, max = 1.0, coerce = "strip_quotes,trim", parse = "number")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completeness: Option<f64>,
    #[field(desc = "risk", min = 0.0, max = 1.0, coerce = "strip_quotes,trim", parse = "number")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk: Option<f64>,
}

use common_core::constants::default_true;

/// What the classifier stage does when its LLM call fails or its response
/// cannot be parsed. The safe default is `Reject`: the router
/// must never convert a classifier outage into a maximum-confidence dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierFailurePolicy {
    /// Return `StageVerdict::Rejected` with a truthful reason (no fabricated
    /// scores).
    Reject,
    /// Route to the configured default route, but with scores that reflect the
    /// failure (coherence/safety = 0.0) and a `reason` stating the error.
    RouteToDefaultTruthful,
}

/// Safe default for `RouterConfig.classifier_failure_policy`: reject on
/// classifier failure rather than route on fabricated scores.
fn default_classifier_failure_policy() -> ClassifierFailurePolicy {
    ClassifierFailurePolicy::Reject
}

/// The default route when a config omits `default_route`: `local`, matching
/// the shipped `env/coral-router.json` (no `fast` model exists in-tree).
fn default_route() -> String {
    "local".into()
}

// -- Charts (DAG workflow library) configuration --------------------------

/// Chart store configuration - the `charts` section of `RouterConfig`.
///
/// The store is owned by `fluent-router` (see `coral-router`/`charts/`): a
/// directory of human-authored chart JSON files, a router-side
/// `workflow_library` HNSW/SQLite path for retrieval, and the model key
/// used by chart-selection LLM adjudication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartsConfig {
    /// Directory of `*.json` chart files loaded at boot. `None` - empty
    /// store (a missing directory is tolerated with a `warn!`).
    #[serde(default)]
    pub dir: Option<String>,
    /// `workflow_library` HNSW/SQLite file path. The index is built lazily
    /// at boot only when this is set.
    #[serde(default)]
    pub index_path: Option<String>,
    /// Chart-selection classifier model key (LLM adjudication step).
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

// -- Rigor configuration ----------------------------------------------

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
    /// (a `DependencySession` with a `SnapshotStore`). Rewind always resets
    /// steps; this flag only gates the KV-restore expectation.
    #[serde(default)]
    pub kv_cache_enabled: bool,
    /// Max blue/red/judge passes. Fixed round count (VISION: terminate, don't
    /// loop); a material rejection triggers **at most one** re-run of
    /// blue+judge. Default 2.
    #[serde(default)]
    pub max_passes: BoundedRounds,
    /// Objection severity at/above which a judge rejection is **material**
    /// (triggers rewind + the second blue pass). Default 0.7.
    #[serde(default)]
    pub severity_threshold: SeverityThreshold,
    /// Judge confidence below which a final rejection escalates to frontier.
    /// An explicit config value - never "red scored a point".
    /// Default 0.4.
    #[serde(default)]
    pub escalation_confidence: EscalationConfidence,
}

#[allow(clippy::derivable_impls)]
impl Default for RigorConfig {
    fn default() -> Self {
        Self {
            blue_model: None,
            red_model: None,
            judge_model: None,
            kv_cache_enabled: false,
            max_passes: BoundedRounds::default(),
            severity_threshold: SeverityThreshold::default(),
            escalation_confidence: EscalationConfidence::default(),
        }
    }
}

/// Default cap on the ledger `Summarizer`'s summary length (tokens). Only a
/// named constant - `LedgerConfig.max_summary_tokens` defaults to it.
pub const DEFAULT_LEDGER_MAX_SUMMARY_TOKENS: u32 = 200;

/// Ledger composition section - the `ledger` block of `RouterConfig`.
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
    /// Enable continuous background LOD4/LOD5 generation. `false` (the
    /// default) keeps today's lazy-on-demand behavior.
    #[serde(default)]
    pub background_tiering: bool,
    /// Model key for the tier worker's labeler/summarizer. `None` falls back
    /// to the ledger model, then the classifier model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_model: Option<String>,
    /// Max characters for LOD4 (short summary). Default 240 (§0.3).
    #[serde(default = "default_lod4_max_chars")]
    pub lod4_max_chars: usize,
    /// Max characters for LOD5 (description). Default 80 (§0.3).
    #[serde(default = "default_lod5_max_chars")]
    pub lod5_max_chars: usize,
    /// Tier worker batch size (nodes drained per poll).
    #[serde(default = "default_tier_batch_size")]
    pub tier_batch_size: usize,
    /// Tier worker poll interval (ms).
    #[serde(default = "default_tier_poll_interval_ms")]
    pub tier_poll_interval_ms: u64,
    /// Credit granted to the tier feed's producer up front: the max
    /// outstanding `NodeId`s the async (credit-gated) enqueue path may have in
    /// flight before it blocks, bounding a burst of agent turns. Default 256.
    #[serde(default = "default_tier_credit_limit")]
    pub tier_credit_limit: usize,
    /// How many processed nodes the tier worker waits for before bumping
    /// credit back to the producer. Default 8.
    #[serde(default = "default_tier_credit_more_after")]
    pub tier_credit_more_after: usize,
    /// Ledger-agent coordinator section. `enabled = true` opts the boot
    /// path into attaching a `LedgerAgentCoordinator` to the server so a
    /// request with a session + ledger runs through its synchronization loop
    /// (`restore-or-assemble → execute → record → snapshot → enqueue`).
    /// Default-absent so existing deployments are untouched.
    #[serde(default)]
    pub orchestrator: OrchestratorSection,
}

const fn default_ledger_max_summary_tokens() -> u32 {
    DEFAULT_LEDGER_MAX_SUMMARY_TOKENS
}

const fn default_lod4_max_chars() -> usize {
    240
}

const fn default_lod5_max_chars() -> usize {
    80
}

const fn default_tier_batch_size() -> usize {
    8
}

const fn default_tier_poll_interval_ms() -> u64 {
    100
}

const fn default_tier_credit_limit() -> usize {
    256
}

const fn default_tier_credit_more_after() -> usize {
    8
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            path: None,
            model: None,
            max_summary_tokens: DEFAULT_LEDGER_MAX_SUMMARY_TOKENS,
            background_tiering: false,
            tier_model: None,
            lod4_max_chars: default_lod4_max_chars(),
            lod5_max_chars: default_lod5_max_chars(),
            tier_batch_size: default_tier_batch_size(),
            tier_poll_interval_ms: default_tier_poll_interval_ms(),
            tier_credit_limit: default_tier_credit_limit(),
            tier_credit_more_after: default_tier_credit_more_after(),
            orchestrator: OrchestratorSection::default(),
        }
    }
}

/// Async review configuration (ROADMAP §12.7, C4). `None` (the default)
/// disables the review worker and its endpoints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReviewConfig {
    /// Model key for the annotation rung (the parse-producing model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation_model: Option<String>,
    /// Model key for the review model — an **independent, more capable**
    /// tier so review is not self-review (§12.7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_model: Option<String>,
    /// Bounded review-queue capacity (the `WorkerPool` queue cap).
    #[serde(default = "default_review_queue_capacity")]
    pub queue_capacity: usize,
    /// Credit granted up front to the review feed (backpressure, §12.6).
    #[serde(default = "default_review_credit_limit")]
    pub credit_limit: usize,
    /// Model key whose onnx `TokenClassification` session backs the
    /// `PiiSpanDetector` pre-filter (ROADMAP_20260827_ORT §3.2/§3.3). When set
    /// and registered, the ort classifier annotates every review job's text
    /// with PII spans (additive candidates only — never a job drop). `None`
    /// falls back to the deterministic `RegexPiiDetector` when `auto_enqueue`
    /// requires a pre-filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pii_model: Option<String>,
    /// Opt-in PII auto-enqueue (ROADMAP_20260827_ORT §3.4): after a parse is
    /// recorded, PII-shaped spans detected on the request text enqueue a
    /// review candidate, bounded by the existing credit gate. `false` (the
    /// default) keeps today's behavior — review jobs only come from the
    /// explicit `POST /v1/sessions/{id}/review-parse` endpoint.
    #[serde(default)]
    pub auto_enqueue: bool,
    /// Policy-Linter flag threshold (0..1): a text token whose score against a
    /// policy rule clears this is flagged. Default 0.5. Only meaningful when a
    /// `ZeroShotTokenMatching` linter model is registered.
    #[serde(default = "default_review_pii_threshold")]
    pub pii_threshold: f64,
}

fn default_review_queue_capacity() -> usize {
    32
}

fn default_review_credit_limit() -> usize {
    16
}

fn default_review_pii_threshold() -> f64 {
    0.5
}

/// Async overlay configuration (ROADMAP_20260827_ORT §6). `None` (the default)
/// disables the overlay worker and its request-path hook.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OverlayConfig {
    /// Opt-in entity-link overlay (M6.2): when `true`, the request path submits
    /// `EntityLink` residuals (unresolved PROPN spans) to a credit-gated
    /// worker that scores them against boot-cached concept-label embeddings and
    /// writes candidates to `overlay_candidates` (never a doc-id write).
    #[serde(default)]
    pub entity_link_enabled: bool,
    /// Minimum cosine similarity for an entity-link candidate to be accepted.
    #[serde(default = "default_entity_link_threshold")]
    pub entity_link_threshold: f64,
    /// Bounded overlay-queue capacity (the `WorkerPool` queue cap).
    #[serde(default = "default_overlay_queue_capacity")]
    pub queue_capacity: usize,
    /// Credit granted up front to the overlay feed (backpressure).
    #[serde(default = "default_overlay_credit_limit")]
    pub credit_limit: usize,
    /// The opt-in `arc_ready` annotation-overlay sub-config (OVERLAYS §8). `None`
    /// (the default) leaves the three arc_ready overlays off — byte-identical to
    /// a deployment with no `overlay.arc_ready` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arc_ready: Option<ArcReadyConfig>,
}

/// The `arc_ready` annotation-overlay configuration (OVERLAYS §8). Opt-in and
/// additive: every field defaults to off/absent so a config with no
/// `overlay.arc_ready` block is byte-identical to a deployment that never
/// mentions it. The numeric knobs mirror the overlay worker's `OverlayWorkerConfig`
/// defaults (the `CreditedFeedWorker` load-bearing constants).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ArcReadyConfig {
    /// Master switch. `false` (the default) leaves the overlays off.
    #[serde(default)]
    pub enabled: bool,
    /// Whether the spacy pipeline overlay is requested. `false` (the default)
    /// means the `NlpPipeline` seam is not wired → the spacy overlay is
    /// fail-open `Ok(None)`.
    #[serde(default)]
    pub nlp: bool,
    /// Name of the `models` key whose `ChatBackend` drives the LLM enrichment
    /// overlay. `None` (the default) → the LLM overlay is off.
    #[serde(default)]
    pub llm_model: Option<String>,
    /// Name of the `models` key whose `EmbeddingProvider` drives the embedding
    /// overlay. `None` (the default) → the embedding overlay is off.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Bounded pending-node feed capacity (the `CreditedFeedWorker` `mpsc` bound).
    #[serde(default = "default_arc_ready_queue_capacity")]
    pub queue_capacity: usize,
    /// Credit granted up front to the feed's producer (backpressure).
    #[serde(default = "default_arc_ready_credit_limit")]
    pub credit_limit: usize,
    /// How many processed nodes the consumer waits for before bumping credit.
    #[serde(default = "default_arc_ready_credit_more_after")]
    pub credit_more_after: usize,
    /// Max concurrent node-derivations (the `Limiter` cap).
    #[serde(default = "default_arc_ready_max_concurrent")]
    pub max_concurrent: usize,
    /// Whether to boot-backfill nodes already missing an overlay. `false` (the
    /// default) leaves boot behavior unchanged (nodes enqueue on create only).
    #[serde(default)]
    pub backfill: bool,
}

const fn default_arc_ready_queue_capacity() -> usize {
    1024
}

const fn default_arc_ready_credit_limit() -> usize {
    256
}

const fn default_arc_ready_credit_more_after() -> usize {
    8
}

const fn default_arc_ready_max_concurrent() -> usize {
    8
}

fn default_entity_link_threshold() -> f64 {
    0.6
}

fn default_overlay_queue_capacity() -> usize {
    32
}

fn default_overlay_credit_limit() -> usize {
    16
}

/// The `ledger.orchestrator` section: configures the
/// `LedgerAgentCoordinator`'s restore-vs-re-prefill policy, its prompt budget,
/// and the default role recorded for agent output nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorSection {
    /// Whether to attach the coordinator at boot (opt-in). `false` (the
    /// default) leaves the server's dispatch path unchanged.
    #[serde(default)]
    pub enabled: bool,
    /// The restore-vs-re-prefill decision rule for per-model KV snapshots.
    #[serde(default)]
    pub kv_policy: crate::dag_session::KvSnapshotPolicy,
    /// The worker's context-window budget (characters) for prompt assembly.
    /// Default 32768 (8192 tokens × 4 chars/token).
    #[serde(default = "default_orchestrator_prompt_budget_chars")]
    pub prompt_budget_chars: usize,
    /// Default role recorded for agent output nodes.
    #[serde(default = "default_orchestrator_role")]
    pub role: String,
    /// Optional concurrency cap for the coordinator's KV-affinity scheduler.
    /// `Some(cap)` attaches an `AffinityScheduler` bounded by `cap` concurrent
    /// agent turns: the active session's turns get a priority bonus (minimize
    /// context switches) while starved sessions age up. `None` (the default)
    /// leaves affinity bookkeeping off — existing deployments are untouched.
    #[serde(default)]
    pub affinity_cap: Option<usize>,
}

const fn default_orchestrator_prompt_budget_chars() -> usize {
    32768
}

fn default_orchestrator_role() -> String {
    "agent".into()
}

impl Default for OrchestratorSection {
    fn default() -> Self {
        Self {
            enabled: false,
            kv_policy: crate::dag_session::KvSnapshotPolicy::RestoreIfSameModel,
            prompt_budget_chars: default_orchestrator_prompt_budget_chars(),
            role: default_orchestrator_role(),
            affinity_cap: None,
        }
    }
}

/// Session composition section - the `session` block of `RouterConfig`.
///
/// `Some` opts the composition root into a `SessionRegistry` (canonical
/// session home) so checkpoint/rewind state and rigor rewind exist at runtime.
/// `None` (absent) keeps today's behavior - no session registry at boot.
/// Default model run parameters - the top-level `default_params` block.
///
/// Supplies the "how a model is run" defaults applied to every managed model
/// that does not declare the key itself: the `llama-server` launch knobs
/// (`--batch-size`, `--ubatch-size`, `--cache-type-k/v`, `--flash-attn`,
/// `--n-gpu-layers`, `--n-cpu-moe`, `--sleep-idle-seconds`, `--ctx-size`) and
/// the sampling `params` merged into dispatch bodies (per-model values win).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultModelParams {
    /// Default context size in tokens (`--ctx-size`; also `ctx_size` alias).
    #[serde(default = "default_num_ctx", alias = "ctx_size")]
    pub num_ctx: u64,
    /// Logical maximum batch size (`--batch-size`).
    #[serde(default = "default_batch_size")]
    pub batch_size: u64,
    /// Physical maximum batch size (`--ubatch-size`).
    #[serde(default = "default_ubatch_size")]
    pub ubatch_size: u64,
    /// KV cache data type for K (`--cache-type-k`).
    #[serde(default = "default_cache_type")]
    pub cache_type_k: String,
    /// KV cache data type for V (`--cache-type-v`).
    #[serde(default = "default_cache_type")]
    pub cache_type_v: String,
    /// Flash attention mode (`--flash-attn on|off|auto`); `None` keeps the
    /// fork default.
    #[serde(default)]
    pub flash_attn: Option<String>,
    /// Max layers stored in VRAM (`--n-gpu-layers`).
    #[serde(default = "default_n_gpu_layers")]
    pub n_gpu_layers: i32,
    /// MoE expert layers kept in CPU RAM (`--n-cpu-moe`).
    #[serde(default)]
    pub n_cpu_moe: i32,
    /// Idle timeout after which the fork sleeps an instance
    /// (`--sleep-idle-seconds`). Only emitted for plain (no-instance) models;
    /// instance pools own residency through the sidecar.
    #[serde(default = "default_sleep_idle_seconds")]
    pub sleep_idle_seconds: i32,
    /// Whether dispatches through models without an explicit `stream` stream.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Whether dispatches through models without an explicit `filter_thinking`
    /// strip thinking blocks.
    #[serde(default)]
    pub filter_thinking: bool,
    /// Default sampling params merged into dispatch bodies (per-model wins).
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Global context-size cap (tokens) applied to every managed model that
    /// does not declare its own `max_ctx`. `None` = no cap (existing behavior —
    /// a model's `num_ctx`/`ctx_size` is the sole bound).
    #[serde(default)]
    pub max_ctx: Option<u64>,
    /// Fleet-wide instance-profile map. A model entry that declares its own
    /// `instances` keeps them; one that declares none inherits this map
    /// (whole-profile replace per map key, never field merge). `None` (the
    /// default) leaves today's per-model declarations untouched.
    #[serde(default)]
    pub instances: Option<HashMap<String, InstanceProfile>>,
}

impl Default for DefaultModelParams {
    fn default() -> Self {
        Self {
            num_ctx: default_num_ctx(),
            batch_size: default_batch_size(),
            ubatch_size: default_ubatch_size(),
            cache_type_k: default_cache_type(),
            cache_type_v: default_cache_type(),
            flash_attn: None,
            n_gpu_layers: default_n_gpu_layers(),
            n_cpu_moe: 0,
            sleep_idle_seconds: default_sleep_idle_seconds(),
            stream: default_true(),
            filter_thinking: false,
            params: None,
            max_ctx: None,
            instances: None,
        }
    }
}

const fn default_num_ctx() -> u64 {
    16384
}

const fn default_batch_size() -> u64 {
    4096
}

const fn default_ubatch_size() -> u64 {
    1024
}

fn default_cache_type() -> String {
    "q8_0".into()
}

fn default_onnx_limiter_cap() -> usize {
    common_core::constants::DEFAULT_ONNX_LIMITER_CAP
}

fn default_onnx_threads() -> usize {
    common_core::constants::DEFAULT_ONNX_THREADS
}

const fn default_n_gpu_layers() -> i32 {
    999
}

const fn default_sleep_idle_seconds() -> i32 {
    15
}

/// Session composition section - the `session` block of `RouterConfig`.
///
/// `Some` opts the composition root into a `SessionRegistry` (canonical
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

/// Sidecar instance-management policy.
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
    /// Free-VRAM floor (bytes) that must remain unallocated: the effective
    /// allocation limit is `device_total - minimum_remaining_vram`. When
    /// `vram_total_bytes` is `None`, the device total is detected at boot
    /// (ROCm `mem_info_vram_total`); `minimum_remaining_vram` then alone
    /// enables the residency eviction budget.
    #[serde(default)]
    pub minimum_remaining_vram: Option<u64>,
    /// Slot-save directory the fork writes KV snapshots under
    /// (`<slot_save_path>/<model_key>/`). Feeds snapshot-path derivation.
    #[serde(default)]
    pub slot_save_path: Option<String>,
    /// Resume snapshots older than this many seconds of context idle are
    /// dropped and their contexts' `resume` flag cleared: the router's signal
    /// that an evicted workload is done and need not be restorable. `None`
    /// keeps resume snapshots until explicitly disabled. The flag also feeds
    /// the `-resume` snapshot naming the router uses on eviction.
    #[serde(default)]
    pub resume_ttl_s: Option<u64>,
    /// Env var naming the management API key sent as `Authorization: Bearer`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Post-boot liveness poll: how often the supervision task probes a running
    /// server's `/health` (seconds). A server that stays alive but stops
    /// answering `liveness_failures_before_restart` consecutive probes is
    /// killed and restarted.
    #[serde(default = "default_sidecar_liveness_poll_s")]
    pub liveness_poll_interval_s: u64,
    /// Consecutive failed `/health` probes before a hung server is killed and
    /// restarted.
    #[serde(default = "default_sidecar_liveness_failures")]
    pub liveness_failures_before_restart: u32,
    /// Consecutive crashes (spawn failures or boot-time child exits) after
    /// which the supervisor stops restarting a model's `llama-server` and
    /// marks it **failed** (containment, per the fluent-concurrency
    /// supervision contract — no endless crash loop). `ensure_running` then
    /// returns a terminal error until the router restarts or the model is
    /// unloaded, at which point a fresh (bounded) load attempt is allowed.
    /// The count resets the moment a server answers `/health`, so a crash
    /// after a healthy period is a fresh failure. `0` disables the limit
    /// (unbounded restart with rising backoff).
    #[serde(default = "default_sidecar_max_restarts")]
    pub max_restarts: u32,
    /// The onnx fleet's working-set budget (bytes): when Σ resident bytes of
    /// loaded onnx sessions exceeds it, the onnx residency loop releases the
    /// LRU-largest `Unloadable` sessions. `None` (the default) → idle-only
    /// eviction — CPU RAM is cheap and the parity target is idle unload, not
    /// a tight budget.
    #[serde(default)]
    pub onnx_working_set_budget_bytes: Option<u64>,
    /// Persisted fleet map (`{model: {port, pid}}` for every running server)
    /// the supervisor writes at boot and reads back on the next boot so an
    /// orphaned `llama-server` (router killed without graceful shutdown) is
    /// adopted instead of duplicated. `None` (the default) disables
    /// persistence — adoption falls back to the `/proc` scan alone.
    #[serde(default)]
    pub server_state_path: Option<String>,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            poll_interval_s: default_sidecar_poll_interval_s(),
            vram_low_watermark_bytes: default_sidecar_watermark(),
            evict_batch: default_sidecar_evict_batch(),
            vram_total_bytes: None,
            minimum_remaining_vram: None,
            slot_save_path: None,
            resume_ttl_s: None,
            api_key_env: None,
            liveness_poll_interval_s: default_sidecar_liveness_poll_s(),
            liveness_failures_before_restart: default_sidecar_liveness_failures(),
            max_restarts: default_sidecar_max_restarts(),
            onnx_working_set_budget_bytes: None,
            server_state_path: None,
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

const fn default_sidecar_liveness_poll_s() -> u64 {
    30
}

const fn default_sidecar_liveness_failures() -> u32 {
    3
}

const fn default_sidecar_max_restarts() -> u32 {
    5
}

/// Detect the device VRAM total (bytes) from the ROCm sysfs interface. Returns
/// the first non-zero `mem_info_vram_total` found under `/sys/class/drm`. Used
/// when `sidecar.vram_total_bytes` is unset so a `minimum_remaining_vram`
/// budget alone can drive the residency loop. `None` when the interface is
/// absent (non-ROCm hosts).
pub fn detect_device_vram_total() -> Option<u64> {
    let entries = fluent_wvr::capability::capability_aware_fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let path = entry.path().join("device/mem_info_vram_total");
        let text = fluent_wvr::capability::capability_aware_fs::read_to_string(path).ok()?;
        let total = text.trim().parse::<u64>().ok()?;
        if total > 0 {
            return Some(total);
        }
    }
    None
}

impl SidecarConfig {
    /// The device VRAM total: the explicit `vram_total_bytes` ceiling first,
    /// else the ROCm sysfs detection. `None` when neither is available.
    pub fn device_total_bytes(&self) -> Option<u64> {
        self.vram_total_bytes.or_else(detect_device_vram_total)
    }

    /// The effective VRAM allocation budget: `device_total - minimum_remaining
    /// _vram`. `None` when no device total is available, or when neither a
    /// ceiling nor a minimum-remaining floor is configured (eviction off).
    pub fn allocation_limit(&self) -> Option<u64> {
        let total = self.device_total_bytes()?;
        let min_remaining = self.minimum_remaining_vram.unwrap_or(0);
        Some(total.saturating_sub(min_remaining))
    }
}

const fn default_charts_min_score() -> f64 {
    0.6
}

const fn default_charts_entity_context() -> bool {
    true
}

// -- Post-processing configuration ---------------------

/// Post-processing configuration - the `post_process` section of
/// `RouterConfig`.
///
/// Controls the VISION learning loop: whether a *successful* dispatch is
/// distilled into a reusable draft chart. Per VISION -"Post-processing:
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

/// Extraction scope for the learning loop (see `PostProcessConfig`).
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
    fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS
}

fn default_idle_timeout_ms() -> u64 {
    fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS
}

fn default_retry_interval() -> u64 {
    fluent_llm::constants::DEFAULT_RETRY_INTERVAL_S
}
#[cfg(test)]
#[path = "../../tests/config_root.rs"]
mod tests;
