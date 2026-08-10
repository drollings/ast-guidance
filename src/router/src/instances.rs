//! Instance-pool grammar generation, management client, sidecar, and the
//! router's aggregate `/instances` facade.
//!
//! Coral Router is the process owner of one `llama-server` per model weights
//! file (see `supervisor`): it spawns each server on a free localhost port and
//! talks to it DIRECTLY - the llama.cpp router mode is never used. This module
//! declares the instance pool it hands each server as `--instance` grammar
//! (matching `common_instances_parse`/`common_instances_to_string`), wraps the
//! server's management API (`/instances`), and hosts the sidecar:
//!
//! - [`InstanceClient`] - one server's `/instances` management API over raw
//!   reqwest, with `HttpClass`-classified errors.
//! - [`InstanceManager`] - boot reconciliation, the residency loop (LRU
//!   eviction of unpinned instances when free VRAM is low), and
//!   allocate-on-503.
//! - [`InstancePool`] - the router's public `/instances` facade: aggregates
//!   every managed server's envelope under `<model_id>:<instance_name>` ids,
//!   sums `total` with 64-bit arithmetic, and proxies per-model operations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use common_core::hash::uuid_v4;
use crate::config::InstanceProfile;

use fluent_llm::HttpClass;

/// Render a flat list of (expanded) `InstanceProfile`s as the fork's
/// comma-joined `--instance` grammar, matching `common_instances_to_string`
/// byte-for-byte for equivalent inputs.
///
/// Grammar (the minimal branch spec): `name[:group=G][:ctx=N][:parallel=M]
/// [:pinned][:default]`. There is no sleep component: the branch has no
/// auto-sleep, so `sleep_idle_seconds` never reaches the server - it is an
/// eviction-priority hint the sidecar reads from config.
pub fn instance_grammar_string(profiles: &[InstanceProfile]) -> String {
    let parts: Vec<String> = profiles.iter().map(render_one).collect();
    parts.join(",")
}

fn render_one(profile: &InstanceProfile) -> String {
    let name = profile.name.clone().unwrap_or_default();
    let mut s = name.clone();

    if let Some(group) = &profile.group {
        if *group != name {
            s.push_str(":group=");
            s.push_str(group);
        }
    }

    if profile.num_ctx > 0 {
        s.push_str(":ctx=");
        s.push_str(&profile.num_ctx.to_string());
    }

    if let Some(parallel) = profile.parallel {
        if parallel > 0 {
            s.push_str(":parallel=");
            s.push_str(&parallel.to_string());
        }
    }

    if profile.pinned {
        s.push_str(":pinned");
    }

    if profile.default {
        s.push_str(":default");
    }

    s
}

/// Validate a flat instance list the way the fork's parser does: no duplicate
/// names, and no instance's group colliding with another instance's name.
/// The group==own-name default is permitted.
pub fn validate_instances(profiles: &[InstanceProfile]) -> Result<(), String> {
    for (i, pi) in profiles.iter().enumerate() {
        let ni = pi.name.as_deref().unwrap_or("");
        let gi = pi.group.as_deref().unwrap_or("");
        for other in profiles.iter().skip(i + 1) {
            let nj = other.name.as_deref().unwrap_or("");
            let gj = other.group.as_deref().unwrap_or("");
            if !ni.is_empty() && ni == nj {
                return Err(format!("duplicate instance name '{ni}'"));
            }
            if (!gi.is_empty() && gi == nj) || (!gj.is_empty() && gj == ni) {
                return Err(format!(
                    "instance group '{gi}' collides with instance name '{nj}'"
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Management client - the fork's management API over raw reqwest
// ---------------------------------------------------------------------------

/// Derive the management base URL from a model's chat-completions endpoint:
/// `http://host:port/v1/chat/completions` -> `http://host:port`. The management
/// endpoints (`/instances`, `/memory`, ...) live at the same host as the
/// generation endpoint.
pub fn management_base_url(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    for suffix in ["/v1/chat/completions", "/chat/completions"] {
        if let Some(base) = trimmed.strip_suffix(suffix) {
            return base.trim_end_matches('/').to_string();
        }
    }
    trimmed.to_string()
}

/// Classification of a management-API failure, mirroring `HttpClass` with the
/// extra `Duplicate` (409) and evict-trigger (507/503) distinctions the sidecar
/// cares about.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InstanceError {
    /// 429/503/504/507/other 5xx - transient; a 507/503 also signals an
    /// allocation/eviction trigger.
    #[error("transient management error: {status} {body}")]
    Transient { status: u16, body: String },
    /// 409 duplicate name - tolerated during reconciliation.
    #[error("duplicate instance (409)")]
    Duplicate,
    /// Permanent 4xx (except 409) - no retry.
    #[error("management request rejected: {status} {body}")]
    Rejected { status: u16, body: String },
    /// Transport / network failure before an HTTP status was received.
    #[error("management network error: {0}")]
    Network(String),
    /// A 2xx whose payload did not match the expected shape.
    #[error("management response parse error: {0}")]
    Other(String),
}

impl InstanceError {
    /// Whether the failure merits a retry with backoff.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            InstanceError::Transient { .. } | InstanceError::Network(_)
        )
    }

    /// Whether this is a 409 duplicate-name collision (tolerable in reconcile).
    pub fn is_duplicate(&self) -> bool {
        matches!(self, InstanceError::Duplicate)
    }

    /// Whether a low-memory allocation (507) or 503 signals the residency loop
    /// should consider evicting.
    pub fn is_evict_trigger(&self) -> bool {
        matches!(
            self,
            InstanceError::Transient { status, .. } if *status == 503 || *status == 507
        )
    }
}

/// One instance as reported by `GET /instances` (the branch-spec envelope).
/// `state` is always `"loaded"` - an instance is either allocated (and
/// immediately ready) or deleted; there is no sleeping state. `model_bytes` is
/// the shared weights, reported on the first loaded instance only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub group: String,
    #[serde(default)]
    pub n_ctx: u64,
    #[serde(default)]
    pub parallel: u32,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub is_default: bool,
    /// Router-side preserve-on-evict flag: when set, the router snapshots this
    /// context's KV (and keeps its ledger transcript durable) before it drops
    /// the context to free VRAM, so a later request can resume it with
    /// `snapshot=<name>-resume`. Not understood by the fork - Coral Router
    /// tracks it and overlays it on the aggregate envelope.
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub model_bytes: u64,
    #[serde(default)]
    pub context_bytes: u64,
    #[serde(default)]
    pub compute_bytes: u64,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub vram_bytes: u64,
    /// Most recent slot use (`-1` when never used).
    #[serde(default = "default_never_used")]
    pub last_used: i64,
}

const fn default_never_used() -> i64 {
    -1
}

/// The summed memory `total` object of `GET /instances`. `model` counts a
/// model's weights once regardless of how many of its instances are loaded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceTotals {
    #[serde(default)]
    pub model: u64,
    #[serde(default)]
    pub context: u64,
    #[serde(default)]
    pub compute: u64,
    #[serde(default)]
    pub total: u64,
}

/// The full `GET /instances` envelope: the running instances, the pool's
/// on-disk KV snapshots (even when no instance is loaded), and the summed
/// memory. A cold pool returns `instances: []` and a zeroed `total`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceList {
    #[serde(default)]
    pub instances: Vec<InstanceInfo>,
    #[serde(default)]
    pub snapshots: Vec<SnapshotInfo>,
    #[serde(default)]
    pub total: InstanceTotals,
}

/// One snapshot entry from `GET /instances` or `GET /instances/:name/snapshots`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SnapshotInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub mtime: serde_json::Value,
    #[serde(default)]
    pub n_ctx_seq: u64,
}

/// Typed management client against one spawned `llama-server`'s `/instances`
/// API. Talks DIRECTLY to the pool's server - no router routing, so no `model`
/// field is carried (each server owns exactly one model's weights).
///
/// Mirrors the raw-reqwest pattern of `OpenAiChatBackend`: a plain
/// `reqwest::Client` and explicit status classification via `HttpClass`.
#[derive(Clone)]
pub struct InstanceClient {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl InstanceClient {
    pub fn new(
        client: reqwest::Client,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            base_url: base_url.into(),
            api_key,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Issue a management request, classify the status, and return the parsed
    /// JSON on 2xx. 409 -> `Duplicate`; 429/503/504/507/5xx -> `Transient`;
    /// other 4xx -> `Rejected`.
    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, InstanceError> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let mut builder = self.client.request(method, &url);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        if let Some(body) = body {
            builder = builder.json(body);
        }

        let response = builder
            .send()
            .await
            .map_err(|e| InstanceError::Network(e.to_string()))?;
        let status = response.status();
        if status.is_success() {
            return response
                .json()
                .await
                .map_err(|e| InstanceError::Other(e.to_string()));
        }
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| String::new());
        let status_u16 = status.as_u16();
        if status_u16 == 409 {
            return Err(InstanceError::Duplicate);
        }
        // Classify via the shared `HttpClass` taxonomy; a 507 (low device
        // memory) is an evict trigger like 503.
        let class = HttpClass::from_status(status_u16);
        if class.is_retryable() || status_u16 == 507 {
            Err(InstanceError::Transient {
                status: status_u16,
                body,
            })
        } else {
            Err(InstanceError::Rejected {
                status: status_u16,
                body,
            })
        }
    }

    /// `GET /instances` - the current instance set + snapshots + summed memory.
    /// Tolerates a bare array (pre-envelope forks) and a bare object.
    pub async fn list(&self) -> Result<InstanceList, InstanceError> {
        let value = self
            .request(reqwest::Method::GET, "/instances", None)
            .await?;
        if let Some(arr) = value.as_array() {
            let instances: Vec<InstanceInfo> =
                serde_json::from_value(Value::Array(arr.clone()))
                    .map_err(|e| InstanceError::Other(format!("list: {e}")))?;
            return Ok(InstanceList {
                instances,
                snapshots: vec![],
                total: InstanceTotals::default(),
            });
        }
        serde_json::from_value(value)
            .map_err(|e| InstanceError::Other(format!("list: {e}")))
    }

    /// `POST /instances` - allocate a NEW context from the shared weights.
    /// Only KV + compute are allocated; the model weights stay loaded.
    /// `default` marks the target of a bare `<base>` request. Returns the
    /// created instance on 201. A 2xx whose body is not a full instance object
    /// (an older fork returning `{}`) degrades to a synthesized record so the
    /// caller still has the instance identity.
    pub async fn create(
        &self,
        name: &str,
        group: &str,
        ctx_size: u64,
        parallel: Option<u32>,
        pinned: bool,
        is_default: bool,
    ) -> Result<InstanceInfo, InstanceError> {
        let mut body = serde_json::json!({
            "name": name,
            "group": group,
            "ctx_size": ctx_size,
            "pinned": pinned,
            "default": is_default,
        });
        if let Some(parallel) = parallel {
            body["parallel"] = Value::Number(parallel.into());
        }
        let value = self
            .request(reqwest::Method::POST, "/instances", Some(&body))
            .await?;
        Ok(serde_json::from_value::<InstanceInfo>(value).unwrap_or_else(|_| InstanceInfo {
            id: name.to_string(),
            aliases: vec![],
            group: group.to_string(),
            n_ctx: ctx_size,
            parallel: parallel.unwrap_or(1),
            pinned,
            is_default,
            resume: false,
            state: "loaded".into(),
            model_bytes: 0,
            context_bytes: 0,
            compute_bytes: 0,
            total_bytes: 0,
            vram_bytes: 0,
            last_used: -1,
        }))
    }

    /// `DELETE /instances/:name` - free KV + compute (the primary eviction
    /// path). `force` is accepted for compatibility and ignored (nothing
    /// enforces `pinned` in the branch).
    pub async fn destroy(&self, name: &str, force: bool) -> Result<(), InstanceError> {
        let path = if force {
            format!("/instances/{name}?force=true")
        } else {
            format!("/instances/{name}")
        };
        self.request(reqwest::Method::DELETE, &path, None)
            .await
            .map(|_| ())
    }

    /// `POST /instances/:name/pin` - set the advisory `pinned` flag.
    pub async fn pin(&self, name: &str) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/pin"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/unpin` - clear the advisory `pinned` flag.
    pub async fn unpin(&self, name: &str) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/unpin"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/resize` - destroy and re-create the context at a
    /// new size (the current KV and any snapshot binding are discarded).
    pub async fn resize(&self, name: &str, ctx_size: u64) -> Result<(), InstanceError> {
        let body = serde_json::json!({ "ctx_size": ctx_size });
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/resize"),
            Some(&body),
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/snapshot` - save the slot-0 KV to a named snapshot.
    pub async fn save_snapshot(&self, instance: &str, name: &str) -> Result<(), InstanceError> {
        let body = serde_json::json!({ "name": name });
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{instance}/snapshot"),
            Some(&body),
        )
        .await
        .map(|_| ())
    }

    /// `GET /instances/:name/snapshots` - list the instance's snapshots. The
    /// server wraps them as `{"snapshots": [...]}`; a bare array is tolerated.
    pub async fn list_snapshots(&self, instance: &str) -> Result<Vec<SnapshotInfo>, InstanceError> {
        let value = self
            .request(
                reqwest::Method::GET,
                &format!("/instances/{instance}/snapshots"),
                None,
            )
            .await?;
        let arr = value.get("snapshots").cloned().unwrap_or(value);
        serde_json::from_value(arr)
            .map_err(|e| InstanceError::Other(format!("list_snapshots: {e}")))
    }

    /// `DELETE /instances/:name/snapshot/:snapshot` - remove a snapshot file.
    pub async fn delete_snapshot(
        &self,
        instance: &str,
        name: &str,
    ) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::DELETE,
            &format!("/instances/{instance}/snapshot/{name}"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `GET /props` - the server's property envelope. Best-effort: `None` when
    /// the endpoint is unreachable or returns a non-2xx. A plain (no-instance
    /// grammar) server exposes no `/instances`, but `/props` still reports the
    /// context size and idle-sleep state Coral Router uses to synthesize its
    /// resident footprint.
    pub async fn props(&self) -> Option<Value> {
        self.request(reqwest::Method::GET, "/props", None).await.ok()
    }
}

// ---------------------------------------------------------------------------
// InstanceManager - sidecar: reconcile, residency, allocate-on-503
// ---------------------------------------------------------------------------

/// The sidecar owner of instance lifecycle for ONE spawned `llama-server`.
/// Holds the model key (public id), the management client talking directly to
/// that server, the expanded configured profiles, and the residency policy.
/// Runs as a task on the router's tokio runtime (owned by the server).
pub struct InstanceManager {
    model_key: String,
    client: InstanceClient,
    profiles: Vec<InstanceProfile>,
    policy: crate::config::SidecarConfig,
    /// Resident weights bytes of a plain (no-instance-grammar) model, from the
    /// configured `weights` file. Instance models report `model_bytes` through
    /// the fork's `/instances`; plain models need this to surface in the
    /// aggregate envelope and the residency budget.
    weights_bytes: u64,
    /// Router-tracked last-use for plain models (the fork reports no
    /// `last_used` for them). Updated by [`Self::touch`] on every dispatch; the
    /// residency loop orders plain-model unloads by it.
    last_used: AtomicI64,
    /// Router-side preserve-on-evict map: instance name -> `resume`. Seeded
    /// from the configured profiles, updated by [`Self::set_resume`]. The fork
    /// knows nothing of it; the aggregate overlays it on the envelope.
    resume: Mutex<HashMap<String, bool>>,
}

impl InstanceManager {
    pub fn new(
        model_key: impl Into<String>,
        client: InstanceClient,
        profiles: Vec<InstanceProfile>,
        policy: crate::config::SidecarConfig,
    ) -> Self {
        let resume = profiles
            .iter()
            .filter_map(|p| p.name.as_ref().map(|n| (n.clone(), p.resume)))
            .collect();
        Self {
            model_key: model_key.into(),
            client,
            profiles,
            policy,
            weights_bytes: 0,
            last_used: AtomicI64::new(-1),
            resume: Mutex::new(resume),
        }
    }

    /// Builder-style: set the resident weights size of a plain (no-instance)
    /// model so the aggregate and residency loops can report it.
    #[must_use]
    pub fn with_weights_bytes(mut self, bytes: u64) -> Self {
        self.weights_bytes = bytes;
        self
    }

    /// Whether this manager's model declares an instance pool. Only instance
    /// models expose `/instances` on their server; a plain (weights-only)
    /// model's server 404s on it and needs a synthesized footprint instead.
    pub fn has_pool(&self) -> bool {
        !self.profiles.is_empty()
    }

    /// The resident weights size of this model (from the configured weights
    /// file). For instance models the fork reports `model_bytes` itself; this
    /// is the size a cold load needs and the plain-model footprint uses.
    pub fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }

    /// Whether the named instance is marked to be preserved (KV snapshotted +
    /// ledger transcript) across eviction.
    pub fn resume_for(&self, name: &str) -> bool {
        self.resume
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
            .copied()
            .unwrap_or(false)
    }

    /// Set (or clear) the preserve-on-evict flag for a named instance.
    pub fn set_resume(&self, name: &str, enabled: bool) {
        self.resume
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_string(), enabled);
    }

    /// Whether the fork currently has this model's weights slept out of VRAM
    /// (`is_sleeping` in `/props`). `None` when the server is unreachable.
    /// Instance models never report sleeping (pinned contexts keep their
    /// weights resident), so `Some(false)` short-circuits without a call.
    pub async fn is_sleeping(&self) -> Option<bool> {
        if self.has_pool() {
            return Some(false);
        }
        self.client
            .props()
            .await?
            .get("is_sleeping")
            .and_then(Value::as_bool)
    }

    /// Record a dispatch to this model so residency can order plain models by
    /// recency (the fork reports no `last_used` for them).
    pub fn touch(&self) {
        self.last_used.store(
            i64::try_from(common_core::now_secs()).unwrap_or(i64::MAX),
            Ordering::Relaxed,
        );
    }

    /// The resident footprint of a plain (no-instance-grammar) managed model.
    ///
    /// The fork exposes no `/instances` for these servers, so Coral Router
    /// synthesizes one envelope entry: `model_bytes` is the configured weights
    /// file size, or 0 when the server reports `is_sleeping` (the fork's idle
    /// sleep has moved the weights out of VRAM). `state` mirrors that flag.
    /// `None` when the server is unreachable (down or never loaded).
    async fn plain_footprint(&self) -> Option<InstanceInfo> {
        if self.has_pool() {
            return None;
        }
        let props = self.client.props().await?;
        let asleep = props
            .get("is_sleeping")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let n_ctx = props
            .get("default_generation_settings")
            .and_then(|g| g.get("n_ctx"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let model_bytes = if asleep { 0 } else { self.weights_bytes };
        let state = if asleep { "sleeping" } else { "loaded" }.to_string();
        Some(InstanceInfo {
            id: format!("{}:default", self.model_key),
            aliases: vec![],
            group: "default".into(),
            n_ctx,
            parallel: 1,
            pinned: false,
            is_default: true,
            resume: false,
            state,
            model_bytes,
            // No context windows are reported for a plain server; the resident
            // footprint is the shared weights alone. `vram_bytes` follows the
            // contract (context + compute, excluding weights) and stays 0.
            context_bytes: 0,
            compute_bytes: 0,
            total_bytes: model_bytes,
            vram_bytes: 0,
            last_used: self.last_used.load(Ordering::Relaxed),
        })
    }

    /// List this manager's instances, synthesizing a resident footprint when
    /// the server is a plain (no-instance-grammar) model (its `/instances`
    /// 404s). Returns `(envelope, plain)`, where `plain` is `true` when the
    /// envelope is the synthesized footprint rather than the fork's report.
    /// `None` when the server is unreachable (down or never loaded).
    async fn list_with_fallback(&self) -> Option<(InstanceList, bool)> {
        match self.client.list().await {
            Ok(envelope) => Some((envelope, false)),
            Err(InstanceError::Rejected { status: 404, .. }) => {
                let footprint = self.plain_footprint().await?;
                let total = InstanceTotals {
                    model: footprint.model_bytes,
                    context: 0,
                    compute: 0,
                    total: footprint.total_bytes,
                };
                Some((
                    InstanceList {
                        instances: vec![footprint],
                        snapshots: vec![],
                        total,
                    },
                    true,
                ))
            }
            Err(_) => None,
        }
    }

    /// The Coral Router model id this manager's server belongs to.
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    pub fn client(&self) -> &InstanceClient {
        &self.client
    }

    pub fn profiles(&self) -> &[InstanceProfile] {
        &self.profiles
    }

    /// Boot reconciliation: create **pinned** configured instances missing from
    /// `GET /instances`, resize `n_ctx` mismatches, and warn on
    /// `parallel`/`pinned` drift. A duplicate-create (409) is tolerated.
    /// Unpinned instances are NOT created here — they are created on demand by
    /// [`Self::ensure_instance`] (the residency goal is that only pinned
    /// instances stay resident). Emits an audit record of the result.
    ///
    /// A plain model (no instance profiles) has nothing to reconcile — returns
    /// `Ok` without touching the management API (a plain server exposes no
    /// `/instances`).
    pub async fn reconcile(&self) -> Result<(), InstanceError> {
        if self.profiles.is_empty() {
            return Ok(());
        }
        let existing = match self.client.list().await {
            Ok(envelope) => envelope,
            Err(e) => {
                tracing::warn!(
                    target: "router.instances",
                    base_url = %self.client.base_url(),
                    error = %e,
                    "instance reconcile aborted - management API unreachable",
                );
                return Err(e);
            }
        };
        let by_name: HashMap<&str, &InstanceInfo> = existing
            .instances
            .iter()
            .map(|i| (instance_name_from_server_id(&i.id), i))
            .collect();

        let mut created = 0usize;
        let mut resized = 0usize;
        for profile in &self.profiles {
            if !profile.pinned {
                continue;
            }
            let name = profile.name.as_deref().unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let group = profile.group.as_deref().unwrap_or(name);
            match by_name.get(name) {
                Some(info) => {
                    // Resize on n_ctx drift; warn on parallel/pinned drift.
                    if profile.num_ctx > 0 && info.n_ctx != profile.num_ctx {
                        match self.client.resize(name, profile.num_ctx).await {
                            Ok(()) => {
                                resized += 1;
                                tracing::info!(
                                    target: "router.instances",
                                    instance = %name,
                                    from_ctx = info.n_ctx,
                                    to_ctx = profile.num_ctx,
                                    "instance resized",
                                );
                            }
                            Err(e) => tracing::warn!(
                                target: "router.instances",
                                instance = %name,
                                error = %e,
                                "instance resize failed",
                            ),
                        }
                    }
                    if info.pinned != profile.pinned {
                        tracing::warn!(
                            target: "router.instances",
                            instance = %name,
                            expected_pinned = profile.pinned,
                            actual_pinned = info.pinned,
                            "instance pinned drift",
                        );
                    }
                    if let Some(parallel) = profile.parallel {
                        if info.parallel != parallel {
                            tracing::warn!(
                                target: "router.instances",
                                instance = %name,
                                expected_parallel = parallel,
                                actual_parallel = info.parallel,
                                "instance parallel drift",
                            );
                        }
                    }
                }
                None => match self
                    .client
                    .create(
                        name,
                        group,
                        profile.num_ctx,
                        profile.parallel,
                        profile.pinned,
                        profile.default,
                    )
                    .await
                {
                    Ok(_) => {
                        created += 1;
                        tracing::info!(
                            target: "router.instances",
                            instance = %name,
                            group = %group,
                            n_ctx = profile.num_ctx,
                            pinned = profile.pinned,
                            "instance created at boot",
                        );
                    }
                    Err(InstanceError::Duplicate) => {
                        // Another reconciler won the race; tolerate.
                    }
                    Err(e) => tracing::warn!(
                        target: "router.instances",
                        instance = %name,
                        error = %e,
                        "instance create failed",
                    ),
                },
            }
        }
        crate::audit::emit(
            "instances",
            serde_json::json!({
                "action": "reconcile",
                "created": created,
                "resized": resized,
                "base_url": self.client.base_url(),
            }),
        );
        Ok(())
    }

    /// Create a configured instance on demand if it is not already present.
    /// Used by the dispatch path when a request targets a specific instance
    /// (e.g. `<base>:scratch`) that is unpinned and therefore absent after
    /// boot. No-op when the name has no configured profile (nothing to create)
    /// or already exists.
    pub async fn ensure_instance(&self, name: &str) -> Result<(), InstanceError> {
        if name.is_empty() || self.profiles.is_empty() {
            return Ok(());
        }
        let existing = match self.client.list().await {
            Ok(envelope) => envelope,
            Err(e) => return Err(e),
        };
        let present = existing
            .instances
            .iter()
            .any(|i| instance_name_from_server_id(&i.id) == name);
        if present {
            return Ok(());
        }
        let Some(profile) = self
            .profiles
            .iter()
            .find(|p| p.name.as_deref() == Some(name))
        else {
            tracing::debug!(
                target: "router.instances",
                instance = %name,
                "no configured profile for on-demand instance - nothing to create",
            );
            return Ok(());
        };
        let group = profile.group.as_deref().unwrap_or(name);
        match self
            .client
            .create(
                name,
                group,
                profile.num_ctx,
                profile.parallel,
                profile.pinned,
                profile.default,
            )
            .await
        {
            Ok(info) => {
                tracing::info!(
                    target: "router.instances",
                    instance = %name,
                    group = %group,
                    n_ctx = info.n_ctx,
                    "instance created on demand",
                );
                crate::audit::emit(
                    "instances",
                    serde_json::json!({
                        "action": "create_on_demand",
                        "instance": name,
                        "group": group,
                        "base_url": self.client.base_url(),
                    }),
                );
                Ok(())
            }
            Err(InstanceError::Duplicate) => Ok(()),
            Err(e) => {
                tracing::warn!(
                    target: "router.instances",
                    instance = %name,
                    error = %e,
                    "on-demand instance create failed",
                );
                Err(e)
            }
        }
    }

    /// Boot orchestration: reconcile the configured pinned instances against
    /// the fork, retrying until the management API is reachable (the container
    /// may come up after the router). Residency is pool-wide
    /// ([`InstancePool::run_residency`]), so this task stops after a
    /// successful reconcile.
    pub async fn bootstrap(&self) {
        let base = Duration::from_secs(self.policy.poll_interval_s.max(1));
        let mut failures = 0u32;
        loop {
            match self.reconcile().await {
                Ok(()) => break,
                Err(e) => {
                    failures += 1;
                    if failures == 1 {
                        tracing::info!(
                            target: "router.instances",
                            error = %e,
                            "instance reconcile deferred - management API not ready, retrying",
                        );
                    } else {
                        tracing::debug!(
                            target: "router.instances",
                            error = %e,
                            failures = failures,
                            "instance reconcile still deferred, retrying",
                        );
                    }
                    tokio::time::sleep(Self::residency_backoff(base, failures)).await;
                }
            }
        }
    }

    /// Compute the sleep delay for a retry loop after `consecutive_failures`
    /// consecutive failures (0 = healthy). Progresses from the base interval up
    /// to a 12x cap so a persistently unavailable management API backs off
    /// without spamming a warning every poll.
    fn residency_backoff(base: Duration, consecutive_failures: u32) -> Duration {
        base.saturating_mul(consecutive_failures.saturating_add(1).min(12))
    }

    /// Allocate a fresh instance for `group` on a 503 group-miss. Uses the
    /// group's configured profile (name/group/ctx/parallel/pinned) with a
    /// unique `<group>-<uuid>` name. No-op when no profile configures the
    /// group (there is nothing to allocate).
    pub async fn ensure_group(&self, group: &str) -> Result<(), InstanceError> {
        let Some(profile) = self
            .profiles
            .iter()
            .find(|p| p.group.as_deref() == Some(group))
        else {
            tracing::debug!(
                target: "router.instances",
                group = %group,
                "no configured profile for group - nothing to allocate",
            );
            return Ok(());
        };
        let name = format!("{group}-{}", &uuid_v4()[..8]);
        let profile_group = profile.group.as_deref().unwrap_or(group);
        let result = self
            .client
            .create(
                &name,
                profile_group,
                profile.num_ctx,
                profile.parallel,
                profile.pinned,
                profile.default,
            )
            .await
            .map(|_| ());
        if result.is_ok() {
            tracing::info!(
                target: "router.instances",
                instance = %name,
                group = %profile_group,
                "instance allocated on group miss",
            );
        }
        result
    }
}

/// Build the HTTP client for one spawned server's management API.
///
/// llama.cpp's cpp-httplib closes idle keep-alive connections after ~5s
/// (`CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND`), but reqwest's default pool retains
/// idle connections far longer. The residency loop polls every
/// `poll_interval_s`, so a poll that falls just past the server's idle cutoff
/// reuses a connection the server already closed, surfacing as an intermittent
/// `management network error: error sending request`. Disabling idle pooling
/// (no connection is kept for reuse) makes each management call open a fresh
/// connection, eliminating the stale-connection resets.
fn management_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("management http client build")
}

/// Build one `InstanceManager` per managed model (`weights`/`hf_repo`/
/// `instances` declared), keyed by the Coral Router model id. Each manager's
/// client points DIRECTLY at that model's spawned `llama-server` (the config
/// `endpoint` must already have been rewritten to the server's address by the
/// supervisor at boot).
///
/// A manager is created for EVERY managed model — even plain weights-only
/// models with no instance pool — so the pool can drive on-demand loading for
/// any lazy model (see [`InstancePool::ensure_target_ready`]): it resolves the
/// dispatch URL to the owning manager and loads the model's server when the
/// target is not resident.
///
/// Fails fast (`Err`) when a model's combined profiles fail
/// [`validate_instances`] (a malformed grammar must abort boot loudly rather
/// than POST a broken instance set). On success logs each pool's generated
/// grammar string for operability.
pub fn build_instance_managers(
    config: &crate::config::RouterConfig,
    supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>,
) -> Result<InstancePool, String> {
    // One manager per managed model. Instances belong to a single model pool,
    // and each model now owns its own server, so the manager talks directly to
    // that server (no `model` routing).
    let mut managers = HashMap::new();
    for (key, entry) in &config.models {
        if !entry.is_managed() {
            continue;
        }
        let profiles = entry.instance_profiles();
        validate_instances(&profiles)
            .map_err(|e| format!("model {key}: invalid instance grammar: {e}"))?;
        let model_name = entry.llama_model_name(key);
        if !profiles.is_empty() {
            tracing::info!(
                target: "router.instances",
                endpoint = %entry.endpoint,
                model = %model_name,
                grammar = instance_grammar_string(&profiles),
                "instance pool grammar",
            );
        }
        let base_url = management_base_url(&entry.endpoint);
        let api_key = config
            .sidecar
            .api_key_env
            .as_deref()
            .map(std::env::var)
            .and_then(Result::ok)
            .filter(|k| !k.is_empty());
        let client = InstanceClient::new(management_http_client(), base_url, api_key);
        // The resident weights size of a plain (no-instance) model: the file
        // the fork loads. Instance models report `model_bytes` themselves; a
        // plain model's footprint is synthesized from this.
        let weights_bytes = entry
            .weights
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map_or(0, |m| m.len());
        let manager = Arc::new(
            InstanceManager::new(key, client, profiles, config.sidecar.clone())
                .with_weights_bytes(weights_bytes),
        );
        managers.insert(key.clone(), manager);
    }
    Ok(InstancePool::from_managers(managers, supervisor))
}

/// The instance NAME within a server-reported id: the segment after the last
/// `:` (the server reports `<model_alias>:<name>`). A bare id (no `:`) is the
/// name itself. Instance names never contain `:`.
fn instance_name_from_server_id(id: &str) -> &str {
    match id.rsplit_once(':') {
        Some((_, name)) => name,
        None => id,
    }
}

/// The deterministic fork snapshot name a resume-marked context is saved under
/// before eviction: `<instance>-resume` (snapshot names share the instance
/// character class `[A-Za-z0-9._-]`), so a later request to the same instance
/// with `snapshot=<name>-resume` restores it.
pub fn resume_snapshot_name(instance: &str) -> String {
    format!("{instance}-resume")
}

/// One unit the residency/admission control can evict to free VRAM.
///
/// A unit is either a single unpinned context (frees its KV + compute; the
/// model's weights stay) or a whole model with no pinned instances (frees its
/// weights and every context). Including whole-model units is what makes the
/// largest resident footprints - e.g. a 10.5 GB weight pool - real eviction
/// targets instead of only the small per-context buffers.
enum Evictable<'a> {
    /// One unpinned context.
    Context {
        info: InstanceInfo,
        manager: &'a Arc<InstanceManager>,
    },
    /// A whole model: every unpinned context, then the shared weights.
    Model {
        manager: &'a Arc<InstanceManager>,
        /// The coldest context's last use (model recency).
        last_used: i64,
        /// Total VRAM freed: weights + all unpinned contexts.
        freed_bytes: u64,
        /// The unpinned contexts to drop first (resume ones are snapshotted).
        contexts: Vec<InstanceInfo>,
    },
}

impl Evictable<'_> {
    fn last_used(&self) -> i64 {
        match self {
            Self::Context { info, .. } => info.last_used,
            Self::Model { last_used, .. } => *last_used,
        }
    }

    fn freed_bytes(&self) -> u64 {
        match self {
            Self::Context { info, .. } => info.vram_bytes,
            Self::Model { freed_bytes, .. } => *freed_bytes,
        }
    }
}

/// Eviction priority score: `freed_bytes * coldness`, where coldness is seconds
/// since `last_used` (capped as an overflow guard; an entity never used is
/// maximally cold). This is a "cost of keeping" heuristic: the unit whose
/// resident footprint times its idle time is largest is the most valuable to
/// evict. It makes big cold footprints (a model's weights) outrank small hot
/// ones, so OOM pressure reclaims the largest chunks while a just-used model
/// scores near zero and stays.
fn eviction_score(freed_bytes: u64, last_used: i64, now: i64) -> u64 {
    const COLD_CAP: i64 = 1 << 40; // ~35k years; overflow guard only
    let coldness = if last_used < 0 {
        COLD_CAP
    } else {
        now.saturating_sub(last_used).clamp(1, COLD_CAP)
    };
    freed_bytes.saturating_mul(coldness as u64)
}

/// The router's aggregate `/instances` facade over every managed model's
/// server. Public instance ids are `<model_id>:<instance_name>`; `total` is
/// summed with 64-bit arithmetic with each model's shared weights counted once.
///
/// This is the public surface Coral Router exposes at its OWN address as the
/// single sidecar entry point (the managed servers bind to `127.0.0.1` and are
/// never exposed directly).
#[derive(Clone)]
pub struct InstancePool {
    /// model key -> manager.
    managers: HashMap<String, Arc<InstanceManager>>,
    /// management base URL -> model key (for dispatch-time manager lookup).
    by_base: HashMap<String, String>,
    /// The llama-server supervisor, used to load lazy models on demand and to
    /// unload a model whose last context was evicted (freeing its weights).
    supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>,
    /// Sidecar residency policy (device budget, poll interval, evict batch).
    policy: crate::config::SidecarConfig,
}

impl InstancePool {
    /// Build a pool from an existing manager set, indexing each manager by its
    /// client's management base URL for dispatch-time lookup.
    pub fn from_managers(
        managers: HashMap<String, Arc<InstanceManager>>,
        supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>,
    ) -> Self {
        let by_base = managers
            .iter()
            .map(|(key, m)| (management_base_url(m.client().base_url()), key.clone()))
            .collect();
        // The policy is shared across managers (it is cloned from the config
        // into each); take the first manager's as the pool-wide residency
        // policy. An empty pool has no policy needs.
        let policy = managers
            .values()
            .next()
            .map(|m| m.policy.clone())
            .unwrap_or_default();
        Self {
            managers,
            by_base,
            supervisor,
            policy,
        }
    }

    /// Whether any model is managed (the sidecar is active).
    pub fn is_empty(&self) -> bool {
        self.managers.is_empty()
    }

    /// The manager for a Coral Router model id.
    pub fn manager(&self, model_key: &str) -> Option<&Arc<InstanceManager>> {
        self.managers.get(model_key)
    }

    /// The manager for a dispatch endpoint URL: strips the path to the
    /// management base and matches a managed server. Used by the
    /// allocate-on-503 path.
    pub fn manager_for_url(&self, endpoint_url: &str) -> Option<&Arc<InstanceManager>> {
        let base = management_base_url(endpoint_url);
        let key = self.by_base.get(&base)?;
        self.managers.get(key)
    }

    /// The supervisor (present when any model is managed).
    pub fn supervisor(&self) -> Option<&Arc<crate::supervisor::LlamaServerSupervisor>> {
        self.supervisor.as_ref()
    }

    /// Iterate the managers (the server spawns each manager's sidecar task).
    pub fn managers_iter(&self) -> impl Iterator<Item = &Arc<InstanceManager>> {
        self.managers.values()
    }

    /// Ensure the model behind a dispatch endpoint is loaded: spawn its
    /// `llama-server` on demand if it is lazy (no pinned instance at boot) and
    /// currently unloaded. Also ensures a specifically-targeted instance
    /// (e.g. `<base>:scratch`) is created on demand. Best-effort: a failure to
    /// load degrades to the caller's normal dispatch error path.
    ///
    /// Before the target's weights are (re)loaded, [`Self::make_room_for`]
    /// evicts LRU unpinned instances and unloads cold plain models so the load
    /// never pushes the device over its VRAM allocation budget. Residency is
    /// judged by the *actual* resident state, not the process flag: a plain
    /// model whose fork has slept its weights out of VRAM (process alive,
    /// `is_sleeping = true`) needs the same room to wake as a cold load would.
    pub async fn ensure_target_ready(&self, endpoint_url: &str, instance: Option<&str>) {
        let Some(manager) = self.manager_for_url(endpoint_url) else {
            return;
        };
        // Record dispatch recency so residency can order plain models by last
        // use (the fork reports no `last_used` for them).
        manager.touch();
        let model_key = manager.model_key();
        if let Some(sup) = &self.supervisor {
            let running = sup.is_running(model_key) == Some(true);
            // A sleeping plain model is NOT resident: waking it reloads its
            // weights into VRAM. Treat it like a cold load for admission.
            let resident = running && manager.is_sleeping().await != Some(true);
            if !resident {
                self.make_room_for(model_key, manager.weights_bytes()).await;
            }
            if !running {
                if let Err(e) = sup.ensure_running(model_key).await {
                    tracing::warn!(
                        target: "router.instances",
                        model = %model_key,
                        error = %e,
                        "on-demand model load failed",
                    );
                }
            }
        }
        if let Some(instance) = instance {
            if let Err(e) = manager.ensure_instance(instance).await {
                tracing::warn!(
                    target: "router.instances",
                    model = %model_key,
                    instance = %instance,
                    error = %e,
                    "on-demand instance create failed",
                );
            }
        }
    }

    /// One device-wide residency pass. The pool owns VRAM residency for the
    /// whole device (all managed servers share it), so this aggregates every
    /// manager's `/instances` into a device `used` total and compares it to
    /// the allocation budget (`device_total - minimum_remaining_vram`).
    ///
    /// When the budget is exceeded, evicts up to `evict_batch` units - the
    /// largest resident footprint first (see [`Self::evict_to_fit`]) - and
    /// then unloads any model whose server is left with zero contexts. Resume
    /// marked contexts are KV-snapshotted before they drop, and resume work
    /// idle past `resume_ttl_s` is concluded (flag cleared, snapshot deleted)
    /// first so the router never keeps saving context it has decided is done.
    /// Pinned instances are never evicted.
    pub async fn residency_cycle(&self) -> Result<(), InstanceError> {
        self.expire_resume().await;
        let Some(budget) = self.policy.allocation_limit() else {
            tracing::info!(
                target: "router.instances",
                "residency: no allocation budget (set sidecar.minimum_remaining_vram or vram_total_bytes)",
            );
            return Ok(());
        };
        let (mut used, evictable) = self.gather_residency(None).await;
        if used <= budget {
            tracing::debug!(
                target: "router.instances",
                used_bytes = used,
                budget_bytes = budget,
                "device VRAM within budget - no eviction this pass",
            );
            return Ok(());
        }
        tracing::warn!(
            target: "router.instances",
            used_bytes = used,
            budget_bytes = budget,
            "device VRAM over budget - evicting largest coldest footprints",
        );
        self.evict_to_fit(&mut used, budget, evictable).await;
        // Unload any model whose server now has zero contexts: its weights are
        // freed, restoring VRAM that context-level eviction cannot.
        self.unload_empty_models().await;
        Ok(())
    }

    /// The device's resident VRAM usage and eviction candidates across every
    /// managed server. `exclude` names a model key whose usage is omitted and
    /// which is never an eviction candidate (the model about to be loaded).
    ///
    /// Every candidate is an [`Evictable`]: either one unpinned context (frees
    /// its KV + compute) or a whole model with no pinned instances (frees its
    /// weights *and* all its unpinned contexts - the largest footprint, and the
    /// only way a 10.5 GB weight pool can actually be reclaimed when OOM
    /// pressure demands it).
    async fn gather_residency(
        &self,
        exclude: Option<&str>,
    ) -> (u64, Vec<Evictable<'_>>) {
        let mut used: u64 = 0;
        let mut evictable: Vec<Evictable<'_>> = Vec::new();
        for manager in self.managers.values() {
            if exclude == Some(manager.model_key()) {
                continue;
            }
            let Some((envelope, plain)) = manager.list_with_fallback().await else {
                tracing::debug!(
                    target: "router.instances",
                    model = %manager.model_key(),
                    "residency poll skipped - server down",
                );
                continue;
            };
            used = used.saturating_add(envelope.total.total);
            if plain {
                // One synthesized entry per plain model; only a non-sleeping
                // model's weights are a freeable resident chunk.
                if let Some(info) = envelope.instances.first() {
                    if info.model_bytes > 0 {
                        evictable.push(Evictable::Model {
                            manager,
                            last_used: info.last_used,
                            freed_bytes: info.model_bytes,
                            contexts: vec![info.clone()],
                        });
                    }
                }
            } else {
                let unpinned: Vec<InstanceInfo> = envelope
                    .instances
                    .iter()
                    .filter(|i| !i.pinned)
                    .cloned()
                    .collect();
                for info in &unpinned {
                    evictable.push(Evictable::Context {
                        info: info.clone(),
                        manager,
                    });
                }
                // A model with NO pinned context is fully evictable: dropping
                // every context unloads its weights too. Pinned contexts keep
                // a model's weights resident, so only models with zero pinned
                // instances surface as whole-model candidates.
                let has_pinned = envelope.instances.iter().any(|i| i.pinned);
                if !has_pinned && envelope.total.model > 0 {
                    let weights = envelope.total.model;
                    let ctx_vram: u64 = unpinned.iter().map(|i| i.vram_bytes).sum();
                    let last_used = unpinned.iter().map(|i| i.last_used).min().unwrap_or(-1);
                    evictable.push(Evictable::Model {
                        manager,
                        last_used,
                        freed_bytes: weights.saturating_add(ctx_vram),
                        contexts: unpinned,
                    });
                }
            }
        }
        (used, evictable)
    }

    /// Load-time admission control: before a cold model spawns (requiring
    /// `required_bytes` of VRAM for its weights), evict units until the
    /// projected device usage fits the allocation budget. The target model is
    /// never an eviction candidate and pinned instances are never evicted.
    /// Best-effort: if eviction cannot fully make room, the load proceeds and
    /// the residency loop corrects the overshoot.
    pub async fn make_room_for(&self, model_key: &str, required_bytes: u64) {
        let Some(budget) = self.policy.allocation_limit() else {
            return;
        };
        if required_bytes == 0 {
            return;
        }
        let (used, evictable) = self.gather_residency(Some(model_key)).await;
        let mut projected = used.saturating_add(required_bytes);
        if projected <= budget {
            return;
        }
        tracing::info!(
            target: "router.instances",
            model = %model_key,
            required_bytes = required_bytes,
            used_bytes = used,
            budget_bytes = budget,
            "making VRAM room for cold model load",
        );
        self.evict_to_fit(&mut projected, budget, evictable).await;
    }

    /// Evict candidates (snapshotting resume-marked contexts first) until
    /// `used` fits the budget.
    ///
    /// Priority is *footprint-weighted coldness*: the candidate that frees the
    /// most VRAM from the coldest resident entity goes first. A whole model's
    /// weights (say a 10.5 GB pool) outrank any handful of context buffers, so
    /// OOM pressure reclaims the big chunks, while a just-used model scores
    /// near zero and stays - protecting active agentic work from being evicted
    /// underneath a running task.
    async fn evict_to_fit(&self, used: &mut u64, budget: u64, mut evictable: Vec<Evictable<'_>>) {
        let now = i64::try_from(common_core::now_secs()).unwrap_or(i64::MAX);
        evictable.sort_by(|a, b| {
            eviction_score(b.freed_bytes(), b.last_used(), now)
                .cmp(&eviction_score(a.freed_bytes(), a.last_used(), now))
                .then_with(|| b.last_used().cmp(&a.last_used()))
        });
        let mut evicted = 0usize;
        for unit in evictable {
            if *used <= budget || evicted >= self.policy.evict_batch {
                break;
            }
            let freed = match unit {
                Evictable::Context { info, manager } => {
                    self.evict_context(manager, &info, "over_budget").await
                }
                Evictable::Model {
                    manager,
                    contexts,
                    freed_bytes,
                    ..
                } => self.evict_model(manager, &contexts, freed_bytes).await,
            };
            if let Some(freed) = freed {
                evicted += 1;
                *used = used.saturating_sub(freed);
            }
        }
    }

    /// Snapshot (if resume-marked) then destroy one unpinned context. Returns
    /// the freed bytes, or `None` when the destroy failed.
    async fn evict_context(
        &self,
        manager: &Arc<InstanceManager>,
        info: &InstanceInfo,
        reason: &str,
    ) -> Option<u64> {
        let name = instance_name_from_server_id(&info.id);
        self.snapshot_for_resume(manager, name).await;
        match manager.client().destroy(name, false).await {
            Ok(()) => {
                tracing::info!(
                    target: "router.instances",
                    model = %manager.model_key(),
                    instance = %info.id,
                    vram_bytes = info.vram_bytes,
                    reason = reason,
                    "unpinned context evicted",
                );
                crate::audit::emit(
                    "instances",
                    serde_json::json!({
                        "action": "evict",
                        "instance": info.id,
                        "reason": reason,
                    }),
                );
                Some(info.vram_bytes)
            }
            Err(e) => {
                tracing::warn!(
                    target: "router.instances",
                    instance = %info.id,
                    error = %e,
                    "context eviction failed",
                );
                None
            }
        }
    }

    /// Evict a whole model: snapshot (if resume-marked) and destroy every
    /// unpinned context, then unload the weights. Returns the total freed
    /// bytes, `None` when the weights could not be unloaded.
    async fn evict_model(
        &self,
        manager: &Arc<InstanceManager>,
        contexts: &[InstanceInfo],
        freed_bytes: u64,
    ) -> Option<u64> {
        for info in contexts {
            let name = instance_name_from_server_id(&info.id);
            self.snapshot_for_resume(manager, name).await;
            if let Err(e) = manager.client().destroy(name, false).await {
                tracing::warn!(
                    target: "router.instances",
                    instance = %info.id,
                    error = %e,
                    "model-eviction context destroy failed",
                );
            }
        }
        let Some(sup) = &self.supervisor else {
            return None;
        };
        let model_key = manager.model_key();
        sup.unload(model_key).await;
        tracing::info!(
            target: "router.instances",
            model = %model_key,
            weights_bytes = freed_bytes,
            "model unloaded to free VRAM (weights + contexts)",
        );
        crate::audit::emit(
            "instances",
            serde_json::json!({
                "action": "unload_model",
                "model": model_key,
                "reason": "free_vram",
            }),
        );
        Some(freed_bytes)
    }

    /// Best-effort KV snapshot of a resume-marked context before it drops. The
    /// session transcript is already durable in the ledger; this preserves the
    /// KV so a later `snapshot=<name>-resume` request restores it. A failed
    /// save (no slot-save path, misconfigured snapshot dir) is logged and the
    /// eviction still proceeds - the context simply drops unsnapshotted.
    async fn snapshot_for_resume(&self, manager: &Arc<InstanceManager>, name: &str) {
        if !manager.resume_for(name) {
            return;
        }
        let snapshot = resume_snapshot_name(name);
        match manager.client().save_snapshot(name, &snapshot).await {
            Ok(()) => {
                tracing::info!(
                    target: "router.instances",
                    instance = %name,
                    snapshot = %snapshot,
                    "resume context snapshotted before eviction",
                );
                crate::audit::emit(
                    "instances",
                    serde_json::json!({
                        "action": "resume_snapshot",
                        "instance": name,
                        "snapshot": snapshot,
                    }),
                );
            }
            Err(e) => tracing::warn!(
                target: "router.instances",
                instance = %name,
                error = %e,
                "resume snapshot save failed - context drops unsnapshotted",
            ),
        }
    }

    /// "Coral Router concludes its work is done": any resume-marked context
    /// idle past `resume_ttl_s` has its flag cleared and its `-resume` snapshot
    /// deleted. Runs each residency pass so eviction stops preserving context
    /// the router has decided is stale.
    async fn expire_resume(&self) {
        let Some(ttl) = self.policy.resume_ttl_s else {
            return;
        };
        let now = i64::try_from(common_core::now_secs()).unwrap_or(i64::MAX);
        for manager in self.managers.values() {
            let Some((envelope, _)) = manager.list_with_fallback().await else {
                continue;
            };
            for info in envelope.instances {
                let name = instance_name_from_server_id(&info.id);
                if !manager.resume_for(name) {
                    continue;
                }
                let idle = now.saturating_sub(info.last_used);
                if idle >= ttl as i64 {
                    manager.set_resume(name, false);
                    let snapshot = resume_snapshot_name(name);
                    match manager.client().delete_snapshot(name, &snapshot).await {
                        Ok(()) => {
                            tracing::info!(
                                target: "router.instances",
                                instance = %name,
                                idle_secs = idle,
                                ttl_secs = ttl,
                                "resume expired - work concluded, snapshot dropped",
                            );
                            crate::audit::emit(
                                "instances",
                                serde_json::json!({
                                    "action": "expire_resume",
                                    "instance": name,
                                    "reason": "idle_ttl",
                                }),
                            );
                        }
                        Err(e) => tracing::warn!(
                            target: "router.instances",
                            instance = %name,
                            error = %e,
                            "resume snapshot delete on expiry failed",
                        ),
                    }
                }
            }
        }
    }

    /// Unload managed models whose servers report zero contexts (all their
    /// instances were evicted). Frees the weights. Never touches models still
    /// holding contexts (pinned instances keep their models resident). Plain
    /// models (no instance pool) report no `/instances` and are skipped — their
    /// on-demand lifecycle is driven by `ensure_target_ready`/residency eviction
    /// at the model level instead.
    pub async fn unload_empty_models(&self) {
        let Some(sup) = &self.supervisor else {
            return;
        };
        let mut keys: Vec<String> = self.managers.keys().cloned().collect();
        keys.sort();
        for key in keys {
            let Some(manager) = self.managers.get(&key) else {
                continue;
            };
            if manager.profiles.is_empty() {
                continue;
            }
            let empty = match manager.client().list().await {
                Ok(envelope) => envelope.instances.is_empty(),
                Err(_) => continue,
            };
            if empty {
                tracing::info!(
                    target: "router.instances",
                    model = %key,
                    "model has no contexts left - unloading weights",
                );
                crate::audit::emit(
                    "instances",
                    serde_json::json!({
                        "action": "unload_model",
                        "model": key,
                        "reason": "no_contexts",
                    }),
                );
                sup.unload(&key).await;
            }
        }
    }

    /// The residency loop: poll device VRAM every `poll_interval_s`, evicting
    /// LRU-largest unpinned instances when over budget, forever. Runs as a
    /// spawned task owned by the server. Without an allocation budget
    /// eviction is impossible, so the loop notes the disabled eviction once
    /// and exits.
    pub async fn run_residency(&self) {
        if self.policy.allocation_limit().is_none() {
            tracing::info!(
                target: "router.instances",
                "residency eviction disabled - no allocation budget (set sidecar.minimum_remaining_vram or vram_total_bytes)",
            );
            return;
        }
        let base = Duration::from_secs(self.policy.poll_interval_s.max(1));
        let mut consecutive_failures = 0u32;
        loop {
            match self.residency_cycle().await {
                Ok(()) => consecutive_failures = 0,
                Err(e) => {
                    consecutive_failures += 1;
                    if consecutive_failures == 1 {
                        tracing::warn!(
                            target: "router.instances",
                            error = %e,
                            "residency poll failed - backing off (retrying with backoff)",
                        );
                    } else {
                        tracing::debug!(
                            target: "router.instances",
                            error = %e,
                            consecutive_failures = consecutive_failures,
                            "residency poll still failing - backing off",
                        );
                    }
                }
            }
            tokio::time::sleep(Self::residency_backoff(base, consecutive_failures)).await;
        }
    }

    /// Compute the sleep delay for the residency loop after
    /// `consecutive_failures` consecutive failed polls (0 = healthy).
    fn residency_backoff(base: Duration, consecutive_failures: u32) -> Duration {
        base.saturating_mul(consecutive_failures.saturating_add(1).min(12))
    }

    /// Resolve the public instance id grammar `<model_id>:<name>` (or a bare
    /// `<model_id>` + name) to `(model key, instance name)`. `None` when the
    /// model is unmanaged, the id has more than one `:`, or the instance name
    /// breaks the `[A-Za-z0-9._-]` grammar.
    pub fn resolve_instance_id(&self, id: &str) -> Option<(String, String)> {
        let (model, name) = id.split_once(':')?;
        if name.contains(':') || !is_valid_instance_name(name) {
            return None;
        }
        if !self.managers.contains_key(model) {
            return None;
        }
        Some((model.to_string(), name.to_string()))
    }

    /// `GET /instances` - the aggregate envelope across every managed model.
    /// `model: Some(...)` scopes the response to one model. Instance ids are
    /// `<model_id>:<name>`; snapshot entries are tagged with their owning
    /// `model`; `total` sums each server's envelope with 64-bit arithmetic.
    /// Plain (no-instance-grammar) models contribute a synthesized footprint
    /// (their shared weights; 0 when the fork reports the model sleeping).
    pub async fn aggregate(&self, model: Option<&str>) -> Result<Value, InstanceError> {
        let mut instances = Vec::new();
        let mut snapshots: Vec<Value> = Vec::new();
        let mut total = InstanceTotals::default();
        for (model_key, manager) in &self.managers {
            if let Some(filter) = model {
                if filter != model_key.as_str() {
                    continue;
                }
            }
            let Some((envelope, _plain)) = manager.list_with_fallback().await else {
                tracing::debug!(
                    target: "router.instances",
                    model = %model_key,
                    "aggregate /instances poll skipped - server down",
                );
                continue;
            };
            for info in envelope.instances {
                let instance_name = instance_name_from_server_id(&info.id);
                let instance_id = format!("{model_key}:{instance_name}");
                let aliases =
                    instance_aliases(model_key, &instance_id, &info.group, info.is_default);
                let mut entry = serde_json::to_value(&info).unwrap_or_default();
                if let Value::Object(ref mut obj) = entry {
                    obj.insert("id".into(), Value::String(instance_id));
                    obj.insert(
                        "aliases".into(),
                        Value::Array(aliases.into_iter().map(Value::String).collect()),
                    );
                    // The fork knows nothing of `resume`; Coral Router tracks it
                    // and overlays the router-side flag on the envelope.
                    obj.insert(
                        "resume".into(),
                        Value::Bool(manager.resume_for(instance_name)),
                    );
                }
                instances.push(entry);
            }
            for snap in envelope.snapshots {
                let mut entry = serde_json::to_value(&snap).unwrap_or_default();
                if let Value::Object(ref mut obj) = entry {
                    obj.insert("model".into(), Value::String(model_key.clone()));
                }
                snapshots.push(entry);
            }
            total.model = total.model.saturating_add(envelope.total.model);
            total.context = total.context.saturating_add(envelope.total.context);
            total.compute = total.compute.saturating_add(envelope.total.compute);
            total.total = total.total.saturating_add(envelope.total.total);
        }
        Ok(serde_json::json!({
            "instances": instances,
            "snapshots": snapshots,
            "total": total,
        }))
    }

    /// `GET /v1/models` - one entry per instance across every managed model,
    /// plus aliases for the bare model, group, and `latest` forms. Plain
    /// (no-instance-grammar) models contribute one synthesized entry.
    pub async fn list_models(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for (model_key, manager) in &self.managers {
            let Some((envelope, _plain)) = manager.list_with_fallback().await else {
                continue;
            };
            let created = common_core::now_secs();
            for info in envelope.instances {
                let instance_name = instance_name_from_server_id(&info.id);
                let instance_id = format!("{model_key}:{instance_name}");
                let mut entry = serde_json::json!({
                    "id": instance_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "coral-router",
                    "n_ctx": info.n_ctx,
                    "parallel": info.parallel,
                    "pinned": info.pinned,
                    "resume": manager.resume_for(instance_name),
                    "is_default": info.is_default,
                    "state": info.state,
                    "last_used": info.last_used,
                });
                entry["aliases"] = Value::Array(
                    instance_aliases(model_key, &instance_id, &info.group, info.is_default)
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                );
                out.push(entry);
            }
        }
        out
    }

    /// `POST /instances` - allocate a NEW context on `model_key`'s server.
    /// `resume` is router-side (the fork knows nothing of it): recorded here so
    /// the aggregate reports it and eviction snapshots the context first.
    pub async fn create(
        &self,
        model_key: &str,
        name: &str,
        group: &str,
        ctx_size: u64,
        parallel: Option<u32>,
        pinned: bool,
        is_default: bool,
        resume: bool,
    ) -> Result<InstanceInfo, InstanceError> {
        let manager = self
            .managers
            .get(model_key)
            .ok_or_else(|| InstanceError::Rejected {
                status: 404,
                body: format!("unknown model: {model_key}"),
            })?;
        let info = manager
            .client()
            .create(name, group, ctx_size, parallel, pinned, is_default)
            .await?;
        manager.set_resume(name, resume);
        let mut info = info;
        info.resume = resume;
        Ok(info)
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn destroy(&self, model_key: &str, name: &str) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?
            .destroy(name, false)
            .await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn pin(&self, model_key: &str, name: &str) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?.pin(name).await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn unpin(&self, model_key: &str, name: &str) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?.unpin(name).await
    }

    /// Set the preserve-on-evict flag for a context (router-side). Disabling
    /// also deletes any `-resume` snapshot the context left behind - the router
    /// concluding the work is done.
    pub async fn set_resume(
        &self,
        model_key: &str,
        name: &str,
        enabled: bool,
    ) -> Result<(), InstanceError> {
        let manager = self
            .managers
            .get(model_key)
            .ok_or_else(|| InstanceError::Rejected {
                status: 404,
                body: format!("unknown model: {model_key}"),
            })?;
        if !enabled {
            let _ = manager
                .client()
                .delete_snapshot(name, &resume_snapshot_name(name))
                .await;
        }
        manager.set_resume(name, enabled);
        crate::audit::emit(
            "instances",
            serde_json::json!({
                "action": "set_resume",
                "instance": format!("{model_key}:{name}"),
                "enabled": enabled,
            }),
        );
        Ok(())
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn resize(
        &self,
        model_key: &str,
        name: &str,
        ctx_size: u64,
    ) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?.resize(name, ctx_size).await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn save_snapshot(
        &self,
        model_key: &str,
        instance: &str,
        name: &str,
    ) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?
            .save_snapshot(instance, name)
            .await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn list_snapshots(
        &self,
        model_key: &str,
        instance: &str,
    ) -> Result<Vec<SnapshotInfo>, InstanceError> {
        self.manager_checked(model_key)?
            .list_snapshots(instance)
            .await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn delete_snapshot(
        &self,
        model_key: &str,
        instance: &str,
        name: &str,
    ) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?
            .delete_snapshot(instance, name)
            .await
    }

    fn manager_checked(&self, model_key: &str) -> Result<&InstanceClient, InstanceError> {
        self.managers
            .get(model_key)
            .map(|m| m.client())
            .ok_or_else(|| InstanceError::Rejected {
                status: 404,
                body: format!("unknown model: {model_key}"),
            })
    }
}

/// Whether a string is a valid instance/snapshot name (`[A-Za-z0-9._-]+`).
pub fn is_valid_instance_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// The alias set for one public instance id: the bare model id and `latest`
/// form on the default instance, the group form, and the exact id.
fn instance_aliases(model_key: &str, instance_id: &str, group: &str, is_default: bool) -> Vec<String> {
    let mut aliases = Vec::new();
    if is_default {
        aliases.push(model_key.to_string());
        aliases.push(format!("{model_key}:latest"));
    }
    aliases.push(format!("{model_key}:{group}"));
    aliases.push(instance_id.to_string());
    aliases.sort();
    aliases.dedup();
    aliases
}

/// A tiny HTTP/1.1 stub that records every request and answers from a shared
/// handler closure `(method, path, body) -> (status, body)`. Used to exercise
/// `InstanceClient`/`InstanceManager` (and the dispatch allocate-on-503 path)
/// against fixture `/instances`, `/memory`, and `/chat/completions` JSON
/// without a real llama-server.
#[cfg(test)]
pub(crate) mod stub {
    use std::sync::Arc;

    pub struct StubServer {
        #[allow(dead_code)]
        _handle: tokio::task::JoinHandle<()>,
        pub addr: std::net::SocketAddr,
        requests: Arc<std::sync::Mutex<Vec<(String, String, String)>>>,
    }

    impl StubServer {
        pub fn start(
            handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync>,
        ) -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
            let reqs = requests.clone();
            let handle = tokio::spawn(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                loop {
                    let (mut stream, _) = match listener.accept().await {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let handler = handler.clone();
                    let reqs = reqs.clone();
                    tokio::spawn(async move {
                        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
                        let mut reader = BufReader::new(&mut stream);
                        let mut request_line = String::new();
                        if reader.read_line(&mut request_line).await.is_err() {
                            return;
                        }
                        let mut content_length: usize = 0;
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).await.is_err() {
                                return;
                            }
                            if line == "\r\n" {
                                break;
                            }
                            if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                                content_length = v.trim().parse().unwrap_or(0);
                            }
                        }
                        let mut body = vec![0u8; content_length];
                        if content_length > 0 && reader.read_exact(&mut body).await.is_err() {
                            return;
                        }
                        let body_str = String::from_utf8_lossy(&body).into_owned();
                        let parts: Vec<&str> = request_line.split_whitespace().collect();
                        let (method, path) = if parts.len() >= 2 {
                            (parts[0].to_string(), parts[1].to_string())
                        } else {
                            ("GET".into(), "/".into())
                        };
                        if let Ok(mut r) = reqs.lock() {
                            r.push((method.clone(), path.clone(), body_str.clone()));
                        }
                        let (status, resp_body) = handler(&method, &path, &body_str);
                        let reason = match status {
                            200 => "OK",
                            201 => "Created",
                            204 => "No Content",
                            400 => "Bad Request",
                            404 => "Not Found",
                            409 => "Conflict",
                            503 => "Service Unavailable",
                            507 => "Insufficient Storage",
                            _ => "Error",
                        };
                        let resp = format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp_body.len(),
                            resp_body
                        );
                        let _ = stream.write_all(resp.as_bytes()).await;
                        let _ = stream.flush().await;
                    });
                }
            });
            Self {
                _handle: handle,
                addr,
                requests,
            }
        }

        pub fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        pub fn recorded(&self) -> Vec<(String, String, String)> {
            self.requests
                .lock()
                .map(|r| r.clone())
                .unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::stub::StubServer;
    use super::*;

    fn profile(name: &str, group: &str) -> InstanceProfile {
        InstanceProfile {
            name: Some(name.into()),
            group: Some(group.into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }
    }

    fn assert_round_trip(profiles: &[InstanceProfile]) {
        let s = instance_grammar_string(profiles);
        let reparsed: Vec<InstanceProfile> = s
            .split(',')
            .filter(|p| !p.is_empty())
            .map(|p| parse_one(p).into_instance_profile())
            .collect();
        assert_eq!(reparsed.len(), profiles.len());
        for (a, b) in profiles.iter().zip(reparsed.iter()) {
            assert_eq!(a.name.as_deref(), b.name.as_deref());
            assert_eq!(a.group.as_deref(), b.group.as_deref());
            assert_eq!(a.num_ctx, b.num_ctx);
            assert_eq!(a.parallel, b.parallel);
            assert_eq!(a.pinned, b.pinned);
            assert_eq!(a.no_sleep, b.no_sleep);
            assert_eq!(a.default, b.default);
        }
    }

    // A minimal fork-parser equivalent used only to validate round-trip shape.
    #[derive(Default)]
    struct Raw {
        name: String,
        group: Option<String>,
        ctx: u64,
        parallel: u32,
        pinned: bool,
        no_sleep: bool,
        sleep: Option<i32>,
        default: bool,
    }

    impl Raw {
        fn into_instance_profile(self) -> InstanceProfile {
            let group = self
                .group
                .clone()
                .unwrap_or_else(|| self.name.clone());
            InstanceProfile {
                name: Some(self.name),
                group: Some(group),
                count: 1,
                num_ctx: self.ctx,
                parallel: if self.parallel > 0 {
                    Some(self.parallel)
                } else {
                    None
                },
                pinned: self.pinned,
                no_sleep: self.no_sleep,
                sleep_idle_seconds: self.sleep,
                default: self.default,
                resume: false,
                params: None,
            }
        }
    }

    fn parse_one(spec: &str) -> Raw {
        let mut comps = spec.split(':');
        let mut raw = Raw {
            name: comps.next().unwrap_or_default().to_string(),
            ..Default::default()
        };
        for comp in comps {
            if comp == "pinned" {
                raw.pinned = true;
            } else if comp == "default" {
                raw.default = true;
            } else if let Some(v) = comp.strip_prefix("group=") {
                raw.group = Some(v.to_string());
            } else if let Some(v) = comp.strip_prefix("ctx=") {
                raw.ctx = v.parse().unwrap();
            } else if let Some(v) = comp.strip_prefix("parallel=") {
                raw.parallel = v.parse().unwrap();
            } else if let Some(v) = comp.strip_prefix("sleep=") {
                raw.sleep = Some(v.parse().unwrap());
            }
        }
        raw
    }

    #[test]
    fn grammar_matches_reference_deployment() {
        let swarm = InstanceProfile {
            name: Some("swarm".into()),
            group: Some("swarm".into()),
            count: 3,
            num_ctx: 16384,
            parallel: None,
            pinned: true,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        };
        // Expand the count as instance_profiles() would.
        let profiles: Vec<InstanceProfile> = (0..3)
            .map(|i| InstanceProfile {
                name: Some(format!("swarm-{i}")),
                group: Some("swarm".into()),
                ..swarm.clone()
            })
            .collect();
        assert_eq!(
            instance_grammar_string(&profiles),
            "swarm-0:group=swarm:ctx=16384:pinned,swarm-1:group=swarm:ctx=16384:pinned,swarm-2:group=swarm:ctx=16384:pinned"
        );
    }

    #[test]
    fn grammar_matches_reference_ledger_scratch() {
        let ledger = InstanceProfile {
            name: Some("ledger".into()),
            group: None,
            num_ctx: 131072,
            pinned: true,
            default: true,
            resume: false,
            ..profile("x", "x")
        };
        let scratch = InstanceProfile {
            name: Some("scratch".into()),
            group: None,
            num_ctx: 131072,
            sleep_idle_seconds: Some(30),
            ..profile("x", "x")
        };
        assert_eq!(
            instance_grammar_string(&[ledger, scratch]),
            // `sleep` is a sidecar eviction hint only - never emitted (the
            // branch has no auto-sleep).
            "ledger:ctx=131072:pinned:default,scratch:ctx=131072"
        );
    }

    #[test]
    fn round_trip_preserves_shape() {
        let profiles = vec![
            profile("swarm0", "swarm"),
            profile("ledger", "ledger"),
            profile("scratch", "scratch"),
        ];
        assert_round_trip(&profiles);
    }

    #[test]
    fn pinned_emits_only_pinned_flag() {
        // A pinned profile with a positive declared sleep emits only `:pinned`
        // (sleep is never forwarded - it is a sidecar eviction hint).
        let pinned = InstanceProfile {
            name: Some("p".into()),
            group: Some("p".into()),
            num_ctx: 0,
            pinned: true,
            sleep_idle_seconds: Some(30),
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[pinned]), "p:pinned");
    }

    #[test]
    fn sleep_is_never_emitted() {
        // `sleep`/`no_sleep` are config eviction hints only; the grammar has no
        // sleep component (the minimal branch has no auto-sleep).
        let with_sleep = InstanceProfile {
            name: Some("a".into()),
            group: Some("a".into()),
            num_ctx: 0,
            sleep_idle_seconds: Some(5),
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[with_sleep]), "a");
        let warm = InstanceProfile {
            name: Some("b".into()),
            group: Some("b".into()),
            num_ctx: 0,
            no_sleep: true,
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[warm]), "b");
        let absent = InstanceProfile {
            name: Some("c".into()),
            group: Some("c".into()),
            num_ctx: 0,
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[absent]), "c");
    }

    #[test]
    fn validates_duplicate_names() {
        let a = profile("dup", "g1");
        let b = profile("dup", "g2");
        assert!(validate_instances(&[a, b]).is_err());
    }

    #[test]
    fn validates_group_name_collision() {
        let a = profile("foo", "bar");
        let b = profile("bar", "baz");
        assert!(validate_instances(&[a, b]).is_err());
    }

    #[test]
    fn allows_group_equals_own_name() {
        let a = profile("foo", "foo");
        let b = profile("bar", "bar");
        assert!(validate_instances(&[a, b]).is_ok());
    }

    // -- M4 sidecar ---------------------------------------------------------

    fn sidecar_policy() -> crate::config::SidecarConfig {
        crate::config::SidecarConfig {
            poll_interval_s: 5,
            vram_low_watermark_bytes: 1024,
            evict_batch: 2,
            vram_total_bytes: Some(10000),
            minimum_remaining_vram: Some(2000),
            slot_save_path: Some("/srv/slots".into()),
            resume_ttl_s: None,
            api_key_env: None,
        }
    }

    fn instance_info(id: &str, group: &str, pinned: bool, last_used: i64) -> InstanceInfo {
        InstanceInfo {
            id: id.into(),
            aliases: vec![],
            group: group.into(),
            n_ctx: 16384,
            parallel: 1,
            pinned,
            is_default: false,
            resume: false,
            state: "loaded".into(),
            model_bytes: 0,
            context_bytes: 262144,
            compute_bytes: 1048576,
            total_bytes: 1310720,
            vram_bytes: 1000,
            last_used,
        }
    }

    fn management_base(endpoint: &str) -> String {
        super::management_base_url(endpoint)
    }

    #[test]
    fn management_base_url_strips_chat_completions_suffix() {
        assert_eq!(
            management_base("http://localhost:8080/v1/chat/completions"),
            "http://localhost:8080"
        );
        assert_eq!(
            management_base("http://localhost:8080/chat/completions"),
            "http://localhost:8080"
        );
        assert_eq!(management_base("http://localhost:8080/v1"), "http://localhost:8080/v1");
    }

    #[tokio::test]
    async fn client_list_parses_envelope_and_bare_array() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [
                                instance_info("swarm0", "swarm", false, 5),
                                instance_info("ledger", "ledger", true, 1),
                            ],
                            "snapshots": [],
                            "total": { "model": 115343360, "context": 524288, "compute": 2097152, "total": 118226080 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, r#"{"error":{"message":"not found"}}"#.into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        let list = client.list().await.expect("list");
        assert_eq!(list.instances.len(), 2);
        assert_eq!(list.instances[0].id, "swarm0");
        assert_eq!(list.instances[0].group, "swarm");
        assert_eq!(list.instances[1].pinned, true);
        assert_eq!(list.total.total, 118226080);
        assert_eq!(list.total.model, 115343360);
    }

    #[tokio::test]
    async fn client_mutating_calls_hit_expected_paths() {
        let requests_c = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                requests_c.lock().unwrap().push((method.into(), path.into()));
                if path.ends_with("/snapshots") {
                    (200, "[]".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        client
            .create("work", "swarm", 32768, Some(2), true, true)
            .await
            .expect("create");
        client.destroy("work", false).await.expect("destroy");
        client.destroy("work", true).await.expect("destroy force");
        client.pin("work").await.expect("pin");
        client.unpin("work").await.expect("unpin");
        client.resize("work", 49152).await.expect("resize");
        client.save_snapshot("work", "readfiles").await.expect("save");
        client.delete_snapshot("work", "readfiles").await.expect("delete");
        client.list_snapshots("work").await.expect("list snapshots");

        let recorded = stub.recorded();
        assert_eq!(recorded.len(), 9);
        let paths: Vec<&str> = recorded.iter().map(|(_, p, _)| p.as_str()).collect();
        assert!(paths.contains(&"/instances"));
        assert!(paths.contains(&"/instances/work"));
        assert!(paths.contains(&"/instances/work?force=true"));
        assert!(paths.contains(&"/instances/work/pin"));
        assert!(paths.contains(&"/instances/work/unpin"));
        assert!(paths.contains(&"/instances/work/resize"));
        assert!(paths.contains(&"/instances/work/snapshot"));
        assert!(paths.contains(&"/instances/work/snapshot/readfiles"));
        assert!(paths.contains(&"/instances/work/snapshots"));
        // create body carries the declared fields.
        let create_req = recorded
            .iter()
            .find(|(m, p, _)| m == "POST" && p == "/instances")
            .unwrap();
        let body: serde_json::Value = serde_json::from_str(&create_req.2).unwrap();
        assert_eq!(body["name"], "work");
        assert_eq!(body["group"], "swarm");
        assert_eq!(body["ctx_size"], 32768);
        assert_eq!(body["parallel"], 2);
        assert_eq!(body["pinned"], true);
    }

    #[tokio::test]
    async fn client_classifies_409_duplicate_and_5xx_transient() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if path == "/instances" && method == "POST" {
                    (409, r#"{"error":{"message":"duplicate"}}"#.into())
                } else if path == "/instances/boom" {
                    (507, r#"{"error":{"message":"oom"}}"#.into())
                } else {
                    (404, r#"{"error":{"message":"not found"}}"#.into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        let dup = client.create("dup", "g", 16384, None, false, false).await;
        assert!(dup.unwrap_err().is_duplicate());

        let transient = client.destroy("boom", false).await.unwrap_err();
        assert!(transient.is_retryable());
        assert!(transient.is_evict_trigger());

        let rejected = client.destroy("missing", false).await.unwrap_err();
        assert!(matches!(rejected, InstanceError::Rejected { .. }));
    }

    #[tokio::test]
    async fn client_talks_directly_to_its_server() {
        // The client targets one spawned server directly: no `model` in the
        // create body and no `?model=` on per-instance ops (the llama.cpp
        // router mode is never used - Coral Router owns the processes).
        let seen: Arc<std::sync::Mutex<Vec<(String, String, String)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, body| {
                seen2.lock().unwrap().push((method.into(), path.into(), body.into()));
                (200, "{}".into())
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        client.create("swarm-0", "swarm", 16384, Some(1), false, false).await.unwrap();
        client.destroy("swarm-0", false).await.unwrap();
        client.resize("swarm-0", 32768).await.unwrap();

        let rec = seen.lock().unwrap();
        let create = rec
            .iter()
            .find(|(m, p, _)| m == "POST" && p == "/instances")
            .expect("create recorded");
        let body: Value = serde_json::from_str(&create.2).unwrap();
        assert!(
            body.get("model").is_none(),
            "no router-routing model in the create body"
        );
        assert!(
            rec.iter().all(|(_, p, _)| !p.contains("model=")),
            "no ?model= routing query on per-instance ops"
        );
    }

    #[tokio::test]
    async fn reconcile_creates_missing_pinned_and_resizes_n_ctx_drift() {
        // Server already has ledger (correct) and swarm0 (wrong n_ctx); swarm1
        // is missing entirely.
        let existing = [
            instance_info("ledger", "ledger", true, 1),
            instance_info("swarm0", "swarm", true, 5),
        ];
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, serde_json::to_string(&existing).unwrap())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        // Profiles: ledger (pinned, n_ctx 131072), swarm0 + swarm1 (group swarm,
        // n_ctx 16384). swarm1 is unpinned -> deferred to on-demand creation.
        let profiles = vec![
            InstanceProfile {
                name: Some("ledger".into()),
                group: Some("ledger".into()),
                count: 1,
                num_ctx: 131072,
                parallel: None,
                pinned: true,
                no_sleep: false,
                sleep_idle_seconds: None,
                default: true,
            resume: false,
                params: None,
            },
            InstanceProfile {
                name: Some("swarm0".into()),
                group: Some("swarm".into()),
                count: 1,
                num_ctx: 16384,
                parallel: None,
                pinned: true,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: false,
            resume: false,
                params: None,
            },
            InstanceProfile {
                name: Some("swarm1".into()),
                group: Some("swarm".into()),
                count: 1,
                num_ctx: 16384,
                parallel: None,
                pinned: false,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: false,
            resume: false,
                params: None,
            },
        ];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager.reconcile().await.expect("reconcile");

        let recorded = stub.recorded();
        // One resize (swarm0 ctx drift). No POST: swarm1 is unpinned and
        // deferred, and every pinned profile already exists.
        let creates = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .count();
        let resizes = recorded
            .iter()
            .filter(|(_, p, _)| p.ends_with("/resize"))
            .count();
        assert_eq!(creates, 0, "unpinned swarm1 deferred to on-demand creation");
        assert_eq!(resizes, 1, "n_ctx drift triggers exactly one resize");
        assert_eq!(
            recorded
                .iter()
                .filter(|(_, p, _)| p.ends_with("/resize"))
                .map(|(_, p, _)| p.as_str())
                .next(),
            Some("/instances/ledger/resize"),
            "ledger's n_ctx drift (131072 profile vs 16384 present) is resized"
        );
    }

    #[tokio::test]
    async fn ensure_instance_creates_missing_unpinned_on_demand() {
        // Server is empty; `scratch` is configured but unpinned -> absent at
        // boot, created on demand by ensure_instance.
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, r#"{"instances":[],"snapshots":[],"total":{"total":0}}"#.into())
                } else {
                    (201, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("scratch".into()),
            group: Some("scratch".into()),
            count: 1,
            num_ctx: 131072,
            parallel: None,
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: Some(1),
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager.ensure_instance("scratch").await.expect("ensure_instance");

        let recorded = stub.recorded();
        let creates: Vec<&(String, String, String)> = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .collect();
        assert_eq!(creates.len(), 1, "scratch created on demand");
        let body: serde_json::Value = serde_json::from_str(&creates[0].2).unwrap();
        assert_eq!(body["name"], "scratch");
        assert_eq!(body["group"], "scratch");
    }

    #[tokio::test]
    async fn ensure_instance_skips_when_already_present_or_unknown() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, serde_json::to_string(&[instance_info("scratch", "scratch", false, 7)]).unwrap())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("scratch".into()),
            group: Some("scratch".into()),
            count: 1,
            num_ctx: 131072,
            parallel: None,
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        // Already present -> no create.
        manager.ensure_instance("scratch").await.expect("present");
        // Unknown name -> nothing to create, no error.
        manager.ensure_instance("nope").await.expect("unknown is a no-op");
        let recorded = stub.recorded();
        assert!(
            recorded
                .iter()
                .all(|(m, _, _)| m != "POST"),
            "no create when already present or unknown: {recorded:?}"
        );
    }

    #[tokio::test]
    async fn list_accepts_wrapped_instances_object() {
        // The fork returns GET /instances as {"instances":[...]}; list() must
        // unwrap it (it also tolerates a bare array).
        let existing = [
            instance_info("ledger", "ledger", true, 1),
            instance_info("swarm0", "swarm", false, 5),
        ];
        let payload = serde_json::json!({ "instances": existing });
        let payload_str = payload.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> =
            Arc::new(move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, payload_str.clone())
                } else {
                    (404, "{}".into())
                }
            });
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let list = client.list().await.expect("list parses wrapped shape");
        assert_eq!(list.instances.len(), 2);
        assert_eq!(list.instances[0].id, "ledger");
    }

    #[tokio::test]
    async fn reconcile_tolerates_duplicate_create() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, "[]".into())
                } else if method == "POST" && path == "/instances" {
                    // A concurrent reconciler created it first.
                    (409, "{}".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("swarm0".into()),
            group: Some("swarm".into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        // A 409 during reconcile is tolerated - reconcile completes Ok.
        manager.reconcile().await.expect("reconcile tolerates 409");
    }

    #[tokio::test]
    async fn residency_evicts_lru_unpinned_and_never_pinned() {
        // Device budget = 10000 - 2000 = 8000; used 10000 -> over budget.
        let envelope = serde_json::json!({
            "instances": [
                instance_info("pinned", "g", true, 0),    // exempt
                instance_info("lru1", "g", false, 100),   // oldest unpinned
                instance_info("lru2", "g", false, 200),
                instance_info("recent", "g", false, 9000),
            ],
            "snapshots": [],
            "total": { "model": 5000, "context": 2500, "compute": 2500, "total": 10000 }
        });
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        // evict_batch = 2 and two 1000-byte evictions are needed to reach the
        // budget (10000 -> 8000): the two oldest unpinned (lru1, lru2) are
        // deleted; pinned is never touched.
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert_eq!(deletes.len(), 2);
        assert!(deletes.contains(&"/instances/lru1"));
        assert!(deletes.contains(&"/instances/lru2"));
        assert!(!deletes.iter().any(|p| p.contains("pinned")));
    }

    #[tokio::test]
    async fn residency_eviction_frees_largest_lru_context_first() {
        // Two candidates with the same last_used: the one with more VRAM is
        // evicted first (freeing the most VRAM from the coldest context). A
        // pinned instance keeps the model's weights resident (no whole-model
        // candidate), isolating the context-level ordering.
        let mut big = instance_info("big", "g", false, 100);
        big.vram_bytes = 5000;
        let mut small = instance_info("small", "g", false, 100);
        small.vram_bytes = 1000;
        let envelope = serde_json::json!({
            "instances": [ small, big, instance_info("keep", "g", true, 0) ],
            "snapshots": [],
            "total": { "model": 5000, "context": 2000, "compute": 2000, "total": 9000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.evict_batch = 1;
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        let deletes: Vec<String> = stub
            .recorded()
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.clone())
            .collect();
        assert_eq!(deletes, vec!["/instances/big"], "largest LRU context evicted first");
    }

    #[tokio::test]
    async fn residency_no_eviction_when_free_vram_within_budget() {
        let envelope = serde_json::json!({
            "instances": [],
            "snapshots": [],
            "total": { "model": 1000, "context": 200, "compute": 200, "total": 1400 }
        });
        // used 1400 <= budget 8000 -> no eviction.
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        assert!(
            stub.recorded()
                .iter()
                .all(|(m, _, _)| m != "DELETE"),
            "no eviction within budget"
        );
    }

    #[tokio::test]
    async fn residency_polls_without_budget_and_never_evicts() {
        // No vram_total_bytes and no minimum_remaining_vram -> no budget; the
        // pass must still GET /instances and report, but must never DELETE.
        let envelope = serde_json::json!({
            "instances": [],
            "snapshots": [],
            "total": { "model": 5000, "context": 2000, "compute": 2000, "total": 9000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = None;
        policy.minimum_remaining_vram = None;
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        let recorded = stub.recorded();
        assert!(
            recorded.iter().any(|(m, p, _)| m == "GET" && p == "/instances"),
            "instances are always polled, budget or not"
        );
        assert!(
            recorded.iter().all(|(m, _, _)| m != "DELETE"),
            "no eviction without a budget"
        );
    }

    #[tokio::test]
    async fn ensure_group_allocates_fresh_instance_from_profile() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "POST" && path == "/instances" {
                    (201, "{}".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("swarm0".into()),
            group: Some("swarm".into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager.ensure_group("swarm").await.expect("ensure_group");

        let recorded = stub.recorded();
        let creates = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .collect::<Vec<_>>();
        assert_eq!(creates.len(), 1);
        let body: serde_json::Value = serde_json::from_str(&creates[0].2).unwrap();
        assert_eq!(body["group"], "swarm");
        assert_eq!(body["ctx_size"], 16384);
        let name = body["name"].as_str().unwrap();
        assert!(name.starts_with("swarm-"), "unique name generated: {name}");
    }

    #[tokio::test]
    async fn pool_aggregates_models_with_rewritten_ids() {
        use crate::instances::stub::StubServer;

        // Two managed models, each served by its own stub: `swarm` (ledger
        // default + scratch) and `qwen` (work). Envelopes use the server's own
        // (bare) instance ids and byte totals.
        let handler_a: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    let ledger = InstanceInfo {
                        is_default: true,
                        ..instance_info("ledger", "ledger", true, 1)
                    };
                    (
                        200,
                        serde_json::json!({
                            "instances": [
                                ledger,
                                instance_info("scratch", "scratch", false, 9),
                            ],
                            "snapshots": [ { "name": "readfiles", "size": 4194304 } ],
                            "total": { "model": 2428416000u64, "context": 2148925440u64, "compute": 2220361792u64, "total": 6797703232u64 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, "{}".into())
                }
            },
        );
        let handler_b: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [ instance_info("work", "work", false, 3) ],
                            "snapshots": [],
                            "total": { "model": 5000000000u64, "context": 1000, "compute": 1000, "total": 5000002000u64 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, "{}".into())
                }
            },
        );
        let stub_a = StubServer::start(handler_a);
        let stub_b = StubServer::start(handler_b);
        let mut managers = HashMap::new();
        managers.insert(
            "swarm".into(),
            Arc::new(InstanceManager::new(
                "swarm",
                InstanceClient::new(reqwest::Client::new(), stub_a.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )),
        );
        managers.insert(
            "qwen".into(),
            Arc::new(InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), stub_b.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )),
        );
        let pool = InstancePool::from_managers(managers, None);

        let agg = pool.aggregate(None).await.expect("aggregate");
        let instances = agg["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 3);
        let ids: Vec<&str> = instances
            .iter()
            .filter_map(|i| i["id"].as_str())
            .collect();
        assert!(ids.contains(&"swarm:ledger"), "ids: {ids:?}");
        assert!(ids.contains(&"swarm:scratch"));
        assert!(ids.contains(&"qwen:work"));
        // The default instance's aliases carry the bare model id + latest.
        let ledger = instances.iter().find(|i| i["id"] == "swarm:ledger").unwrap();
        let aliases: Vec<&str> = ledger["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|a| a.as_str())
            .collect();
        assert!(aliases.contains(&"swarm"));
        assert!(aliases.contains(&"swarm:latest"));
        assert!(aliases.contains(&"swarm:ledger"));
        // Snapshots tagged with the owning model; totals summed with 64-bit
        // arithmetic (each model's weights counted once).
        let snaps = agg["snapshots"].as_array().unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["model"], "swarm");
        let total = &agg["total"];
        assert_eq!(total["model"], 2428416000u64 + 5000000000u64);
        assert_eq!(total["total"], 6797703232u64 + 5000002000u64);
    }

    #[tokio::test]
    async fn pool_scopes_aggregate_to_one_model() {
        use crate::instances::stub::StubServer;

        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [ instance_info("ledger", "ledger", true, 1) ],
                            "snapshots": [],
                            "total": { "model": 1, "context": 1, "compute": 1, "total": 3 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let mut managers = HashMap::new();
        managers.insert(
            "swarm".into(),
            Arc::new(InstanceManager::new(
                "swarm",
                InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )),
        );
        let pool = InstancePool::from_managers(managers, None);
        let agg = pool.aggregate(Some("swarm")).await.expect("scoped aggregate");
        assert_eq!(agg["instances"].as_array().unwrap().len(), 1);
        let unknown = pool.aggregate(Some("nope")).await.expect("unknown scope");
        assert_eq!(unknown["instances"].as_array().unwrap().len(), 0);
        assert!(pool.resolve_instance_id("swarm:ledger").is_some());
        assert!(pool.resolve_instance_id("swarm:ledger:x").is_none());
        assert!(pool.resolve_instance_id("nope:x").is_none());
    }

    /// Build a stub server that serves a fixed `/instances` envelope and
    /// records destroys; DELETE answers 200.
    fn residency_stub(envelope: serde_json::Value) -> StubServer {
        let envelope = envelope.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, envelope.clone())
                } else {
                    (200, "{}".into())
                }
            },
        );
        StubServer::start(handler)
    }

    fn manager_for_stub(stub: &StubServer) -> Arc<InstanceManager> {
        Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            sidecar_policy(),
        ))
    }

    #[tokio::test]
    async fn pool_residency_evicts_lru_largest_unpinned_across_managers() {
        // Device budget = 10000 (vram_total) - 2000 (minimum_remaining) = 8000.
        // Both managers together report used 10000 -> over budget. The coldest
        // largest unpinned context (old) is evicted first; pinned never is.
        let env_a = serde_json::json!({
            "instances": [
                instance_info("pinned", "g", true, 0),    // exempt
                instance_info("old", "g", false, 100),    // LRU, vram 1000
                instance_info("big", "g", false, 200),    // larger vram 1000
            ],
            "snapshots": [],
            "total": { "model": 5000, "context": 2500, "compute": 2500, "total": 10000 }
        });
        let env_b = serde_json::json!({
            "instances": [],
            "snapshots": [],
            "total": { "model": 0, "context": 0, "compute": 0, "total": 0 }
        });
        let stub_a = residency_stub(env_a);
        let stub_b = residency_stub(env_b);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&stub_a));
        managers.insert("other".into(), manager_for_stub(&stub_b));
        let pool = InstancePool::from_managers(managers, None);

        pool.residency_cycle().await.expect("residency");

        let deletes_a = stub_a
            .recorded()
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.clone())
            .collect::<Vec<_>>();
        // evict_batch = 2 (sidecar_policy) -> both unpinned candidates go,
        // ordered LRU (old) before big; pinned never is.
        assert_eq!(deletes_a.len(), 2, "evict_batch = 2 per pass");
        assert!(deletes_a[0].ends_with("/old"), "LRU evicted first: {deletes_a:?}");
        assert!(
            !deletes_a.iter().any(|p| p.contains("pinned")),
            "pinned instance never evicted"
        );
        assert!(
            stub_b.recorded().iter().all(|(m, _, _)| m != "DELETE"),
            "other manager has no unpinned candidates"
        );
    }

    #[tokio::test]
    async fn pool_residency_no_eviction_within_budget() {
        // used 5000 <= budget 8000 -> no eviction.
        let env = serde_json::json!({
            "instances": [ instance_info("warm", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 4000, "context": 500, "compute": 500, "total": 5000 }
        });
        let stub = residency_stub(env);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        assert!(
            stub.recorded().iter().all(|(m, _, _)| m != "DELETE"),
            "no eviction within budget"
        );
    }

    #[tokio::test]
    async fn pool_residency_without_budget_never_evicts() {
        let mut policy = sidecar_policy();
        policy.minimum_remaining_vram = None;
        policy.vram_total_bytes = None;
        let env = serde_json::json!({
            "instances": [ instance_info("warm", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 4000, "context": 500, "compute": 500, "total": 5000 }
        });
        let stub = residency_stub(env);
        let manager = Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        assert!(
            stub.recorded().iter().all(|(m, _, _)| m != "DELETE"),
            "no budget -> no eviction"
        );
    }

    #[test]
    fn build_instance_managers_rejects_duplicate_name_within_model() {
        // Two profiles in ONE model resolve to the same instance name: the
        // profile key `swarm0` and another profile whose explicit `name` is
        // also `swarm0`. The pool grammar is invalid and boot fails fast.
        let config: crate::config::RouterConfig =
            serde_json::from_value(serde_json::json!({
                "models": {
                    "a": {
                        "endpoint": "http://x/v1/chat/completions",
                        "intelligence": 1,
                        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 1,
                        "instances": {
                            "swarm0": { "num_ctx": 16384 },
                            "x": { "name": "swarm0", "num_ctx": 32768 }
                        }
                    }
                }
            }))
            .unwrap();
        let err = match build_instance_managers(&config, None) {
            Err(e) => e,
            Ok(_) => panic!("duplicate-name config must fail validation"),
        };
        assert!(
            err.contains("duplicate instance name 'swarm0'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn residency_backoff_progresses_and_caps() {
        let base = Duration::from_secs(5);
        // Healthy -> base interval.
        assert_eq!(InstanceManager::residency_backoff(base, 0), Duration::from_secs(5));
        // First failure -> 2x; second -> 3x; ... capped at 12x.
        assert_eq!(InstanceManager::residency_backoff(base, 1), Duration::from_secs(10));
        assert_eq!(InstanceManager::residency_backoff(base, 2), Duration::from_secs(15));
        assert_eq!(
            InstanceManager::residency_backoff(base, 11),
            Duration::from_secs(60),
            "cap at 12x base"
        );
        assert_eq!(
            InstanceManager::residency_backoff(base, 100),
            Duration::from_secs(60),
            "capped regardless of further failures"
        );
    }

    #[test]
    fn build_instance_managers_ok_on_valid_config() {
        let config: crate::config::RouterConfig =
            serde_json::from_value(serde_json::json!({
                "models": {
                    "a": {
                        "endpoint": "http://x/v1/chat/completions",
                        "intelligence": 1,
                        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 1,
                        "instances": { "swarm": { "num_ctx": 16384, "count": 2 } }
                    }
                }
            }))
            .unwrap();
        let pool = build_instance_managers(&config, None).expect("valid config builds managers");
        assert_eq!(pool.managers_iter().count(), 1);
        // Keyed by the Coral Router model id.
        assert!(pool.manager("a").is_some());
    }

    #[test]
    fn build_instance_managers_is_empty_without_instances() {
        let config: crate::config::RouterConfig =
            serde_json::from_value(serde_json::json!({
                "models": {
                    "a": {
                        "endpoint": "http://x/v1/chat/completions",
                        "intelligence": 1,
                        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 1,
                    }
                }
            }))
            .unwrap();
        let pool = build_instance_managers(&config, None).expect("no managers");
        assert!(pool.is_empty());
    }

    // -- plain (no-instance-grammar) model footprint --------------------------

    /// A stub that 404s `/instances` (the fork's behavior for a server started
    /// without `--instance` grammar) and answers `/props`.
    fn plain_stub(props: serde_json::Value) -> StubServer {
        let props = props.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/props" {
                    (200, props.clone())
                } else {
                    (404, r#"{"error":{"message":"File Not Found"}}"#.into())
                }
            },
        );
        StubServer::start(handler)
    }

    fn plain_manager(stub: &StubServer, weights: u64) -> Arc<InstanceManager> {
        Arc::new(
            InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )
            .with_weights_bytes(weights),
        )
    }

    #[tokio::test]
    async fn plain_model_footprint_reports_weights_when_awake() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 16384 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 10_000_000_000);
        let (envelope, plain) = manager.list_with_fallback().await.expect("fallback");
        assert!(plain, "a 404 on /instances is synthesized");
        assert_eq!(envelope.instances.len(), 1);
        let inst = &envelope.instances[0];
        assert_eq!(inst.id, "qwen:default");
        assert_eq!(inst.state, "loaded");
        assert_eq!(inst.model_bytes, 10_000_000_000);
        assert_eq!(inst.n_ctx, 16384);
        assert_eq!(envelope.total.model, 10_000_000_000);
        assert_eq!(envelope.total.total, 10_000_000_000);
    }

    #[tokio::test]
    async fn plain_model_footprint_zeroes_weights_when_sleeping() {
        let props = serde_json::json!({
            "is_sleeping": true,
            "default_generation_settings": { "n_ctx": 16384 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 10_000_000_000);
        let (envelope, plain) = manager.list_with_fallback().await.expect("fallback");
        assert!(plain);
        let inst = &envelope.instances[0];
        assert_eq!(inst.state, "sleeping");
        assert_eq!(inst.model_bytes, 0, "sleeping plain model freed its weights");
        assert_eq!(envelope.total.total, 0);
    }

    #[tokio::test]
    async fn plain_model_footprint_none_when_server_down() {
        // A down (never-loaded) plain server: /props is unreachable -> None.
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> =
            Arc::new(|_m, _p, _b| (404, "{}".into()));
        let stub = StubServer::start(handler);
        let manager = plain_manager(&stub, 1_000);
        assert!(manager.list_with_fallback().await.is_none());
    }

    #[tokio::test]
    async fn aggregate_includes_plain_model_footprint() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), plain_manager(&stub, 5_000_000_000));
        let pool = InstancePool::from_managers(managers, None);

        let agg = pool.aggregate(None).await.expect("aggregate");
        let instances = agg["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 1);
        let entry = &instances[0];
        assert_eq!(entry["id"], "qwen:default");
        assert_eq!(entry["model_bytes"], 5_000_000_000u64);
        let aliases: Vec<&str> = entry["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|a| a.as_str())
            .collect();
        assert!(aliases.contains(&"qwen"), "aliases: {aliases:?}");
        assert_eq!(agg["total"]["model"], 5_000_000_000u64);
        assert_eq!(agg["total"]["total"], 5_000_000_000u64);

        let scoped = pool.aggregate(Some("qwen")).await.expect("scoped");
        assert_eq!(scoped["instances"].as_array().unwrap().len(), 1);
        let unknown = pool.aggregate(Some("nope")).await.expect("unknown scope");
        assert_eq!(unknown["instances"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn list_models_includes_plain_model_entry() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), plain_manager(&stub, 5_000_000_000));
        let pool = InstancePool::from_managers(managers, None);

        let models = pool.list_models().await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["id"], "qwen:default");
        assert_eq!(models[0]["state"], "loaded");
    }

    #[tokio::test]
    async fn touch_advances_plain_model_last_used() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 1_000);
        let before = manager.plain_footprint().await.expect("footprint").last_used;
        manager.touch();
        let after = manager.plain_footprint().await.expect("footprint").last_used;
        assert!(after >= before, "touch must advance last_used");
    }

    #[tokio::test]
    async fn residency_polls_plain_models_and_survives_without_supervisor() {
        // A plain model awake at 10_000 bytes, budget 2000: over budget. With
        // no supervisor the plain-model unload is a no-op break; the pass must
        // still complete Ok and poll the plain server's /props.
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000);
        policy.minimum_remaining_vram = Some(2000);
        let manager = Arc::new(
            InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
                Vec::new(),
                policy,
            )
            .with_weights_bytes(10_000),
        );
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency completes without supervisor");
        assert!(
            stub.recorded().iter().any(|(m, p, _)| m == "GET" && p == "/props"),
            "the plain-model branch must poll /props"
        );
    }

    // -- load-time admission control (make_room_for) -------------------------

    #[tokio::test]
    async fn make_room_for_no_eviction_within_budget() {
        // swarm holds one unpinned instance (used 3000); loading gemma (1000)
        // projects 4000 <= budget 8000 -> nothing evicted.
        let swarm_envelope = serde_json::json!({
            "instances": [ instance_info("scratch", "scratch", false, 5) ],
            "snapshots": [],
            "total": { "model": 2000, "context": 500, "compute": 500, "total": 3000 }
        });
        let swarm_stub = residency_stub(swarm_envelope);
        let gemma_stub = plain_stub(serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        }));
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&swarm_stub));
        managers.insert("gemma".into(), plain_manager(&gemma_stub, 1000));
        let pool = InstancePool::from_managers(managers, None);
        pool.make_room_for("gemma", 1000).await;
        assert!(
            swarm_stub
                .recorded()
                .iter()
                .all(|(m, _, _)| m != "DELETE"),
            "no eviction within budget"
        );
    }

    #[tokio::test]
    async fn make_room_for_evicts_unpinned_instance_over_budget() {
        // Budget 4000 - 2000 = 2000; used 3000 + gemma 1000 = 4000 -> over.
        // The only freeable chunk is the unpinned `scratch` instance.
        let envelope = serde_json::json!({
            "instances": [ instance_info("scratch", "scratch", false, 5) ],
            "snapshots": [],
            "total": { "model": 2000, "context": 500, "compute": 500, "total": 3000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000);
        let manager = Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.make_room_for("gemma", 1000).await;
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/scratch"),
            "unpinned instance evicted to make room: {deletes:?}"
        );
    }

    #[tokio::test]
    async fn make_room_for_excludes_target_and_survives_without_supervisor() {
        // qwen (plain, awake, 10_000) resident; gemma (plain, 7_000) is the
        // cold target. Budget 2000 -> over budget. Without a supervisor the
        // plain unload is a no-op break; the pass must complete Ok and never
        // poll the excluded target.
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let qwen_stub = plain_stub(props.clone());
        let gemma_stub = plain_stub(props);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000);
        let qwen = Arc::new(
            InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), qwen_stub.base_url(), None),
                Vec::new(),
                policy.clone(),
            )
            .with_weights_bytes(10_000),
        );
        let gemma = Arc::new(
            InstanceManager::new(
                "gemma",
                InstanceClient::new(reqwest::Client::new(), gemma_stub.base_url(), None),
                Vec::new(),
                policy,
            )
            .with_weights_bytes(7_000),
        );
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), qwen);
        managers.insert("gemma".into(), gemma);
        let pool = InstancePool::from_managers(managers, None);
        pool.make_room_for("gemma", 7_000).await;
        assert!(
            gemma_stub.recorded().is_empty(),
            "the cold target must be excluded from the gather: {:?}",
            gemma_stub.recorded()
        );
    }

    #[tokio::test]
    async fn is_sleeping_reflects_fork_state_and_skips_instance_models() {
        let props = serde_json::json!({
            "is_sleeping": true,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 1000);
        assert_eq!(manager.is_sleeping().await, Some(true), "fork reports sleeping");

        let stub2 = plain_stub(serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        }));
        let manager2 = plain_manager(&stub2, 1000);
        assert_eq!(manager2.is_sleeping().await, Some(false), "awake");

        // Instance models never poll /props: their pinned contexts keep the
        // weights resident, so the answer is always Some(false).
        let stub3 = plain_stub(serde_json::json!({ "is_sleeping": true }));
        let instance_manager = InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub3.base_url(), None),
            vec![profile("ledger", "ledger")],
            sidecar_policy(),
        );
        assert_eq!(instance_manager.is_sleeping().await, Some(false));
        assert!(
            stub3.recorded().is_empty(),
            "instance models must not poll /props: {:?}",
            stub3.recorded()
        );
    }

    // -- resume (preserve-on-evict) ------------------------------------------

    #[test]
    fn eviction_score_weights_size_and_coldness() {
        let now = 1_000_000i64;
        // Equal coldness: the larger footprint scores higher (evicted first) -
        // a 10 GB weight pool outranks a context buffer of any recency within
        // reach, which is the OOM-avoidance priority.
        assert!(eviction_score(10_000_000_000, 100, now) > eviction_score(1_000, 100, now));
        assert!(eviction_score(10_000_000_000, now - 1, now) > eviction_score(1_000, now - 3600, now));
        // Equal size: the colder (older last_used) scores higher.
        assert!(eviction_score(1_000, 50, now) > eviction_score(1_000, 900, now));
        // Coldness scales within a size class: the same 10 GB pool idle a
        // minute is far more evictable than when used a second ago, so active
        // work is relatively protected by recency.
        assert!(eviction_score(10_000_000_000, now - 60, now) > eviction_score(10_000_000_000, now - 1, now));
        // Never used = maximally cold.
        assert!(eviction_score(1_000, -1, now) > eviction_score(1_000, 1, now));
    }

    #[test]
    fn resume_flag_round_trips_through_profiles() {
        // A profile with `resume: true` seeds the manager's map so the
        // aggregate and eviction see it.
        let manager = InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), "http://x", None),
            vec![
                InstanceProfile {
                    resume: true,
                    ..profile("agent", "g")
                },
                profile("scratch", "g"),
            ],
            sidecar_policy(),
        );
        assert!(manager.resume_for("agent"));
        assert!(!manager.resume_for("scratch"));
        manager.set_resume("scratch", true);
        assert!(manager.resume_for("scratch"));
        manager.set_resume("agent", false);
        assert!(!manager.resume_for("agent"));
    }

    #[tokio::test]
    async fn aggregate_reports_resume_overlay() {
        let envelope = serde_json::json!({
            "instances": [ instance_info("agent", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 1000, "context": 500, "compute": 500, "total": 2000 }
        });
        let stub = residency_stub(envelope);
        let manager = manager_for_stub(&stub);
        manager.set_resume("agent", true);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        let agg = pool.aggregate(None).await.expect("aggregate");
        let entry = agg["instances"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["id"] == "swarm:agent")
            .expect("aggregated entry");
        assert_eq!(entry["resume"], true, "aggregate overlays the router-side flag");
    }

    #[tokio::test]
    async fn eviction_snapshots_resume_context_before_destroy() {
        // Budget 2000; over budget by a resume-marked unpinned context with a
        // pinned sibling keeping the weights resident. The resume context is
        // snapshotted (`POST .../agent/snapshot`) before it is destroyed, and
        // the pinned sibling is never touched.
        let mut agent = instance_info("agent", "g", false, 100);
        agent.vram_bytes = 2000;
        let envelope = serde_json::json!({
            "instances": [ instance_info("keep", "g", true, 0), agent ],
            "snapshots": [],
            "total": { "model": 5000, "context": 3000, "compute": 3000, "total": 11000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000); // budget = 4000 - 2000 = 2000
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        manager.set_resume("agent", true);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        let recorded = stub.recorded();
        let snapshot_posts: Vec<&str> = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && *p == "/instances/agent/snapshot")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert_eq!(snapshot_posts.len(), 1, "resume context snapshotted: {recorded:?}");
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(deletes.contains(&"/instances/agent"), "resume context evicted");
        assert!(
            !deletes.iter().any(|p| p.contains("keep")),
            "pinned context never evicted"
        );
    }

    #[tokio::test]
    async fn expire_resume_clears_idle_context_and_deletes_snapshot() {
        // `resume_ttl_s = 60`: an ancient (idle) resume context has its flag
        // cleared and its `-resume` snapshot deleted - the router concluding
        // the work is done. Within budget, so no eviction happens.
        let envelope = serde_json::json!({
            "instances": [
                instance_info("agent", "g", false, 100), // idle ~50 years
                instance_info("keep", "g", true, 0),
            ],
            "snapshots": [],
            "total": { "model": 1000, "context": 500, "compute": 500, "total": 2000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.resume_ttl_s = Some(60);
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        manager.set_resume("agent", true);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        assert!(
            !pool.manager("base").unwrap().resume_for("agent"),
            "idle resume cleared"
        );
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/agent/snapshot/agent-resume"),
            "resume snapshot deleted on expiry: {deletes:?}"
        );
    }

    #[tokio::test]
    async fn set_resume_false_deletes_snapshot() {
        let envelope = serde_json::json!({
            "instances": [ instance_info("agent", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 1000, "context": 500, "compute": 500, "total": 2000 }
        });
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.set_resume("swarm", "agent", true).await.expect("enable");
        assert!(pool.manager("swarm").unwrap().resume_for("agent"));
        pool.set_resume("swarm", "agent", false).await.expect("disable");
        assert!(!pool.manager("swarm").unwrap().resume_for("agent"));
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/agent/snapshot/agent-resume"),
            "disable deletes the resume snapshot: {deletes:?}"
        );
    }

    #[tokio::test]
    async fn whole_model_is_largest_footprint_candidate() {
        // A model with NO pinned instances is a whole-model candidate: its
        // weights + all contexts. Without a supervisor the model eviction
        // still drops every context before breaking; the point is that the
        // whole-model unit outranks the individual contexts.
        let envelope = serde_json::json!({
            "instances": [
                instance_info("ctx-a", "g", false, 100),
                instance_info("ctx-b", "g", false, 200),
            ],
            "snapshots": [],
            "total": { "model": 8000, "context": 2000, "compute": 2000, "total": 12000 }
        });
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/ctx-a") && deletes.contains(&"/instances/ctx-b"),
            "whole-model eviction drops every context: {deletes:?}"
        );
    }
}
