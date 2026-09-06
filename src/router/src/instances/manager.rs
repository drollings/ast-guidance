//! The sidecar owner of instance lifecycle for ONE spawned `llama-server`:
//! boot reconciliation, on-demand allocation, and the residency-adjacent
//! per-manager queries (`list_with_fallback`/`plain_footprint`) the pool and
//! the `/instances` aggregation surface drive. Runs as a task on the router's
//! tokio runtime (owned by the server).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use common_core::hash::uuid_v4;
use common_core::retry::PollWithBackoff;
use crate::cli::gguf::{compute_short_id, quant_name, read_gguf_metadata};
use crate::config::InstanceProfile;

use super::client::{InstanceClient, InstanceError, InstanceInfo, InstanceList, InstanceTotals};
use super::{instance_grammar_string, management_base_url, validate_instances};
use super::pool::InstancePool;

/// The weights-file identity of one managed model, surfaced on the aggregate
/// `/instances` envelope so `coral-router ps` can display it without needing
/// the weights file on the CLI's host. The router computes it from the file it
/// actually loaded (authoritative); empty when the model has no weights path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightsIdentity {
    pub short_id: String,
    pub arch: String,
    pub quant: String,
}

/// Derive a model's weights identity from its GGUF file on disk. Best-effort:
/// a missing/unreadable file yields empty strings (callers fall back).
pub fn weights_identity(weights_path: &Path) -> WeightsIdentity {
    let short_id = compute_short_id(weights_path);
    let meta = read_gguf_metadata(weights_path);
    let arch = meta
        .get("general.architecture")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let quant = meta
        .get("general.file_type")
        .and_then(Value::as_u64)
        .map(|ft| quant_name(ft as u32))
        .unwrap_or_default();
    WeightsIdentity {
        short_id,
        arch,
        quant,
    }
}

/// A held dispatch on one managed server: releases the manager's in-flight
/// count on drop, so every exit path (response, error, abort) frees the hold.
pub struct InflightLease {
    manager: Arc<InstanceManager>,
}

impl Drop for InflightLease {
    fn drop(&mut self) {
        self.manager.in_flight_count.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The sidecar owner of instance lifecycle for ONE spawned `llama-server`.
/// Holds the model key (public id), the management client talking directly to
/// that server, the expanded configured profiles, and the residency policy.
/// Runs as a task on the router's tokio runtime (owned by the server).
pub struct InstanceManager {
    model_key: String,
    client: InstanceClient,
    pub(super) profiles: Vec<InstanceProfile>,
    pub(super) policy: crate::config::SidecarConfig,
    /// Resident weights bytes of a plain (no-instance-grammar) model, from the
    /// configured `weights` file. Instance models report `model_bytes` through
    /// the fork's `/instances`; plain models need this to surface in the
    /// aggregate envelope and the residency budget.
    weights_bytes: u64,
    /// Whether the server speaks the fork's `/instances` management API.
    /// `false` for an adopted grammar-less orphan (spawned without
    /// `--instance` flags, so the route 404s): reconcile and on-demand
    /// creates are suspended until an explicit unload respawns it with the
    /// correct grammar, while generation and the `/props` footprint keep
    /// working. Set once at construction from the supervisor's adoption
    /// record; a mid-life respawn self-heals via `adoption_info`.
    has_instances_api: AtomicBool,
    /// Router-tracked last-use for plain models (the fork reports no
    /// `last_used` for them). Updated by [`Self::touch`] on every dispatch; the
    /// residency loop orders plain-model unloads by it.
    last_used: AtomicI64,
    /// Active dispatches currently holding this manager's server. Each
    /// dispatch holds an [`InflightLease`] across its server call; the
    /// residency engine never evicts a weights instance with a nonzero
    /// count, so a model is never unloaded mid-inference.
    in_flight_count: AtomicUsize,
    /// Router-side preserve-on-evict map: instance name -> `resume`. Seeded
    /// from the configured profiles, updated by [`Self::set_resume`]. The fork
    /// knows nothing of it; the aggregate overlays it on the envelope.
    resume: Mutex<HashMap<String, bool>>,
    /// The weights-file identity (`short_id`/`arch`/`quant`) of this model,
    /// surfaced on the aggregate envelope for `coral-router ps`.
    identity: WeightsIdentity,
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
            has_instances_api: AtomicBool::new(true),
            last_used: AtomicI64::new(-1),
            in_flight_count: AtomicUsize::new(0),
            resume: Mutex::new(resume),
            identity: WeightsIdentity::default(),
        }
    }

    /// Builder-style: set the resident weights size of a plain (no-instance)
    /// model so the aggregate and residency loops can report it.
    #[must_use]
    pub fn with_weights_bytes(mut self, bytes: u64) -> Self {
        self.weights_bytes = bytes;
        self
    }

    /// Builder-style: set the weights-file identity (`short_id`/`arch`/`quant`)
    /// so the aggregate `/instances` envelope can surface it.
    #[must_use]
    pub fn with_weights_identity(mut self, identity: WeightsIdentity) -> Self {
        self.identity = identity;
        self
    }

    /// The weights-file identity of this model (empty fields when the model has
    /// no weights path).
    pub fn weights_identity(&self) -> &WeightsIdentity {
        &self.identity
    }

    /// Whether this manager's model declares an instance pool. Only instance
    /// models expose `/instances` on their server; a plain (weights-only)
    /// model's server 404s on it and needs a synthesized footprint instead.
    pub fn has_pool(&self) -> bool {
        !self.profiles.is_empty()
    }

    /// Whether the server speaks the fork's `/instances` API. `false` only
    /// for an adopted grammar-less orphan (see the field docs).
    pub fn has_instances_api(&self) -> bool {
        self.has_instances_api.load(Ordering::Relaxed)
    }

    /// Mark the server as grammar-less (adopted without `--instance` flags)
    /// or fully API-capable. Called once at construction from the
    /// supervisor's adoption record.
    pub fn set_instances_supported(&self, supported: bool) {
        self.has_instances_api.store(supported, Ordering::Relaxed);
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
    /// Instance models with a live management API never report sleeping
    /// (pinned contexts keep their weights resident), so `Some(false)`
    /// short-circuits without a call. A grammar-less adopted pool model has no
    /// `/instances` but its `/props` sleep flag is still meaningful.
    pub async fn is_sleeping(&self) -> Option<bool> {
        if self.has_pool() && self.has_instances_api() {
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

    /// The router-tracked last use (seconds; `-1` = never used). This is the
    /// model-level recency the shared `LlmWeights` surface reports for a llama
    /// weights instance (instance-level recency comes from the fork envelope).
    pub fn last_used(&self) -> i64 {
        self.last_used.load(Ordering::Relaxed)
    }

    /// Active dispatches currently holding this manager's server.
    pub fn in_flight(&self) -> usize {
        self.in_flight_count.load(Ordering::Relaxed)
    }

    /// Hold this manager's server across one dispatch: the count is released
    /// when the lease drops (end of the server call, success or failure), so
    /// the residency engine never evicts a model mid-inference.
    pub fn hold_in_flight(self: &Arc<Self>) -> InflightLease {
        self.in_flight_count.fetch_add(1, Ordering::Relaxed);
        InflightLease { manager: Arc::clone(self) }
    }

    /// Ask this model's server to stop the generation running in a slot.
    /// Invoked on a downstream streaming disconnect (the abort arm of the
    /// dispatch path), as belt-and-suspenders on top of the transport close:
    /// the router is the process owner, so it can reach the owning server's
    /// `/abort` directly. Best-effort — a non-running slot or an unreachable
    /// server is logged and ignored.
    pub async fn abort_generation(&self, id_slot: Option<i32>) -> Result<(), InstanceError> {
        self.client.abort(id_slot).await
    }

    /// The resident footprint of a plain (no-instance-grammar) managed model.
    ///
    /// The fork exposes no `/instances` for these servers, so Coral Router
    /// synthesizes one envelope entry: `model_bytes` is the configured weights
    /// file size, or 0 when the server reports `is_sleeping` (the fork's idle
    /// sleep has moved the weights out of VRAM). `state` mirrors that flag.
    /// `None` when the server is unreachable (down or never loaded).
    ///
    /// Also serves an adopted grammar-less pool model: a server spawned
    /// without `--instance` flags 404s `/instances`, but its `/props` still
    /// reports the resident weights, so the pool keeps residency visibility
    /// instead of going blind until the next unload respawns it with grammar.
    pub(super) async fn plain_footprint(&self) -> Option<InstanceInfo> {
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
    pub(super) async fn list_with_fallback(&self) -> Option<(InstanceList, bool)> {
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
    /// `/instances`). An adopted grammar-less pool model is likewise skipped
    /// with a loud warning: its server 404s every management call, so
    /// reconcile would only spin; generation still works and the next explicit
    /// unload respawns it with the correct grammar.
    pub async fn reconcile(&self) -> Result<(), InstanceError> {
        if self.profiles.is_empty() {
            return Ok(());
        }
        if !self.has_instances_api() {
            tracing::warn!(
                target: "router.instances",
                model = %self.model_key,
                base_url = %self.client.base_url(),
                "instance reconcile suspended - adopted server has no /instances API (spawned without --instance flags); generation continues, unload to respawn with grammar",
            );
            return Ok(());
        }
        let existing = match self.client.list().await {
            Ok(envelope) => envelope,
            Err(e) => {
                // Quiet by design: `bootstrap` logs the first deferral (info)
                // and subsequent retries (debug), and the sibling paths below
                // propagate `Err` without logging. A WARN here would repeat on
                // every backoff tick — including for lazy models whose server
                // is legitimately not started yet.
                tracing::debug!(
                    target: "router.instances",
                    base_url = %self.client.base_url(),
                    error = %e,
                    "instance reconcile deferred - management API unreachable",
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
    /// or already exists. No-op on a grammar-less adopted server (its single
    /// context serves every target; the fork has no management API to create
    /// through).
    pub async fn ensure_instance(&self, name: &str) -> Result<(), InstanceError> {
        if name.is_empty() || self.profiles.is_empty() {
            return Ok(());
        }
        if !self.has_instances_api() {
            tracing::debug!(
                target: "router.instances",
                model = %self.model_key,
                instance = %name,
                "grammar-less server serves every target from its single context - nothing to create",
            );
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
    /// may come up after the router). Residency runs on the shared engine, so
    /// this task stops after a successful reconcile.
    pub async fn bootstrap(&self) {
        let base = Duration::from_secs(self.policy.poll_interval_s.max(1));
        let poll = PollWithBackoff::new(base, 12);
        let failures = std::sync::atomic::AtomicU32::new(0);
        poll.run(|| async {
            match self.reconcile().await {
                Ok(()) => true,
                Err(e) => {
                    let count =
                        failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if count == 1 {
                        tracing::info!(
                            target: "router.instances",
                            error = %e,
                            "instance reconcile deferred - management API not ready, retrying",
                        );
                    } else {
                        tracing::debug!(
                            target: "router.instances",
                            error = %e,
                            failures = count,
                            "instance reconcile still deferred, retrying",
                        );
                    }
                    false
                }
            }
        })
        .await;
    }

    /// Allocate a fresh instance for `group` on a 503 group-miss. Uses the
    /// group's configured profile (name/group/ctx/parallel/pinned) with a
    /// unique `<group>-<uuid>` name. No-op when no profile configures the
    /// group (there is nothing to allocate), or on a grammar-less adopted
    /// server (no management API to allocate through).
    pub async fn ensure_group(&self, group: &str) -> Result<(), InstanceError> {
        if !self.has_instances_api() {
            return Ok(());
        }
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

    /// Ensure the group has at least one resident member, creating on demand
    /// when it is entirely absent. Idempotent: no-op when any instance already
    /// belongs to the group. For a **pinned** group the boot reconciliation
    /// creates the canonical members (e.g. `swarm-0`/`swarm-1`); for an
    /// unpinned group a fresh member is allocated. Used by the classifier path,
    /// whose sync client cannot trigger the dispatch path's allocate-on-miss.
    pub async fn ensure_group_ready(&self, group: &str) -> Result<(), InstanceError> {
        if self.profiles.is_empty() {
            return Ok(());
        }
        if !self.has_instances_api() {
            return Ok(());
        }
        let existing = match self.client.list().await {
            Ok(envelope) => envelope,
            Err(e) => return Err(e),
        };
        if existing
            .instances
            .iter()
            .any(|i| i.group == group)
        {
            return Ok(());
        }
        let group_is_pinned = self
            .profiles
            .iter()
            .any(|p| p.group.as_deref() == Some(group) && p.pinned);
        if group_is_pinned {
            self.reconcile().await
        } else {
            self.ensure_group(group).await
        }
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
        // Onnx models are served by the ort registry, never by a llama-server
        // instance pool (ROADMAP_20260827_ORT §0.5).
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
            .and_then(|p| fluent_wvr::capability::capability_aware_fs::metadata(p).ok())
            .map_or(0, |m| m.len());
        let identity = entry
            .weights
            .as_deref()
            .map(|p| weights_identity(Path::new(p)))
            .unwrap_or_default();
        let manager = Arc::new(
            InstanceManager::new(key, client, profiles, config.sidecar.clone())
                .with_weights_bytes(weights_bytes)
                .with_weights_identity(identity),
        );
        // An adopted grammar-less orphan (no `--instance` flags, so no
        // `/instances` route) serves generation but cannot be reconciled:
        // suspend the management calls that would 404 against it. A mid-life
        // respawn self-heals — `adoption_info` verifies the live pid.
        if let Some(adoption) = supervisor
            .as_ref()
            .and_then(|sup| sup.adoption_info(key))
        {
            manager.set_instances_supported(adoption.instances_supported);
        }
        managers.insert(key.clone(), manager);
    }
    Ok(InstancePool::from_managers(managers, supervisor))
}

/// The instance NAME within a server-reported id: the segment after the last
/// `:` (the server reports `<model_alias>:<name>`). A bare id (no `:`) is the
/// name itself. Instance names never contain `:`.
pub(super) fn instance_name_from_server_id(id: &str) -> &str {
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
#[cfg(test)]
#[path = "../../tests/instances_manager.rs"]
mod tests;
