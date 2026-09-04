//! The llama half of the shared `LlmWeights`/`LlmContext`/`LlmKVCache`
//! contracts (ROADMAP_20260830_LLMS M4), plus the unified `LlmFleet` facade
//! the server builds from both the llama pool and the onnx registry.
//!
//! These are **thin adapters** (roadmap §3.3): every method is a one-line
//! delegation to the existing concrete types — `ManagedServer` +
//! `InstanceManager` + `InstanceClient` + `SnapshotStore`. The llama machinery
//! is not rewritten; the adapters let the shared residency engine, the
//! `/instances` facade, `/v1/models`, `ps`, and `POST /models/unload` drive
//! both fleets through the same trait surface.
//!
//! The `LlmFleet` holds the llama-only `InstancePool` (whose `aggregate`/
//! `list_models` stay **byte-identical** for llama) alongside the trait-object
//! weights list; onnx rows are appended by rendering `dyn LlmWeights`
//! residency rows. Llama behavior is never changed by the facade.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use fluent_llm::runtime::{
    EvictionPolicy, LlmContext, LlmKVCache, LlmResidencyRow, LlmRuntime, LlmRuntimeError,
    LlmWeights, SnapshotMeta,
};

use super::client::{InstanceClient, InstanceError, InstanceInfo, InstanceTotals};
use super::manager::{instance_name_from_server_id, resume_snapshot_name, InstanceManager};
use super::{instance_aliases, InstancePool};
use crate::config::{InstanceProfile, RouterConfig, SidecarConfig};
use crate::kv_cache::{KvSnapshot, SnapshotStore};
use crate::supervisor::{LlamaServerSupervisor, ManagedServer};

/// Map a llama management-client error onto the runtime-agnostic error type.
fn map_instance_err(e: InstanceError) -> LlmRuntimeError {
    match e {
        InstanceError::Transient { status, body } => LlmRuntimeError::Transient { status, body },
        InstanceError::Duplicate => LlmRuntimeError::Duplicate,
        InstanceError::Rejected { status, body } => LlmRuntimeError::Rejected { status, body },
        InstanceError::Network(s) => LlmRuntimeError::Network(s),
        InstanceError::Other(s) => LlmRuntimeError::Parse(s),
    }
}

/// Map a supervisor error (`String` from `ensure_running`) onto the runtime
/// error type, naming the model.
fn map_server_err(model: &str, e: &str) -> LlmRuntimeError {
    LlmRuntimeError::Other(format!("model '{model}': {e}"))
}

/// One `InstanceInfo` (the fork's envelope row) rendered as a shared
/// [`LlmResidencyRow`]. The fork's `last_used` clock is seconds — kept
/// pool-native (the shared engine's ordering expects it).
fn info_to_row(model_key: &str, info: &InstanceInfo) -> LlmResidencyRow {
    LlmResidencyRow {
        context_key: format!("{model_key}:{}", instance_name_from_server_id(&info.id)),
        group: info.group.clone(),
        n_ctx: info.n_ctx,
        parallel: info.parallel,
        pinned: info.pinned,
        resume: info.resume,
        state: info.state.clone(),
        runtime: LlmRuntime::Llama,
        model_bytes: info.model_bytes,
        context_bytes: info.context_bytes,
        compute_bytes: info.compute_bytes,
        total_bytes: info.total_bytes,
        vram_bytes: info.vram_bytes,
        last_used: info.last_used,
    }
}

/// A shared [`LlmResidencyRow`] rendered back as a fork `InstanceInfo` for the
/// unified `/instances` envelope. `is_default` is derived from the context
/// name (`default` — the synthesized plain-model / single-context shape); the
/// onnx implementors mark their rows `is_default` the same way.
pub(crate) fn row_to_info(model_key: &str, row: &LlmResidencyRow) -> InstanceInfo {
    let name = row
        .context_key
        .strip_prefix(&format!("{model_key}:"))
        .unwrap_or(&row.context_key)
        .to_string();
    InstanceInfo {
        id: row.context_key.clone(),
        aliases: instance_aliases(model_key, &row.context_key, &row.group, name == "default"),
        group: row.group.clone(),
        n_ctx: row.n_ctx,
        parallel: row.parallel,
        pinned: row.pinned,
        is_default: name == "default",
        resume: row.resume,
        state: row.state.clone(),
        model_bytes: row.model_bytes,
        context_bytes: row.context_bytes,
        compute_bytes: row.compute_bytes,
        total_bytes: row.total_bytes,
        vram_bytes: row.vram_bytes,
        last_used: row.last_used,
    }
}

/// The `LlmWeights` adapter for one managed llama-server: spawn/kill through
/// the supervisor's `ManagedServer`, the residency view through the manager's
/// `list_with_fallback` (synthesized plain-model footprint included), and the
/// `FootprintColdness` eviction ordering (the llama sidecar's).
pub struct LlamaWeights {
    model_key: String,
    server: Arc<ManagedServer>,
    manager: Arc<InstanceManager>,
    /// The supervisor's `llama-server` binary (needed for `ensure_running`).
    bin: PathBuf,
    /// Sidecar residency policy (drives the `expire_resume` hook).
    policy: SidecarConfig,
    /// The latest envelope snapshot (instance name → info) so the sync
    /// `context()` lookup resolves without a network call.
    last_rows: Mutex<HashMap<String, InstanceInfo>>,
}

impl LlamaWeights {
    /// Build the adapter over an existing manager + server. `bin` is the
    /// supervisor's resolved `llama-server` path.
    pub fn new(
        manager: Arc<InstanceManager>,
        server: Arc<ManagedServer>,
        bin: PathBuf,
        policy: SidecarConfig,
    ) -> Self {
        Self {
            model_key: manager.model_key().to_string(),
            server,
            manager,
            bin,
            policy,
            last_rows: Mutex::new(HashMap::new()),
        }
    }

    /// The configured `max_ctx` cap for a named instance profile, if any.
    fn profile_max_ctx(&self, name: &str) -> Option<u64> {
        self.manager
            .profiles()
            .iter()
            .find(|p| p.name.as_deref() == Some(name))
            .and_then(|p| p.max_ctx)
    }
}

#[async_trait::async_trait]
impl LlmWeights for LlamaWeights {
    fn model_key(&self) -> &str {
        &self.model_key
    }

    fn weights_bytes(&self) -> u64 {
        self.manager.weights_bytes()
    }

    fn pinned(&self) -> bool {
        self.manager.profiles().iter().any(|p| p.pinned)
    }

    fn refuse_unload(&self) -> bool {
        false // the supervisor may always unload a llama-server
    }

    fn is_loaded(&self) -> bool {
        self.server.is_running()
    }

    fn sleep_idle_seconds(&self) -> Option<i32> {
        None // llama weights are never router-side idle-released (the fork owns their sleep)
    }

    async fn ensure_loaded(&self) -> Result<(), LlmRuntimeError> {
        self.server
            .ensure_running(&self.bin)
            .await
            .map_err(|e| map_server_err(&self.model_key, &e))
    }

    async fn unload(&self) -> Result<(), LlmRuntimeError> {
        self.server.unload().await;
        Ok(())
    }

    fn touch(&self) {
        self.manager.touch();
    }

    fn last_used(&self) -> i64 {
        self.manager.last_used()
    }

    async fn residency_rows(&self) -> Vec<LlmResidencyRow> {
        let Some((envelope, _plain)) = self.manager.list_with_fallback().await else {
            return Vec::new(); // server down / never loaded → no rows
        };
        let mut cache = match self.last_rows.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        cache.clear();
        let mut rows = Vec::with_capacity(envelope.instances.len());
        for info in envelope.instances {
            let name = instance_name_from_server_id(&info.id).to_string();
            cache.insert(name, info.clone());
            rows.push(info_to_row(&self.model_key, &info));
        }
        rows
    }

    fn context(&self, name: &str) -> Option<Arc<dyn LlmContext>> {
        let cache = match self.last_rows.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let info = cache.get(name).cloned()?;
        Some(Arc::new(LlamaContext::from_info(
            Arc::clone(&self.manager),
            self.manager.client().clone(),
            &info,
            self.profile_max_ctx(name),
        )) as Arc<dyn LlmContext>)
    }

    async fn ensure_context(&self, name: &str) -> Result<Arc<dyn LlmContext>, LlmRuntimeError> {
        self.manager
            .ensure_instance(name)
            .await
            .map_err(map_instance_err)?;
        let profile = self
            .manager
            .profiles()
            .iter()
            .find(|p| p.name.as_deref() == Some(name))
            .ok_or_else(|| {
                LlmRuntimeError::NotLoaded(format!(
                    "no configured profile for instance '{name}'"
                ))
            })?;
        Ok(Arc::new(LlamaContext::from_profile(
            Arc::clone(&self.manager),
            self.manager.client().clone(),
            profile,
        )) as Arc<dyn LlmContext>)
    }

    fn eviction_policy(&self) -> EvictionPolicy {
        EvictionPolicy::FootprintColdness
    }

    async fn expire_resume(&self) {
        let Some(ttl) = self.policy.resume_ttl_s else {
            return;
        };
        let now = i64::try_from(common_core::now_secs()).unwrap_or(i64::MAX);
        let Some((envelope, _)) = self.manager.list_with_fallback().await else {
            return;
        };
        for info in envelope.instances {
            let name = instance_name_from_server_id(&info.id);
            if !self.manager.resume_for(name) {
                continue;
            }
            let idle = now.saturating_sub(info.last_used);
            if idle < ttl as i64 {
                continue;
            }
            self.manager.set_resume(name, false);
            let snapshot = resume_snapshot_name(name);
            match self.manager.client().delete_snapshot(name, &snapshot).await {
                Ok(()) => {
                    tracing::info!(
                        target: "router.instances",
                        model = %self.model_key,
                        instance = %name,
                        idle_secs = idle,
                        ttl_secs = ttl,
                        "resume expired - work concluded, snapshot dropped",
                    );
                    crate::audit::emit(
                        "instances",
                        serde_json::json!({
                            "action": "expire_resume",
                            "model": self.model_key,
                            "instance": name,
                            "reason": "idle_ttl",
                        }),
                    );
                }
                Err(e) => tracing::warn!(
                    target: "router.instances",
                    model = %self.model_key,
                    instance = %name,
                    error = %e,
                    "resume snapshot delete on expiry failed",
                ),
            }
        }
    }
}

/// The `LlmContext` adapter for one named llama-fork instance: resize →
/// `client.resize`, KV → `LlamaKVCache`, resume → the manager's router-side
/// flag, eviction → resume-snapshot-then-destroy (the sidecar's exact order).
pub struct LlamaContext {
    client: InstanceClient,
    manager: Arc<InstanceManager>,
    name: String,
    group: String,
    n_ctx: AtomicU64,
    max_ctx: Option<u64>,
    pinned: bool,
    vram_bytes: u64,
    last_used: AtomicI64,
}

impl LlamaContext {
    /// Build the adapter from a live fork envelope row.
    fn from_info(
        manager: Arc<InstanceManager>,
        client: InstanceClient,
        info: &InstanceInfo,
        max_ctx: Option<u64>,
    ) -> Self {
        Self {
            client,
            manager,
            name: instance_name_from_server_id(&info.id).to_string(),
            group: info.group.clone(),
            n_ctx: AtomicU64::new(info.n_ctx),
            max_ctx,
            pinned: info.pinned,
            vram_bytes: info.vram_bytes,
            last_used: AtomicI64::new(info.last_used),
        }
    }

    /// Build the adapter from a configured (expanded) profile — the shape
    /// `ensure_context` hands back before the fork's next envelope poll.
    fn from_profile(
        manager: Arc<InstanceManager>,
        client: InstanceClient,
        profile: &InstanceProfile,
    ) -> Self {
        let name = profile.name.clone().unwrap_or_default();
        let group = profile.group.clone().unwrap_or_else(|| name.clone());
        let n_ctx = if profile.num_ctx == 0 {
            // The `default_params.num_ctx` default (16384) — the plain-model
            // context size the supervisor hands a no-instance server.
            16_384
        } else {
            profile.num_ctx
        };
        Self {
            client,
            manager,
            name,
            group,
            n_ctx: AtomicU64::new(n_ctx),
            max_ctx: profile.max_ctx,
            pinned: profile.pinned,
            vram_bytes: 0,
            last_used: AtomicI64::new(-1),
        }
    }
}

#[async_trait::async_trait]
impl LlmContext for LlamaContext {
    fn name(&self) -> &str {
        &self.name
    }

    fn group(&self) -> &str {
        &self.group
    }

    fn n_ctx(&self) -> u64 {
        self.n_ctx.load(Ordering::Relaxed)
    }

    fn max_ctx(&self) -> Option<u64> {
        self.max_ctx
    }

    async fn resize(&self, n_ctx: u64) -> Result<(), LlmRuntimeError> {
        if let Some(cap) = self.max_ctx {
            if n_ctx > cap {
                return Err(LlmRuntimeError::Other(format!(
                    "context {} resize to {n_ctx} exceeds max_ctx {cap}",
                    self.name
                )));
            }
        }
        self.client
            .resize(&self.name, n_ctx)
            .await
            .map_err(map_instance_err)?;
        self.n_ctx.store(n_ctx, Ordering::Relaxed);
        Ok(())
    }

    fn pinned(&self) -> bool {
        self.pinned
    }

    fn resume(&self) -> bool {
        self.manager.resume_for(&self.name)
    }

    fn set_resume(&self, enabled: bool) {
        self.manager.set_resume(&self.name, enabled);
    }

    fn touch(&self) {
        self.manager.touch();
    }

    fn last_used(&self) -> i64 {
        self.last_used.load(Ordering::Relaxed)
    }

    fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    async fn destroy(&self) -> Result<(), LlmRuntimeError> {
        self.client
            .destroy(&self.name, false)
            .await
            .map_err(map_instance_err)
    }

    fn kv_cache(&self) -> Arc<dyn LlmKVCache> {
        Arc::new(LlamaKVCache {
            client: self.client.clone(),
            instance: self.name.clone(),
            model_key: self.manager.model_key().to_string(),
            store: None,
        })
    }

    async fn evict(&self) -> Result<(), LlmRuntimeError> {
        // Resume-marked contexts are KV-snapshotted BEFORE they drop (the
        // sidecar's `evict_context` order) so a later `snapshot=<name>-resume`
        // restores them. A failed save logs and the eviction still proceeds.
        if self.manager.resume_for(&self.name) {
            let snapshot = resume_snapshot_name(&self.name);
            if let Err(e) = self.client.save_snapshot(&self.name, &snapshot).await {
                tracing::warn!(
                    target: "router.instances",
                    model = %self.manager.model_key(),
                    instance = %self.name,
                    snapshot = %snapshot,
                    error = %e,
                    "resume snapshot save on evict failed - context drops unsnapshotted",
                );
            }
        }
        self.destroy().await
    }
}

/// The `LlmKVCache` adapter for one llama-fork instance: snapshot
/// save/list/delete round-trip through the fork's `/instances` API, with the
/// router-side `SnapshotStore` metadata index recorded when one is attached.
pub struct LlamaKVCache {
    client: InstanceClient,
    instance: String,
    model_key: String,
    store: Option<Arc<SnapshotStore>>,
}

#[async_trait::async_trait]
impl LlmKVCache for LlamaKVCache {
    async fn save(&self, name: &str) -> Result<(), LlmRuntimeError> {
        self.client
            .save_snapshot(&self.instance, name)
            .await
            .map_err(map_instance_err)?;
        if let Some(store) = &self.store {
            let _ = store.store(KvSnapshot {
                model: self.model_key.clone(),
                adapter: None,
                session_id: self.instance.clone(),
                snapshot_name: name.to_string(),
                instance: Some(self.instance.clone()),
                file_path: PathBuf::new(),
                token_count: None,
                created_at: common_core::now_secs(),
                last_used_at: common_core::now_secs(),
                llama_cpp_version: None,
                model_quant: None,
                base_model_hash: None,
                turn_seq: None,
            });
        }
        Ok(())
    }

    async fn restore(&self, name: &str) -> Result<(), LlmRuntimeError> {
        // The fork switches a snapshot into a slot at dispatch time via the
        // `snapshot` request field; there is no management-side "restore" —
        // restore is expressed by the caller sending `snapshot=<name>`. The
        // fork's `POST /instances/:instance/snapshot` only saves. A restore
        // therefore validates the snapshot exists (loud) and is otherwise a
        // no-op — the dispatch path owns the actual slot switch.
        let snapshots = self
            .client
            .list_snapshots(&self.instance)
            .await
            .map_err(map_instance_err)?;
        if snapshots.iter().any(|s| s.name == name) {
            Ok(())
        } else {
            Err(LlmRuntimeError::NotLoaded(format!(
                "snapshot {name} not found for instance {}",
                self.instance
            )))
        }
    }

    async fn list(&self) -> Vec<SnapshotMeta> {
        match self.client.list_snapshots(&self.instance).await {
            Ok(infos) => infos
                .into_iter()
                .map(|i| SnapshotMeta {
                    name: i.name,
                    size: i.size,
                    mtime: i.mtime.as_i64().unwrap_or(0),
                    n_ctx_seq: i.n_ctx_seq,
                })
                .collect(),
            Err(e) => {
                tracing::warn!(
                    target: "router.instances",
                    instance = %self.instance,
                    error = %e,
                    "snapshot list on fork failed - returning empty",
                );
                Vec::new()
            }
        }
    }

    async fn delete(&self, name: &str) -> Result<(), LlmRuntimeError> {
        self.client
            .delete_snapshot(&self.instance, name)
            .await
            .map_err(map_instance_err)
    }
}

/// The unified weights facade the server builds from the llama `InstancePool`
/// and the onnx registry. Llama rows stay **byte-identical** (delegated to
/// [`InstancePool::aggregate`]/`list_models`); onnx rows are appended by
/// rendering `dyn LlmWeights` residency rows. Also the surface `ps` and
/// `POST /models/unload` route both fleets through.
#[derive(Clone)]
pub struct LlmFleet {
    /// The llama-only pool (unchanged aggregation + dispatch path).
    pool: InstancePool,
    /// Every weights instance as a trait object (llama adapters + onnx
    /// implementors). Drives the shared residency engine (M5), the onnx rows
    /// of the facade, and unified unload.
    weights: Vec<Arc<dyn LlmWeights>>,
    /// The onnx registry (refusal checks + registry-key knowledge).
    onnx: Option<crate::ort::OrtRegistry>,
}

impl LlmFleet {
    /// Build the fleet from the llama pool, the supervisor (for llama adapters'
    /// `ensure_running` bin), and the onnx registry. Onnx implementors read
    /// their role config (instances block, defaults) from `config`.
    pub fn build(
        pool: InstancePool,
        supervisor: Option<&LlamaServerSupervisor>,
        onnx: Option<crate::ort::OrtRegistry>,
        config: &RouterConfig,
    ) -> Self {
        let mut weights: Vec<Arc<dyn LlmWeights>> = Vec::new();
        if let Some(sup) = supervisor {
            let bin = sup.bin().to_path_buf();
            for manager in pool.managers_iter() {
                if let Some(server) = sup.server_for(manager.model_key()) {
                    weights.push(Arc::new(LlamaWeights::new(
                        manager.clone(),
                        server,
                        bin.clone(),
                        config.sidecar.clone(),
                    )));
                }
            }
        }
        if let Some(onnx) = &onnx {
            weights.extend(crate::ort::onnx_weights_impls(onnx, config));
        }
        Self { pool, weights, onnx }
    }

    /// Whether the fleet is empty (no llama pool, no onnx weights).
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty() && self.weights.is_empty()
    }

    /// The llama-only pool (dispatch + management ops that stay llama-scoped).
    pub fn instance_pool(&self) -> &InstancePool {
        &self.pool
    }

    /// Every weights instance as a trait object (the shared engine's input).
    pub fn weights(&self) -> &[Arc<dyn LlmWeights>] {
        &self.weights
    }

    /// The onnx registry, when present.
    pub fn onnx(&self) -> Option<&crate::ort::OrtRegistry> {
        self.onnx.as_ref()
    }

    /// Whether a model key is known to the fleet (llama manager or a weights
    /// instance).
    pub fn is_known_model(&self, key: &str) -> bool {
        self.pool.manager(key).is_some() || self.weights.iter().any(|w| w.model_key() == key)
    }

    /// `GET /instances` — the aggregate envelope across every fleet (llama rows
    /// byte-identical to `InstancePool::aggregate`, onnx rows appended).
    pub async fn aggregate(&self, model: Option<&str>) -> Result<Value, InstanceError> {
        let llama = self.pool.aggregate(model).await?;
        let mut instances = llama["instances"].as_array().cloned().unwrap_or_default();
        let snapshots = llama["snapshots"].as_array().cloned().unwrap_or_default();
        let mut total: InstanceTotals =
            serde_json::from_value(llama["total"].clone()).unwrap_or_default();

        for w in &self.weights {
            if w.eviction_policy() != EvictionPolicy::LruLargest {
                continue; // llama rows already came through `pool.aggregate`
            }
            if let Some(filter) = model {
                if filter != w.model_key() {
                    continue;
                }
            }
            let rows = w.residency_rows().await;
            if rows.is_empty() {
                continue;
            }
            // The shared weights count once per loaded onnx instance.
            let model_bytes = if w.is_loaded() { w.weights_bytes() } else { 0 };
            total.model = total.model.saturating_add(model_bytes);
            total.total = total.total.saturating_add(model_bytes);
            for row in &rows {
                let mut entry = serde_json::to_value(row_to_info(w.model_key(), row))
                    .unwrap_or_default();
                if let Value::Object(ref mut obj) = entry {
                    obj.insert("runtime".into(), Value::String("onnx".into()));
                }
                instances.push(entry);
                total.context = total.context.saturating_add(row.context_bytes);
                total.compute = total.compute.saturating_add(row.compute_bytes);
                total.total = total.total.saturating_add(row.total_bytes);
            }
        }
        Ok(serde_json::json!({
            "instances": instances,
            "snapshots": snapshots,
            "total": total,
        }))
    }

    /// `GET /v1/models` — llama entries byte-identical to
    /// `InstancePool::list_models`, onnx entries appended (one per context,
    /// with the llama alias vocabulary).
    pub async fn list_models(&self) -> Vec<Value> {
        let mut out = self.pool.list_models().await;
        let created = common_core::now_secs();
        for w in &self.weights {
            if w.eviction_policy() != EvictionPolicy::LruLargest {
                continue;
            }
            let rows = w.residency_rows().await;
            if rows.is_empty() {
                continue;
            }
            for row in &rows {
                let name = row
                    .context_key
                    .strip_prefix(&format!("{}:", w.model_key()))
                    .unwrap_or(&row.context_key)
                    .to_string();
                let is_default = name == "default";
                let mut entry = serde_json::json!({
                    "id": row.context_key,
                    "object": "model",
                    "created": created,
                    "owned_by": "coral-router",
                    "n_ctx": row.n_ctx,
                    "parallel": row.parallel,
                    "pinned": row.pinned,
                    "resume": row.resume,
                    "is_default": is_default,
                    "state": row.state,
                    "last_used": row.last_used,
                    "runtime": "onnx",
                });
                entry["aliases"] = Value::Array(
                    instance_aliases(w.model_key(), &row.context_key, &row.group, is_default)
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                );
                out.push(entry);
            }
        }
        out
    }

    /// Unload a model through the trait surface (onnx `Unloadable` roles via
    /// their `OnnxWeights`; llama via the supervisor adapter). `Always`/pinned
    /// refusals surface as [`LlmRuntimeError::UnloadRefused`].
    pub async fn unload(&self, key: &str) -> Result<(), LlmRuntimeError> {
        if let Some(w) = self.weights.iter().find(|w| w.model_key() == key) {
            return w.unload().await;
        }
        Err(LlmRuntimeError::NotLoaded(format!("unknown model: {key}")))
    }

    /// Resize an onnx context through `LlmContext::resize` (ROADMAP M7 §3): the
    /// onnx half of `POST /instances/:name/resize`. Only onnx weights
    /// implementors (the `LruLargest` policy) serve named contexts in-process;
    /// llama contexts stay on the pool's `client.resize` path. An unknown model
    /// or a missing context is a loud error.
    pub async fn resize_context(
        &self,
        model: &str,
        name: &str,
        n_ctx: u64,
    ) -> Result<(), LlmRuntimeError> {
        let weights = self
            .weights
            .iter()
            .find(|w| w.model_key() == model && w.eviction_policy() == EvictionPolicy::LruLargest)
            .ok_or_else(|| {
                LlmRuntimeError::NotLoaded(format!("no onnx model or context: '{model}:{name}'"))
            })?;
        let ctx = weights.context(name).ok_or_else(|| {
            LlmRuntimeError::NotLoaded(format!("onnx context '{model}:{name}' not materialized"))
        })?;
        ctx.resize(n_ctx).await
    }

    /// Resolve the onnx instance id grammar `<model_id>:<name>` to `(model key,
    /// context name)`. `None` when the model is not an onnx weights key or the
    /// name breaks the instance-name grammar. Llama keys are handled by the
    /// llama pool's resolver; this is the onnx branch of the unified grammar.
    pub fn resolve_instance_id(&self, id: &str) -> Option<(String, String)> {
        let parsed = fluent_types::instance_id::InstanceId::parse(id).ok()?;
        let is_onnx = self.weights.iter().any(|w| {
            w.model_key() == &**parsed.model() && w.eviction_policy() == EvictionPolicy::LruLargest
        });
        is_onnx.then(|| ((**parsed.model()).to_string(), (**parsed.name()).to_string()))
    }
}
#[cfg(test)]
#[path = "../../tests/instances_traits.rs"]
mod tests;
