//! The router's aggregate `/instances` facade over every managed model's
//! server, and the device-wide VRAM residency engine that keeps the managed
//! fleet within its allocation budget.
//!
//! This file owns the residency background loop: it aggregates every manager's
//! `/instances` into a device `used` total, compares it to the budget
//! (`device_total - minimum_remaining_vram`), evicts LRU-largest unpinned
//! contexts (snapshotting resume-marked ones first), and unloads models left
//! with zero contexts. The `Evictable` union (one unpinned context vs a whole
//! model) is what makes the largest resident footprints — e.g. a 10.5 GB
//! weight pool — real eviction targets.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use common_core::registry::ConcurrentRegistry;

use super::client::{InstanceError, InstanceInfo};
use super::manager::{
    instance_name_from_server_id, resume_snapshot_name, InstanceManager,
};
use super::management_base_url;

/// One unit the residency/admission control can evict to free VRAM.
///
/// A unit is either a single unpinned context (frees its KV + compute; the
/// model's weights stay) or a whole model with no pinned instances (frees its
/// weights and every context). Including whole-model units is what makes the
/// largest resident footprints - e.g. a 10.5 GB weight pool - real eviction
/// targets instead of only the small per-context buffers.
#[derive(Clone)]
enum Evictable {
    /// One unpinned context.
    Context {
        info: InstanceInfo,
        manager: Arc<InstanceManager>,
    },
    /// A whole model: every unpinned context, then the shared weights.
    Model {
        manager: Arc<InstanceManager>,
        /// The coldest context's last use (model recency).
        last_used: i64,
        /// Total VRAM freed: weights + all unpinned contexts.
        freed_bytes: u64,
        /// The unpinned contexts to drop first (resume ones are snapshotted).
        contexts: Vec<InstanceInfo>,
    },
}

impl Evictable {
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
    pub(super) managers: ConcurrentRegistry<String, Arc<InstanceManager>>,
    /// management base URL -> model key (for dispatch-time manager lookup).
    pub(super) by_base: ConcurrentRegistry<String, String>,
    /// The llama-server supervisor, used to load lazy models on demand and to
    /// unload a model whose last context was evicted (freeing its weights).
    pub(super) supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>,
    /// Sidecar residency policy (device budget, poll interval, evict batch).
    pub(super) policy: crate::config::SidecarConfig,
}

impl InstancePool {
    /// Build a pool from an existing manager set, indexing each manager by its
    /// client's management base URL for dispatch-time lookup.
    pub fn from_managers(
        managers: HashMap<String, Arc<InstanceManager>>,
        supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>,
    ) -> Self {
        let reg = ConcurrentRegistry::new();
        let by_base = ConcurrentRegistry::new();
        for (key, manager) in managers {
            by_base.insert(management_base_url(manager.client().base_url()), key.clone());
            reg.insert(key, manager);
        }
        // The policy is shared across managers (it is cloned from the config
        // into each); take the first manager's as the pool-wide residency
        // policy. An empty pool has no policy needs.
        let policy = reg
            .keys()
            .first()
            .and_then(|k| reg.get(k).map(|m| m.policy.clone()))
            .unwrap_or_default();
        Self {
            managers: reg,
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
    pub fn manager(&self, model_key: &str) -> Option<Arc<InstanceManager>> {
        self.managers
            .get(&model_key.to_string())
            .map(|m| m.as_ref().clone())
    }

    /// The manager for a dispatch endpoint URL: strips the path to the
    /// management base and matches a managed server. Used by the
    /// allocate-on-503 path.
    pub fn manager_for_url(&self, endpoint_url: &str) -> Option<Arc<InstanceManager>> {
        let base = management_base_url(endpoint_url);
        let key = self.by_base.get(&base)?;
        self.managers.get(&key).map(|m| m.as_ref().clone())
    }

    /// The supervisor (present when any model is managed).
    pub fn supervisor(&self) -> Option<&Arc<crate::supervisor::LlamaServerSupervisor>> {
        self.supervisor.as_ref()
    }

    /// Iterate the managers (the server spawns each manager's sidecar task).
    pub fn managers_iter(&self) -> Vec<Arc<InstanceManager>> {
        self.managers
            .keys()
            .into_iter()
            .filter_map(|k| self.managers.get(&k).map(|a| a.as_ref().clone()))
            .collect()
    }

    /// A stable, owned snapshot of the managers for a residency pass. The
    /// returned `Vec` lives as long as the caller holds it, so per-manager
    /// references (e.g. the `&Arc<InstanceManager>` inside `Evictable`) are
    /// valid for the duration of the pass.
    fn managers_snapshot(&self) -> Vec<Arc<InstanceManager>> {
        self.managers_iter()
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

    /// ROADMAP M7 resize-to-demand: when the dispatch path targets a named
    /// context and the caller's declared context need (`num_ctx`) exceeds the
    /// profile's allocated `n_ctx` but stays under its `max_ctx`, resize the
    /// instance via the fork's `client.resize` before dispatching. A need
    /// beyond `max_ctx` is a loud error (no new fail-open) — the same shape a
    /// too-large llama request gets today. No-op for unmanaged / unnamed
    /// targets (best-effort, like `ensure_target_ready`).
    pub async fn resize_to_demand(
        &self,
        endpoint_url: &str,
        instance: Option<&str>,
        num_ctx: Option<u64>,
    ) -> Result<(), InstanceError> {
        let Some(need) = num_ctx else {
            return Ok(());
        };
        let Some(instance) = instance else {
            return Ok(());
        };
        let Some(manager) = self.manager_for_url(endpoint_url) else {
            return Ok(());
        };
        // The targeted context's profile: the allocated `n_ctx` and the cap.
        let Some(profile) = manager
            .profiles()
            .iter()
            .find(|p| p.name.as_deref() == Some(instance))
        else {
            // No configured profile for the instance — nothing to size.
            return Ok(());
        };
        if need <= profile.num_ctx {
            return Ok(());
        }
        if let Some(cap) = profile.max_ctx {
            if need > cap {
                return Err(InstanceError::Rejected {
                    status: 400,
                    body: format!(
                        "instance '{instance}' resize to {need} exceeds max_ctx {cap}"
                    ),
                });
            }
        }
        tracing::info!(
            target: "router.instances",
            model = %manager.model_key(),
            instance = %instance,
            allocated = profile.num_ctx,
            requested = need,
            max_ctx = ?profile.max_ctx,
            "resizing instance to demand",
        );
        manager.client().resize(instance, need).await
    }

    /// One device-wide residency pass. The pool owns VRAM residency for the
    /// whole device (all managed servers share it), so this aggregates every
    /// manager's `/instances` into a device `used` total and compares it to
    /// the allocation budget (`device_total - minimum_remaining_vram`).
    ///
    /// When the budget is exceeded, evicts up to `evict_batch` units - the
    /// largest resident footprint first (see `Self::evict_to_fit`) - and
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
    ) -> (u64, Vec<Evictable>) {
        let mut used: u64 = 0;
        let mut evictable: Vec<Evictable> = Vec::new();
        let managers = self.managers_snapshot();
        for manager in &managers {
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
                            manager: Arc::clone(manager),
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
                        manager: Arc::clone(manager),
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
                        manager: Arc::clone(manager),
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
    async fn evict_to_fit(&self, used: &mut u64, budget: u64, evictable: Vec<Evictable>) {
        let now = i64::try_from(common_core::now_secs()).unwrap_or(i64::MAX);
        // Order best-eviction-first (footprint × coldness desc, then last-used
        // desc), then evict until under budget or the batch is reached. The
        // engine (ordering + budget loop) lives in `common_core::cache`.
        let ordered = common_core::cache::eviction_order(
            evictable,
            now,
            Evictable::freed_bytes,
            Evictable::last_used,
        );
        let me = InstancePool::clone(self);
        let (used_after, _) = common_core::cache::evict_until_fit(
            *used,
            budget,
            self.policy.evict_batch,
            ordered,
            |unit| {
                let me = InstancePool::clone(&me);
                let unit = unit.clone();
                async move {
                    match unit {
                        Evictable::Context { info, manager } => {
                            me.evict_context(&manager, &info, "over_budget").await
                        }
                        Evictable::Model {
                            manager,
                            contexts,
                            freed_bytes,
                            ..
                        } => me.evict_model(&manager, &contexts, freed_bytes).await,
                    }
                }
            },
        )
        .await;
        *used = used_after;
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
        let managers = self.managers_snapshot();
        for manager in &managers {
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
        let mut keys: Vec<String> = self.managers.keys();
        keys.sort();
        for key in keys {
            let Some(manager) = self.managers.get(&key).map(|m| m.as_ref().clone()) else {
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
            tokio::time::sleep(Duration::from_millis(common_core::retry::capped_backoff_ms(
                base.as_millis() as u64,
                consecutive_failures,
                12,
            )))
            .await;
        }
    }
}
