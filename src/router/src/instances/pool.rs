//! The router's aggregate `/instances` facade over every managed model's
//! server, plus dispatch-time load readiness (on-demand loads, admission,
//! resize-to-demand).
//!
//! Residency itself lives in the shared `fluent_llm::runtime::
//! LlmResidencyEngine`: this pool keeps transport duties (detecting a cold
//! load, excluding the target, naming the admission hook) while the engine
//! owns eviction ordering, idle release, resume expiry, and the empty-model
//! unload over the fleet's `LlmWeights` adapters.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use common_core::registry::ConcurrentRegistry;
use fluent_llm::runtime::{LlmResidencyEngine, MemoryPool};

use super::client::InstanceError;
use super::manager::InstanceManager;
use super::management_base_url;

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
    /// The shared residency engine for load-time admission. Wired once at
    /// boot (the engine is built before the pool); `None` only for pools
    /// driven standalone in tests without an engine.
    admission: Arc<Mutex<Option<Arc<LlmResidencyEngine>>>>,
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
            admission: Arc::new(Mutex::new(None)),
        }
    }

    /// Wire the shared residency engine for load-time admission
    /// (`make_room_for`). Called once at boot after the engine is built; the
    /// pool keeps transport duties (detecting the cold load, excluding the
    /// target) while the engine owns the eviction ordering.
    pub fn set_admission_engine(&self, engine: Arc<LlmResidencyEngine>) {
        if let Ok(mut slot) = self.admission.lock() {
            *slot = Some(engine);
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
        if !manager.has_instances_api() {
            // Grammar-less adopted server: a single fixed context, nothing
            // to resize through the (absent) management API.
            return Ok(());
        }
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

    /// Load-time admission control: before a cold model spawns (requiring
    /// `required_bytes` of VRAM for its weights), evict units until the
    /// projected device usage fits the allocation budget. The target model is
    /// never an eviction candidate and pinned instances are never evicted.
    /// Best-effort: if eviction cannot fully make room, the load proceeds and
    /// the residency loop corrects the overshoot.
    ///
    /// Transport only: the pool detects the cold load and names the target,
    /// then hands the pass to the shared engine's pre-dispatch hook over the
    /// same `LlamaWeights` adapters the fleet build uses.
    pub async fn make_room_for(&self, model_key: &str, required_bytes: u64) {
        let engine = self.admission.lock().ok().and_then(|g| g.clone());
        let Some(engine) = engine else {
            tracing::debug!(
                target: "router.instances",
                model = %model_key,
                "admission skipped - no residency engine wired",
            );
            return;
        };
        let Some(supervisor) = &self.supervisor else {
            return;
        };
        let weights =
            super::traits::llama_weights_for_pool(self, supervisor, &self.policy);
        engine
            .make_room_for(&weights, model_key, required_bytes, MemoryPool::Vram)
            .await;
    }

}
