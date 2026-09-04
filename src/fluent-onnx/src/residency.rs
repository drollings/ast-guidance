//! `OrtResidencyLoop` — the onnx registry's residency engine (ROADMAP M0).
//!
//! The ort session registry gets the same unload-from-memory semantics the
//! llama supervisor has for VRAM, as a **sibling** of the llama sidecar task
//! (`InstancePool::run_residency`) — it operates on the `OrtSessionRegistry`
//! directly and reuses the llama residency *semantics* (idle release,
//! LRU-largest working-set eviction, pinned/Always never released), never a
//! fork of the sidecar's code.
//!
//! Two policies, per pass:
//!
//! 1. **Idle release** — every `Unloadable`, unpinned entry whose session is
//!    loaded and idle past its per-role `sleep_idle_seconds` (or the loop
//!    default) releases its handle, so a model that is not earning its memory
//!    is unloaded again.
//! 2. **Working-set eviction** — when Σ `resident_bytes` of loaded sessions
//!    exceeds the budget, release the LRU-largest `Unloadable` entry (ties →
//!    oldest `last_used`), stopping once back under budget or nothing is
//!    releasable. `Always`/pinned entries are never released.
//!
//! `start()` runs the loop on an injected `Runtime` — no ambient
//! `tokio::spawn` — so the router composes the shared primitive and tests
//! drive it deterministically. The loop is ort-free (it only reads the
//! registry), so it compiles and is hermetically testable without a model.

use std::sync::Arc;
use std::time::Duration;

use fluent_llm::runtime::{
    EvictionPolicy, LlmContext, LlmResidencyEngine, LlmResidencyRow, LlmRuntimeError, LlmWeights,
};
use fluent_wvr::runtime::Runtime;

use crate::error::OrtError;
use crate::session::OrtSessionRegistry;

/// The default idle threshold (seconds) after which an `Unloadable` session
/// may be released, when the entry's `OnnxRoleConfig.sleep_idle_seconds` is
/// absent or zero.
pub const DEFAULT_SLEEP_IDLE_SECONDS: i32 = 30;

/// The onnx residency loop (ROADMAP M5): a **delegating compatibility shim**
/// over the shared [`LlmResidencyEngine`]. It presents the same constructor and
/// `residency_cycle`/`start` surface it always did, but every pass is driven by
/// the one shared engine over per-registry `LlmWeights` implementors — the
/// engine now owns the idle-release + LRU-largest working-set logic the loop
/// used to inline. Its own tests pass unchanged; the router's `serve` no longer
/// spawns this type (it spawns the shared engine over the fleet instead).
pub struct OrtResidencyLoop {
    engine: Arc<LlmResidencyEngine>,
    weights: Vec<Arc<dyn LlmWeights>>,
}

/// A `LlmWeights` implementor over one `OrtSessionRegistry` entry — the shim's
/// bridge into the shared engine. Non-generative entries render no context rows
/// (today's invisibility); their resident footprint is the whole-weights
/// eviction target, exactly as the pre-M5 loop treated them.
struct OnnxRegistryWeights {
    registry: Arc<OrtSessionRegistry>,
    key: String,
}

/// Map an ort registry error onto the runtime-agnostic error type.
fn map_err(e: &OrtError) -> LlmRuntimeError {
    let s = e.to_string();
    if s.contains("refuses unload") {
        LlmRuntimeError::UnloadRefused(s)
    } else {
        LlmRuntimeError::Other(s)
    }
}

#[async_trait::async_trait]
impl LlmWeights for OnnxRegistryWeights {
    fn model_key(&self) -> &str {
        &self.key
    }

    fn weights_bytes(&self) -> u64 {
        self.registry.resident_bytes(&self.key).unwrap_or(0)
    }

    fn pinned(&self) -> bool {
        self.registry.is_pinned(&self.key)
    }

    fn refuse_unload(&self) -> bool {
        // `Always` residency OR pinned — the exact pre-M5 rule
        // (`policy.is_always() || pinned` never released/evicted).
        self.registry.refuses_unload(&self.key) || self.registry.is_pinned(&self.key)
    }

    fn is_loaded(&self) -> bool {
        self.registry
            .residency_report()
            .iter()
            .any(|r| r.key == self.key && r.loaded)
    }

    fn sleep_idle_seconds(&self) -> Option<i32> {
        self.registry.sleep_idle_seconds(&self.key)
    }

    async fn ensure_loaded(&self) -> Result<(), LlmRuntimeError> {
        self.registry.ensure_loaded(&self.key).map_err(|e| map_err(&e))?;
        Ok(())
    }

    async fn unload(&self) -> Result<(), LlmRuntimeError> {
        self.registry.release(&self.key).map_err(|e| map_err(&e))?;
        Ok(())
    }

    fn touch(&self) {
        self.registry.touch(&self.key);
    }

    fn last_used(&self) -> i64 {
        self.registry.last_used_of(&self.key).unwrap_or(0)
    }

    async fn residency_rows(&self) -> Vec<LlmResidencyRow> {
        Vec::new()
    }

    fn context(&self, _name: &str) -> Option<Arc<dyn LlmContext>> {
        None
    }

    async fn ensure_context(
        &self,
        _name: &str,
    ) -> Result<Arc<dyn LlmContext>, LlmRuntimeError> {
        Err(LlmRuntimeError::NotLoaded(format!(
            "onnx registry entry {} serves no named contexts",
            self.key
        )))
    }

    fn eviction_policy(&self) -> EvictionPolicy {
        EvictionPolicy::LruLargest
    }
}

impl OrtResidencyLoop {
    /// Build the loop. `working_set_budget_bytes = None` → idle-only eviction
    /// (CPU RAM is cheap; the parity target is idle unload).
    pub fn new(
        registry: Arc<OrtSessionRegistry>,
        poll_interval: Duration,
        idle_seconds_default: i32,
        working_set_budget_bytes: Option<u64>,
    ) -> Arc<Self> {
        Self::new_with_clock(
            registry,
            poll_interval,
            idle_seconds_default,
            working_set_budget_bytes,
            Arc::new(now_unix_ms),
        )
    }

    /// The deterministic-clock constructor (tests, and the shim's own
    /// deterministic seam — `new` is the production entry). The by-value
    /// `Arc<OrtSessionRegistry>` is the public API contract (kept stable so
    /// the shim's own tests pass unchanged); it is only cloned into the
    /// per-entry weights.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_with_clock(
        registry: Arc<OrtSessionRegistry>,
        poll_interval: Duration,
        idle_seconds_default: i32,
        working_set_budget_bytes: Option<u64>,
        clock: Arc<dyn Fn() -> i64 + Send + Sync>,
    ) -> Arc<Self> {
        let weights: Vec<Arc<dyn LlmWeights>> = registry
            .model_keys()
            .into_iter()
            .map(|key| {
                Arc::new(OnnxRegistryWeights {
                    registry: Arc::clone(&registry),
                    key,
                }) as Arc<dyn LlmWeights>
            })
            .collect();
        // VRAM budget is `None` (onnx lives in CPU RAM). The working-set budget
        // feeds the engine's RAM pool; the onnx loop released until back under
        // budget (no batch cap), so the shim passes an unbounded batch.
        let engine = LlmResidencyEngine::new_with_clock(
            poll_interval,
            None,
            working_set_budget_bytes,
            idle_seconds_default,
            usize::MAX,
            clock,
        );
        Arc::new(Self { engine, weights })
    }

    /// Run the loop as a background task on the injected `Runtime`: one
    /// residency pass per `poll_interval`, forever. The caller owns the
    /// returned handle (the server drains it on graceful shutdown). Delegates
    /// to the shared engine's loop.
    pub fn start(self: &Arc<Self>, runtime: &Arc<dyn Runtime>) -> tokio::task::JoinHandle<()> {
        self.engine.start(runtime, Arc::new(self.weights.clone()))
    }

    /// One residency pass: idle release, then working-set eviction if over
    /// budget. Best-effort — a failed release is logged and the pass moves on
    /// (the loop never panics the process). Delegates to the shared engine
    /// (its onnx operations are synchronous registry calls, so the cycle is
    /// driven to completion inline).
    pub fn residency_cycle(&self) {
        let weights = self.weights.clone();
        let _ = block_on(self.engine.residency_cycle(&weights));
    }
}

/// Drive an async engine pass to completion. The onnx cycle's operations are
/// synchronous registry calls (they never actually await I/O), so a plain
/// `block_on` on the current thread — or a throwaway current-thread runtime
/// outside one — is sufficient and exact.
fn block_on<T>(fut: impl std::future::Future<Output = T>) -> T {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(fut)
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("onnx residency current-thread runtime");
        rt.block_on(fut)
    }
}

/// Current system time as unix-milliseconds.
fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

#[cfg(test)]
#[path = "../tests/residency.rs"]
mod tests;
