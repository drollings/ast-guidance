//! `fluent-llm::runtime` — the shared, runtime-agnostic LLM lifecycle +
//! residency control plane (ROADMAP_20260830_LLMS M1).
//!
//! This module is the single home of the "one loaded weights instance serving
//! N named context windows, each with its own KV cache" abstraction. It is a
//! **scaffold with zero production callers in M1** (per fluent-wvr, scaffold /
//! compatibility-surface traits are intentional and must not be pruned as dead
//! code): the llama fleet (`fluent-router`'s `instances/*` + `supervisor.rs` +
//! `kv_cache.rs`) and the in-process ONNX fleet (`fluent-onnx`'s `session.rs` +
//! `residency.rs` + `llm.rs`) both gain thin adapters over these traits in M4,
//! and the shared residency engine replaces both sibling loops in M5.
//!
//! Design notes (load-bearing):
//!
//! - **`dyn` at the control plane, concrete in the hot loop** (fluent-wvr §0):
//!   these traits are used only by the residency engine, the `/instances`
//!   facade, `ps`, and the context registry — never inside the token-by-token
//!   decode loop (`OrtLlmSession::generate` stays fully concrete).
//! - **`LlmRuntime` is data, not a vtable discriminator.** `ps` renders a
//!   `runtime` field from `LlmResidencyRow.runtime`; no code branches on it to
//!   dispatch.
//! - **Per-weights eviction policy.** The two eviction orderings
//!   (llama footprint×coldness vs onnx LRU-largest) are **not** unified
//!   (`ROADMAP_20260830_PRIMITIVES` §"What is NOT being extracted").
//!   `EvictionPolicy` is an injected ordering chosen by each `LlmWeights`
//!   implementor so the engine's single loop preserves each fleet's exact
//!   semantics.
//! - **Two budgets, one loop.** VRAM (llama) and CPU RAM (onnx) are different
//!   memory pools; the engine carries both budgets and checks the over-budget
//!   condition per-pool inside one pass, preserving today's exact thresholds.
//! - **`last_used` is pool-native.** The llama fork reports last use in
//!   **seconds**; the onnx registry in **milliseconds**. A `LlmResidencyRow`
//!   keeps the runtime's native unit so the engine's ordering and the onnx
//!   idle-release rule (`idle_ms >= idle_seconds * 1000`) stay exact with zero
//!   unit conversion.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use fluent_wvr::runtime::Runtime;

/// The runtime that owns a weights instance: a spawned `llama-server` process
/// or an in-process ort session. Data for `ps` / envelope rendering — never a
/// vtable discriminator (no code branches on it to dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmRuntime {
    /// A supervised `llama-server` process (VRAM-resident).
    Llama,
    /// An in-process ONNX / `ort` session (CPU-RAM-resident).
    Onnx,
}

/// The eviction ordering a weights instance injects into the shared engine.
/// The two fleets' orderings are deliberately distinct and must never be
/// unified (roadmap §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionPolicy {
    /// llama: footprint × coldness (`eviction_score`), newest `last_used`
    /// kept on tie. Ordered by `common_core::cache::eviction_order`.
    FootprintColdness,
    /// onnx: largest footprint first, oldest `last_used` first on tie (the
    /// exact `sort_by` in `fluent-onnx/src/residency.rs`).
    LruLargest,
}

/// The memory pool an eviction budget applies to. VRAM is the llama fleet's
/// pool; CPU RAM is the onnx fleet's. The engine carries both budgets and runs
/// one per-pool over-budget check per pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryPool {
    /// Device VRAM — the `FootprintColdness` pool.
    Vram,
    /// CPU RAM — the `LruLargest` pool.
    Ram,
}

impl EvictionPolicy {
    /// The memory pool this policy's budget comes from.
    #[must_use]
    pub fn pool(self) -> MemoryPool {
        match self {
            Self::FootprintColdness => MemoryPool::Vram,
            Self::LruLargest => MemoryPool::Ram,
        }
    }
}

/// Runtime-agnostic error for weights/context/KV lifecycle operations. The
/// variants mirror `InstanceError` (llama management) + `OrtError` (onnx),
/// suffixed where the source kept them distinct.
#[derive(Debug, thiserror::Error)]
pub enum LlmRuntimeError {
    /// 429/503/504/507/other 5xx — transient; a 507/503 also signals an
    /// allocation/eviction trigger.
    #[error("transient runtime error: {status} {body}")]
    Transient { status: u16, body: String },
    /// 409 duplicate name — tolerated during reconciliation.
    #[error("duplicate resource (409)")]
    Duplicate,
    /// Permanent 4xx (except 409) — no retry.
    #[error("runtime request rejected: {status} {body}")]
    Rejected { status: u16, body: String },
    /// Transport / network failure before an HTTP status was received.
    #[error("runtime network error: {0}")]
    Network(String),
    /// A 2xx whose payload did not match the expected shape.
    #[error("runtime response parse error: {0}")]
    Parse(String),
    /// The requested resource is not loaded / registered.
    #[error("runtime resource not loaded: {0}")]
    NotLoaded(String),
    /// An `Always` / pinned resource refuses to unload.
    #[error("runtime unload refused: {0}")]
    UnloadRefused(String),
    /// Anything else.
    #[error("runtime operation failed: {0}")]
    Other(String),
}

/// One row of a weights instance's residency view — the trait-object form of
/// `InstanceInfo` (llama) / `ResidencyReportEntry` + pool context (onnx).
///
/// `last_used` is **pool-native**: seconds for `runtime: Llama` (the fork's
/// `/instances` clock), milliseconds for `runtime: Onnx` (the registry's
/// `last_used_ms` clock). The engine keeps each fleet's unit so its ordering
/// and idle-release rules stay exact with zero conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResidencyRow {
    /// Public `<model_key>:<context_name>` identifier.
    pub context_key: String,
    /// The context's group.
    pub group: String,
    /// Context size in tokens.
    pub n_ctx: u64,
    /// Slots per context.
    pub parallel: u32,
    /// Whether the context is pinned (never evicted).
    pub pinned: bool,
    /// Whether the context is resume-marked (KV snapshotted before eviction).
    pub resume: bool,
    /// `"loaded"` | `"sleeping"` | `"unloaded"`.
    pub state: String,
    /// The owning runtime (drives `ps`; never a type check).
    pub runtime: LlmRuntime,
    /// Shared weights bytes (reported once per loaded instance).
    pub model_bytes: u64,
    /// Context (KV) bytes.
    pub context_bytes: u64,
    /// Compute-buffer bytes.
    pub compute_bytes: u64,
    /// Context + compute bytes (excludes the shared weights).
    pub total_bytes: u64,
    /// Context + compute bytes for the VRAM pool; 0 for RAM-resident onnx
    /// contexts.
    pub vram_bytes: u64,
    /// Pool-native last use (see the field note above). Negative = never used
    /// (llama); 0 = never used (onnx).
    pub last_used: i64,
}

/// Metadata for one context KV snapshot — the trait-object form of
/// `SnapshotInfo` (llama) / the RAM snapshot index entry (onnx).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotMeta {
    /// Snapshot name (e.g. `<instance>-resume`).
    pub name: String,
    /// Snapshot size in bytes (0 when unknown).
    pub size: u64,
    /// Last-modification time (unix seconds; 0 when unknown).
    pub mtime: i64,
    /// Sequence length the snapshot was taken at.
    pub n_ctx_seq: u64,
}

/// One loaded weights instance — a llama-server process or an ort session.
/// The engine, the `/instances` facade, `ps`, and the context registry drive
/// fleets through this surface; the hot decode loop never sees it.
#[async_trait::async_trait]
pub trait LlmWeights: Send + Sync {
    /// The public model key (e.g. `"onnx/llm"` or `"lfm2.5-2.6b"`).
    fn model_key(&self) -> &str;
    /// Resident weights footprint in bytes (0 when unloaded).
    fn weights_bytes(&self) -> u64;
    /// Whether this weights instance is pinned (never released/evicted).
    fn pinned(&self) -> bool;
    /// Whether this weights instance refuses unload (`Always` residency on
    /// the onnx side; a llama model with pinned instances on the llama side).
    fn refuse_unload(&self) -> bool;
    /// Whether the weights are currently loaded/resident.
    fn is_loaded(&self) -> bool;
    /// Per-weights idle threshold (seconds) after which the engine's
    /// idle-release pass may unload it. `None`/`<= 0` inherits the engine
    /// default. Only meaningful for the `LruLargest` pool (onnx) — llama
    /// weights are never router-side idle-released today.
    fn sleep_idle_seconds(&self) -> Option<i32>;
    /// Load the weights on demand (spawn + health-wait, or session load).
    async fn ensure_loaded(&self) -> Result<(), LlmRuntimeError>;
    /// Unload the weights, freeing them from memory. Best-effort; a refusal
    /// (pinned/`Always`) is an error.
    async fn unload(&self) -> Result<(), LlmRuntimeError>;
    /// Record dispatch recency (used by the residency ordering when the
    /// weights' last use is not otherwise reported).
    fn touch(&self);
    /// Pool-native last use of the weights instance itself (seconds for llama,
    /// ms for onnx). Negative/0 = never used. The idle-release pass uses this
    /// (falling back to the most recent context row's `last_used`) so a
    /// weights instance with no resident contexts still reports recency.
    fn last_used(&self) -> i64;
    /// The current residency view: one row per live context (a plain llama
    /// model reports one synthesized row; an onnx session with no contexts
    /// reports none).
    async fn residency_rows(&self) -> Vec<LlmResidencyRow>;
    /// The named context handle, if it is currently materialized.
    fn context(&self, name: &str) -> Option<Arc<dyn LlmContext>>;
    /// Ensure the named context exists (create on demand), returning it.
    async fn ensure_context(&self, name: &str) -> Result<Arc<dyn LlmContext>, LlmRuntimeError>;
    /// The eviction ordering this weights instance injects into the engine.
    fn eviction_policy(&self) -> EvictionPolicy;
    /// Engine hook: expire resume-marked contexts idle past the resume TTL.
    /// Default no-op; `LlamaWeights` overrides with the existing `expire_resume`
    /// pass (idle past `resume_ttl_s` → clear flag + delete snapshot).
    async fn expire_resume(&self) {}
}

/// One named context window with its own KV cache, allocated from a weights
/// instance. The onnx side gains this in M2 (the defining capability of the
/// llama fork) as an additive layer.
#[async_trait::async_trait]
pub trait LlmContext: Send + Sync {
    /// The bare context name (e.g. `"scratch"`, `"swarm-0"`).
    fn name(&self) -> &str;
    /// The context's group.
    fn group(&self) -> &str;
    /// The allocated context size in tokens.
    fn n_ctx(&self) -> u64;
    /// The context-size cap (`None` = inherit / no cap). `resize` refuses
    /// growth past it.
    fn max_ctx(&self) -> Option<u64>;
    /// Resize the context window (destroy + re-create on the llama fork;
    /// in-place on onnx). Bounded by `max_ctx`.
    async fn resize(&self, n_ctx: u64) -> Result<(), LlmRuntimeError>;
    /// Whether this context is pinned (never evicted).
    fn pinned(&self) -> bool;
    /// Whether this context is resume-marked (KV preserved across eviction).
    fn resume(&self) -> bool;
    /// Set/clear the resume flag.
    fn set_resume(&self, enabled: bool);
    /// Record dispatch recency.
    fn touch(&self);
    /// Pool-native last use (seconds for llama, ms for onnx).
    fn last_used(&self) -> i64;
    /// The context's VRAM footprint (0 for RAM-resident onnx contexts).
    fn vram_bytes(&self) -> u64;
    /// Destroy the context, freeing its KV + compute.
    async fn destroy(&self) -> Result<(), LlmRuntimeError>;
    /// The KV-cache surface for this context (snapshot/restore/list/delete).
    fn kv_cache(&self) -> Arc<dyn LlmKVCache>;
    /// Evict this context. Default = `destroy`; the llama adapter overrides to
    /// run the resume-snapshot-before-destroy order first (M5).
    async fn evict(&self) -> Result<(), LlmRuntimeError> {
        self.destroy().await
    }
}

/// Snapshot/restore/list/delete of one context's KV state. Llama delegates to
/// the fork's `POST /instances/:name/snapshot` + the `SnapshotStore` metadata
/// index; onnx clones the context's `PastState` tensors to a named RAM map.
#[async_trait::async_trait]
pub trait LlmKVCache: Send + Sync {
    /// Save the context's current KV under `name`.
    async fn save(&self, name: &str) -> Result<(), LlmRuntimeError>;
    /// Restore the context's KV from `name`.
    async fn restore(&self, name: &str) -> Result<(), LlmRuntimeError>;
    /// List the context's snapshots.
    async fn list(&self) -> Vec<SnapshotMeta>;
    /// Delete the snapshot `name`.
    async fn delete(&self, name: &str) -> Result<(), LlmRuntimeError>;
}

/// One unit the residency engine can evict to free memory.
///
/// A unit is either a single unpinned context (frees its KV + compute; the
/// weights stay) or a whole weights instance with no pinned contexts (frees
/// its weights *and* every context). Including whole-weights units is what
/// makes the largest resident footprints real eviction targets.
struct Candidate {
    kind: CandidateKind,
    /// Bytes freeing this candidate releases.
    freed_bytes: u64,
    /// Pool-native last use (ordering key).
    last_used: i64,
}

enum CandidateKind {
    /// One unpinned context.
    Context { weights: Arc<dyn LlmWeights>, name: String },
    /// A whole weights instance (every unpinned context, then the weights).
    Weights { weights: Arc<dyn LlmWeights> },
}

/// Current system time as unix-milliseconds.
fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// The shared residency engine: one pass over `&[Arc<dyn LlmWeights>]` with
/// **injected per-weights policies** (budget, eviction ordering, idle rule).
///
/// The two existing loop structures (llama `InstancePool::residency_cycle` and
/// onnx `OrtResidencyLoop::residency_cycle`) share this *pass structure*:
///
/// 1. per-weights resume expiry (`expire_resume` hook; llama overrides);
/// 2. idle-release (the onnx rule in `fluent-onnx/src/residency.rs:124-162`);
/// 3. per-pool over-budget check + `evict_until_fit` (VRAM for
///    `FootprintColdness`, RAM for `LruLargest`);
/// 4. unload weights left with zero contexts (the llama rule — llama-only).
///
/// The two eviction orderings stay distinct (`EvictionPolicy`), and the two
/// budgets stay distinct, so each fleet's exact semantics are preserved.
pub struct LlmResidencyEngine {
    /// The poll cadence the `start()` loop sleeps between passes.
    poll_interval: Duration,
    /// VRAM allocation budget (llama pool). `None` → no VRAM eviction.
    vram_budget_bytes: Option<u64>,
    /// CPU-RAM working-set budget (onnx pool). `None` → idle-only eviction.
    ram_budget_bytes: Option<u64>,
    /// Default idle threshold (seconds) for `LruLargest` weights whose own
    /// `sleep_idle_seconds` is absent or `<= 0`.
    idle_seconds_default: i32,
    /// Max units evicted per over-budget pass.
    evict_batch: usize,
    /// Unix-ms clock (injectable for deterministic tests).
    clock: Arc<dyn Fn() -> i64 + Send + Sync>,
}

impl LlmResidencyEngine {
    /// Build the engine. `vram_budget_bytes = None` disables VRAM eviction
    /// (the llama sidecar's "no allocation budget" mode); `ram_budget_bytes =
    /// None` gives idle-only onnx eviction (today's default).
    pub fn new(
        poll_interval: Duration,
        vram_budget_bytes: Option<u64>,
        ram_budget_bytes: Option<u64>,
        idle_seconds_default: i32,
        evict_batch: usize,
    ) -> Arc<Self> {
        Self::new_with_clock(
            poll_interval,
            vram_budget_bytes,
            ram_budget_bytes,
            idle_seconds_default,
            evict_batch,
            Arc::new(now_unix_ms),
        )
    }

    /// The deterministic-clock constructor (tests, and the onnx residency
    /// shim's own deterministic-clock seam — `new` is the production entry).
    pub fn new_with_clock(
        poll_interval: Duration,
        vram_budget_bytes: Option<u64>,
        ram_budget_bytes: Option<u64>,
        idle_seconds_default: i32,
        evict_batch: usize,
        clock: Arc<dyn Fn() -> i64 + Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(Self {
            poll_interval,
            vram_budget_bytes,
            ram_budget_bytes,
            idle_seconds_default,
            evict_batch,
            clock,
        })
    }

    /// One residency pass over every weights instance. Best-effort: a failed
    /// release/unload is logged and the pass moves on (never a panic).
    pub async fn residency_cycle(&self, weights: &[Arc<dyn LlmWeights>]) -> Result<(), LlmRuntimeError> {
        // 1. Per-weights resume expiry (the llama `expire_resume` pass; onnx is
        //    a no-op until it gains resume semantics).
        for w in weights {
            w.expire_resume().await;
        }

        // 2. Idle-release pass — the onnx rule. Only the `LruLargest` (onnx)
        //    pool participates; llama weights are never router-side
        //    idle-released today (the fork owns their idle sleep).
        self.release_idle(weights).await;

        // 3. Per-pool over-budget eviction, one pass each (two budgets, one
        //    loop).
        let now = (self.clock)();
        self.evict_over_budget(weights, now, MemoryPool::Vram, self.vram_budget_bytes)
            .await;
        self.evict_over_budget(weights, now, MemoryPool::Ram, self.ram_budget_bytes)
            .await;

        // 4. Unload weights left with zero contexts (the llama rule — the
        //    onnx pool's weights are released by idle/budget, never by
        //    zero-context).
        self.unload_empty(weights).await;
        Ok(())
    }

    /// Release every loaded `LruLargest` (onnx) weights instance idle past its
    /// `sleep_idle_seconds` (or the engine default). `refuse_unload` (pinned /
    /// `Always`) entries are never released. Returns the count released.
    async fn release_idle(&self, weights: &[Arc<dyn LlmWeights>]) -> usize {
        let now = (self.clock)();
        let mut released = 0usize;
        for w in weights {
            if w.eviction_policy() != EvictionPolicy::LruLargest {
                continue;
            }
            if !w.is_loaded() || w.refuse_unload() {
                continue;
            }
            let idle_seconds = w
                .sleep_idle_seconds()
                .filter(|&s| s > 0)
                .unwrap_or(self.idle_seconds_default);
            if idle_seconds <= 0 {
                continue;
            }
            // The weights instance's recency: its own last use, else the most
            // recent context row's (a weights with no resident contexts still
            // reports recency through `last_used`).
            let rows = w.residency_rows().await;
            let rows_last_used = rows.iter().map(|r| r.last_used).max().unwrap_or(-1);
            let last_used = if w.last_used() >= 0 { w.last_used() } else { rows_last_used };
            if last_used < 0 {
                continue; // never used
            }
            let idle_ms = now.saturating_sub(last_used);
            if idle_ms < i64::from(idle_seconds) * 1000 {
                continue;
            }
            match w.unload().await {
                Ok(()) => {
                    tracing::info!(
                        target: "fluent-llm.runtime",
                        model = %w.model_key(),
                        idle_seconds,
                        weights_bytes = w.weights_bytes(),
                        "weights instance released (idle)",
                    );
                    released += 1;
                }
                Err(e) => tracing::warn!(
                    target: "fluent-llm.runtime",
                    model = %w.model_key(),
                    error = %e,
                    "idle release refused",
                ),
            }
        }
        released
    }

    /// The pool's resident memory usage and eviction candidates across every
    /// weights instance of that pool. `used` = resident weights + every
    /// context's `total_bytes` (context + compute), mirroring the llama
    /// envelope's summed `total.total` and the onnx working-set sum.
    async fn gather(&self, weights: &[Arc<dyn LlmWeights>], pool: MemoryPool) -> (u64, Vec<Candidate>) {
        let mut used: u64 = 0;
        let mut candidates: Vec<Candidate> = Vec::new();
        for w in weights {
            if w.eviction_policy().pool() != pool {
                continue;
            }
            let rows = w.residency_rows().await;
            let has_pinned = rows.iter().any(|r| r.pinned);
            let resident = if w.is_loaded() { w.weights_bytes() } else { 0 };
            used = used.saturating_add(resident);
            let mut unpinned_vram: u64 = 0;
            let mut unpinned_min_last_used: i64 = i64::MAX;
            let mut unpinned: Vec<&LlmResidencyRow> = Vec::new();
            for row in &rows {
                used = used.saturating_add(row.total_bytes);
                if row.pinned {
                    continue;
                }
                unpinned_vram = unpinned_vram.saturating_add(row.vram_bytes);
                unpinned_min_last_used = unpinned_min_last_used.min(row.last_used);
                unpinned.push(row);
            }
            // A weights instance that refuses unload (`Always`/pinned onnx
            // roles) contributes its footprint to `used` but is never an
            // eviction candidate — the exact onnx rule (`policy.is_always() ||
            // pinned` never evicted) and a no-op for llama (never refuses).
            if w.refuse_unload() {
                continue;
            }
            // A weights instance with NO pinned context is fully evictable:
            // dropping every context unloads its weights too. Pinned contexts
            // keep a model's weights resident, so only weights with zero
            // pinned instances surface as whole-weights candidates.
            if !has_pinned && resident > 0 {
                candidates.push(Candidate {
                    kind: CandidateKind::Weights { weights: Arc::clone(w) },
                    freed_bytes: resident.saturating_add(unpinned_vram),
                    // With no unpinned context rows the weights' own last use
                    // is the recency (the onnx registry's `last_used_ms`; a
                    // llama model with no rows reports its router-tracked
                    // `last_used`, seconds). With rows, the coldest context's
                    // last use stands in (the llama `evict_model` shape).
                    last_used: if unpinned_min_last_used == i64::MAX {
                        w.last_used()
                    } else {
                        unpinned_min_last_used
                    },
                });
            }
            for row in unpinned {
                let name = row
                    .context_key
                    .strip_prefix(&format!("{}:", w.model_key()))
                    .unwrap_or(&row.context_key)
                    .to_string();
                candidates.push(Candidate {
                    kind: CandidateKind::Context {
                        weights: Arc::clone(w),
                        name,
                    },
                    freed_bytes: row.vram_bytes,
                    last_used: row.last_used,
                });
            }
        }
        (used, candidates)
    }

    /// When the pool's resident usage exceeds `budget`, order the candidates
    /// by the pool's injected `EvictionPolicy` and evict until under budget or
    /// the batch is reached.
    async fn evict_over_budget(
        &self,
        weights: &[Arc<dyn LlmWeights>],
        now: i64,
        pool: MemoryPool,
        budget: Option<u64>,
    ) {
        let Some(budget) = budget else {
            return;
        };
        let (used, candidates) = self.gather(weights, pool).await;
        if used <= budget {
            return;
        }
        tracing::warn!(
            target: "fluent-llm.runtime",
            pool = ?pool,
            used_bytes = used,
            budget_bytes = budget,
            "residency over budget - evicting coldest largest footprints",
        );

        // Order best-eviction-first per the pool's injected policy.
        let ordered = match pool {
            MemoryPool::Vram => {
                // footprint × coldness desc, then last-used desc (newer kept).
                common_core::cache::eviction_order(
                    candidates,
                    now.saturating_div(1000),
                    |c: &Candidate| c.freed_bytes,
                    |c: &Candidate| c.last_used,
                )
            }
            MemoryPool::Ram => {
                // LRU-largest: largest footprint first, oldest last_used first.
                let mut ordered = candidates;
                ordered.sort_by(|a, b| {
                    b.freed_bytes
                        .cmp(&a.freed_bytes)
                        .then_with(|| a.last_used.cmp(&b.last_used))
                });
                ordered
            }
        };

        let (used_after, _) = common_core::cache::evict_until_fit(
            used,
            budget,
            self.evict_batch,
            ordered,
            |c: &Candidate| {
                // Clone the candidate's owned parts so the returned future owns
                // its data (the llama pool's `evict_to_fit` does the same).
                let kind = match &c.kind {
                    CandidateKind::Context { weights, name } => CandidateKind::Context {
                        weights: Arc::clone(weights),
                        name: name.clone(),
                    },
                    CandidateKind::Weights { weights } => {
                        CandidateKind::Weights { weights: Arc::clone(weights) }
                    }
                };
                let freed_bytes = c.freed_bytes;
                async move {
                    match &kind {
                        CandidateKind::Context { weights, name } => {
                            let ctx = weights.context(name)?;
                            match ctx.evict().await {
                                Ok(()) => Some(freed_bytes),
                                Err(e) => {
                                    tracing::warn!(
                                        target: "fluent-llm.runtime",
                                        model = %weights.model_key(),
                                        context = %name,
                                        error = %e,
                                        "context eviction failed",
                                    );
                                    None
                                }
                            }
                        }
                        CandidateKind::Weights { weights } => match weights.unload().await {
                            Ok(()) => {
                                tracing::info!(
                                    target: "fluent-llm.runtime",
                                    model = %weights.model_key(),
                                    weights_bytes = freed_bytes,
                                    "weights unloaded to free memory (weights + contexts)",
                                );
                                Some(freed_bytes)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "fluent-llm.runtime",
                                    model = %weights.model_key(),
                                    error = %e,
                                    "weights unload failed - not counted toward the batch",
                                );
                                None
                            }
                        },
                    }
                }
            },
        )
        .await;
        let _ = used_after;
    }

    /// Unload `FootprintColdness` (llama) weights instances whose servers
    /// report zero contexts. Frees the weights. Never touches weights still
    /// holding contexts (pinned instances keep their models resident), and
    /// never the onnx pool (its weights are released by idle/budget, and the
    /// single-shot path legitimately holds zero contexts).
    async fn unload_empty(&self, weights: &[Arc<dyn LlmWeights>]) {
        for w in weights {
            if w.eviction_policy() != EvictionPolicy::FootprintColdness {
                continue;
            }
            if w.refuse_unload() {
                continue;
            }
            // Nothing resident to unload — a down server reports empty rows, so
            // without this gate every residency pass would re-log (and re-try)
            // an unload for every unloaded model, churning forever.
            if !w.is_loaded() {
                continue;
            }
            let rows = w.residency_rows().await;
            if !rows.is_empty() {
                continue;
            }
            tracing::info!(
                target: "fluent-llm.runtime",
                model = %w.model_key(),
                "weights has no contexts left - unloading",
            );
            if let Err(e) = w.unload().await {
                tracing::warn!(
                    target: "fluent-llm.runtime",
                    model = %w.model_key(),
                    error = %e,
                    "unload-empty weights unload failed",
                );
            }
        }
    }

    /// Run the loop as a background task on the injected `Runtime`: one
    /// residency pass per `poll_interval`, forever, with capped backoff on
    /// consecutive failures. The caller owns the returned handle (the server
    /// drains it on graceful shutdown). No ambient `tokio::spawn`.
    pub fn start(
        self: &Arc<Self>,
        runtime: &Arc<dyn Runtime>,
        weights: Arc<Vec<Arc<dyn LlmWeights>>>,
    ) -> tokio::task::JoinHandle<()> {
        let me = Arc::clone(self);
        let poll_interval = me.poll_interval;
        let runtime_loop = Arc::clone(runtime);
        runtime.spawn(Box::pin(async move {
            let base_ms = poll_interval.as_millis() as u64;
            let mut consecutive_failures = 0u32;
            loop {
                match me.residency_cycle(&weights).await {
                    Ok(()) => consecutive_failures = 0,
                    Err(e) => {
                        consecutive_failures += 1;
                        if consecutive_failures == 1 {
                            tracing::warn!(
                                target: "fluent-llm.runtime",
                                error = %e,
                                "residency pass failed - backing off (retrying with backoff)",
                            );
                        } else {
                            tracing::debug!(
                                target: "fluent-llm.runtime",
                                error = %e,
                                consecutive_failures = consecutive_failures,
                                "residency pass still failing - backing off",
                            );
                        }
                    }
                }
                let sleep_ms = common_core::retry::capped_backoff_ms(
                    base_ms,
                    consecutive_failures,
                    12,
                );
                runtime_loop
                    .sleep(Duration::from_millis(sleep_ms))
                    .await;
            }
        }))
    }
}

#[cfg(test)]
#[path = "../tests/runtime.rs"]
mod tests;
