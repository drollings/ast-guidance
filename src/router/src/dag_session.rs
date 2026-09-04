//! Dependency-aware session with DAG step tracking, checkpoint/rewind,
//! and cancellation propagation.
//!
//! Composes `fluent_dag::dep_graph::DependencyGraph<K>` - does **not**
//! re-implement graph algorithms (no hand-rolled `HashMap` dependents index,
//! no manual topo sort, no manual transitive DFS).
//!
//! `SessionRegistry` (in this module) is the canonical server-side session
//! home: sessions are created per `session_id`, attached to a
//! shared `SnapshotStore`, and retained for the process lifetime so
//! checkpoint/rewind state survives across requests.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common_core::registry::ConcurrentRegistry;
use fluent_dag::checkpointed::CheckpointedStepGraph;
use fluent_dag::dep_graph::{DependencyGraph, GraphError};
use serde::{Deserialize, Serialize};

use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex, KvCacheError, SnapshotStore, KvSnapshot};
use crate::session::StepStatus;

/// Errors produced by dependency-session operations.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    #[error("step not found: {0}")]
    StepNotFound(String),

    #[error("step already completed: {0}")]
    StepAlreadyCompleted(String),

    #[error("graph error: {0}")]
    Graph(#[from] GraphError),

    #[error("checkpoint not found: {0}")]
    CheckpointNotFound(String),

    #[error("kv cache error: {0}")]
    KvCache(#[from] KvCacheError),
}

/// The result of executing a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub content: String,
    pub accepted: bool,
    pub score: Option<f64>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// A single step in a dependency session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStep {
    pub id: String,
    pub description: String,
    pub status: StepStatus,
    pub depends_on: Vec<String>,
    pub result: Option<StepResult>,
    pub checkpoint: bool,
}

impl SessionStep {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            status: StepStatus::Pending,
            depends_on: Vec::new(),
            result: None,
            checkpoint: false,
        }
    }

    #[must_use]
    pub fn with_depends(mut self, deps: Vec<String>) -> Self {
        self.depends_on = deps;
        self
    }

    #[must_use]
    pub fn with_checkpoint(mut self) -> Self {
        self.checkpoint = true;
        self
    }
}

/// A step's checkpoint flag drives rewind; KV snapshots on context advance
/// use the model's KV key. This policy is the explicit decision rule the
/// coordinator applies when handing control between models: whether a
/// previously-snapshotted KV state should be **restored** (same model resumes
/// from KV) or **ignored** (a different model re-prefills from the ledger).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvSnapshotPolicy {
    /// Restore only when the pending snapshot belongs to the same model (KV
    /// affinity). The default — a same-model re-entry resumes from KV, a
    /// different model re-prefills.
    #[default]
    #[serde(alias = "restore_if_same_model")]
    RestoreIfSameModel,
    /// Always restore the most recent snapshot, regardless of model.
    #[serde(alias = "always_restore")]
    AlwaysRestore,
    /// Never restore from a snapshot — always re-prefill from the ledger.
    #[serde(alias = "never_restore")]
    NeverRestore,
}

impl KvSnapshotPolicy {
    /// Whether a snapshot owned by `snapshot_model` should be restored for a
    /// request targeting `requested_model`.
    ///
    /// The caller supplies the snapshot **found under the requested model's
    /// own `(model, adapter, session)` key** — not the session's *pending*
    /// snapshot, which may belong to a different model that ran most recently.
    /// This is what lets a same-model re-entry resume its own KV while a
    /// different model re-prefills (the coordinator does exactly this: it
    /// retrieves under `requested_model` and passes that snapshot's model
    /// here). `None` = no snapshot exists for the requested model.
    pub fn decide_restore(&self, snapshot_model: Option<&str>, requested_model: &str) -> bool {
        match self {
            KvSnapshotPolicy::NeverRestore => false,
            KvSnapshotPolicy::AlwaysRestore => snapshot_model.is_some(),
            KvSnapshotPolicy::RestoreIfSameModel => snapshot_model == Some(requested_model),
        }
    }
}

/// A dependency-aware session that tracks steps as a DAG.
///
/// Composes `fluent_dag::dep_graph::DependencyGraph<String>` for
/// dependency tracking. Steps each have an ID, description, status,
/// dependencies, optional result, and a checkpoint flag.
///
/// # Examples
///
/// ```rust,ignore
/// use fluent_router::dag_session::DependencySession;
///
/// let mut session = DependencySession::new("sess-1");
/// session.add_step(SessionStep::new("plan", "Create plan"));
/// session.add_step(SessionStep::new("execute", "Execute plan")
///     .with_depends(vec!["plan".into()]));
///
/// let ready = session.next_ready(); // - vec!["plan"]
/// session.complete_step("plan", StepResult { ... });
/// let ready = session.next_ready(); // - vec!["execute"]
/// ```
pub struct DependencySession {
    pub session_id: String,
    /// The model this session dispatches to. Set by the server on request;
    /// used as the KV-cache snapshot key component (rewind no longer
    /// hard-codes `"unknown"`).
    pub model: Option<String>,
    /// Optional adapter (LoRA) name, part of the KV-cache snapshot key.
    pub adapter: Option<String>,
    graph: CheckpointedStepGraph<String, SessionStep>,
    kv_cache: Option<SnapshotStore>,
    /// The most recently restored KV snapshot on a rewind, carried so the next
    /// dispatch can set the fork's `snapshot`/`instance`/`id_slot` request
    /// fields. `None` before any rewind restored a snapshot.
    pending_snapshot: Option<KvSnapshot>,
    /// Monotonic per-turn counter used to derive deterministic KV snapshot
    /// names on context advance: each turn's snapshot is independently
    /// addressable under the `(model, adapter, session)` key.
    snapshot_seq: u64,
    /// Set when the escalation ladder's turnover mode hands the session to
    /// a frontier model.  Subsequent requests in the session bypass the
    /// local pipeline and go straight to frontier.
    frontier_owned: bool,
}

impl DependencySession {
    /// Create a new dependency session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            model: None,
            adapter: None,
            graph: CheckpointedStepGraph::new(),
            kv_cache: None,
            pending_snapshot: None,
            snapshot_seq: 0,
            frontier_owned: false,
        }
    }

    /// Set the model name this session dispatches to (used for KV-cache
    /// snapshot keying on rewind).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Mark the session as frontier-owned (turnover handoff) or not. When
    /// owned, subsequent requests in the session bypass the local pipeline.
    pub fn set_frontier_owned(&mut self, owned: bool) {
        self.frontier_owned = owned;
    }

    /// Whether the escalation ladder's turnover mode handed this session to a
    /// frontier model. The server routes such sessions' requests straight to
    /// the frontier.
    pub fn is_frontier_owned(&self) -> bool {
        self.frontier_owned
    }

    /// Set the adapter (LoRA) name, part of the KV-cache snapshot key.
    #[must_use]
    pub fn with_adapter(mut self, adapter: impl Into<String>) -> Self {
        self.adapter = Some(adapter.into());
        self
    }

    /// Update the model name in place (called on each request).
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = Some(model.into());
    }

    /// Attach a KV cache manager for checkpoint/rewind support.
    #[must_use]
    pub fn with_kv_cache(mut self, cache: SnapshotStore) -> Self {
        self.kv_cache = Some(cache);
        self
    }

    /// The attached `SnapshotStore`, when present. Lets rigor's blue-pass
    /// completion record a fork snapshot via `save_snapshot` so a later rewind
    /// finds real metadata.
    pub fn kv_cache(&self) -> Option<&SnapshotStore> {
        self.kv_cache.as_ref()
    }

    /// Add a step to the session, registering it in the dependency graph.
    ///
    /// Returns `Err(DagError::Graph(DuplicateNode))` if a step with the
    /// same ID already exists.
    pub fn add_step(&mut self, step: SessionStep) -> Result<(), DagError> {
        let id = step.id.clone();
        let deps = step.depends_on.clone();

        // Each step provides its own ID as an asset, and depends on the
        // IDs of its prerequisite steps. This maps naturally to
        // `DependencyGraph`'s node/asset model (via `CheckpointedStepGraph`).
        self.graph.add_step(id, &deps, step)?;
        Ok(())
    }

    /// Mark a step as completed with the given result.
    ///
    /// If the result is not accepted or has an error, dependents are
    /// cancelled transitively via `DependencyGraph::dependents_of`.
    pub fn complete_step(&mut self, step_id: &str, result: StepResult) -> Result<(), DagError> {
        let should_cancel = !result.accepted || result.error.is_some();

        let is_checkpoint = {
            let step = self
                .graph
                .state_mut(&step_id.to_string())
                .ok_or_else(|| DagError::StepNotFound(step_id.into()))?;

            if step.status == StepStatus::Completed {
                return Err(DagError::StepAlreadyCompleted(step_id.into()));
            }

            step.status = if should_cancel {
                StepStatus::Failed
            } else {
                StepStatus::Completed
            };
            step.result = Some(result);

            step.checkpoint
        };

        self.graph.complete(&step_id.to_string());

        if should_cancel {
            self.cancel_dependents(step_id);
        }

        if is_checkpoint {
            self.graph.checkpoint(step_id.to_string())?;
        }

        Ok(())
    }

    /// Cancel all steps that transitively depend on `step_id`.
    ///
    /// Uses `DependencyGraph::dependents_of` for transitive traversal
    /// with built-in cycle detection.
    pub fn cancel_dependents(&mut self, step_id: &str) {
        let dependents = self.graph.cancel_dependents(&step_id.to_string());
        for dep_id in &dependents {
            if let Some(step) = self.graph.state_mut(dep_id) {
                if step.status == StepStatus::Pending || step.status == StepStatus::InProgress {
                    step.status = StepStatus::Cancelled;
                }
            }
        }
    }

    /// Return the IDs of steps that are pending and whose dependencies
    /// are all satisfied.
    ///
    /// Uses `DependencyGraph::ready_nodes` against the current set of
    /// completed step IDs, then filters out steps already completed.
    pub fn next_ready(&self) -> Vec<String> {
        let mut ready = self.graph.ready_steps();
        ready.retain(|id| !self.graph.is_completed(id));
        // Only return steps that are in Pending status
        ready.retain(|id| {
            self.graph
                .status(id)
                .is_some_and(|s| s.status == StepStatus::Pending)
        });
        ready
    }

    /// Returns `true` if the step is ready to execute (all dependencies
    /// satisfied).
    ///
    /// Uses `DependencyGraph::is_ready`.
    pub fn is_ready(&self, step_id: &str) -> bool {
        self.graph.is_ready(&step_id.to_string())
    }

    /// Returns the IDs of steps that have the checkpoint flag set.
    pub fn checkpoints(&self) -> Vec<String> {
        self.graph
            .step_ids()
            .iter()
            .filter(|id| {
                self.graph
                    .status(id)
                    .is_some_and(|s| s.checkpoint)
            })
            .cloned()
            .collect()
    }

    /// Rewind to a checkpoint, discarding all steps completed after the
    /// checkpoint step.
    ///
    /// Steps are reset to `Pending` but their result data is preserved
    /// for audit (it is not deleted). If a `SnapshotStore` is attached and
    /// a model name has been set, the KV cache snapshot metadata is **actually
    /// restored**: the record is retrieved from the cold tier (promoted to the
    /// hot tier by `SnapshotStore::retrieve`) and returned to the caller, which
    /// passes its fork-facing identity (`snapshot_name`/`instance`/`id_slot`) to
    /// the next dispatch as request fields. Returns `None` when no snapshot
    /// exists. The restored snapshot is also stored on the session for the next
    /// dispatch (`pending_kv_fields`).
    ///
    /// Synchronous by design: the caller holds a `std::sync::MutexGuard`
    /// around the session (`Arc<Mutex<DependencySession>>`), and holding that
    /// guard across an `.await` would make the surrounding future non-`Send`.
    /// The KV restore is synchronous (the two tiers are in-memory metadata
    /// indices; the fork owns the KV bytes).
    pub fn rewind_to_checkpoint(
        &mut self,
        checkpoint_name: &str,
    ) -> Result<Option<Arc<KvSnapshot>>, DagError> {
        // The step graph returns the ordered suffix from the checkpoint
        // (inclusive) and clears the checkpoint marker + completed set.
        let reset = self
            .graph
            .rewind_to(&checkpoint_name.to_string())
            .map_err(|_| DagError::CheckpointNotFound(checkpoint_name.into()))?;

        // Reset each returned step's status to Pending (result data preserved
        // for audit — the primitive does not touch `S`).
        let mut discarded: Vec<String> = Vec::new();
        for step_id in &reset {
            if let Some(step) = self.graph.state_mut(step_id) {
                if step.status == StepStatus::Completed
                    || step.status == StepStatus::Cancelled
                    || step.status == StepStatus::Failed
                {
                    tracing::info!(
                        session_id = %self.session_id,
                        step_id = %step_id,
                        "rewinding step"
                    );
                    step.status = StepStatus::Pending;
                    discarded.push(step_id.clone());
                }
            }
        }

        if !discarded.is_empty() {
            tracing::info!(
                session_id = %self.session_id,
                discarded_count = discarded.len(),
                discarded = ?discarded,
                "rewound to checkpoint, steps reset to Pending (data preserved for audit)"
            );
        }

        // Restore the KV cache snapshot for real. A session with no model
        // name cannot key a snapshot (the key is `(model, adapter, session)`),
        // so the restore is skipped - never a fabricated `"unknown"` lookup.
        let Some(model) = self.model.clone() else {
            tracing::debug!(
                session_id = %self.session_id,
                "no model name on session - skipping kv cache restore"
            );
            return Ok(None);
        };

        Ok(self.restore_kv_snapshot(&model))
    }

    /// Retrieve + restore the KV cache snapshot for this session: the
    /// metadata record is loaded from the cold tier (promoted to the hot tier
    /// by `SnapshotStore::retrieve`) and returned so the caller can pass its
    /// fork-facing identity to the next dispatch. `None` when no snapshot
    /// exists. The restored snapshot is also stored on the session so the next
    /// dispatch can set the `snapshot`/`instance`/`id_slot` request fields.
    fn restore_kv_snapshot(&mut self, model: &str) -> Option<Arc<KvSnapshot>> {
        let kv = self.kv_cache.as_ref()?;
        match kv.retrieve(model, self.adapter.as_deref(), &self.session_id) {
            Ok(snapshot) => {
                tracing::info!(
                    session_id = %self.session_id,
                    model = %model,
                    snapshot_name = %snapshot.snapshot_name,
                    instance = ?snapshot.instance,
                    file_path = %snapshot.file_path.display(),
                    token_count = snapshot.token_count.unwrap_or(0),
                    "kv cache snapshot restored for rewind - pass snapshot/instance to next dispatch"
                );
                self.pending_snapshot = Some((*snapshot).clone());
                Some(snapshot)
            }
            Err(KvCacheError::NotFound(_)) => {
                tracing::debug!(
                    session_id = %self.session_id,
                    "no kv cache snapshot found for rewind"
                );
                None
            }
            Err(e) => {
                tracing::warn!(
                    session_id = %self.session_id,
                    error = %e,
                    "kv cache retrieve failed during rewind"
                );
                None
            }
        }
    }

    /// The fork-facing snapshot identity to send on the next dispatch after a
    /// rewind: `(snapshot_name, instance, id_slot)`. `id_slot` defaults to 0
    /// (the fork's default slot target). `None` when no snapshot was restored.
    pub fn pending_kv_fields(&self) -> Option<(String, Option<String>, i32)> {
        self.pending_snapshot
            .as_ref()
            .map(|s| (s.snapshot_name.clone(), s.instance.clone(), 0))
    }

    /// The most recently recorded/restored KV snapshot, if any. The
    /// coordinator inspects its `model` to decide restore-vs-re-prefill via a
    /// `KvSnapshotPolicy`.
    pub fn pending_snapshot(&self) -> Option<&KvSnapshot> {
        self.pending_snapshot.as_ref()
    }

    /// Snapshot this model's KV on context advance — called when an
    /// agent/model finishes a turn and control moves on. Records a per-turn,
    /// independently-addressable fork snapshot under the `(model, adapter,
    /// session)` key and sets `pending_snapshot` so the next dispatch can pass
    /// the fork-facing `snapshot`/`instance`/`id_slot` fields.
    ///
    /// Requires `self.model` set (the KV key is `(model, adapter, session)`);
    /// without a model this logs and returns `Ok(None)` — never a fabricated
    /// key. Without an attached fork handle `SnapshotStore::save_snapshot`
    /// degrades to a metadata-only no-op, so this returns `Ok(None)`.
    ///
    /// Synchronous by design (mirrors `rewind_to_checkpoint`): the fork
    /// round-trip runs through the sync `block_on` bridge inside
    /// `save_snapshot`, so it can be called while holding the session mutex.
    ///
    /// **Lock note:** this is only safe when the fork round-trip is cheap and
    /// metadata-only (the `SnapshotStore` records metadata and hands the bytes
    /// to llama.cpp; it does not stream them). The coordinator calls it while
    /// holding the session mutex, so the `block_on` bridge must never await a
    /// long I/O — if the fork round-trip ever grows real async I/O, the call
    /// must move outside the guard (the async `run_agent` already stages its
    /// other session-guarded work the same way).
    pub fn advance_and_snapshot(
        &mut self,
        instance: &str,
    ) -> Result<Option<Arc<KvSnapshot>>, DagError> {
        let Some(model) = self.model.clone() else {
            tracing::debug!(
                session_id = %self.session_id,
                "no model name on session - skipping kv snapshot (no fabricated key)",
            );
            return Ok(None);
        };
        let Some(kv) = &self.kv_cache else {
            return Ok(None);
        };

        // Crash-safe per-turn snapshot name: ordered and content-addressed.
        let cold_seq = kv.seed_seq_for(&self.session_id);
        self.snapshot_seq = self.snapshot_seq.max(cold_seq) + 1;
        let hash_hex = common_core::hash::blake3_hex(format!("{}-{}", self.session_id, self.snapshot_seq).as_bytes());
        let hash_suffix = &hash_hex[..8];
        let snapshot_name = format!("{}-{:06}-{}", self.session_id, self.snapshot_seq, hash_suffix);

        kv.save_snapshot_with_seq(
            &model,
            self.adapter.as_deref(),
            &self.session_id,
            &snapshot_name,
            instance,
            Some(self.snapshot_seq),
        )?;

        // Promote the recorded snapshot into `pending_snapshot` so the next
        // dispatch (any model) can pass its fork-facing identity. Without a
        // fork handle `save_snapshot` records no metadata, so `retrieve` finds
        // nothing → `Ok(None)` (metadata-only no-op, never a crash).
        match kv.retrieve(&model, self.adapter.as_deref(), &self.session_id) {
            Ok(snapshot) => {
                tracing::info!(
                    session_id = %self.session_id,
                    model = %model,
                    snapshot_name = %snapshot.snapshot_name,
                    instance = ?snapshot.instance,
                    "kv snapshot recorded on context advance",
                );
                self.pending_snapshot = Some((*snapshot).clone());
                crate::audit::emit(
                    "kv_advance",
                    serde_json::json!({
                        "session_id": self.session_id,
                        "model": model,
                        "adapter": self.adapter,
                        "snapshot_name": snapshot.snapshot_name,
                        "instance": snapshot.instance,
                        "turn_seq": self.snapshot_seq,
                    }),
                );
                Ok(Some(snapshot))
            }
            Err(KvCacheError::NotFound(_)) => {
                tracing::debug!(
                    session_id = %self.session_id,
                    "no kv snapshot recorded (no fork handle) - metadata-only no-op",
                );
                Ok(None)
            }
            Err(e) => Err(DagError::KvCache(e)),
        }
    }

    /// Set a step's status to InProgress (only if currently Pending).
    pub fn start_step(&mut self, step_id: &str) -> Result<(), DagError> {
        let step = self
            .graph
            .state_mut(&step_id.to_string())
            .ok_or_else(|| DagError::StepNotFound(step_id.into()))?;

        if step.status != StepStatus::Pending {
            return Err(DagError::StepAlreadyCompleted(format!(
                "step '{step_id}' is not Pending (current: {:?})",
                step.status
            )));
        }

        step.status = StepStatus::InProgress;
        Ok(())
    }

    /// Look up a step by ID.
    pub fn get_step(&self, step_id: &str) -> Option<&SessionStep> {
        self.graph.status(&step_id.to_string())
    }

    /// All step IDs in insertion order.
    pub fn step_ids(&self) -> &[String] {
        self.graph.step_ids()
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.graph.step_count()
    }

    /// Number of completed steps.
    pub fn completed_count(&self) -> usize {
        self.graph.completed_count()
    }

    /// Borrow the underlying dependency graph (for inspection).
    pub fn graph(&self) -> &DependencyGraph<String> {
        self.graph.graph()
    }

    /// Steps that depend on unsatisfiable assets (assets depended on by
    /// some step but provided by none).
    pub fn unresolved_deps(&self) -> Vec<String> {
        self.graph.unresolved_deps()
    }
}

/// Process-wide registry of `DependencySession`s keyed by `session_id`.
///
/// The canonical server-side session home: sessions are
/// created on first use, attached to a shared `SnapshotStore`, and retained
/// for the process lifetime so checkpoint/rewind state survives across
/// requests. Each session is individually `Mutex`-wrapped so the server can
/// mutate it from the (async) request path without holding the registry lock
/// across an await.
#[derive(Clone)]
pub struct SessionRegistry {
    inner: Arc<SessionRegistryInner>,
}

struct SessionRegistryInner {
    sessions: ConcurrentRegistry<String, Mutex<DependencySession>>,
    kv_cache: SnapshotStore,
}

impl SessionRegistry {
    /// Create a registry. `kv_root` is the cold-tier mountpoint for KV cache
    /// snapshots; when `None`, a process-local temp directory is used (still
    /// durable across requests, ephemeral across restarts).
    pub fn new(kv_root: Option<PathBuf>) -> Self {
        let hot = Arc::new(HotSnapshotIndex::new(1024, 512));
        let cold = Arc::new(ColdSnapshotIndex::new(
            kv_root.unwrap_or_else(|| std::env::temp_dir().join("coral-router-kv-cache")),
            4096,
            7 * 24 * 3600, // 7-day TTL
        ));
        Self {
            inner: Arc::new(SessionRegistryInner {
                sessions: ConcurrentRegistry::new(),
                kv_cache: SnapshotStore::new(hot, cold),
            }),
        }
    }

    /// Create a registry whose sessions share a caller-supplied
    /// `SnapshotStore`. Enables the `LedgerAgentCoordinator` and the
    /// registry to snapshot/restore through the same manager (e.g. one with an
    /// attached fork handle for the fork round-trip).
    pub fn with_kv_cache(kv: SnapshotStore) -> Self {
        Self {
            inner: Arc::new(SessionRegistryInner {
                sessions: ConcurrentRegistry::new(),
                kv_cache: kv,
            }),
        }
    }

    /// The shared KV cache manager (hot + cold tiers) attached to every
    /// session in this registry.
    pub fn kv_cache(&self) -> &SnapshotStore {
        &self.inner.kv_cache
    }

    /// Look up a session by ID, creating it (with the shared `SnapshotStore`
    /// attached) on first use. Seeds `snapshot_seq` from the cold tier (M3).
    pub fn get_or_create(&self, session_id: &str) -> Arc<Mutex<DependencySession>> {
        let kv = self.inner.kv_cache.clone();
        self.inner.sessions.resolve_or_create(session_id.to_string(), |id| {
            let seed = kv.seed_seq_for(id);
            let mut sess = DependencySession::new(id).with_kv_cache(kv.clone());
            sess.snapshot_seq = seed;
            Mutex::new(sess)
        })
    }

    /// Drop a session (its KV cache snapshot, if any, is retained in the
    /// cold tier for a future session with the same ID).
    pub fn remove(&self, session_id: &str) {
        self.inner.sessions.remove(&session_id.to_string());
    }

    /// Number of live sessions.
    pub fn session_count(&self) -> usize {
        self.inner.sessions.len()
    }
}

#[cfg(test)]
#[path = "../tests/dag_session.rs"]
mod tests;
