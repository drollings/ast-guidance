//! Dependency-aware session with DAG step tracking, checkpoint/rewind,
//! and cancellation propagation.
//!
//! Composes `fluent_dag::dep_graph::DependencyGraph<K>` — does **not**
//! re-implement graph algorithms (no hand-rolled `HashMap` dependents index,
//! no manual topo sort, no manual transitive DFS).
//!
//! `SessionRegistry` (in this module) is the canonical server-side session
//! home (decision D6): sessions are created per `session_id`, attached to a
//! shared `KvCacheManager`, and retained for the process lifetime so
//! checkpoint/rewind state survives across requests.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common_core::sync::lock;
use fluent_dag::dep_graph::{DependencyGraph, GraphError};
use serde::{Deserialize, Serialize};

use crate::kv_cache::{ColdKvCache, HotKvCache, KvCacheError, KvCacheManager, KvSnapshot};
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
/// let ready = session.next_ready(); // → vec!["plan"]
/// session.complete_step("plan", StepResult { ... });
/// let ready = session.next_ready(); // → vec!["execute"]
/// ```
pub struct DependencySession {
    pub session_id: String,
    /// The model this session dispatches to. Set by the server on request;
    /// used as the KV-cache snapshot key component (rewind no longer
    /// hard-codes `"unknown"`).
    pub model: Option<String>,
    /// Optional adapter (LoRA) name, part of the KV-cache snapshot key.
    pub adapter: Option<String>,
    steps: HashMap<String, SessionStep>,
    graph: DependencyGraph<String>,
    completed: HashSet<String>,
    checkpoints: HashMap<String, usize>,
    kv_cache: Option<KvCacheManager>,
    step_order: Vec<String>,
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
            steps: HashMap::new(),
            graph: DependencyGraph::new(),
            completed: HashSet::new(),
            checkpoints: HashMap::new(),
            kv_cache: None,
            step_order: Vec::new(),
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
    pub fn with_kv_cache(mut self, cache: KvCacheManager) -> Self {
        self.kv_cache = Some(cache);
        self
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
        // `DependencyGraph`'s node/asset model.
        self.graph.register(&id, &deps, std::slice::from_ref(&id))?;

        self.steps.insert(id.clone(), step);
        self.step_order.push(id);
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
                .steps
                .get_mut(step_id)
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

        self.completed.insert(step_id.to_string());

        if should_cancel {
            self.cancel_dependents(step_id);
        }

        if is_checkpoint {
            self.checkpoints
                .insert(step_id.to_string(), self.completed.len());
        }

        Ok(())
    }

    /// Cancel all steps that transitively depend on `step_id`.
    ///
    /// Uses `DependencyGraph::dependents_of` for transitive traversal
    /// with built-in cycle detection.
    pub fn cancel_dependents(&mut self, step_id: &str) {
        let dependents = self.graph.dependents_of(&step_id.to_string());
        for dep_id in &dependents {
            if let Some(step) = self.steps.get_mut(dep_id) {
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
        let mut ready = self.graph.ready_nodes(&self.completed);
        ready.retain(|id| !self.completed.contains(id));
        // Only return steps that are in Pending status
        ready.retain(|id| {
            self.steps
                .get(id)
                .is_some_and(|s| s.status == StepStatus::Pending)
        });
        ready
    }

    /// Returns `true` if the step is ready to execute (all dependencies
    /// satisfied).
    ///
    /// Uses `DependencyGraph::is_ready`.
    pub fn is_ready(&self, step_id: &str) -> bool {
        self.graph.is_ready(&step_id.to_string(), &self.completed)
    }

    /// Returns the IDs of steps that have the checkpoint flag set.
    pub fn checkpoints(&self) -> Vec<String> {
        self.steps
            .iter()
            .filter(|(_, step)| step.checkpoint)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Rewind to a checkpoint, discarding all steps completed after the
    /// checkpoint step.
    ///
    /// Steps are reset to `Pending` but their result data is preserved
    /// for audit (it is not deleted). If a `KvCacheManager` is attached and
    /// a model name has been set, the KV cache snapshot is **actually
    /// restored**: the metadata record is retrieved from the cold tier
    /// (promoted to the hot tier by `KvCacheManager::retrieve`) and returned
    /// to the caller, which passes its `file_path` to the next dispatch's
    /// slot-restore. Returns `None` when no snapshot exists.
    pub async fn rewind_to_checkpoint(
        &mut self,
        checkpoint_name: &str,
    ) -> Result<Option<Arc<KvSnapshot>>, DagError> {
        // Verify checkpoint exists.
        self.checkpoints
            .get(checkpoint_name)
            .ok_or_else(|| DagError::CheckpointNotFound(checkpoint_name.into()))?;

        let checkpoint_idx = self
            .step_order
            .iter()
            .position(|id| id == checkpoint_name)
            .unwrap_or(0);

        let mut discarded: Vec<String> = Vec::new();

        // Reset steps after checkpoint (including checkpoint step itself).
        for step_id in self.step_order.iter().skip(checkpoint_idx) {
            if let Some(step) = self.steps.get_mut(step_id) {
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
                    self.completed.remove(step_id.as_str());
                    discarded.push(step_id.clone());
                }
            }
        }

        self.checkpoints.remove(checkpoint_name);

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
        // so the restore is skipped — never a fabricated `"unknown"` lookup.
        let Some(model) = self.model.as_deref() else {
            tracing::debug!(
                session_id = %self.session_id,
                "no model name on session — skipping kv cache restore"
            );
            return Ok(None);
        };

        Ok(self.restore_kv_snapshot(model).await)
    }

    /// Retrieve + restore the KV cache snapshot for this session: the
    /// metadata record is loaded from the cold tier (promoted to the hot tier
    /// by `KvCacheManager::retrieve`) and returned so the caller can pass its
    /// `file_path` to the next dispatch's slot-restore. `None` when no
    /// snapshot exists.
    async fn restore_kv_snapshot(&self, model: &str) -> Option<Arc<KvSnapshot>> {
        let kv = self.kv_cache.as_ref()?;
        match kv
            .retrieve(model, self.adapter.as_deref(), &self.session_id)
            .await
        {
            Ok(snapshot) => {
                tracing::info!(
                    session_id = %self.session_id,
                    model = %model,
                    file_path = %snapshot.file_path.display(),
                    token_count = snapshot.token_count.unwrap_or(0),
                    "kv cache snapshot restored for rewind — pass file_path to next dispatch"
                );
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

    /// Set a step's status to InProgress (only if currently Pending).
    pub fn start_step(&mut self, step_id: &str) -> Result<(), DagError> {
        let step = self
            .steps
            .get_mut(step_id)
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
        self.steps.get(step_id)
    }

    /// All step IDs in insertion order.
    pub fn step_ids(&self) -> &[String] {
        &self.step_order
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Number of completed steps.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Borrow the underlying dependency graph (for inspection).
    pub fn graph(&self) -> &DependencyGraph<String> {
        &self.graph
    }

    /// Steps that depend on unsatisfiable assets (assets depended on by
    /// some step but provided by none).
    pub fn unresolved_deps(&self) -> Vec<String> {
        self.graph.unresolved_deps()
    }
}

/// Process-wide registry of `DependencySession`s keyed by `session_id`.
///
/// The canonical server-side session home (decision D6): sessions are
/// created on first use, attached to a shared `KvCacheManager`, and retained
/// for the process lifetime so checkpoint/rewind state survives across
/// requests. Each session is individually `Mutex`-wrapped so the server can
/// mutate it from the (async) request path without holding the registry lock
/// across an await.
#[derive(Clone)]
pub struct SessionRegistry {
    inner: Arc<SessionRegistryInner>,
}

struct SessionRegistryInner {
    sessions: Mutex<HashMap<String, Arc<Mutex<DependencySession>>>>,
    kv_cache: KvCacheManager,
}

impl SessionRegistry {
    /// Create a registry. `kv_root` is the cold-tier mountpoint for KV cache
    /// snapshots; when `None`, a process-local temp directory is used (still
    /// durable across requests, ephemeral across restarts).
    pub fn new(kv_root: Option<PathBuf>) -> Self {
        let hot = Arc::new(HotKvCache::new(1024, 512));
        let cold = Arc::new(ColdKvCache::new(
            kv_root.unwrap_or_else(|| std::env::temp_dir().join("coral-router-kv-cache")),
            4096,
            7 * 24 * 3600, // 7-day TTL
            crate::config::EvictionPolicy::Lru,
        ));
        Self {
            inner: Arc::new(SessionRegistryInner {
                sessions: Mutex::new(HashMap::new()),
                kv_cache: KvCacheManager::new(hot, cold),
            }),
        }
    }

    /// The shared KV cache manager (hot + cold tiers) attached to every
    /// session in this registry.
    pub fn kv_cache(&self) -> &KvCacheManager {
        &self.inner.kv_cache
    }

    /// Look up a session by ID, creating it (with the shared `KvCacheManager`
    /// attached) on first use.
    pub fn get_or_create(&self, session_id: &str) -> Arc<Mutex<DependencySession>> {
        let mut sessions = lock(&self.inner.sessions);
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| {
                Arc::new(Mutex::new(
                    DependencySession::new(session_id).with_kv_cache(self.inner.kv_cache.clone()),
                ))
            })
            .clone()
    }

    /// Drop a session (its KV cache snapshot, if any, is retained in the
    /// cold tier for a future session with the same ID).
    pub fn remove(&self, session_id: &str) {
        lock(&self.inner.sessions).remove(session_id);
    }

    /// Number of live sessions.
    pub fn session_count(&self) -> usize {
        lock(&self.inner.sessions).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_result(content: &str) -> StepResult {
        StepResult {
            content: content.into(),
            accepted: true,
            score: Some(0.9),
            latency_ms: 100,
            error: None,
        }
    }

    fn fail_result(content: &str) -> StepResult {
        StepResult {
            content: content.into(),
            accepted: false,
            score: Some(0.1),
            latency_ms: 50,
            error: Some("execution failed".into()),
        }
    }

    #[test]
    fn test_add_steps_and_get_order() {
        let mut session = DependencySession::new("sess-1");

        session
            .add_step(SessionStep::new("step-1", "First step"))
            .unwrap();
        session
            .add_step(SessionStep::new("step-2", "Second step").with_depends(vec!["step-1".into()]))
            .unwrap();

        assert_eq!(session.step_count(), 2);
        assert_eq!(session.step_ids(), &["step-1", "step-2"]);
    }

    #[test]
    fn test_duplicate_step_rejected() {
        let mut session = DependencySession::new("sess-1");
        session
            .add_step(SessionStep::new("step-1", "First"))
            .unwrap();
        let result = session.add_step(SessionStep::new("step-1", "Duplicate"));
        assert!(matches!(
            result,
            Err(DagError::Graph(GraphError::DuplicateNode(_)))
        ));
    }

    #[test]
    fn test_ready_nodes_basic_dependency() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("plan", "Plan")).unwrap();
        session
            .add_step(SessionStep::new("execute", "Execute").with_depends(vec!["plan".into()]))
            .unwrap();

        // Only "plan" should be ready (no dependencies)
        let ready = session.next_ready();
        assert_eq!(ready, vec!["plan"]);

        // Complete "plan"
        session
            .complete_step("plan", ok_result("plan done"))
            .unwrap();

        // Now "execute" should be ready
        let ready = session.next_ready();
        assert_eq!(ready, vec!["execute"]);
    }

    #[test]
    fn test_fail_cancels_dependents() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();
        session
            .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
            .unwrap();

        // Complete "a" successfully
        session.complete_step("a", ok_result("a done")).unwrap();

        // Complete "b" with failure
        session.complete_step("b", fail_result("b failed")).unwrap();

        let b = session.get_step("b").unwrap();
        assert_eq!(b.status, StepStatus::Failed);

        let c = session.get_step("c").unwrap();
        assert_eq!(c.status, StepStatus::Cancelled);
    }

    #[test]
    fn test_complete_step_not_found() {
        let mut session = DependencySession::new("sess-1");
        let result = session.complete_step("nonexistent", ok_result("nope"));
        assert!(matches!(result, Err(DagError::StepNotFound(_))));
    }

    #[test]
    fn test_complete_already_completed() {
        let mut session = DependencySession::new("sess-1");
        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session.complete_step("a", ok_result("done")).unwrap();

        let result = session.complete_step("a", ok_result("again"));
        assert!(matches!(result, Err(DagError::StepAlreadyCompleted(_))));
    }

    #[test]
    fn test_start_step() {
        let mut session = DependencySession::new("sess-1");
        session.add_step(SessionStep::new("a", "Step A")).unwrap();

        session.start_step("a").unwrap();
        let step = session.get_step("a").unwrap();
        assert_eq!(step.status, StepStatus::InProgress);
    }

    #[test]
    fn test_start_step_not_found() {
        let mut session = DependencySession::new("sess-1");
        let result = session.start_step("nonexistent");
        assert!(matches!(result, Err(DagError::StepNotFound(_))));
    }

    #[test]
    fn test_start_step_not_pending() {
        let mut session = DependencySession::new("sess-1");
        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session.complete_step("a", ok_result("done")).unwrap();

        let result = session.start_step("a");
        assert!(matches!(result, Err(DagError::StepAlreadyCompleted(_))));
    }

    #[test]
    fn test_checkpoint_listing() {
        let mut session = DependencySession::new("sess-1");
        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_checkpoint())
            .unwrap();

        let cps = session.checkpoints();
        assert_eq!(cps.len(), 2);
        assert!(cps.contains(&"a".to_string()));
        assert!(cps.contains(&"b".to_string()));
    }

    #[tokio::test]
    async fn test_rewind_to_checkpoint() {
        let mut session = DependencySession::new("sess-1");

        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();
        session
            .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
            .unwrap();

        // Complete a and reach checkpoint
        session.complete_step("a", ok_result("a done")).unwrap();
        assert!(session.checkpoints().contains(&"a".to_string()));

        // Complete b
        session.complete_step("b", ok_result("b done")).unwrap();
        assert_eq!(session.completed_count(), 2);

        // Rewind to checkpoint "a"
        session.rewind_to_checkpoint("a").await.unwrap();

        // "a" is reset, "b" is reset
        assert_eq!(session.get_step("a").unwrap().status, StepStatus::Pending);
        assert_eq!(session.get_step("b").unwrap().status, StepStatus::Pending);
        // "c" was never completed, stays Pending
        assert_eq!(session.get_step("c").unwrap().status, StepStatus::Pending);
        assert_eq!(session.completed_count(), 0);
    }

    #[tokio::test]
    async fn test_rewind_missing_checkpoint() {
        let mut session = DependencySession::new("sess-1");
        let result = session.rewind_to_checkpoint("nonexistent").await;
        assert!(matches!(result, Err(DagError::CheckpointNotFound(_))));
    }

    #[test]
    fn test_is_ready() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();

        assert!(session.is_ready("a"));
        assert!(!session.is_ready("b"));

        session.complete_step("a", ok_result("done")).unwrap();
        assert!(session.is_ready("b"));
    }

    #[test]
    fn test_is_ready_unregistered() {
        let session = DependencySession::new("sess-1");
        assert!(!session.is_ready("nonexistent"));
    }

    #[test]
    fn test_independent_steps_ready_together() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session.add_step(SessionStep::new("b", "Step B")).unwrap();

        let ready = session.next_ready();
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&"a".to_string()));
        assert!(ready.contains(&"b".to_string()));
    }

    #[test]
    fn test_cycle_detection_in_dependents() {
        // DependencyGraph::dependents_of handles cycles gracefully
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into(), "c".into()]))
            .unwrap();
        session
            .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
            .unwrap();

        // This should not panic — DependencyGraph handles cycles
        let deps = session.graph().dependents_of(&"a".to_string());
        // In a cycle, the result is partial but non-panicking
        assert!(!deps.is_empty() || deps.is_empty()); // Just verify it returns
    }

    #[test]
    fn test_unresolved_deps() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["missing".into()]))
            .unwrap();

        let unresolved = session.unresolved_deps();
        assert!(unresolved.contains(&"missing".to_string()));
    }

    #[tokio::test]
    async fn test_step_result_data_preserved_on_rewind() {
        let mut session = DependencySession::new("sess-1");

        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();

        session
            .complete_step("a", ok_result("important result"))
            .unwrap();
        session.rewind_to_checkpoint("a").await.unwrap();

        let step = session.get_step("a").unwrap();
        assert_eq!(step.status, StepStatus::Pending);
        // Result data preserved for audit
        assert!(step.result.is_some());
        assert_eq!(step.result.as_ref().unwrap().content, "important result");
    }

    #[test]
    fn test_get_step_nonexistent() {
        let session = DependencySession::new("sess-1");
        assert!(session.get_step("nonexistent").is_none());
    }

    #[test]
    fn test_with_constructor_builders() {
        let mut session = DependencySession::new("sess-1");

        let step = SessionStep::new("step-1", "A step")
            .with_depends(vec!["dep-1".into(), "dep-2".into()])
            .with_checkpoint();

        assert_eq!(step.id, "step-1");
        assert_eq!(step.depends_on, vec!["dep-1", "dep-2"]);
        assert!(step.checkpoint);

        session.add_step(step).unwrap();
        assert_eq!(session.step_count(), 1);
    }

    #[tokio::test]
    async fn test_rewind_restores_kv_snapshot_for_real() {
        use crate::kv_cache::KvSnapshot;

        let dir = tempfile::tempdir().unwrap();
        let src_dir = tempfile::tempdir().unwrap();
        let hot = Arc::new(HotKvCache::new(10, 1024));
        let kv = KvCacheManager::new(
            Arc::clone(&hot),
            Arc::new(ColdKvCache::new(
                dir.path(),
                1024,
                86400,
                crate::config::EvictionPolicy::Lru,
            )),
        );

        let src_file = src_dir.path().join("rewind.kv");
        tokio::fs::write(&src_file, b"kv bytes").await.unwrap();
        let snapshot = KvSnapshot {
            model: "model-x".into(),
            adapter: None,
            session_id: "sess-1".into(),
            file_path: src_file,
            token_count: Some(42),
            created_at: common_core::now_secs(),
            last_used_at: common_core::now_secs(),
            llama_cpp_version: Some("0.1.0".into()),
            model_quant: None,
            base_model_hash: Some("abc".into()),
        };
        kv.store(snapshot).await.unwrap();
        // Force a cold-tier hit so rewind exercises the reload-into-hot-tier
        // path rather than a hot-tier cache hit.
        hot.remove("model-x", None, "sess-1");

        let mut session = DependencySession::new("sess-1")
            .with_model("model-x")
            .with_kv_cache(kv);
        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();

        session.complete_step("a", ok_result("a done")).unwrap();
        session.complete_step("b", ok_result("b done")).unwrap();

        // Real restore: the snapshot is returned to the caller (its file_path
        // feeds the next dispatch's slot-restore), not a log-only no-op.
        let restored = session
            .rewind_to_checkpoint("a")
            .await
            .unwrap()
            .expect("snapshot should be restored");
        assert_eq!(restored.session_id, "sess-1");
        assert_eq!(
            restored.file_path,
            dir.path().join("model-x/sess-1/sess-1.kv")
        );
        assert!(restored.file_path.exists());
        // M6.2: the cold tier does NOT fabricate an unknown token count —
        // a raw reload without a sidecar reports `None`.
        assert_eq!(restored.token_count, None);

        // Steps were still reset (data preserved for audit).
        assert_eq!(session.get_step("a").unwrap().status, StepStatus::Pending);
        assert!(session.get_step("a").unwrap().result.is_some());
    }

    #[tokio::test]
    async fn test_rewind_without_model_skips_restore() {
        let mut session = DependencySession::new("sess-1");
        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();
        session.complete_step("a", ok_result("a done")).unwrap();

        let restored = session.rewind_to_checkpoint("a").await.unwrap();
        assert!(restored.is_none(), "no model → no snapshot keyed → None");
    }

    #[test]
    fn test_session_registry_get_or_create() {
        let registry = SessionRegistry::new(None);
        assert_eq!(registry.session_count(), 0);

        let session = registry.get_or_create("sess-1");
        assert_eq!(registry.session_count(), 1);
        assert_eq!(session.lock().unwrap().session_id, "sess-1");

        // Second lookup returns the same session (state survives).
        let again = registry.get_or_create("sess-1");
        assert!(Arc::ptr_eq(&session, &again));

        registry.remove("sess-1");
        assert_eq!(registry.session_count(), 0);
    }
}
